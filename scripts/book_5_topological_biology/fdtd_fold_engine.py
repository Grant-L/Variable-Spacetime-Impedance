"""
Yee-Lattice FDTD Protein Fold Engine
=====================================

Places the protein backbone as (μ,ε) impedance defects on a 2D/3D Yee grid.
Waves propagate THROUGH the protein AND the surrounding lattice.
S₁₁ = reflected energy ratio at the N-terminus injection node.

This is the most AVE-consistent approach: the protein IS ON the lattice.

PHYSICS:
  Each residue at position (x_i, y_i) creates:
    μ(x) += mass_i / m_e  × G(x - x_i, σ)    (mass = inductance)
    ε(x) += n_e_i / α     × G(x - x_i, σ)    (electrons = permittivity)
  where G is a Gaussian smearing kernel (σ = Slater radius).

  Z(x) = √(μ(x)/ε(x)) — the local impedance at every grid point.
  Where Z varies, waves reflect. The fold that minimises total reflection
  is the native state.

  Same physics as bond_energy_solver.py (nuclear scale) and
  fdtd_yee_lattice.py (macroscopic scale), applied at molecular scale.

IMPLEMENTATION:
  2D TMz mode: Ez, Hx, Hy on staggered Yee grid.
  JAX for autodiff through time-stepping.
  Torsion angles → Cα positions → grid defects → FDTD → S₁₁ → gradient.
"""

import sys
import os
import time
import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
from jax import jit
import optax

# AVE constants
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mechanics'))
from ave.core.constants import ETA_EQ, P_C, ALPHA

# Import from 1D engine
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'ave', 'solvers'))
from s11_fold_engine_v3_jax import (
    _torsions_to_backbone, _compute_cb_positions,
    compute_z_topo, compute_gly_mask, compute_pro_mask,
)
from protein_bond_constants import Q_BACKBONE


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Grid parameters
DX = 1.0             # Å — grid pitch (molecular resolution)
GRID_PAD = 15        # cells of padding around protein
SIGMA_DEFECT = 1.7   # Å — Gaussian smearing radius (= Slater C radius)

# FDTD parameters
# At molecular scale, normalise: c = 1.0, μ₀ = 1.0, ε₀ = 1.0
# CFL for 2D: dt ≤ dx / (c × √2) = 1.0 / √2 ≈ 0.707
DT = 0.5             # well within CFL
N_TIMESTEPS = 200    # enough for waves to traverse ~100 Å
DAMPING = 0.001      # slight absorption at boundaries

# Source parameters
PULSE_WIDTH = 4.0    # Å — Gaussian pulse width (broadband)

# Steric (from 2D engine)
d0 = 3.8
r_Ca = 1.7
STERIC = 2.0 * r_Ca
LAMBDA_STERIC = 2.0 * d0 / r_Ca


# ═══════════════════════════════════════════════════════════════
# DEFECT PLACEMENT: Protein → (μ, ε) grid
# ═══════════════════════════════════════════════════════════════

def _place_protein_on_grid(ca_positions, z_topo, grid_size, grid_origin):
    """
    Place protein Cα atoms as (μ, ε) defects on a 2D grid.
    
    Each residue creates Gaussian-smeared perturbations:
        μ(x,y) += |Z_topo| × G(r, σ)     (mass → inductance)
        ε(x,y) += (1/|Z_topo|) × G(r, σ) (electrons → permittivity)
    
    This preserves Z = √(μ/ε) = |Z_topo| at the defect center,
    while ε grows → Z drops → wave slows down (higher refractive index).
    
    Args:
        ca_positions: (N, 3) Cα coordinates (only x,y used for 2D)
        z_topo: (N,) complex impedances  
        grid_size: (Ny, Nx) grid dimensions
        grid_origin: (ox, oy) physical coordinates of grid[0,0]
    """
    Ny, Nx = grid_size
    ox, oy = grid_origin
    z_mag = jnp.abs(z_topo)
    N = len(z_mag)
    
    # Start with vacuum: μ = 1, ε = 1 → Z = 1
    mu_grid = jnp.ones((Ny, Nx))
    eps_grid = jnp.ones((Ny, Nx))
    
    # Grid coordinates
    gx = jnp.arange(Nx) * DX + ox  # (Nx,)
    gy = jnp.arange(Ny) * DX + oy  # (Ny,)
    GX, GY = jnp.meshgrid(gx, gy)  # (Ny, Nx) each
    
    # Vectorized: compute all Gaussian kernels at once
    # ca_positions[:, 0] is (N,); GX is (Ny, Nx)
    # Broadcast: (N, 1, 1) vs (1, Ny, Nx) → (N, Ny, Nx)
    cx = ca_positions[:, 0][:, None, None]  # (N, 1, 1)
    cy = ca_positions[:, 1][:, None, None]  # (N, 1, 1)
    r_sq = (GX[None, :, :] - cx)**2 + (GY[None, :, :] - cy)**2  # (N, Ny, Nx)
    kernels = jnp.exp(-r_sq / (2.0 * SIGMA_DEFECT**2))  # (N, Ny, Nx)
    
    # μ enhancement: Σ z_mag[i] × kernel_i
    mu_weights = z_mag[:, None, None]  # (N, 1, 1)
    mu_grid = mu_grid + jnp.sum(mu_weights * kernels, axis=0)
    
    # ε enhancement: Σ (1/z_mag[i]) × kernel_i
    eps_weights = (1.0 / (z_mag + 1e-6))[:, None, None]
    eps_grid = eps_grid + jnp.sum(eps_weights * kernels, axis=0)
    
    return mu_grid, eps_grid


# ═══════════════════════════════════════════════════════════════
# 2D TMz FDTD ENGINE (JAX)
# ═══════════════════════════════════════════════════════════════

def _fdtd_s11(ca_positions, z_topo, N, grid_size):
    """
    Run 2D TMz FDTD with protein defects and return |S₁₁|².
    
    Grid is fixed size (static for JIT). Protein centered at grid center.
    
    Args:
        ca_positions: (N, 3) Cα coordinates
        z_topo: (N,) complex impedances
        N: number of residues (static)
        grid_size: (Ny, Nx) static grid dimensions
    """
    Ny, Nx = grid_size
    
    # Center protein at grid center
    com = jnp.mean(ca_positions, axis=0)
    ca_centered = ca_positions - com  # center of mass at (0,0,0)
    
    # Grid origin so that (0,0) maps to grid center
    ox = -(Nx // 2) * DX
    oy = -(Ny // 2) * DX
    grid_origin = (ox, oy)
    
    # Place protein on grid
    mu_grid, eps_grid = _place_protein_on_grid(
        ca_centered, z_topo, (Ny, Nx), grid_origin
    )
    
    # Update coefficients
    Ce = DT / (DX * eps_grid)
    Ch_x = DT / (DX * 0.5 * (mu_grid[:, :-1] + mu_grid[:, 1:]))
    Ch_y = DT / (DX * 0.5 * (mu_grid[:-1, :] + mu_grid[1:, :]))
    
    # Fields
    Ez = jnp.zeros((Ny, Nx))
    Hx = jnp.zeros((Ny, Nx - 1))
    Hy = jnp.zeros((Ny - 1, Nx))
    
    # Source at N-terminus: offset from center
    src_dx = ca_centered[0, 0] / DX  # fractional offset
    src_dy = ca_centered[0, 1] / DX
    src_x = Nx // 2 + jnp.round(src_dx).astype(int)
    src_y = Ny // 2 + jnp.round(src_dy).astype(int)
    src_x = jnp.clip(src_x, 2, Nx - 3)
    src_y = jnp.clip(src_y, 2, Ny - 3)
    
    t_peak = 5.0 * PULSE_WIDTH
    
    def fdtd_step(carry, t_idx):
        Ez, Hx, Hy, inc_e, ref_e = carry
        t = t_idx * DT
        
        # Broadband Gaussian pulse
        pulse = jnp.exp(-0.5 * ((t - t_peak) / PULSE_WIDTH)**2)
        pulse_val = pulse * jnp.sin(2.0 * jnp.pi * t / (4.0 * PULSE_WIDTH))
        inc_e = inc_e + pulse_val**2
        
        # Update H
        Hx = Hx - Ch_x * (Ez[:, 1:] - Ez[:, :-1])
        Hy = Hy + Ch_y * (Ez[1:, :] - Ez[:-1, :])
        
        # Absorbing BCs
        Ez = Ez.at[0, :].set(Ez[1, :])
        Ez = Ez.at[-1, :].set(Ez[-2, :])
        Ez = Ez.at[:, 0].set(Ez[:, 1])
        Ez = Ez.at[:, -1].set(Ez[:, -2])
        
        # Update E
        Ez = Ez.at[1:-1, 1:-1].add(
            Ce[1:-1, 1:-1] * (
                (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) -
                (Hx[1:-1, 1:] - Hx[1:-1, :-1])
            )
        )
        
        # Soft source
        Ez = Ez.at[src_y, src_x].add(pulse_val)
        
        # Measure reflection after pulse passes
        is_measuring = t > 2.0 * t_peak
        ref_e = ref_e + is_measuring * Ez[src_y, src_x]**2
        
        return (Ez, Hx, Hy, inc_e, ref_e), None
    
    init_carry = (Ez, Hx, Hy, 0.0, 0.0)
    t_indices = jnp.arange(N_TIMESTEPS)
    (_, _, _, inc_energy, ref_energy), _ = jax.lax.scan(
        fdtd_step, init_carry, t_indices
    )
    
    s11_sq = ref_energy / (inc_energy + 1e-12)
    return s11_sq


# ═══════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════

def _fdtd_loss(coords_flat, z_topo, gly_mask, pro_mask, N):
    """FDTD S₁₁ + steric penalty."""
    bb = coords_flat.reshape(N, 3, 3)
    atom_Ca = bb[:, 1, :]  # (N, 3) — use only Cα for grid placement
    
    # Grid size: static, based on N (will be passed via static_argnums)
    # N≤20 → 50×50, N≤40 → 70×70, N≤80 → 100×100
    if N <= 20:
        grid_size = (50, 50)
    elif N <= 40:
        grid_size = (70, 70)
    else:
        grid_size = (100, 100)
    
    # FDTD S₁₁
    s11 = _fdtd_s11(atom_Ca, z_topo, N, grid_size)
    
    # P_C packing saturation
    com = jnp.mean(atom_Ca, axis=0)
    Rg_sq = jnp.mean(jnp.sum((atom_Ca - com)**2, axis=1))
    R_eff = jnp.sqrt(5.0/3.0 * Rg_sq + 1e-12)
    eta = N * r_Ca**3 / (R_eff**3 + 1e-12)
    eta_ratio = jnp.clip(eta / P_C, 0.0, 0.999)
    sat_global = jnp.sqrt(1.0 - eta_ratio**2)
    
    s11 = s11 * sat_global
    
    # Steric (Cα-Cα only for speed)
    idx = jnp.arange(N)
    diff = atom_Ca[:, None, :] - atom_Ca[None, :, :]
    d_ca = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)
    ca_mask = jnp.abs(idx[:, None] - idx[None, :]) >= 3
    ca_viol = jnp.where(ca_mask, jnp.maximum(0.0, d0 - d_ca)**2, 0.0)
    steric = LAMBDA_STERIC * jnp.sum(jnp.triu(ca_viol, k=3)) / N
    
    return s11 + steric


def _fdtd_torsion_loss(angles, z_topo, gly_mask, pro_mask, N):
    """(φ,ψ) → backbone → FDTD S₁₁ loss."""
    phi = angles[:N]
    psi = angles[N:]
    coords_flat = _torsions_to_backbone(phi, psi, N)
    return _fdtd_loss(coords_flat, z_topo, gly_mask, pro_mask, N)


# ═══════════════════════════════════════════════════════════════
# FOLD FUNCTION
# ═══════════════════════════════════════════════════════════════

def fold_fdtd(seq, n_steps=5000, lr=2e-3, n_starts=3):
    """Fold protein via Yee-lattice FDTD."""
    N = len(seq)
    print(f"  FDTD lattice (2D TMz, {n_starts}-start): N={N}, steps={n_steps}")
    
    z_topo = compute_z_topo(seq)
    gly_mask = compute_gly_mask(seq)
    pro_mask = compute_pro_mask(seq)
    
    loss_fn = jit(_fdtd_torsion_loss, static_argnums=(4,))
    loss_and_grad = jax.value_and_grad(loss_fn)
    
    t_jit = time.time()
    key = jax.random.PRNGKey(42)
    test_angles = jax.random.normal(key, (2 * N,)) * 0.5
    _ = loss_and_grad(test_angles, z_topo, gly_mask, pro_mask, N)
    print(f"    JIT compiled in {time.time() - t_jit:.1f}s")
    
    best_loss = float('inf')
    best_angles = None
    
    for s in range(n_starts):
        t_start = time.time()
        key, subkey = jax.random.split(key)
        angles = jax.random.normal(subkey, (2 * N,)) * 0.5
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(angles)
        
        key, noise_base = jax.random.split(key)
        noise_keys = jax.random.split(noise_base, n_steps)
        
        T0 = 0.05
        for step in range(n_steps):
            loss_val, grads = loss_and_grad(angles, z_topo, gly_mask, pro_mask, N)
            T = T0 * (1.0 - step / n_steps)
            noise = T * jax.random.normal(noise_keys[step], grads.shape)
            grads = grads + noise
            updates, opt_state = optimizer.update(grads, opt_state)
            angles = optax.apply_updates(angles, updates)
        
        final_loss = float(loss_fn(angles, z_topo, gly_mask, pro_mask, N))
        dt = time.time() - t_start
        print(f"    start {s}: loss={final_loss:.4f} ({dt:.0f}s)")
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_angles = angles
    
    print(f"    best loss = {best_loss:.6f}")
    
    phi = best_angles[:N]
    psi = best_angles[N:]
    coords_flat = _torsions_to_backbone(phi, psi, N)
    bb = np.array(coords_flat).reshape(N, 3, 3)
    ca = bb[:, 1, :]
    
    return ca, np.array(z_topo), [best_loss], bb


if __name__ == "__main__":
    seq = "YYDPETGT"
    print("=== FDTD LATTICE: 8-residue β-hairpin ===")
    ca, z, trace, bb = fold_fdtd(seq, n_steps=3000, lr=2e-3, n_starts=3)
    N = len(seq)
    rg = np.sqrt(np.mean(np.sum((ca - ca.mean(0))**2, 1)))
    Rg_eq = 1.7 * (N / ETA_EQ) ** (1.0/3.0) * np.sqrt(3.0/5.0)
    print(f"Rg={rg:.2f} (target {Rg_eq:.2f})")
    print(f"Loss={trace[0]:.4f}")
