"""
2D S-Parameter Network Protein Fold Engine (v2 — Vectorized)
=============================================================

Extends the 1D ABCD cascade to a 2D TL network where H-bonds are full 
TL segments — not Y_shunt perturbations.

AUDIT FIXES (v1 → v2):
  S1: Vectorized _build_nodal_Y (Python loops → batched jnp ops)
  C2: Added chirality phase correction to backbone γ
  C3: Ported full 5-atom + Cβ steric from 1D engine
  S3: Enabled float64 for numerical stability in matrix solve

ARCHITECTURE:
  backbone bond → TL segment → 2×2 Y-matrix
  H-bond N_i→C_j → TL cross-link → 2×2 Y-matrix
  All Y-matrices summed at nodes (Kirchhoff) → (3N, 3N) global Y
  Schur complement reduction → 1-port S₁₁ at N-terminus

ZERO NEW PARAMETERS: all constants from 1D engine / protein_bond_constants.
"""

import sys
import os
import time
import numpy as np

# Enable float64 BEFORE importing JAX
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
from jax import jit, lax
import optax

# AVE constants
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mechanics'))
from ave.core.constants import ETA_EQ

# Import from 1D engine (reuse, don't duplicate)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'ave', 'solvers'))
from s11_fold_engine_v3_jax import (
    _torsions_to_backbone, _compute_cb_positions,
    compute_z_topo, compute_gly_mask, compute_pro_mask,
    D_N_CA, D_CA_C, D_C_N,
)
from protein_bond_constants import Q_BACKBONE, KAPPA_HB, D_HB_DETECT


# ═══════════════════════════════════════════════════════════════
# CONSTANTS (all from 1D engine — zero new parameters)
# ═══════════════════════════════════════════════════════════════

d0 = 3.8              # Å — Cα-Cα virtual bond length
r_Ca = 1.7            # Å — carbon Slater radius
STERIC = 2.0 * r_Ca   # ≈ 3.4 Å — Pauli exclusion diameter
Z_HB_SCALE = 1.0 / Q_BACKBONE  # = 1/7 ≈ 0.143
D_HB_EQ = 3.0         # Å — H-bond equilibrium N...C distance
HB_SEQ_MIN = 3        # |i-j| ≥ 3 for H-bonds
HB_DIST_MAX = 8.0     # Å — H-bond detection cutoff
FREQ_SWEEP = [0.5, 1.0, 1.7]  # 3 frequencies (audit: reduced from 5)

# Chirality (from 1D engine lines 275-276)
DELTA_CHI = 1.0 / Q_BACKBONE * 0.35  # ≈ 0.05 rad
CHI_SCALE = d0**3 / 11.0             # ≈ 5.0 ų

# Steric LJ zero-crossing distances (from 1D engine lines 894-904)
SIGMA_FACTOR = 1.0 / (2.0 ** (1.0/6.0))  # ≈ 0.891
R_CB_N  = (1.70 + 1.55) * SIGMA_FACTOR
R_CB_C  = (1.70 + 1.70) * SIGMA_FACTOR
R_CB_CB = (1.70 + 1.70) * SIGMA_FACTOR


# ═══════════════════════════════════════════════════════════════
# VECTORIZED NODAL Y-MATRIX ASSEMBLY
# ═══════════════════════════════════════════════════════════════

def _build_nodal_Y_vectorized(backbone_coords, z_topo, freq, N):
    """
    Assemble the (3N, 3N) global nodal admittance matrix.
    
    Fully vectorized — no Python for-loops.
    
    Node numbering: 3i = N_i, 3i+1 = Cα_i, 3i+2 = C_i
    """
    n_nodes = 3 * N
    w = 2.0 * jnp.pi * freq
    
    atom_N  = backbone_coords[:, 0, :]   # (N, 3)
    atom_Ca = backbone_coords[:, 1, :]   # (N, 3)
    atom_C  = backbone_coords[:, 2, :]   # (N, 3)
    
    all_pos = backbone_coords.reshape(n_nodes, 3)
    z_mag = jnp.abs(z_topo)
    
    Y_global = jnp.zeros((n_nodes, n_nodes), dtype=jnp.complex128)
    
    # --- Chirality phase correction ---
    # Triple product of Cα bond vectors → tanh limiter (from 1D engine)
    ca_bonds = atom_Ca[1:] - atom_Ca[:-1]  # (N-1, 3)
    cross = jnp.cross(ca_bonds[:-2], ca_bonds[1:-1])  # (N-3, 3)
    triple = jnp.sum(cross * ca_bonds[2:], axis=1)      # (N-3,)
    chi_signal = jnp.tanh(triple / CHI_SCALE)
    z_avg = 0.5 * (z_mag[:-1] + z_mag[1:])  # (N-1,)
    helix_wt = jnp.clip(1.0 - z_avg / 2.0, 0.0, 1.0)
    chi_padded = jnp.concatenate([jnp.array([0.0]), chi_signal, jnp.array([0.0])])
    chiral_per_res = DELTA_CHI * chi_padded * helix_wt  # (N-1,)
    
    # =========================================
    # BACKBONE TL SEGMENTS (vectorized)
    # =========================================
    
    # --- Type 1: N_i → Cα_i (N segments) ---
    idx_i = jnp.arange(N)
    node_a1 = 3 * idx_i        # N_i
    node_b1 = 3 * idx_i + 1    # Cα_i
    d1 = jnp.sqrt(jnp.sum((atom_N - atom_Ca)**2, axis=-1) + 1e-12)  # (N,)
    Zc1 = z_mag + 1e-12
    alpha1 = jnp.abs(d1 - D_N_CA) / D_N_CA
    beta1 = w * d1 / D_N_CA
    gamma_l1 = alpha1 + 1j * beta1  # (N,) complex
    
    # Y-matrix components for all N segments simultaneously
    Yc1 = 1.0 / Zc1
    cosh1 = jnp.cosh(gamma_l1)
    sinh1 = jnp.sinh(gamma_l1) + 1e-12
    Y11_1 = Yc1 * cosh1 / sinh1   # (N,) — diagonal entries
    Y12_1 = -Yc1 / sinh1           # (N,) — off-diagonal entries
    
    # Scatter into global Y
    Y_global = Y_global.at[node_a1, node_a1].add(Y11_1)
    Y_global = Y_global.at[node_a1, node_b1].add(Y12_1)
    Y_global = Y_global.at[node_b1, node_a1].add(Y12_1)
    Y_global = Y_global.at[node_b1, node_b1].add(Y11_1)
    
    # --- Type 2: Cα_i → C_i (N segments) ---
    node_a2 = 3 * idx_i + 1    # Cα_i
    node_b2 = 3 * idx_i + 2    # C_i
    d2 = jnp.sqrt(jnp.sum((atom_Ca - atom_C)**2, axis=-1) + 1e-12)
    Zc2 = z_mag + 1e-12
    alpha2 = jnp.abs(d2 - D_CA_C) / D_CA_C
    beta2 = w * d2 / D_CA_C
    gamma_l2 = alpha2 + 1j * beta2
    
    Yc2 = 1.0 / Zc2
    cosh2 = jnp.cosh(gamma_l2)
    sinh2 = jnp.sinh(gamma_l2) + 1e-12
    Y11_2 = Yc2 * cosh2 / sinh2
    Y12_2 = -Yc2 / sinh2
    
    Y_global = Y_global.at[node_a2, node_a2].add(Y11_2)
    Y_global = Y_global.at[node_a2, node_b2].add(Y12_2)
    Y_global = Y_global.at[node_b2, node_a2].add(Y12_2)
    Y_global = Y_global.at[node_b2, node_b2].add(Y11_2)
    
    # --- Type 3: C_i → N_{i+1} (N-1 segments, with chirality) ---
    idx_j = jnp.arange(N - 1)
    node_a3 = 3 * idx_j + 2        # C_i
    node_b3 = 3 * (idx_j + 1)      # N_{i+1}
    d3 = jnp.sqrt(jnp.sum((atom_C[:-1] - atom_N[1:])**2, axis=-1) + 1e-12)
    Zc3 = 0.5 * (z_mag[:-1] + z_mag[1:]) + 1e-12
    alpha3 = jnp.abs(d3 - D_C_N) / D_C_N
    # Add chirality phase correction to inter-residue bonds
    beta3 = w * d3 / D_C_N - chiral_per_res
    gamma_l3 = alpha3 + 1j * beta3
    
    Yc3 = 1.0 / Zc3
    cosh3 = jnp.cosh(gamma_l3)
    sinh3 = jnp.sinh(gamma_l3) + 1e-12
    Y11_3 = Yc3 * cosh3 / sinh3
    Y12_3 = -Yc3 / sinh3
    
    Y_global = Y_global.at[node_a3, node_a3].add(Y11_3)
    Y_global = Y_global.at[node_a3, node_b3].add(Y12_3)
    Y_global = Y_global.at[node_b3, node_a3].add(Y12_3)
    Y_global = Y_global.at[node_b3, node_b3].add(Y11_3)
    
    # =========================================
    # H-BOND TL CROSS-LINKS (vectorized N×N)
    # =========================================
    
    # Full pairwise N_i → C_j distance matrix
    diff_NC = atom_N[:, None, :] - atom_C[None, :, :]  # (N, N, 3)
    d_NC = jnp.sqrt(jnp.sum(diff_NC**2, axis=-1) + 1e-12)  # (N, N)
    
    # Sequence separation mask: |i-j| ≥ HB_SEQ_MIN
    seq_mask = jnp.abs(idx_i[:, None] - idx_i[None, :]) >= HB_SEQ_MIN  # (N, N)
    
    # Proximity gating (smooth sigmoid)
    proximity = jax.nn.sigmoid(4.0 * (HB_DIST_MAX - d_NC))  # (N, N)
    
    # H-bond TL impedance
    Zc_hb = Z_HB_SCALE * 0.5 * (z_mag[:, None] + z_mag[None, :]) + 1e-12  # (N, N)
    
    # H-bond propagation
    alpha_hb = jnp.abs(d_NC - D_HB_EQ) / D_HB_EQ  # (N, N)
    beta_hb = w * d_NC / D_HB_EQ
    gamma_l_hb = alpha_hb + 1j * beta_hb  # (N, N) complex
    
    # Y-matrix entries for all N² potential H-bonds simultaneously
    Yc_hb = 1.0 / Zc_hb
    cosh_hb = jnp.cosh(gamma_l_hb)
    sinh_hb = jnp.sinh(gamma_l_hb) + 1e-12
    Y11_hb = Yc_hb * cosh_hb / sinh_hb * proximity * KAPPA_HB  # (N, N)
    Y12_hb = -Yc_hb / sinh_hb * proximity * KAPPA_HB           # (N, N)
    
    # Apply sequence mask
    Y11_hb = jnp.where(seq_mask, Y11_hb, 0.0)
    Y12_hb = jnp.where(seq_mask, Y12_hb, 0.0)
    
    # Scatter H-bond Y-entries into global matrix
    # H-bond connects node_donor = 3*i (N_i) to node_acceptor = 3*j+2 (C_j)
    # We need to add: Y11_hb[i,j] to Y_global[3i, 3i]  (donor diagonal)
    #                 Y11_hb[i,j] to Y_global[3j+2, 3j+2]  (acceptor diagonal)
    #                 Y12_hb[i,j] to Y_global[3i, 3j+2]  (cross-coupling)
    #                 Y12_hb[i,j] to Y_global[3j+2, 3i]  (cross-coupling)
    
    hb_donor_idx = 3 * idx_i       # (N,) — N_i node indices
    hb_accept_idx = 3 * idx_i + 2  # (N,) — C_j node indices
    
    # Sum H-bond contributions per donor/acceptor node
    # Donor diagonal: sum_j Y11_hb[i,j] added to Y_global[3i, 3i]
    donor_diag = Y11_hb.sum(axis=1)  # (N,) — sum over all acceptors j
    Y_global = Y_global.at[hb_donor_idx, hb_donor_idx].add(donor_diag)
    
    # Acceptor diagonal: sum_i Y11_hb[i,j] added to Y_global[3j+2, 3j+2]
    accept_diag = Y11_hb.sum(axis=0)  # (N,) — sum over all donors i
    Y_global = Y_global.at[hb_accept_idx, hb_accept_idx].add(accept_diag)
    
    # Cross-coupling: Y12_hb[i,j] at Y_global[3i, 3j+2] and Y_global[3j+2, 3i]
    # Build index arrays for all N×N entries
    donor_2d = jnp.repeat(hb_donor_idx, N)      # (N²,) — [0,0,...,0, 3,3,...,3, ...]
    accept_2d = jnp.tile(hb_accept_idx, N)       # (N²,) — [2,5,..., 2,5,..., ...]
    y12_flat = Y12_hb.ravel()                     # (N²,)
    
    Y_global = Y_global.at[donor_2d, accept_2d].add(y12_flat)
    Y_global = Y_global.at[accept_2d, donor_2d].add(y12_flat)
    
    return Y_global


# ═══════════════════════════════════════════════════════════════
# GLOBAL S₁₁ FROM NODAL Y-MATRIX
# ═══════════════════════════════════════════════════════════════

def _s11_from_nodal_Y(Y_global, Z0):
    """Extract 1-port S₁₁ at N-terminus (node 0) via Schur complement."""
    Y11 = Y_global[0, 0]
    Y1x = Y_global[0, 1:]
    Yx1 = Y_global[1:, 0]
    Yxx = Y_global[1:, 1:]
    
    # Regularise for numerical stability
    reg = 1e-10 * jnp.eye(Yxx.shape[0], dtype=jnp.complex128)
    v = jnp.linalg.solve(Yxx + reg, Yx1)
    
    Y_reduced = Y11 - jnp.dot(Y1x, v)
    gamma = (1.0 - Z0 * Y_reduced) / (1.0 + Z0 * Y_reduced + 1e-20)
    return jnp.real(gamma * jnp.conj(gamma))


# ═══════════════════════════════════════════════════════════════
# NETWORK LOSS (S₁₁ + full steric)
# ═══════════════════════════════════════════════════════════════

def _network_s11_loss(coords_flat, z_topo, gly_mask, pro_mask, N):
    """
    Compute combined loss: multi-frequency network S₁₁ + full steric.
    """
    bb = coords_flat.reshape(N, 3, 3)
    atom_N  = bb[:, 0, :]
    atom_Ca = bb[:, 1, :]
    atom_C  = bb[:, 2, :]
    coords = atom_Ca
    
    z_mag = jnp.abs(z_topo)
    
    # --- Multi-frequency network S₁₁ ---
    s11_total = 0.0
    for freq in FREQ_SWEEP:
        Y_global = _build_nodal_Y_vectorized(bb, z_topo, freq, N)
        s11_f = _s11_from_nodal_Y(Y_global, Z0=z_mag.mean())
        s11_total = s11_total + s11_f
    s11_avg = s11_total / len(FREQ_SWEEP)
    
    # --- Full steric exclusion (Axiom 2, ported from 1D engine) ---
    idx = jnp.arange(N)
    
    # Cα-Cα steric
    diff_ca = coords[:, None, :] - coords[None, :, :]
    d_ca = jnp.sqrt(jnp.sum(diff_ca**2, axis=-1) + 1e-12)
    ca_mask = jnp.abs(idx[:, None] - idx[None, :]) >= 3
    ca_viol = jnp.maximum(0.0, STERIC - d_ca) ** 2
    ca_viol = jnp.where(ca_mask, ca_viol, 0.0)
    
    # Backbone atom pairwise steric (N-N, C-C, N-C, C-N)
    SIGMA_NN = (1.55 + 1.55) * SIGMA_FACTOR
    SIGMA_CC = (1.70 + 1.70) * SIGMA_FACTOR
    SIGMA_NC = (1.55 + 1.70) * SIGMA_FACTOR
    bb_mask2 = jnp.abs(idx[:, None] - idx[None, :]) >= 2
    
    d_NN = jnp.sqrt(jnp.sum((atom_N[:, None, :] - atom_N[None, :, :])**2, axis=-1) + 1e-12)
    nn_viol = jnp.where(bb_mask2, jnp.maximum(0.0, SIGMA_NN - d_NN)**2, 0.0)
    
    d_CC = jnp.sqrt(jnp.sum((atom_C[:, None, :] - atom_C[None, :, :])**2, axis=-1) + 1e-12)
    cc_viol = jnp.where(bb_mask2, jnp.maximum(0.0, SIGMA_CC - d_CC)**2, 0.0)
    
    d_NC = jnp.sqrt(jnp.sum((atom_N[:, None, :] - atom_C[None, :, :])**2, axis=-1) + 1e-12)
    nc_viol = jnp.where(bb_mask2, jnp.maximum(0.0, SIGMA_NC - d_NC)**2, 0.0)
    
    d_CN = jnp.sqrt(jnp.sum((atom_C[:, None, :] - atom_N[None, :, :])**2, axis=-1) + 1e-12)
    cn_viol = jnp.where(bb_mask2, jnp.maximum(0.0, SIGMA_NC - d_CN)**2, 0.0)
    
    # Cβ steric (from 1D engine)
    chi1_def = jnp.full(N, jnp.radians(60.0))
    cb_pos = _compute_cb_positions(atom_N, atom_Ca, atom_C, chi1_def, gly_mask)
    cb_mask = jnp.abs(idx[:, None] - idx[None, :]) >= 1
    cb_seq3 = jnp.abs(idx[:, None] - idx[None, :]) >= 3
    
    d_CBN = jnp.sqrt(jnp.sum((cb_pos[:, None, :] - atom_N[None, :, :])**2, axis=-1) + 1e-12)
    cbn_viol = jnp.where(cb_mask, jnp.maximum(0.0, R_CB_N - d_CBN)**2, 0.0)
    
    d_CBC = jnp.sqrt(jnp.sum((cb_pos[:, None, :] - atom_C[None, :, :])**2, axis=-1) + 1e-12)
    cbc_viol = jnp.where(cb_mask, jnp.maximum(0.0, R_CB_C - d_CBC)**2, 0.0)
    
    d_CBCB = jnp.sqrt(jnp.sum((cb_pos[:, None, :] - cb_pos[None, :, :])**2, axis=-1) + 1e-12)
    cbcb_viol = jnp.where(cb_seq3, jnp.maximum(0.0, R_CB_CB - d_CBCB)**2, 0.0)
    
    # Total steric penalty
    LAMBDA_STERIC = d0 / r_Ca  # ≈ 2.24 (from 1D: LAMBDA_BOND * d0 / r_Ca)
    bb_steric = (
        jnp.sum(jnp.triu(nn_viol, k=2)) +
        jnp.sum(jnp.triu(cc_viol, k=2)) +
        jnp.sum(jnp.triu(nc_viol, k=2)) +
        jnp.sum(jnp.triu(cn_viol, k=2)) +
        jnp.sum(cbn_viol) + jnp.sum(cbc_viol) +
        jnp.sum(jnp.triu(cbcb_viol, k=3))
    )
    steric_penalty = LAMBDA_STERIC * (
        jnp.sum(jnp.triu(ca_viol, k=3)) / N + bb_steric / (6 * N)
    )
    
    return s11_avg + steric_penalty


# ═══════════════════════════════════════════════════════════════
# TORSION-ANGLE WRAPPER
# ═══════════════════════════════════════════════════════════════

def _network_torsion_loss(angles, z_topo, gly_mask, pro_mask, N):
    """Convert (φ,ψ) → backbone → network S₁₁ loss."""
    phi = angles[:N]
    psi = angles[N:]
    coords_flat = _torsions_to_backbone(phi, psi, N)
    return _network_s11_loss(coords_flat, z_topo, gly_mask, pro_mask, N)

_network_torsion_loss_jit = jit(_network_torsion_loss, static_argnums=(4,))


# ═══════════════════════════════════════════════════════════════
# FOLD FUNCTION
# ═══════════════════════════════════════════════════════════════

def fold_network(seq, n_steps=20000, lr=2e-3, n_starts=5):
    """
    Fold a protein using the 2D S-parameter network engine.
    Same interface as fold_s11_jax from the 1D engine.
    """
    N = len(seq)
    print(f"  2D TL network v2 (vectorized, {n_starts}-start): N={N}, steps={n_steps}")
    
    z_topo = compute_z_topo(seq)
    gly_mask = compute_gly_mask(seq)
    pro_mask = compute_pro_mask(seq)
    
    loss_fn = _network_torsion_loss_jit
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
        
        # Pre-generate noise keys (audit fix S4)
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


# ═══════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    seq = "YYDPETGT"  # 8-residue GB1 hairpin turn
    print("=== 2D TL NETWORK v2: 8-residue β-hairpin ===")
    ca, z, trace, bb = fold_network(seq, n_steps=5000, lr=2e-3, n_starts=3)
    
    N = len(seq)
    rg = np.sqrt(np.mean(np.sum((ca - ca.mean(0))**2, 1)))
    Rg_eq = 1.7 * (N / ETA_EQ) ** (1.0 / 3.0) * np.sqrt(3.0 / 5.0)
    print(f"Rg={rg:.2f} (target {Rg_eq:.2f})")
    print(f"Loss={trace[0]:.4f}")
    print(f"v1 baseline: Rg=5.48, Loss=0.2313")
