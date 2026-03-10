#!/usr/bin/env python3
"""
Loaded Bloch Eigenvalue — Passband/Stopband with Y_shunt
=========================================================

Key insight from bare-backbone test: the unloaded backbone is always
in the passband. Confinement (stopband → SS) requires Y_shunt loading
at Cα junctions — the same through-space contacts that drive S₁₁.

This parallels the proton mass eigenvalue:
  Proton: I_scalar / (1 - V·P_C) → confinement from P_C cavity
  Protein: ABCD_loaded = ABCD_bare × Shunt(Y) → confinement from contacts

The loaded ABCD at each Cα junction:
  [A'  B']   [1  0] [A  B]   [A          B      ]
  [C'  D'] = [Y  1] [C  D] = [C + Y·A    D + Y·B]
"""
import sys, os, time, urllib.request
import numpy as np

os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
from jax import lax
import optax

from ave.core.constants import ETA_EQ, P_C
from multiscale_fold_engine import _torsions_to_backbone, compute_gly_mask, compute_pro_mask
from s11_fold_engine_v3_jax import (
    _torsion_loss as _s11_torsion_loss,
    _s11_loss as _original_s11_loss,
    compute_z_topo as compute_z_topo_v3,
    compute_cys_mask as compute_cys_mask_v3,
    compute_aromatic_mask as compute_aromatic_mask_v3,
    Q_BACKBONE,
    D_N_CA, D_CA_C, D_C_N,
    M_N_CA, M_CA_C, M_C_N,
    N_E_N_CA, N_E_CA_C, N_E_C_N,
)
from ave.solvers.protein_bond_constants import KAPPA_HB, D_HB_DETECT

# ═══════════════════════════════════════════════════════════════
# Compute Y_shunt (simplified version of main engine's computation)
# ═══════════════════════════════════════════════════════════════

def _compute_y_shunt(coords, z_topo, N):
    """
    Compute per-residue Y_shunt from hydrophobic coupling.
    Simplified version of the main _s11_loss Y_shunt computation.
    
    Returns: (N,) array of shunt admittances at each Cα.
    """
    d0 = 3.8
    r_Ca = 1.7
    KAPPA = 0.5
    R_BURIAL = 2.0 * d0
    
    z_mag = jnp.abs(z_topo)
    
    # Pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)
    
    # Conjugate impedance matching
    z_conj = z_topo[:, None] * jnp.conj(z_topo[None, :])
    z_mags = jnp.abs(z_topo[:, None]) * jnp.abs(z_topo[None, :]) + 1e-12
    conj_match = jnp.maximum(0.0, jnp.real(z_conj) / z_mags)
    
    # Axiom 4 saturation
    sat = jnp.clip(d0 / (dists + 1e-12), 0.0, 0.95)
    C_raw = 1.0 / jnp.sqrt(1.0 - sat**2)
    C_sat = 1.0 + (C_raw - 1.0) * conj_match
    
    coupling = KAPPA * conj_match * C_sat / (dists**2 + 1e-12)
    
    # Long-range saturation
    lr = jnp.clip(dists / R_BURIAL, 0.0, 0.999)
    coupling = coupling * jnp.sqrt(1.0 - lr**2)
    
    # Q-decay
    idx = jnp.arange(N)
    seq_sep = jnp.abs(idx[:, None] - idx[None, :]).astype(jnp.float64)
    coupling = coupling * jnp.exp(-seq_sep / (2.0 * jnp.pi * Q_BACKBONE))
    
    # Exclude local backbone
    mask = jnp.abs(idx[:, None] - idx[None, :]) <= 2
    coupling = jnp.where(mask, 0.0, coupling)
    
    Y_shunt = coupling.sum(axis=1)
    
    # Global packing saturation (same as main engine)
    com = jnp.mean(coords, axis=0)
    Rg_sq = jnp.mean(jnp.sum((coords - com)**2, axis=1))
    R_eff = jnp.sqrt(5.0 / 3.0 * Rg_sq + 1e-12)
    eta = N * r_Ca**3 / (R_eff**3 + 1e-12)
    eta_ratio = jnp.clip(eta / P_C, 0.0, 0.999)
    sat_global = jnp.sqrt(1.0 - eta_ratio**2)
    
    Y_shunt = Y_shunt * sat_global
    
    return Y_shunt


# ═══════════════════════════════════════════════════════════════
# Loaded Bloch Eigenvalue Analysis
# ═══════════════════════════════════════════════════════════════

def _loaded_bloch_reward(coords_flat, z_topo, N):
    """
    Bloch eigenvalue with Y_shunt-loaded Cα junctions.
    
    The loaded ABCD for each TL segment + shunt at junction:
    ABCD_loaded = Shunt(Y) × ABCD_segment
    
    For a unit cell of P residues, the total ABCD is:
    T = Π_{i=0}^{P-1} [ABCD_res(i) × Shunt(Y_i) × ABCD_peptide(i→i+1)]
    
    Bloch condition: cos(βP) = (A+D)/2
      |cos(βP)| ≤ 1 → passband (wave propagates) → coil
      |cos(βP)| > 1 → stopband (Bragg reflection) → SS confined
    """
    bb = coords_flat.reshape(N, 3, 3)
    atom_N_arr = bb[:, 0, :]
    atom_Ca = bb[:, 1, :]
    atom_C  = bb[:, 2, :]
    
    # Compute Y_shunt at each Cα
    Y_shunt = _compute_y_shunt(atom_Ca, z_topo, N)
    
    # Bond distances
    d_NCa = jnp.sqrt(jnp.sum((atom_Ca - atom_N_arr)**2, axis=-1) + 1e-12)
    d_CaC = jnp.sqrt(jnp.sum((atom_C - atom_Ca)**2, axis=-1) + 1e-12)
    d_CN  = jnp.sqrt(jnp.sum((atom_N_arr[1:] - atom_C[:-1])**2, axis=-1) + 1e-12)
    
    # Segment impedances
    Z_NCa = jnp.sqrt(M_N_CA / float(N_E_N_CA))
    Z_CaC = jnp.sqrt(M_CA_C / float(N_E_CA_C))
    Z_CN  = jnp.sqrt(M_C_N  / float(N_E_C_N))
    z_mag = jnp.abs(z_topo)
    
    def _abcd_seg(d_actual, d_target, Z_base, R_sc):
        """ABCD for one TL segment."""
        w = 2.0 * jnp.pi * 1.0
        beta = w * d_actual / d_target
        alpha = jnp.abs(d_actual - d_target) / d_target
        gamma = alpha + 1j * beta
        Zc = Z_base * jnp.sqrt(1.0 + R_sc**2) + 1e-12
        ch = jnp.cosh(gamma)
        sh = jnp.sinh(gamma)
        return ch, Zc * sh, sh / Zc, ch
    
    def _mul(a1, a2):
        """ABCD multiply."""
        A1,B1,C1,D1 = a1; A2,B2,C2,D2 = a2
        return (A1*A2+B1*C2, A1*B2+B1*D2, C1*A2+D1*C2, C1*B2+D1*D2)
    
    def _shunt(Y):
        """Shunt admittance ABCD: [[1,0],[Y,1]]."""
        return (1.0+0j, 0.0+0j, Y+0j, 1.0+0j)
    
    def _loaded_residue_abcd(i):
        """ABCD for one residue: N→Cα segment × Shunt(Y) × Cα→C segment."""
        R_i = z_mag[i]
        # N_i → Cα_i
        seg1 = _abcd_seg(d_NCa[i], D_N_CA, Z_NCa, R_i)
        # Shunt Y at Cα_i (THE KEY: loading creates stopband)
        shunt = _shunt(Y_shunt[i])
        # Cα_i → C_i
        seg2 = _abcd_seg(d_CaC[i], D_CA_C, Z_CaC, R_i)
        # Product: seg1 × shunt × seg2
        return _mul(_mul(seg1, shunt), seg2)
    
    def _peptide_abcd(i):
        """ABCD for C_i → N_{i+1} peptide bond."""
        R_avg = (z_mag[i] + z_mag[jnp.clip(i+1, 0, N-1)]) / 2.0
        return _abcd_seg(d_CN[jnp.clip(i, 0, N-2)], D_C_N, Z_CN, R_avg)
    
    # ───────────────────────────────────────────
    # Helix unit cell: 4 residues
    # ───────────────────────────────────────────
    n_helix = max(0, N - 4)
    
    def helix_bloch(start):
        """Loaded Bloch eigenvalue for 4-residue helix window."""
        abcd = (1.0+0j, 0.0+0j, 0.0+0j, 1.0+0j)
        for k in range(4):
            i = jnp.clip(start + k, 0, N-1)
            abcd = _mul(abcd, _loaded_residue_abcd(i))
            if k < 3:
                abcd = _mul(abcd, _peptide_abcd(i))
        A, B, C, D = abcd
        cos_bP = jnp.real(A + D) / 2.0
        return jnp.maximum(0.0, jnp.abs(cos_bP) - 1.0)
    
    # ───────────────────────────────────────────
    # Sheet unit cell: 2 residues
    # ───────────────────────────────────────────
    n_sheet = max(0, N - 2)
    
    def sheet_bloch(start):
        """Loaded Bloch eigenvalue for 2-residue sheet window."""
        abcd = (1.0+0j, 0.0+0j, 0.0+0j, 1.0+0j)
        for k in range(2):
            i = jnp.clip(start + k, 0, N-1)
            abcd = _mul(abcd, _loaded_residue_abcd(i))
            if k < 1:
                abcd = _mul(abcd, _peptide_abcd(i))
        A, B, C, D = abcd
        cos_bP = jnp.real(A + D) / 2.0
        return jnp.maximum(0.0, jnp.abs(cos_bP) - 1.0)
    
    helix_conf = jnp.array([helix_bloch(s) for s in range(n_helix)])
    sheet_conf = jnp.array([sheet_bloch(s) for s in range(n_sheet)])
    
    helix_frac = jnp.mean(jnp.where(helix_conf > 0.01, 1.0, 0.0)) if n_helix > 0 else 0.0
    sheet_frac = jnp.mean(jnp.where(sheet_conf > 0.01, 1.0, 0.0)) if n_sheet > 0 else 0.0
    helix_depth = jnp.mean(helix_conf) if n_helix > 0 else 0.0
    sheet_depth = jnp.mean(sheet_conf) if n_sheet > 0 else 0.0
    
    # Sequence-weighted reward
    z_mean = jnp.mean(z_mag)
    hw = jnp.clip(1.0 - z_mean / 2.0, 0.0, 1.0)
    sw = jnp.clip(z_mean / 2.0, 0.0, 1.0)
    total_depth = hw * helix_depth + sw * sheet_depth
    
    # Coupling strength: 1/(2Q) = backbone mutual inductance coefficient
    kappa = 1.0 / (2.0 * Q_BACKBONE)
    reward = kappa * total_depth
    
    return reward, helix_conf, sheet_conf, helix_frac, sheet_frac, Y_shunt


def _combined_bloch_loss(angles, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N):
    """Original S₁₁ + loaded Bloch confinement reward."""
    phi = angles[:N]
    psi = angles[N:2*N]
    coords = _torsions_to_backbone(phi, psi, N)
    base = _original_s11_loss(coords, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N)
    reward, _, _, _, _, _ = _loaded_bloch_reward(coords, z_topo, N)
    return base - reward


# ═══════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════

AA_MAP = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q',
    'GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
    'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

def extract_ca(pdb_path, chain, max_res):
    ca, seq = [], []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"): continue
            if line[12:16].strip() != "CA": continue
            if line[16] not in (' ', 'A'): continue
            ch = line[21]
            if chain != "*" and ch != chain: continue
            res_id = line[22:27].strip()
            if res_id in seen: continue
            seen.add(res_id)
            ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            seq.append(AA_MAP.get(line[17:20].strip(), 'A'))
            if len(ca) >= max_res: break
    return np.array(ca) if ca else None, ''.join(seq)

def kabsch_rmsd(P, Q):
    p0 = P - P.mean(0); q0 = Q - Q.mean(0)
    H = p0.T @ q0
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return np.sqrt(np.mean(np.sum((p0 @ R.T - q0)**2, axis=1)))

def extract_bb(pdb_path, chain, max_res):
    atoms = {'N': {}, 'CA': {}, 'C': {}}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'): continue
            if line[16] not in (' ', 'A'): continue
            ch = line[21]
            if chain != '*' and ch != chain: continue
            a = line[12:16].strip()
            if a not in atoms: continue
            r = line[22:27].strip()
            if r not in atoms[a]:
                atoms[a][r] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    rids = sorted(atoms['CA'].keys(), key=lambda x: int(''.join(c for c in x if c.isdigit() or c=='-') or '0'))
    bb = []
    for r in rids[:max_res]:
        if r in atoms['N'] and r in atoms['CA'] and r in atoms['C']:
            bb.append([atoms['N'][r], atoms['CA'][r], atoms['C'][r]])
    return np.array(bb) if bb else None

# Load Trp-cage
pdb_path = '/tmp/1L2Y.pdb'
if not os.path.exists(pdb_path):
    urllib.request.urlretrieve('https://files.rcsb.org/download/1L2Y.pdb', pdb_path)
ca_native, seq = extract_ca(pdb_path, 'A', 20)
N = len(seq)
Rg_eq = 1.7 * (N / ETA_EQ)**(1/3) * np.sqrt(3/5)

print('=' * 70)
print(f'  Loaded Bloch Eigenvalue Test: Trp-cage (N={N})')
print('=' * 70)

z_topo = compute_z_topo_v3(seq)
cys_mask = compute_cys_mask_v3(seq)
arom_mask = compute_aromatic_mask_v3(seq)
gly_mask = compute_gly_mask(seq)
pro_mask = compute_pro_mask(seq)

# ── Native analysis ──
print("\n--- Native Structure: Loaded Bloch ---")
bb_nat = extract_bb(pdb_path, 'A', 20)
if bb_nat is not None and len(bb_nat) >= N:
    coords_nat = jnp.array(bb_nat[:N].reshape(-1))
    reward_nat, hc, sc, hf, sf, Ysh = _loaded_bloch_reward(coords_nat, z_topo, N)
    print(f"  Bloch reward (native, loaded): {float(reward_nat):.6f}")
    print(f"  Y_shunt (native): {[f'{float(y):.2f}' for y in Ysh]}")
    print(f"  Helix confinement: {[f'{float(c):.3f}' for c in hc]}")
    print(f"  Helix stopband frac: {float(hf):.1%}")
    print(f"  Sheet stopband frac: {float(sf):.1%}")

# ── Optimization comparison ──
N_STARTS = 4
N_STEPS = 20000
LR = 2e-3

for mode in ['baseline', 'bloch_loaded']:
    print(f"\n{'─'*70}")
    print(f"  Mode: {mode.upper()} ({N_STARTS} starts × {N_STEPS} steps)")
    print(f"{'─'*70}")
    
    if mode == 'baseline':
        loss_fn = lambda a: _s11_torsion_loss(
            a, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N)
    else:
        loss_fn = lambda a: _combined_bloch_loss(
            a, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N)
    
    loss_jit = jax.jit(loss_fn)
    grad_jit = jax.jit(jax.grad(loss_fn))
    
    best_rmsd = float('inf')
    best_loss = float('inf')
    
    for si in range(N_STARTS):
        seed = 42 + si * 137
        np.random.seed(seed)
        phi_i = np.random.uniform(-np.pi, np.pi, N)
        psi_i = np.random.uniform(-np.pi, np.pi, N)
        angles = jnp.concatenate([jnp.array(phi_i), jnp.array(psi_i)])
        
        t0 = time.time()
        if si == 0:
            _ = loss_jit(angles)
            _ = grad_jit(angles)
            print(f"  JIT compiled in {time.time()-t0:.1f}s", flush=True)
            t0 = time.time()
        
        opt = optax.adam(LR)
        opt_state = opt.init(angles)
        key = jax.random.PRNGKey(seed)
        
        for step in range(N_STEPS):
            g = grad_jit(angles)
            g = jnp.where(jnp.isnan(g), 0.0, g)
            gn = jnp.sqrt(jnp.sum(g**2) + 1e-12)
            g = jnp.where(gn > 10.0, g * 10.0 / gn, g)
            updates, opt_state = opt.update(g, opt_state)
            angles = optax.apply_updates(angles, updates)
            if step < N_STEPS * 0.2:
                T = 0.03 * (1.0 - step / (N_STEPS * 0.2)) ** 2
                key, subkey = jax.random.split(key)
                angles = angles + jax.random.normal(subkey, shape=angles.shape) * T
        
        loss = float(loss_jit(angles))
        dt = time.time() - t0
        
        phi_f = angles[:N]; psi_f = angles[N:2*N]
        coords = _torsions_to_backbone(phi_f, psi_f, N)
        bb = np.array(coords.reshape(N, 3, 3))
        ca = bb[:, 1, :]
        rg = np.sqrt(np.mean(np.sum((ca - ca.mean(0))**2, 1)))
        rg_err = 100 * abs(rg - Rg_eq) / Rg_eq
        rmsd = kabsch_rmsd(ca, ca_native[:N])
        
        reward_f, hc_f, sc_f, hf_f, sf_f, Ysh_f = _loaded_bloch_reward(
            coords, z_topo, N)
        
        print(f"  start {si}: loss={loss:.4f} RMSD={rmsd:.2f}Å Rg={rg:.1f}Å({rg_err:.0f}%) "
              f"Bloch={float(reward_f):.4f} H_stop={float(hf_f):.0%} ({dt:.0f}s)", 
              flush=True)
        
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_loss = loss
    
    print(f"\n  BEST {mode.upper()}: RMSD={best_rmsd:.2f}Å  loss={best_loss:.4f}")

print(f"\n{'='*70}")
