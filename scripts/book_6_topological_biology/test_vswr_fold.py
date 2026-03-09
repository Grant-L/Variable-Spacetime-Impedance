#!/usr/bin/env python3
"""
VSWR Standing Wave Analysis — Test
====================================

Hypothesis: Adding voltage standing wave ratio (VSWR) periodicity analysis
to the loss function provides topological discrimination that S₁₁ alone lacks.

EE principle: A TL with periodic impedance discontinuities (Bragg grating)
has well-defined standing wave nodes. The native fold's standing wave pattern
has periodicity matching the secondary structure:
  - α-helix: λ = 3.6 residues (i→i+4 H-bond periodicity)
  - β-sheet: λ = 2.0 residues (alternating pleated sheet)

By extracting V(i) at each Cα node during the ABCD cascade and computing
the autocorrelation of |V|, we reward conformations whose standing wave
pattern has physical periodicity.

This adds TOPOLOGICAL information to the loss without changing any existing
coupling constants — it's a new observable from the same TL physics.
"""
import sys, os, time, urllib.request
import numpy as np

os.environ['JAX_ENABLE_X64'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

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
    Q_BACKBONE, N_FREQ, FREQ_SWEEP,
    D_N_CA, D_CA_C, D_C_N,
    M_N_CA, M_CA_C, M_C_N,
    N_E_N_CA, N_E_CA_C, N_E_C_N,
    LAMBDA_BOND, debye_z_water,
)
from ave.solvers.protein_bond_constants import KAPPA_HB, D_HB_DETECT

# ═══════════════════════════════════════════════════════════════
# VSWR Extraction from ABCD Cascade
# ═══════════════════════════════════════════════════════════════

def _vswr_loss(coords_flat, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N, chi1=None):
    """
    Compute VSWR periodicity reward from backbone standing wave pattern.
    
    Physics: During the ABCD cascade, the voltage at each node is
    V(i) = A_i + B_i / Z_L, where A_i and B_i are the ABCD cascade
    state at node i. Instead of walking through the lax.fori_loop
    (which discards intermediate states), we use lax.scan to capture
    V at each Cα junction.
    
    The autocorrelation of |V(i)| reveals periodicity:
      - Peak at lag=4 → α-helix standing wave
      - Peak at lag=2 → β-sheet standing wave
      - No peaks → random coil (no standing wave)
    
    The reward is the autocorrelation contrast at physical lags.
    """
    bb = coords_flat.reshape(N, 3, 3)
    atom_N  = bb[:, 0, :]
    atom_Ca = bb[:, 1, :]
    atom_C  = bb[:, 2, :]
    coords = atom_Ca
    
    d0 = 3.8
    r_Ca = 1.7
    Z0 = 1.0
    
    # --- Build backbone TL segments (same as _s11_loss) ---
    d_NCa = jnp.sqrt(jnp.sum((atom_Ca - atom_N)**2, axis=-1) + 1e-12)
    d_CaC = jnp.sqrt(jnp.sum((atom_C - atom_Ca)**2, axis=-1) + 1e-12)
    d_CN  = jnp.sqrt(jnp.sum((atom_N[1:] - atom_C[:-1])**2, axis=-1) + 1e-12)
    
    triplets = jnp.stack([d_NCa[:-1], d_CaC[:-1], d_CN], axis=1)
    last_pair = jnp.array([d_NCa[-1], d_CaC[-1]])
    seg_d = jnp.concatenate([triplets.reshape(-1), last_pair])
    
    d0_triplet = jnp.array([D_N_CA, D_CA_C, D_C_N])
    d0_last = jnp.array([D_N_CA, D_CA_C])
    seg_d0 = jnp.concatenate([jnp.tile(d0_triplet, N-1), d0_last])
    
    Z_NCa = jnp.sqrt(M_N_CA / float(N_E_N_CA))
    Z_CaC = jnp.sqrt(M_CA_C / float(N_E_CA_C))
    Z_CN  = jnp.sqrt(M_C_N  / float(N_E_C_N))
    z_triplet = jnp.array([Z_NCa, Z_CaC, Z_CN])
    z_last = jnp.array([Z_NCa, Z_CaC])
    seg_Zc = jnp.concatenate([jnp.tile(z_triplet, N-1), z_last])
    
    n_bb_segs = 3 * N - 1
    
    # --- VSWR at center frequency (ω₀ = 1.0) ---
    w = 2.0 * jnp.pi * 1.0
    beta_arr = w * seg_d / seg_d0
    alpha_arr = jnp.abs(seg_d - seg_d0) / seg_d0
    gamma_arr = alpha_arr + 1j * beta_arr
    cosh_arr = jnp.cosh(gamma_arr)
    sinh_arr = jnp.sinh(gamma_arr)
    
    # Use lax.scan to capture ABCD at every Cα junction
    # State: [A, B, C, D] (complex ABCD parameters)
    # At each step, cascade through segment i
    # At every 3rd step (Cα junctions), record V = |A + B/Z₀|
    
    def scan_step(state, i):
        A, B, C, D = state[0], state[1], state[2], state[3]
        ch = cosh_arr[i]
        sh = sinh_arr[i]
        Zc = seg_Zc[i] + 1e-12
        
        # Lossy TL section
        A_n = A * ch + B * (sh / Zc)
        B_n = A * (Zc * sh) + B * ch
        C_n = C * ch + D * (sh / Zc)
        D_n = C * (Zc * sh) + D * ch
        
        # Record voltage at this junction: V = A + B/Z₀
        V = A_n + B_n / Z0
        V_mag = jnp.sqrt(jnp.real(V * jnp.conj(V)) + 1e-12)
        
        return jnp.array([A_n, B_n, C_n, D_n]), V_mag
    
    # Scan through all segments
    init_state = jnp.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j])
    indices = jnp.arange(n_bb_segs)
    
    # Can't use lax.scan with dynamic indexing into cosh_arr easily.
    # Instead, unroll for N <= 130 (vectorized approach):
    # Walk through segments and record voltage at Cα positions
    
    # Forward cascade: accumulate ABCD and record V at each Cα
    def cascade_forward(carry, seg_idx):
        A, B, C, D = carry[0], carry[1], carry[2], carry[3]
        ch = cosh_arr[seg_idx]
        sh = sinh_arr[seg_idx]
        Zc = seg_Zc[seg_idx] + 1e-12
        
        A_n = A * ch + B * (sh / Zc)
        B_n = A * (Zc * sh) + B * ch
        C_n = C * ch + D * (sh / Zc)
        D_n = C * (Zc * sh) + D * ch
        
        V = A_n + B_n / Z0
        V_mag = jnp.sqrt(jnp.real(V * jnp.conj(V)) + 1e-12)
        
        return jnp.array([A_n, B_n, C_n, D_n]), V_mag
    
    _, V_all = lax.scan(cascade_forward, init_state, indices)
    # V_all is (3N-1,) — voltage magnitude at each backbone segment junction
    
    # Extract voltage at Cα positions (every 3rd junction, starting at index 0)
    # Segment layout: [N₀-Cα₀=0, Cα₀-C₀=1, C₀-N₁=2, N₁-Cα₁=3, ...]
    # After segment 0 (N-Cα): we're at Cα₀ → index 0
    # After segment 3 (N₁-Cα₁): we're at Cα₁ → index 3
    # General: Cα_i → index 3i
    ca_indices = jnp.arange(N) * 3
    ca_indices = jnp.clip(ca_indices, 0, n_bb_segs - 1)
    V_ca = V_all[ca_indices]  # (N,) voltage at each Cα
    
    # Normalize voltage pattern (remove DC bias)
    V_mean = jnp.mean(V_ca)
    V_norm = V_ca - V_mean  # zero-mean
    V_std = jnp.sqrt(jnp.mean(V_norm**2) + 1e-12)
    V_unit = V_norm / (V_std + 1e-12)  # unit variance
    
    # ═══════════════════════════════════════════════════════════════
    # Autocorrelation of standing wave pattern
    # ═══════════════════════════════════════════════════════════════
    # C(k) = (1/N) Σ V(i) × V(i+k)
    #
    # Physical lags:
    #   k=0: C(0) = 1 (normalized)
    #   k=4: α-helix periodicity (3.6 ≈ 4 residues per turn)
    #   k=2: β-sheet periodicity (alternating pleat)
    #   k=7: two α-helix turns (verifies long-range helix order)
    #
    # A well-formed helix has C(4) ≈ +1 (positive correlation)
    # A well-formed sheet has C(2) ≈ +1
    # Random coil has C(k) ≈ 0 for k > 0
    
    max_lag = min(8, N - 1)
    
    def autocorr_at_lag(k):
        """Autocorrelation at lag k."""
        # Shift and correlate
        v1 = V_unit[:N-k]
        v2 = V_unit[k:N]
        return jnp.mean(v1 * v2)
    
    # Compute autocorrelation at physical lags
    acf = jnp.array([autocorr_at_lag(k) for k in range(max_lag + 1)])
    
    # ═══════════════════════════════════════════════════════════════
    # VSWR Periodicity Reward
    # ═══════════════════════════════════════════════════════════════
    # Reward: positive autocorrelation at helix (k=4) and sheet (k=2) lags
    # Weight by Z_topo: low-Z residues → helix, high-Z → sheet
    #
    # No fitting: weights derived from Z_topo impedance classification
    z_mag = jnp.abs(z_topo)
    z_mean = jnp.mean(z_mag)
    
    # Helix propensity: fraction of residues with |Z| < 1.0 (helix-formers)
    helix_frac = jnp.mean(jnp.where(z_mag < 1.0, 1.0, 0.0))
    # Sheet propensity: fraction of residues with |Z| > 1.0 (sheet-formers)
    sheet_frac = jnp.mean(jnp.where(z_mag > 1.0, 1.0, 0.0))
    
    # Autocorrelation at helix lag (k=4)
    acf_helix = acf[4] if max_lag >= 4 else 0.0
    # Autocorrelation at sheet lag (k=2)  
    acf_sheet = acf[2] if max_lag >= 2 else 0.0
    
    # VSWR reward: weighted by sequence composition
    # Uses 1/(2Q) = κ_HB as coupling strength (same as H-bond mutual inductance)
    # This is the natural strength: standing waves couple at 1/(2Q) of the main signal
    vswr_reward = KAPPA_HB * (helix_frac * acf_helix + sheet_frac * acf_sheet)
    
    return vswr_reward, V_ca, acf


def _combined_loss(angles, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N):
    """Original S₁₁ loss + VSWR periodicity reward."""
    phi = angles[:N]
    psi = angles[N:2*N]
    coords_flat = _torsions_to_backbone(phi, psi, N)
    
    # Original loss
    base_loss = _original_s11_loss(
        coords_flat, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N)
    
    # VSWR reward (negative = reduces loss)
    vswr_reward, _, _ = _vswr_loss(
        coords_flat, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N)
    
    return base_loss - vswr_reward


# ═══════════════════════════════════════════════════════════════
# Test: Compare with and without VSWR on Trp-cage
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

# Load Trp-cage
pdb_path = '/tmp/1L2Y.pdb'
if not os.path.exists(pdb_path):
    urllib.request.urlretrieve('https://files.rcsb.org/download/1L2Y.pdb', pdb_path)
ca_native, seq = extract_ca(pdb_path, 'A', 20)
N = len(seq)
Rg_eq = 1.7 * (N / ETA_EQ)**(1/3) * np.sqrt(3/5)

print('=' * 70)
print(f'  VSWR Test: Trp-cage (N={N})')
print(f'  η_eq = {ETA_EQ:.6f}')
print('=' * 70)

z_topo = compute_z_topo_v3(seq)
cys_mask = compute_cys_mask_v3(seq)
arom_mask = compute_aromatic_mask_v3(seq)
gly_mask = compute_gly_mask(seq)
pro_mask = compute_pro_mask(seq)

# First: analyze the NATIVE structure's VSWR pattern
print("\n--- Native Structure VSWR Analysis ---")
# Extract native backbone (N, Cα, C)
atoms_dict = {'N': {}, 'CA': {}, 'C': {}}
with open(pdb_path) as f:
    for line in f:
        if not line.startswith('ATOM'): continue
        if line[16] not in (' ', 'A'): continue
        ch = line[21]
        if ch != 'A': continue
        aname = line[12:16].strip()
        if aname not in atoms_dict: continue
        res_id = line[22:27].strip()
        if res_id not in atoms_dict[aname]:
            atoms_dict[aname][res_id] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]

res_ids = sorted(atoms_dict['CA'].keys(), key=lambda x: int(''.join(c for c in x if c.isdigit() or c=='-') or '0'))
bb_native = []
for rid in res_ids[:N]:
    if rid in atoms_dict['N'] and rid in atoms_dict['CA'] and rid in atoms_dict['C']:
        bb_native.append([atoms_dict['N'][rid], atoms_dict['CA'][rid], atoms_dict['C'][rid]])
bb_native = np.array(bb_native)
N_nat = len(bb_native)

if N_nat >= N:
    coords_flat_native = jnp.array(bb_native[:N].reshape(-1))
    vswr_nat, V_nat, acf_nat = _vswr_loss(
        coords_flat_native, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N)
    print(f"  VSWR reward (native): {float(vswr_nat):.4f}")
    print(f"  Autocorrelation: {[f'{float(a):.3f}' for a in acf_nat]}")
    print(f"  V pattern: {[f'{float(v):.2f}' for v in V_nat]}")
else:
    print(f"  (native backbone extraction only got {N_nat} residues, need {N})")

# Run optimization with and without VSWR
N_STARTS = 4
N_STEPS = 20000
LR = 2e-3

for mode in ['baseline', 'vswr']:
    print(f"\n{'─'*70}")
    print(f"  Mode: {mode.upper()} ({N_STARTS} starts × {N_STEPS} steps)")
    print(f"{'─'*70}")
    
    if mode == 'baseline':
        loss_fn = lambda a: _s11_torsion_loss(
            a, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N)
    else:
        loss_fn = lambda a: _combined_loss(
            a, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N)
    
    loss_jit = jax.jit(loss_fn)
    grad_jit = jax.jit(jax.grad(loss_fn))
    
    best_rmsd = float('inf')
    best_loss = float('inf')
    
    for si in range(N_STARTS):
        seed = 42 + si * 137
        np.random.seed(seed)
        phi_init = np.random.uniform(-np.pi, np.pi, N)
        psi_init = np.random.uniform(-np.pi, np.pi, N)
        angles = jnp.concatenate([jnp.array(phi_init), jnp.array(psi_init)])
        
        t0 = time.time()
        if si == 0:
            _ = loss_jit(angles)
            _ = grad_jit(angles)
            print(f"  JIT compiled in {time.time()-t0:.1f}s", flush=True)
            t0 = time.time()
        
        optimizer = optax.adam(LR)
        opt_state = optimizer.init(angles)
        key = jax.random.PRNGKey(seed)
        
        for step in range(N_STEPS):
            g = grad_jit(angles)
            g = jnp.where(jnp.isnan(g), 0.0, g)
            g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-12)
            g = jnp.where(g_norm > 10.0, g * 10.0 / g_norm, g)
            updates, opt_state = optimizer.update(g, opt_state)
            angles = optax.apply_updates(angles, updates)
            if step < N_STEPS * 0.2:
                T = 0.03 * (1.0 - step / (N_STEPS * 0.2)) ** 2
                key, subkey = jax.random.split(key)
                angles = angles + jax.random.normal(subkey, shape=angles.shape) * T
        
        loss = float(loss_jit(angles))
        dt = time.time() - t0
        
        # Build coords and compute metrics
        phi_f = angles[:N]; psi_f = angles[N:2*N]
        coords = _torsions_to_backbone(phi_f, psi_f, N)
        bb = np.array(coords.reshape(N, 3, 3))
        ca = bb[:, 1, :]
        rg = np.sqrt(np.mean(np.sum((ca - ca.mean(0))**2, 1)))
        rg_err = 100 * abs(rg - Rg_eq) / Rg_eq
        rmsd = kabsch_rmsd(ca, ca_native[:N])
        
        # VSWR analysis of folded structure
        vswr_r, V_f, acf_f = _vswr_loss(
            coords, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N)
        
        print(f"  start {si}: loss={loss:.4f} RMSD={rmsd:.2f}Å Rg={rg:.1f}Å({rg_err:.0f}%) "
              f"VSWR={float(vswr_r):.4f} ACF[4]={float(acf_f[4]):.3f} ({dt:.0f}s)", 
              flush=True)
        
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_loss = loss
    
    print(f"\n  BEST {mode.upper()}: RMSD={best_rmsd:.2f}Å  loss={best_loss:.4f}")

print(f"\n{'='*70}")
