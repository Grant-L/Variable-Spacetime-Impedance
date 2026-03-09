#!/usr/bin/env python3
"""
χ₁ Sidechain DOF Test: 2N vs 3N Optimization
==============================================

The v3 engine already supports 3N DOF (φ, ψ, χ₁) but we've been running
with only 2N. This test compares:

  2N mode: angles = [φ₀...φₙ, ψ₀...ψₙ]           (χ₁ = 60° for all)
  3N mode: angles = [φ₀...φₙ, ψ₀...ψₙ, χ₁₀...χ₁ₙ]  (χ₁ optimized)

What χ₁ controls:
  - Cβ sidechain stub direction via Rodrigues rotation
  - Enters 5 of 13 steric exclusion terms (Cβ-N, Cβ-C, Cβ-Cβ, O-Cβ, H-Cβ)
  - Different rotamer states (−60°, 60°, 180°) resolve steric clashes
  - With χ₁ locked, every sidechain points the same way → wrong packing

Biology: Sidechain rotamers are the "jigsaw puzzle" of protein interior.
EE: χ₁ controls the shunt stub ORIENTATION — its phase relative to the
    backbone determines constructive vs destructive interference at each
    junction. Locked stubs = all stubs in phase = unphysical.
"""
import sys, os, time, urllib.request
import numpy as np

os.environ['JAX_ENABLE_X64'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
import optax

from ave.core.constants import ETA_EQ
from multiscale_fold_engine import _torsions_to_backbone, compute_gly_mask, compute_pro_mask
from s11_fold_engine_v3_jax import (
    _torsion_loss,
    compute_z_topo as compute_z_topo_v3,
    compute_cys_mask as compute_cys_mask_v3,
    compute_aromatic_mask as compute_aromatic_mask_v3,
)

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
print(f'  χ₁ DOF Test: Trp-cage (N={N})')
print(f'  2N = {2*N} DOF (φ,ψ) vs 3N = {3*N} DOF (φ,ψ,χ₁)')
print('=' * 70)

z_topo = compute_z_topo_v3(seq)
cys_mask = compute_cys_mask_v3(seq)
arom_mask = compute_aromatic_mask_v3(seq)
gly_mask = compute_gly_mask(seq)
pro_mask = compute_pro_mask(seq)

N_STARTS = 4
N_STEPS = 20000
LR = 2e-3

for mode in ['2N_baseline', '3N_chi1']:
    n_dof = 2 * N if mode == '2N_baseline' else 3 * N
    
    print(f"\n{'─'*70}")
    print(f"  Mode: {mode} ({n_dof} DOF, {N_STARTS} starts × {N_STEPS} steps)")
    print(f"{'─'*70}")
    
    loss_fn = lambda a: _torsion_loss(
        a, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N)
    
    loss_jit = jax.jit(loss_fn)
    grad_jit = jax.jit(jax.grad(loss_fn))
    
    best_rmsd = float('inf')
    best_loss = float('inf')
    best_angles = None
    
    for si in range(N_STARTS):
        seed = 42 + si * 137
        np.random.seed(seed)
        phi_i = np.random.uniform(-np.pi, np.pi, N)
        psi_i = np.random.uniform(-np.pi, np.pi, N)
        
        if mode == '2N_baseline':
            angles = jnp.concatenate([jnp.array(phi_i), jnp.array(psi_i)])
        else:
            # χ₁ starts at common rotamer states (randomized across −60°, 60°, 180°)
            chi1_init = np.random.choice(
                [np.radians(-60), np.radians(60), np.radians(180)], N)
            # Glycines have no sidechain → χ₁ irrelevant, set to 0
            for i, aa in enumerate(seq):
                if aa == 'G':
                    chi1_init[i] = 0.0
            angles = jnp.concatenate([
                jnp.array(phi_i), jnp.array(psi_i), jnp.array(chi1_init)])
        
        t0 = time.time()
        if si == 0:
            _ = loss_jit(angles)
            _ = grad_jit(angles)
            print(f"  JIT compiled ({n_dof} DOF) in {time.time()-t0:.1f}s", flush=True)
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
        
        # Build backbone from φ, ψ only (χ₁ doesn't affect backbone coords)
        phi_f = angles[:N]; psi_f = angles[N:2*N]
        coords = _torsions_to_backbone(phi_f, psi_f, N)
        bb = np.array(coords.reshape(N, 3, 3))
        ca = bb[:, 1, :]
        rg = np.sqrt(np.mean(np.sum((ca - ca.mean(0))**2, 1)))
        rg_err = 100 * abs(rg - Rg_eq) / Rg_eq
        rmsd = kabsch_rmsd(ca, ca_native[:N])
        
        # Report χ₁ distribution if in 3N mode
        chi1_info = ""
        if mode == '3N_chi1':
            chi1_final = np.degrees(np.array(angles[2*N:]))
            # Wrap to [-180, 180]
            chi1_final = (chi1_final + 180) % 360 - 180
            chi1_info = f" χ₁=[{chi1_final.mean():.0f}°±{chi1_final.std():.0f}°]"
        
        print(f"  start {si}: loss={loss:.4f} RMSD={rmsd:.2f}Å Rg={rg:.1f}Å({rg_err:.0f}%){chi1_info} ({dt:.0f}s)", 
              flush=True)
        
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_loss = loss
            best_angles = angles
    
    print(f"\n  BEST {mode}: RMSD={best_rmsd:.2f}Å  loss={best_loss:.4f}")
    
    # Report final χ₁ rotamer distribution
    if mode == '3N_chi1' and best_angles is not None:
        chi1_f = np.degrees(np.array(best_angles[2*N:]))
        chi1_f = (chi1_f + 180) % 360 - 180
        print(f"  χ₁ distribution (best start):")
        for i, (aa, c1) in enumerate(zip(seq, chi1_f)):
            rotamer = "g+" if -120 < c1 < 0 else ("t" if c1 > 120 or c1 < -120 else "g-")
            print(f"    {i:2d} {aa} χ₁={c1:+6.1f}° ({rotamer})")

print(f"\n{'='*70}")
