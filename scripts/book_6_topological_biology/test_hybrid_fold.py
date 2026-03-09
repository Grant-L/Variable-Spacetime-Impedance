#!/usr/bin/env python3
"""
Hybrid Folding Test: Cotranslational Init + Batch Refinement
=============================================================

Hypothesis: cotranslational init (sequential N→C) + batch refinement
(full-chain optimizer) can break the 5Å RMSD barrier.

Previous results:
  - Batch only (10k steps):  Trp-cage 6.2Å, Villin 6.5Å
  - Batch only (20k steps):  Trp-cage 5.24Å
  - Cotranslational only:    Villin 6.71Å (faster, but not accurate)
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
from multiscale_fold_engine import (
    _torsions_to_backbone, compute_gly_mask, compute_pro_mask,
)

# ═══════════════════════════════════════════════════════════════
# PDB utilities
# ═══════════════════════════════════════════════════════════════
AA_MAP = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q',
    'GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
    'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
    'TYR':'Y','VAL':'V'
}

def download_pdb(pdb_id):
    path = f"/tmp/{pdb_id}.pdb"
    if os.path.exists(path):
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    urllib.request.urlretrieve(url, path)
    return path

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


# ═══════════════════════════════════════════════════════════════
# Hybrid Fold: Cotranslational Init + Batch Refinement
# ═══════════════════════════════════════════════════════════════
def fold_hybrid(seq, n_cotrans_steps=200, n_batch_steps=15000, lr=2e-3,
                k0=8, n_starts=2):
    """
    Phase 1: Cotranslational initialization (sequential N→C)
    Phase 2: Batch refinement (full-chain optimizer)
    """
    from s11_fold_engine_v3_jax import (
        _torsion_loss as _s11_torsion_loss,
        compute_z_topo as compute_z_topo_v3,
        compute_cys_mask as compute_cys_mask_v3,
        compute_aromatic_mask as compute_aromatic_mask_v3,
    )
    
    N = len(seq)
    z_topo = compute_z_topo_v3(seq)
    cys_mask = compute_cys_mask_v3(seq)
    arom_mask = compute_aromatic_mask_v3(seq)
    gly_mask = compute_gly_mask(seq)
    pro_mask = compute_pro_mask(seq)
    
    # Full-chain loss function
    loss_fn = lambda a: _s11_torsion_loss(
        a, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N)
    loss_jit = jax.jit(loss_fn)
    grad_jit = jax.jit(jax.grad(loss_fn))
    
    print(f"  Hybrid fold: N={N}, cotrans={n_cotrans_steps}/res, "
          f"batch={n_batch_steps}, {n_starts}-start", flush=True)
    
    best_loss = float('inf')
    best_angles = None
    
    for start_idx in range(n_starts):
        seed = 42 + start_idx * 137
        np.random.seed(seed)
        all_phi = np.random.uniform(-np.pi, np.pi, N)
        all_psi = np.random.uniform(-np.pi, np.pi, N)
        
        t0 = time.time()
        
        # ──────────────────────────────────────────────
        # Phase 1: Cotranslational Init (sequential N→C)
        # ──────────────────────────────────────────────
        print(f"    start {start_idx} phase 1 (cotranslational)...", flush=True)
        
        k = min(k0, N)
        # Initial segment gets 5× more steps
        for init_step in range(n_cotrans_steps * 5):
            z_k = z_topo[:k]
            cys_k = cys_mask[:k]
            arom_k = arom_mask[:k]
            gly_k = gly_mask[:k]
            pro_k = pro_mask[:k]
            
            angles_k = jnp.concatenate([jnp.array(all_phi[:k]),
                                         jnp.array(all_psi[:k])])
            loss_k_fn = lambda a: _s11_torsion_loss(
                a, z_k, cys_k, arom_k, gly_k, pro_k, k)
            
            if init_step == 0:
                # JIT for this size
                loss_k_jit = jax.jit(loss_k_fn)
                grad_k_jit = jax.jit(jax.grad(loss_k_fn))
                _ = loss_k_jit(angles_k)
                optimizer_k = optax.adam(lr)
                opt_state_k = optimizer_k.init(angles_k)
            
            g = grad_k_jit(angles_k)
            g = jnp.where(jnp.isnan(g), 0.0, g)
            g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-12)
            g = jnp.where(g_norm > 10.0, g * 10.0 / g_norm, g)
            updates, opt_state_k = optimizer_k.update(g, opt_state_k)
            angles_k = optax.apply_updates(angles_k, updates)
        
        all_phi[:k] = np.array(angles_k[:k])
        all_psi[:k] = np.array(angles_k[k:])
        
        # Grow chain one residue at a time
        for k in range(k0 + 1, N + 1):
            z_k = z_topo[:k]
            cys_k = cys_mask[:k]
            arom_k = arom_mask[:k]
            gly_k = gly_mask[:k]
            pro_k = pro_mask[:k]
            
            angles_k = jnp.concatenate([jnp.array(all_phi[:k]),
                                         jnp.array(all_psi[:k])])
            
            loss_k_fn = lambda a: _s11_torsion_loss(
                a, z_k, cys_k, arom_k, gly_k, pro_k, k)
            # JIT for this size (cached after first)
            loss_k_jit = jax.jit(loss_k_fn)
            grad_k_jit = jax.jit(jax.grad(loss_k_fn))
            
            optimizer_k = optax.adam(lr)
            opt_state_k = optimizer_k.init(angles_k)
            
            for step in range(n_cotrans_steps):
                g = grad_k_jit(angles_k)
                g = jnp.where(jnp.isnan(g), 0.0, g)
                g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-12)
                g = jnp.where(g_norm > 10.0, g * 10.0 / g_norm, g)
                updates, opt_state_k = optimizer_k.update(g, opt_state_k)
                angles_k = optax.apply_updates(angles_k, updates)
            
            all_phi[:k] = np.array(angles_k[:k])
            all_psi[:k] = np.array(angles_k[k:])
        
        dt_cotrans = time.time() - t0
        
        # Get cotranslational loss
        cotrans_angles = jnp.concatenate([jnp.array(all_phi), jnp.array(all_psi)])
        cotrans_loss = float(loss_jit(cotrans_angles))
        print(f"      cotrans: loss={cotrans_loss:.4f} ({dt_cotrans:.0f}s)", flush=True)
        
        # ──────────────────────────────────────────────
        # Phase 2: Batch Refinement (full chain)  
        # ──────────────────────────────────────────────
        print(f"    start {start_idx} phase 2 (batch refinement)...", flush=True)
        t1 = time.time()
        
        # JIT compile for full N (warm up if first start)
        if start_idx == 0:
            _ = loss_jit(cotrans_angles)
            _ = grad_jit(cotrans_angles)
            print(f"      JIT compiled for N={N} in {time.time()-t1:.1f}s", flush=True)
            t1 = time.time()
        
        angles = cotrans_angles
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(angles)
        key = jax.random.PRNGKey(seed + 1000)
        
        for step in range(n_batch_steps):
            g = grad_jit(angles)
            g = jnp.where(jnp.isnan(g), 0.0, g)
            g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-12)
            g = jnp.where(g_norm > 10.0, g * 10.0 / g_norm, g)
            updates, opt_state = optimizer.update(g, opt_state)
            angles = optax.apply_updates(angles, updates)
            # Mild anneal for first 25%
            if step < n_batch_steps * 0.25:
                T = 0.03 * (1.0 - step / (n_batch_steps * 0.25)) ** 2
                key, subkey = jax.random.split(key)
                angles = angles + jax.random.normal(subkey, shape=angles.shape) * T
        
        batch_loss = float(loss_jit(angles))
        dt_batch = time.time() - t1
        dt_total = time.time() - t0
        print(f"      batch:   loss={batch_loss:.4f} ({dt_batch:.0f}s, "
              f"total {dt_total:.0f}s)", flush=True)
        
        if batch_loss < best_loss:
            best_loss = batch_loss
            best_angles = angles
    
    # Build final backbone
    phi_final = best_angles[:N]
    psi_final = best_angles[N:2*N]
    coords = _torsions_to_backbone(phi_final, psi_final, N)
    bb = np.array(coords.reshape(N, 3, 3))
    ca = bb[:, 1, :]
    
    return ca, best_loss, bb


# ═══════════════════════════════════════════════════════════════
# Test on small proteins
# ═══════════════════════════════════════════════════════════════
TEST_PROTEINS = [
    ("Trp-cage",    "1L2Y", "A",  20, "α"),
    ("Villin HP35", "1YRF", "A",  35, "α"),
    ("BBA5",        "1T8J", "A",  23, "α/β"),
]

print("=" * 70)
print("  Hybrid Fold Test: Cotranslational Init + Batch Refinement")
print("=" * 70)
print(f"  η_eq = {ETA_EQ:.6f}")
print()

for name, pdb_id, chain, nmax, fold in TEST_PROTEINS:
    pdb_path = download_pdb(pdb_id)
    ca_native, seq = extract_ca(pdb_path, chain, nmax)
    if ca_native is None:
        ca_native, seq = extract_ca(pdb_path, "*", nmax)
    
    N = len(seq)
    Rg_eq = 1.7 * (N / ETA_EQ)**(1/3) * np.sqrt(3/5)
    
    print(f"\n{'─'*70}")
    print(f"  {name} (N={N}, {fold})")
    print(f"{'─'*70}")
    
    ca, loss, bb = fold_hybrid(
        seq, 
        n_cotrans_steps=200,
        n_batch_steps=15000,
        lr=2e-3,
        n_starts=2,
    )
    
    rg = np.sqrt(np.mean(np.sum((ca - ca.mean(0))**2, 1)))
    rg_err = 100 * abs(rg - Rg_eq) / Rg_eq
    rmsd = kabsch_rmsd(ca, ca_native[:N])
    
    print(f"\n  Result: RMSD={rmsd:.2f}Å  Rg={rg:.1f}Å({rg_err:.0f}%)  loss={loss:.4f}")
    
    # Compare to batch-only baseline
    from multiscale_fold_engine import fold_multiscale
    print(f"\n  Running batch-only baseline (15k steps)...")
    ca_batch, _, trace_batch, bb_batch = fold_multiscale(
        seq, n_steps=15000, lr=2e-3, n_starts=2)
    rmsd_batch = kabsch_rmsd(ca_batch, ca_native[:N])
    rg_batch = np.sqrt(np.mean(np.sum((ca_batch - ca_batch.mean(0))**2, 1)))
    rg_err_batch = 100 * abs(rg_batch - Rg_eq) / Rg_eq
    
    print(f"  Baseline: RMSD={rmsd_batch:.2f}Å  Rg={rg_batch:.1f}Å({rg_err_batch:.0f}%)")
    print(f"  Delta:    RMSD {rmsd - rmsd_batch:+.2f}Å")

print(f"\n{'='*70}")
