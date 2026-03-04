#!/usr/bin/env python3
"""
Full 20-Sequence Stress Test for v4 S₁₁ Folding Engine
========================================================
Runs all 20 homopolymer sequences (10 residues each) through the
engine and reports: loss, mean angle, Rg, and pass/fail.

Pass criteria:
  - Engine converges (final_loss < initial_loss)
  - Rg is physically reasonable (2-8 Å for 10-mer)
  - Mean angle is non-degenerate (30° < θ < 160°)
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mechanics'))

from s11_fold_engine_v3_jax import fold_s11_jax

AA_NAMES = {
    'G': 'Glycine',    'A': 'Alanine',    'V': 'Valine',
    'L': 'Leucine',    'I': 'Isoleucine', 'P': 'Proline',
    'F': 'Phe',        'W': 'Trp',        'M': 'Methionine',
    'S': 'Serine',     'T': 'Threonine',  'C': 'Cysteine',
    'Y': 'Tyrosine',   'H': 'Histidine',  'D': 'Aspartate',
    'E': 'Glutamate',  'N': 'Asparagine', 'Q': 'Glutamine',
    'K': 'Lysine',     'R': 'Arginine',
}

CHAIN_LENGTH = 10
N_STEPS = 3000  # Reduced for speed (full test)

print("=" * 75)
print(f"  S₁₁ v4 Engine: 20-Sequence Stress Test ({CHAIN_LENGTH}-mer, {N_STEPS} steps)")
print("=" * 75)

results = []

for aa_code in 'GAVLIMFWPSTCYHDNEQKR':
    seq = aa_code * CHAIN_LENGTH
    name = AA_NAMES[aa_code]
    
    print(f"\n--- Poly{name} ({seq}) ---")
    t0 = time.time()
    
    try:
        coords, history, trace = fold_s11_jax(seq, n_steps=N_STEPS, lr=1e-3)
        dt = time.time() - t0
        
        # Compute metrics
        angles = []
        for i in range(1, len(seq) - 1):
            u1 = coords[i] - coords[i-1]
            u2 = coords[i+1] - coords[i]
            cos_a = np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2) + 1e-10)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
        
        mean_angle = np.mean(angles)
        rg = np.sqrt(np.mean(np.sum((coords - coords.mean(0))**2, 1)))
        
        # Pass criteria
        converged = trace[-1] < trace[0]
        rg_ok = 2.0 < rg < 8.0
        angle_ok = 30.0 < mean_angle < 160.0
        passed = converged and rg_ok and angle_ok
        
        results.append({
            'aa': aa_code, 'name': name,
            'init_loss': trace[0], 'final_loss': trace[-1],
            'angle': mean_angle, 'rg': rg,
            'time': dt, 'passed': passed,
            'error': None
        })
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  Loss: {trace[0]:.4f}→{trace[-1]:.4f}  "
              f"Angle: {mean_angle:.0f}°  Rg: {rg:.1f}Å  ({dt:.1f}s)")
        
    except Exception as e:
        dt = time.time() - t0
        results.append({
            'aa': aa_code, 'name': name,
            'init_loss': None, 'final_loss': None,
            'angle': None, 'rg': None,
            'time': dt, 'passed': False,
            'error': str(e)
        })
        print(f"  ❌ ERROR: {e}")

# Summary
print(f"\n{'=' * 75}")
print(f"  SUMMARY")
print(f"{'=' * 75}")
n_pass = sum(1 for r in results if r['passed'])
n_fail = sum(1 for r in results if not r['passed'])
print(f"\n  Passed: {n_pass}/20    Failed: {n_fail}/20")

if n_fail > 0:
    print(f"\n  Failed sequences:")
    for r in results:
        if not r['passed']:
            if r['error']:
                print(f"    {r['aa']} ({r['name']}): ERROR — {r['error']}")
            else:
                reasons = []
                if r['final_loss'] >= r['init_loss']:
                    reasons.append(f"no convergence ({r['init_loss']:.3f}→{r['final_loss']:.3f})")
                if r['rg'] and not (2.0 < r['rg'] < 8.0):
                    reasons.append(f"Rg={r['rg']:.1f}")
                if r['angle'] and not (30.0 < r['angle'] < 160.0):
                    reasons.append(f"angle={r['angle']:.0f}°")
                print(f"    {r['aa']} ({r['name']}): {', '.join(reasons)}")

print(f"\n  {'AA':>3} {'Name':<12} {'Loss₀':>8} {'Loss_f':>8} {'Angle':>6} {'Rg':>5} {'Status':>8}")
print(f"  {'-'*55}")
for r in results:
    if r['error']:
        print(f"  {r['aa']:>3} {r['name']:<12} {'—':>8} {'—':>8} {'—':>6} {'—':>5} {'ERROR':>8}")
    else:
        s = "PASS" if r['passed'] else "FAIL"
        print(f"  {r['aa']:>3} {r['name']:<12} {r['init_loss']:8.4f} {r['final_loss']:8.4f} "
              f"{r['angle']:5.0f}° {r['rg']:4.1f}Å {s:>8}")
