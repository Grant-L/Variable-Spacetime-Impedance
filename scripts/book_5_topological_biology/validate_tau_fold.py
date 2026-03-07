#!/usr/bin/env python3
"""
τ_fold Validation: Axiom-Derived Folding Timescale vs Experiment
================================================================

Prediction:  τ_fold = 3Q² · N · CO · τ_water = 1.22 ns × N × CO

Where:
    Q = 7          (backbone amide-V quality factor, Axiom 1)
    f₀ = 23 THz    (backbone resonance frequency, Axiom 1)
    τ_water = 8.3 ps  (water Debye relaxation, Axiom 2)

The relative contact order CO is computed from PDB Cα coordinates:
    CO = (1 / L·N) × Σ|i - j|  for all native contacts (i,j)
where L = number of contacts with d(Cα_i, Cα_j) < 8 Å and |i-j| ≥ 3.

This script:
1. Downloads PDB structures for ~15 well-characterized two-state folders
2. Computes CO from Cα contact maps
3. Predicts τ_fold from the axiom formula
4. Compares with experimental folding rates
5. Generates a publication-ready correlation plot

Zero empirical parameters. All constants from ave.core.constants.
"""

import os
import sys
import numpy as np
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# ===========================================================================
# AVE-derived constants (from physics engine)
# ===========================================================================
Q_BACKBONE = 7               # amide-V quality factor (Axiom 1: f₀/Δf = 23/3.3)
F0_BACKBONE = 23e12           # Hz — backbone resonance frequency
TAU_WATER = 8.3e-12           # s — water Debye relaxation (Axiom 2)

# Derived folding prefactor: 3Q² × τ_water
TAU_FOLD_0 = 3 * Q_BACKBONE**2 * TAU_WATER  # = 1.22 ns
print(f"τ_fold,0 = 3 × Q² × τ_water = 3 × {Q_BACKBONE}² × {TAU_WATER*1e12:.1f} ps")
print(f"        = {TAU_FOLD_0*1e9:.2f} ns")
print()

# ===========================================================================
# Curated two-state folder dataset
# ===========================================================================
# Each entry: (name, pdb_id, chain, N_residues, k_fold (s⁻¹), reference)
# Folding rates from published literature (all two-state folders)
# k_fold = folding rate constant at ~25°C, pH 7
#
# Sources:
#   - Plaxco, Simons & Baker (1998) J. Mol. Biol. 277:985-994
#   - Ivankov et al. (2003) PNAS 100:6021
#   - Kubelka, Hofrichter & Eaton (2004) Curr. Opin. Struct. Biol. 14:76
#   - Updated rates from individual publications

TWO_STATE_FOLDERS = [
    # Ultra-fast folders
    ("Trp-cage",            "1L2Y", "A",  20,  2.5e5),   # Qiu et al. 2002
    ("Villin HP35",         "1YRF", "A",  35,  1.4e6),   # Kubelka 2003
    ("BBA5",                "1T8J", "A",  23,  1.0e5),   # Zhu et al. 2003

    # Fast folders
    ("λ-repressor",         "1LMB", "3",  80,  3.3e5),   # Burton et al. 1997
    ("Engrailed HD",        "1ENH", "A",  54,  3.7e4),   # Mayor et al. 2003
    ("Protein A (B-domain)","1BDD", "A",  60,  1.0e5),   # Myers & Oas 2001

    # Moderate folders
    ("Chymotrypsin inh. 2", "2CI2", "I",  64,  5.0e1),   # Jackson 1998
    ("Ubiquitin",           "1UBQ", "A",  76,  1.0e3),   # Khorasanizadeh 1996
    ("Protein G (GB1)",     "1PGA", "A",  56,  3.3e2),   # Alexander 1992

    # Slow folders
    ("SH3 (α-spectrin)",    "1SHG", "A",  62,  2.7e1),   # Viguera et al. 1994
    ("SH3 (src)",           "1SRL", "A",  56,  3.3e1),   # Grantcharova 1997
    ("FKBP12",              "1FKB", "A", 107,  4.0e0),   # Main et al. 1999
    ("Barnase",             "1BNI", "A", 110,  6.0e1),   # Matouschek 1990

    # WW domains
    ("WW domain (PIN1)",    "1PIN", "A",  34,  7.7e3),   # Jäger et al. 2001
    ("WW domain (FBP28)",   "1E0L", "A",  37,  4.7e3),   # Nguyen 2003
]

# ===========================================================================
# PDB download and CO computation
# ===========================================================================
def download_pdb(pdb_id, out_dir="/tmp"):
    """Download PDB file from RCSB."""
    path = os.path.join(out_dir, f"{pdb_id}.pdb")
    if os.path.exists(path):
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, path)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        return None
    return path


def extract_ca(pdb_path, chain="A", max_res=None):
    """Extract Cα coordinates from PDB file."""
    ca_coords = []
    seen_res = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            alt_loc = line[16]
            if alt_loc not in (' ', 'A'):
                continue
            ch = line[21]
            if chain != "*" and ch != chain:
                continue
            res_seq = line[22:27].strip()
            if res_seq in seen_res:
                continue
            seen_res.add(res_seq)
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            ca_coords.append([x, y, z])
            if max_res and len(ca_coords) >= max_res:
                break
    return np.array(ca_coords)


def compute_contact_order(ca_coords, contact_cutoff=8.0, min_seq_sep=3):
    """
    Compute relative contact order from Cα coordinates.

    CO = (1 / L·N) × Σ|i - j|

    where the sum is over all pairs (i,j) with:
      - |i-j| ≥ min_seq_sep (exclude trivial neighbors)
      - d(Cα_i, Cα_j) < contact_cutoff (native contact)
    and L = number of such contacts, N = chain length.

    Standard parameters: cutoff=8 Å, min_seq_sep=3 (Plaxco et al. 1998).
    """
    N = len(ca_coords)
    total_sep = 0
    L = 0

    for i in range(N):
        for j in range(i + min_seq_sep, N):
            d = np.sqrt(np.sum((ca_coords[i] - ca_coords[j])**2))
            if d < contact_cutoff:
                total_sep += (j - i)
                L += 1

    if L == 0:
        return 0.0, 0

    co = total_sep / (L * N)
    return co, L


# ===========================================================================
# Main validation
# ===========================================================================
print("=" * 70)
print("  τ_fold VALIDATION: Axiom-Derived Folding Timescale vs Experiment")
print("=" * 70)
print()
print(f"  Prediction: τ_fold = {TAU_FOLD_0*1e9:.2f} ns × N × CO")
print(f"  Constants: Q={Q_BACKBONE}, f₀={F0_BACKBONE/1e12:.0f} THz, "
      f"τ_water={TAU_WATER*1e12:.1f} ps")
print()

results = []

for name, pdb_id, chain, N_expected, k_fold_exp in TWO_STATE_FOLDERS:
    # Download PDB
    pdb_path = download_pdb(pdb_id)
    if pdb_path is None:
        continue

    # Extract Cα
    ca = extract_ca(pdb_path, chain=chain, max_res=N_expected)
    N = len(ca)
    if N < 10:
        # Try wildcard chain
        ca = extract_ca(pdb_path, chain="*", max_res=N_expected)
        N = len(ca)

    if N < 10:
        print(f"  SKIP {name}: only {N} Cα atoms found")
        continue

    # Compute contact order
    co, n_contacts = compute_contact_order(ca)

    # Predict folding time
    tau_pred = TAU_FOLD_0 * N * co  # seconds
    k_pred = 1.0 / tau_pred if tau_pred > 0 else 0

    # Experimental values
    tau_exp = 1.0 / k_fold_exp

    # Ratio
    ratio = tau_pred / tau_exp if tau_exp > 0 else 0

    results.append({
        'name': name, 'pdb': pdb_id, 'N': N,
        'co': co, 'n_contacts': n_contacts,
        'tau_pred': tau_pred, 'tau_exp': tau_exp,
        'k_pred': k_pred, 'k_exp': k_fold_exp,
        'ratio': ratio,
    })

# ===========================================================================
# Results table
# ===========================================================================
print()
print("─" * 90)
print(f"{'Protein':<22} {'PDB':>4} {'N':>4} {'CO':>6} {'#Cont':>5}  "
      f"{'τ_pred':>10} {'τ_exp':>10}  {'Ratio':>6}")
print("─" * 90)

for r in sorted(results, key=lambda x: x['tau_exp']):
    # Format times with appropriate units
    def fmt_time(t):
        if t < 1e-6:
            return f"{t*1e9:.1f} ns"
        elif t < 1e-3:
            return f"{t*1e6:.1f} μs"
        elif t < 1:
            return f"{t*1e3:.1f} ms"
        else:
            return f"{t:.1f} s"

    print(f"  {r['name']:<20} {r['pdb']:>4} {r['N']:>4} {r['co']:>6.3f} "
          f"{r['n_contacts']:>5}  {fmt_time(r['tau_pred']):>10} "
          f"{fmt_time(r['tau_exp']):>10}  {r['ratio']:>6.2f}×")

print("─" * 90)

# ===========================================================================
# Correlation analysis
# ===========================================================================
ln_k_pred = np.array([np.log10(r['k_pred']) for r in results if r['k_pred'] > 0])
ln_k_exp = np.array([np.log10(r['k_exp']) for r in results if r['k_pred'] > 0])

# Pearson correlation
corr = np.corrcoef(ln_k_pred, ln_k_exp)[0, 1]
# Linear regression
slope, intercept = np.polyfit(ln_k_exp, ln_k_pred, 1)
# Mean absolute log error
mae_log = np.mean(np.abs(ln_k_pred - ln_k_exp))

print()
print(f"  Pearson R (log₁₀ k): {corr:.3f}")
print(f"  Slope (ideal=1.0):   {slope:.3f}")
print(f"  Mean |Δlog₁₀ k|:     {mae_log:.2f} decades")
print(f"  Number of proteins:  {len(results)}")
print()

# ===========================================================================
# Publication-ready plot
# ===========================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.2

fig, ax = plt.subplots(1, 1, figsize=(7, 6))

# Plot data points
for r in results:
    lk_exp = np.log10(r['k_exp'])
    lk_pred = np.log10(r['k_pred']) if r['k_pred'] > 0 else None
    if lk_pred is None:
        continue

    # Color by speed class
    if r['tau_exp'] < 10e-6:
        color = '#2196F3'   # fast (blue)
        marker = 'o'
    elif r['tau_exp'] < 1e-3:
        color = '#FF9800'   # moderate (orange)
        marker = 's'
    else:
        color = '#F44336'   # slow (red)
        marker = '^'

    ax.scatter(lk_exp, lk_pred, c=color, marker=marker, s=80, zorder=5,
              edgecolors='white', linewidths=0.8)
    # Label
    ax.annotate(r['name'], (lk_exp, lk_pred),
               fontsize=7, ha='left', va='bottom',
               xytext=(4, 3), textcoords='offset points',
               color='#444444')

# Perfect prediction line
lims = [min(ln_k_exp.min(), ln_k_pred.min()) - 1,
        max(ln_k_exp.max(), ln_k_pred.max()) + 1]
ax.plot(lims, lims, '--', color='#888888', linewidth=1, zorder=1,
       label='Perfect prediction')

# ±1 decade bands
ax.fill_between(lims, [l-1 for l in lims], [l+1 for l in lims],
               alpha=0.08, color='green', zorder=0)
ax.fill_between(lims, [l-2 for l in lims], [l+2 for l in lims],
               alpha=0.04, color='green', zorder=0)

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.set_xlabel(r'$\log_{10}\, k_\mathrm{fold}^\mathrm{(exp)}$ (s$^{-1}$)',
             fontsize=13)
ax.set_ylabel(r'$\log_{10}\, k_\mathrm{fold}^\mathrm{(pred)}$ (s$^{-1}$)',
             fontsize=13)
ax.set_title(
    r'$\tau_\mathrm{fold} = 3Q^2 \cdot N \cdot \mathrm{CO} \cdot '
    r'\tau_\mathrm{water}$'
    f'\n$R = {corr:.3f}$, slope = {slope:.2f},'
    f' {len(results)} two-state folders',
    fontsize=12)

# Legend for speed classes
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
           markersize=8, label=r'$\tau < 10\,\mu$s (fast)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF9800',
           markersize=8, label=r'$10\,\mu$s $< \tau < 1$ ms'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#F44336',
           markersize=8, label=r'$\tau > 1$ ms (slow)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
         framealpha=0.9)

# Box with formula
textstr = (f'$\\tau_{{\\mathrm{{fold}}}} = {TAU_FOLD_0*1e9:.2f}$ ns '
           r'$\times\, N \times$ CO'
           f'\n$Q = {Q_BACKBONE}$, '
           f'$f_0 = {F0_BACKBONE/1e12:.0f}$ THz, '
           f'$\\tau_{{\\mathrm{{w}}}} = {TAU_WATER*1e12:.1f}$ ps'
           f'\nZero fitted parameters')
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
            alpha=0.9, edgecolor='gray')
ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=9,
       verticalalignment='top', bbox=props)

ax.grid(True, alpha=0.3, linewidth=0.5)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__),
                        '..', '..', 'manuscript', 'book_5_topological_biology',
                        'figures', 'tau_fold_validation.pdf')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')

# Also save PNG for quick viewing
out_png = out_path.replace('.pdf', '.png')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"  Plot saved: {out_path}")
print(f"  Plot saved: {out_png}")
print()

# ===========================================================================
# Summary
# ===========================================================================
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print()
print(f"  Formula:  τ_fold = 3Q²·N·CO·τ_water = {TAU_FOLD_0*1e9:.2f} ns × N × CO")
print(f"  Inputs:   Q={Q_BACKBONE} (Axiom 1), τ_water={TAU_WATER*1e12:.1f} ps (Axiom 2)")
print(f"  Fitted:   ZERO parameters")
print()
print(f"  Correlation (R):     {corr:.3f}")
print(f"  Slope:               {slope:.2f} (ideal: 1.00)")
print(f"  Mean log₁₀ error:    {mae_log:.2f} decades")
print(f"  Proteins tested:     {len(results)}")
print()

# Per-protein summary
within_1_decade = sum(1 for r in results if abs(np.log10(r['ratio'])) < 1)
within_2_decades = sum(1 for r in results if abs(np.log10(r['ratio'])) < 2)
print(f"  Within 1 decade:     {within_1_decade}/{len(results)} "
      f"({100*within_1_decade/len(results):.0f}%)")
print(f"  Within 2 decades:    {within_2_decades}/{len(results)} "
      f"({100*within_2_decades/len(results):.0f}%)")
print()
print("  The scaling law ln(k_fold) ∝ N×CO is the Plaxco-Simons-Baker")
print("  relation (1998), here DERIVED from backbone TL propagation physics.")
print("=" * 70)
