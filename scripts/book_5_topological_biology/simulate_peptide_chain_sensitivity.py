r"""
Peptide Chain Extension & Sensitivity Analysis
===============================================
Tests two key predictions of the amino acid transmission line model:

1. CHAIN EXTENSION: Wiring N amino acids in series should narrow the
   backbone passband (longer filter = more selective), while preserving
   R-group differentiation.

2. SENSITIVITY: Sweeping atomic mass by a continuous factor should shift
   resonant peaks as f ∝ 1/√m, proving genuine LC resonance behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "mechanics"))

from spice_organic_mapper import get_inductance, get_capacitance
from ave.core.constants import Z_0, C_0

# ─── Frequency setup ───
f = np.logspace(10.5, 14.5, 10000)
w = 2 * np.pi * f


def z_L(L): return 1j * w * L
def z_C(C): return 1.0 / (1j * w * C)
def parallel(z1, z2): return (z1 * z2) / (z1 + z2)


def z_rgroup(kind):
    """Return R-group shunt impedance for a given amino acid."""
    if kind == 'glycine':
        return z_C(get_capacitance('C-H')) + z_L(get_inductance('H'))
    elif kind == 'alanine':
        z_rh = z_C(get_capacitance('C-H')) + z_L(get_inductance('H'))
        return z_C(get_capacitance('C-C')) + z_L(get_inductance('C')) + z_rh / 3.0
    elif kind == 'valine':
        z_rh = z_C(get_capacitance('C-H')) + z_L(get_inductance('H'))
        z_methyl = z_C(get_capacitance('C-C')) + z_L(get_inductance('C')) + z_rh / 3.0
        z_beta_split = parallel(z_rh, parallel(z_methyl, z_methyl))
        return z_C(get_capacitance('C-C')) + z_L(get_inductance('C')) + z_beta_split
    raise ValueError(f"Unknown: {kind}")


def compute_peptide_chain(residues):
    """
    Compute transfer function for a peptide chain of N residues.
    Each residue is: NH-Cα(R)-C(=O) connected by peptide bonds.
    The chain is: Source → [residue₁ → peptide bond → residue₂ → ...] → Sink.
    """
    # Start from the load (work backwards through the chain)
    Z_current = Z_0  # Vacuum impedance termination

    # For each residue (from C-terminus to N-terminus)
    for i, aa in enumerate(reversed(residues)):
        # Peptide bond between residues (C-N, not first residue)
        if i > 0:
            Z_current = z_C(get_capacitance('C-N')) + Z_current

        # Carboxyl group of this residue (only on last = C-terminal)
        if i == 0:
            Z_out = z_L(get_inductance('O')) + Z_current
            Z_co_single = z_C(get_capacitance('C-O')) + Z_out
            Z_o_double = z_C(get_capacitance('C=O')) + z_L(get_inductance('O'))
            Z_current = parallel(Z_o_double, Z_co_single)

        # Backbone carbon
        Z_carb = z_L(get_inductance('C')) + Z_current

        # Alpha carbon with R-group shunt
        Z_alpha_out = z_C(get_capacitance('C-C')) + Z_carb
        Z_alpha_main = z_L(get_inductance('C')) + Z_alpha_out
        Z_alpha = parallel(z_rgroup(aa), Z_alpha_main)

        Z_current = Z_alpha

    # Amino source (N-terminus)
    Z_amino = z_C(get_capacitance('C-N')) + Z_current
    Z_in = z_L(get_inductance('N')) + Z_amino

    # Transfer function (simplified: ratio at load)
    # For multi-stage, use cascaded voltage division
    H = Z_current / Z_in
    return H


# ═══════════════════════════════════════════════════════════
# PART 1: CHAIN LENGTH EXTENSION TEST
# ═══════════════════════════════════════════════════════════
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.97, hspace=0.30, wspace=0.25)

nu = f / (C_0 * 100)

# Polyglycine chains of increasing length
chain_lengths = [1, 2, 5, 10]
colors_chain = ['#00ffcc', '#00aaff', '#ff00aa', '#ffcc00']

ax1 = axes[0, 0]
for n, color in zip(chain_lengths, colors_chain):
    chain = ['glycine'] * n
    H = compute_peptide_chain(chain)
    P_db = 10 * np.log10(np.clip(np.abs(H)**2, 1e-30, None))
    ax1.plot(nu, P_db, color=color, linewidth=1.5, label=f'Poly-Gly (n={n})', alpha=0.85)

ax1.set_title("Chain Length Effect: Polyglycine", fontsize=12, fontweight='bold', pad=10)
ax1.set_ylabel("|H|² (dB)", fontsize=10, labelpad=8)
ax1.set_xlim(300, 4000)
ax1.set_ylim(-80, 40)
ax1.grid(True, color='#222', linestyle=':', alpha=0.5)
ax1.legend(fontsize=8, loc='upper right', facecolor='#111', edgecolor='#444')

# Polyalanine chains
ax2 = axes[0, 1]
for n, color in zip(chain_lengths, colors_chain):
    chain = ['alanine'] * n
    H = compute_peptide_chain(chain)
    P_db = 10 * np.log10(np.clip(np.abs(H)**2, 1e-30, None))
    ax2.plot(nu, P_db, color=color, linewidth=1.5, label=f'Poly-Ala (n={n})', alpha=0.85)

ax2.set_title("Chain Length Effect: Polyalanine", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlim(300, 4000)
ax2.set_ylim(-80, 40)
ax2.grid(True, color='#222', linestyle=':', alpha=0.5)
ax2.legend(fontsize=8, loc='upper right', facecolor='#111', edgecolor='#444')

# ═══════════════════════════════════════════════════════════
# PART 2: R-GROUP DIFFERENTIATION IN CHAINS
# ═══════════════════════════════════════════════════════════
ax3 = axes[1, 0]
chain_types = {
    'Poly-Gly (×5)': ['glycine'] * 5,
    'Poly-Ala (×5)': ['alanine'] * 5,
    'Poly-Val (×5)': ['valine'] * 5,
    'Mixed (G-A-V-A-G)': ['glycine', 'alanine', 'valine', 'alanine', 'glycine'],
}
colors_type = ['#00ffcc', '#ff00aa', '#ffcc00', '#ffffff']

for (label, chain), color in zip(chain_types.items(), colors_type):
    H = compute_peptide_chain(chain)
    P_db = 10 * np.log10(np.clip(np.abs(H)**2, 1e-30, None))
    ax3.plot(nu, P_db, color=color, linewidth=1.5, label=label, alpha=0.85)

ax3.set_title("R-Group Differentiation at Chain Length 5", fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=10, labelpad=8)
ax3.set_ylabel("|H|² (dB)", fontsize=10, labelpad=8)
ax3.set_xlim(300, 4000)
ax3.set_ylim(-80, 40)
ax3.grid(True, color='#222', linestyle=':', alpha=0.5)
ax3.legend(fontsize=8, loc='upper right', facecolor='#111', edgecolor='#444')

# ═══════════════════════════════════════════════════════════
# PART 3: SENSITIVITY ANALYSIS — MASS SCALING
# ═══════════════════════════════════════════════════════════
ax4 = axes[1, 1]

# Single glycine backbone, sweep mass scaling
from spice_organic_mapper import XI_TOPO_SQ, ATOMIC_MASS_DA
DA = 1.66053906660e-27

mass_scales = [0.5, 0.75, 1.0, 1.5, 2.0]
colors_mass = ['#ff4444', '#ff8800', '#00ffcc', '#4488ff', '#8844ff']

for scale, color in zip(mass_scales, colors_mass):
    # Override inductances for this sweep
    L_H_s = (1.00794 * DA * scale) / XI_TOPO_SQ
    L_C_s = (12.0107 * DA * scale) / XI_TOPO_SQ
    L_N_s = (14.0067 * DA * scale) / XI_TOPO_SQ
    L_O_s = (15.9994 * DA * scale) / XI_TOPO_SQ

    # Glycine backbone with scaled masses
    z_rg = z_C(get_capacitance('C-H')) + z_L(L_H_s)
    Z_load = Z_0
    Z_out = z_L(L_O_s) + Z_load
    Z_co_single = z_C(get_capacitance('C-O')) + Z_out
    Z_o_double = z_C(get_capacitance('C=O')) + z_L(L_O_s)
    Z_split = parallel(Z_o_double, Z_co_single)
    Z_carb = z_L(L_C_s) + Z_split
    Z_alpha_out = z_C(get_capacitance('C-C')) + Z_carb
    Z_alpha_main = z_L(L_C_s) + Z_alpha_out
    Z_alpha = parallel(z_rg, Z_alpha_main)
    Z_amino = z_C(get_capacitance('C-N')) + Z_alpha
    Z_in = z_L(L_N_s) + Z_amino
    H = (Z_alpha / Z_in) * (Z_split / Z_alpha_main) * (Z_load / Z_co_single)

    P_db = 10 * np.log10(np.clip(np.abs(H)**2, 1e-30, None))
    f_peak = f[np.argmax(np.abs(H)**2)]
    nu_peak = f_peak / (C_0 * 100)
    ax4.plot(nu, P_db, color=color, linewidth=1.5,
             label=f'm×{scale:.2f} (peak: {nu_peak:.0f} cm⁻¹)', alpha=0.85)

ax4.set_title(r"Sensitivity: Mass Scaling (Glycine, $f \propto 1/\sqrt{m}$)",
              fontsize=12, fontweight='bold', pad=10)
ax4.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=10, labelpad=8)
ax4.set_xlim(300, 4000)
ax4.set_ylim(-80, 40)
ax4.grid(True, color='#222', linestyle=':', alpha=0.5)
ax4.legend(fontsize=7, loc='upper right', facecolor='#111', edgecolor='#444')

fig.suptitle(r"Amino Acid SPICE Model — Peptide Chain Extension & Sensitivity Analysis",
             fontsize=15, fontweight='bold', color='white', y=0.97)

out_dir = PROJECT_ROOT / "assets" / "sim_outputs"
os.makedirs(out_dir, exist_ok=True)
out_path = out_dir / "amino_acid_chain_sensitivity.png"
plt.savefig(out_path, dpi=300, facecolor='black', edgecolor='none',
            bbox_inches='tight', pad_inches=0.3)
print(f"Saved → {out_path}")

# ─── Verify f ∝ 1/√m ───
print("\n  MASS SENSITIVITY VERIFICATION (f ∝ 1/√m):")
print(f"  {'Scale':>8}  {'Peak (cm⁻¹)':>14}  {'Predicted ratio':>16}  {'√(1/scale)':>12}  {'Match':>8}")
peaks = []
for scale in mass_scales:
    L_H_s = (1.00794 * DA * scale) / XI_TOPO_SQ
    L_C_s = (12.0107 * DA * scale) / XI_TOPO_SQ
    L_N_s = (14.0067 * DA * scale) / XI_TOPO_SQ
    L_O_s = (15.9994 * DA * scale) / XI_TOPO_SQ
    z_rg = z_C(get_capacitance('C-H')) + z_L(L_H_s)
    Z_load = Z_0
    Z_out = z_L(L_O_s) + Z_load
    Z_co = z_C(get_capacitance('C-O')) + Z_out
    Z_od = z_C(get_capacitance('C=O')) + z_L(L_O_s)
    Z_sp = parallel(Z_od, Z_co)
    Z_cb = z_L(L_C_s) + Z_sp
    Z_ao = z_C(get_capacitance('C-C')) + Z_cb
    Z_am = z_L(L_C_s) + Z_ao
    Z_al = parallel(z_rg, Z_am)
    Z_an = z_C(get_capacitance('C-N')) + Z_al
    Z_in = z_L(L_N_s) + Z_an
    H = (Z_al / Z_in) * (Z_sp / Z_am) * (Z_load / Z_co)
    f_pk = f[np.argmax(np.abs(H)**2)]
    peaks.append(f_pk / (C_0 * 100))

ref = peaks[mass_scales.index(1.0)]
for scale, pk in zip(mass_scales, peaks):
    ratio = pk / ref
    expected = np.sqrt(1.0 / scale)
    match = "✓" if abs(ratio - expected) / expected < 0.05 else "✗"
    print(f"  {scale:>8.2f}  {pk:>14.1f}  {ratio:>16.4f}  {expected:>12.4f}  {match:>8}")
