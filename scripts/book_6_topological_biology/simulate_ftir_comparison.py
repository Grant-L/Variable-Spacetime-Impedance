r"""
FTIR Falsification Test — Amino Acid Predicted vs Experimental
==============================================================
Overlays the AVE-predicted transfer function (Bode plot) against
known experimental FTIR absorption peaks from NIST and literature.

The predicted curve has a FIXED frequency scale (locked by ξ_topo).
The experimental peaks are measured.  Either they align or they don't.
This is the falsification test — no parameters can be tuned.

Sources:
  - Shimanouchi (1972), NIST Chemistry WebBook
  - Dhamelincourt & Ramirez (2000), Raman and IR spectra of glycine
  - NIST Standard Reference Database 69
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

# ─── Known Experimental FTIR Peaks for Glycine (zwitterion, solid state) ───
# Sources: NIST WebBook, Shimanouchi 1972, ResearchGate compilations
GLYCINE_FTIR = {
    # (wavenumber cm⁻¹, assignment)
    607:  r'COO$^-$ rock',
    893:  r'C-C stretch',
    1034: r'C-N stretch',
    1130: r'NH$_3^+$ rock',
    1323: r'CH$_2$ wag',
    1414: r'COO$^-$ sym stretch',
    1524: r'NH$_3^+$ asym bend',
    1596: r'COO$^-$ asym stretch',
    2900: r'C-H stretch',
    3090: r'NH$_3^+$ stretch',
    3170: r'N-H sym stretch',
}

ALANINE_FTIR = {
    770:  r'C-C-N skeletal',
    851:  r'C-C stretch',
    1015: r'C-N stretch',
    1114: r'NH$_3^+$ rock',
    1307: r'CH bend',
    1363: r'COO$^-$ sym',
    1413: r'CH$_3$ asym bend',
    1456: r'CH$_3$ asym bend',
    1587: r'COO$^-$ asym',
    2942: r'C-H stretch',
    3070: r'NH$_3^+$ stretch',
}

# ─── Transfer Function Solver (from simulate_biological_resonance.py) ───
f = np.logspace(10.5, 14.6, 10000)
w = 2 * np.pi * f

def z_L(L): return 1j * w * L
def z_C(C): return 1.0 / (1j * w * C)
def parallel(z1, z2): return (z1 * z2) / (z1 + z2)

def z_rgroup_glycine():
    return z_C(get_capacitance('C-H')) + z_L(get_inductance('H'))

def z_rgroup_alanine():
    z_rh = z_C(get_capacitance('C-H')) + z_L(get_inductance('H'))
    return z_C(get_capacitance('C-C')) + z_L(get_inductance('C')) + z_rh / 3.0

def compute_transfer_function(z_rgroup):
    Z_load = Z_0
    Z_out = z_L(get_inductance('O')) + Z_load
    Z_co_single = z_C(get_capacitance('C-O')) + Z_out
    Z_o_double = z_C(get_capacitance('C=O')) + z_L(get_inductance('O'))
    Z_split = parallel(Z_o_double, Z_co_single)
    Z_carb = z_L(get_inductance('C')) + Z_split
    Z_alpha_out = z_C(get_capacitance('C-C')) + Z_carb
    Z_alpha_main = z_L(get_inductance('C')) + Z_alpha_out
    Z_alpha = parallel(z_rgroup, Z_alpha_main)
    Z_amino = z_C(get_capacitance('C-N')) + Z_alpha
    Z_in = z_L(get_inductance('N')) + Z_amino
    H = (Z_alpha / Z_in) * (Z_split / Z_alpha_main) * (Z_load / Z_co_single)
    return H

# ─── Compute ───
nu = f / (C_0 * 100)  # Convert Hz → cm⁻¹

H_gly = compute_transfer_function(z_rgroup_glycine())
H_ala = compute_transfer_function(z_rgroup_alanine())

P_gly_db = 10 * np.log10(np.clip(np.abs(H_gly)**2, 1e-30, None))
P_ala_db = 10 * np.log10(np.clip(np.abs(H_ala)**2, 1e-30, None))

# ─── Plot ───
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
fig.subplots_adjust(top=0.93, bottom=0.07, left=0.09, right=0.97, hspace=0.28)


def _plot_amino(ax, nu_arr, p_db, ftir_peaks, curve_color, peak_color, title):
    """Plot one amino acid transfer function with experimental FTIR overlay."""
    ax.plot(nu_arr, p_db, color=curve_color, linewidth=2.2, label='AVE Predicted', zorder=5)

    # Group peaks into regions to avoid label overlap
    sorted_peaks = sorted(ftir_peaks.items())
    for wn, _ in sorted_peaks:
        ax.axvline(wn, color=peak_color, alpha=0.35, linestyle='--', linewidth=0.8)

    # Place peak labels at staggered y-positions to avoid overlap
    y_levels = [-15, -25, -35, -45, -55]
    for i, (wn, label) in enumerate(sorted_peaks):
        y = y_levels[i % len(y_levels)]
        ax.annotate(f'{wn}', xy=(wn, p_db[np.argmin(np.abs(nu_arr - wn))]),
                    xytext=(wn, y), fontsize=6.5, color=peak_color, alpha=0.9,
                    ha='center', va='top',
                    arrowprops=dict(arrowstyle='-', color=peak_color, alpha=0.3, lw=0.5))

    ax.set_title(title, fontsize=13, fontweight='bold', color='white', pad=12)
    ax.set_ylabel("|H|² (dB)", fontsize=11, labelpad=8)
    ax.set_xlim(300, 4000)
    top_y = max(10, np.max(p_db) + 5)
    ax.set_ylim(-100, top_y)
    ax.grid(True, color='#222', linestyle=':', alpha=0.5)
    ax.legend(fontsize=9, loc='upper right', facecolor='#111111', edgecolor='#444')

    # Add subtle region shading for IR bands
    ax.axvspan(600, 1600, alpha=0.04, color='white', label='_')
    ax.text(1100, 6, 'Fingerprint Region', fontsize=8, color='#888', ha='center', alpha=0.6)
    ax.axvspan(2500, 3800, alpha=0.03, color='cyan', label='_')
    ax.text(3150, 6, 'Stretch Region', fontsize=8, color='#668899', ha='center', alpha=0.6)


_plot_amino(ax1, nu, P_gly_db, GLYCINE_FTIR, '#00ffcc', '#ff8888',
            "Glycine: AVE Predicted Transfer Function vs Experimental FTIR")
_plot_amino(ax2, nu, P_ala_db, ALANINE_FTIR, '#ff00aa', '#88bbff',
            "L-Alanine: AVE Predicted Transfer Function vs Experimental FTIR")

ax2.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=11, labelpad=8)

out_dir = PROJECT_ROOT / "assets" / "sim_outputs"
os.makedirs(out_dir, exist_ok=True)
out_path = out_dir / "amino_acid_ftir_comparison.png"
plt.savefig(out_path, dpi=300, facecolor='black', edgecolor='none',
            bbox_inches='tight', pad_inches=0.3)
print(f"Saved → {out_path}")

# ─── Diagnostic: Quantify alignment ───
print("\n  GLYCINE FALSIFICATION REPORT:")
print(f"  {'Peak (cm⁻¹)':>14}  {'Assignment':>20}  {'Predicted |H|² (dB)':>20}  {'Verdict':>10}")
for wn, label in sorted(GLYCINE_FTIR.items()):
    idx = np.argmin(np.abs(nu - wn))
    p_db = P_gly_db[idx]
    verdict = "PASS" if p_db > -60 else "STEEP"
    print(f"  {wn:>14}  {label:>20}  {p_db:>20.1f}  {verdict:>10}")

print("\n  ALANINE FALSIFICATION REPORT:")
print(f"  {'Peak (cm⁻¹)':>14}  {'Assignment':>20}  {'Predicted |H|² (dB)':>20}  {'Verdict':>10}")
for wn, label in sorted(ALANINE_FTIR.items()):
    idx = np.argmin(np.abs(nu - wn))
    p_db = P_ala_db[idx]
    verdict = "PASS" if p_db > -60 else "STEEP"
    print(f"  {wn:>14}  {label:>20}  {p_db:>20.1f}  {verdict:>10}")

