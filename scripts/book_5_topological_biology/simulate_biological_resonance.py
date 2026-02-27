r"""
Simulate Biological Resonance — Zero-Parameter AVE Derivation
=============================================================
Solves the exact SPICE LTI Transfer Function for amino acids using
the axiom-derived L = m/ξ² and C = ξ²/k mapping.

Outputs a Bode plot (power transmission vs frequency) comparing
6 representative amino acids across the biological IR band.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Fix path to import ave modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "mechanics"))

from spice_organic_mapper import (
    get_inductance,
    get_capacitance,
    XI_TOPO_SQ,
)
from ave.core.constants import Z_0, C_0

# All values are now in SI (Henries, Farads) — no unit prefixes needed.

def parallel(z1, z2):
    return (z1 * z2) / (z1 + z2)

# Frequency sweep: 100 GHz → 300 THz (covers IR molecular modes)
f = np.logspace(11, 14.5, 8000)
w = 2 * np.pi * f

def z_L(L_H):
    """Impedance of inductor. L in Henries."""
    return 1j * w * L_H

def z_C(C_F):
    """Impedance of capacitor. C in Farads."""
    return 1.0 / (1j * w * C_F)

# ---------------------------------------------------------
# R-GROUP SHUNT FILTER DEFINITIONS
# ---------------------------------------------------------
def z_rgroup_glycine():
    """Glycine R-Group: Single Hydrogen atom (-H)"""
    return z_C(get_capacitance('C-H')) + z_L(get_inductance('H'))

def z_rgroup_alanine():
    """Alanine R-Group: Methyl group (-CH3)"""
    z_rh_branch = z_C(get_capacitance('C-H')) + z_L(get_inductance('H'))
    z_rh_split = z_rh_branch / 3.0
    return z_C(get_capacitance('C-C')) + z_L(get_inductance('C')) + z_rh_split

def z_rgroup_valine():
    """Valine R-Group: Isopropyl group -CH(CH3)2"""
    z_rh = z_C(get_capacitance('C-H')) + z_L(get_inductance('H'))
    z_methyl = z_C(get_capacitance('C-C')) + z_L(get_inductance('C')) + (z_rh / 3.0)
    z_beta_split = parallel(z_rh, parallel(z_methyl, z_methyl))
    return z_C(get_capacitance('C-C')) + z_L(get_inductance('C')) + z_beta_split

def z_rgroup_serine():
    """Serine R-Group: Hydroxymethyl group -CH2-OH"""
    z_rh = z_C(get_capacitance('C-H')) + z_L(get_inductance('H'))
    z_oh = z_C(get_capacitance('O-H')) + z_L(get_inductance('H'))
    z_oh_branch = z_C(get_capacitance('C-O')) + z_L(get_inductance('O')) + z_oh
    z_beta_split = parallel(z_rh / 2.0, z_oh_branch)
    return z_C(get_capacitance('C-C')) + z_L(get_inductance('C')) + z_beta_split

def z_rgroup_cysteine():
    """Cysteine R-Group: Thiomethyl group -CH2-SH"""
    z_rh = z_C(get_capacitance('C-H')) + z_L(get_inductance('H'))
    z_sh = z_C(get_capacitance('S-H')) + z_L(get_inductance('H'))
    z_sh_branch = z_C(get_capacitance('C-S')) + z_L(get_inductance('S')) + z_sh
    z_beta_split = parallel(z_rh / 2.0, z_sh_branch)
    return z_C(get_capacitance('C-C')) + z_L(get_inductance('C')) + z_beta_split

def z_rgroup_phenylalanine():
    """Phenylalanine R-Group: Benzyl group -CH2-Phenyl"""
    z_rh = z_C(get_capacitance('C-H')) + z_L(get_inductance('H'))
    l_ring = 6 * get_inductance('C') + 5 * get_inductance('H')
    c_ring_bond = get_capacitance('C-C')
    z_ring = z_C(c_ring_bond) + z_L(l_ring)
    z_beta_split = parallel(z_rh / 2.0, z_ring)
    return z_C(get_capacitance('C-C')) + z_L(get_inductance('C')) + z_beta_split

# ---------------------------------------------------------
# BACKBONE LADDER NETWORK SOLVER
# ---------------------------------------------------------
def compute_transfer_function(z_rgroup):
    """
    Computes V_out / V_in for the standard amino acid backbone,
    treating the R-group as a shunt from the Alpha-Carbon.
    """
    # 1. The Sink (Carboxyl COO-)
    Z_load = Z_0
    Z_out_branch = z_L(get_inductance('O')) + Z_load
    Z_co_single_branch = z_C(get_capacitance('C-O')) + Z_out_branch

    Z_o_double_shunt = z_C(get_capacitance('C=O')) + z_L(get_inductance('O'))
    Z_split = parallel(Z_o_double_shunt, Z_co_single_branch)

    Z_carboxyl_c = z_L(get_inductance('C')) + Z_split
    Z_alpha_out = z_C(get_capacitance('C-C')) + Z_carboxyl_c

    # 2. The Chassis (Alpha-Carbon)
    Z_alpha_main = z_L(get_inductance('C')) + Z_alpha_out
    Z_alpha = parallel(z_rgroup, Z_alpha_main)

    # 3. The Source (Amino NH3+)
    Z_amino = z_C(get_capacitance('C-N')) + Z_alpha
    Z_in = z_L(get_inductance('N')) + Z_amino

    # Transfer function H(f) = V_load / V_in
    H = (Z_alpha / Z_in) * (Z_split / Z_alpha_main) * (Z_load / Z_co_single_branch)
    return H

# ---------------------------------------------------------
# EXECUTE & PLOT
# ---------------------------------------------------------
if __name__ == "__main__":
    amino_acids = {
        'Glycine (-H)':             z_rgroup_glycine(),
        'Alanine (-CH₃)':           z_rgroup_alanine(),
        'Valine (-CH(CH₃)₂)':      z_rgroup_valine(),
        'Serine (-CH₂OH)':         z_rgroup_serine(),
        'Cysteine (-CH₂SH)':       z_rgroup_cysteine(),
        'Phenylalanine (-CH₂-Ring)': z_rgroup_phenylalanine(),
    }

    colors = ['#00ffcc', '#ff00aa', '#ffcc00', '#00ccff', '#ff5500', '#b266ff']

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.96, hspace=0.15)

    for (name, z_rg), color in zip(amino_acids.items(), colors):
        H = compute_transfer_function(z_rg)
        P_db = 10 * np.log10(np.clip(np.abs(H)**2, 1e-30, None))
        phase = np.angle(H, deg=True)

        ax1.plot(f / 1e12, P_db, color=color, label=name, linewidth=1.8, alpha=0.85)
        ax2.plot(f / 1e12, phase, color=color, linewidth=1.0, alpha=0.7)

    ax1.set_title(
        r"Amino Acid RLC Transfer Functions — Zero-Parameter AVE Derivation"
        "\n" + r"$L = m/\xi^2_{topo}$,  $C = \xi^2_{topo}/k$,  $\xi_{topo} = e/\ell_{node}$",
        fontsize=14, fontweight='bold', color='white', pad=15
    )
    ax1.set_ylabel("Power Transmission |H|² (dB)", fontsize=12, labelpad=10)
    ax1.grid(True, color='#333333', linestyle='--', alpha=0.7)
    ax1.set_ylim(-120, 40)
    ax1.legend(fontsize=9, loc='upper right', facecolor='#111111', edgecolor='#444',
               ncol=2, framealpha=0.9)

    # Mark known IR absorption bands — labels placed inside plot with offset
    ir_bands = [(1000, 'C-C'), (1650, 'C=C'), (1700, 'C=O'), (3000, 'C-H'), (3400, 'N-H')]
    y_offsets = [-105, -95, -105, -95, -105]  # alternating heights to avoid overlap
    for i, (nu_cm, label) in enumerate(ir_bands):
        f_hz = nu_cm * C_0 * 100
        ax1.axvline(f_hz / 1e12, color='white', alpha=0.2, linestyle=':', linewidth=0.8)
        ax1.text(f_hz / 1e12, y_offsets[i], f'{label}\n{nu_cm} cm⁻¹', fontsize=7,
                 color='#aaaaaa', alpha=0.7, ha='center', va='bottom')

    ax2.set_xlabel("Frequency (THz)", fontsize=12, labelpad=8)
    ax2.set_ylabel("Phase (degrees)", fontsize=12, labelpad=10)
    ax2.grid(True, color='#333333', linestyle='--', alpha=0.7)
    ax2.set_xscale('log')

    # Output
    out_dir = PROJECT_ROOT / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "amino_acid_resonance.png"

    plt.savefig(out_path, dpi=300, facecolor='black', edgecolor='none',
                bbox_inches='tight', pad_inches=0.3)
    print(f"Saved → {out_path}")

    # Print diagnostic summary
    print("\n  Frequency band summary:")
    for name, z_rg in amino_acids.items():
        H = compute_transfer_function(z_rg)
        P = np.abs(H)**2
        f_peak = f[np.argmax(P)]
        nu_peak = f_peak / (C_0 * 100)
        print(f"    {name:30s}  peak: {f_peak/1e12:.1f} THz  ({nu_peak:.0f} cm⁻¹)")

