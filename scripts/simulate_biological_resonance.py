"""
Simulate Biological Resonance 
=============================
Solves the exact SPICE LTI Transfer Function mathematically for 
Glycine vs Alanine using the AVE topological `spice_organic_mapper`.
Outputs a transmission frequency response (Bode Plot) emphasizing 
how the R-Group stub filters the backbone AC signal.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Fix path to import ave modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ave.mechanics.spice_organic_mapper import (
    get_inductance,
    get_capacitance
)
from ave.core.constants import Z_0

def parallel(z1, z2):
    return (z1 * z2) / (z1 + z2)

# Frequencies from 100 GHz to 10 THz (biological IR resonance range)
# The topological impedance values used earlier put resonance in the sub-THz and THz bands.
f = np.logspace(11, 13.5, 5000)
w = 2 * np.pi * f

def z_L(L_pH):
    """Convert pH to jwL impedance."""
    return 1j * w * (L_pH * 1e-12)

def z_C(C_fF):
    """Convert fF to 1/jwC impedance."""
    return 1.0 / (1j * w * (C_fF * 1e-15))

# ---------------------------------------------------------
# R-GROUP SHUNT FILTER DEFINITIONS
# ---------------------------------------------------------
def z_rgroup_glycine():
    """Glycine R-Group: Single Hydrogen atom (-H)"""
    return z_C(get_capacitance('C-H')) + z_L(get_inductance('H'))

def z_rgroup_alanine():
    """Alanine R-Group: Methyl group (-CH3)"""
    # 3 Parallel H's on the terminal C
    z_rh_branch = z_C(get_capacitance('C-H')) + z_L(get_inductance('H'))
    z_rh_split = z_rh_branch / 3.0  # 3 identical branches in parallel
    return z_C(get_capacitance('C-C')) + z_L(get_inductance('C')) + z_rh_split

# ---------------------------------------------------------
# BACKBONE LADDER NETWORK SOLVER
# ---------------------------------------------------------
def compute_transfer_function(z_rgroup):
    """
    Computes V_out / V_in for the standard amino acid backbone, 
    treating the R-group as a shunt from the Alpha-Carbon.
    """
    # 1. The Sink (Carboxyl COO-)
    Z_load = Z_0  # Vacuum Impedance termination
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
    
    # Voltage Dividers (V_load / V_in)
    # H = (V_alpha / V_in) * (V_split / V_alpha) * (V_load / V_split)
    # V_alpha = V_in * (Z_alpha / Z_in) 
    # V_split = V_alpha * (Z_split / Z_alpha_main)
    # V_load  = V_split * (Z_load / Z_co_single_branch)
    H = (Z_alpha / Z_in) * (Z_split / Z_alpha_main) * (Z_load / Z_co_single_branch)
    return H

# ---------------------------------------------------------
# EXECUTE & PLOT
# ---------------------------------------------------------
if __name__ == "__main__":
    H_gly = compute_transfer_function(z_rgroup_glycine())
    H_ala = compute_transfer_function(z_rgroup_alanine())

    # Calculate Power Transmission |H|^2
    P_gly = np.abs(H_gly)**2
    P_ala = np.abs(H_ala)**2
    
    # Convert to log scale (dB) safely
    P_gly_db = 10 * np.log10(np.clip(P_gly, 1e-12, None))
    P_ala_db = 10 * np.log10(np.clip(P_ala, 1e-12, None))

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(f / 1e12, P_gly_db, color='#00ffcc', label='Glycine (-H stub)', linewidth=2.5)
    ax.plot(f / 1e12, P_ala_db, color='#ff00aa', label='L-Alanine (-CH3 stub)', linewidth=2.5)

    ax.set_title("Amino Acid RLC Resonance & R-Group Filtering", fontsize=16, fontweight='bold', color='white', pad=15)
    ax.set_xlabel("Driving Frequency (THz)", fontsize=14)
    ax.set_ylabel("Power Transmission (dB)", fontsize=14)
    ax.grid(True, color='#333333', linestyle='--', alpha=0.7)
    
    # Set y limit to emphasize resonant peaks
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=12, loc='lower right', facecolor='black', edgecolor='white')

    out_dir = PROJECT_ROOT / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "amino_acid_resonance.png"
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='black', edgecolor='none')
    print(f"Saved visualization to {out_path}")
