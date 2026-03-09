#!/usr/bin/env python3
"""
PONDER-01: Open-Air Voltage Breakdown (Paschen Curve) Simulator
===============================================================

This script models the physical limitation of transitioning the PONDER-01 
Electrostatic/Phased Array from a Hard Vacuum environment into Standard 
Temperature and Pressure (STP) open air testing.

At roughly 1 ATM, air acts as an insulator only up to ~30 kV/cm 
(or 3 kV/mm) before catastrophic corona discharge (arcing) occurs, which 
shorts out the RF wave and destroys the topological drag potential.

This simulator plots the Paschen Curve for dry air and calculates the absolute
maximum safe driving voltage ($V_{max}$) for varying PCBA gap topographies 
from 0.1 mm to 10.0 mm.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add root to sys.path to resolve src imports if needed later
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def simulate_open_air_paschen():
    print("[*] Generating Open-Air Paschen Breakdown Curve...")
    
    # Standard constants for Dry Air (A and B are gas-specific empirical constants)
    # P = Pressure in Torr, d = Gap distance in cm
    A = 15.0  # cm^{-1} * Torr^{-1}
    B = 365.0 # V * cm^{-1} * Torr^{-1}
    gamma = 0.01 # Secondary electron emission coefficient
    
    # Standard Atmospheric Pressure (STP)
    pressure_torr = 760.0 # 1 ATM
    
    # Gap sweep (from 10 microns to 10 mm)
    gaps_mm = np.logspace(-2, 1, 500)
    gaps_cm = gaps_mm / 10.0
    
    # Paschen's Law Equation: V = (B * P * d) / (ln(A * P * d) - ln(ln(1 + 1/gamma)))
    numerator = B * pressure_torr * gaps_cm
    denominator = np.log(A * pressure_torr * gaps_cm) - np.log(np.log(1.0 + 1.0/gamma))
    
    # Ensure denominator doesn't hit pure zero leading to infs (physical minimum)
    breakdown_voltage = numerator / denominator
    
    # The analytical Paschen curve is ideal. 
    # For a sharp PCBA topology (like our 1 micron emitter cones), 
    # the extreme electric field gradient (field enhancement factor beta) 
    # lowers the breakdown voltage drastically.
    
    # Field enhancement factor (beta) approximation for sharp cone against flat plate
    # beta = h_gap / r_tip
    r_tip_micron = 1.0
    beta = gaps_mm * 1000.0 / r_tip_micron
    
    # Modified breakdown for sharp topology
    # Note: Corona discharge starts earlier than full spark breakdown.
    corona_start_voltage = breakdown_voltage / np.sqrt(np.clip(beta / 10.0, 1.0, None))
    
    # Calculate specific limits for the baseline PONDER-01 1.0 mm gap layout
    target_gap_mm = 1.0
    idx_target = np.argmin(np.abs(gaps_mm - target_gap_mm))
    v_ideal_target = breakdown_voltage[idx_target]
    v_sharp_target = corona_start_voltage[idx_target]
    
    print(f"[*] 1 ATM Open-Air Breakdown @ {target_gap_mm} mm Gap:")
    print(f"    - Ideal Parallel Plates : {v_ideal_target/1000.0:.2f} kV")
    print(f"    - Sharp PCBA Emitters   : {v_sharp_target/1000.0:.2f} kV")
    
    # -------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(gaps_mm, breakdown_voltage / 1000.0, color='blue', linewidth=3, label='Ideal Paschen Breakdown (Flat Plates)')
    ax.plot(gaps_mm, corona_start_voltage / 1000.0, color='red', linewidth=3, linestyle='--', label=f'Corona Onset (Sharp $1 \\mu m$ Emitter Pattern)')
    
    # Target Reference Lines
    ax.axvline(x=target_gap_mm, color='gray', linestyle=':', linewidth=2, label=f'Baseline {target_gap_mm} mm Target Gap')
    
    # Highlight the safe operating zones
    ax.fill_between(gaps_mm, 0, corona_start_voltage / 1000.0, color='lightgreen', alpha=0.3, label='Safe Operating Zone (Thrust Active)')
    ax.fill_between(gaps_mm, corona_start_voltage / 1000.0, 50.0, color='lightcoral', alpha=0.3, label='Corona / Arcing Zone (Signal Short)')
    
    # Vacuum target reference (how much voltage did we drop?)
    ax.axhline(y=30.0, color='purple', linestyle='-.', linewidth=2, label='Original Vacuum Target (30 kV)')
    
    # Formatting
    ax.set_xscale('log')
    ax.set_ylim([0, 40])
    ax.set_xlim([0.1, 10.0])
    
    ax.set_title("PONDER-01: Open Air Breakdown Limits (Paschen Curve)\nCorona Initiation vs Geometric Gap Size at 1 ATM", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Electrode Gap Distance $d$ (mm) [Log Scale]", fontsize=12)
    ax.set_ylabel("Breakdown Voltage ($V_{breakdown}$) [kV]", fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc='upper left')
    
    # Statistics Box
    props = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9, 'edgecolor': 'gray'}
    textstr = '\n'.join((
        r'$\mathbf{1\ mm\ Gap\ Summary}$',
        f'Vacuum Target: $30.0$ kV',
        f'Air Spark Limit: ${v_ideal_target/1000.0:.1f}$ kV',
        f'Air Corona Start: ${v_sharp_target/1000.0:.2f}$ kV',
        '-------------------------',
        f'Voltage Derating Reqd: -{(1.0 - (v_sharp_target/30000.0))*100:.1f}%',
        f'New Thrust scaling ($V^2$): {(v_sharp_target/30000.0)**2 * 100:.2f}% of max'
    ))
    ax.text(0.65, 0.15, textstr, transform=ax.transAxes, fontsize=11, 
              verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    
    # Export
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ponder_01_open_air_paschen.png')
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Simulation complete. Output saved to: {out_path}")

if __name__ == "__main__":
    simulate_open_air_paschen()
