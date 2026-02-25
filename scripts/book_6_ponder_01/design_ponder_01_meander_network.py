#!/usr/bin/env python3
"""
PONDER-01: Passive Phase Meander-Line Calculator
================================================

This script calculates the exact physical PCBA copper trace lengths required 
to synthesize an 8-element $C_0$ symmetric phased array from a single RF source.

Instead of synchronizing 8 independent RF amplifiers, we will inject a single 
100 MHz source into a central node, and split it 8 ways. By purposefully 
lengthening each sequential copper trace (meandering), the signal transit time 
inherently creates a progressive phase delay ($\\Delta \\phi = 45^\\circ$).

This calculates the $v_f$ (Velocity Factor) of a standard FR-4 microstrip to 
determine the explicit millimeter routing layout.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Bind into the AVE constants
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.ave.core.constants import C_0

def design_meander_network():
    print("[*] Calculating PONDER-01 Passive Phase-Delay Routing...")
    
    # Target Parameters
    FREQUENCY = 100.0e6  # 100 MHz Target
    C_VACUUM = float(C_0) # m/s
    WAVELENGTH_VACUUM = C_VACUUM / FREQUENCY # c / f (meters)
    
    # Substrate Constants (Standard FR-4 PCBA)
    ER_FR4 = 4.4 # Relative Permittivity (Dielectric Constant)
    
    # Microstrip Effective Permittivity approximation (simplified generic)
    # E_eff = (Er + 1)/2 + (Er - 1)/(2 * sqrt(1 + 12 * h/w))
    # Assuming W/h ~= 2.0 (standard 50 Ohm trace geometry)
    W_over_h = 2.0
    E_EFF = (ER_FR4 + 1.0)/2.0 + (ER_FR4 - 1.0)/(2.0 * np.sqrt(1.0 + 12.0 / W_over_h))
    
    # Propagation Velocity (v_p) and Wavelength in the PCB (\lambda_g)
    VELOCITY_FACTOR = 1.0 / np.sqrt(E_EFF)
    WAVELENGTH_PCB = WAVELENGTH_VACUUM * VELOCITY_FACTOR
    
    print(f"[*] Substrate: FR-4 (Er = {ER_FR4}) -> Effective Er_eff = {E_EFF:.2f}")
    print(f"[*] Velocity Factor (Vf): {VELOCITY_FACTOR:.3f} c")
    print(f"[*] 100 MHz Wavelength in FR-4 Trace: {WAVELENGTH_PCB * 1000.0:.1f} mm")
    
    # Element Phase Requirements for a traveling 360-degree wave across 8 elements
    NUM_ELEMENTS = 8
    PHASE_STEP_DEG = 360.0 / NUM_ELEMENTS
    
    # Phase step as a fraction of physical wavelength
    WAVELENGTH_FRACTION = PHASE_STEP_DEG / 360.0
    DELTA_LENGTH = WAVELENGTH_PCB * WAVELENGTH_FRACTION # Meters per step
    
    base_length = 50.0 / 1000.0 # 50 mm base trace length for the first element
    
    trace_lengths_m = [base_length + (i * DELTA_LENGTH) for i in range(NUM_ELEMENTS)]
    trace_lengths_mm = np.array(trace_lengths_m) * 1000.0
    
    # -------------------------------------------------------------
    # Visualization: Trace Routing Delta
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar chart showing the required trace lengths
    elements = np.arange(1, NUM_ELEMENTS + 1)
    color_map = plt.cm.viridis(np.linspace(0, 1, NUM_ELEMENTS))
    
    bars = ax.bar(elements, trace_lengths_mm, color=color_map, edgecolor='black')
    
    # Annotate the delta step
    for i, bar in enumerate(bars):
        height = bar.get_height()
        phase = i * PHASE_STEP_DEG
        # Label the absolute length
        ax.text(bar.get_x() + bar.get_width()/2., height - 20,
                f'{height:.1f} mm\n({phase:.0f}$^{{\\circ}}$)',
                ha='center', va='bottom', color='white', fontweight='bold')
    
    # Add a stair-step theoretical line showing the delta L
    ax.plot(elements, trace_lengths_mm, color='red', marker='o', linestyle=':', linewidth=2, label=f'$\\Delta L = {DELTA_LENGTH * 1000.0:.1f}$ mm/step')
    
    ax.set_title(r"PONDER-01: Passive Phased Array Trace Routing\nCopper Microstrip Meander Lengths for $45^{\circ}$ Delay @ 100 MHz", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Antenna Element Number (Sequential Ring)", fontsize=12)
    ax.set_ylabel("Required PCB Trace Length from Source (mm)", fontsize=12)
    ax.set_xticks(elements)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    
    # Add summary box
    props = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9, 'edgecolor': 'blue'}
    textstr = '\n'.join((
        f'Base Frequency: 100.0 MHz',
        f'Substrate: FR-4 (Vf = {VELOCITY_FACTOR:.3f})',
        f'Required Phase Step: {PHASE_STEP_DEG:.1f}$^{{\\circ}}$',
        '-------------------------',
        f'Meander Delay Req: +{DELTA_LENGTH * 1000.0:.1f} mm / element'
    ))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11, 
              verticalalignment='top', bbox=props)

    plt.tight_layout()
    
    # Export
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ponder_01_meander_network.png')
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Simulation complete. Output saved to: {out_path}")

if __name__ == "__main__":
    design_meander_network()
