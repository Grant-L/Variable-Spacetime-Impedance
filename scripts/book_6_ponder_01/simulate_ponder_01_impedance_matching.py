#!/usr/bin/env python3
r"""
PONDER-01: 50-Ohm VHF Impedance Matching Simulator
===================================================

This script models the explicit L-C matching network needed to drive the 
highly capacitive PONDER-01 array at 100 MHz without destroying the RF amplifier 
through reflected power.

It outputs the required Series Inductance ($L_{match}$) to reach resonance
and plots the resulting S11 Reflection Coefficient ($\Gamma$).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_impedance_match():
    print("[*] Generating PONDER-01 RF Impedance Resonance Profile...")
    
    # Target Operation Parameters
    TARGET_FREQ = 100.0e6  # 100 MHz
    Z_SOURCE    = 50.0     # 50 Ohm Coaxial feedline and Amplifier output
    
    # -------------------------------------------------------------
    # 1. PONDER-01 Array Capacitance Calculation
    # -------------------------------------------------------------
    # We estimate the massive mutual capacitance of the 2D array
    # Array Area: 5cm x 5cm = 0.0025 m^2
    # Gap: 1 mm = 0.001 m
    # Vacuum Permittivity ~ 8.854e-12 F/m
    AREA = 0.0025
    D_GAP = 0.001
    EPSILON_0 = 8.854e-12
    
    # Parallel Plate Base Capacitance
    C_base = (EPSILON_0 * AREA) / D_GAP
    
    # Multiplier for the highly dense etched emitter array topology
    # (Sharp tips dramatically increase localized flux density)
    TOPOLOGY_MULTIPLIER = 4.5
    
    C_ARRAY = C_base * TOPOLOGY_MULTIPLIER
    print(f"[*] Calculated Array Mutual Capacitance: {C_ARRAY * 1e12:.2f} pF")
    
    # -------------------------------------------------------------
    # 2. Resonant Inductor Matching Network
    # -------------------------------------------------------------
    # To drive this pure capacitor, we must place a highly-Q inductor in series
    # to cancel the blind reactive impedance at exactly 100 MHz.
    # L = 1 / ( (2 * pi * f)^2 * C )
    
    omega_target = 2 * np.pi * TARGET_FREQ
    L_MATCH = 1.0 / ((omega_target**2) * C_ARRAY)
    
    print(f"[*] Required Series Inductance for 100 MHz Resonance: {L_MATCH * 1e9:.2f} nH")
    
    # -------------------------------------------------------------
    # 3. S11 Reflected Power Sweep
    # -------------------------------------------------------------
    # We sweep the frequency from 90 MHz to 110 MHz to see how tight the tuning is.
    frequencies = np.linspace(90e6, 110e6, 2000)
    omega = 2 * np.pi * frequencies
    
    # We assume a slight real resistance in the array due to traces (ESR)
    R_ARRAY = 0.2 # 0.2 Ohms of copper loss
    
    # Total Load Impedance: Z_load = R_esr + j*(wL - 1/wC)
    Z_load = R_ARRAY + 1j * (omega * L_MATCH - 1.0 / (omega * C_ARRAY))
    
    # Reflection Coefficient: Gamma = (Z_load - Z_source) / (Z_load + Z_source)
    # Using Z_source = 50 Ohms to represent the match network step-up transformer
    # (Note: A real hardware match requires an L-network or Balun to step 50 Ohm to 0.2 Ohm real,
    # but here we visualize the pure LC resonance Q-factor notch).
    
    # S11 parameter in decibels
    Gamma = (Z_load - Z_SOURCE) / (Z_load + Z_SOURCE)
    S11_dB = 20 * np.log10(np.abs(Gamma) + 1e-12) # Add small epsilon to prevent log(0)
    
    # -------------------------------------------------------------
    # 4. Visualization
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    plt.plot(frequencies / 1e6, S11_dB, color='purple', linewidth=2.5, label='S11 Reflected Power')
    
    # Limit lines
    plt.axhline(-10.0, color='red', linestyle='--', label='-10 dB Limit (10% Power Reflected)')
    plt.axvline(TARGET_FREQ / 1e6, color='black', linestyle=':', label='100 MHz Target')
    
    # Annotate matching parameters
    props = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9, 'edgecolor': 'gray'}
    textstr = '\n'.join((
        r'$\mathbf{Hardware\ Parameters}$',
        f'$C_{{array}} = {C_ARRAY * 1e12:.1f}$ pF',
        f'$L_{{match}} = {L_MATCH * 1e9:.1f}$ nH',
        f'$R_{{esr}} = {R_ARRAY}$ $\\Omega$'
    ))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
            
    plt.title("PONDER-01 Array: 100 MHz LC Resonance Notch ($S_{11}$)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("VHF Drive Frequency (MHz)", fontsize=12)
    plt.ylabel("Return Loss $S_{11}$ (dB)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(loc='lower left')
    
    plt.ylim(-40, 0)
    
    # Output
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ponder_01_s11_match.png')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Simulation complete. Output saved to: {out_path}")

if __name__ == "__main__":
    simulate_impedance_match()
