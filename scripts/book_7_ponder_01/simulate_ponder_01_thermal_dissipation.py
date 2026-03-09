#!/usr/bin/env python3
"""
PONDER-01: Hard Vacuum Thermal Dissipation Simulator
===================================================

This script models the explicit heat accumulation of the PONDER-01 
substrate when driven at 30 kV / 100 MHz in a convective-dead hard vacuum environment.
It calculates both the dielectric loss heating ($P = V^2 \\omega C \\tan(\\delta)$) 
and the massive circulating skin-current $I = V / X_c$ associated with maintaining, charting the catastrophic 
thermal runaway limit (delamination) as a function of continuous runtime.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_thermal_dissipation():
    print("[*] Generating PONDER-01 Vacuum Thermal Runtime Profile...")
    
    # -------------------------------------------------------------
    # 1. PCBA Thermal & Material Constants
    # -------------------------------------------------------------
    # Power Parameters
    V_RMS = 30000.0   # 30 kV
    FREQ = 100.0e6    # 100 MHz
    
    # Imported from impedance match simulation
    MUTUAL_CAPACITANCE = 100.0e-12 # 100 pF average
    
    # PCBA Material (FR-4 vs High-Freq PTFE/Teflon)
    # Tan Delta (Dissipation Factor)
    TAN_DELTA_FR4 = 0.02   # Standard cheap PCB substrate
    TAN_DELTA_PTFE = 0.001 # Rogers / Teflon High-Frequency Substrate
    
    # Copper Skin Effect Resistance at 100 MHz
    # Skin depth in copper at 100MHz is ~6.6 microns
    # High frequency forces current to the extreme surface, increasing effective resistance
    R_SKIN = 0.15 # Ohms of equivalent series resistance for the array tracks
    I_RMS = V_RMS * (2 * np.pi * FREQ * MUTUAL_CAPACITANCE) # Massive circulating reactive current
    
    print(f"[*] Circulating Reactive Current at 100 MHz: {I_RMS:.2f} A rms")
    
    # Heat Capacities and Masses (Approximation of a 5x5cm block of substrate)
    MASS_SUBSTRATE = 0.02 # 20 grams
    SPECIFIC_HEAT = 1000.0 # J/(kg*K) ~ average for epoxy/fiberglass
    
    # -------------------------------------------------------------
    # 2. Power Dissipation Calculations (Watts)
    # -------------------------------------------------------------
    omega = 2 * np.pi * FREQ
    
    # Dielectric Heating: P_d = V^2 * omega * C * tan(delta)
    P_DIELECTRIC_FR4 = (V_RMS**2) * omega * MUTUAL_CAPACITANCE * TAN_DELTA_FR4
    P_DIELECTRIC_PTFE = (V_RMS**2) * omega * MUTUAL_CAPACITANCE * TAN_DELTA_PTFE
    
    # Ohmic/Skin Heating: P_r = I^2 * R
    P_SKIN = (I_RMS**2) * R_SKIN
    
    print(f"[*] FR4 Dielectric Heating: {P_DIELECTRIC_FR4:.0f} W")
    print(f"[*] PTFE Dielectric Heating: {P_DIELECTRIC_PTFE:.0f} W")
    print(f"[*] Copper Skin-Effect Heating: {P_SKIN:.0f} W")
    
    # Total Heating rates (in a hard-vacuum, convective cooling is 0. 
    # Stefan-Boltzmann radiative cooling at room temp is negligible compared to this thermal input).
    total_power_fr4 = P_DIELECTRIC_FR4 + P_SKIN
    total_power_ptfe = P_DIELECTRIC_PTFE + P_SKIN
    
    # dT/dt = Power / (Mass * Specific_Heat)
    dT_dt_fr4 = total_power_fr4 / (MASS_SUBSTRATE * SPECIFIC_HEAT)
    dT_dt_ptfe = total_power_ptfe / (MASS_SUBSTRATE * SPECIFIC_HEAT)
    
    # -------------------------------------------------------------
    # 3. Time Series Thermal Runaway Simulation
    # -------------------------------------------------------------
    time_steps = np.linspace(0, 10, 500) # 0 to 10 seconds tracking
    T_ambient = 25.0 # 25 Celsius
    
    temp_fr4 = T_ambient + (dT_dt_fr4 * time_steps)
    temp_ptfe = T_ambient + (dT_dt_ptfe * time_steps)
    
    # Delamination/Destruction Threshold (~150C for FR4, ~280C for PTFE)
    T_FAIL_FR4 = 150.0
    T_FAIL_PTFE = 280.0
    
    # -------------------------------------------------------------
    # 4. Visualization
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    plt.plot(time_steps, temp_fr4, color='red', linewidth=3.0, label=f'Standard FR-4 PCBA ($\\tan\\delta = {TAN_DELTA_FR4}$)')
    plt.plot(time_steps, temp_ptfe, color='blue', linewidth=3.0, label=f'High-Freq PTFE PCBA ($\\tan\\delta = {TAN_DELTA_PTFE}$)')
    
    # Highlight Delamination Thresholds
    plt.axhline(T_FAIL_FR4, color='darkred', linestyle='--', label='FR-4 Glass Transition / Delamination Limit')
    plt.axhline(T_FAIL_PTFE, color='darkblue', linestyle='--', label='PTFE Melting Point Limit')
    
    # Annotate limits
    idx_fr4 = np.argmax(temp_fr4 >= T_FAIL_FR4) if np.any(temp_fr4 >= T_FAIL_FR4) else -1
    if idx_fr4 != -1:
        time_fail_fr4 = time_steps[idx_fr4]
        plt.scatter([time_fail_fr4], [T_FAIL_FR4], color='black', zorder=5)
        plt.annotate(f"Catastrophic FR-4 Failure\n{time_fail_fr4:.1f} Sec Runtime", xy=(time_fail_fr4, T_FAIL_FR4), xytext=(time_fail_fr4 + 0.5, T_FAIL_FR4 - 80), arrowprops={'facecolor': 'black', 'shrink': 0.05, 'width': 1.5, 'headwidth': 6})
        
    idx_ptfe = np.argmax(temp_ptfe >= T_FAIL_PTFE) if np.any(temp_ptfe >= T_FAIL_PTFE) else -1
    if idx_ptfe != -1:
        time_fail_ptfe = time_steps[idx_ptfe]
        plt.scatter([time_fail_ptfe], [T_FAIL_PTFE], color='black', zorder=5)
        plt.annotate(f"PTFE Melt Point\n{time_fail_ptfe:.1f} Sec Runtime", xy=(time_fail_ptfe, T_FAIL_PTFE), xytext=(time_fail_ptfe + 0.5, T_FAIL_PTFE - 80), arrowprops={'facecolor': 'black', 'shrink': 0.05, 'width': 1.5, 'headwidth': 6})
    
    plt.xlabel("Continuous Resonance Runtime (Seconds)", fontsize=12)
    plt.ylabel(r"Core Hardware Temperature ($^{\circ}$C)", fontsize=12)
    plt.title("PONDER-01: Thermal Runaway and PCBA Delamination Limits\n(100 MHz @ 30 kV RMS in Hard Vacuum)", fontsize=14, fontweight='bold', pad=15)
    
    plt.ylim(0, 500)
    plt.xlim(0, 5)
    
    # Output
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ponder_01_thermal_dissipation.png')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Simulation complete. Output saved to: {out_path}")

if __name__ == "__main__":
    simulate_thermal_dissipation()
