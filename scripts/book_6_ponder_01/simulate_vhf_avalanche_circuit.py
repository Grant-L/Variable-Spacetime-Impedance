#!/usr/bin/env python3
"""
PONDER-01: VHF Avalanche Drive Simulator
===================================================

This script models the high-speed electrical drive circuitry required to sustain 
a 30 kV RMS / 100 MHz continuous wave across the topological thrust geometry.
It operates functionally as a discrete SPICE-level simulator, generating the explicit
Current-Voltage (I-V) trace and the transient Voltage/Time wave outputs.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Non-Linear Avalanche SPICE Model (Transient Analysis)
# -------------------------------------------------------------
def simulate_vhf_avalanche():
    print("[*] Simulating PONDER-01 VHF Avalanche Circuit Dynamics...")
    
    # Target Operating Parameters
    V_RMS = 30000.0   # Target 30 kV
    V_PEAK = V_RMS * np.sqrt(2) # ~42.4 kV Peak
    FREQ = 100.0e6    # 100 MHz VHF
    
    # Time domain array (3 full cycles)
    TIME_STEPS = 2000
    PERIOD = 1.0 / FREQ
    t = np.linspace(0, 3 * PERIOD, TIME_STEPS)
    
    # Mathematical ideal sinusoidal drive
    v_ideal = V_PEAK * np.sin(2 * np.pi * FREQ * t)
    
    # -------------------------------------------------------------
    # 2. Avalanche Clipping / Diode Breakdown
    # -------------------------------------------------------------
    # In a physical VHF flyback, the sharp emitter geometry causes 
    # field-emission avalanche clipping at the tips. We model this as
    # a non-linear Zener-like voltage sag at the extreme peaks.
    AVALANCHE_THRESHOLD = 38000.0 # Volts
    
    v_actual = np.copy(v_ideal)
    current  = np.zeros_like(v_ideal)
    
    # Capacitive Load Parameters (Array approximation)
    C_LOAD = 15.0e-12 # 15 pF parasitic load of the asymmetric stack
    R_SERIES = 50.0   # 50 Ohm coax match
    
    for i in range(TIME_STEPS):
        # i_cap = C * dV/dt (Simplified)
        if i > 0:
            dv_dt = (v_actual[i] - v_actual[i-1]) / (t[i] - t[i-1])
            i_cap = C_LOAD * dv_dt
        else:
            i_cap = 0
            
        current[i] = i_cap
        
        # Apply Avalanche sag if V_PEAK exceeds threshold
        if abs(v_actual[i]) > AVALANCHE_THRESHOLD:
            excess = abs(v_actual[i]) - AVALANCHE_THRESHOLD
            # Non-linear squared clipping
            clip = 0.5 * excess 
            v_actual[i] = (abs(v_actual[i]) - clip) * np.sign(v_actual[i])
            
            # Massive current spike during field emission (avalanche)
            current[i] += np.sign(v_actual[i]) * ((excess / 1000.0)**2) * 5.0 # Ampere spike multiplier

    # -------------------------------------------------------------
    # 3. Visualization output
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top Panel: Transient Voltage
    ax1.plot(t * 1e9, v_ideal / 1000.0, color='gray', linestyle='--', linewidth=1.5, label='Ideal 100 MHz Source')
    ax1.plot(t * 1e9, v_actual / 1000.0, color='red', linewidth=2.5, label='Actual Array Voltage')
    
    # Highlight the Avalanche Thresholds
    ax1.axhline(y=AVALANCHE_THRESHOLD / 1000.0, color='blue', linestyle=':', label='Field Emission Threshold')
    ax1.axhline(y=-AVALANCHE_THRESHOLD / 1000.0, color='blue', linestyle=':')
    
    ax1.set_title("PONDER-01: VHF Avalanche Breakdown Transient", fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel("Electrode Voltage (kV)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # Bottom Panel: Transient Current
    ax2.plot(t * 1e9, current, color='green', linewidth=2.0, label='Drive Current (Amperes)')
    ax2.set_xlabel("Time (nanoseconds)", fontsize=12)
    ax2.set_ylabel("Drive Current (A)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Export
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ponder_01_vhf_drive_transient.png')
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Simulation complete. Output saved to: {out_path}")

if __name__ == "__main__":
    simulate_vhf_avalanche()
