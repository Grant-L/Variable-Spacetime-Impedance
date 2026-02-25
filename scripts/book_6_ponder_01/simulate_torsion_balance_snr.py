#!/usr/bin/env python3
"""
PONDER-01: Torsion Balance Signal-to-Noise Simulator
===================================================

This script models the explicit macroscopic noise limits (seismic, thermal, thermal drift) 
of a physical laboratory vacuum chamber against the target 45 micro-Newton topological thrust anomaly.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_torsion_metrology():
    print("[*] Simulating PONDER-01 Torsion Balance Metrology Limits...")
    
    # -------------------------------------------------------------
    # 1. Metrology Noise Profiles
    # -------------------------------------------------------------
    frequencies = np.logspace(-3, 2, 500) # 1 mHz to 100 Hz signal acquisition bounds
    
    # Target Thrust (Constant DC displacement of the pendulum)
    TARGET_THRUST = 45e-6 # 45 micro-Newtons
    
    # Seismic Noise (Micro-seisms peak around 0.1 to 1 Hz due to ocean waves)
    # Scales generally as 1/f^2 but peaks around the microseism band.
    seismic_noise = 1e-4 * (1/frequencies) * np.exp(-(frequencies - 0.2)**2 / 0.05)
    seismic_noise += 1e-6 # Base floor of typical optical bench building
    
    # Thermal Noise (Brownian motion of the torsion fiber)
    # Tends to be a flat white-noise floor dependent on temperature and fiber damping
    thermal_noise = np.ones_like(frequencies) * 1e-7 # 0.1 micro-Newton variance at Room Temp
    
    # Outgassing / Thermal Drift 
    # (Extremely problematic at ultra-low frequencies as the chamber heats up over minutes)
    drift_noise = 2e-5 * (1 / (frequencies * 1e3 + 1))
    
    # Combined Instrument RMS Noise Floor
    total_noise_floor = np.sqrt(seismic_noise**2 + thermal_noise**2 + drift_noise**2)
    
    # -------------------------------------------------------------
    # 2. SNR Visualization Output
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Plot Noise Sources
    plt.plot(frequencies, seismic_noise * 1e6, color='orange', linestyle='--', label='Un-isolated Seismic Jitter')
    plt.plot(frequencies, drift_noise * 1e6, color='brown', linestyle='--', label='Thermal Drift & Outgassing')
    plt.plot(frequencies, thermal_noise * 1e6, color='gray', linestyle=':', label='Brownian Limit (Room Temp)')
    plt.plot(frequencies, total_noise_floor * 1e6, color='black', linewidth=2.5, label='Aggregate System Noise Floor')
    
    # Plot Target Signal
    plt.axhline(y=TARGET_THRUST * 1e6, color='green', linewidth=3.0, label=f'Topological THRUST ({TARGET_THRUST*1e6:.1f} $\\mu$N)')
    
    # Highlight the safe acquisition window
    mask = (TARGET_THRUST > total_noise_floor)
    
    # Finding intersections to shade safe region
    if np.any(mask):
        plt.fill_between(frequencies, total_noise_floor * 1e6, TARGET_THRUST * 1e6, 
                         where=mask, color='green', alpha=0.15, label='Favorable SNR Acquisiton Band')
    
    plt.title("PONDER-01: Vacuum Torsion Balance SNR Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Metrology Bandwidth / Acquisition Frequency (Hz)", fontsize=12)
    plt.ylabel("Measured Force ($\\mu$N)", fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(loc='lower left')
    
    # Set reasonable viewport limits
    plt.ylim(0.01, 1000)
    
    # Output
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ponder_01_torsion_metrology.png')
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Simulation complete. Output saved to: {out_path}")

if __name__ == "__main__":
    simulate_torsion_metrology()
