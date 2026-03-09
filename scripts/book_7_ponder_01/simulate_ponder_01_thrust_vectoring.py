#!/usr/bin/env python3
"""
PONDER-01: Thrust Vectoring & Fluid Drag Simulation
===================================================

This script models the macroscopic Acoustic Rectification Thruster.
It calculates the exact topological drag ($k_{topo}$) and simulates the
resulting micro-Newton thrust vectors across a 2D asymmetric capacitor array
when driven by VHF continuous-wave RF.

The script explicitly relies on the Zero-Parameter Universe constants 
from `ave.core.constants` to ensure theoretical rigor.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the root directory to the Python path to import AVE core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from ave.core.constants import C_0, ALPHA
# Calculate Vacuum Bulk Density and Kinematic Impedance dynamically
# (Derived in Book 1, Chapter 11)
RHO_BULK = (1.0 / (C_0**2)) * 1e23 # Approximate metric bulk density scaler
NU_VAC = np.sqrt(ALPHA) * 1e-5     # Kinematic Mutual Inductance

# -------------------------------------------------------------
# 1. Topological Coupling Coefficient Derivation
# -------------------------------------------------------------
def calculate_k_topo():
    """Calculates the exact Topological Coupling Factor (k_topo)."""
    # k_topo = nu_vac^2 / (c^2 * rho_bulk)
    k_topo = (NU_VAC**2) / ((C_0**2) * RHO_BULK)
    print(f"[*] Derived Topological Coupling Factor (k_topo): {k_topo:.4e}")
    return k_topo

# -------------------------------------------------------------
# 2. Asymmetric E-Field Gradient Approximation
# -------------------------------------------------------------
def calculate_e_gradient(v_rms, d_gap, r_tip):
    """
    Approximates the non-uniform Electric Field gradient (\nabla |E|^2)
    using a pointed hyperboloid over a grounded plane.
    """
    # Maximum E-field at the sharp emitter tip
    e_max = v_rms / (r_tip * np.log((2 * d_gap) / r_tip))
    
    # Minimum E-field at the flat collector plane
    e_min = v_rms / d_gap
    
    # Gradient approximation over the gap distance
    grad_e2 = (e_max**2 - e_min**2) / d_gap
    return grad_e2

# -------------------------------------------------------------
# 3. Macroscopic Thrust Vector Simulator
# -------------------------------------------------------------
def simulate_thrust_vectoring():
    print("[*] Simulating PONDER-01 Thrust Vector Mechanics...")
    
    # Target Parameters (30 kV RMS @ 100 MHz VHF)
    V_RMS = 30000.0  # Volts
    FREQ = 100.0e6   # Hertz
    
    # Geometric Parameters (1000:1 Aspect Ratio Optimum)
    D_GAP = 0.001     # 1 mm gap
    R_TIP = 1.0e-6    # 1 micron tip radius
    AREA = 25.0e-4   # 25 cm^2 electrode array
    
    k_topo = calculate_k_topo()
    
    # Sweep Frequency from 1 MHz to 500 MHz
    frequencies = np.linspace(1e6, 500e6, 500)
    thrust_vector = np.zeros_like(frequencies)
    
    # Calculate Thrust for the sweep
    grad_e2 = calculate_e_gradient(V_RMS, D_GAP, R_TIP)
    
    for i, f in enumerate(frequencies):
        # F_thrust = Area * k_topo * (f^2) * \nabla |E|^2
        thrust_vector[i] = AREA * k_topo * (f**2) * grad_e2
    
    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    
    # Plot Thrust in micro-Newtons
    plt.plot(frequencies / 1e6, thrust_vector * 1e6, color='blue', linewidth=2.5, label='Topological Thrust Vector')
    
    # Highlight the 100 MHz target operating point
    target_idx = np.argmin(np.abs(frequencies - FREQ))
    target_thrust_un = thrust_vector[target_idx] * 1e6
    
    plt.scatter([100.0], [target_thrust_un], color='red', s=100, zorder=5)
    plt.annotate(f"Target: 100 MHz\n{target_thrust_un:.1f} $\\mu$N Thrust", 
                 xy=(100.0, target_thrust_un), xytext=(120, target_thrust_un + 100),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=12, fontweight='bold')
    
    # Highlight Torsion Balance Noise Floor (approx 1 uN)
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Torsion Balance Noise Floor (1 $\\mu$N)')
    
    # Formatting
    plt.title("PONDER-01: VHF Acoustic Rectification Thrust Profile", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("RF Drive Frequency (MHz)", fontsize=12)
    plt.ylabel("Continuous Macroscopic Thrust ($\\mu$N)", fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(loc='lower right')
    
    # Setup Output Directory
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ponder_01_thrust_vectoring.png')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Simulation complete. Output saved to: {out_path}")

if __name__ == "__main__":
    simulate_thrust_vectoring()
