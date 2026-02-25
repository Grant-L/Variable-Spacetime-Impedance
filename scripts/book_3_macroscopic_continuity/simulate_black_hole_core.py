"""
AVE Black Hole LC Network Core Simulation.
Visualizes how Inductive Saturation and Topological geometric limits
naturally prevent an r=0 singularity, forming a stable, holographic core limit.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure local ave package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ave.core.constants import MU_0, EPSILON_0, Z_0

def simulate_black_hole_core():
    """
    Simulates the localized metric strain h_perp converging on a massive body.
    In General Relativity, 1/r diverges to infinity at r=0.
    In the AVE discrete topology, the 6^3_2 Borromean limit (Volume = 2.0 max)
    and the Dielectric Saturation bound hard-cap the permissible metric strain.
    """
    print("==========================================================")
    print("   AVE BLACK HOLE CORE (DIELECTRIC RUPTURE LIMIT)")
    print("==========================================================")
    
    # 1D Radial array (from deep space into the core at r=0)
    # Using arbitrary structural units (rs = 1.0)
    r = np.linspace(0.001, 5.0, 1000)
    
    # Standard Newtonian/GR 1/r potential strain
    # (Unbounded classical gravity)
    strain_classical = 1.0 / r
    
    # The AVE Topological Tensor geometric bound.
    # From `tensors.py`, the intersecting orthogonal flux strings have a 
    # strictly hard geometric saturation limit of 2.0 volume metric. 
    # Effectively, the metric strain acts as an inverted bounded logistic curve.
    
    # Maximum physical metric strain (approx ~2.0 per the Tensor limit)
    h_max = 2.0
    
    # Non-linear saturated strain n(r) = 1 + h_perp
    # Modulated by topological inductive delay
    strain_ave = h_max * (1.0 - np.exp(-1.0 / r))
    
    # Effective Local Impedance Z(r)
    # The Event Horizon acts as the impedance boundary where the wave impedance
    # reflection coefficient reaches Total Internal Reflection (Gamma = -1)
    
    # In pure Achromatic mode, Z_0 theoretically remains flat for light, but
    # for massive matter (which couples to the phase strain), the impedance diverges.
    # We plot the core mass-density divergence here.
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ----- Plot 1: Unbounded GR vs Bounded LC Network -----
    ax1.plot(r, strain_classical, color='red', linestyle='--', label="Classical Singularity (1/r)")
    ax1.plot(r, strain_ave, color='cyan', linewidth=3, label="Topological Saturation Tensor")
    
    # Highlight the Event Horizon Zone
    ax1.axvspan(0.0, 1.0, color='purple', alpha=0.3, label="Interior Holographic Core")
    ax1.axvline(1.0, color='white', linestyle=':', label="Event Horizon (Dielectric Rupture)")
    
    ax1.set_ylim(0, 10)
    ax1.set_xlim(0, 5)
    ax1.set_title("Metric Strain: Avoiding the r=0 Singularity", fontsize=16)
    ax1.set_xlabel("Radius ($r / r_s$)")
    ax1.set_ylabel(r"Localized Metric Strain $h_\perp$")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # ----- Plot 2: Macroscopic Capacitance Collapse -----
    # As the tensor saturates, the structural compliance (Capacitance) required 
    # to support further geometric twisting drops to zero.
    
    # Capacitance is proportional to 1 / (1 + strain) 
    capacitance_classical = EPSILON_0 / (1.0 + strain_classical)
    capacitance_ave = EPSILON_0 / (1.0 + strain_ave)
    
    # Normalize for plotting
    C_norm_class = capacitance_classical / EPSILON_0
    C_norm_ave = capacitance_ave / EPSILON_0
    
    ax2.plot(r, C_norm_class, color='red', linestyle='--', label="Infinite Compression")
    ax2.plot(r, C_norm_ave, color='green', linewidth=3, label="Lattice Geometric Lock")
    
    ax2.axvspan(0.0, 1.0, color='purple', alpha=0.3)
    ax2.axvline(1.0, color='white', linestyle=':')
    
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 1.2)
    ax2.set_title("Lattice Compliance (Capacitance) Collapse", fontsize=16)
    ax2.set_xlabel("Radius ($r / r_s$)")
    ax2.set_ylabel("Normalized Structural Capacitance ($C / C_0$)")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), '../assets/sim_outputs/simulate_black_hole_core.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    print(f"\nSaved Black Hole Core bounds plot to {output_path}")

if __name__ == "__main__":
    simulate_black_hole_core()
