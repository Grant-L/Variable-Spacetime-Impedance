"""
AVE MODULE 2: TRUE HARDWARE ACTION SCALE
----------------------------------------
Plots the mathematically derived spatial scales of the AVE framework, strictly 
enforcing Axiom 1 (l_node == electron scale) and visualizing the Fine Structure 
Constant (\alpha) as the exact spatial porosity ratio (r_core / l_node).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_action_scale():
    print("Simulating Rigorous Hardware Action Scale...")
    
    # Strictly Derived AVE Theoretical Scales (Meters)
    planck_length = 1.616e-35
    l_node = 3.86159e-13  # Reduced Compton Wavelength (Axiom 1 & Section 2.2)
    alpha = 1 / 137.035999
    r_core = alpha * l_node  # Structural Core Saturation Limit (Section 2.2)
    
    scales = [planck_length, r_core, l_node]
    labels = [
        'Planck "Illusion"\n(Macroscopic $G$ Projection)', 
        'Structural Core Radius ($r_{core}$)\n(Dielectric Saturation Limit)', 
        'AVE Lattice Pitch ($l_{node}$)\n(Fundamental Electron Limit)'
    ]
    
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    y_pos = [1, 2.5, 4]
    colors = ['#ff3366', '#ffff00', '#00ffcc']
    
    for i in range(3):
        ax.plot(scales[i], y_pos[i], marker='o', markersize=15, color=colors[i], markeredgecolor='white', markeredgewidth=1.5)
        ax.axhline(y_pos[i], color=colors[i], alpha=0.3, linestyle='--')
        ax.text(scales[i] * 2.5, y_pos[i] - 0.15, labels[i], color=colors[i], fontsize=11, weight='bold')

    # Draw the structural porosity gap (Alpha)
    ax.annotate('', xy=(r_core, 3.25), xytext=(l_node, 3.25),
                arrowprops=dict(arrowstyle='<->', color='white', lw=1.5))
    ax.text(np.sqrt(r_core * l_node), 3.4, f"Porosity Ratio ($\\alpha \\approx 1/137$)", 
            color='white', ha='center', fontsize=11, weight='bold')

    ax.set_xscale('log')
    ax.set_xlim(1e-37, 1e-10)
    ax.set_ylim(0.5, 5.0)
    ax.set_yticks([])
    ax.set_xlabel('Absolute Spatial Scale (Meters)', color='white', fontsize=12, weight='bold')
    ax.set_title('The True Geometric Granularity of the Vacuum Substrate', color='white', fontsize=14, weight='bold', pad=15)
    
    ax.grid(True, which="major", ls="--", color='#333333', alpha=0.7)
    ax.tick_params(axis='x', colors='white')
    
    filepath = os.path.join(OUTPUT_DIR, "hardware_action_scale.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved theoretical scale validation to: {filepath}")

if __name__ == "__main__":
    simulate_action_scale()