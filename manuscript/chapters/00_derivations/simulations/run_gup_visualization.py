import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Append project root to path for src.ave imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from src.ave.core import constants as ave_const

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets/sim_outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_generalized_uncertainty():
    """
    Plots the AVE Generalized Uncertainty Principle (GUP) derived from 
    finite-difference sampling on the discrete vacuum hardware vs the standard HUP.
    """
    hbar = ave_const.H_BAR
    l_node = ave_const.L_NODE
    
    # Momentum ranging from 0 near to the Brillouin Zone boundary (pi*hbar/l_node)
    p_max = np.pi * hbar / l_node
    p = np.linspace(0.01 * p_max, p_max, 1000)
    
    # Standard Heisenberg Uncertainty (Linear continuum)
    delta_x_hup = hbar / (2 * p)  # Solving delta_x * p = hbar/2
    
    # AVE Discrete GUP: Delta x * P >= hbar/2 * |cos(l_node * P / hbar)|
    # We plot the minimum resolvable delta_x bound
    delta_x_gup = (hbar / (2 * p)) * np.abs(np.cos((l_node * p) / hbar))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Aesthetic dark-mode styling
    fig.patch.set_facecolor('#0f0f13')
    ax.set_facecolor('#0f0f13')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333333')
    ax.grid(True, ls=':', color='#333333')
    
    # Plotting
    ax.plot(p / p_max, delta_x_hup / l_node, color='red', lw=2, linestyle='--', label='Standard HUP (Continuum Limit)')
    ax.plot(p / p_max, delta_x_gup / l_node, color='#00ffcc', lw=3, label='AVE GUP (Discrete Lattice Limit)')
    
    # Bounds and Annotations
    ax.set_ylim(0, 10)
    ax.set_xlim(0, 1.05)
    
    ax.axvline(1.0, color='gray', linestyle=':', lw=2)
    ax.text(1.02, 5, 'Geometric Brillouin\\nBoundary Limit', color='gray', rotation=90, verticalalignment='center')
    
    # Title and Labels
    ax.set_title('Resolving the UV Catastrophe: The Authentic GUP', color='white', weight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Relative Particle Momentum ($p_c / p_{max}$)', color='white', fontsize=12)
    ax.set_ylabel('Minimum Positional Variance ($\Delta x / \ell_{node}$)', color='white', fontsize=12)
    
    # Overlay explanation
    explanation = (
        "In standard QFT, positional uncertainty approaches zero\nas momentum approaches infinity, leading to UV singularities.\n\n"
        "In AVE, the discrete physical lattice pitch ($\ell_{node}$)\n"
        "strictly limits minimum resolvable spatial variance.\nAs momentum approaches the Brillouin boundary,\nthe finite-difference expectation value naturally plunges."
    )
    ax.text(0.05, 1.5, explanation, color='white', fontsize=9, bbox=dict(facecolor='#111111', edgecolor='#00ffcc', alpha=0.8))
    
    ax.legend(loc='upper right', facecolor='#111111', edgecolor='#333333', labelcolor='white')
    plt.tight_layout()
    
    outpath = os.path.join(OUTPUT_DIR, "ave_gup_resolution.png")
    plt.savefig(outpath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    plot_generalized_uncertainty()
