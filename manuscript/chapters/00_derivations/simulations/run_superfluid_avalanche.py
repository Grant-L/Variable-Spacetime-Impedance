import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Append project root to path for src.ave imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from src.ave.core import constants as ave_const

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets/sim_outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_superfluid_avalanche():
    """
    Plots the rheological Bingham-Plastic phase transition of the Cosserat vacuum.
    """
    # 43.65 kV is the exact derived yield limit from Chapter 4
    yield_kv = 43.65 
    
    # Gravitational shear stress range (in kV equivalent for intuitive mapping)
    tau = np.linspace(0, 100, 1000)
    
    # Viscosity behavior (Heuristic Bingham model for visualization)
    # High viscosity below yield, dropping rapidly to ~0 above yield
    viscosity = np.zeros_like(tau)
    viscosity[tau <= yield_kv] = 1.0  # Normalized base viscosity
    
    # Avalanche dropdown
    avalanche_region = tau > yield_kv
    viscosity[avalanche_region] = np.exp(-0.2 * (tau[avalanche_region] - yield_kv))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Aesthetic dark-mode styling
    fig.patch.set_facecolor('#0f0f13')
    ax.set_facecolor('#0f0f13')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333333')
    ax.grid(True, ls=':', color='#333333')
    
    # Plot tracking curve
    ax.plot(tau, viscosity, color='#00ffcc', lw=4, label='Effective Vacuum Kinematic Viscosity ($\eta_{eff}$)')
    
    # Bounds and Annotations
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlim(0, 100)
    
    # Yield Line
    ax.axvline(yield_kv, color='#ff3366', linestyle='--', lw=2, label=f'Absolute Structural Yield Limit ({yield_kv} kV)')

    # Regions
    ax.fill_between(tau, 0, 1.2, where=(tau <= yield_kv), color='gray', alpha=0.1)
    ax.fill_between(tau, 0, 1.2, where=(tau > yield_kv), color='#00ffcc', alpha=0.05)
    
    # Annotations
    ax.text(20, 0.4, "Rigid Cosserat Solid\n(Dark Matter / Phantom Drag)", color='gray', ha='center', weight='bold')
    ax.text(75, 0.4, "Dielectric Avalanche\nSuperfluid Slipstream\n(Frictionless Orbits)", color='#00ffcc', ha='center', weight='bold')
    
    # Intersection dot
    ax.plot(yield_kv, 1.0, marker='o', color='white', markersize=8, zorder=5)
    
    # Title and Labels
    ax.set_title('Resolving the Friction Paradox: Bingham-Plastic Superfluid Collapse', color='white', weight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Local Metric Shear Stress / Inductive Strain Equiv. ($kV$)', color='white', fontsize=12)
    ax.set_ylabel('Normalized Effective Viscosity / Drag ($\eta_{eff}/\eta_0$)', color='white', fontsize=12)
    
    ax.legend(loc='upper right', facecolor='#111111', edgecolor='#333333', labelcolor='white')
    plt.tight_layout()
    
    outpath = os.path.join(OUTPUT_DIR, "superfluid_avalanche.png")
    plt.savefig(outpath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    plot_superfluid_avalanche()
