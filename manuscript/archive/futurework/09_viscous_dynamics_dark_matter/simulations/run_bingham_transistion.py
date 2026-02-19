"""
AVE MODULE 29: VACUUM BINGHAM PLASTIC RHEOLOGY
----------------------------------------------
Strict simulation of the Non-Newtonian Vacuum Substrate.
Demonstrates the Rheological Phase Transition from a localized Superfluid 
(near stars) to a rigid Viscous Solid (deep space), entirely controlled by 
the local gravitational strain gradient (g) relative to a_{genesis}.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/09_viscous_dynamics_dark_matter/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_bingham_rheology():
    print("Simulating Bingham Plastic Vacuum Rheology...")
    
    # Local Gravitational Acceleration g = |\nabla\Phi|
    g = np.logspace(-14, -6, 1000)
    
    # Theoretical Unruh Drift Limit
    a_gen = 1.071e-10
    
    # Effective Viscosity / Stiffness Modifier (\eta_{eff} \propto 1 - \mu_g)
    # \mu_g = g / (g + a_gen) -> Yields \eta_{eff} = a_gen / (g + a_gen)
    eta_eff = a_gen / (g + a_gen)
    
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor('#050508'); ax.set_facecolor('#050508')
    
    ax.plot(g, eta_eff, color='#ffcc00', lw=4, label=r'Effective Vacuum Viscosity $\eta(g)$')
    
    ax.axvline(a_gen, color='white', linestyle='--', lw=2, label=r'Generative Drift ($a_{genesis}$)')
    
    # Superfluid Zone
    ax.fill_betweenx([0, 1.1], a_gen, 1e-6, color='#00ffcc', alpha=0.1)
    ax.text(1e-8, 0.5, "Superfluid Regime (High Strain)\nViscosity $\\approx 0$ (Solar System)", color='#00ffcc', ha='center', weight='bold')
    
    # Viscous Solid Zone
    ax.fill_betweenx([0, 1.1], 1e-14, a_gen, color='#ff3366', alpha=0.1)
    ax.text(1e-12, 0.5, "Viscous Solid Regime (Low Strain)\nStructural Drag (Dark Matter)", color='#ff3366', ha='center', weight='bold')
    
    ax.set_xscale('log')
    ax.set_ylim(0, 1.1); ax.set_xlim(1e-14, 1e-6)
    ax.set_xlabel(r'Local Gravitational Acceleration $g = |\nabla\Phi|$ (m/s$^2$)', fontsize=13, color='white', weight='bold')
    ax.set_ylabel(r'Normalized Structural Viscosity ($\eta_{eff}$)', fontsize=13, color='white', weight='bold')
    ax.set_title('The Bingham Transition: Resolving the Viscosity Paradox', fontsize=15, pad=15, color='white', weight='bold')
    
    ax.grid(True, which="major", ls=":", color='#444444', alpha=0.8)
    ax.legend(loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333333')
    
    filepath = os.path.join(OUTPUT_DIR, "bingham_transition.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_bingham_rheology()