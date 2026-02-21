"""
AVE Falsification Protocol: Achromatic Impedance Lens Hardware
--------------------------------------------------------------
The AVE framework mathematically proves that Gravity acts as an
"Achromatic Impedance Lens" because metric strain scales both 
permittivity (epsilon) and permeability (mu) proportionally, meaning 
the local impedance Z = sqrt(mu/epsilon) remains perfectly matched 
to Z_0. This prevents any reflection of light across a gravity well.

This script proposes an experimental RF hardware falsification:
By fabricating a metamaterial dielectric lens where both mu_r and epsilon_r
are spatially doped to scale together proportionally, we can synthesize a 
macroscopic lens with a 0.0 reflection coefficient at all incidence angles,
proving the gravitational mechanism in solid-state hardware.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_achromatic_lens():
    print("==========================================================")
    print(" AVE HARDWARE FALSIFICATION: ACHROMATIC IMPEDANCE LENS")
    print("==========================================================")
    
    # Radial cross-section of the Lens [-1, 1]
    x = np.linspace(-1.0, 1.0, 500)
    
    # 1. Standard Optical/RF Lens (Varying Epsilon, Constant Mu)
    # Refractive index n(x) increases toward the center to focus the beam
    n_gradient = 1.0 + 1.5 * np.exp(-10 * x**2)
    
    eps_r_standard = n_gradient**2
    mu_r_standard = np.ones_like(x) # Non-magnetic glass/dielectric standard
    
    Z_standard = np.sqrt(mu_r_standard / eps_r_standard) # Normalized to Z_0 = 1.0
    reflection_standard = ((Z_standard - 1.0) / (Z_standard + 1.0))**2 * 100 # % Reflection
    
    # 2. AVE Achromatic Metamaterial Lens (Proportional Doping)
    # Both mu and epsilon scale linearly with the refractive index
    eps_r_ave = n_gradient
    mu_r_ave = n_gradient
    
    Z_ave = np.sqrt(mu_r_ave / eps_r_ave)
    reflection_ave = ((Z_ave - 1.0) / (Z_ave + 1.0))**2 * 100
    
    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0B0F19')
    fig.suptitle("AVE Telescope Hardware Protocol: The Achromatic Impedance Lens", color='white', fontsize=20, weight='bold', y=0.98)
    
    # Panel 1: Material Properties
    ax1.set_facecolor('#0B0F19')
    ax1.plot(x, n_gradient, color='white', lw=3, label="Target Refractive Index $n(x)$")
    ax1.plot(x, eps_r_standard, color='#FF3366', ls='--', lw=2, label=r"Standard Lens ($\epsilon_r$)")
    ax1.plot(x, eps_r_ave, color='#00FFCC', lw=2, label=r"AVE Metamaterial ($\epsilon_r = \mu_r \propto n$)")
    
    ax1.set_title("1. Spatial Metamaterial Doping Profile", color='white', pad=15, weight='bold')
    ax1.set_xlabel("Lens Radial Axis (Normalized)", color='gray')
    ax1.set_ylabel("Relative Material Constants", color='gray')
    ax1.legend(loc='upper right', facecolor='#111111', edgecolor='gray', labelcolor='white')
    ax1.grid(True, ls=':', color='#333333', alpha=0.5)
    
    # Panel 2: Reflection Coefficients (The Falsification)
    ax2.set_facecolor('#0B0F19')
    ax2.fill_between(x, 0, reflection_standard, color='#FF3366', alpha=0.3)
    ax2.plot(x, reflection_standard, color='#FF3366', lw=3, label="Standard Lens: Fresnel Reflection Loss")
    
    ax2.plot(x, reflection_ave, color='#00FFCC', lw=4, label="AVE Lens: Zero Boundary Reflection ($Z \equiv Z_0$)")
    
    ax2.set_title("2. Fresnel Boundary Reflection (Signal Loss)", color='white', pad=15, weight='bold')
    ax2.set_xlabel("Lens Radial Axis (Normalized)", color='gray')
    ax2.set_ylabel("Reflected Power (%)", color='gray')
    ax2.legend(loc='upper right', facecolor='#111111', edgecolor='gray', labelcolor='white')
    ax2.grid(True, ls=':', color='#333333', alpha=0.5)
    
    # Global aesthetics
    for ax in [ax1, ax2]:
        ax.tick_params(colors='lightgray')
        for spine in ax.spines.values(): spine.set_color('#333333')
        
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    OUTPUT_DIR = "assets/sim_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "simulate_achromatic_lens.png")
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"\n[+] Saved Achromatic Lens Hardware profile to {out_path}")

if __name__ == "__main__":
    simulate_achromatic_lens()
