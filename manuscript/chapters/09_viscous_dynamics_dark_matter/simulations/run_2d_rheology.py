import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/09_viscous_dynamics_dark_matter/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def plot_vacuum_rheology():
    print("Simulating Vacuum Rheology (Bingham Plastic Model)...")
    
    # 1. Define Shear Rates (Gravitational Gradients)
    # Gamma_dot ~ sqrt(GM/r^3) (Tidal/Shear Strain Rate)
    # Range: from Intergalactic (1e-20) to Surface of Sun (1e-5)
    gamma_dot = np.logspace(-20, -5, 100)
    
    # 2. Constants
    # Critical Shear Threshold (Transition Point)
    # We set this empirically between Solar System and Galactic scales
    gamma_c = 1e-13  # s^-1
    
    # Base Viscosity (Dark Matter Limit)
    eta_0 = 1.0 # Normalized to 1 for visualization
    
    # 3. The Constitutive Equation (Eq 9.1 update)
    # Shear-Thinning Model: viscosity drops as shear squared
    eta = eta_0 / (1 + (gamma_dot / gamma_c)**2)
    
    # 4. Benchmark Points
    # Solar System (Earth Orbit): r ~ 1.5e11 m, M ~ 2e30 kg
    # g_grad ~ sqrt(GM/r^3) ~ sqrt(1.3e20 / 3.3e33) ~ 2e-7
    solar_shear = 2.0e-7 
    solar_visc = eta_0 / (1 + (solar_shear / gamma_c)**2)
    
    # Galactic Rim (Milky Way): r ~ 50 kpc, M ~ 1e11 M_sol
    # g_grad ~ 1e-16
    galactic_shear = 1.0e-16
    galactic_visc = eta_0 / (1 + (galactic_shear / gamma_c)**2)
    
    # 5. Plotting
    plt.figure(figsize=(12, 7))
    plt.loglog(gamma_dot, eta, color='#D95319', linewidth=3, label=r'Vacuum Viscosity $\eta(\dot{\gamma})$')
    
    # Annotate Solar System (Safety Zone)
    plt.scatter([solar_shear], [solar_visc], color='green', s=150, zorder=5, label='Solar System (Earth)')
    plt.text(solar_shear, solar_visc*5, "Superfluid Regime\n(No Drag)", 
             color='green', ha='center', fontweight='bold')
             
    # Annotate Galaxy (Dark Matter Zone)
    plt.scatter([galactic_shear], [galactic_visc], color='blue', s=150, zorder=5, label='Galaxy Outskirts')
    plt.text(galactic_shear, galactic_visc*0.5, "Viscous Regime\n(Dark Matter)", 
             color='blue', ha='center', fontweight='bold')
    
    # Aesthetics
    plt.axvline(gamma_c, color='gray', linestyle='--', alpha=0.5, label=r'Critical Shear $\dot{\gamma}_c$')
    plt.title('The Rheological Shield: Non-Newtonian Vacuum Dynamics', fontsize=14)
    plt.xlabel(r'Gravitational Shear Rate $\dot{\gamma}$ ($s^{-1}$)', fontsize=12)
    plt.ylabel(r'Effective Viscosity $\eta_{eff}$ (Normalized)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='lower left')
    
    # Fill safe zone
    plt.axvspan(gamma_c, 1e-5, color='green', alpha=0.05)
    plt.axvspan(1e-20, gamma_c, color='blue', alpha=0.05)

    output_path = os.path.join(OUTPUT_DIR, "rheology_2d_curve.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_vacuum_rheology()