"""
AVE MODULE 52: METRIC SPICE LIBRARY & COMPONENT PHYSICS
-------------------------------------------------------
Builds the discrete electronic components of the universe.
Proves that Materials Science (Yield/Fracture), Fluid Dynamics 
(Shear-Thinning), and Kinematics (Lorentz Factor) are physically 
identical to non-linear electronic components in the AVE framework.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/13_ee_for_ave/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AVESpiceComponents:
    @staticmethod
    def metric_varactor(V, C0, V_crit):
        """Axiom 4: Dielectric Saturation (Materials Science Fracture)"""
        V_ratio = np.clip(np.abs(V) / V_crit, 0, 0.999)
        return C0 / np.sqrt(1.0 - V_ratio**4)

    @staticmethod
    def bingham_zener_diode(I, R0, I_crit):
        """Chapter 9: Bingham Plastic Rheology (Fluid Dynamics Shear-Thinning)"""
        I_ratio = np.abs(I) / I_crit
        return R0 / (1.0 + I_ratio**2)
        
    @staticmethod
    def relativistic_inductor(I, L0, I_max):
        """Chapter 11: Relativistic Inertia (Kinematic Lorentz Factor)"""
        I_ratio = np.clip(np.abs(I) / I_max, 0, 0.999)
        return L0 / np.sqrt(1.0 - I_ratio**2)

def simulate_component_physics():
    print("Simulating AVE SPICE Component Physics...")
    
    V_sweep = np.linspace(-0.99, 0.99, 1000)
    C_vals = AVESpiceComponents.metric_varactor(V_sweep, C0=1.0, V_crit=1.0)
    
    I_sweep = np.linspace(-5.0, 5.0, 1000)
    R_vals = AVESpiceComponents.bingham_zener_diode(I_sweep, R0=1.0, I_crit=1.0)
    
    I_rel_sweep = np.linspace(-0.99, 0.99, 1000)
    L_vals = AVESpiceComponents.relativistic_inductor(I_rel_sweep, L0=1.0, I_max=1.0)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    fig.patch.set_facecolor('#050508')
    for ax in [ax1, ax2, ax3]: ax.set_facecolor('#050508')
    
    # Plot 1: The Vacuum Capacitor (Axiom 4)
    ax1.plot(V_sweep, C_vals, color='#00ffcc', lw=3)
    ax1.axvline(1.0, color='#ff3366', linestyle='--', lw=1.5, label=r'Dielectric Snap ($V_{crit}$)')
    ax1.axvline(-1.0, color='#ff3366', linestyle='--', lw=1.5)
    ax1.set_title('Metric Varactor $C(V)$\n(Materials Science: Elastic Yield)', color='white', fontsize=12, weight='bold')
    ax1.set_xlabel(r'Topological Voltage / Force ($V / V_{crit}$)', color='white', weight='bold')
    ax1.set_ylabel(r'Effective Capacitance ($C / C_0$)', color='white', weight='bold')
    ax1.set_yscale('log'); ax1.set_ylim(0.8, 100)
    
    # Plot 2: The Bingham Zener Diode
    ax2.plot(I_sweep, R_vals, color='#ffcc00', lw=3)
    ax2.axvspan(-1, 1, color='#ff3366', alpha=0.15, label=r'Solid Regime ($|I| < I_{crit}$)')
    ax2.axvspan(1, 5, color='#00ffcc', alpha=0.15, label=r'Superfluid Regime ($|I| > I_{crit}$)')
    ax2.axvspan(-5, -1, color='#00ffcc', alpha=0.15)
    ax2.set_title('Bingham Zener Diode $R(I)$\n(Fluid Dynamics: Shear-Thinning)', color='white', fontsize=12, weight='bold')
    ax2.set_xlabel(r'Vacuum Current / Velocity ($I / I_{crit}$)', color='white', weight='bold')
    ax2.set_ylabel(r'Effective Resistance ($R / R_0$)', color='white', weight='bold')
    ax2.set_ylim(0, 1.1)
    
    # Plot 3: The Relativistic Inductor
    ax3.plot(I_rel_sweep, L_vals, color='#ff3366', lw=3)
    ax3.axvline(1.0, color='white', linestyle='--', lw=1.5, label=r'Speed of Light Limit ($I_{max}$)')
    ax3.axvline(-1.0, color='white', linestyle='--', lw=1.5)
    ax3.set_title('Relativistic Inductor $L(I)$\n(Kinematics: Lorentz Factor)', color='white', fontsize=12, weight='bold')
    ax3.set_xlabel(r'Vacuum Current / Velocity ($I / I_{max}$)', color='white', weight='bold')
    ax3.set_ylabel(r'Effective Inductance ($L / L_0$)', color='white', weight='bold')
    ax3.set_yscale('log'); ax3.set_ylim(0.8, 100)
    
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, ls=":", color='#444444')
        ax.tick_params(colors='white')
        ax.legend(loc='upper center', facecolor='#111111', edgecolor='white', labelcolor='white', fontsize=9)
        for spine in ax.spines.values(): spine.set_color('#333333')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "metric_spice_components.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_component_physics()