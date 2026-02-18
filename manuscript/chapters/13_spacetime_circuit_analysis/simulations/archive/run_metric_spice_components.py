"""
AVE MODULE 52: EQUIVALENT CIRCUIT COMPONENT LIBRARY
---------------------------------------------------
Models the discrete non-linear properties of the vacuum substrate 
using standard lumped-element circuit equivalents.
Connects materials science (elastic yield), fluid dynamics 
(shear-thinning), and kinematics (relativistic inertia) to their 
corresponding electrical analogues.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/13_spacetime_circuit_analysis/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class VacuumCircuitEquivalents:
    @staticmethod
    def metric_varactor(V, C0, V_crit):
        """Models Axiom 4: Dielectric Saturation as a Voltage-Dependent Capacitor"""
        V_ratio = np.clip(np.abs(V) / V_crit, 0, 0.999)
        return C0 / np.sqrt(1.0 - V_ratio**4)

    @staticmethod
    def bingham_resistor(I, R0, I_crit):
        """Models Chapter 9: Bingham Plastic Rheology as a Current-Dependent Resistor"""
        I_ratio = np.abs(I) / I_crit
        return R0 / (1.0 + I_ratio**2)
        
    @staticmethod
    def relativistic_inductor(I, L0, I_max):
        """Models Relativistic Inertia via the Lorentz Factor"""
        I_ratio = np.clip(np.abs(I) / I_max, 0, 0.999)
        return L0 / np.sqrt(1.0 - I_ratio**2)

def simulate_component_physics():
    print("Generating AVE SPICE Component Profiles...")
    
    V_sweep = np.linspace(-0.99, 0.99, 1000)
    C_vals = VacuumCircuitEquivalents.metric_varactor(V_sweep, C0=1.0, V_crit=1.0)
    
    I_sweep = np.linspace(-5.0, 5.0, 1000)
    R_vals = VacuumCircuitEquivalents.bingham_resistor(I_sweep, R0=1.0, I_crit=1.0)
    
    I_rel_sweep = np.linspace(-0.99, 0.99, 1000)
    L_vals = VacuumCircuitEquivalents.relativistic_inductor(I_rel_sweep, L0=1.0, I_max=1.0)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    fig.patch.set_facecolor('#0a0a12')
    for ax in [ax1, ax2, ax3]: ax.set_facecolor('#0a0a12')
    
    # Plot 1: Vacuum Capacitor
    ax1.plot(V_sweep, C_vals, color='#4FC3F7', lw=2.5)
    ax1.axvline(1.0, color='#E57373', linestyle='--', lw=1.5, label=r'Dielectric Limit ($V_{crit}$)')
    ax1.axvline(-1.0, color='#E57373', linestyle='--', lw=1.5)
    ax1.set_title('Metric Varactor $C(V)$\n(Elastic Saturation)', color='white', fontsize=12)
    ax1.set_xlabel(r'Topological Voltage / Force ($V / V_{crit}$)', color='white')
    ax1.set_ylabel(r'Effective Capacitance ($C / C_0$)', color='white')
    ax1.set_yscale('log'); ax1.set_ylim(0.8, 100)
    
    # Plot 2: Bingham Zener
    ax2.plot(I_sweep, R_vals, color='#FFD54F', lw=2.5)
    ax2.axvspan(-1, 1, color='#E57373', alpha=0.15, label=r'Solid Regime ($|I| < I_{crit}$)')
    ax2.axvspan(1, 5, color='#4FC3F7', alpha=0.15, label=r'Fluid Regime ($|I| > I_{crit}$)')
    ax2.axvspan(-5, -1, color='#4FC3F7', alpha=0.15)
    ax2.set_title('Bingham Resistor $R(I)$\n(Shear-Thinning Viscosity)', color='white', fontsize=12)
    ax2.set_xlabel(r'Vacuum Current / Velocity ($I / I_{crit}$)', color='white')
    ax2.set_ylabel(r'Effective Resistance ($R / R_0$)', color='white')
    ax2.set_ylim(0, 1.1)
    
    # Plot 3: Relativistic Inductor
    ax3.plot(I_rel_sweep, L_vals, color='#E57373', lw=2.5)
    ax3.axvline(1.0, color='white', linestyle='--', lw=1.5, label=r'Kinematic Limit ($I_{max} \equiv c$)')
    ax3.axvline(-1.0, color='white', linestyle='--', lw=1.5)
    ax3.set_title('Relativistic Inductor $L(I)$\n(Lorentz Inertial Factor)', color='white', fontsize=12)
    ax3.set_xlabel(r'Vacuum Current / Velocity ($I / I_{max}$)', color='white')
    ax3.set_ylabel(r'Effective Inductance ($L / L_0$)', color='white')
    ax3.set_yscale('log'); ax3.set_ylim(0.8, 100)
    
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, ls=":", color='#333333')
        ax.tick_params(colors='lightgray')
        ax.legend(loc='upper center', facecolor='#111111', edgecolor='gray', labelcolor='white', fontsize=9)
        for spine in ax.spines.values(): spine.set_color('#333333')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "metric_spice_components.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_component_physics()