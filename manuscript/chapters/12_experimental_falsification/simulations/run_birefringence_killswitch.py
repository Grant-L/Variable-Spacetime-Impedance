"""
AVE MODULE 48: THE BIREFRINGENCE KILL SWITCH (QED vs AVE)
---------------------------------------------------------
The ultimate binary falsification test for high-energy colliders.
Standard QED (Euler-Heisenberg) predicts vacuum refractive index 
shifts scale linearly with E^2.
AVE Axiom 4 rigorously mandates that the shift scales with E^4 due to 
the exact hardware non-linear dielectric limit (1 - (\Delta\phi/\alpha)^4).
Plotting the divergence provides a definitive, testable threshold.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/12_experimental_falsification/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_birefringence_killswitch():
    print("Simulating Birefringence Kill Switch (E^4 vs E^2)...")
    
    e_ratio = np.linspace(0, 0.6, 1000)
    
    # 1. Standard QED (Euler-Heisenberg) -> \Delta n \propto E^2
    delta_n_qed = 0.5 * e_ratio**2
    
    # 2. AVE Axiom 4 (Dielectric Saturation) -> \Delta n \approx (1/4) * (E/E_crit)^4
    n_ave = 1.0 / np.power(1.0 - e_ratio**4, 0.25)
    delta_n_ave = n_ave - 1.0
    
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    fig.patch.set_facecolor('#050508'); ax.set_facecolor('#050508')
    
    ax.plot(e_ratio, delta_n_qed, color='#ffcc00', lw=3, label=r'Standard QED ($\Delta n \propto E^2$)')
    ax.plot(e_ratio, delta_n_ave, color='#ff3366', lw=4, label=r'AVE Axiom 4 ($\Delta n \propto E^4$)')
    
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1e-1); ax.set_xlim(0.05, 0.6)
    
    ax.set_xlabel(r'Electric Field Intensity ($E / E_{crit}$)', fontsize=13, color='white', weight='bold')
    ax.set_ylabel(r'Refractive Index Shift ($\Delta n$)', fontsize=13, color='white', weight='bold')
    ax.set_title('The Vacuum Birefringence Kill Switch', fontsize=15, pad=15, color='white', weight='bold')
    
    textstr = (
        r"$\mathbf{The~Binary~Falsification~Condition:}$" + "\n" +
        r"AVE formally rejects the Euler-Heisenberg Lagrangian." + "\n" +
        r"Because discrete topological capacitance diverges asymptotically at $\alpha$, " + "\n" +
        r"the AVE refractive index rigorously scales with the 4th power of the field." + "\n" +
        r"High-intensity laser interferometry testing the $E^2$ vs $E^4$ " + "\n" +
        r"slope will definitively falsify one of the two frameworks."
    )
    ax.text(0.05, 0.65, textstr, transform=ax.transAxes, color='white', fontsize=12, bbox=dict(facecolor='#111111', edgecolor='white', alpha=0.9, pad=10))

    ax.grid(True, which="major", ls=":", color='#444444', alpha=0.8)
    ax.legend(loc='lower right', facecolor='#111111', edgecolor='white', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333333')
    
    filepath = os.path.join(OUTPUT_DIR, "birefringence_killswitch.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_birefringence_killswitch()