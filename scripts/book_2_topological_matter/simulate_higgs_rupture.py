# simulate_higgs_rupture.py
# Simulates the theoretical limits of the LC vacuum by demonstrating
# the 'Higgs Mechanism' as the literal dielectric rupture of the discrete
# spatial lattice, irreversibly converting localized geometric compliance (C)
# into a massive stationary defect (L) (W/Z Boson generation).

import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('dark_background')
# --- Standard AVE output directory ---
def _find_repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "pyproject.toml")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(_find_repo_root(), "assets", "sim_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- End standard output directory ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_higgs_rupture():
    print("Evaluating Unitary Dielectric Rupture (The Higgs Mechanism)...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='#050510')
    
    t = np.linspace(0, 5, 2000)
    
    # -------------------------------------------------------------------
    # Axiom: The vacuum is a discrete LC lattice with a strict dielectric
    # saturation limit (Axiom 4). A massive localized kinetic spike (Energy > E_unitary)
    # physically snaps the local dielectric compliance (C), structurally transforming the broken 
    # node into a permanently locked inductive mass defect (W/Z Boson generation).
    # -------------------------------------------------------------------
    
    # Input Transient Energy Spike (Simulating a high energy particle collision)
    E_input = 2.0 * np.exp(-((t - 1.5) / 0.15)**2)
    
    # Grid Compliance (C) - Starts dynamically at baseline epsilon_0 = 1.0
    C_val = np.ones_like(t)
    
    # Inductive Defect Mass (L) - Starts at near-zero baseline (vacuum node)
    L_val = np.zeros_like(t) + 0.01 
    
    # Absolute Unitary structural threshold (Axiom 1 Hard-Sphere displacement limit)
    E_crit = 1.0
    
    # Simulate the structural rupture dynamics
    ruptured = False
    for i in range(len(t)):
        if E_input[i] > E_crit and not ruptured:
            ruptured = True
            
        if ruptured:
            # The compliance C instantly drops to near-zero (The node is permanently crushed/locked)
            # This is the 'Symmetry Breaking' phase transition
            C_val[i] = 0.05
            # The trapped collision energy condenses totally into permanent macroscopic Rest Mass (L)
            L_val[i] = 1.0
            
    # Top Plot - The Kinetic Collision Boundary
    ax1 = axes[0]
    ax1.set_facecolor('#050510')
    ax1.plot(t, E_input, color='#00ffff', linewidth=3, label="Transient Collision Energy ($E_{kin}$)")
    ax1.axhline(E_crit, color='#ff00aa', linestyle='--', linewidth=2, label=r"Unitary Rupture Threshold ($\epsilon_{sat}$)")
    
    # Highlight the topological violation singularity
    ax1.fill_between(t, E_crit, E_input, where=(E_input > E_crit), color='#ff00aa', alpha=0.4, label="Coordinate Snapping Event (Symmetry Break)")
    
    ax1.set_title("Extreme Kinetic Collision Event ($E > E_{crit}$)", color='white', fontsize=16, pad=15, weight='bold')
    ax1.set_ylabel("Applied Tension", color='#aaaaaa', fontsize=12)
    ax1.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
    ax1.grid(color='#222233', linestyle=':', linewidth=1)
    
    # Bottom Plot - The Topological Mass Defect Generation (W/Z Boson)
    ax2 = axes[1]
    ax2.set_facecolor('#050510')
    ax2.plot(t, C_val, color='#00ff00', linewidth=3, label=r"Lattice Compliance ($C / \epsilon_0$)")
    ax2.plot(t, L_val, color='#ffcc00', linewidth=3, label=r"Topological Rest Mass ($L / \mu_0$)")
    
    # Format the simulation proof
    ax2.text(1.7, 0.45, r"$\mathbf{The\ Higgs\ Phase\ Transition}$" + "\n" +
                 "Spatial node displacement hits 100% strain limit.\n" +
                 "Structure is permanently locked ($C \to 0$),\n" +
                 "manifesting as a massive Weak Field Boson ($L \to 1$).",
                 color='white', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='#111122', alpha=0.9, edgecolor='#ffcc00'))
    
    ax2.set_title("Topological Mass Generation ($W/Z$ Origin)", color='white', fontsize=16, pad=15, weight='bold')
    ax2.set_xlabel("Time (t)", color='#aaaaaa', fontsize=12)
    ax2.set_ylabel("Internal Lattice Parameters", color='#aaaaaa', fontsize=12)
    ax2.legend(loc='right', facecolor='black', edgecolor='white', labelcolor='white')
    ax2.grid(color='#222233', linestyle=':', linewidth=1)
    
    output_path = os.path.join(OUTPUT_DIR, 'electroweak_dielectric_spark.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, format='pdf', facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Saved Higgs Rupture Transient simulation to: {output_path}")

if __name__ == "__main__":
    generate_higgs_rupture()
