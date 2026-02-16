import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/06_electrodynamics_weak_interaction/simulations/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_weak_yukawa():
    print("Simulating Cosserat Cutoff (Yukawa vs Coulomb)...")
    
    r = np.linspace(0.01, 5, 500)
    
    # Electromagnetism: Massless mode (above cutoff / zero gap)
    V_coulomb = 1.0 / r
    
    # Weak Force: Massive mode (below Cosserat cutoff)
    l_c = 0.5 # Cosserat Characteristic Cutoff Length
    V_yukawa = (1.0 / r) * np.exp(-r / l_c)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    ax.plot(r, V_coulomb, color='#00ffcc', lw=2, label=r'Electromagnetism (Coulomb: $1/r$)')
    ax.plot(r, V_yukawa, color='#ff3366', lw=3, label=r'Weak Force (Yukawa: $e^{-r/l_c}/r$)')
    ax.axvline(l_c, color='white', linestyle='--', alpha=0.5, label=r'Cosserat Cutoff Length ($l_c$)')
    
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 4)
    ax.set_xlabel(r'Distance from Source ($r$)', fontsize=12, color='white')
    ax.set_ylabel(r'Potential Strength ($V$)', fontsize=12, color='white')
    ax.set_title('Mechanical Origin of the Weak Force Range', fontsize=14, pad=15, color='white', weight='bold')
    
    ax.grid(True, which="both", ls="--", color='#333333', alpha=0.7)
    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white', fontsize=11)
    
    textstr = "Sub-Cutoff Evanescence:\nBecause the Weak interaction lacks the energy\nto overcome the Cosserat rotational mass gap,\nits force decays exponentially."
    ax.text(0.4, 0.5, textstr, transform=ax.transAxes, color='white', fontsize=11, 
            bbox=dict(facecolor='black', edgecolor='#ff3366', alpha=0.8, pad=8))
    
    filepath = os.path.join(OUTPUT_DIR, "weak_yukawa_cutoff.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    simulate_weak_yukawa()