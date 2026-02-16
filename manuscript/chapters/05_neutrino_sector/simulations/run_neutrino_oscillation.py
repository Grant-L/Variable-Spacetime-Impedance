import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/05_neutrino_sector/simulations/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_neutrino_oscillation():
    print("Simulating Neutrino Oscillation via Lattice Dispersion...")
    
    distance = np.linspace(0, 100, 2000)
    
    # Wavenumbers for the three torsional harmonics
    k1, k2, k3 = 1.0, 1.05, 1.12
    
    # The physical state is a superposition of the three harmonics
    state_1 = np.cos(k1 * distance)
    state_2 = np.cos(k2 * distance)
    state_3 = np.cos(k3 * distance)
    
    # Flavor probabilities are the squared projections of the beating wave packet
    prob_e = (1/3) * (state_1 + state_2 + state_3)**2
    prob_mu = (1/3) * (state_1 - 0.5*state_2 - 0.5*state_3 + 0.866*state_2 - 0.866*state_3)**2
    prob_tau = 1.0 - (prob_e + prob_mu) # Unitarity conservation
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    ax.plot(distance, prob_e, color='#00ffcc', lw=2, label=r'$\nu_e$ (Electron Flavor)')
    ax.plot(distance, prob_mu, color='#ff3366', lw=2, label=r'$\nu_\mu$ (Muon Flavor)')
    ax.plot(distance, prob_tau, color='#ffff00', lw=2, label=r'$\nu_\tau$ (Tau Flavor)')
    
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 100)
    
    ax.set_xlabel('Propagation Distance (Lattice Nodes)', fontsize=12, color='white')
    ax.set_ylabel(r'Flavor Detection Probability ($|\Psi|^2$)', fontsize=12, color='white')
    ax.set_title('Neutrino Oscillation as Lattice Dispersive Beat Frequencies', fontsize=14, pad=15, color='white', weight='bold')
    
    ax.grid(True, which="both", ls="--", color='#333333', alpha=0.7)
    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
    
    filepath = os.path.join(OUTPUT_DIR, "neutrino_oscillation_beat.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    simulate_neutrino_oscillation()