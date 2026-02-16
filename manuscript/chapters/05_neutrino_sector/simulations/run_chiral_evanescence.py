import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/05_neutrino_sector/simulations/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_chiral_evanescence():
    print("Simulating Cosserat Chiral Bandgap (Parity Violation)...")
    
    x = np.linspace(0, 10, 1000)
    
    # Left-Handed: Real frequency (Propagating Sine Wave)
    k_LH = 2.0
    LH_wave = np.sin(k_LH * x)
    
    # Right-Handed: Imaginary frequency (Evanescent Exponential Decay)
    gamma_c = 1.5 # Cosserat chiral damping factor
    RH_wave = np.sin(k_LH * x) * np.exp(-gamma_c * x)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    ax.plot(x, LH_wave, color='#00ffcc', lw=2, label=r'Left-Handed Twist ($\omega^2 > 0$): Propagating Mode')
    ax.plot(x, RH_wave, color='#ff3366', lw=2, label=r'Right-Handed Twist ($\omega^2 < 0$): Evanescent Mode')
    
    ax.axvspan(0, 1, color='white', alpha=0.05, label='Fundamental Lattice Pitch ($l_{node}$)')
    
    ax.set_ylim(-1.5, 2.0)
    ax.set_xlim(0, 8)
    
    ax.set_xlabel(r'Propagation Distance ($x/l_{node}$)', fontsize=12, color='white')
    ax.set_ylabel(r'Microrotational Amplitude ($\theta$)', fontsize=12, color='white')
    ax.set_title('Parity Violation via Cosserat Chiral Bandgap', fontsize=14, pad=15, color='white', weight='bold')
    
    ax.grid(True, which="both", ls="--", color='#333333', alpha=0.7)
    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
    
    filepath = os.path.join(OUTPUT_DIR, "chiral_evanescence.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    simulate_chiral_evanescence()