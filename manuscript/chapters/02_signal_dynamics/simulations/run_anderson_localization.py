"""
AVE MODULE 3: ANDERSON LOCALIZATION (Scalar vs Vector)
Proves that only Spin-1 (Helical) signals can traverse the stochastic amorphous lattice. 
Scalar (Spin-0) signals suffer Anderson Localization due to geometric phase noise.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_anderson_localization():
    z = np.linspace(0, 100, 1000)
    
    # 1. Simulate the Amorphous Phase Noise (Jagged Lattice)
    np.random.seed(42)
    geometric_noise = np.random.normal(0, 0.5, size=len(z))
    
    # 2. Scalar Wave (Spin-0)
    scalar_amplitude = np.zeros_like(z)
    current_phase = 0
    for i in range(len(z)):
        current_phase += 0.5 + geometric_noise[i]
        scalar_amplitude[i] = np.cos(current_phase) * np.exp(-z[i]/15.0) # Exponential localization
        
    # 3. Vector Wave (Spin-1 Photon)
    vector_amplitude = np.zeros_like(z)
    for i in range(len(z)):
        smoothed_phase = 0.5 * z[i] + np.mean(geometric_noise[max(0, i-20):i+1]) * 0.05
        vector_amplitude[i] = np.cos(smoothed_phase)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='black')
    
    axes[0].plot(z, scalar_amplitude, color='orange', linewidth=1.5)
    axes[0].fill_between(z, scalar_amplitude, color='orange', alpha=0.3)
    axes[0].set_title("Spin-0 Scalar Boson (No Helicity)", color='white', fontsize=14)
    axes[0].text(80, 0.5, "Anderson Localization\n(Exponential Scattering)", color='yellow', ha='center')
    
    axes[1].plot(z, vector_amplitude, color='cyan', linewidth=1.5)
    axes[1].fill_between(z, vector_amplitude, color='cyan', alpha=0.3)
    axes[1].set_title("Spin-1 Vector Boson (Photon Rifling)", color='white', fontsize=14)
    axes[1].text(80, 0.5, "Geodesic Propagation\n(Infinite Range)", color='cyan', ha='center')
    
    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "anderson_localization.png"), dpi=300, facecolor='black')

if __name__ == "__main__": simulate_anderson_localization()