"""
AVE MODULE 11: THE TOPOLOGICAL LEPTON GENERATIONS
-------------------------------------------------
Generates the exact topological soliton boundaries for the three stable leptons 
(Electron p=3, Muon p=7, Tau p=11) constrained perfectly to the Golden Torus limit.
Computes and maps the localized lattice strain to demonstrate the Flux Crowding 
mechanism driving the exponential mass hierarchy.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = "manuscript/chapters/03_fermion_sector/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_lepton_generations():
    print("Simulating Topological Lepton Generations (Golden Torus Limits)...")
    
    # Exact Hardware Saturation Limits (Golden Ratio)
    Phi = (1 + np.sqrt(5)) / 2
    R = Phi / 2        # Major Radius (~0.809)
    r = (Phi - 1) / 2  # Minor Radius (~0.309)
    
    # Lepton Topological Winding Numbers (Accruing exactly 4 crossings per generation)
    leptons = [
        ("Electron ($e^-$)\nGround State ($3_1$)", 3),
        ("Muon ($\mu^-$)\n1st Resonance ($7_1$)", 7),
        ("Tau ($\\tau^-$)\n2nd Resonance ($11_1$)", 11)
    ]
    
    t = np.linspace(0, 2 * np.pi, 2500)
    fig = plt.figure(figsize=(18, 6), dpi=150)
    fig.patch.set_facecolor('#050508')
    
    for i, (label, p) in enumerate(leptons):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.set_facecolor('#050508')
        
        # Torus Knot Parametrization (q=2)
        x = (R + r * np.cos(2 * t)) * np.cos(p * t)
        y = (R + r * np.cos(2 * t)) * np.sin(p * t)
        z = r * np.sin(2 * t)
        
        # Compute geometric strain (local gradient density)
        dx, dy, dz = np.gradient(x), np.gradient(y), np.gradient(z)
        strain = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if i == 0: base_strain = np.max(strain)
        strain_normalized = strain / base_strain
        
        # Plot Soliton Flux Tube
        sc = ax.scatter(x, y, z, c=strain_normalized, cmap='inferno', s=40, alpha=0.9, vmin=0.5, vmax=3.0)
        ax.plot(x, y, z, color='white', linewidth=0.5, alpha=0.3)
        
        ax.set_title(label, color='white', fontsize=14, weight='bold', pad=10)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
        ax.axis('off')
        
        # Annotate Flux Crowding
        ax.text2D(0.5, 0.0, f"Max Local Strain: {np.max(strain_normalized):.2f}x", 
                  transform=ax.transAxes, color='cyan', ha='center', fontsize=12, weight='bold')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "topological_leptons.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Simulation Complete. Saved: {filepath}")
    plt.close()

if __name__ == "__main__":
    simulate_lepton_generations()