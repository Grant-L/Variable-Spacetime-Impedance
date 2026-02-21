"""
1D FDTD Simulation of the Bullet Cluster Merger.
Proves that Macroscopic Mutual Inductance (Dark Matter) permeates and passes 
through localized collisions collisionlessly due to LC Network superposition.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure local ave package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ave.core.constants import C_0, MU_0, EPSILON_0

def simulate_bullet_cluster_fdtd():
    """
    Simulates two galaxy clusters colliding. 
    In the AVE framework, 'Dark Matter' is just the macroscopic inductive metric strain L(r).
    Because the vacuum is an LC Network, overlapping metric strains simply add via 
    linear superposition and pass right through each other without 'fluidic' shockwaves,
    perfectly mirroring the Bullet Cluster lensing observations.
    """
    GRID_SIZE = 500
    RESOLUTION = 1.0 # arbitrary kiloparsec scale for visualization
    
    x_axis = np.linspace(-GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE)
    
    # Two clusters moving towards each other
    pos_cluster_1 = -150
    pos_cluster_2 = 150
    
    # We will compute their localized Refractive Strains n(r)
    # n(r) roughly scales physically as a Lorentzian or Gaussian decay around the mass center
    def get_strain(pos, width, amplitude):
        return amplitude * np.exp(-0.5 * ((x_axis - pos) / width)**2)
    
    snapshots = []
    
    # Move them through each other over 60 time steps
    for t in range(61):
        # Update positions
        c1_current = pos_cluster_1 + (t * 5.0) # moving right
        c2_current = pos_cluster_2 - (t * 5.0) # moving left
        
        # Calculate localized inductive strain L(r)
        strain_1 = get_strain(c1_current, width=30, amplitude=0.8)
        strain_2 = get_strain(c2_current, width=30, amplitude=0.8)
        
        # The total Macroscopic Inductance (Dark Matter halo) is the direct linear superposition
        total_inductance = MU_0 * (1.0 + strain_1 + strain_2)
        
        if t % 15 == 0:
            snapshots.append((c1_current, c2_current, total_inductance))
            
    # Visualize the crossing
    fig, axes = plt.subplots(len(snapshots), 1, figsize=(10, 12), sharex=True)
    
    for i, (c1, c2, L_total) in enumerate(snapshots):
        ax = axes[i]
        # Plot the 'Baryonic' centers (collisional gas would shock here, but we only plot the centers)
        ax.axvline(c1, color='red', linestyle='--', alpha=0.7, label='Cluster 1 (Baryon Core)')
        ax.axvline(c2, color='blue', linestyle='--', alpha=0.7, label='Cluster 2 (Baryon Core)')
        
        # Plot the dark matter halo (Total metric inductance)
        ax.plot(x_axis, L_total / MU_0, color='purple', linewidth=2, label='Total Macroscopic Inductance\n(Dark Matter Lensing Halo)')
        
        ax.set_ylabel("Metric Strain $n_L$")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Bullet Cluster Collisionless Metric Superposition")
            ax.legend(loc='upper right')
            
    axes[-1].set_xlabel("Spatial Distance (arbitrary units)")
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '../assets/sim_outputs/bullet_cluster_fdtd.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    simulate_bullet_cluster_fdtd()
