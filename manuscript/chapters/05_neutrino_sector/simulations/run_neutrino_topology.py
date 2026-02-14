import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = "manuscript/chapters/05_neutrino_sector/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_neutrino_topology():
    print("Simulating Neutrino Topology (Twisted Unknot 0_1)...")
    
    # 1. GEOMETRY: The Unknot (A simple circle)
    theta = np.linspace(0, 2*np.pi, 200)
    R = 1.0 # Radius of the loop
    
    # Centerline path
    x_core = R * np.cos(theta)
    y_core = R * np.sin(theta)
    z_core = np.zeros_like(theta)
    
    # 2. TOPOLOGY: The Internal Twist (Helicity)
    # The neutrino is a 'Twisted Rubber Band'
    # It has 0 crossings (Charge = 0) but internal torsion (Mass > 0)
    
    # We visualize the twist by plotting a "stripe" on the surface of the tube
    tube_radius = 0.2
    twist_rate = 0.5 # Half-twist (Mobius-like) or Full twist
    
    # Surface path (winding around the core)
    # The twist angle varies with theta
    twist_angle = twist_rate * theta 
    
    # Parametric surface point
    x_surf = (R + tube_radius * np.cos(twist_angle)) * np.cos(theta)
    y_surf = (R + tube_radius * np.cos(twist_angle)) * np.sin(theta)
    z_surf = tube_radius * np.sin(twist_angle)
    
    # 3. VISUALIZATION
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the Core (Invisible/Ghost-like)
    ax.plot(x_core, y_core, z_core, 'w--', linewidth=1, alpha=0.5, label='Topological Void (N=0, q=0)')
    
    # Plot the Lattice Stress (The Twist)
    # Color depends on the twist stress (z-displacement)
    p = ax.scatter(x_surf, y_surf, z_surf, c=z_surf, cmap='twilight', s=50, label='Lattice Torsion (Mass)')
    
    # Physics Annotation
    ax.text2D(0.05, 0.95, "The Neutrino ($0_1$)", transform=ax.transAxes, color='white', fontsize=16, fontweight='bold')
    ax.text2D(0.05, 0.90, "Topology: Unknot (No Crossings)\nCharge: q = 0 (No trapped flux)\nMass: $m \\propto \\tau_{twist}^2$ (Torsional Energy)", 
              transform=ax.transAxes, color='cyan', fontsize=12)
    
    # View Settings
    ax.set_facecolor('black')
    ax.axis('off')
    ax.view_init(elev=60, azim=45)
    ax.legend(loc='lower right')
    
    output_path = os.path.join(OUTPUT_DIR, "neutrino_winding.png")
    plt.savefig(output_path, dpi=300, facecolor='black')
    print(f"Neutrino simulation saved to {output_path}")

if __name__ == "__main__":
    simulate_neutrino_topology()