import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    os.makedirs('assets/sim_outputs', exist_ok=True)
    
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Hide axes
    ax.axis('off')
    
    # Draw the cosmological horizon (Background expanding sphere)
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    r_sphere = 2.0
    x = r_sphere * np.outer(np.cos(u), np.sin(v))
    y = r_sphere * np.outer(np.sin(u), np.sin(v))
    z = r_sphere * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='purple', alpha=0.04, edgecolor='none')
    
    # Draw Radial Expansion Vectors (a_r)
    np.random.seed(42)
    phi = np.random.uniform(0, 2*np.pi, 80)
    costheta = np.random.uniform(-1, 1, 80)
    theta = np.arccos(costheta)
    vx = r_sphere * np.sin(theta) * np.cos(phi)
    vy = r_sphere * np.sin(theta) * np.sin(phi)
    vz = r_sphere * np.cos(theta)
    
    # Ensure vectors point outward to show cosmic expansion
    ax.quiver(vx*0.8, vy*0.8, vz*0.8, vx*0.2, vy*0.2, vz*0.2, 
              color='magenta', alpha=0.3, linewidth=1.5, arrow_length_ratio=0.1)
    
    # Draw the 1D Particle Loop 
    # (A simple embedded circular ring representing the perimeter of a localized defect)
    r_ring = 1.0
    theta_ring = np.linspace(0, 2*np.pi, 200)
    ring_x = r_ring * np.cos(theta_ring)
    ring_y = r_ring * np.sin(theta_ring)
    ring_z = np.zeros_like(theta_ring)
    
    # Draw the Ring (Solid White with a cyan glow)
    ax.plot(ring_x, ring_y, ring_z, color='cyan', linewidth=10, alpha=0.3)
    ax.plot(ring_x, ring_y, ring_z, color='white', linewidth=3, label='1D Topological Loop')
    
    # Draw the continuous outward radial projection (F_r) physically pulling ON THE RING
    theta_sample = np.linspace(0, 2*np.pi, 16, endpoint=False)
    sx = r_ring * np.cos(theta_sample)
    sy = r_ring * np.sin(theta_sample)
    sz = np.zeros_like(theta_sample)
    
    # Radial tension acting directly OUTWARD on the ring
    dr_x = sx * 0.5
    dr_y = sy * 0.5
    dr_z = np.zeros_like(theta_sample)
    ax.quiver(sx, sy, sz, dr_x, dr_y, dr_z, color='#ffaa00', linewidth=2.5, arrow_length_ratio=0.2)
    
    # Draw the internally derived longitudinal Hoop Stress vectors 
    # (Tension stretches the ring circumferentially)
    for i in range(len(theta_sample)):
        p = theta_sample[i]
        px, py = r_ring * np.cos(p), r_ring * np.sin(p)
        # Tangent vector for the hoop stress
        tx, ty = -np.sin(p), np.cos(p)
        
        # Plot two arrows pulling tangentially in both directions from the node
        ax.quiver(px, py, 0, tx*0.25, ty*0.25, 0, color='#00ff00', linewidth=3, arrow_length_ratio=0.3)
        ax.quiver(px, py, 0, -tx*0.25, -ty*0.25, 0, color='#00ff00', linewidth=3, arrow_length_ratio=0.3)
        
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-2.2, 2.2])
    ax.set_ylim([-2.2, 2.2])
    ax.set_zlim([-2.2, 2.2])
    ax.view_init(elev=35, azim=45)
    
    plt.title("Cosmic MOND Acceleration ($a_0$)\nDeriving Hoop Stress on a 1D Closed Topology", 
              color='white', pad=20, fontsize=16, fontweight='bold')
              
    plt.tight_layout()
    output_path = 'assets/sim_outputs/unruh_hawking_hoop_stress.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Successfully saved Hoop Stress MOND simulation to {output_path}")

if __name__ == '__main__':
    main()
