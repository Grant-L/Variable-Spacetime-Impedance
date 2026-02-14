import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import os

# Configuration
OUTPUT_DIR = "manuscript/chapters/13_thermodynamic_cycle/simulations"

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def simulate_ave_rubber_sheet():
    print("Simulating the AVE Saturable Rubber Sheet (Lattice Melt)...")
    
    np.random.seed(42)
    n_points = 3000
    Rs = 2.0  # Schwarzschild Radius
    R_max = 12.0
    
    # 1. Generate the Amorphous Discrete Lattice (M_A)
    # Bias points towards the center for the gravity well
    r = Rs + (R_max - Rs) * (np.random.power(2.0, n_points))
    theta = np.random.uniform(0, 2*np.pi, n_points)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Calculate Tensor Strain (Gravitational potential)
    GM = 8.0
    z = -GM / r
    
    # Triangulate
    triang = mtri.Triangulation(x, y)
    
    # Mask out triangles that cross the event horizon
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    rmid = np.sqrt(xmid**2 + ymid**2)
    mask = rmid < Rs
    triang.set_mask(mask)
    
    # Strain for coloring
    strain = GM / (r**2)
    
    # 2. Setup Plot
    fig = plt.figure(figsize=(14, 10), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    # 3. Plot the Discrete Lattice (M_A)
    # Plot as a wireframe to emphasize the discrete nature
    ax.plot_trisurf(triang, z, cmap='cool', shade=True,
                    linewidth=0.5, edgecolor=(0.0, 1.0, 1.0, 0.4), antialiased=True)
    
    # 4. The Pre-Geometric Melt (Inside Event Horizon)
    Z_floor = -GM / Rs
    
    # Scatter points for the melt (unstructured fluid)
    r_melt = np.sqrt(np.random.uniform(0, (Rs*0.95)**2, 4000))
    theta_melt = np.random.uniform(0, 2*np.pi, 4000)
    x_melt = r_melt * np.cos(theta_melt)
    y_melt = r_melt * np.sin(theta_melt)
    
    # Add thermal noise to the z-axis of the melt to represent infinite entropy
    z_melt = np.full_like(x_melt, Z_floor) + np.random.normal(0, 0.2, 4000)
    
    ax.scatter(x_melt, y_melt, z_melt, c=z_melt, cmap='magma', s=10, alpha=0.9, edgecolors='none')
    
    # Glowing ring at the Event Horizon (The Snap Point)
    theta_ring = np.linspace(0, 2*np.pi, 200)
    x_ring = Rs * np.cos(theta_ring)
    y_ring = Rs * np.sin(theta_ring)
    z_ring = np.full_like(x_ring, Z_floor)
    
    ax.plot(x_ring, y_ring, z_ring, color='#ff0055', linewidth=3)
    ax.plot(x_ring, y_ring, z_ring, color='#ff0055', linewidth=12, alpha=0.2) # Glow effect
    
    # 5. Add a Trefoil Knot (Matter) falling in
    t = np.linspace(0, 2*np.pi, 300)
    knot_scale = 0.6
    knot_x = 0 + knot_scale*(np.sin(t) + 2*np.sin(2*t))
    knot_y = 5.0 + knot_scale*(np.cos(t) - 2*np.cos(2*t))
    # Place it on the curved manifold
    knot_r = np.sqrt(knot_x**2 + knot_y**2)
    knot_z = (-GM / knot_r) + 1.0 - 0.4*np.sin(3*t)
    
    ax.plot(knot_x, knot_y, knot_z, color='#00ffcc', linewidth=3)
    
    # Formatting
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.axis('off')
    
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_zlim(Z_floor * 1.1, 2)
    ax.view_init(elev=35, azim=45)
    
    ax.text2D(0.5, 0.95, "AVE: The Death of the Rubber Sheet", transform=ax.transAxes, color='white', fontsize=22, ha='center', weight='bold')
    
    info_text = (
        "1. Tensor Strain: $M_A$ lattice stretches, increasing refractive index (Gravity).\n"
        "2. The Dielectric Snap: Strain exceeds $V_{break}$. Lattice snaps into liquid phase.\n"
        "3. Topological Sublimation: Knot geometry (Matter) erased upon entering the melt.\n"
        "RESULT: No Singularity. Information Paradox mechanically resolved."
    )
    ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes, color='white', fontsize=12, 
              bbox=dict(facecolor='black', edgecolor='#ff0055', alpha=0.8, pad=10))

    filepath = os.path.join(OUTPUT_DIR, "rubber_sheet_melt.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Simulation Complete. Saved: {filepath}")
    plt.close()

if __name__ == "__main__":
    ensure_output_dir()
    simulate_ave_rubber_sheet()