"""
AVE MODULE: Gravity as the Radial Elastic Wake of the Strong Force
-------------------------------------------------------------
This script visually renders the spatial metric embedded in the 
discrete M_A Cosserat condensate. It visualizes the radial 
displacement of the physical hardware nodes pulling perfectly 
into the center of mass.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

def simulate_ave_gravity_lattice():
    print("==========================================================")
    print(" AVE: VISUALIZING MACROSCOPIC GRAVITY AS LATTICE DENSIFICATION ")
    print("==========================================================")
    
    # --- 1. CONFIGURATION ---
    L_NODE = 10.0      # Simulated pitch for visual clarity
    BOX_SIZE = 120.0   # Span of the simulation volume
    EXAGGERATION = 8.0 # Exaggerate the displacement so the eye can see the "pull"
    
    print(f"Generating Amorphous Cosserat Vacuum Grid...")
    
    # --- 2. GENERATE UNPERTURBED AMORPHOUS LATTICE ---
    grid_pts = int(BOX_SIZE / L_NODE)
    x = np.linspace(-BOX_SIZE/2, BOX_SIZE/2, grid_pts)
    y = np.linspace(-BOX_SIZE/2, BOX_SIZE/2, grid_pts)
    z = np.linspace(-BOX_SIZE/2, BOX_SIZE/2, grid_pts)
    X, Y, Z = np.meshgrid(x, y, z)
    
    base_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    # Remove nodes directly at the origin to place the "Nucleus"
    distances = np.linalg.norm(base_points, axis=1)
    base_points = base_points[distances > L_NODE*0.5]
    
    # Add random amorphous jitter
    np.random.seed(42)
    jitter = (np.random.rand(*base_points.shape) - 0.5) * (0.4 * L_NODE)
    unperturbed_nodes = base_points + jitter

    # --- 3. APPLY GRAVITATIONAL DISPLACEMENT (The Radial Pull) ---
    print("Applying Radial Elastic Displacement (Gravity)...")
    
    displaced_nodes = []
    displacement_vectors = []
    strain_magnitudes = []
    
    for pt in unperturbed_nodes:
        r_vec = pt
        r_mag = np.linalg.norm(r_vec)
        
        # Displacement drops as 1/r^2 for a 3D bulk solid (Newtonian profile)
        u_mag = (L_NODE**3) / (r_mag**2) * 0.1 * EXAGGERATION
        
        # Cap displacement to preserve Axiom 1 topology (prevent singularity)
        if u_mag > L_NODE * 0.8:
            u_mag = L_NODE * 0.8
            
        # Direction is perfectly inward towards the mass
        u_vec = - (r_vec / r_mag) * u_mag
        
        displaced_nodes.append(pt + u_vec)
        displacement_vectors.append(u_vec)
        strain_magnitudes.append(u_mag)
        
    displaced_nodes = np.array(displaced_nodes)
    displacement_vectors = np.array(displacement_vectors)
    strain_magnitudes = np.array(strain_magnitudes)

    # --- 4. VISUALIZATION ---
    fig = plt.figure(figsize=(14, 12), facecolor='#0B0F19')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0B0F19')
    ax.set_axis_off()
    
    norm = mcolors.Normalize(vmin=0, vmax=np.max(strain_magnitudes)*0.8)
    cmap = plt.cm.plasma
    
    # Plot the final displaced nodes
    scat = ax.scatter(displaced_nodes[:,0], displaced_nodes[:,1], displaced_nodes[:,2],
                      c=strain_magnitudes, cmap=cmap, norm=norm, s=40, alpha=0.9, edgecolors='none')

    # Draw the displacement vectors (The "Pull" of gravity)
    # Filter to only show vectors near the center slice to avoid visual clutter
    mask = np.abs(unperturbed_nodes[:,2]) < L_NODE * 1.5
    ax.quiver(unperturbed_nodes[mask, 0], unperturbed_nodes[mask, 1], unperturbed_nodes[mask, 2],
              displacement_vectors[mask, 0], displacement_vectors[mask, 1], displacement_vectors[mask, 2],
              color='#00FFCC', alpha=0.5, linewidth=1.2, arrow_length_ratio=0.2)

    # Render the central topological mass 
    ax.scatter([0], [0], [0], color='#FFFFFF', s=300, edgecolors='#00FFFF', linewidth=2, zorder=10)
    ax.text(0, 0, L_NODE*1.5, "Topological Mass", color='white', ha='center', fontsize=12, weight='bold')

    ax.set_title("Macroscopic Gravity: Radial Lattice Densification\n(The continuous elastic wake of the Strong Force)", 
                 color='white', fontsize=16, pad=20, weight='bold')
    
    cbar = plt.colorbar(scat, ax=ax, shrink=0.5, pad=0.05)
    cbar.set_label('Local Metric Densification / Refractive Index ($n$)', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    limit = BOX_SIZE / 2.2
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    output_path = "manuscript/chapters/00_derivations/simulations/outputs/ave_gravity_lattice_densification.png"
    plt.savefig(output_path, dpi=300, facecolor='#0B0F19')
    print(f"Saved visualization to '{output_path}'")
    plt.show()

if __name__ == "__main__":
    simulate_ave_gravity_lattice()