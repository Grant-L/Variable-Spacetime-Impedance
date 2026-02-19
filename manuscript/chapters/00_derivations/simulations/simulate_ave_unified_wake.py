"""
AVE MODULE: Unified Visualization of the Local Bubble and Gravitational Wake
-------------------------------------------------------------------------
Visualizes the discrete M_A Cosserat condensate. 
Renders the central topological knot (The Gamma = -1 Bubble) 
and the resulting radial elastic wake (Macroscopic Gravity) 
pulling perfectly into the center of mass.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

def simulate_unified_lattice_dynamics():
    print("==========================================================")
    print(" AVE: VISUALIZING UNIFIED LATTICE DYNAMICS ")
    print(" The Gamma=-1 Core & The Radial Gravitational Wake ")
    print("==========================================================")

    # 1. PARAMETERS
    BOX_SIZE = 80.0
    L_NODE = 3.0
    R_CORE = 8.0  
    
    # 2. GENERATE UNPERTURBED AMORPHOUS LATTICE
    grid_pts = int(BOX_SIZE / L_NODE)
    x = np.linspace(-BOX_SIZE/2, BOX_SIZE/2, grid_pts)
    X, Y, Z = np.meshgrid(x, x, x)
    
    base_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    np.random.seed(42)
    jitter = (np.random.rand(*base_points.shape) - 0.5) * (0.6 * L_NODE)
    unperturbed_nodes = base_points + jitter

    # 3. APPLY UNIFIED DYNAMICS (Twist + Pull)
    displaced_nodes = []
    vectors = []
    strains = []
    is_core = []
    
    for pt in unperturbed_nodes:
        r_vec = pt
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 0.1: r_mag = 0.1
        
        # Region 1: The Local Bubble (Matter Core)
        if r_mag <= R_CORE:
            # Nodes jammed to the surface (Gamma = -1 Impedance Wall)
            u_vec = (r_vec / r_mag) * (R_CORE - r_mag) * 0.9 
            displaced_nodes.append(pt + u_vec)
            vectors.append([0, 0, 0])
            strains.append(0)
            is_core.append(True)
            
        # Region 2: The Gravitational Wake (Ambient Lattice Pull)
        else:
            # Elastic displacement pulling perfectly to the center
            # Magnitude drops as 1/r^2 (Trace-reversed volumetric pull)
            pull_mag = (R_CORE**2 / r_mag**2) * (L_NODE * 0.5)
            u_vec = -(r_vec / r_mag) * pull_mag
            
            displaced_nodes.append(pt + u_vec)
            vectors.append(u_vec)
            strains.append(pull_mag)
            is_core.append(False)

    displaced_nodes = np.array(displaced_nodes)
    vectors = np.array(vectors)
    strains = np.array(strains)
    is_core = np.array(is_core)

    # 4. VISUALIZATION
    fig = plt.figure(figsize=(14, 12), facecolor='#0B0F19')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0B0F19')
    ax.set_axis_off()
    
    # Plot ambient vacuum nodes (Gravity wake)
    scat = ax.scatter(displaced_nodes[~is_core, 0], 
                      displaced_nodes[~is_core, 1], 
                      displaced_nodes[~is_core, 2],
                      c=strains[~is_core], cmap='plasma', s=35, alpha=0.7, edgecolors='none')
                      
    # Plot core nodes (Gamma = -1 boundary)
    ax.scatter(displaced_nodes[is_core, 0], 
               displaced_nodes[is_core, 1], 
               displaced_nodes[is_core, 2],
               color='#00FFFF', s=90, alpha=0.9, edgecolors='white', linewidth=0.5)
               
    # Draw Gravitational Pull Vectors
    # Filter for clarity: only show vectors near the center slice
    vec_mask = (~is_core) & (np.linalg.norm(vectors, axis=1) > 0.1) & (np.abs(unperturbed_nodes[:,2]) < L_NODE * 1.5)
    ax.quiver(unperturbed_nodes[vec_mask, 0], unperturbed_nodes[vec_mask, 1], unperturbed_nodes[vec_mask, 2],
              vectors[vec_mask, 0]*4, vectors[vec_mask, 1]*4, vectors[vec_mask, 2]*4,
              color='#FF3366', alpha=0.6, linewidth=1.5, arrow_length_ratio=0.3)

    # Draw a faint sphere representing the TIR boundary
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    X = R_CORE * np.outer(np.cos(u), np.sin(v))
    Y = R_CORE * np.outer(np.sin(u), np.sin(v))
    Z = R_CORE * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(X, Y, Z, color='#00FFFF', alpha=0.1, shade=True)

    # Polish
    ax.set_title("Unified Lattice Dynamics:\nThe $\Gamma = -1$ Core (Matter) & The Radial Elastic Wake (Gravity)", 
                 color='white', fontsize=16, pad=20, weight='bold')
                 
    cbar = plt.colorbar(scat, ax=ax, shrink=0.5, pad=0.05)
    cbar.set_label('Gravitational Strain Amplitude ($1/r^2$)', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    limit = BOX_SIZE / 2.2
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig("manuscript/chapters/00_derivations/simulations/outputs/unified_lattice_dynamics.png", dpi=300, facecolor='#0B0F19')
    print("Saved visualization to 'unified_lattice_dynamics.png'")
    plt.show()

if __name__ == "__main__":
    simulate_unified_lattice_dynamics()