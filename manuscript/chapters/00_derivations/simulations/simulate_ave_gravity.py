"""
AVE MODULE: Gravity as the Elastic Wake of the Strong Force
-------------------------------------------------------------
This script mathematically renders the spatial metric embedded in the 
discrete M_A Cosserat condensate. It explicitly preserves the 386 fm 
hardware pitch (Axiom 1) and models the empirical 1.94 fm bond length 
strictly as the RMS elastic displacement amplitude of the strain field.

Gravity is visualized as the 3D trace-reversed volumetric densification 
of the ambient lattice nodes pulling toward the center of mass.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

def simulate_ave_gravity_wake():
    print("==========================================================")
    print(" AVE 1ST PRINCIPLES: GRAVITY AS ELASTIC DISPLACEMENT WAKE")
    print("==========================================================")
    
    # --- 1. HARDWARE CALIBRATION (Axiom 1) ---
    L_NODE = 386.16  # Fundamental Lattice Pitch (fm)
    MAX_DISP = 1.94  # Max elastic strain at the core (fm)
    BOX_SIZE = L_NODE * 4.0  # Simulate a box spanning a few lattice nodes
    
    print(f"Lattice Pitch: {L_NODE} fm")
    print(f"Max Core Displacement: {MAX_DISP} fm ({(MAX_DISP/L_NODE)*100:.3f}% Strain)")
    
    # --- 2. GENERATE AMORPHOUS LATTICE ---
    grid_pts = int(BOX_SIZE / L_NODE) + 1
    x = np.linspace(-BOX_SIZE/2, BOX_SIZE/2, grid_pts)
    y = np.linspace(-BOX_SIZE/2, BOX_SIZE/2, grid_pts)
    z = np.linspace(-BOX_SIZE/2, BOX_SIZE/2, grid_pts)
    X, Y, Z = np.meshgrid(x, y, z)
    
    base_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    # Remove exact center node
    distances = np.linalg.norm(base_points, axis=1)
    base_points = base_points[distances > 0.1]
    
    # Add amorphous jitter (prevent perfect cubic symmetry)
    np.random.seed(42)
    jitter = (np.random.rand(*base_points.shape) - 0.5) * (0.3 * L_NODE)
    unperturbed_nodes = base_points + jitter

    # --- 3. APPLY GRAVITATIONAL DISPLACEMENT FIELD ---
    displaced_nodes = []
    displacement_vectors = []
    strain_magnitudes = []
    
    for pt in unperturbed_nodes:
        r_vec = pt  
        r_mag = np.linalg.norm(r_vec)
        
        # 3D volumetric displacement drops off as 1/r^2
        u_mag = MAX_DISP * ((L_NODE / r_mag)**2)
        if u_mag > MAX_DISP: u_mag = MAX_DISP
            
        # The displacement vector points strictly inward toward mass center
        u_vec = - (r_vec / r_mag) * u_mag
        
        displaced_nodes.append(pt + u_vec)
        displacement_vectors.append(u_vec)
        strain_magnitudes.append(u_mag / L_NODE) # True mechanical strain
        
    displaced_nodes = np.array(displaced_nodes)
    displacement_vectors = np.array(displacement_vectors)
    strain_magnitudes = np.array(strain_magnitudes)

    # --- 4. VISUALIZATION ---
    fig = plt.figure(figsize=(14, 11), facecolor='#0B0F19')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0B0F19')
    ax.set_axis_off()
    
    # Exaggeration factor purely to make the 1.94fm shift visible on a 1500fm plot
    VISUAL_MULTIPLIER = 40.0 
    
    # Render Substrate Nodes (Colored by Strain/Refractive Index)
    norm = mcolors.Normalize(vmin=0, vmax=MAX_DISP/L_NODE)
    cmap = plt.cm.plasma
    scat = ax.scatter(displaced_nodes[:,0], displaced_nodes[:,1], displaced_nodes[:,2],
                      c=strain_magnitudes, cmap=cmap, norm=norm, s=60, alpha=0.9, edgecolors='none')

    # Render Faint Unperturbed Shadow Nodes (Where they started)
    ax.scatter(unperturbed_nodes[:,0], unperturbed_nodes[:,1], unperturbed_nodes[:,2],
               color='gray', s=20, alpha=0.3, edgecolors='none')

    # Render Gravitational Pull (Quiver/Vectors)
    ax.quiver(unperturbed_nodes[:,0], unperturbed_nodes[:,1], unperturbed_nodes[:,2],
              displacement_vectors[:,0] * VISUAL_MULTIPLIER, 
              displacement_vectors[:,1] * VISUAL_MULTIPLIER, 
              displacement_vectors[:,2] * VISUAL_MULTIPLIER,
              color='#00FFCC', alpha=0.6, linewidth=1.5, arrow_length_ratio=0.2)

    # Render Helium-4 Nucleus (The Central Defect / Scattering Cloud)
    ax.scatter([0], [0], [0], color='#FFD700', s=2500, edgecolors='white', linewidth=2, alpha=0.15)
    ax.scatter([0], [0], [0], color='#FFFFFF', s=200, alpha=1.0)
    
    ax.set_title("AVE Macroscopic Gravity: Radial Elastic Displacement Field\n(Gravity as the Wake of the Strong Force)", 
                 color='white', fontsize=16, pad=20, weight='bold')
    
    cbar = plt.colorbar(scat, ax=ax, shrink=0.5, pad=0.05)
    cbar.set_label('Local Metric Strain (Optical Refractive Density)', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    limit = BOX_SIZE / 2.2
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig("manuscript/chapters/00_derivations/simulations/outputs/ave_gravity_displacement.png", dpi=300, facecolor='#0B0F19')
    print("Saved simulation to 'ave_gravity_displacement.png'")
    plt.show()

if __name__ == "__main__":
    simulate_ave_gravity_wake()