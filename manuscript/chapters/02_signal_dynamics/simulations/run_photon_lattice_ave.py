import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import os

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_rifled_photon():
    print("Simulating Discrete Photon Propagation (Helicity m=1)...")
    
    # 1. GENERATE HARDWARE (Axiom I: Amorphous Manifold)
    # We generate a Poisson distribution of nodes (The Vacuum)
    np.random.seed(42) # Consistent Universe
    n_nodes = 4000
    L = 20.0 # Volume Size
    
    # Random node positions
    points = np.random.rand(n_nodes, 3) * L
    
    # Triangulate (The Connectivity)
    # This creates the edges (Flux Tubes)
    tri = Delaunay(points)
    
    # 2. DEFINE THE SIGNAL (Axiom II: State Variables)
    # We define a "Rifled Pulse" (Photon) moving along Z-axis.
    # Center of pulse at time t
    t = 10.0 
    c = 1.0   # Normalized Slew Rate
    k = 2.0   # Wave number
    sigma = 2.5 # Pulse width
    
    # Calculate Field Strength on Edges
    # In AVE, light is a wave of Edge Flux (U_ij).
    # Flux = Potential Difference (Grad Phi) * Helicity
    
    edges_to_draw = []
    
    # We filter for edges that are currently "lit up" by the pulse to save rendering time
    # The pulse is at z = c*t = 10
    z_center = c * t
    
    print("Computing Flux across Lattice Edges...")
    
    # Extract unique edges from triangulation
    indptr, indices = tri.vertex_neighbor_vertices
    
    # Vectorized Edge Calculation would be faster, but we iterate for clarity of physics
    # We visualize a subset of edges near the pulse center
    
    active_edges_x = []
    active_edges_y = []
    active_edges_z = []
    active_colors = []
    
    # Iterate through nodes to find neighbors
    for i in range(n_nodes):
        # Position of Node i
        pos_i = points[i]
        
        # Check if node is in the "Active Window" of the pulse (Optimization)
        # Pulse is Gaussian in Z
        if abs(pos_i[2] - z_center) > 3 * sigma:
            continue
            
        # Get neighbors
        neighbors = indices[indptr[i]:indptr[i+1]]
        
        for j in neighbors:
            if i > j: continue # Avoid duplicates
            
            pos_j = points[j]
            
            # Midpoint of edge (for field calculation)
            mid = (pos_i + pos_j) / 2.0
            
            # 3. APPLY WAVE MECHANICS (The Rifling)
            # Longitudinal position relative to pulse center
            dz = mid[2] - z_center
            
            # Transverse position
            r = np.sqrt((mid[0]-L/2)**2 + (mid[1]-L/2)**2)
            
            # Envelope (Gaussian Packet)
            envelope = np.exp(- (dz**2) / (2*sigma**2)) * np.exp(- (r**2) / (2*(sigma/1.5)**2))
            
            if envelope < 0.05: continue # Skip dark vacuum
            
            # Phase Angle (The Spin)
            # Helicity m=1: Phase depends on angle theta around Z axis
            theta = np.arctan2(mid[1]-L/2, mid[0]-L/2)
            
            # The Wave Function psi = Envelope * exp(i(kz - wt + m*theta))
            # Real component determines physical stress
            phase = k * dz + 1.0 * theta # m=1 Spin
            amplitude = envelope * np.cos(phase)
            
            # Color Mapping (Blue = Positive Stress, Red = Negative Stress)
            # This visualizes the "Twist" propagating through the lattice
            color_val = (amplitude + 1) / 2 # Normalize 0-1
            
            active_edges_x.extend([pos_i[0], pos_j[0], None])
            active_edges_y.extend([pos_i[1], pos_j[1], None])
            active_edges_z.extend([pos_i[2], pos_j[2], None])
            active_colors.append(color_val)

    # 4. VISUALIZATION
    print(f"Rendering {len(active_colors)} active flux tubes...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the "Dark" Lattice (The background Vacuum)
    # We plot faint dots for nodes to show the medium density
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0.5, c='gray', alpha=0.1)
    
    # Plot the "Active" Flux Tubes (The Photon)
    # We map the phase amplitude to a colormap
    cmap = plt.get_cmap('coolwarm') # Blue-Red for + / - Phase
    
    # We have to plot segments individually or use a LineCollection (complex in 3D)
    # For simulation proof, we plot a subset of high-intensity edges
    
    # Visual Trick: We plot the points of the edges colored by phase
    # This creates a "Cloud" representing the wave packet structure
    
    # Re-loop for scatter plot visualization (cleaner for dense lattice)
    x_scat, y_scat, z_scat, c_scat = [], [], [], []
    
    for k_idx in range(len(active_colors)):
        # Reconstruct segment midpoints for scatter
        # (Simplified visualization of the field density)
        idx_base = k_idx * 3
        mx = (active_edges_x[idx_base] + active_edges_x[idx_base+1])/2
        my = (active_edges_y[idx_base] + active_edges_y[idx_base+1])/2
        mz = (active_edges_z[idx_base] + active_edges_z[idx_base+1])/2
        
        x_scat.append(mx)
        y_scat.append(my)
        z_scat.append(mz)
        c_scat.append(active_colors[k_idx])

    sc = ax.scatter(x_scat, y_scat, z_scat, c=c_scat, cmap='coolwarm', s=10, alpha=0.8)
    
    # Aesthetics
    ax.set_facecolor('black')
    ax.grid(False)
    ax.set_axis_off()
    
    ax.set_title("AVE Simulation: Photon Propagation (Helicity m=1)\nBlue/Red = +/- Phase Twist on Lattice Edges", color='white')
    
    # Add annotation about Rifling
    ax.text2D(0.05, 0.05, "Note the Spiral Phase Structure\nStabilizing the path through random nodes", 
              transform=ax.transAxes, color='white', fontsize=10)

    output_path = os.path.join(OUTPUT_DIR, "photon_lattice_ave.png")
    plt.savefig(output_path, dpi=300, facecolor='black')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    simulate_rifled_photon()