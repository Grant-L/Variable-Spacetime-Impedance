import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import os

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_raw_lattice():
    print("Generating Discrete Amorphous Manifold (MA)...")
    
    # 1. HARDWARE GENERATION
    np.random.seed(137) # Fine Structure Seed
    n_nodes = 2000
    L = 10.0
    
    # Poisson Distribution (The "Melt")
    points = np.random.rand(n_nodes, 3) * L
    
    # Triangulation (The Crystallization)
    tri = Delaunay(points)
    
    print(f"Lattice Crystallized: {n_nodes} Nodes, {len(tri.simplices)} Cells")

    # 2. VISUALIZATION
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Nodes (The Inductive Mass)
    ax.scatter(points[:,0], points[:,1], points[:,2], s=5, c='gray', alpha=0.6, label=r'Inductive Nodes ($\mu_0$)')
    
    # Plot a subset of Edges (The Capacitive Flux)
    # We plot edges only near the center to avoid clutter
    indptr, indices = tri.vertex_neighbor_vertices
    center = np.array([L/2, L/2, L/2])
    
    edge_count = 0
    for i in range(n_nodes):
        if np.linalg.norm(points[i] - center) > 2.0: continue
        
        for j in indices[indptr[i]:indptr[i+1]]:
            if i < j:
                # Plot each edge individually to avoid None value issues in 3D
                ax.plot([points[i,0], points[j,0]], 
                       [points[i,1], points[j,1]], 
                       [points[i,2], points[j,2]], 
                       c='cyan', alpha=0.3, linewidth=0.5)
                edge_count += 1
    
    if edge_count > 0:
        # Create a dummy plot for the legend
        ax.plot([], [], [], c='cyan', alpha=0.3, linewidth=0.5, label=r'Flux Edges ($\epsilon_0$)')
    
    # Aesthetics
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_title("The Amorphous Lattice (No Signal)\nThe 'Jagged' Hardware of Space", color='white')
    ax.legend(loc='lower right')
    
    output_path = os.path.join(OUTPUT_DIR, "photon_lattice.png")
    plt.savefig(output_path, dpi=300, facecolor='black')
    print(f"Lattice map saved to {output_path}")

if __name__ == "__main__":
    simulate_raw_lattice()