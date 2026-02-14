import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay, Voronoi
import os

# Configuration
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = "ave_lattice_model.png"
N_NODES = 60       # Node count (kept low for visual clarity in 3D)
BOX_SIZE = 10.0    # Scale of the micro-volume
SEED = 42          # Fixed seed for consistent geometry

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def visualize_ave_lattice():
    """
    Generates a 3D visualization of the Discrete Amorphous Manifold (AVE Lattice).
    - Nodes: Inductive Mass centers (Red)
    - Edges: Capacitive Flux lines (Cyan)
    - Cell:  Effective Metric Volume (Yellow)
    """
    print(f"Generating 3D AVE Lattice Model (N={N_NODES})...")
    np.random.seed(SEED)

    # 1. Generate Amorphous Manifold (Poisson Distribution)
    points = np.random.rand(N_NODES, 3) * BOX_SIZE

    # 2. Delaunay Triangulation (The Flux Network)
    delaunay = Delaunay(points)

    # 3. Voronoi Tessellation (The Metric Volume)
    voronoi = Voronoi(points)

    # 4. Plotting Setup
    fig = plt.figure(figsize=(12, 10))
    # Use a dark background to represent the vacuum
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # A. Plot Lattice Edges (Capacitive Flux Tubes)
    # These represent the \epsilon_0 paths for photon propagation
    print("Tracing Flux Tubes...")
    for simplex in delaunay.simplices:
        pts = points[simplex]
        # Draw lines between all pairs in the tetrahedron
        combinations = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
        for pair in combinations:
            p1, p2 = pts[pair[0]], pts[pair[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    color='cyan', alpha=0.15, linewidth=0.5)

    # B. Plot Nodes (Inductive Centers)
    # These represent the \mu_0 inertial mass points
    ax.scatter(points[:,0], points[:,1], points[:,2], 
               c='red', s=40, depthshade=False, edgecolors='black', linewidth=0.5,
               label='Nodes (Inductance $\mu_0$)')

    # C. Plot a Representative Voronoi Cell (Effective Volume)
    # Highlighting one cell shows the "Pixel Size" of space
    print("Constructing Voronoi Metric Cell...")
    center = np.mean(points, axis=0)
    central_idx = np.argmin(np.linalg.norm(points - center, axis=1))
    
    region_idx = voronoi.point_region[central_idx]
    region = voronoi.regions[region_idx]
    
    if -1 not in region and len(region) > 0:
        polygon = [voronoi.vertices[i] for i in region]
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(polygon)
            faces = []
            for simplex in hull.simplices:
                faces.append([polygon[i] for i in simplex])
            
            # Plot the volume
            poly3d = Poly3DCollection(faces, alpha=0.4, facecolor='yellow', 
                                      edgecolor='orange', linewidth=1.5)
            ax.add_collection3d(poly3d)
            
            # Label the cell
            p_center = points[central_idx]
            ax.text(p_center[0], p_center[1], p_center[2]+1.5, 
                    "Voronoi Cell\n(Metric Volume)", color='yellow', fontsize=10, ha='center')
        except:
            print("Could not compute convex hull for the selected cell.")

    # D. Styling and Annotation
    ax.set_title("The Discrete Amorphous Manifold ($M_A$)", fontsize=16, color='white', pad=20)
    
    # Hide axes ticks for a clean "void" look, but keep labels
    ax.set_xlabel('X ($l_0$)', color='gray')
    ax.set_ylabel('Y ($l_0$)', color='gray')
    ax.set_zlabel('Z ($l_0$)', color='gray')
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    ax.tick_params(axis='z', colors='gray')
    
    # Remove pane fills
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
        Line2D([0], [0], color='cyan', lw=2),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', alpha=0.5, markersize=10)
    ]
    ax.legend(custom_lines, 
              ['Inductive Node ($\mu_0$)', 'Flux Edge ($\epsilon_0$)', 'Metric Volume ($\kappa$)'], 
              loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')

    # Save
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    ensure_output_dir()
    visualize_ave_lattice()