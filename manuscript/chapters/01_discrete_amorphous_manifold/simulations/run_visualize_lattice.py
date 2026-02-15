"""
AVE MODULE 2: LATTICE VISUALIZATION
-----------------------------------
Generates a mathematically pristine 3D visualization of the M_A Manifold.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay, Voronoi, ConvexHull
from scipy.stats import qmc
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
OUTPUT_FILE = "ave_lattice_model.png"
BOX_SIZE = 10.0    
# Kept large to yield a sparse, visually clear lattice (~60 nodes)
MIN_DIST = 1.8  

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def visualize_ave_lattice():
    print("Generating 3D M_A Lattice Model...")
    
    # Poisson-Disk Sampling to enforce minimum discrete separation
    engine = qmc.PoissonDisk(d=3, radius=MIN_DIST/BOX_SIZE, seed=42)
    points = engine.fill_space() * BOX_SIZE
    print(f"Visualizing {len(points)} discrete nodes.")
    
    delaunay = Delaunay(points)
    voronoi = Voronoi(points)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # A. Plot Lattice Edges (Extract Unique Edges to fix alpha ghosting)
    print("Tracing Unique Flux Tubes...")
    unique_edges = set()
    for simplex in delaunay.simplices:
        for i in range(4):
            for j in range(i+1, 4):
                idx1, idx2 = sorted([simplex[i], simplex[j]])
                unique_edges.add((idx1, idx2))
                
    for p1_idx, p2_idx in unique_edges:
        p1, p2 = points[p1_idx], points[p2_idx]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                color='cyan', alpha=0.25, linewidth=0.8)

    # B. Plot Nodes (Inductive Centers -> L_node)
    ax.scatter(points[:,0], points[:,1], points[:,2], 
               c='red', s=45, depthshade=False, edgecolors='black', linewidth=0.5)

    # C. Plot a Representative Voronoi Cell (Effective Nodal Volume)
    print("Constructing Voronoi Metric Cell...")
    center = np.array([BOX_SIZE/2, BOX_SIZE/2, BOX_SIZE/2])
    central_idx = np.argmin(np.linalg.norm(points - center, axis=1))
    
    region_idx = voronoi.point_region[central_idx]
    region = voronoi.regions[region_idx]
    
    if -1 not in region and len(region) > 0:
        polygon = [voronoi.vertices[i] for i in region]
        try:
            hull = ConvexHull(polygon)
            faces = [[polygon[i] for i in simplex] for simplex in hull.simplices]
            
            poly3d = Poly3DCollection(faces, alpha=0.3, facecolor='yellow', 
                                      edgecolor='orange', linewidth=1.5)
            ax.add_collection3d(poly3d)
            
            p_center = points[central_idx]
            ax.text(p_center[0], p_center[1], p_center[2]+1.5, 
                    "Metric Volume\n($V_{node}$)", color='yellow', fontsize=11, ha='center', weight='bold')
        except Exception as e:
            print(f"Could not compute convex hull: {e}")

    # D. Styling and Strict Physical Annotation
    ax.set_title("The Discrete Amorphous Manifold ($M_A$)", fontsize=18, color='white', pad=20)
    
    ax.set_xlabel('X ($l_{node}$)', color='gray')
    ax.set_ylabel('Y ($l_{node}$)', color='gray')
    ax.set_zlabel('Z ($l_{node}$)', color='gray')
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    ax.tick_params(axis='z', colors='gray')
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    # Custom Legend strictly using VSE terminology
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, linestyle='None'),
        Line2D([0], [0], color='cyan', lw=2),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', alpha=0.5, markersize=10, linestyle='None')
    ]
    ax.legend(custom_lines, 
              ['Discrete Inductance ($L_{node}$)', 'Discrete Capacitance ($C_g$)', 'Metric Volume ($\\kappa_V$)'], 
              loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    ensure_output_dir()
    visualize_ave_lattice()