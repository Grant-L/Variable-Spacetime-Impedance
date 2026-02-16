"""
AVE MODULE 3: LATTICE VISUALIZATION (Cosserat Over-Bracing)
-----------------------------------------------------------
Generates a mathematically pristine 3D visualization of the M_A Manifold.
Crucially, it maps BOTH the primary kinematic links and the secondary 
transverse Cosserat links required to satisfy the derived \\kappa_V = 8\\pi\\alpha limit.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi, ConvexHull, cKDTree
from scipy.stats import qmc
from matplotlib.lines import Line2D
import os
import warnings

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = "ave_cosserat_lattice_model.png"

BOX_SIZE = 10.0    
MIN_DIST = 1.8  
# Strict structural span derived in Module 1 to satisfy \kappa_V
COSSERAT_RATIO = 1.67 

def visualize_ave_lattice():
    print("Generating 3D Trace-Reversed Cosserat Lattice Model...")
    
    # 1. Generate Discrete Hardware Nodes
    engine = qmc.PoissonDisk(d=3, radius=MIN_DIST/BOX_SIZE, seed=42)
    points = engine.fill_space() * BOX_SIZE
    N = len(points)
    print(f"Visualizing {N} discrete nodes.")
    
    kd_tree = cKDTree(points)
    voronoi = Voronoi(points)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # 2. Trace Structural Links (Primary Kinematic vs Secondary Cosserat)
    print("Tracing Kinematic and Transverse Flux Tubes...")
    primary_edges = set()
    cosserat_edges = set()
    
    primary_radius = MIN_DIST * 1.2
    cosserat_radius = MIN_DIST * COSSERAT_RATIO
    
    for i in range(N):
        # Primary Links
        p_neighbors = kd_tree.query_ball_point(points[i], r=primary_radius)
        for j in p_neighbors:
            if i < j: primary_edges.add((i, j))
        
        # Transverse Cosserat Links (Over-Bracing)
        c_neighbors = kd_tree.query_ball_point(points[i], r=cosserat_radius)
        for j in c_neighbors:
            if i < j and (i, j) not in primary_edges:
                cosserat_edges.add((i, j))

    # Plot Transverse Cosserat Links (Magenta/Purple - Source of 2/7 Poisson Ratio)
    for p1_idx, p2_idx in cosserat_edges:
        p1, p2 = points[p1_idx], points[p2_idx]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                color='#ff00ff', alpha=0.25, linewidth=0.8, linestyle=':')

    # Plot Primary Kinematic Links (Cyan)
    for p1_idx, p2_idx in primary_edges:
        p1, p2 = points[p1_idx], points[p2_idx]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                color='cyan', alpha=0.5, linewidth=1.5)

    # 3. Plot Nodes (Inductive Centers)
    ax.scatter(points[:,0], points[:,1], points[:,2], 
               c='red', s=60, depthshade=False, edgecolors='white', linewidth=0.5, zorder=5)

    # 4. Plot a Representative Voronoi Cell (Effective Nodal Volume \kappa_V)
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
            
            poly3d = Poly3DCollection(faces, alpha=0.25, facecolor='yellow', 
                                      edgecolor='orange', linewidth=1.5)
            ax.add_collection3d(poly3d)
            
            p_center = points[central_idx]
            ax.text(p_center[0], p_center[1], p_center[2]+1.5, 
                    r"Metric Volume\n($\kappa_V$)", color='yellow', fontsize=11, ha='center', weight='bold')
        except Exception as e:
            pass

    # 5. Styling and Strict Physical Annotation
    ax.set_title(r"The Discrete Amorphous Manifold ($\mathcal{M}_A$)\nHighlighting Cosserat Transverse Over-Bracing", 
                 fontsize=16, color='white', pad=20, weight='bold')
    
    ax.set_xlabel('X ($l_{node}$)', color='gray')
    ax.set_ylabel('Y ($l_{node}$)', color='gray')
    ax.set_zlabel('Z ($l_{node}$)', color='gray')
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    ax.tick_params(axis='z', colors='gray')
    
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(False)

    # Legend reflecting strict AVE terminology
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, linestyle='None'),
        Line2D([0], [0], color='cyan', lw=2),
        Line2D([0], [0], color='#ff00ff', lw=2, linestyle=':'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', alpha=0.5, markersize=10, linestyle='None')
    ]
    ax.legend(custom_lines, 
              ['Hardware Node (Dirac Sea Grid)', 
               r'Primary Kinematic Link ($l_{node}$)', 
               'Transverse Cosserat Link (Trace-Reversal)', 
               r'Packing Volume ($\kappa_V = 8\pi\alpha$)'], 
              loc='upper left', facecolor='#111111', edgecolor='white', labelcolor='white')

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    visualize_ave_lattice()