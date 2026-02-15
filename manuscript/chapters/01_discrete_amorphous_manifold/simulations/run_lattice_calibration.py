"""
AVE MODULE 1: LATTICE CALIBRATION (Strict)
---------------------------------
Derives the Volumetric Coarse-Graining Factor (\kappa_V) directly from
a 3D Poisson-Disk (Amorphous Solid) Mesh, rigorously filtering boundary artifacts.
"""

import numpy as np
import scipy.spatial as spatial
from scipy.stats import qmc
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

BOX_SIZE = 10.0
# Enforces fundamental minimum exclusion distance (prevents l -> 0 singularities)
MIN_DIST = 0.6 

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def generate_amorphous_manifold(size, min_dist):
    print(f"Generating Amorphous Manifold (Poisson-Disk radius={min_dist})...")
    # qmc.PoissonDisk packs nodes dynamically based on the radius limit
    engine = qmc.PoissonDisk(d=3, radius=min_dist/size, seed=42)
    points = engine.fill_space() * size
    return points

def analyze_lattice_statistics(points):
    print(f"Triangulating {len(points)} nodes...")
    delaunay = spatial.Delaunay(points)
    voronoi = spatial.Voronoi(points)
    
    # Identify boundary nodes to strictly filter out convex hull artifacts
    hull = spatial.ConvexHull(points)
    boundary_nodes = set(hull.vertices)
    
    # 1. Extract true bulk edge lengths (l_node)
    edges = set()
    for simplex in delaunay.simplices:
        for i in range(4):
            for j in range(i+1, 4):
                idx1, idx2 = sorted([simplex[i], simplex[j]])
                # Only include edge if BOTH nodes are in the interior bulk
                if idx1 not in boundary_nodes and idx2 not in boundary_nodes:
                    edges.add((idx1, idx2))
                    
    edge_lengths = [np.linalg.norm(points[i] - points[j]) for i, j in edges]
    l0_mean = np.mean(edge_lengths)
    print(f"Mean Bulk Lattice Pitch (l_node): {l0_mean:.4f}")

    # 2. Extract true bulk Voronoi cell volumes (V_node)
    bulk_volumes = []
    for i, region_idx in enumerate(voronoi.point_region):
        if i not in boundary_nodes:
            region = voronoi.regions[region_idx]
            # Ensure it is a closed, valid geometric region
            if -1 not in region and len(region) > 0:
                poly = voronoi.vertices[region]
                try:
                    bulk_volumes.append(spatial.ConvexHull(poly).volume)
                except Exception:
                    pass
                    
    v_mean = np.mean(bulk_volumes)
    print(f"Mean Bulk Nodal Volume (V_node): {v_mean:.4f}")

    # 3. Calculate Kappa exactly as defined in the VSI manuscript
    kappa = v_mean / (l0_mean**3)
    
    return l0_mean, v_mean, kappa, edge_lengths

def plot_lattice_distribution(edge_lengths, l0, kappa):
    plt.figure(figsize=(10, 6))
    plt.hist(edge_lengths, bins=40, color='teal', alpha=0.7, density=True, label="Bulk Edge Distribution")
    plt.axvline(l0, color='red', linestyle='--', linewidth=2, label=f"Mean Pitch ($l_{{node}}$) = {l0:.3f}")
    
    plt.title("Vacuum Lattice Topological Statistics (Amorphous Solid Limit)")
    plt.xlabel("Node Separation (arbitrary length units)")
    plt.ylabel("Probability Density")
    
    # Annotate strictly using the volumetric definition
    plt.text(0.65 * max(edge_lengths), plt.gca().get_ylim()[1] * 0.5, 
             f"Derived $\kappa_V \equiv V_{{node}}/l_{{node}}^3 \\approx {kappa:.4f}$", 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

    plt.legend()
    plt.grid(True, alpha=0.3)
    
    outfile = os.path.join(OUTPUT_DIR, "lattice_calibration.png")
    plt.savefig(outfile, dpi=300)
    print(f"Saved strictly calibrated distribution plot to {outfile}")
    plt.close()

if __name__ == "__main__":
    ensure_output_dir()
    points = generate_amorphous_manifold(BOX_SIZE, MIN_DIST)
    l0, v_mean, kappa, edges = analyze_lattice_statistics(points)
    
    print("-" * 40)
    print(f"VSI LATTICE CALIBRATION RESULTS:")
    print(f"Kappa (Volumetric Geometric Factor): {kappa:.5f}")
    print("-" * 40)
    
    with open(os.path.join(OUTPUT_DIR, "kappa.txt"), "w") as f:
        f.write(str(kappa))
        
    plot_lattice_distribution(edges, l0, kappa)