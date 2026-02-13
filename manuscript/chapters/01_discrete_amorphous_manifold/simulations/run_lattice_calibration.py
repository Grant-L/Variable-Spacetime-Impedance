"""
AVE MODULE 1: LATTICE CALIBRATION
---------------------------------
Derives the Geometric Coarse-Graining Factors (Kappa) directly from
Monte Carlo simulation of a 3D Poisson-Delaunay Mesh.

Hypothesis: Kappa is not 1.0. It is a statistical constant of random packing.
"""

import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import os

# Directory setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Configuration
N_NODES = 5000  # Number of vacuum nodes to simulate
BOX_SIZE = 10.0 # Arbitrary units (scale invariant)

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def generate_amorphous_manifold(n, size):
    """
    Generates a set of stochastic points (Poisson distribution).
    """
    return np.random.rand(n, 3) * size

def analyze_lattice_statistics(points):
    """
    Computes the Delaunay Triangulation (Edges) and Voronoi Tessellation (Volumes).
    Returns the statistical Kappa factors.
    """
    print(f"Triangulating {len(points)} nodes...")
    
    # 1. Delaunay (Connectivity / Inductance)
    delaunay = spatial.Delaunay(points)
    
    # Extract all unique edges to find Mean Lattice Pitch (l0)
    edges = set()
    for simplex in delaunay.simplices:
        for i in range(4):
            for j in range(i+1, 4):
                idx1, idx2 = sorted([simplex[i], simplex[j]])
                edges.add((idx1, idx2))
    
    edge_lengths = []
    for p1_idx, p2_idx in edges:
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        dist = np.linalg.norm(p1 - p2)
        edge_lengths.append(dist)
    
    l0_mean = np.mean(edge_lengths)
    print(f"Mean Lattice Pitch (l0): {l0_mean:.4f}")

    # 2. Voronoi (Capacity / Volume)
    # We define the effective node radius R_eff from the mean Voronoi cell volume.
    # In a hard-sphere packing, this relates to the Wigner-Seitz radius.
    # For a Poisson distribution, V_mean is simply Box_Volume / N.
    
    total_volume = BOX_SIZE**3
    volume_per_node = total_volume / N_NODES
    
    # R_eff = Radius of sphere with equivalent volume
    r_eff_mean = (volume_per_node * 3 / (4 * np.pi)) ** (1/3)
    print(f"Mean Effective Node Radius (R_eff): {r_eff_mean:.4f}")

    # 3. Calculate Kappa (The Geometric Factor)
    # Kappa_geo = R_eff / l0
    kappa_geo = r_eff_mean / l0_mean
    
    return l0_mean, r_eff_mean, kappa_geo, edge_lengths

def plot_lattice_distribution(edge_lengths, l0, kappa):
    plt.figure(figsize=(10, 6))
    plt.hist(edge_lengths, bins=50, color='teal', alpha=0.7, density=True, label="Edge Distribution")
    plt.axvline(l0, color='red', linestyle='--', linewidth=2, label=f"Mean Pitch (l0) = {l0:.3f}")
    
    plt.title(f"Vacuum Lattice Statistics (N={N_NODES})")
    plt.xlabel("Node Separation (arbitrary units)")
    plt.ylabel("Probability Density")
    
    # Annotate Kappa
    plt.text(0.7 * max(edge_lengths), 0.5, 
             f"Derived $\kappa \\approx {kappa:.4f}$", 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.legend()
    plt.grid(True, alpha=0.3)
    
    outfile = os.path.join(SCRIPT_DIR, "lattice_calibration.png")
    plt.savefig(outfile, dpi=300)
    print(f"Saved distribution plot to {outfile}")
    plt.close()

if __name__ == "__main__":
    ensure_output_dir()
    
    # Run Simulation
    points = generate_amorphous_manifold(N_NODES, BOX_SIZE)
    l0, r_eff, kappa, edges = analyze_lattice_statistics(points)
    
    print("-" * 30)
    print(f"LATTICE CALIBRATION RESULTS:")
    print(f"Kappa (Geometric Factor): {kappa:.5f}")
    print("-" * 30)
    
    # Save Kappa to file for other modules to use
    with open(os.path.join(OUTPUT_DIR, "kappa.txt"), "w") as f:
        f.write(str(kappa))
    
    plot_lattice_distribution(edges, l0, kappa)