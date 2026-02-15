import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from scipy.stats import qmc
import os

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_raw_lattice():
    print("Generating Discrete Amorphous Manifold (Strict Poisson-Disk)...")
    L = 10.0
    # Enforce Hard-Sphere Limit (Prevents l -> 0 singularities)
    engine = qmc.PoissonDisk(d=3, radius=0.6/L, seed=137)
    points = engine.fill_space() * L
    
    tri = Delaunay(points)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], s=10, c='gray', alpha=0.8, label=r'Inductive Nodes ($L_{node}$)')
    
    indptr, indices = tri.vertex_neighbor_vertices
    center = np.array([L/2, L/2, L/2])
    
    for i in range(len(points)):
        if np.linalg.norm(points[i] - center) > 2.5: continue
        for j in indices[indptr[i]:indptr[i+1]]:
            if i < j:
                ax.plot([points[i,0], points[j,0]], [points[i,1], points[j,1]], [points[i,2], points[j,2]], 
                       c='cyan', alpha=0.35, linewidth=0.6)
                
    ax.plot([], [], [], c='cyan', alpha=0.35, linewidth=0.6, label=r'Capacitive Flux Edges ($C_{EM}$)')
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_title("The Amorphous Lattice (No Signal)\nThe Strict Amorphous Solid Hardware", color='white', fontsize=14)
    
    legend = ax.legend(loc='lower right', facecolor='black', edgecolor='white')
    for text in legend.get_texts(): text.set_color('white')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "photon_lattice.png"), dpi=300, facecolor='black', bbox_inches='tight')

if __name__ == "__main__": simulate_raw_lattice()