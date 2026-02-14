import numpy as np
from scipy.spatial import Delaunay, Voronoi

class AmorphousManifold:
    """
    The fundamental hardware substrate of the AVE theory.
    Represents a Discrete Amorphous Manifold (MA) via Poisson-Delaunay triangulation.
    """
    def __init__(self, n_nodes, box_size=10.0, seed=None):
        self.n_nodes = n_nodes
        self.box_size = box_size
        self.seed = seed
        self.points = None
        self.delaunay = None
        self.voronoi = None
        self.kappa = None
        
        self._genesis()

    def _genesis(self):
        """Initializes the lattice nodes (Crystallization Phase)."""
        if self.seed:
            np.random.seed(self.seed)
        self.points = np.random.rand(self.n_nodes, 3) * self.box_size
        
        # Hardware Connectivity (Flux Tubes)
        self.delaunay = Delaunay(self.points)
        # Metric Volume (Voronoi Cells)
        self.voronoi = Voronoi(self.points)

    def calculate_kappa(self):
        """
        Derives the Geometric Packing Factor (Kappa) for this specific universe instance.
        Kappa = Mean_Effective_Radius / Mean_Lattice_Pitch
        """
        # 1. Calculate Mean Lattice Pitch (l0)
        edges = set()
        for simplex in self.delaunay.simplices:
            for i in range(4):
                for j in range(i+1, 4):
                    # Sort to ensure unique edge ID
                    p1, p2 = sorted([simplex[i], simplex[j]])
                    edges.add((p1, p2))
        
        lengths = []
        for p1_idx, p2_idx in edges:
            dist = np.linalg.norm(self.points[p1_idx] - self.points[p2_idx])
            lengths.append(dist)
        l0_mean = np.mean(lengths)

        # 2. Calculate Mean Effective Radius (R_eff)
        # Approximation: Volume per node = Box_Volume / N
        vol_per_node = (self.box_size ** 3) / self.n_nodes
        r_eff_mean = (vol_per_node * 3 / (4 * np.pi)) ** (1/3)

        # 3. Derive Kappa
        self.kappa = r_eff_mean / l0_mean
        return self.kappa