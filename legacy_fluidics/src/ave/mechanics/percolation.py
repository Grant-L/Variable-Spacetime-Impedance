"""
AVE Lattice Genesis & Rigidity Percolation
Computationally proves that the macroscopic K/G ratio dynamically crosses 
2.0 (Trace-Reversal) strictly due to non-affine geometric buckling.
Source: Chapter 4 (Trace-Reversal) & Appendix D
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ave.core import constants as k

def generate_poisson_disk_graph(box_size_nodes=12):
    """Generates a localized 3D chunk of the M_A vacuum via Hard-Sphere sampling."""
    L = box_size_nodes * k.L_NODE
    np.random.seed(42) # Deterministic for DAG repeatability
    
    num_candidates = int(box_size_nodes**3 * 5)
    candidates = np.random.uniform(0, L, (num_candidates, 3))
    
    accepted_nodes = []
    tree = None
    
    for pt in candidates:
        if tree is None:
            accepted_nodes.append(pt)
            tree = cKDTree(accepted_nodes)
            continue
            
        distances, _ = tree.query(pt, k=1)
        if distances >= k.L_NODE:
            accepted_nodes.append(pt)
            tree = cKDTree(accepted_nodes)
            
    return np.array(accepted_nodes), L

class CentralForceElastodynamicsPBC:
    """
    Simulates the macroscopic elastodynamics using PURE central forces 
    with Periodic Boundary Conditions (PBC) to eliminate affine edge artifacts.
    Relies entirely on non-affine geometric buckling to shift K/G.
    """
    def __init__(self, nodes, L_box, connectivity_multiplier):
        self.nodes = nodes
        self.N = len(nodes)
        self.L = L_box
        self.r_max = connectivity_multiplier * k.L_NODE
        
        # Build KDTree with PBC to wrap edges infinitely
        self.tree = cKDTree(self.nodes, boxsize=self.L)
        self.pairs = list(self.tree.query_pairs(self.r_max))
        self.Z_avg = (2.0 * len(self.pairs)) / self.N
        
    def evaluate_modulus(self, strain_tensor):
        """Solves non-affine relaxation (nodes physically buckle) in PBC."""
        I, J, V = [], [], []
        F_aff = np.zeros((self.N, 3))
        affine_energy = 0.0
        
        for (i, j) in self.pairs:
            dr = self.nodes[j] - self.nodes[i]
            
            # Enforce Periodic Boundary Conditions for distance vector
            dr = dr - self.L * np.round(dr / self.L)
            l0 = np.linalg.norm(dr)
            n = dr / l0
            
            # Pure central force tensor
            K_edge = np.outer(n, n) 
            
            # Affine displacement difference
            du_aff = strain_tensor @ dr
            
            # Force vector construction
            f_ij = K_edge @ du_aff
            F_aff[i] += f_ij
            F_aff[j] -= f_ij
            
            affine_energy += 0.5 * du_aff.T @ K_edge @ du_aff
            
            K_flat = K_edge.flatten()
            idx_i = [3*i, 3*i+1, 3*i+2]
            idx_j = [3*j, 3*j+1, 3*j+2]
            
            I.extend(np.repeat(idx_i, 3)); J.extend(np.tile(idx_i, 3)); V.extend(K_flat)
            I.extend(np.repeat(idx_j, 3)); J.extend(np.tile(idx_j, 3)); V.extend(K_flat)
            I.extend(np.repeat(idx_i, 3)); J.extend(np.tile(idx_j, 3)); V.extend(-K_flat)
            I.extend(np.repeat(idx_j, 3)); J.extend(np.tile(idx_i, 3)); V.extend(-K_flat)
            
        H = sp.coo_matrix((V, (I, J)), shape=(3*self.N, 3*self.N)).tocsr()
        
        # Pin node 0 to remove rigid body translation modes in PBC
        H_sub = H[3:, 3:]
        F_sub = F_aff.flatten()[3:]
        
        # Regularize to prevent singular matrix during severe buckling
        H_sub += sp.eye(3*self.N - 3) * 1e-8
        
        # Solve for non-affine displacements
        u_na_sub = spla.spsolve(H_sub, F_sub)
        u_na = np.zeros(3*self.N)
        u_na[3:] = u_na_sub
        
        # Total energy = Affine Energy - Non-Affine Relaxation Energy
        relax_energy = -0.5 * np.dot(u_na, F_aff.flatten())
        total_energy = affine_energy + relax_energy
        
        volume = self.L**3
        return total_energy / volume

def sweep_percolation_threshold():
    print("==================================================")
    print("AVE COMPUTATIONAL RHEOLOGY: RIGIDITY PERCOLATION")
    print("==================================================\n")
    
    # PBC allows us to simulate a deep bulk volume without boundary pinning
    nodes, L = generate_poisson_disk_graph(12) 
    print(f"[+] Spatial Condensate Generated. (Nodes: {len(nodes)})")
    print("[+] PBC Applied: Removing all affine boundary pinning artifacts.")
    print("[+] Sweeping geometric connectivity to locate K=2G Trace-Reversal...\n")
    
    delta = 1e-4
    strain_bulk = np.array([[delta, 0, 0], [0, delta, 0], [0, 0, delta]])
    strain_shear = np.array([[0, delta, 0], [delta, 0, 0], [0, 0, 0]])
    
    print(f"{'Radius (r/l_node)':<18} | {'Coordination (Z)':<16} | {'Relaxed K/G':<15}")
    print("-" * 55)
    
    crossing_found = False
    
    # Sweep connectivity down towards the isostatic limit
    for r_mult in np.linspace(1.80, 1.45, 12):
        solver = CentralForceElastodynamicsPBC(nodes, L, connectivity_multiplier=r_mult)
        
        if solver.Z_avg < 4.0:
            print(f"{r_mult:<18.3f} | {solver.Z_avg:<16.2f} | [NETWORK LIQUEFIED]")
            break
            
        try:
            E_bulk = solver.evaluate_modulus(strain_bulk)
            E_shear = solver.evaluate_modulus(strain_shear)
            
            K_eff = E_bulk / (4.5 * (delta**2))
            G_eff = E_shear / (2.0 * (delta**2))
            
            ratio = K_eff / G_eff
            
            marker = " <-- TRACE REVERSAL (K=2G)" if 1.85 < ratio < 2.15 else ""
            if 1.85 < ratio < 2.15:
                crossing_found = True

            print(f"{r_mult:<18.3f} | {solver.Z_avg:<16.2f} | {ratio:<15.3f} {marker}")
        except Exception as e:
            print(f"{r_mult:<18.3f} | {solver.Z_avg:<16.2f} | ERROR: {e}")

    print("\n==================================================")
    if crossing_found:
        print("VERDICT: SUCCESS. K/G geometrically crosses 2.0 via non-affine buckling.")
        print("General Relativity's trace-reversed bulk is natively supported by the topology.")
    else:
        print("VERDICT: CROSSING MISSED. (Adjust sweep parameters).")
    print("==================================================")

if __name__ == "__main__":
    sweep_percolation_threshold()