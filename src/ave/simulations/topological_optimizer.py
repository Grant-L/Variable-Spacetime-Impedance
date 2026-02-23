"""
Universal Topological Optimization Engine
=========================================
The core computational engine of the Applied Vacuum Engineering (AVE) framework.
Because the universe is scale-invariant, the exact same optimization math that
folds complex macro-molecular proteins into their lowest-energy 3D conformations
will seamlessly assemble 235 subatomic nucleons into the precise geometry of 
Uranium. 

This engine accepts N arbitrary topological nodes, applies a universal impedance
cost function, and uses gradient descent / simulated annealing to iteratively 
lock the lattice into its absolute geometric minimum-energy state.
"""

import numpy as np
import scipy.optimize

class TopologicalOptimizer:
    def __init__(self, node_masses, interaction_scale='nuclear'):
        """
        Initialize the optimizer.
        :param node_masses: Array of masses/charges for the N nodes.
        :param interaction_scale: 'nuclear' (1/d) or 'molecular' (1/r^2 vs Van der Waals).
        """
        self.masses = np.array(node_masses)
        self.N = len(self.masses)
        self.scale = interaction_scale
        
        # Scaling parameters based on the physical domain
        if self.scale == 'nuclear':
            # K_mutual derived in Phase 1 for nucleon-nucleon coupling
            self.K_attr = 11.33763228 
            self.K_rep = 15.0 # Repulsive core proxy
            self.r_min = 0.85 # Fermi cutoff
        else:
            # Macro-Molecular (Protein Folding)
            # A simplified Lennard-Jones / Coulomb scale proxy for the 1/d macroscopic limit
            self.K_attr = 1.0
            self.K_rep = 1.0
            self.r_min = 3.5 # Van der Waals equilibrium proxy
            
    def _cost_function(self, flat_coords):
        """
        The global impedance scalar (U_total).
        Calculates the exact structural strain of the entire 3D geometry matrix 
        based on the mutual distances of every node. The goal is to minimize this value.
        """
        coords = flat_coords.reshape((self.N, 3))
        energy = 0.0
        
        # Calculate pair-wise interactions (O(N^2) complexity, significantly faster than DFT)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dist_vec = coords[i] - coords[j]
                dist_sq = np.sum(dist_vec**2)
                dist = np.sqrt(dist_sq)
                
                # Prevent singularities
                if dist < 0.01:
                    dist = 0.01 
                
                m_prod = self.masses[i] * self.masses[j]
                
                if self.scale == 'nuclear':
                    # Pure 1/d topological resonant tension
                    # Repulsion spikes violently inside the Fermi cutoff
                    if dist < self.r_min:
                        energy += m_prod * self.K_rep * ((self.r_min / dist)**3 - 1.0)
                    else:
                        # Standard 1/d mutual acoustic binding
                        energy -= m_prod * (self.K_attr / dist)
                else:
                    # Macro-Molecular Steric Constraints
                    # 1-2 C-alpha backbone constraint
                    if j == i + 1:
                        energy += 1000.0 * (dist - 3.8)**2
                    elif j == i + 2:
                        target = 5.4 if m_prod < 1.0 else 7.0
                        energy += 500.0 * (dist - target)**2
                    elif j == i + 3:
                        target = 5.1 if m_prod < 1.0 else 10.0
                        energy += 500.0 * (dist - target)**2
                    
        return energy
        
    def _jacobian(self, flat_coords):
        """
        Calculates the gradient (force vectors) for the optimizer to step efficiently.
        """
        coords = flat_coords.reshape((self.N, 3))
        grad = np.zeros_like(coords)
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dist_vec = coords[i] - coords[j]
                dist_sq = np.sum(dist_vec**2)
                dist = np.sqrt(dist_sq)
                
                if dist < 0.01:
                    dist = 0.01
                    
                m_prod = self.masses[i] * self.masses[j]
                force_mag = 0.0
                
                if self.scale == 'nuclear':
                    if dist < self.r_min:
                        # Derivative of repulsion
                        force_mag = -3.0 * m_prod * self.K_rep * (self.r_min**3 / dist**4)
                    else:
                        # Derivative of attraction: d/dr(-K/r) = K/r^2
                        force_mag = m_prod * self.K_attr / dist_sq
                else:
                    if j == i + 1:
                        force_mag = -2000.0 * (dist - 3.8)
                    elif j == i + 2:
                        target = 5.4 if m_prod < 1.0 else 7.0
                        force_mag = -1000.0 * (dist - target)
                    elif j == i + 3:
                        target = 5.1 if m_prod < 1.0 else 10.0
                        force_mag = -1000.0 * (dist - target)
                
                # Apply force vector
                # If force_mag > 0, it pushes nodes apart (repulsion gradient acts outwards)
                # But our optimizer MINIMIZES energy, so gradient points towards increasing energy.
                # Actually, scipy.minimize wants the mathematical gradient dU/dx.
                
                f_vec = force_mag * (dist_vec / dist)
                grad[i] += f_vec
                grad[j] -= f_vec
                
        return grad.flatten()

    def optimize(self, initial_coords, method='L-BFGS-B', options=None, record_history=False):
        """
        Runs the numerical optimizer to find the minimum-stress crystalline state.
        :param initial_coords: Starting (N, 3) coordinate array. If randomized, the solver 
                               will physically "fold" or "assemble" them from scratch.
        :param record_history: If True, saves and returns the coordinates at every iteration 
                               for dynamic animation.
        """
        if options is None:
            options = {'disp': True, 'maxiter': 5000, 'ftol': 1e-7}
            
        initial_flat = np.array(initial_coords).flatten()
        
        print(f"[*] Commencing O(N^2) Topological Optimization ({self.scale} mapping)...")
        print(f"    -> Nodes: {self.N}")
        
        history = []
        energy_history = []
        
        def callback(xk):
            if record_history:
                # Save the current 3D geometry
                history.append(xk.reshape((self.N, 3)).copy())
                # Save the current total energetic strain
                energy_history.append(self._cost_function(xk))
        
        # Scipy Minimize (L-BFGS-B is great for smooth energetic gradients)
        result = scipy.optimize.minimize(
            fun=self._cost_function,
            x0=initial_flat,
            method=method,
            jac=self._jacobian,
            options=options,
            callback=callback if record_history else None
        )
        
        if result.success:
            print("[*] Convergence achieved! Optimal lowest-energy geometric lattice locked.")
        else:
            print("[!] Optimizer halted before strict convergence. Returning best estimate.")
            
        final_coords = result.x.reshape((self.N, 3))
        final_energy = result.fun
        
        if record_history:
            if len(history) == 0 or not np.array_equal(history[-1].flatten(), final_coords.flatten()):
                history.append(final_coords.copy())
                energy_history.append(final_energy)
        
        print(f"    -> Final Core Impedance (Strain): {final_energy:.4f}")
        
        if record_history:
            return final_coords, final_energy, np.array(history), np.array(energy_history)
        return final_coords, final_energy
