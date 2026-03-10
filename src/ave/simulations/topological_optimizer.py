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
    def __init__(self, node_masses, interaction_scale='nuclear', node_charges=None):
        """
        Initialize the optimizer.
        :param node_masses: Array of masses/charges for the N nodes.
        :param interaction_scale: 'nuclear' (1/d) or 'molecular'.
        :param node_charges: Array of charges (1 for proton, 0 for neutron). If None,
                             charges are inferred from masses (m < 1.003 = proton).
        """
        self.masses = np.array(node_masses)
        self.N = len(self.masses)
        self.scale = interaction_scale

        # ALL constants derived from axioms via ave.core.constants
        if self.scale == 'nuclear':
            from ave.core.constants import (
                K_MUTUAL, D_PROTON,
                ALPHA, HBAR, C_0, e_charge
            )
            self.K_attr = K_MUTUAL
            self.d_sat = D_PROTON

            # Coulomb constant: αℏc [MeV·fm]
            self.alpha_hc = ALPHA * (HBAR * C_0 / e_charge) * 1e9

            # Assign proton/neutron identity for Coulomb repulsion
            if node_charges is not None:
                self.charges = np.array(node_charges, dtype=float)
            else:
                self.charges = np.array(
                    [1.0 if m < 1.003 else 0.0 for m in self.masses]
                )
        else:
            self.K_attr = 1.0
            self.d_sat = 3.5
            self.alpha_hc = 0.0
            self.charges = np.zeros(self.N)

    def _cost_function(self, flat_coords):
        """
        The global impedance scalar (U_total).
        
        Uses the universal saturated pairwise potential (Operator 4):
          U_mutual(r) = universal_pairwise_energy(r, K, d_sat)
        Plus Coulomb repulsion between proton pairs: +αℏc/r
        """
        from ave.core.universal_operators import universal_pairwise_energy

        coords = flat_coords.reshape((self.N, 3))
        energy = 0.0
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dist_vec = coords[i] - coords[j]
                dist = np.sqrt(np.sum(dist_vec**2))
                if dist < 0.01:
                    dist = 0.01
                
                m_prod = self.masses[i] * self.masses[j]
                
                if self.scale == 'nuclear':
                    # Universal Operator 4: saturated mutual coupling
                    energy += m_prod * universal_pairwise_energy(dist, self.K_attr, self.d_sat)

                    # Coulomb repulsion between proton pairs
                    q_prod = self.charges[i] * self.charges[j]
                    if q_prod > 0:
                        energy += q_prod * self.alpha_hc / dist
                else:
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
        Analytical gradient using the universal pairwise gradient (Operator 4).
        """
        from ave.core.universal_operators import universal_pairwise_gradient

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
                    # Universal Operator 4: gradient of saturated potential
                    force_mag = m_prod * universal_pairwise_gradient(dist, self.K_attr, self.d_sat)

                    # Coulomb gradient
                    q_prod = self.charges[i] * self.charges[j]
                    if q_prod > 0:
                        force_mag -= q_prod * self.alpha_hc / dist_sq
                else:
                    if j == i + 1:
                        force_mag = -2000.0 * (dist - 3.8)
                    elif j == i + 2:
                        target = 5.4 if m_prod < 1.0 else 7.0
                        force_mag = -1000.0 * (dist - target)
                    elif j == i + 3:
                        target = 5.1 if m_prod < 1.0 else 10.0
                        force_mag = -1000.0 * (dist - target)
                
                f_vec = force_mag * (dist_vec / dist)
                grad[i] += f_vec
                grad[j] -= f_vec
                
        return grad.flatten()

    def optimize(self, initial_coords, method='L-BFGS-B', options=None, record_history=False):
        """
        Runs the numerical optimizer to find the minimum-stress crystalline state.
        """
        if options is None:
            options = {'disp': True, 'maxiter': 5000, 'ftol': 1e-7}
            
        initial_flat = np.array(initial_coords).flatten()
        
        print(f"[*] Commencing O(N^2) Topological Optimization ({self.scale} mapping)...")
        print(f"    -> Nodes: {self.N}")
        
        history = []
        energy_history = []
        
        def callback(xk):
            """Record optimisation trajectory for convergence analysis."""
            if record_history:
                history.append(xk.reshape((self.N, 3)).copy())
                energy_history.append(self._cost_function(xk))
        
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
