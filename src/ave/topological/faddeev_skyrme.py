"""
Faddeev-Skyrme Hamiltonian Solver for the AVE Topological Network.
Solves for the 1D scalar rest-mass minimum of the structural metric defect.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

class TopologicalHamiltonian1D:
    def __init__(self, node_pitch: float, scaling_coupling: float = 1.0):
        """
        Initializes the 1D solver for the localized non-linear phase defect.
        
        Args:
            node_pitch (float): Fundamental structural spacing (Axiom 1).
            scaling_coupling (float): The generalized Faddeev coupling constant.
        """
        self.l_node = node_pitch
        self.kappa = scaling_coupling
        
    def _phase_profile(self, r: float, r_opt: float, n: float) -> float:
        """
        Standard 1D topological profile interpolating smoothly between:
        phi(0) = pi (inverted core phase)
        phi(inf) = 0 (relaxed unbroken vacuum)
        """
        if r == 0:
            return np.pi
            
        scaled_r = r / r_opt
        
        # Power-law bounded profile matching standard topological ansatz
        return np.pi / (1.0 + (scaled_r)**n)

    def _energy_density_integrand(self, r: float, r_opt: float, n: float) -> float:
        """
        Evaluates the local energy density of the Faddeev-Skyrme functional 
        at a specific radius r. 
        Note: The true 3D tensor trace uses external geometric bounding from `tensors.py`.
        Here we strictly evaluate the localized 1D radial scalar component.
        """
        # Finite difference approx for derivative (d_phi/d_r)
        dr = 1e-5
        phi1 = self._phase_profile(r, r_opt, n)
        phi2 = self._phase_profile(r + dr, r_opt, n)
        dphi_dr = (phi2 - phi1) / dr
        
        # Quadratic stiffness term (Standard Dirichlet tension)
        kinetic_term = 0.5 * (dphi_dr**2)
        
        # Quartic stabilization term (Skyrme/Faddeev Tensor repulsion)
        # Prevents the defect from collapsing to a singularity
        # In 1D radial projection, sin^2(phi)/r^2 dominates
        skyrme_term = 0.5 * (np.sin(phi1)**2) / (r**2 + 1e-10)
        
        # Total density scaled spherically
        density = 4 * np.pi * (r**2) * (kinetic_term + (self.kappa**2) * skyrme_term * dphi_dr**2)
        
        return density

    def solve_scalar_trace(self) -> float:
        """
        Minimizes the 1D topological Hamiltonian to find the absolute lowest 
        energy stable profile of the fundamental defect.
        
        Returns:
            float: The integrated energy eigenvalue in dimensionless mass units (1162 approx expected).
        """
        
        def objective(params):
            r_opt, n = params
            # Integrate the energy density from core out to 10 * r_opt
            integral, _ = quad(self._energy_density_integrand, 0.0, 10.0 * r_opt, args=(r_opt, n), limit=50)
            return integral
            
        # Initial guesses: optimal radius roughly 1.0, power profile n=2
        initial_guess = [1.0, 2.0]
        
        # Bound the radius strictly > 0 and n > 0
        bounds = [(0.1, 5.0), (1.0, 4.0)]
        
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        # Return the minimized dimensionless energy scalar
        return result.fun
