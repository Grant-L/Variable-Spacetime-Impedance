"""
1D Finite-Difference Time-Domain (FDTD) solver for the AVE LC Network Metric.
Strictly replaces continuous structural CFD solvers with discrete Electrodynamic solvers.
"""

import numpy as np
from ave.core.constants import C_0, MU_0, EPSILON_0, Z_0
from ave.mechanics.impedance import calculate_refractive_strain

class FDTDLattice1D:
    def __init__(self, size: int, grid_resolution: float, mass_center_kg: float = 0.0):
        """
        Initializes a 1D discrete LC network transmission line representing the macroscopic vacuum.
        
        Args:
            size (int): Number of nodes in the simulated 1D lattice.
            grid_resolution (float): Physical spacing between discrete nodes (dz).
            mass_center_kg (float): Mass located at the center pulling the metric.
        """
        self.size = size
        self.dz = grid_resolution
        
        # Courant limit for stability (dt <= dz / c)
        self.dt = self.dz / (2.0 * C_0) 
        
        # Grid allocations: Electric Field (E), Magnetic Field (H)
        self.E = np.zeros(size)
        self.H = np.zeros(size - 1)
        
        # Spatially varying macroscopic grid parameters L(z) and C(z)
        self.u_local = np.ones(size - 1) * MU_0
        self.e_local = np.ones(size) * EPSILON_0
        self.n_refractive = np.ones(size)
        
        self._apply_gravitational_metric(mass_center_kg)
        
        # Precompute update coefficients
        self.ce = self.dt / (self.dz * self.e_local)
        
        # H field is staggered, average the adjacent n(r) or define uniquely
        u_centers = (self.u_local)
        self.ch = self.dt / (self.dz * u_centers)

    def _apply_gravitational_metric(self, mass_kg: float):
        """
        Applies topological gravity. 
        Calculates localized optical strain n_r based on mass distance from center.
        Achromatic Impedance Matching explicitly scales BOTH u and e symmetrically!
        """
        if mass_kg <= 0.0:
            return
            
        center_idx = self.size // 2
        
        for k in range(self.size):
            radius = abs(k - center_idx) * self.dz
            if radius > 0.001:  # Avoid singularity precisely at center
                n_r = calculate_refractive_strain(mass_kg, radius)
                # Achromatic scaling: u and e scale directly with n(r)
                # Z_0 remains invariant: sqrt(u'/e') = sqrt(n*u0 / n*e0) = Z_0
                self.e_local[k] = EPSILON_0 * n_r
                self.n_refractive[k] = n_r
                
        for k in range(self.size - 1):
            radius = abs((k + 0.5) - center_idx) * self.dz
            if radius > 0.001:
                n_r = calculate_refractive_strain(mass_kg, radius)
                self.u_local[k] = MU_0 * n_r
                
    def get_local_impedance(self, k: int) -> float:
        """Returns the local transverse impedance at node k."""
        u_eff = self.u_local[min(k, self.size-2)]
        e_eff = self.e_local[k]
        return np.sqrt(u_eff / e_eff)

    def step(self, source_node: int, t_steps: int):
        """
        Advances the FDTD simulation by injecting a continuous sine wave at the source node.
        """
        omega = 2.0 * np.pi * (C_0 / (50.0 * self.dz))
        
        for t in range(t_steps):
            # Update H field (Magnetic)
            self.H[:] += self.ch[:] * (self.E[1:] - self.E[:-1])
            
            # Simple absorbing boundary conditions (ABC) for E field edges
            self.E[0] = self.E[1]
            self.E[-1] = self.E[-2]
            
            # Update E field (Electric) internal nodes
            self.E[1:-1] += self.ce[1:-1] * (self.H[1:] - self.H[:-1])
            
            # Hard source injection
            self.E[source_node] = np.sin(omega * t * self.dt)
