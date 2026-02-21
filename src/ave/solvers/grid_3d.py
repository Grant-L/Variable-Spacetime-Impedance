"""
AVE 3D Eulerian Macroscopic Grid
Provides the continuous grid for tracking fluid stress, velocity, and density.
Hooks directly into the discrete constants of the `ave.core` module.
"""
import numpy as np
from ave.core import constants as k

class EulerianGrid3D:
    def __init__(self, physical_size_meters, resolution_nodes, dt=None):
        """
        Initializes a uniform 3D grid.
        :param physical_size_meters: Tuple of (Lx, Ly, Lz) in meters
        :param resolution_nodes: Tuple of (Nx, Ny, Nz) grid points
        :param dt: Timestep (dx / c) for relativistic causality
        """
        self.Lx, self.Ly, self.Lz = physical_size_meters
        self.Nx, self.Ny, self.Nz = resolution_nodes
        
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        
        # Courant-Friedrichs-Lewy (CFL) absolute causal time step
        # Ensures signal propagation v never exceeds c
        if dt is None:
            self.dt = min(self.dx, self.dy, self.dz) / (2.0 * k.C) 
        else:
            self.dt = dt
            
        # The macroscopic bulk density (approx 7.9 x 10^6 kg/m^3)
        # Sourced rigidly from Axioms 1 & 2
        from ave.mechanics.moduli import calculate_bulk_density
        self.rho = calculate_bulk_density()
        
        # Grid allocations (using float64 for high dynamic range)
        # Pressure/Density Field (Scalar)
        self.p = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.float64)
        
        # Velocity Field (Vector)
        self.vx = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.float64)
        self.vy = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.float64)
        self.vz = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.float64)
        
        # Absolute Kinematic Phase Phase (A_mu)
        self.A_phase = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.float64)
        
        # Isotropic Kinematic Viscosity term (nu = alpha * c * l_node)
        from ave.mechanics.moduli import calculate_kinematic_viscosity
        self.nu_baseline = calculate_kinematic_viscosity()

    def get_mesh(self):
        """Returns the X, Y, Z coordinate matrices for easy vector plotting."""
        x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)
        y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)
        z = np.linspace(-self.Lz/2, self.Lz/2, self.Nz)
        return np.meshgrid(x, y, z, indexing='ij')

    def apply_boundary_conditions(self):
        """Enforces completely open radiating boundary conditions (zero gradient edges)."""
        # X boundaries
        self.vx[0, :, :] = self.vx[1, :, :]
        self.vx[-1, :, :] = self.vx[-2, :, :]
        self.p[0, :, :] = self.p[1, :, :]
        self.p[-1, :, :] = self.p[-2, :, :]
        
        # Y boundaries
        self.vy[:, 0, :] = self.vy[:, 1, :]
        self.vy[:, -1, :] = self.vy[:, -2, :]
        self.p[:, 0, :] = self.p[:, 1, :]
        self.p[:, -1, :] = self.p[:, -2, :]
        
        # Z boundaries
        if self.Nz > 1:
            self.vz[:, :, 0] = self.vz[:, :, 1]
            self.vz[:, :, -1] = self.vz[:, :, -2]
            self.p[:, :, 0] = self.p[:, :, 1]
            self.p[:, :, -1] = self.p[:, :, -2]
