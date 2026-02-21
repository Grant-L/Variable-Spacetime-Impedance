"""
AVE Bingham-Plastic CFD Solver
Implements the core time-integration for the macroscopic physical vacuum.
Dynamically tracks the K=2G yield rupture and viscosity collapse (Superfluid Slipstream).
"""
import numpy as np
from ave.solvers.grid_3d import EulerianGrid3D
from ave.mechanics.moduli import calculate_bingham_yield_stress, calculate_kinematic_viscosity

class BinghamFluidSolver:
    def __init__(self, grid: EulerianGrid3D):
        self.g = grid
        
        # Core Rheology limits derived strictly from geometry
        self.tau_yield = calculate_bingham_yield_stress()
        self.nu_solid = calculate_kinematic_viscosity()
        
        # In the Bingham avalanche, viscosity decays exponentially as yielding occurs
        # A fully sheared vacuum (warp bubble) drops Nu effectively to zero
        self.nu_field = np.full((self.g.Nx, self.g.Ny, self.g.Nz), self.nu_solid)

    def compute_strain_rate_tensor(self):
        """Calculates the local velocity gradients across the 3D grid."""
        # Central differences for first derivatives
        dvx_dx = np.gradient(self.g.vx, self.g.dx, axis=0)
        dvy_dy = np.gradient(self.g.vy, self.g.dy, axis=1)
        dvz_dz = np.zeros_like(self.g.vx) if self.g.Nz == 1 else np.gradient(self.g.vz, self.g.dz, axis=2)
        
        dvx_dy = np.gradient(self.g.vx, self.g.dy, axis=1)
        dvy_dx = np.gradient(self.g.vy, self.g.dx, axis=0)
        
        dvx_dz = np.zeros_like(self.g.vx) if self.g.Nz == 1 else np.gradient(self.g.vx, self.g.dz, axis=2)
        dvz_dx = np.zeros_like(self.g.vx) if self.g.Nz == 1 else np.gradient(self.g.vz, self.g.dx, axis=0)
        
        dvy_dz = np.zeros_like(self.g.vx) if self.g.Nz == 1 else np.gradient(self.g.vy, self.g.dz, axis=2)
        dvz_dy = np.zeros_like(self.g.vx) if self.g.Nz == 1 else np.gradient(self.g.vz, self.g.dy, axis=1)
        
        # Construct symmetric rate-of-strain tensor elements (gamma_dot)
        g_xx = dvx_dx
        g_yy = dvy_dy
        g_zz = dvz_dz
        g_xy = 0.5 * (dvx_dy + dvy_dx)
        g_xz = 0.5 * (dvx_dz + dvz_dx)
        g_yz = 0.5 * (dvy_dz + dvz_dy)
        
        # Second invariant of the strain rate tensor (magnitude of shear)
        # gamma_dot_II = sqrt(2 * I_2)
        gamma_dot_mag = np.sqrt(2.0 * (g_xx**2 + g_yy**2 + g_zz**2 + 2*g_xy**2 + 2*g_xz**2 + 2*g_yz**2))
        return gamma_dot_mag

    def update_viscosity_field(self):
        """
        The Core Mechanism: The Bingham Phase Transition.
        If local shear stress tau > 43.65 keV (equivalent yield tension),
        the topology fractures and the space locally liquifies into a perfect fluid.
        """
        gamma_dot = self.compute_strain_rate_tensor()
        
        # Base Bingham model implies tau = tau_y + mu_p * gamma_dot if yielded
        # As an effective field theory, we compute the apparent viscosity smoothly.
        # Below yield, the viscosity is infinitely high (Macroscopic solid/Dark Matter).
        
        # To make it numerically stable, we use the Papanastasiou modification
        # mu_eff(gamma_dot) = mu_solid + (tau_y / gamma_dot) * (1 - exp(-m * gamma_dot))
        m_regulizer = 100.0 # Sharpness of the solid-to-fluid phase transition
        
        # Prevent division by zero
        gamma_safe = np.where(gamma_dot == 0, 1e-12, gamma_dot)
        
        # Compute dynamic kinematic viscosity (nu_eff = mu_eff / rho)
        dynamic_mu = self.nu_solid * self.g.rho + (self.tau_yield / gamma_safe) * (1.0 - np.exp(-m_regulizer * gamma_safe))
        
        # High shear fundamentally shatters the lattice, driving nu_eff -> 0
        self.nu_field = dynamic_mu / self.g.rho
        
        # Clip the absolute minimum to avoid numerical exploding gradients
        # The ultimate lower bound is standard quantum fluid turbulence limits
        self.nu_field = np.clip(self.nu_field, 1e-10, self.nu_solid * 1e3)

    def apply_momentum_diffusion(self):
        """Applies the macroscopic diffusion term of the Navier-Stokes eq: nu * Del^2(v)"""
        # Laplacian of velocity
        lap_vx = (np.roll(self.g.vx, 1, axis=0) + np.roll(self.g.vx, -1, axis=0) - 2*self.g.vx) / self.g.dx**2 + \
                 (np.roll(self.g.vx, 1, axis=1) + np.roll(self.g.vx, -1, axis=1) - 2*self.g.vx) / self.g.dy**2
        if self.g.Nz > 1:
            lap_vx += (np.roll(self.g.vx, 1, axis=2) + np.roll(self.g.vx, -1, axis=2) - 2*self.g.vx) / self.g.dz**2
                 
        lap_vy = (np.roll(self.g.vy, 1, axis=0) + np.roll(self.g.vy, -1, axis=0) - 2*self.g.vy) / self.g.dx**2 + \
                 (np.roll(self.g.vy, 1, axis=1) + np.roll(self.g.vy, -1, axis=1) - 2*self.g.vy) / self.g.dy**2
        if self.g.Nz > 1:
            lap_vy += (np.roll(self.g.vy, 1, axis=2) + np.roll(self.g.vy, -1, axis=2) - 2*self.g.vy) / self.g.dz**2
                 
        lap_vz = (np.roll(self.g.vz, 1, axis=0) + np.roll(self.g.vz, -1, axis=0) - 2*self.g.vz) / self.g.dx**2 + \
                 (np.roll(self.g.vz, 1, axis=1) + np.roll(self.g.vz, -1, axis=1) - 2*self.g.vz) / self.g.dy**2
        if self.g.Nz > 1:
            lap_vz += (np.roll(self.g.vz, 1, axis=2) + np.roll(self.g.vz, -1, axis=2) - 2*self.g.vz) / self.g.dz**2
                 
        self.g.vx += self.nu_field * lap_vx * self.g.dt
        self.g.vy += self.nu_field * lap_vy * self.g.dt
        self.g.vz += self.nu_field * lap_vz * self.g.dt

    def step(self):
        """Advances the PDE one discrete time interval based on strict topology."""
        self.update_viscosity_field()
        self.apply_momentum_diffusion()
        self.g.apply_boundary_conditions()
