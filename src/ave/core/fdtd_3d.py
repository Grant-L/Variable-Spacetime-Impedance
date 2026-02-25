"""
3D Finite-Difference Time-Domain (FDTD) Maxwell Solver Engine
=============================================================

This module provides a rigorous, time-evolved 3D Maxwell equation solver 
utilizing the standard Yee-cell grid architecture. It explicitly calculates 
the propagation of Electric (E) and Magnetic (H) fields through a defined 
volumetric mesh.

It includes 1st-order Mur Absorbing Boundary Conditions (ABCs) to prevent 
spurious un-physical reflections off the computational domain walls, 
ensuring expanding wavefronts (such as dipole radiation and phased arrays) 
decay naturally into the surrounding vacuum.
"""

import numpy as np

# Bind core AVE constants
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.ave.core.constants import C_0, MU_0, EPSILON_0

class FDTD3DEngine:
    def __init__(self, nx: int, ny: int, nz: int, dx: float=0.01):
        """
        Initialize the 3D Cartesian FDTD grid.
        
        Args:
            nx, ny, nz: Number of spatial grid cells in each dimension.
            dx: Spatial step size (meters) per cell. Assuming uniform cubic cells.
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        
        # Physical Constants from AVE Core
        self.c = float(C_0)
        self.mu_0 = float(MU_0)
        self.epsilon_0 = float(EPSILON_0)
        
        # Courant-Friedrichs-Lewy (CFL) Condition for 3D stability
        # dt <= dx / (c * sqrt(3))
        self.dt = self.dx / (self.c * np.sqrt(3.0))
        
        # Core Field Matrices (E and H vectors)
        self.Ex = np.zeros((nx, ny, nz))
        self.Ey = np.zeros((nx, ny, nz))
        self.Ez = np.zeros((nx, ny, nz))
        
        self.Hx = np.zeros((nx, ny, nz))
        self.Hy = np.zeros((nx, ny, nz))
        self.Hz = np.zeros((nx, ny, nz))
        
        # Update coefficients (assuming vacuum for simplicity initially)
        self.ce = self.dt / (self.epsilon_0 * self.dx)
        self.ch = self.dt / (self.mu_0 * self.dx)
        
        # Mur 1st-Order ABC boundary condition memory vectors
        # Required to calculate the spatial derivative strictly adjacent to the wall
        abc_coef = (self.c * self.dt - self.dx) / (self.c * self.dt + self.dx)
        self.abc_coef = abc_coef
        
        self.ex_y0 = np.zeros((nx, nz)); self.ex_yn = np.zeros((nx, nz))
        self.ex_z0 = np.zeros((nx, ny)); self.ex_zn = np.zeros((nx, ny))
        
        self.ey_x0 = np.zeros((ny, nz)); self.ey_xn = np.zeros((ny, nz))
        self.ey_z0 = np.zeros((nx, ny)); self.ey_zn = np.zeros((nx, ny))
        
        self.ez_x0 = np.zeros((ny, nz)); self.ez_xn = np.zeros((ny, nz))
        self.ez_y0 = np.zeros((nx, nz)); self.ez_yn = np.zeros((nx, nz))

    def update_magnetic_field(self):
        """Update H fields from the curl of E according to Faraday's Law."""
        # \vec{H}^{n+1/2} = \vec{H}^{n-1/2} - (dt / mu) \nabla \times \vec{E}^n
        
        # Hx
        self.Hx[:, :-1, :-1] -= self.ch * (
            (self.Ez[:, 1:, :-1] - self.Ez[:, :-1, :-1]) - 
            (self.Ey[:, :-1, 1:] - self.Ey[:, :-1, :-1])
        )
        
        # Hy
        self.Hy[:-1, :, :-1] -= self.ch * (
            (self.Ex[:-1, :, 1:] - self.Ex[:-1, :, :-1]) - 
            (self.Ez[1:, :, :-1] - self.Ez[:-1, :, :-1])
        )
        
        # Hz
        self.Hz[:-1, :-1, :] -= self.ch * (
            (self.Ey[1:, :-1, :] - self.Ey[:-1, :-1, :]) - 
            (self.Ex[:-1, 1:, :] - self.Ex[:-1, :-1, :])
        )

    def update_electric_field(self):
        """Update E fields from the curl of H according to Ampere's Law."""
        # \vec{E}^{n+1} = \vec{E}^{n} + (dt / epsilon) \nabla \times \vec{H}^{n+1/2}
        
        # Ex
        self.Ex[:, 1:, 1:] += self.ce * (
            (self.Hz[:, 1:, 1:] - self.Hz[:, :-1, 1:]) - 
            (self.Hy[:, 1:, 1:] - self.Hy[:, 1:, :-1])
        )
        
        # Ey
        self.Ey[1:, :, 1:] += self.ce * (
            (self.Hx[1:, :, 1:] - self.Hx[1:, :, :-1]) - 
            (self.Hz[1:, :, 1:] - self.Hz[:-1, :, 1:])
        )
        
        # Ez
        self.Ez[1:, 1:, :] += self.ce * (
            (self.Hy[1:, 1:, :] - self.Hy[:-1, 1:, :]) - 
            (self.Hx[1:, 1:, :] - self.Hx[1:, :-1, :])
        )

    def apply_mur_abc(self):
        """
        Apply 1st-Order Mur Absorbing Boundary Conditions (ABC) to grid edges.
        This forces outer tangential E-field components to radiate outward at c,
        preventing reflections from the hard boundary constraints.
        """
        c1 = self.abc_coef
        
        # X-Boundaries (x=0, x=nx-1) for tangential fields Ey, Ez
        # x = 0
        self.Ey[0, :, :] = self.ey_x0 + c1 * (self.Ey[1, :, :] - self.Ey[0, :, :])
        self.ey_x0[:,:] = self.Ey[1, :, :] # update memory
        self.Ez[0, :, :] = self.ez_x0 + c1 * (self.Ez[1, :, :] - self.Ez[0, :, :])
        self.ez_x0[:,:] = self.Ez[1, :, :]
        # x = nx-1
        self.Ey[-1, :, :] = self.ey_xn + c1 * (self.Ey[-2, :, :] - self.Ey[-1, :, :])
        self.ey_xn[:,:] = self.Ey[-2, :, :]
        self.Ez[-1, :, :] = self.ez_xn + c1 * (self.Ez[-2, :, :] - self.Ez[-1, :, :])
        self.ez_xn[:,:] = self.Ez[-2, :, :]

        # Y-Boundaries (y=0, y=ny-1) for tangential fields Ex, Ez
        # y = 0
        self.Ex[:, 0, :] = self.ex_y0 + c1 * (self.Ex[:, 1, :] - self.Ex[:, 0, :])
        self.ex_y0[:,:] = self.Ex[:, 1, :]
        self.Ez[:, 0, :] = self.ez_y0 + c1 * (self.Ez[:, 1, :] - self.Ez[:, 0, :])
        self.ez_y0[:,:] = self.Ez[:, 1, :]
        # y = ny-1
        self.Ex[:, -1, :] = self.ex_yn + c1 * (self.Ex[:, -2, :] - self.Ex[:, -1, :])
        self.ex_yn[:,:] = self.Ex[:, -2, :]
        self.Ez[:, -1, :] = self.ez_yn + c1 * (self.Ez[:, -2, :] - self.Ez[:, -1, :])
        self.ez_yn[:,:] = self.Ez[:, -2, :]

        # Z-Boundaries (z=0, z=nz-1) for tangential fields Ex, Ey
        # z = 0
        self.Ex[:, :, 0] = self.ex_z0 + c1 * (self.Ex[:, :, 1] - self.Ex[:, :, 0])
        self.ex_z0[:,:] = self.Ex[:, :, 1]
        self.Ey[:, :, 0] = self.ey_z0 + c1 * (self.Ey[:, :, 1] - self.Ey[:, :, 0])
        self.ey_z0[:,:] = self.Ey[:, :, 1]
        # z = nz-1
        self.Ex[:, :, -1] = self.ex_zn + c1 * (self.Ex[:, :, -2] - self.Ex[:, :, -1])
        self.ex_zn[:,:] = self.Ex[:, :, -2]
        self.Ey[:, :, -1] = self.ey_zn + c1 * (self.Ey[:, :, -2] - self.Ey[:, :, -1])
        self.ey_zn[:,:] = self.Ey[:, :, -2]

    def inject_soft_source(self, field: str, x: int, y: int, z: int, amplitude: float):
        """
        Inject a soft source (additive current equivalent) into a specific field component.
        A soft source is physically accurate since the node remains transparent to passing waves.
        """
        if field == 'Ex': self.Ex[x, y, z] += amplitude
        elif field == 'Ey': self.Ey[x, y, z] += amplitude
        elif field == 'Ez': self.Ez[x, y, z] += amplitude

    def step(self):
        """Execute one complete dt timestep of the Maxwell Yee-cell algorithm."""
        # 1. Update H from old E
        self.update_magnetic_field()
        # 2. Update E from new H
        self.update_electric_field()
        # 3. Apply Absorbing Boundary Conditions
        self.apply_mur_abc()
