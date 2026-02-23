import numpy as np
from typing import Tuple

class VacuumGrid:
    """
    AVE Unified Python Engine: 'VacuumGrid' Core Object.
    Represents the continuous, mathematically structured LC dielectric vacuum matrix.
    Handles transverse wave propagation (c), characteristic impedance (Z0),
    and fluidic acoustic transmission using FDTD integration.
    """
    
    def __init__(self, nx: int, ny: int, z0: float = 377.0, c2: float = 0.25):
        self.nx = nx
        self.ny = ny
        self.z0 = z0     # Vacuum characteristic macroscopic impedance
        self.c2 = c2     # Courant number (wave speed squared, dt/dx ratio)
        
        # Primary Macroscopic Field Traces (Transverse Displacement / Inductive Shear)
        # Interpreted as 'Displacement' or 'Strain' in AVE Continuum Mechanics.
        self.strain_z = np.zeros((nx, ny))
        
        # Temperature (Background thermal RMS array)
        self.temperature = 0.0

    def set_temperature(self, t: float):
        """Sets the baseline transverse electromagnetic jitter (Heat)."""
        self.temperature = t

    def step_kinematic_wave_equation(self, damping: float = 0.99):
        """Standard Cartesian mechanical wave-equation integration (Laplacian)."""
        new_strain = np.copy(self.strain_z)
        
        # Apply thermal noise if requested
        if self.temperature > 0.0:
            noise = np.random.normal(0, self.temperature * 0.1, (self.nx, self.ny))
            new_strain += noise
            
        # Fast vector Laplacian (excluding boundaries)
        laplacian = (
            self.strain_z[2:, 1:-1] + self.strain_z[:-2, 1:-1] +
            self.strain_z[1:-1, 2:] + self.strain_z[1:-1, :-2] -
            4 * self.strain_z[1:-1, 1:-1]
        )
        
        new_strain[1:-1, 1:-1] = self.strain_z[1:-1, 1:-1] + self.c2 * laplacian
        new_strain *= damping  # Geometric radiation loss
        
        # Fixed borders
        new_strain[0, :] = 0
        new_strain[-1, :] = 0
        new_strain[:, 0] = 0
        new_strain[:, -1] = 0
        
        self.strain_z = new_strain

    def get_local_strain(self, x: int, y: int) -> float:
        """Reads the local FDTD impedance strain at a grid coordinate."""
        if 0 <= x < self.nx and 0 <= y < self.ny:
            return self.strain_z[x, y]
        return 0.0

    def inject_strain(self, x: int, y: int, value: float):
        """Pumps transverse energy directly into the LC matrix."""
        if 0 < x < self.nx - 1 and 0 < y < self.ny - 1:
            self.strain_z[x, y] += value
