"""
Core physical primitives and invariant constants for the
Applied Vacuum Electrodynamics (AVE) Framework.
"""

import numpy as np
from math import pi

# ---------------------------------------------------------
# Electromagnetic Constants derived from standard physics
# ---------------------------------------------------------
C_0 = 299792458.0                       # Speed of light in vacuum [m/s]
MU_0 = 1.25663706e-06                   # Vacuum permeability (4*pi*10^-7) [H/m]
EPSILON_0 = 8.85418782e-12              # Vacuum permittivity (1/(mu_0*c^2)) [F/m]
Z_0 = np.sqrt(MU_0 / EPSILON_0)         # Characteristic impedance [Ohms] (approx 376.73)
HBAR = 1.054571817e-34                  # Planck constant over 2 pi [J s]
M_E = 9.1093837e-31                     # Electron invariant mass [kg]

# ---------------------------------------------------------
# AVE Specific Topological Constants
# ---------------------------------------------------------
G = 6.67430e-11                         # Newtonian Gravitational Constant [m^3/(kg*s^2)]
ALPHA = 7.29735256e-3                   # Fine-structure constant (Porosity factor)

# Topological Conversion Constant (xi_topo)
# Maps Electrical Impedance to Mechanical Acoustic Impedance
# Estimated here abstractly. In a full implementation, this binds e/l_node
# For generic simulations, we assume unity dimensionality scaling factor
XI_TOPO = 1.0                           

# ---------------------------------------------------------
# Macroscopic Network Bounds
# ---------------------------------------------------------
# Isotropic Strain Projection (derived from 6^3_2 Borromean kinematics)
ISOTROPIC_PROJECTION = 1.0 / 7.0

# Absolute Impedance Rupture Limit (Gamma = -1)
# The boundary where local dielectric strain equals or exceeds 1.0
DIELECTRIC_RUPTURE_STRAIN = 1.0         # Dimensionless unit strain
