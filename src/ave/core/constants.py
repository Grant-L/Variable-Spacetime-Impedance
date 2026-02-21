"""
Core physical primitives and invariant constants for the
Applied Vacuum Electrodynamics (AVE) Framework.
"""

import scipy.constants as const
from math import pi

import numpy as np

# ---------------------------------------------------------
# Electromagnetic Constants derived from standard physics
# ---------------------------------------------------------
C_0 = const.c                           # Speed of light in vacuum [m/s]
MU_0 = const.mu_0                       # Vacuum permeability (Macroscopic Mutual Inductance) [H/m or kg*m/C^2]
EPSILON_0 = const.epsilon_0             # Vacuum permittivity (Macroscopic Compliance) [F/m or C^2/(N*m^2)]
Z_0 = np.sqrt(MU_0 / EPSILON_0)         # Characteristic impedance [Ohms] (approx 376.73)

# ---------------------------------------------------------
# AVE Specific Topological Constants
# ---------------------------------------------------------
G = const.G                             # Newtonian Gravitational Constant [m^3/(kg*s^2)]
ALPHA = const.alpha                     # Fine-structure constant (Porosity factor)

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
