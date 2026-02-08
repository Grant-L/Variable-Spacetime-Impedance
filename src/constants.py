"""
Physical constants and unit definitions for the project.
"""

import scipy.constants as const

# Fundamental constants
C = const.c  # Speed of light [m/s]
G = const.G  # Gravitational constant [m^3 kg^-1 s^-2]
HBAR = const.hbar  # Reduced Planck constant [J s]
H = const.h  # Planck constant [J s]
K_B = const.k  # Boltzmann constant [J/K]
EPSILON_0 = const.epsilon_0  # Vacuum permittivity [F/m]
MU_0 = const.mu_0  # Vacuum permeability [H/m]

# Derived constants
PLANCK_LENGTH = const.physical_constants['Planck length'][0]  # [m]
PLANCK_TIME = const.physical_constants['Planck time'][0]  # [s]
PLANCK_MASS = const.physical_constants['Planck mass'][0]  # [kg]

# Unit conversions
EV_TO_JOULE = const.e  # Electron volt to joule
KG_TO_EV = 1 / (EV_TO_JOULE / C**2)  # Mass-energy conversion
