"""
AVE Core Constants
Implements the Single-Parameter calibration anchored to the electron.
Source: Chapter 1 & Appendix F of main.pdf
"""
import math
import scipy.constants as const

# ==========================================
# 1. INPUTS (Standard Model Empirical Data)
# ==========================================
c = const.c                  # Speed of Light
hbar = const.hbar            # Reduced Planck Constant
m_e = const.m_e              # Electron Rest Mass
e_charge = const.e           # Elementary Charge
mu_0 = const.mu_0            # Vacuum Permeability
epsilon_0 = const.epsilon_0  # Vacuum Permittivity
G = const.G                  # Gravitational Constant (Macroscopic)

# ==========================================
# 2. AVE DERIVED CONSTANTS (The Kernel)
# ==========================================

# AXIOM 1: The Topological Coherence Length (l_node)
# Calibration: Anchored to the electron Compton scale.
# Source: [cite: 12, 43, 72, 104]
l_node = hbar / (m_e * c)

# AXIOM 2: The Topo-Kinematic Conversion Constant (xi_topo)
# Definition: Charge is spatial dislocation ([Q] == [L]).
# Source: [cite: 86, 121, 1399]
xi_topo = e_charge / l_node

# AXIOM 4: The Geometric Fine Structure Constant (alpha_geom)
# Derivation: The topological impedance of a Golden Torus (3_1 knot).
# Calculation: 1 / (4*pi^3 + pi^2 + pi)
# Source: [cite: 114, 422, 1400]
alpha_geom_inv = (4 * math.pi**3) + (math.pi**2) + math.pi
alpha_geom = 1.0 / alpha_geom_inv

# The Vacuum Packing Fraction (kappa_v)
# Derivation: Collapsing the QED yield density to the discrete node volume.
# Source: [cite: 133, 139, 1401]
kappa_v = 8 * math.pi * alpha_geom

# The 1D Topological String Tension (T_EM)
# Derivation: Maximum tension before a single flux line snaps.
# Source: [cite: 296, 452]
T_EM = (m_e * c**2) / l_node