"""
AVE Mechanical Moduli
Derives the density, viscosity, and elasticity of the vacuum condensate.
Source: Chapter 2, 4, & 11 of main.pdf
"""
import math
from ave.core import constants as k

def get_poisson_ratio():
    """
    Returns the exact trace-reversed Cosserat Poisson ratio.
    Source: Eq 4.1 [cite: 272, 1407]
    """
    return 2.0 / 7.0

def calculate_bulk_density():
    """
    Calculates the macroscopic bulk mass density (rho_bulk) of the spatial fluid.
    Formula: (xi_topo^2 * mu_0) / (8 * pi * alpha * l_node^2)
    Source: Eq 11.1 [cite: 729, 1402]
    """
    numerator = (k.xi_topo**2) * k.mu_0
    denominator = k.kappa_v * (k.l_node**2) # kappa_v is 8*pi*alpha
    return numerator / denominator

def calculate_kinematic_viscosity():
    """
    Calculates the kinematic viscosity (nu_vac) of the vacuum.
    Formula: alpha * c * l_node
    Source: Eq 11.2 [cite: 738, 1403]
    """
    return k.alpha_geom * k.c * k.l_node

def calculate_bingham_yield_stress():
    """
    Calculates the macroscopic shear stress required to liquefy the vacuum.
    Derivation: 1D Snap Voltage scaled by 1/7 tensor projection, over node area.
    Source: Eq 4.9 [cite: 340]
    """
    # 1. Nodal Breakdown Voltage (V_snap) [cite: 147]
    V_snap = (k.m_e * k.c**2) / k.e_charge
    
    # 2. Bingham Yield Voltage (V_yield) [cite: 337]
    V_yield = V_snap / 7.0
    
    # 3. Topological Force Yield [cite: 337]
    F_yield = V_yield * k.xi_topo
    
    # 4. Macroscopic Stress (Force / Area) [cite: 340]
    area_node = k.l_node**2
    tau_yield = F_yield / area_node
    
    return tau_yield