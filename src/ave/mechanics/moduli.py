"""
AVE Mechanical Moduli
Derives the density, viscosity, and elasticity of the vacuum condensate.
Source: Chapter 2, 4, & 11 of main.pdf
"""
import sys
from pathlib import Path

# Add src directory to path if running as script (before imports)
src_dir = Path(__file__).parent.parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ave.core import constants as k

def get_poisson_ratio():
    """
    Returns the exact trace-reversed Cosserat Poisson ratio.
    Note: Computationally proven via percolation.py solver.
    """
    return 2.0 / 7.0

def calculate_bulk_density():
    return k.RHO_BULK

def calculate_kinematic_viscosity():
    return k.NU_VAC

def calculate_bingham_yield_stress():
    """Calculates the macroscopic shear stress required to liquefy the vacuum."""
    V_snap = (k.M_E * k.C**2) / k.E_CHARGE
    V_yield = V_snap / 7.0
    F_yield = V_yield * k.XI_TOPO
    area_node = k.L_NODE**2
    return F_yield / area_node