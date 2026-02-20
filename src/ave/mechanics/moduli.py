"""
AVE Mechanical Moduli
Derives the density, viscosity, and elasticity of the vacuum condensate.
Source: Chapter 2, 4, & 11 of main.pdf
"""
import sys
from pathlib import Path

# Add src directory to path if running as script (before imports)
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ave.core import constants as k

def get_poisson_ratio():
    """
    Returns the exact trace-reversed Cosserat Poisson ratio.
    Source: Eq 4.1 
    Note: Currently an analytical limit (2/7). Will be computationally 
    proven via percolation.py in Phase 3.
    """
    return 2.0 / 7.0

def calculate_bulk_density():
    """
    Calculates the macroscopic bulk mass density (rho_bulk) of the spatial fluid.
    Source: Eq 11.1
    """
    return k.RHO_BULK

def calculate_kinematic_viscosity():
    """
    Calculates the kinematic viscosity (nu_vac) of the vacuum.
    Source: Eq 11.2
    """
    return k.NU_VAC

def calculate_bingham_yield_stress():
    """
    Calculates the macroscopic shear stress required to liquefy the vacuum.
    Derivation: 1D Snap Voltage scaled by 1/7 tensor projection, over node area.
    Source: Eq 4.9
    """
    # 1. Nodal Breakdown Voltage (V_snap)
    V_snap = (k.M_E * k.C**2) / k.E_CHARGE
    
    # 2. Bingham Yield Voltage (V_yield - Macroscopic Shielding)
    V_yield = V_snap / 7.0
    
    # 3. Topological Force Yield
    F_yield = V_yield * k.XI_TOPO
    
    # 4. Macroscopic Stress (Force / Area)
    area_node = k.L_NODE**2
    tau_yield = F_yield / area_node
    
    return tau_yield