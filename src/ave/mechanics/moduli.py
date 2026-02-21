"""
AVE Mechanical Moduli
Derives the density, viscosity, and elasticity of the vacuum condensate.
Source: Chapter 2, 4, & 11 of main.pdf
"""
import sys
from pathlib import Path

# Add src directory to path if running as script (before imports)
src_dir = Path(__file__).resolve().parent.parent.parent
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
    """
    Calculates the macroscopic shear stress required to liquefy the vacuum.
    Derived as the scalar sum of the 6^3_2 tensor crossings evaluated 
    over the macroscopic bulk energy density.
    """
    from ave.matter.baryons import BorromeanTensorSolver
    
    # 1. Macroscopic Baseline Energy Density (J/m^3 or Pa)
    energy_density = k.RHO_BULK * (k.C**2)
    
    # 2. Integrate the 6^3_2 Topological Crossings
    solver = BorromeanTensorSolver(grid_resolution=60)
    v_single = solver.evaluate_tensor_crossing_volume()
    n_crossings = 6.0
    tensor_scalar = n_crossings * v_single
    
    # 3. Apply geometric porosity (alpha) to yield the macroscopic plastic limit
    tau_yield = energy_density * tensor_scalar * k.ALPHA_GEOM
    
    return tau_yield