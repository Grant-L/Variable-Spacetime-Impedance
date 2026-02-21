"""
AVE 3D Macroscopic Fluid Solvers
"""

from .grid_3d import EulerianGrid3D
from .bingham_cfd import BinghamFluidSolver
from .topological_coupling import apply_topological_kinematics_to_grid, apply_macroscopic_rotor_to_grid
