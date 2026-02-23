"""
Variable Spacetime Impedance (AVE) - Unified Physics Engine.
This core library maps discrete mass geometry and continuum fluid mechanics
onto a rigorous LC matrix. 
"""

from .grid import VacuumGrid
from .node import TopologicalNode

__all__ = ['VacuumGrid', 'TopologicalNode']
