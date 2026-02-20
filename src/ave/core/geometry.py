"""
AVE Topological Geometry
Defines the knot geometries for fundamental particles.
Source: Chapter 5 (The Golden Torus) & Chapter 6 (Borromean Linkages)
"""
import math
from ave.core import constants as k

class GoldenTorus:
    """
    The geometric definition of the Electron (3_1 Trefoil).
    Source: Eq 5.2
    """
    def __init__(self):
        # The Golden Ratio
        self.phi = (1 + math.sqrt(5)) / 2
        
        # Dimensionless radii (normalized to l_node = 1)
        # R * r = 1/4 (Holomorphic screening)
        # R - r = 1/2 (Self-avoidance limit)
        self.R_norm = (1 + math.sqrt(5)) / 4  # approx 0.809
        self.r_norm = (self.phi - 1) / 2      # approx 0.309

    def physical_major_radius(self):
        """Returns physical Major Radius (R) in meters."""
        return self.R_norm * k.L_NODE

    def physical_minor_radius(self):
        """Returns physical Minor Radius (r) in meters."""
        return self.r_norm * k.L_NODE
        
    def topological_impedance(self):
        """
        Calculates the theoretical Alpha inverse from geometry.
        Source: Eq 5.3
        """
        term_vol = 4 * (math.pi**3)
        term_surf = math.pi**2
        term_line = math.pi
        return term_vol + term_surf + term_line

class BorromeanLinkage:
    """
    The geometric definition of the Proton (6^3_2 Link).
    Source: Chapter 6
    """
    def __init__(self):
        self.loops = 3
        self.symmetry = "Z3" # Permutation symmetry
        
    def charge_fractionalization(self):
        """
        Derives quark charges via Witten Effect on Z3 symmetry.
        Source: Eq 6.6 - 6.8
        """
        # Theta vacuums allowed by Z3
        thetas = [0.0, (2 * math.pi) / 3.0, (4 * math.pi) / 3.0]
        
        charges = []
        for theta in thetas:
            # Witten effect: q_eff = n + theta/2pi
            # Base integer n=0
            q_fract = theta / (2 * math.pi)
            charges.append(q_fract)
            
        return charges # Returns [0.0, 0.333..., 0.666...] (and symmetric negatives)