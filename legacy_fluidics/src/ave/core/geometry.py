"""
AVE Topological Geometry
Defines the knot geometries for fundamental particles.
Source: Chapter 5 (The Golden Torus) & Chapter 6 (Borromean Linkages)
"""
import math

from ave.core import constants as k

class GoldenTorus:
    """The geometric definition of the Electron (3_1 Trefoil)."""
    def __init__(self):
        self.phi = (1.0 + math.sqrt(5.0)) / 2.0
        self.R_norm = (1.0 + math.sqrt(5.0)) / 4.0  
        self.r_norm = (self.phi - 1.0) / 2.0      

    def physical_major_radius(self):
        return self.R_norm * k.L_NODE

    def physical_minor_radius(self):
        return self.r_norm * k.L_NODE
        
    def topological_impedance(self):
        return (4.0 * math.pi**3) + (math.pi**2) + math.pi

class BorromeanLinkage:
    """The geometric definition of the Proton (6^3_2 Link)."""
    def __init__(self):
        self.loops = 3
        self.symmetry = "Z3" 
        
    def charge_fractionalization(self):
        thetas = [0.0, (2.0 * math.pi) / 3.0, (4.0 * math.pi) / 3.0]
        charges = []
        for theta in thetas:
            q_fract = theta / (2.0 * math.pi)
            charges.append(q_fract)
        return charges