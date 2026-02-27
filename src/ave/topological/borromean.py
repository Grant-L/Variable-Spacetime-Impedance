"""
Topological Generator Module: borromean.py
-------------------------------------------
Generates exact 3D parametric coordinate arrays for fundamental macroscopic
topological defects (knots and links) within the AVE discrete framework.

Includes:
- 0_1 Unknot (Fundamental Lepton / Electron)
- 3_1 Trefoil Knot (used in HOPF-01 antenna topology, NOT the electron)
- 6^3_2 Borromean Link (Fundamental Baryon / Proton)
"""
import numpy as np

class FundamentalTopologies:
    @staticmethod
    def generate_unknot_0_1(radius: float, resolution: int = 1000):
        """
        Generates the 3D parametric coordinates of a 0_1 Unknot.
        In the AVE framework, this is the fundamental lepton topology:
        a single closed flux tube loop at minimum ropelength = 2π.

        The unknot has circumference ℓ_node and tube radius ℓ_node/(2π).
        Its mass is m_e = T_EM · ℓ_node / c².

        Args:
            radius (float): The characteristic radial scale of the loop.
            resolution (int): Number of coordinate points along the curve.

        Returns:
            np.ndarray: [N, 3] array of (x,y,z) spatial coordinates.
        """
        t = np.linspace(0, 2 * np.pi, resolution)

        # Simple torus loop (unknot) in the X-Y plane:
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = np.zeros_like(t)

        coords = np.vstack((x, y, z)).T
        return coords

    @staticmethod
    def generate_trefoil_3_1(radius: float, resolution: int = 1000):
        """
        Generates the 3D parametric coordinates of a 3_1 Trefoil Knot.
        NOTE: This is used for the HOPF-01 antenna topology and torus knot
        classification, NOT as the electron's ground-state topology (which
        is the unknot, see generate_unknot_0_1).

        Args:
            radius (float): The characteristic radial scale of the defect.
            resolution (int): Number of coordinate points along the curve.

        Returns:
            np.ndarray: [N, 3] array of (x,y,z) spatial coordinates.
        """
        t = np.linspace(0, 2 * np.pi, resolution)

        # Standard parametric trefoil:
        # x = (sin(t) + 2sin(2t)) * scale
        # y = (cos(t) - 2cos(2t)) * scale
        # z = -sin(3t) * scale

        x = radius * (np.sin(t) + 2.0 * np.sin(2.0 * t))
        y = radius * (np.cos(t) - 2.0 * np.cos(2.0 * t))
        z = radius * (-np.sin(3.0 * t))

        # Normalize the structural scale so 'radius' controls the maximum bounding extent
        max_extent = np.max(np.sqrt(x**2 + y**2 + z**2))
        scale_factor = radius / max_extent

        coords = np.vstack((x, y, z)).T * scale_factor
        return coords

    @staticmethod
    def generate_borromean_6_3_2(radius: float, eccentricity: float = 1.6, resolution: int = 1000):
        """
        Generates the 3D parametric coordinates of the 6^3_2 Borromean Link.
        Consists of three mutually interlocking independent discrete rings.
        In the AVE framework, this defines the topological geometry of the Proton.
        
        Args:
            radius (float): The bounding scale of the overall structure.
            eccentricity (float): Flattening of the individual elliptical links.
            resolution (int): Points per individual ring.
            
        Returns:
            list of np.ndarray: A list containing three [N, 3] coordinate arrays, 
                                one for each intersecting loop.
        """
        # The Borromean rings can be parametrized as 3 mutually perpendicular, 
        # undulating ellipses that interlock without touching.
        
        t = np.linspace(0, 2 * np.pi, resolution)
        
        # Undulation parameters to guarantee the over/under braided weaving
        # (Standard L-G topological formulation)
        base_r = radius
        
        # Ring 1 (Primarily along X-Y plane, undulating in Z)
        x1 = base_r * np.cos(t)
        y1 = base_r * eccentricity * np.sin(t)
        z1 = base_r * 0.3 * np.cos(3 * t)
        ring_1 = np.vstack((x1, y1, z1)).T

        # Ring 2 (Primarily along Y-Z plane, undulating in X)
        x2 = base_r * 0.3 * np.cos(3 * t)
        y2 = base_r * np.cos(t)
        z2 = base_r * eccentricity * np.sin(t)
        ring_2 = np.vstack((x2, y2, z2)).T

        # Ring 3 (Primarily along Z-X plane, undulating in Y)
        x3 = base_r * eccentricity * np.sin(t)
        y3 = base_r * 0.3 * np.cos(3 * t)
        z3 = base_r * np.cos(t)
        ring_3 = np.vstack((x3, y3, z3)).T
        
        return [ring_1, ring_2, ring_3]
