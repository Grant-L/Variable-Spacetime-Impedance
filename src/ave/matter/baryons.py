"""
AVE Baryon Sector
Implements the physics of Borromean Linkages (Protons/Neutrons).
Source: Chapter 6
"""
import math
import scipy.constants as const
from ave.core import constants as k

def calculate_strong_force_tension(m_proton=const.m_p):
    """
    Derives the Strong Force String Tension (Sigma).
    Model: Amplified elastic strain of a Borromean Linkage (Proton).
    Formula: 3 * (m_p/m_e) * alpha^-1 * T_EM
    Source: Eq 6.1
    """
    mass_ratio = m_proton / k.m_e
    Q_factor = k.alpha_geom_inv # approx 137.036
    
    # Total mechanical force (Newtons)
    # Factor of 3 comes from the 3 loops of the Borromean Link
    F_confinement = 3 * mass_ratio * Q_factor * k.T_EM
    
    return F_confinement

def calculate_proton_mass_components():
    """
    Decomposes the Proton Mass into Scalar (1D) and Tensor (3D) traces.
    Source: Eq 6.4 & 6.5
    """
    total_mass = const.m_p
    
    # The 1D Scalar Radial Projection (approx 1162 m_e)
    # This is the mass if the proton were a spherical scalar soliton
    scalar_trace = 1162.0 * k.m_e
    
    # The 3D Orthogonal Tensor Trace (The "Mass Gap")
    # This arises from the crossing flux lines in the Borromean core
    tensor_trace = total_mass - scalar_trace
    
    return {
        "scalar_contribution": scalar_trace,
        "tensor_deficit": tensor_trace,
        "total_mass": total_mass
    }