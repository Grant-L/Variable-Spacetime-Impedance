"""
AVE Macroscopic Gravity
Implements the Refractive Optical Metric and MOND kinematics.
Source: Chapter 9 & Chapter 11
"""
import math
import scipy.constants as const
from ave.core import constants as k
from ave.cosmology import expansion

def refractive_index_scalar(mass, radius):
    """
    Calculates the scalar refractive index n(r) for a mass M.
    Used for: Ponderomotive force (massive particles).
    Source: Eq 9.4 (Derived from n = 1 + h/7)
    """
    if radius == 0: return float('inf')
    potential = (k.G * mass) / (k.c**2 * radius)
    return 1.0 + potential

def refractive_index_optical(mass, radius):
    """
    Calculates the transverse optical refractive index n_perp(r).
    Used for: Gravitational Lensing (photons).
    Note: Includes the factor of 2 from Poisson Ratio 2/7.
    Source: Eq 9.6
    """
    if radius == 0: return float('inf')
    potential = (2.0 * k.G * mass) / (k.c**2 * radius)
    return 1.0 + potential

def calculate_mond_velocity(baryonic_mass):
    """
    Derives the asymptotic flat rotation velocity (Tully-Fisher).
    Formula: v_flat = (G * M * a_0)^1/4
    Source: Eq 11.4
    """
    # 1. Get H0 in SI units
    H0_km_Mpc = expansion.calculate_hubble_constant_limit()
    H0_si = H0_km_Mpc * 1000 / 3.086e22
    
    # [cite_start]2. Derive a_0 (Unruh-Hawking drift) [cite: 763]
    a_genesis = (k.c * H0_si) / (2 * math.pi)
    
    # [cite_start]3. Calculate Flat Velocity [cite: 769]
    v_flat = (k.G * baryonic_mass * a_genesis)**0.25
    
    return v_flat, a_genesis