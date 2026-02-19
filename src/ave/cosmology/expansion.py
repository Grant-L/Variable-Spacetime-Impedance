"""
AVE Generative Cosmology
Derives expansion rates and dark sector dynamics from lattice genesis.
Source: Chapter 10 & 11 of main.pdf
"""
import math
import scipy.constants as const
from ave.core import constants as k

def calculate_hubble_constant_limit():
    """
    Derives the Asymptotic de Sitter Limit (H_infinity).
    Formula: (28 * pi * m_e^3 * c * G) / (hbar^2 * alpha^2)
    Source: Eq 4.7 [cite: 328, 1414]
    """
    numerator = 28 * math.pi * (k.m_e**3) * k.c * k.G
    denominator = (k.hbar**2) * (k.alpha_geom**2)
    
    H_si = numerator / denominator # Units: 1/s
    
    # Convert to km/s/Mpc for comparison
    km_per_Mpc = 3.08567758e19
    H_km_s_Mpc = H_si * (km_per_Mpc / 1000.0)
    
    return H_km_s_Mpc

def calculate_mond_acceleration(H_0_si):
    """
    Derives the MOND acceleration threshold (a_0) from Unruh-Hawking drift.
    Formula: (c * H_0) / (2 * pi)
    Source: Eq 11.3 [cite: 763, 1415]
    """
    return (k.c * H_0_si) / (2 * math.pi)

def calculate_dark_energy_eos(rho_latent, rho_vac):
    """
    Derives the Dark Energy Equation of State (w).
    Source: Eq 10.3 [cite: 694, 1414]
    NOTE: In the text, this evaluates to approx -1.0001
    """
    return -1.0 - (rho_latent / rho_vac)