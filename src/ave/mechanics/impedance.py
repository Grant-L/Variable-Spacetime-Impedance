"""
Macroscopic impedance calculations, replacing older Navier-Stokes mechanical rheology.
Calculates localized refractive index and mutual inductance transitions.

The mutual inductance saturation uses the SAME ``saturation_factor()`` that
governs particle confinement, FDTD field updates, and plasma cutoff:
when rotational shear exceeds the lattice saturation threshold, the
drag vanishes smoothly — not as a step function.
"""

import numpy as np
from ave.core.constants import G, C_0, ISOTROPIC_PROJECTION
from ave.axioms.scale_invariant import saturation_factor

def calculate_refractive_strain(mass_kg: float, radius_m: float) -> float:
    """
    Calculates the localized effective physical refractive index (n(r)) of the
    vacuum LC network created by the inductive rest energy of a massive body.
    
    n(r) = 1 + h_perp
    where h_perp is the transverse optical strain.
    
    Args:
        mass_kg (float): Mass of the polarizing body.
        radius_m (float): Distance from the center of mass.
        
    Returns:
        float: The effective localized refractive index (n >= 1.0).
    """
    if radius_m <= 0:
        raise ValueError("Radius must be greater than 0.")
        
    # 1D Principal Radial Strain: ε₁₁ = 7GM/(rc²)
    # Contract via trace-reversed Poisson ratio (2/7):
    #   n(r) = 1 + (2/7)·ε₁₁ = 1 + 2GM/(rc²)
    n_r = 1.0 + (2 * G * mass_kg) / ((C_0**2) * radius_m)
    
    return n_r


def is_dielectric_rupture(mass_kg: float, radius_m: float) -> bool:
    """
    Checks if the local metric strain exceeds the Axiom 4 Dielectric Saturation limit (Unitary Strain).
    This physically defines the Event Horizon (Schwarzschild Radius) not as curved geometry,
    but as the catastrophic impedance collapse limit.
    """
    schwarzschild_r = (2 * G * mass_kg) / (C_0**2)
    return radius_m <= schwarzschild_r


def get_mutual_inductance(shear_rate: float, background_inductance: float, saturation_threshold: float) -> float:
    """
    Effective macroscopic mutual inductance under rotational shear.

    Uses the universal Axiom 4 saturation operator — the SAME function
    that confines particles, drives FDTD, and causes plasma cutoff:

        η_eff = η₀ · √(1 − (γ̇/γ̇_yield)²)

    When shear is LOW (outer galaxy): η_eff ≈ η₀ → full drag
        → the unbroken LC lattice drags on orbiting mass
        → manifests as "dark matter"

    When shear is HIGH (inner galaxy): η_eff → 0 → no drag
        → saturated lattice cannot support transverse inductive coupling
        → conservative Keplerian orbits

    The transition is smooth, governed by the same √(1−r²) kernel that
    operates at every other scale in the framework.

    Args:
        shear_rate: Local rate of topological shear (from orbital velocity gradients).
        background_inductance: Base undisturbed inductance of deep space.
        saturation_threshold: Critical shear threshold for LC loop saturation.

    Returns:
        Effective macroscopic mutual inductance at that specific shear rate.
    """
    S = float(saturation_factor(shear_rate, saturation_threshold))
    return background_inductance * S
