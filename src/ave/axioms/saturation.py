"""
Axiom 4: Dielectric Saturation
================================
The vacuum acts as a non-linear dielectric bounded by the fine-structure
limit α. The effective capacitance diverges as local strain approaches
saturation, while effective permittivity collapses — trapping energy into
topological knots (mass).

The saturation operator is strictly squared (n=2) to align with:
  - The E⁴ energy density of the Euler-Heisenberg QED Lagrangian
  - The χ⁽³⁾ displacement of the optical Kerr effect
  - Standard Born-Infeld non-linear electrodynamics

Key functions:
  epsilon_eff(V, V_yield)  — Non-linear permittivity
  capacitance_eff(dphi, alpha) — Non-linear capacitance
  reflection_coefficient(Z_knot, Z_vac) — Transmission line Γ
  local_wave_speed(V, V_yield) — c_eff under saturation
  energy_density_nonlinear(dphi, alpha, eps0) — Full non-linear U
"""

import numpy as np
from ave.core.constants import EPSILON_0, MU_0, C_0, ALPHA, V_SNAP, Z_0


def epsilon_eff(V: np.ndarray | float, V_yield: float = V_SNAP) -> np.ndarray | float:
    """
    Non-linear effective permittivity under dielectric saturation.

    ε_eff(V) = ε₀ · √(1 − (V/V_yield)²)

    As V → V_yield, ε_eff → 0 (impedance collapse).
    As V → 0, ε_eff → ε₀ (linear Maxwell recovery).

    Args:
        V: Local potential or strain amplitude (scalar or array).
        V_yield: Absolute dielectric yield limit (default: V_snap = m_e c²/e).

    Returns:
        Effective permittivity at each point.

    Raises:
        ValueError: If |V| > V_yield (physical rupture — the lattice has broken).
    """
    ratio_sq = np.asarray(V, dtype=float)**2 / V_yield**2
    if np.any(ratio_sq > 1.0):
        raise ValueError(
            f"Dielectric rupture: |V/V_yield| > 1.0. "
            f"Max ratio² = {np.max(ratio_sq):.6f}. "
            f"The lattice has structurally failed at this strain."
        )
    return EPSILON_0 * np.sqrt(1.0 - ratio_sq)


def capacitance_eff(
    dphi: np.ndarray | float,
    alpha: float = ALPHA,
) -> np.ndarray | float:
    """
    Non-linear effective capacitance per the manuscript's Axiom 4:

    C_eff(Δφ) = C₀ / √(1 − (Δφ/α)²)

    Note the INVERSE relationship vs epsilon_eff: as strain increases,
    capacitance DIVERGES (the node absorbs more displacement current)
    while permittivity COLLAPSES (wave speed drops to zero).

    Args:
        dphi: Normalized phase displacement (dimensionless, 0 ≤ |Δφ| < α).
        alpha: Fine-structure saturation limit.

    Returns:
        Effective capacitance ratio (C_eff / C₀). Multiply by C₀ for absolute.
    """
    ratio_sq = np.asarray(dphi, dtype=float)**2 / alpha**2
    if np.any(ratio_sq >= 1.0):
        raise ValueError(
            f"Capacitance singularity: |Δφ/α| ≥ 1.0. "
            f"Max ratio² = {np.max(ratio_sq):.6f}."
        )
    return 1.0 / np.sqrt(1.0 - ratio_sq)


def reflection_coefficient(Z_knot: float, Z_vac: float = Z_0) -> float:
    """
    Transmission line reflection coefficient:

    Γ = (Z_knot − Z_vac) / (Z_knot + Z_vac)

    At dielectric saturation, Z_knot → 0 Ω, so Γ → −1 (perfect reflection).
    This is the physical origin of the Pauli exclusion principle and
    particle confinement.

    Args:
        Z_knot: Local impedance of the saturated region [Ω].
        Z_vac: Ambient vacuum impedance [Ω] (default: Z₀ ≈ 376.73 Ω).

    Returns:
        Reflection coefficient Γ ∈ [−1, +1].
    """
    return (Z_knot - Z_vac) / (Z_knot + Z_vac)


def local_wave_speed(
    V: np.ndarray | float,
    V_yield: float = V_SNAP,
) -> np.ndarray | float:
    """
    Effective local phase velocity under dielectric saturation:

    c_eff(V) = c₀ · (1 − (V/V_yield)²)^(1/4)

    Derived from c_eff = 1/√(μ₀ · ε_eff) where ε_eff = ε₀√(1-(V/V_yield)²).
    As V → V_yield, c_eff → 0 (wave packet freezes, forming mass).

    Args:
        V: Local potential/strain amplitude.
        V_yield: Absolute yield limit.

    Returns:
        Local phase velocity [m/s].
    """
    ratio_sq = np.asarray(V, dtype=float)**2 / V_yield**2
    ratio_sq = np.clip(ratio_sq, 0.0, 1.0 - 1e-15)
    return C_0 * (1.0 - ratio_sq)**0.25


def energy_density_nonlinear(
    dphi: np.ndarray | float,
    alpha: float = ALPHA,
    eps0: float = EPSILON_0,
) -> np.ndarray | float:
    """
    Full non-linear energy density including the E⁴ correction term:

    U ≈ ½ε₀(Δφ)² + (3/8α²)ε₀(Δφ)⁴

    The first term is standard Maxwell. The second term is the Euler-Heisenberg
    QED correction that structurally emerges from Axiom 4's squared saturation
    operator.

    Args:
        dphi: Phase displacement (dimensionless or field amplitude).
        alpha: Fine-structure saturation limit.
        eps0: Baseline permittivity.

    Returns:
        Energy density [J/m³ equivalent units].
    """
    dphi = np.asarray(dphi, dtype=float)
    linear_term = 0.5 * eps0 * dphi**2
    nonlinear_correction = (3.0 / (8.0 * alpha**2)) * eps0 * dphi**4
    return linear_term + nonlinear_correction


def impedance_at_strain(V: np.ndarray | float, V_yield: float = V_SNAP) -> np.ndarray | float:
    """
    Local characteristic impedance under dielectric saturation:

    Z_eff(V) = √(μ₀ / ε_eff(V)) = Z₀ / (1 − (V/V_yield)²)^(1/4)

    As strain increases, impedance rises toward infinity (the spatial medium
    becomes increasingly opaque to transverse waves), until the exact saturation
    point where the impedance formally diverges — however physically, the
    medium ruptures into a zero-impedance phase before reaching infinity.

    Args:
        V: Local potential/strain amplitude.
        V_yield: Absolute yield limit.

    Returns:
        Local impedance [Ω].
    """
    ratio_sq = np.asarray(V, dtype=float)**2 / V_yield**2
    ratio_sq = np.clip(ratio_sq, 0.0, 1.0 - 1e-15)
    return Z_0 / (1.0 - ratio_sq)**0.25
