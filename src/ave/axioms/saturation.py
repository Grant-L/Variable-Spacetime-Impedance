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

All core saturation operations are implemented in
``ave.axioms.scale_invariant`` — the SAME functions serve every scale.
This module re-exports them with Axiom-4-specific defaults and adds
the capacitance and energy-density formulas unique to this domain.

Key functions:
  epsilon_eff(V, V_yield)  — Non-linear permittivity
  capacitance_eff(dphi, alpha) — Non-linear capacitance
  reflection_coefficient(Z_knot, Z_vac) — Transmission line Γ
  local_wave_speed(V, V_yield) — c_eff under saturation
  energy_density_nonlinear(dphi, alpha, eps0) — Full non-linear U
  impedance_at_strain(V, V_yield) — Z_eff under saturation
"""

import numpy as np
from ave.core.constants import EPSILON_0, MU_0, C_0, ALPHA, V_SNAP, Z_0

# Import the scale-invariant canonical implementations
from ave.axioms.scale_invariant import (
    saturation_factor as _saturation_factor,
    epsilon_eff as _si_epsilon_eff,
    mu_eff as _si_mu_eff,
    reflection_coefficient as _si_reflection_coefficient,
    local_wave_speed as _si_local_wave_speed,
    impedance_at_strain as _si_impedance_at_strain,
)


def epsilon_eff(V: np.ndarray | float, V_yield: float = V_SNAP) -> np.ndarray | float:
    """
    Non-linear effective permittivity under dielectric saturation.

    ε_eff(V) = ε₀ · √(1 − (V/V_yield)²)

    As V → V_yield, ε_eff → 0 (impedance collapse).
    As V → 0, ε_eff → ε₀ (linear Maxwell recovery).

    Delegates to ``ave.axioms.scale_invariant.epsilon_eff`` — the same
    function that computes saturation at every other scale.

    Args:
        V: Local potential or strain amplitude (scalar or array).
        V_yield: Absolute dielectric yield limit (default: V_snap = m_e c²/e).

    Returns:
        Effective permittivity at each point.

    Raises:
        ValueError: If |V| > V_yield (physical rupture — the lattice has broken).
    """
    return _si_epsilon_eff(V, V_yield, EPSILON_0, clip=False)


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

    Delegates to ``ave.axioms.scale_invariant.reflection_coefficient`` —
    the same operator that computes seismic Moho reflections, antenna S₁₁,
    and every other impedance boundary in the framework.

    Args:
        Z_knot: Local impedance of the saturated region [Ω].
        Z_vac: Ambient vacuum impedance [Ω] (default: Z₀ ≈ 376.73 Ω).

    Returns:
        Reflection coefficient Γ ∈ [−1, +1].
    """
    # Note: scale_invariant convention is Γ = (Z2 - Z1) / (Z2 + Z1)
    # saturation convention is Z_knot = incident, Z_vac = reference
    # Γ = (Z_knot - Z_vac) / (Z_knot + Z_vac) = -(Z_vac - Z_knot)/(Z_vac + Z_knot)
    # Using scale_invariant: _si_reflection_coefficient(Z_vac, Z_knot) gives
    # (Z_knot - Z_vac)/(Z_vac + Z_knot) — same result!
    return float(_si_reflection_coefficient(Z_vac, Z_knot))


def local_wave_speed(
    V: np.ndarray | float,
    V_yield: float = V_SNAP,
) -> np.ndarray | float:
    """
    Effective local phase velocity under dielectric saturation:

    c_eff(V) = c₀ · (1 − (V/V_yield)²)^(1/4)

    Derived from c_eff = 1/√(μ₀ · ε_eff) where ε_eff = ε₀√(1-(V/V_yield)²).
    As V → V_yield, c_eff → 0 (wave packet freezes, forming mass).

    Delegates to ``ave.axioms.scale_invariant.local_wave_speed``.

    Args:
        V: Local potential/strain amplitude.
        V_yield: Absolute yield limit.

    Returns:
        Local phase velocity [m/s].
    """
    return _si_local_wave_speed(V, V_yield, C_0, clip=True)


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

    Delegates to ``ave.axioms.scale_invariant.impedance_at_strain``.

    Args:
        V: Local potential/strain amplitude.
        V_yield: Absolute yield limit.

    Returns:
        Local impedance [Ω].
    """
    return _si_impedance_at_strain(V, V_yield, Z_0, clip=True)
