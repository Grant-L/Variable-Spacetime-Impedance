"""
Scale-Invariant Impedance Operations
=====================================
Every function in this module is domain-agnostic.  The SAME function computes
the reflection coefficient at a Pauli exclusion boundary, a PONDER-01 antenna
port, a seismic discontinuity, or a galactic halo edge.

This is not an analogy.  It is a structural identity:

    Z = √(μ/ε)

The saturation operator ``ε_eff(A) = ε_base · √(1 − (A/A_yield)²)`` is the
single non-linear kernel of the AVE framework, applied identically from
particle confinement (A = E·ℓ_node, A_yield = V_snap) to plasma cutoff
(A = V_local, A_yield = V_snap) to macroscopic dielectric yield.

Functions
---------
impedance(mu, eps)
    Z = √(μ/ε) — the universal operator.
saturation_factor(amplitude, yield_limit)
    √(1 − (A/A_yield)²) — Axiom 4 at any scale.
epsilon_eff(amplitude, yield_limit, eps_base)
    ε_eff = ε_base · saturation_factor()
mu_eff(amplitude, yield_limit, mu_base)
    μ_eff = μ_base · saturation_factor()
reflection_coefficient(Z1, Z2)
    Γ = (Z₂ − Z₁) / (Z₂ + Z₁) — at every boundary, every scale.
transmission_coefficient(Z1, Z2)
    T = 1 + Γ
local_wave_speed(amplitude, yield_limit, c_base)
    c_eff = c_base · (1 − (A/A_yield)²)^(1/4)
impedance_at_strain(amplitude, yield_limit, Z_base)
    Z_eff = Z_base / (1 − (A/A_yield)²)^(1/4)
"""

import numpy as np
from ave.core.constants import EPSILON_0, MU_0, C_0, Z_0, V_SNAP


# ────────────────────────────────────────────────────────────────────
# The universal impedance operator
# ────────────────────────────────────────────────────────────────────

def impedance(mu, eps):
    r"""
    Compute the characteristic impedance of any medium.

    .. math::
        Z = \sqrt{\frac{\mu}{\varepsilon}}

    This single formula is valid at every scale:

    ========================  ==========  ==========  ==============
    Domain                    μ-analog    ε-analog    Z expression
    ========================  ==========  ==========  ==============
    Vacuum lattice (fm)       μ₀          ε₀          Z₀ = 376.73 Ω
    Seismic (km)              1/G         1/K         ρ·Vₚ (Rayl)
    Protein (nm)              backbone τ  dipole C    S₁₁ impedance
    ========================  ==========  ==========  ==============

    Args:
        mu: Permeability (or inductive analog) — scalar or array.
        eps: Permittivity (or capacitive analog) — scalar or array.

    Returns:
        Impedance (same shape as inputs).
    """
    return np.sqrt(np.asarray(mu, dtype=float) /
                   np.asarray(eps, dtype=float))


# ────────────────────────────────────────────────────────────────────
# The Axiom 4 saturation kernel
# ────────────────────────────────────────────────────────────────────

def saturation_factor(
    amplitude,
    yield_limit: float = V_SNAP,
    *,
    clip: bool = True,
) -> np.ndarray:
    r"""
    The universal non-linear saturation factor (Axiom 4).

    .. math::
        S(A) = \sqrt{1 - \left(\frac{A}{A_{yield}}\right)^{\!2}}

    At A = 0: S = 1 (linear Maxwell recovered).
    At A → A_yield: S → 0 (full dielectric collapse / mass confinement).

    This kernel is the ONLY non-linearity in the AVE framework.
    It appears identically in:
      - Particle rest-mass confinement (faddeev_skyrme)
      - FDTD E- and H-field updates (fdtd_3d)
      - Nuclear bond energy (bond_energy_solver)
      - Plasma cutoff (cutoff)
      - Macroscopic dielectric yield (saturation)

    Args:
        amplitude: Local field amplitude, voltage, or strain (scalar/array).
        yield_limit: Absolute saturation limit (default V_snap = m_e c²/e).
        clip: If True (default), clip ratio² to [0, 1-ε] for numerical
              stability.  If False, raise ValueError on rupture.

    Returns:
        Saturation factor S ∈ (0, 1] — same shape as ``amplitude``.

    Raises:
        ValueError: If ``clip=False`` and |amplitude| > yield_limit.
    """
    ratio_sq = np.asarray(amplitude, dtype=float) ** 2 / yield_limit ** 2
    if clip:
        ratio_sq = np.clip(ratio_sq, 0.0, 1.0 - 1e-15)
    else:
        if np.any(ratio_sq > 1.0):
            raise ValueError(
                f"Dielectric rupture: |A/A_yield| > 1.0. "
                f"Max ratio² = {np.max(ratio_sq):.6f}. "
                f"The lattice has structurally failed at this strain."
            )
    return np.sqrt(1.0 - ratio_sq)


# ────────────────────────────────────────────────────────────────────
# Effective material parameters under saturation
# ────────────────────────────────────────────────────────────────────

def epsilon_eff(amplitude, yield_limit: float = V_SNAP,
                eps_base=EPSILON_0, *, clip: bool = True):
    r"""
    Non-linear effective permittivity under dielectric saturation.

    .. math::
        \varepsilon_{eff}(A) = \varepsilon_{base}
            \cdot \sqrt{1 - (A / A_{yield})^2}

    At A = 0: ε_eff → ε_base (linear Maxwell recovered).
    At A → A_yield: ε_eff → 0 (dielectric collapse / impedance divergence).

    NOTE: This is the CONSTITUTIVE permittivity (material property of the
    lattice node). This DECREASES under strain. The OBSERVABLE capacitance
    C_eff = 1/S → ∞ (diverges), which matches Euler-Heisenberg QED.
    These are different quantities: ε is a property, C is a response.

    Args:
        amplitude: Local strain (scalar or array).
        yield_limit: Saturation voltage / strain limit.
        eps_base: Baseline permittivity — ε₀ (or ε₀·ε_r for materials).
        clip: Clip vs. raise on rupture (see ``saturation_factor``).

    Returns:
        Effective permittivity (same shape as amplitude).
    """
    return eps_base * saturation_factor(amplitude, yield_limit, clip=clip)


def mu_eff(amplitude, yield_limit: float = V_SNAP,
           mu_base=MU_0, *, clip: bool = True):
    r"""
    Non-linear effective permeability under magnetic saturation.

    .. math::
        \mu_{eff}(B) = \mu_{base}
            \cdot \sqrt{1 - (B / B_{yield})^2}

    Args:
        amplitude: Local B-field magnitude (scalar or array).
        yield_limit: Magnetic saturation limit (B_snap).
        mu_base: Baseline permeability — μ₀ (or μ₀·μ_r for materials).
        clip: Clip vs. raise on rupture (see ``saturation_factor``).

    Returns:
        Effective permeability (same shape as amplitude).
    """
    return mu_base * saturation_factor(amplitude, yield_limit, clip=clip)


# ────────────────────────────────────────────────────────────────────
# The universal reflection and transmission coefficients
# ────────────────────────────────────────────────────────────────────

def reflection_coefficient(Z1, Z2=None):
    r"""
    Amplitude reflection coefficient at any impedance boundary.

    .. math::
        \Gamma = \frac{Z_2 - Z_1}{Z_2 + Z_1}

    This is the same operator at EVERY scale:

    ===========================  ==============================
    Scale                        Physical manifestation
    ===========================  ==============================
    Particle (Z_knot → 0)        Γ → −1  (Pauli exclusion)
    Antenna S₁₁                  Γ = (Z_L − Z₀)/(Z_L + Z₀)
    Seismic Moho                 Γ ≈ 0.17  (partial reflection)
    ===========================  ==============================

    Args:
        Z1: Impedance of the incident medium (scalar or array).
        Z2: Impedance of the transmitted medium (scalar or array).
            If None, defaults to Z₀ (vacuum).

    Returns:
        Reflection coefficient Γ ∈ [−1, +1].
    """
    if Z2 is None:
        Z2 = Z_0
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)
    denom = Z1 + Z2
    # Guard against 0/0
    return np.where(denom != 0, (Z2 - Z1) / denom, 0.0)


def transmission_coefficient(Z1, Z2=None):
    r"""
    Amplitude transmission coefficient at any impedance boundary.

    .. math::
        T = 1 + \Gamma = \frac{2 Z_1}{Z_1 + Z_2}

    Args:
        Z1: Impedance of incident medium.
        Z2: Impedance of transmitted medium (default: Z₀).

    Returns:
        Transmission coefficient T.
    """
    return 1.0 + reflection_coefficient(Z1, Z2)


# ────────────────────────────────────────────────────────────────────
# Derived wave-speed and impedance under saturation
# ────────────────────────────────────────────────────────────────────

def local_wave_speed(amplitude, yield_limit: float = V_SNAP,
                     c_base: float = C_0, *, clip: bool = True):
    r"""
    Effective local phase velocity under dielectric saturation.

    .. math::
        c_{eff}(A) = c_{base} \cdot (1 - (A/A_{yield})^2)^{1/4}

    Derived from c = 1/√(μ·ε): when ε collapses as √(1−r²), c drops
    as (1−r²)^{1/4}.  At saturation, c → 0 (wave packet freezes → mass).

    Args:
        amplitude: Local strain amplitude.
        yield_limit: Saturation limit.
        c_base: Base wave speed (default c₀).
        clip: See ``saturation_factor``.

    Returns:
        Local phase velocity [m/s].
    """
    ratio_sq = np.asarray(amplitude, dtype=float) ** 2 / yield_limit ** 2
    if clip:
        ratio_sq = np.clip(ratio_sq, 0.0, 1.0 - 1e-15)
    return c_base * (1.0 - ratio_sq) ** 0.25


def impedance_at_strain(amplitude, yield_limit: float = V_SNAP,
                        Z_base: float = Z_0, *, clip: bool = True):
    r"""
    Local characteristic impedance under dielectric saturation.

    .. math::
        Z_{eff}(A) = \frac{Z_{base}}{(1 - (A/A_{yield})^2)^{1/4}}

    As strain increases, impedance rises (the medium becomes opaque
    to transverse waves).  At exact saturation the impedance formally
    diverges — but physically the lattice ruptures first.

    Args:
        amplitude: Local strain amplitude.
        yield_limit: Saturation limit.
        Z_base: Base impedance (default Z₀).
        clip: See ``saturation_factor``.

    Returns:
        Local impedance [Ω].
    """
    ratio_sq = np.asarray(amplitude, dtype=float) ** 2 / yield_limit ** 2
    if clip:
        ratio_sq = np.clip(ratio_sq, 0.0, 1.0 - 1e-15)
    return Z_base / (1.0 - ratio_sq) ** 0.25
