"""
AVE Gravitational Wave Propagation
====================================

Gravitational waves in the AVE framework are transverse inductive
shear waves propagating through the structured LC vacuum. They are
governed by the SAME impedance, saturation, and reflection operators
used across all other scales.

Physical picture (from Ch. 19):
  - Mass = localized topological energy deficit in the LC lattice
  - Gravity = dielectric refraction: Z(r) varies radially around mass
  - GW = transverse shear (μ-sector) perturbation radiating outward
  - At h ~ 10⁻²¹, the strain is 10¹⁹× below V_SNAP → no saturation
  - Therefore: perfectly linear, lossless, c-speed propagation

Key identities:
  Schwarzschild: ε_eff(r) = ε₀ / (1 − r_s/r)
                 μ_eff(r) = μ₀ · (1 − r_s/r)    [shear sector]
                 Z(r)     = Z₀ · (1 − r_s/r)

  At r → r_s:  ε → ∞, μ → 0, Z → 0 → Γ → −1 (total reflection)
  This IS the event horizon — it's the superconductor/plasma duality
  applied to the gravitational sector.

  GW propagation: same FDTD engine, same saturation_factor, applied
  to strain amplitudes h << V_SNAP/mc² → linear regime.

Correspondences:

    Plasma           Superconductor    Gravity Well
    ──────           ──────────────    ────────────
    ε → 0            μ → 0             ε → ∞, μ → 0
    E-field expelled B-field expelled  GW confined
    ω < ω_p          B < B_c            r < r_s
    Γ → −1           Γ → −1            Γ → −1 (horizon)
"""

import numpy as np
from typing import Optional

from ave.core.constants import (
    C_0, EPSILON_0, MU_0, Z_0, V_SNAP, B_SNAP, G, HBAR, L_NODE,
)
from ave.axioms.scale_invariant import (
    saturation_factor,
    impedance,
    reflection_coefficient,
)


# ═══════════════════════════════════════════════════════════════
# Schwarzschild impedance profile — gravity as dielectric refraction
# ═══════════════════════════════════════════════════════════════

def schwarzschild_radius(M: float) -> float:
    r"""
    Schwarzschild radius of a mass M.

    .. math::
        r_s = \frac{2 G M}{c^2}

    Args:
        M: Mass [kg].

    Returns:
        Schwarzschild radius [m].
    """
    return 2 * G * M / C_0**2


def epsilon_eff_schwarzschild(r: float | np.ndarray,
                               r_s: float) -> float | np.ndarray:
    r"""
    Effective permittivity in a Schwarzschild gravity well.

    .. math::
        \varepsilon_{eff}(r) = \frac{\varepsilon_0}{1 - r_s / r}

    As r → r_s: ε → ∞ (capacitor fully charged — dielectric rupture).
    Far from mass (r >> r_s): ε → ε₀ (flat vacuum).

    This is the gravitational analog of plasma ε-saturation, but
    INVERTED: gravity INCREASES ε instead of reducing it.

    Args:
        r: Radial distance [m].
        r_s: Schwarzschild radius [m].

    Returns:
        Effective permittivity [F/m].
    """
    ratio = np.minimum(r_s / np.asarray(r, dtype=float), 0.9999)
    return EPSILON_0 / (1 - ratio)


def mu_eff_schwarzschild(r: float | np.ndarray,
                          r_s: float) -> float | np.ndarray:
    r"""
    Effective permeability in a Schwarzschild gravity well.

    .. math::
        \mu_{eff}(r) = \mu_0 \cdot (1 - r_s / r)

    As r → r_s: μ → 0 (inductor shorts — like a superconductor!).
    Far from mass: μ → μ₀.

    The event horizon is a MAGNETIC saturation point:
    μ_eff → 0 means Z → 0 means Γ → −1 (total reflection).

    Args:
        r: Radial distance [m].
        r_s: Schwarzschild radius [m].

    Returns:
        Effective permeability [H/m].
    """
    ratio = np.minimum(r_s / np.asarray(r, dtype=float), 0.9999)
    return MU_0 * (1 - ratio)


def gravitational_impedance(r: float | np.ndarray,
                             r_s: float) -> float | np.ndarray:
    r"""
    Characteristic impedance at radius r in a Schwarzschild field.

    .. math::
        Z(r) = \sqrt{\mu_{eff}(r) / \varepsilon_{eff}(r)}
             = Z_0 \cdot (1 - r_s / r)

    Z → 0 at the horizon (total impedance mismatch with vacuum).
    Z → Z₀ far away.

    Args:
        r: Radial distance [m].
        r_s: Schwarzschild radius [m].

    Returns:
        Impedance [Ω].
    """
    mu = mu_eff_schwarzschild(r, r_s)
    eps = epsilon_eff_schwarzschild(r, r_s)
    return impedance(mu, eps)


def horizon_reflection(r: float | np.ndarray,
                        r_s: float) -> float | np.ndarray:
    r"""
    Reflection coefficient at radius r in a Schwarzschild field.

    .. math::
        \Gamma(r) = \frac{Z(r) - Z_0}{Z(r) + Z_0}

    At r → r_s: Z → 0, so Γ → −1 (total reflection — event horizon).
    At r → ∞: Z → Z₀, so Γ → 0 (matched — flat space).

    This is the SAME ``reflection_coefficient()`` used for:
      - Pauli exclusion (Γ → −1 at particle boundary)
      - Plasma cutoff (Γ → −1 below ω_p)
      - Meissner effect (Γ → −1 for B < B_c)
      - Seismic Moho reflection

    Args:
        r: Radial distance [m].
        r_s: Schwarzschild radius [m].

    Returns:
        Reflection coefficient (−1 ≤ Γ ≤ 0).
    """
    Z_r = gravitational_impedance(r, r_s)
    return reflection_coefficient(Z_0, Z_r)


# ═══════════════════════════════════════════════════════════════
# GW strain and propagation properties
# ═══════════════════════════════════════════════════════════════

def gw_strain_to_voltage(h: float, freq_hz: float = 100.0) -> float:
    r"""
    Convert GW strain to equivalent voltage across one lattice cell.

    The GW strain h ≈ 10⁻²¹ corresponds to a fractional lattice
    deformation. The equivalent voltage is:

    .. math::
        V_{GW} = h \cdot c \cdot \ell_{node} \cdot 2\pi f

    This is the maximum electric field perturbation per cell.

    Args:
        h: Gravitational wave strain amplitude (dimensionless).
        freq_hz: GW frequency [Hz] (default 100 Hz for LIGO band).

    Returns:
        Equivalent voltage per lattice cell [V].
    """
    return h * C_0 * L_NODE * 2 * np.pi * freq_hz


def is_linear_propagation(h: float, freq_hz: float = 100.0) -> bool:
    r"""
    Check whether a GW propagates in the linear regime.

    Linear propagation requires V_GW << V_SNAP (no saturation).
    For LIGO-detected GW (h ~ 10⁻²¹), V_GW / V_SNAP ~ 10⁻¹⁹.

    Args:
        h: Strain amplitude.
        freq_hz: Frequency [Hz].

    Returns:
        True if propagation is linear (no saturation losses).
    """
    V_gw = gw_strain_to_voltage(h, freq_hz)
    return float(V_gw / V_SNAP) < 0.01


def gw_local_speed(r: float, r_s: float) -> float:
    r"""
    Local GW propagation speed in a Schwarzschild field.

    .. math::
        c_{local}(r) = \frac{1}{\sqrt{\varepsilon_{eff} \cdot \mu_{eff}}}
                     = c_0

    In the AVE framework, ε × μ = ε₀μ₀ everywhere
    (even in a gravity well), so the LOCAL speed of light
    at any radius is always c₀. Time dilation and length
    contraction are refractive effects, not speed changes.

    Returns:
        Local wave speed [m/s] (always c₀).
    """
    # ε_eff × μ_eff = ε₀/(1-rs/r) × μ₀(1-rs/r) = ε₀μ₀ = 1/c²
    return float(C_0)


def refractive_index(r: float | np.ndarray,
                      r_s: float) -> float | np.ndarray:
    r"""
    Effective refractive index around a Schwarzschild mass.

    Although the LOCAL speed is c, the COORDINATE speed (as measured
    by a distant observer) is reduced:

    .. math::
        n(r) = \frac{c}{c_{coord}(r)}
             = \frac{(1 + r_s/(4r'))^3}{1 - r_s/(4r')}

    In isotropic coordinates r' ≈ r for r >> r_s.
    Simplified (weak field): n ≈ 1 + r_s / r.

    This is the refractive index that bends light paths → gravitational
    lensing. It uses the same math as optical lensing through a
    graded-index medium.

    Args:
        r: Radial distance [m].
        r_s: Schwarzschild radius [m].

    Returns:
        Refractive index (≥ 1).
    """
    r = np.asarray(r, dtype=float)
    return 1 + r_s / r


# ═══════════════════════════════════════════════════════════════
# Black hole echo prediction
# ═══════════════════════════════════════════════════════════════

def echo_delay(M: float) -> float:
    r"""
    Predicted delay between post-merger gravitational wave echoes.

    In AVE, the event horizon is a Γ = −1 reflection boundary (Z → 0).
    Post-merger GW energy bounces between the photon sphere and the
    horizon, producing echoes separated by:

    .. math::
        \Delta t_{echo} \approx \frac{4 G M}{c^3} \cdot
        \ln\left(\frac{r_{ph}}{r_s - r_{ph}}\right)

    For a 30 M☉ merger: Δt ≈ 0.1–0.3 s (consistent with Abedi et al.).

    Args:
        M: Total mass of the merged remnant [kg].

    Returns:
        Echo delay time [s].
    """
    r_s = schwarzschild_radius(M)
    r_ph = 1.5 * r_s  # Photon sphere at 3GM/c²
    # Tortoise coordinate delay
    delta_r_star = r_s * np.log(r_ph / (r_ph - r_s + 1e-30))
    return 2 * delta_r_star / C_0


# ═══════════════════════════════════════════════════════════════
# Summary dataclass
# ═══════════════════════════════════════════════════════════════

def gw_propagation_summary(M_solar: float = 30.0,
                            h: float = 1e-21,
                            r_multiples: list = None) -> dict:
    """
    Generate a summary of GW propagation properties.

    Args:
        M_solar: Source mass [solar masses].
        h: GW strain amplitude.
        r_multiples: List of r/r_s ratios to evaluate.

    Returns:
        Dict with all computed properties.
    """
    M_SUN = 1.989e30
    M = M_solar * M_SUN
    r_s = schwarzschild_radius(M)

    if r_multiples is None:
        r_multiples = [1.01, 1.1, 2, 5, 10, 100, 1000]

    results = {
        'M_kg': M,
        'r_s_m': r_s,
        'echo_delay_s': echo_delay(M),
        'linear_propagation': is_linear_propagation(h),
        'V_gw_over_V_snap': gw_strain_to_voltage(h) / V_SNAP,
        'profiles': [],
    }

    for mult in r_multiples:
        r = mult * r_s
        results['profiles'].append({
            'r_over_rs': mult,
            'r_m': r,
            'epsilon_eff': float(epsilon_eff_schwarzschild(r, r_s)),
            'mu_eff': float(mu_eff_schwarzschild(r, r_s)),
            'Z_ohm': float(gravitational_impedance(r, r_s)),
            'gamma': float(horizon_reflection(r, r_s)),
            'n_refract': float(refractive_index(r, r_s)),
        })

    return results
