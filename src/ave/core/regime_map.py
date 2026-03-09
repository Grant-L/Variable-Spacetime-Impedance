"""
Regime Map: Universal Classification of Operating Regimes
==========================================================

Every physical domain in AVE reduces to a single dimensionless control
parameter r = A/Ac, where A is the local amplitude and Ac is the critical
(yield) threshold. The saturation operator S(r) = √(1-r²) changes
character at well-defined boundaries, defining 4 universal regimes:

    Regime I   LINEAR      r < 0.1     S ≈ 1    Standard equations
    Regime II  NONLINEAR   0.1 ≤ r < 0.9  S < 1    Axiom 4 corrections
    Regime III YIELD       0.9 ≤ r < 1.0  S → 0    Phase transition
    Regime IV  RUPTURED    r ≥ 1.0        S = 0    Topology destroyed

This module provides:
    classify_regime(A, Ac) → RegimeInfo
    domain_control_parameter(domain, **kwargs) → (A, Ac, r)
    regime_summary(domain, **kwargs) → formatted string

The regime classification is the PREREQUISITE GATE: no domain
analysis should proceed without first identifying its regime.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ave.core.constants import (
    C_0, ALPHA, HBAR, M_E, e_charge,
    EPSILON_0, MU_0, Z_0, L_NODE,
    V_SNAP, V_YIELD,
    B_SNAP, NU_VAC,
)

# ══════════════════════════════════════════════════════════════════════════════
# Regime Boundaries (dimensionless r = A/Ac)
# ══════════════════════════════════════════════════════════════════════════════
R_LINEAR_MAX = 0.1       # r < 0.1: perturbation theory valid, S > 0.995
R_NONLINEAR_MAX = 0.9    # r < 0.9: S > 0.436, Axiom 4 corrections measurable
R_YIELD_MAX = 1.0        # r < 1.0: S → 0, phase transition zone

# Regime IDs
REGIME_LINEAR = 1
REGIME_NONLINEAR = 2
REGIME_YIELD = 3
REGIME_RUPTURED = 4

REGIME_NAMES = {
    REGIME_LINEAR: "I (LINEAR)",
    REGIME_NONLINEAR: "II (NONLINEAR)",
    REGIME_YIELD: "III (YIELD)",
    REGIME_RUPTURED: "IV (RUPTURED)",
}

REGIME_DESCRIPTIONS = {
    REGIME_LINEAR: "Standard equations; Axiom 4 corrections negligible (ΔS < 0.5%)",
    REGIME_NONLINEAR: "Full S(r) required; perturbative expansion breaks down",
    REGIME_YIELD: "Phase transition zone; c_eff → 0, topology approaching rupture",
    REGIME_RUPTURED: "Topology destroyed; deconfinement / event horizon interior",
}


@dataclass
class RegimeInfo:
    """Result of regime classification."""
    regime: int
    name: str
    description: str
    r: float                  # dimensionless ratio A/Ac
    S: float                  # saturation factor
    A: float                  # physical amplitude
    Ac: float                 # critical threshold
    domain: Optional[str] = None
    A_units: Optional[str] = None
    Ac_units: Optional[str] = None

    def __repr__(self):
        return (
            f"RegimeInfo(regime={self.name}, r={self.r:.6f}, "
            f"S={self.S:.6f}, A={self.A:.4e}, Ac={self.Ac:.4e})"
        )

    def summary(self) -> str:
        """Human-readable summary for diagnostic printing."""
        lines = [
            f"  ── REGIME CLASSIFICATION ──",
            f"  Domain:    {self.domain or 'unspecified'}",
            f"  Amplitude: A = {self.A:.4e}" + (f" {self.A_units}" if self.A_units else ""),
            f"  Threshold: Ac = {self.Ac:.4e}" + (f" {self.Ac_units}" if self.Ac_units else ""),
            f"  Ratio:     r = A/Ac = {self.r:.6f}",
            f"  Saturation: S(r) = {self.S:.6f}",
            f"  ▶ Regime:  {self.name}",
            f"  ▶ Physics: {self.description}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Core Classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_regime(A, Ac, domain=None, A_units=None, Ac_units=None):
    """
    Classify the operating regime from amplitude and critical threshold.

    Parameters
    ----------
    A : float
        Local amplitude (V, E, ε₁₁, T, B, h, etc.)
    Ac : float
        Critical threshold (V_yield, E_yield, 1.0, T_c, B_snap, etc.)
    domain : str, optional
        Domain name for diagnostic output.
    A_units, Ac_units : str, optional
        Physical units for diagnostic output.

    Returns
    -------
    RegimeInfo
        Complete regime classification with diagnostics.
    """
    r = abs(float(A)) / abs(float(Ac))
    S = np.sqrt(max(0.0, 1.0 - min(r, 1.0)**2))

    if r < R_LINEAR_MAX:
        regime = REGIME_LINEAR
    elif r < R_NONLINEAR_MAX:
        regime = REGIME_NONLINEAR
    elif r < R_YIELD_MAX:
        regime = REGIME_YIELD
    else:
        regime = REGIME_RUPTURED

    return RegimeInfo(
        regime=regime,
        name=REGIME_NAMES[regime],
        description=REGIME_DESCRIPTIONS[regime],
        r=r,
        S=S,
        A=float(A),
        Ac=float(Ac),
        domain=domain,
        A_units=A_units,
        Ac_units=Ac_units,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Domain-Specific Control Parameters
# ══════════════════════════════════════════════════════════════════════════════

def em_voltage_regime(V_local):
    """
    EM (dielectric) regime: r = V / V_yield.

    V_yield = √α × V_snap ≈ 43.65 kV is the kinetic onset of nonlinearity.
    Lab fields are typically deep in Regime I (r ~ 10⁻⁶ to 10⁻³).
    PONDER-05 at 30 kV operates at r = 0.687 (Regime II).
    """
    return classify_regime(
        V_local, float(V_YIELD),
        domain="EM (dielectric)",
        A_units="V", Ac_units="V",
    )


def em_field_regime(E_local):
    """
    EM (field strength) regime: r = E / E_yield.

    E_yield = V_yield / ℓ_node ≈ 1.13 × 10¹⁷ V/m.
    Lab fields (E ~ 10⁶ V/m) are in Regime I (r ~ 10⁻¹¹).
    """
    E_yield = float(V_YIELD) / float(L_NODE)  # ≈ 1.13 × 10¹⁷ V/m
    return classify_regime(
        E_local, E_yield,
        domain="EM (field)",
        A_units="V/m", Ac_units="V/m",
    )


def gravity_regime(M_kg, r_meters):
    """
    Gravitational regime: r = ε₁₁ = 7GM/(c²r).

    The principal radial strain of the lattice under Schwarzschild geometry.
    Solar surface: ε₁₁ ≈ 10⁻⁵ (Regime I).
    Neutron star: ε₁₁ ≈ 0.3 (Regime II).
    Black hole at r_s: ε₁₁ → 1 (Regime III/IV boundary).
    """
    G_Newton = 6.67430e-11  # m³/(kg·s²)
    epsilon_11 = 7.0 * G_Newton * M_kg / (C_0**2 * r_meters)
    return classify_regime(
        epsilon_11, 1.0,
        domain="Gravity",
        A_units="(strain)", Ac_units="(unitary)",
    )


def bcs_regime(T_kelvin, T_c_kelvin):
    """
    BCS/superconducting regime: r = T/Tc.

    B_c(T) = B_c0 × √(1-(T/Tc)²) — same saturation operator.
    Below Tc: superconducting (S > 0). At Tc: normal (S → 0).
    """
    return classify_regime(
        T_kelvin, T_c_kelvin,
        domain="BCS (superconductor)",
        A_units="K", Ac_units="K",
    )


def magnetic_regime(B_local):
    """
    Magnetic regime: r = B / B_snap.

    B_snap = m_e²c²/(eℏ) ≈ 1.89 × 10⁹ T.
    Lab magnets (B ~ 10 T): r ~ 10⁻⁸ (Regime I).
    Magnetar surface (B ~ 10¹⁰ T): r ~ 5 (Regime IV, ruptured).
    """
    return classify_regime(
        B_local, float(B_SNAP),
        domain="Magnetic",
        A_units="T", Ac_units="T",
    )


def nuclear_regime(r_separation, d_sat):
    """
    Nuclear regime: r = d_sat / r_separation.

    d_sat is the saturation radius (proton diameter, Slater radius, etc.)
    At r_separation = d_sat: r = 1 (Pauli wall, Regime IV boundary).
    """
    ratio = d_sat / r_separation if r_separation > 0 else float('inf')
    return classify_regime(
        ratio, 1.0,
        domain="Nuclear",
        A_units="(d_sat/r)", Ac_units="(unitary)",
    )


def gw_regime(h_strain):
    """
    Gravitational wave regime: r = h / h_yield.

    h_yield = √α ≈ 0.0854 (yield strain of the lattice).
    LIGO detections: h ~ 10⁻²¹ (Regime I, r ~ 10⁻²⁰).
    """
    h_yield = np.sqrt(ALPHA)
    return classify_regime(
        h_strain, h_yield,
        domain="GW strain",
        A_units="(strain)", Ac_units="(strain)",
    )


def protein_regime(d_bond, d_eq):
    """
    Protein backbone regime: r = |d - d_eq| / d_eq.

    d_eq is the equilibrium bond distance (e.g., 3.8 Å for Cα-Cα).
    Typical backbone fluctuations: r ~ 0.05 (Regime I).
    Unfolded: r ~ 0.3 (Regime II, nonlinear).
    """
    dr = abs(d_bond - d_eq)
    return classify_regime(
        dr, d_eq,
        domain="Protein backbone",
        A_units="Å", Ac_units="Å",
    )


def galactic_regime(g_newtonian, a_0=1.2e-10):
    """
    Galactic regime: r = g_N / a₀.

    a₀ ≈ 1.2 × 10⁻¹⁰ m/s² (MOND acceleration scale).
    Inner galaxy (g >> a₀): r >> 1 (Regime IV — but here "ruptured"
    means the Newtonian potential dominates; the saturation correction
    is negligible because the lattice is in the unsaturated deep interior).
    Outer galaxy (g ~ a₀): r ~ 1 (Regime III — rotation curve flattening).
    Far outer (g << a₀): r << 1 (Regime I — but this is the deep MOND limit).

    Note: The galactic regime has an INVERTED interpretation compared to
    others. High g_N means LESS saturation effect, not more. The
    nonlinearity appears at LOW accelerations where the lattice
    compliance gradient matters.
    """
    return classify_regime(
        g_newtonian, a_0,
        domain="Galactic rotation",
        A_units="m/s²", Ac_units="m/s²",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Regime-Specific Equation Forms
# ══════════════════════════════════════════════════════════════════════════════

def regime_equations(regime_id):
    """
    Return the simplified equation forms valid in each regime.

    Returns a dict of {quantity: (formula_str, approximation_note)}.
    """
    if regime_id == REGIME_LINEAR:
        return {
            "ε_eff": ("ε₀", "S ≈ 1, standard Maxwell"),
            "μ_eff": ("μ₀", "S ≈ 1, standard Maxwell"),
            "c_eff": ("c₀", "No wave speed modification"),
            "Z_eff": ("Z₀", "Impedance invariant"),
            "S(r)": ("1 - r²/2 + O(r⁴)", "Perturbative expansion valid"),
        }
    elif regime_id == REGIME_NONLINEAR:
        return {
            "ε_eff": ("ε₀ × √(1 - r²)", "Full operator required"),
            "μ_eff": ("μ₀ × √(1 - r²)", "Full operator required"),
            "c_eff": ("c₀ × (1 - r²)^(1/4)", "Measurable slowdown"),
            "Z_eff": ("Z₀ / (1 - r²)^(1/4)", "Impedance rises"),
            "S(r)": ("√(1 - r²)", "No simplification"),
        }
    elif regime_id == REGIME_YIELD:
        return {
            "ε_eff": ("→ 0", "Compliance destroyed"),
            "μ_eff": ("→ 0", "Inductance shorts"),
            "c_eff": ("→ 0", "Wave packet freezes"),
            "Z_eff": ("→ ∞ or 0", "Depends on symmetric/asymmetric saturation"),
            "S(r)": ("→ 0", "Phase transition imminent"),
        }
    elif regime_id == REGIME_RUPTURED:
        return {
            "ε_eff": ("0", "Topology destroyed"),
            "μ_eff": ("0", "Topology destroyed"),
            "c_eff": ("0", "No propagation inside ruptured zone"),
            "Z_eff": ("undefined", "New physics: deconfinement, event horizon"),
            "S(r)": ("0", "Fully ruptured"),
        }
    else:
        raise ValueError(f"Unknown regime: {regime_id}")


# ══════════════════════════════════════════════════════════════════════════════
# Comprehensive Summary
# ══════════════════════════════════════════════════════════════════════════════

def print_regime_map():
    """Print the full regime map with all domain examples."""
    print("=" * 78)
    print("  UNIVERSAL REGIME MAP")
    print("  S(r) = √(1 - r²),  r = A/Ac")
    print("=" * 78)

    print(f"\n  {'Regime':<20} {'r range':<16} {'S range':<16} {'Physics'}")
    print(f"  {'─'*72}")
    print(f"  {'I  LINEAR':<20} {'r < 0.1':<16} {'S > 0.995':<16} Standard equations")
    print(f"  {'II NONLINEAR':<20} {'0.1 ≤ r < 0.9':<16} {'0.436 < S < 0.995':<16} Axiom 4 active")
    print(f"  {'III YIELD':<20} {'0.9 ≤ r < 1.0':<16} {'0 < S < 0.436':<16} Phase transition")
    print(f"  {'IV RUPTURED':<20} {'r ≥ 1.0':<16} {'S = 0':<16} Topology destroyed")

    print(f"\n  ── DOMAIN EXAMPLES ──")

    # EM
    examples = [
        ("EM (dielectric)", [
            ("Lab 1kV/m capacitor", 1e3, float(V_YIELD), "V"),
            ("PONDER-05 @ 30kV", 30e3, float(V_YIELD), "V"),
            ("PONDER-05 @ 43kV", 43e3, float(V_YIELD), "V"),
        ]),
        ("Gravity", [
            ("Solar surface", 2.12e-6, 1.0, "strain"),
            ("White dwarf", 3.0e-4, 1.0, "strain"),
            ("Neutron star", 0.3, 1.0, "strain"),
            ("BH at r_s", 1.0, 1.0, "strain"),
        ]),
        ("Magnetic", [
            ("MRI scanner (3T)", 3.0, float(B_SNAP), "T"),
            ("LHC dipole (8T)", 8.0, float(B_SNAP), "T"),
            ("Magnetar (10¹⁰ T)", 1e10, float(B_SNAP), "T"),
        ]),
        ("GW strain", [
            ("LIGO detection", 1e-21, np.sqrt(ALPHA), "h"),
            ("NS merger surface", 0.01, np.sqrt(ALPHA), "h"),
        ]),
    ]

    for domain, items in examples:
        print(f"\n  {domain}:")
        for name, A, Ac, units in items:
            r = abs(A) / abs(Ac)
            S = np.sqrt(max(0.0, 1.0 - min(r, 1.0)**2))
            regime_id = (REGIME_LINEAR if r < 0.1 else
                        REGIME_NONLINEAR if r < 0.9 else
                        REGIME_YIELD if r < 1.0 else
                        REGIME_RUPTURED)
            print(f"    {name:<28s} r = {r:.2e}  S = {S:.6f}  → {REGIME_NAMES[regime_id]}")

    print(f"\n  {'='*72}")


if __name__ == "__main__":
    print_regime_map()
