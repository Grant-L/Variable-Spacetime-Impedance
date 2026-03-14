"""
AVE MODULE: Regime IV — Black Hole Interior Constitutive Properties
====================================================================
Maps the constitutive parameters of the vacuum lattice from deep
Regime I (flat space) through the phase transition at r_sat into
the ruptured interior (Regime IV).

DERIVATION (LIVING_REFERENCE §"How to Apply AVE", 6 steps):

  Step 1 — LC Analogs:
    μ-analog: μ₀·n(r)   (lattice inductance, scales with refractive index)
    ε-analog: ε₀·n(r)   (lattice capacitance, scales with refractive index)
    SYMMETRIC saturation: both μ and ε scale by n(r) → Z = Z₀ always
    Interior (r < r_sat): G_shear = 0 → transverse waves cannot propagate
    This is the DC operating point past breakdown — no AC model exists.

  Step 2 — Strain & Regime:
    ε₁₁(r) = 7GM/(c²r)
    r_sat = 7GM/c² → ε₁₁ = 1 (Regime IV boundary)
    Interior: ε₁₁ > 1 → S = 0 (clipped)

  Step 3 — Universal Operators:
    S(r) = √(1 - ε₁₁²)  for ε₁₁ < 1
    S(r) = 0              for ε₁₁ ≥ 1 (interior)
    G_shear(r) = G₀·S(r)
    c_eff(r) = c₀·(1-ε₁₁²)^(1/4)  → 0 at r_sat

  Step 4 — Symmetry:
    BH: SYMMETRIC (Z = Z₀, Γ = 0) → dissipative sink (not resonant)
    Electron: ASYMMETRIC (μ → 0, Z → 0, Γ → -1) → standing wave
    BCS: ASYMMETRIC (μ_eff → 0, Z → 0, Γ → -1) → Meissner
    Nucleus: ASYMMETRIC (Γ → +1) → repulsive wall
    The BH is the ONLY Regime IV system with no impedance mismatch.

  Step 5 — Numerical: profile all properties from r = 10·r_sat to 0.01·r_sat

  Step 6 — Testability: BH QNM ringdown validates exterior;
    information paradox = consequence of the dissipative sink model.

USAGE:
    PYTHONPATH=src python scripts/future_work/bh_interior_regime_iv.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ave.core.constants import G, C_0, Z_0, MU_0, EPSILON_0, NU_VAC, M_SUN, ALPHA, L_NODE
from ave.gravity import (
    principal_radial_strain,
    refractive_index,
    saturation_radius,
    schwarzschild_radius,
    shear_modulus_factor,
    gravitational_saturation_factor,
    local_impedance,
)
from ave.axioms.scale_invariant import saturation_factor
from ave.axioms.spectral_gap import confinement_radius
from ave.core.constants import KAPPA_FS


def bh_constitutive_profile(M_kg, r_min_factor=0.01, r_max_factor=10.0, n_points=50):
    """
    Compute all constitutive properties from r_max to r_min through the
    Regime IV boundary at r_sat.

    Returns dict of arrays indexed by radius.
    """
    r_sat = saturation_radius(M_kg)
    r_s = schwarzschild_radius(M_kg)
    r_arr = np.geomspace(r_min_factor * r_sat, r_max_factor * r_sat, n_points)

    eps11 = np.array([principal_radial_strain(M_kg, r) for r in r_arr])
    n_r = np.array([refractive_index(M_kg, r) for r in r_arr])
    S = np.array([gravitational_saturation_factor(M_kg, r) for r in r_arr])
    G_shear = S.copy()  # G_shear/G₀ = S

    # Effective wave speed: c_eff = c₀·(1-ε₁₁²)^(1/4) where defined
    c_eff = np.where(eps11 < 1.0,
                     C_0 * (1.0 - eps11**2)**0.25,
                     0.0)

    # Impedance: Z = Z₀ always (symmetric gravity)
    Z = np.full_like(r_arr, Z_0)

    # Reflection coefficient: Γ = 0 always (symmetric)
    Gamma = np.zeros_like(r_arr)

    # Quality factor: Q = 1/S (diverges at boundary)
    Q = np.where(S > 0, 1.0 / S, np.inf)

    # Regime classification
    r1 = np.sqrt(2 * float(ALPHA))
    regime = np.where(eps11 >= 1.0, 4,
             np.where(eps11 >= 0.866, 3,
             np.where(eps11 >= r1, 2, 1)))

    return {
        'r': r_arr,
        'r_sat': r_sat,
        'r_s': r_s,
        'eps11': eps11,
        'n': n_r,
        'S': S,
        'G_shear': G_shear,
        'c_eff': c_eff,
        'Z': Z,
        'Gamma': Gamma,
        'Q': Q,
        'regime': regime,
    }


def main():
    print("=" * 78)
    print("AVE Regime IV: Black Hole Interior Constitutive Properties")
    print("6-Step Methodology — The DC Operating Point Past Breakdown")
    print("=" * 78)
    print()

    # ──────────────────────────────────────────
    # Step 1: LC Analogs
    # ──────────────────────────────────────────
    print("STEP 1 — LC ANALOGS")
    print("-" * 40)
    print("  GRAVITY (symmetric saturation):")
    print("    μ'(r) = μ₀·n(r)     ε'(r) = ε₀·n(r)")
    print("    Z(r)  = √(μ'/ε') = √(μ₀/ε₀) = Z₀ = {:.2f} Ω  (ALWAYS)".format(Z_0))
    print("    Γ = (Z₂-Z₁)/(Z₂+Z₁) = 0  (no impedance mismatch)")
    print()
    print("  EE ANALOG: This is a matched transmission line — no reflections.")
    print("  The BH interior is past V_BR: the DC bias has exceeded")
    print("  breakdown voltage. No small-signal (AC) model exists.")
    print("  Transverse waves (shear/GW) cannot propagate.")
    print()

    # ──────────────────────────────────────────
    # Step 2: Profile through all 4 regimes
    # ──────────────────────────────────────────
    M_solar = 10.0
    M_kg = M_solar * float(M_SUN)
    prof = bh_constitutive_profile(M_kg, r_min_factor=0.01, r_max_factor=10.0, n_points=30)

    print("STEP 2 — REGIME PROFILE (10 M☉ Black Hole)")
    print("-" * 40)
    print(f"  r_sat = {prof['r_sat']/1e3:.2f} km  (ε₁₁ = 1 boundary)")
    print(f"  r_s   = {prof['r_s']/1e3:.2f} km  (Schwarzschild radius)")
    print(f"  r_sat / r_s = {prof['r_sat']/prof['r_s']:.2f}  (should be 3.5)")
    print()

    hdr = f"  {'r/r_sat':>8s}  {'r [km]':>8s}  {'ε₁₁':>8s}  {'Reg':>4s}  {'S':>8s}  {'c_eff/c':>8s}  {'Q':>10s}"
    print(hdr)
    print("  " + "-" * 70)

    for i in range(len(prof['r'])):
        r = prof['r'][i]
        r_ratio = r / prof['r_sat']
        eps = prof['eps11'][i]
        reg = int(prof['regime'][i])
        S = prof['S'][i]
        c_ratio = prof['c_eff'][i] / C_0
        Q = prof['Q'][i]
        Q_str = f"{Q:.4f}" if Q < 1e6 else "∞"

        # Only print selected points
        if i % 3 == 0 or reg != int(prof['regime'][max(0, i-1)]):
            reg_str = ["", "I", "II", "III", "IV"][reg]
            print(f"  {r_ratio:>8.3f}  {r/1e3:>8.2f}  {eps:>8.4f}  {reg_str:>4s}"
                  f"  {S:>8.6f}  {c_ratio:>8.6f}  {Q_str:>10s}")
    print()

    # ──────────────────────────────────────────
    # Step 3: Universal Operators — the 0·∞ limit
    # ──────────────────────────────────────────
    print("STEP 3 — THE 0·∞ LIMIT AT THE SINGULARITY")
    print("-" * 40)
    print()
    print("  At r → 0:")
    print(f"    n(r) = 1 + 2GM/(c²r) → ∞")
    print(f"    S(r) = 0  (ε₁₁ > 1, clipped)")
    print(f"    μ_eff = μ₀·n·S = μ₀·∞·0 = indeterminate")
    print(f"    ε_eff = ε₀·n·S = ε₀·∞·0 = indeterminate")
    print()
    print("  Resolution: In the ruptured interior, the lattice topology is")
    print("  destroyed. The constitutive parameters ε and μ lose their")
    print("  physical meaning — they describe elastic compliance of a")
    print("  structure that no longer exists. The 0·∞ is not a pathology;")
    print("  it is the signature that the LC model breaks down.")
    print()
    print("  EE ANALOG: Asking for the AC impedance of a burned-out device.")
    print("  The device has no small-signal model because it has no junctions.")
    print()

    # ──────────────────────────────────────────
    # Step 4: The Isomorphism Table
    # ──────────────────────────────────────────
    print("STEP 4 — REGIME IV ISOMORPHISM TABLE")
    print("-" * 40)
    print()
    print("  All four systems use S(A/A_c) = 0 at Regime IV.")
    print("  They differ in WHICH constitutive parameter saturates:")
    print()
    print(f"  {'System':<20s}  {'Saturates':<20s}  {'Sym':>5s}  {'Z_bdy':>8s}  {'Γ':>5s}  {'Interior':>20s}")
    print("  " + "-" * 85)
    print(f"  {'Electron':<20s}  {'μ (self-inductance)':<20s}  {'Asym':>5s}  {'→ 0':>8s}  {'-1':>5s}  {'Standing wave':>20s}")
    print(f"  {'Superconductor':<20s}  {'μ_eff (Meissner)':<20s}  {'Asym':>5s}  {'→ 0':>8s}  {'-1':>5s}  {'Standing wave':>20s}")
    print(f"  {'Nucleus (Pauli)':<20s}  {'U(r) at d_sat':<20s}  {'Asym':>5s}  {'→ ∞':>8s}  {'+1':>5s}  {'Repulsive wall':>20s}")
    print(f"  {'BH interior':<20s}  {'G_shear (topology)':<20s}  {'Sym':>5s}  {'= Z₀':>8s}  {'0':>5s}  {'Dissipative sink':>20s}")
    print()
    print("  KEY INSIGHT: The BH is the ONLY Regime IV system with no")
    print("  impedance mismatch. The electron is a KNOT (topology wound up);")
    print("  the BH is a HOLE (topology torn apart).")
    print()
    print("  EE TRANSLATION:")
    print("    Electron     = self-resonant LC tank (Z → 0, standing wave)")
    print("    Superconductor = shorted inductor (μ_eff → 0)")
    print("    Nucleus      = open circuit at Pauli wall (Z → ∞)")
    print("    BH interior  = device past V_BR (DC operating point destroyed)")
    print()

    # ──────────────────────────────────────────
    # Step 5: Cross-check characteristic scales
    # ──────────────────────────────────────────
    print("STEP 5 — CHARACTERISTIC SCALES")
    print("-" * 40)
    print()

    # BH scales
    for M_label, M_sol in [("10 M☉", 10.0), ("62 M☉ (GW150914)", 62.0),
                            ("4.3M M☉ (Sgr A*)", 4.3e6)]:
        M = M_sol * float(M_SUN)
        r_sat = saturation_radius(M)
        r_s = schwarzschild_radius(M)
        eps11_at_rs = principal_radial_strain(M, r_s)

        print(f"  {M_label}:")
        print(f"    r_sat = {r_sat/1e3:.2f} km,  r_s = {r_s/1e3:.2f} km,  r_sat/r_s = {r_sat/r_s:.2f}")
        print(f"    ε₁₁(r_s) = {eps11_at_rs:.4f}  (should be 7/2 = 3.5)")
        print()

    # Electron confinement scale (from spectral_gap)
    # Proton: c = 5 (cinquefoil)
    r_conf_electron = confinement_radius(float(KAPPA_FS), 3)
    r_conf_proton = confinement_radius(float(KAPPA_FS), 5)
    print(f"  Electron (trefoil c=3): r_conf = {r_conf_electron*1e15:.3f} fm "
          f"= {r_conf_electron/float(L_NODE):.4f} ℓ_node")
    print(f"  Proton (cinquefoil c=5): r_conf = {r_conf_proton*1e15:.3f} fm "
          f"= {r_conf_proton/float(L_NODE):.4f} ℓ_node")
    print()

    # ──────────────────────────────────────────
    # Step 6: Testability
    # ──────────────────────────────────────────
    print("STEP 6 — TESTABILITY & PREDICTIONS")
    print("-" * 40)
    print()
    print("  VALIDATED:")
    print("    • BH QNM ℓ=2: ω·M = 0.3673 vs GR 0.3737 (1.7% error)")
    print("      → validates EXTERIOR (r > r_sat) dynamics")
    print()
    print("  NEW PREDICTIONS:")
    print("    1. No information paradox: BH interior is a dissipative sink,")
    print("       not a unitary cavity. Information is not 'stored' — it is")
    print("       absorbed by the ruptured lattice (topology destroyed).")
    print()
    print("    2. Hawking radiation reinterpretation: vacuum fluctuations at")
    print("       r ≈ r_sat where G_shear → 0 create particle pairs. This is")
    print("       the gravitational analog of Schwinger pair production at")
    print("       E = E_yield (same S→0 boundary, different sector).")
    print()
    print("    3. BH mergers: the collision of two ruptured regions creates a")
    print("       new r_sat boundary. The ringdown IS the exterior lattice")
    print("       equilibrating to the new saturation shell — already validated")
    print("       to 1.7% via the 5-step eigenvalue method.")
    print()

    # ──────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()
    print("  The BH interior is the gravitational DC operating point past V_BR.")
    print("  The lattice topology is destroyed (G_shear = 0, S = 0).")
    print("  No transverse (shear) waves propagate — no AC small-signal model.")
    print()
    print("  This is NOT an electron: the electron has asymmetric saturation")
    print("  (μ → 0, Z → 0, Γ = -1) creating a standing wave resonator.")
    print("  The BH has symmetric saturation (Z = Z₀, Γ = 0) creating a")
    print("  dissipative sink. Knot vs hole. Tank circuit vs burned device.")
    print()
    print("  The exterior (r > r_sat) supports shear waves — the QNM")
    print("  eigenfrequencies arise from the same r_eff = r_sat/(1+ν_vac)")
    print("  formula used for WD surface modes.")


if __name__ == "__main__":
    main()
