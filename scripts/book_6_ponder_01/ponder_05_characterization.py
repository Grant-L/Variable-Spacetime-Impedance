#!/usr/bin/env python3
r"""
PONDER-05: DC-Biased Quartz Drive in the Nonlinear Tensor Regime
===================================================================

V_yield = √α × V_snap ≈ 43.65 kV is the kinetic yield threshold.
At 30 kV DC bias = 68.7% of V_yield → deep nonlinear regime.

This script computes:
  1. ε_eff(V_DC) curve showing dielectric saturation
  2. Linear thrust (standard electrostatics, F = ½ε₀∇E²)
  3. Nonlinear thrust excess from Axiom 4 saturation
  4. Acoustic steepening factor (shockwave onset)
  5. Cross-term amplification from DC bias + AC perturbation

Usage:
    PYTHONPATH=src python scripts/book_6_ponder_01/ponder_05_characterization.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from ave.core.constants import (
    C_0, EPSILON_0, MU_0, ALPHA, V_SNAP, E_YIELD_KINETIC, e_charge, M_E,
    L_NODE
)

# ====================================================================
# CONSTANTS
# ====================================================================

V_YIELD = np.sqrt(ALPHA) * V_SNAP   # √α × 511 kV ≈ 43.65 kV

# Quartz parameters
EPS_R_QUARTZ = 4.5
V_SOUND_QUARTZ = 5960   # m/s
RHO_QUARTZ = 2650       # kg/m³

# Geometry (25mm radius × 50mm quartz cylinder)
R_CYL = 0.025
L_CYL = 0.050
A_CYL = np.pi * R_CYL**2      # ~1.96e-3 m²
VOL_CYL = A_CYL * L_CYL       # ~9.82e-5 m³

# AC drive (50 kHz piezo transducer)
F_AC = 50_000
V_AC = 500   # Volts peak


def eps_eff_ratio(V, v_yield=V_YIELD):
    """ε_eff/ε₀ = √(1 - (V/V_yield)²)"""
    s = np.clip(V / v_yield, 0, 0.9999)
    return np.sqrt(1.0 - s**2)


def energy_density(V, gap, eps_r=1.0, v_yield=V_YIELD):
    """
    Energy density inside a dielectric gap under voltage V.

    The electric field inside the gap: E = V / gap
    The STANDARD energy density: u_lin = ½ε₀ε_r E²
    The AVE NONLINEAR correction: ε_eff drops, so
      u_AVE = ½ε₀ε_r × ε_eff(V)/ε₀ × E²

    This means the nonlinear ε REDUCES the stored energy density
    (the vacuum can't hold as much charge). The force arises from
    the GRADIENT of this reduced capacity.
    """
    E = V / gap
    eps_ratio = eps_eff_ratio(V, v_yield)
    u_linear = 0.5 * EPSILON_0 * eps_r * E**2
    u_ave = u_linear * eps_ratio
    return u_linear, u_ave


def compute_dc_bias_thrust(V_dc, V_ac, gap, area, eps_r=1.0, v_yield=V_YIELD):
    """
    Compute the net asymmetric force from a DC-biased AC drive.

    When ε is LINEAR: the force from the +swing and -swing cancel exactly
    (symmetric around V_dc). No net DC thrust.

    When ε is NONLINEAR (Axiom 4): the +swing (toward V_yield) diminishes
    ε MORE than the -swing restores it. This asymmetry creates a net DC
    rectified force.

    The cross-term amplification factor:
      Standard: F ∝ V_ac²  (tiny at 500V)
      DC-biased: F ∝ 2 × V_dc × V_ac  (60× amplification at 30kV/500V)
      Nonlinear: F includes ε'(V_dc) curvature → further amplification

    Returns dict with linear/nonlinear force and amplification.
    """
    E_dc = V_dc / gap
    E_ac = V_ac / gap

    # ── Linear forces (standard Maxwell) ──
    # Standard cross-term: F_lin = ε₀ε_r × E_dc × E_ac × A
    # This is the 2E·δE cross term → 60× amplification over pure AC
    F_lin_cross = EPSILON_0 * eps_r * E_dc * E_ac * area
    F_lin_ac_only = 0.5 * EPSILON_0 * eps_r * E_ac**2 * area
    amplification_linear = F_lin_cross / F_lin_ac_only if F_lin_ac_only > 0 else 0

    # ── Nonlinear force (Axiom 4) ──
    # Energy density at V_dc ± V_ac
    V_plus = V_dc + V_ac
    V_minus = max(V_dc - V_ac, 0)

    _, u_plus = energy_density(V_plus, gap, eps_r, v_yield)
    _, u_minus = energy_density(V_minus, gap, eps_r, v_yield)
    _, u_dc = energy_density(V_dc, gap, eps_r, v_yield)

    # Linear energy at same voltages (for comparison)
    u_plus_lin, _ = energy_density(V_plus, gap, eps_r, v_yield)
    u_minus_lin, _ = energy_density(V_minus, gap, eps_r, v_yield)

    # Net asymmetric residual (nonlinear)
    du_nl = u_plus - u_minus

    # Net asymmetric residual (linear: always = 2ε₀ε_r × E_dc × E_ac / gap)
    du_lin = u_plus_lin - u_minus_lin

    # The NONLINEAR EXCESS: additional force from ε saturation
    du_excess = du_nl - du_lin

    # Force = pressure × area
    F_nl_total = du_nl * area
    F_nl_excess = du_excess * area

    # The curvature amplification: how much steeper the ε curve makes the swing
    eps_plus = eps_eff_ratio(V_plus, v_yield)
    eps_minus = eps_eff_ratio(V_minus, v_yield)
    curvature_asymmetry = abs(eps_plus - eps_minus)

    return {
        'F_lin_cross': F_lin_cross,
        'F_lin_ac_only': F_lin_ac_only,
        'amplification_linear': amplification_linear,
        'F_nl_total': F_nl_total,
        'F_nl_excess': F_nl_excess,
        'du_nl': du_nl,
        'du_lin': du_lin,
        'curvature_asymmetry': curvature_asymmetry,
        'eps_plus': eps_plus,
        'eps_minus': eps_minus,
    }


def main():
    print("=" * 75)
    print("  PONDER-05: DC-Biased Quartz — Nonlinear Tensor Regime")
    print("=" * 75)

    print(f"\n  V_SNAP (absolute):     {V_SNAP/1e3:.2f} kV")
    print(f"  V_YIELD (kinetic, √α): {V_YIELD/1e3:.3f} kV")
    print(f"  30 kV / V_YIELD:       {30e3/V_YIELD*100:.1f}% → DEEP NONLINEAR")

    # ─────────────────────────────────────────────
    # 1. ε_eff Saturation Curve
    # ─────────────────────────────────────────────
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  ε_eff(V) / ε₀   Axiom 4 Dielectric Saturation │")
    print(f"  ├──────────┬──────────┬───────────┬───────────────┤")
    print(f"  │ V_DC (kV)│ V/V_yield│  ε_eff/ε₀ │  Drop from ε₀ │")
    print(f"  ├──────────┼──────────┼───────────┼───────────────┤")
    for v_kv in [0, 5, 10, 15, 20, 25, 30, 35, 40, 43]:
        v = v_kv * 1e3
        r = eps_eff_ratio(v)
        d = (1 - r) * 100
        print(f"  │ {v_kv:>7.0f}  │  {v/V_YIELD:>7.3f}  │  {r:>8.6f} │    {d:>7.1f}%    │")
    print(f"  └──────────┴──────────┴───────────┴───────────────┘")

    # ─────────────────────────────────────────────
    # 2. DC Bias Cross-Term Amplification
    # ─────────────────────────────────────────────
    gap = L_CYL  # 50mm quartz cylinder
    print(f"\n  DC Bias Sweep: 500V AC @ 50 kHz riding on V_DC")
    print(f"  Gap = {gap*1e3:.0f} mm quartz (ε_r = {EPS_R_QUARTZ}), Area = {A_CYL*1e4:.2f} cm²")
    print(f"")
    print(f"  {'V_DC':>6} {'Cross F':>12} {'AC-only F':>12} {'Amplif':>8} {'NL excess':>12} {'ε asym':>8}")
    print(f"  {'(kV)':>6} {'(μN)':>12} {'(μN)':>12} {'':>8} {'(μN)':>12}")
    print(f"  {'─'*65}")

    for v_kv in [0, 1, 5, 10, 15, 20, 25, 28, 30, 32, 35, 38, 40, 42, 43]:
        v = v_kv * 1e3
        r = compute_dc_bias_thrust(v, V_AC, gap, A_CYL, EPS_R_QUARTZ)
        print(f"  {v_kv:>5.0f}  {r['F_lin_cross']*1e6:>11.4f}  {r['F_lin_ac_only']*1e6:>11.6f}  "
              f"{r['amplification_linear']:>7.1f}×  {r['F_nl_excess']*1e6:>11.6f}  "
              f"{r['curvature_asymmetry']:>7.4f}")

    # ─────────────────────────────────────────────
    # 3. Acoustic Steepening
    # ─────────────────────────────────────────────
    print(f"\n  Acoustic Steepening (wave deformation → shockwave onset)")
    print(f"  {'V_DC':>6} {'c_eff(peak)':>14} {'c_eff(trough)':>14} {'Steep':>8}")
    print(f"  {'(kV)':>6} {'(m/s)':>14} {'(m/s)':>14}")
    print(f"  {'─'*48}")

    for v_kv in [0, 10, 20, 25, 30, 35, 40, 42, 43]:
        V_p = v_kv * 1e3 + V_AC
        V_t = max(v_kv * 1e3 - V_AC, 0)
        c_p = C_0 * np.sqrt(eps_eff_ratio(V_p))
        c_t = C_0 * np.sqrt(eps_eff_ratio(V_t))
        steep = c_t / c_p if c_p > 0 else float('inf')
        print(f"  {v_kv:>5.0f}  {c_p:>13.4e}  {c_t:>13.4e}  {steep:>7.4f}×")

    # ─────────────────────────────────────────────
    # 4. PONDER-02 comparison: 2.45 GHz + sapphire
    # ─────────────────────────────────────────────
    print(f"\n  ═══════════════════════════════════════════════════════════════════════")
    print(f"  PONDER-02 vs PONDER-05 COMPARISON")
    print(f"  ═══════════════════════════════════════════════════════════════════════")

    # PONDER-02: BaTiO₃ at 2.45 GHz, 25kV, sapphire GRIN nozzle
    eps_r_batio3 = 3000
    f_p02 = 2.45e9
    v_p02 = 25_000
    gap_p02 = 0.0001  # 100 μm BaTiO₃ wafer
    a_p02 = 0.001 * 0.001  # 1mm × 1mm tip area (per emitter)
    n_tips = 100  # 100 emitter tips

    E_p02 = v_p02 / gap_p02
    u_p02_lin = 0.5 * EPSILON_0 * eps_r_batio3 * E_p02**2
    F_p02_per_tip = u_p02_lin * a_p02
    F_p02_total = F_p02_per_tip * n_tips

    # Frequency scaling: F ∝ f² (from temporal gradient rectification)
    f_scale = (f_p02 / F_AC)**2

    r05_30 = compute_dc_bias_thrust(30e3, V_AC, gap, A_CYL, EPS_R_QUARTZ)

    print(f"  PONDER-05 (50 kHz quartz, 30kV DC bias):")
    print(f"    Linear cross-term:   {r05_30['F_lin_cross']*1e6:.4f} μN")
    print(f"    NL excess:           {r05_30['F_nl_excess']*1e6:.6f} μN")
    print(f"    DC amplification:    {r05_30['amplification_linear']:.1f}×")
    print(f"")
    print(f"  PONDER-02 (2.45 GHz BaTiO₃, 25kV, 10mm sapphire nozzle):")
    print(f"    E_internal:          {E_p02:.2e} V/m")
    print(f"    u_stored:            {u_p02_lin:.2e} J/m³")
    print(f"    F per tip:           {F_p02_per_tip*1e3:.4f} mN")
    print(f"    F total ({n_tips} tips):  {F_p02_total:.4f} N ({F_p02_total/9.81*1000:.2f} g)")
    print(f"    Freq scaling f²:     {f_scale:.2e}×")

    # ─────────────────────────────────────────────
    # 5. Key Results
    # ─────────────────────────────────────────────
    print(f"\n  ═══════════════════════════════════════════════════════════════════════")
    print(f"  PONDER-05 KEY RESULTS")
    print(f"  ═══════════════════════════════════════════════════════════════════════")
    print(f"  At 30 kV DC bias ({30e3/V_YIELD*100:.1f}% of V_yield = {V_YIELD/1e3:.2f} kV):")
    print(f"    1. ε_eff drops by {(1 - eps_eff_ratio(30e3))*100:.1f}%")
    print(f"    2. DC cross-term amplification: {r05_30['amplification_linear']:.0f}×")
    print(f"    3. Linear ponderomotive force: {r05_30['F_lin_cross']*1e6:.2f} μN")
    print(f"    4. Nonlinear ε asymmetry: {r05_30['curvature_asymmetry']:.4f}")
    print(f"    5. Acoustic steepening at 43 kV: approaching shockwave")
    print(f"")
    print(f"  The DC bias achieves TWO things:")
    print(f"    a) Linear amplification: 2×E_dc×E_ac >> E_ac² → {r05_30['amplification_linear']:.0f}× more signal")
    print(f"    b) Nonlinear regime: ε_eff curvature creates asymmetric rectification")
    print(f"       that standard Maxwell cannot predict (new physics signal)")
    print(f"  ═══════════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
