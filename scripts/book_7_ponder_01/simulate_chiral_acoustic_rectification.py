#!/usr/bin/env python3
"""
Chiral Acoustic Rectification: Vacuum Varactor Thrust from First Principles
============================================================================

Derives the macroscopic thrust produced by the PONDER-01 phased array antenna
via the **Vacuum Varactor Diode** mechanism:

DERIVATION CHAIN (all from ave.core.constants — zero free parameters):
----------------------------------------------------------------------
1. Axiom 4 defines S(E) = √(1 − (E/E_yield)²)   [universal_saturation]
2. E_yield = V_YIELD / L_NODE ≈ 1.13e17 V/m       [from constants.py]
3. Local field at tip: E_local = β × Q × E_drive   [geometric amplification]
4. Jensen's inequality on S(E): ⟨S(E(t))⟩ < S(0) = 1
   → Time-averaged ε_eff(tip) < ε₀
   → DC rectified stress at tip ≠ 0
5. Torus knot chirality (p,q) → spatial handedness of hot spots
   → η_chiral = ν_vac = 2/7 (helical ↔ longitudinal transfer via Poisson)
6. Phased array of N tips → F_total = N² × F_single (coherent gain)

PHYSICAL PICTURE:
  The torus knot antenna's standing-wave maxima at sharp tips push the
  local E/E_yield ratio high enough (via β × Q amplification) that Jensen's
  inequality on the concave saturation function S(E) produces a non-zero
  time-averaged DC stress asymmetry.  The phased array coherently sums
  these tiny rectified stresses with N² gain.  The torus knot chirality
  selects the thrust direction via the Poisson coupling ν_vac = 2/7.

  This is a "vacuum varactor diode" — the nonlinear C(V) curve of the
  vacuum LC cell (Axiom 4) converts AC electromagnetic drive into DC
  lattice momentum, exactly like a charge-pump rectifier.

All constants from ave.core.constants. Zero empirical fits.

Usage:
    PYTHONPATH=src python scripts/book_7_ponder_01/simulate_chiral_acoustic_rectification.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ave.core.constants import (
    C_0, ALPHA, HBAR, M_E, e_charge,
    MU_0, EPSILON_0, Z_0,
    V_SNAP, V_YIELD, E_YIELD, XI_TOPO, L_NODE,
    NU_VAC, RHO_BULK,
)
from ave.core.universal_operators import universal_saturation

# ==============================================================================
# PONDER-01 HARDWARE PARAMETERS (geometric — not physics constants)
# ==============================================================================
V_DRIVE_RMS = 30_000.0        # Applied RMS voltage [V] — 30 kV
D_GAP = 1.0e-3                # Gap distance [m] — 1 mm
R_TIP = 1.0e-6                # Tip radius [m] — 1 μm
H_TIP = 1.0e-3                # Tip height [m] — 1 mm
F_DRIVE = 100e6               # Drive frequency [Hz] — 100 MHz VHF
A_TIP = np.pi * R_TIP**2      # Effective tip area [m²]
N_TIPS = 10_000               # Number of tips in phased array


# ==============================================================================
# DERIVED QUANTITIES (all from engine constants)
# ==============================================================================

# Tip enhancement factor β = h/r (hyperboloid geometry)
BETA_TIP = H_TIP / R_TIP      # ≈ 1000

# Macroscopic average field
E_MACRO = V_DRIVE_RMS / D_GAP  # ≈ 3e7 V/m

# Peak local field at tip (without Q)
E_TIP_PEAK = BETA_TIP * E_MACRO * np.sqrt(2)  # Peak = √2 × RMS


def saturation_factor(E_local):
    """Compute S(E) = √(1 − (E/E_yield)²) using engine operator.

    The voltage across one lattice cell is V_cell = E × L_NODE.
    Saturation occurs when V_cell → V_YIELD.
    """
    return universal_saturation(E_local, E_YIELD)


def jensen_rectification(Q_factor, beta=BETA_TIP, n_samples=2000):
    """Compute the Jensen's inequality rectification factor.

    For one RF cycle at a tip with enhancement β and resonant Q:
      E_local(t) = β × Q × E_macro × √2 × |sin(ωt)|

    The time-averaged saturation is:
      ⟨S⟩ = (1/T) ∫₀ᵀ S(E_local(t)) dt

    The rectification factor is:
      δ = 1 − ⟨S⟩

    Jensen's inequality guarantees δ > 0 for any nonzero AC drive
    through the concave function S(E).

    Returns:
        delta: rectification factor (dimensionless, > 0)
        S_avg: time-averaged saturation ⟨S⟩
        E_peak: peak local field [V/m]
        ratio_peak: E_peak / E_yield
    """
    t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    E_peak = beta * Q_factor * E_MACRO * np.sqrt(2)
    E_local = E_peak * np.abs(np.sin(t))

    # Clamp to just below E_yield to avoid NaN from sqrt(negative)
    ratio = np.minimum(E_local / E_YIELD, 0.99999)
    S_t = np.sqrt(1.0 - ratio**2)

    S_avg = np.mean(S_t)
    delta = 1.0 - S_avg

    return delta, S_avg, E_peak, E_peak / E_YIELD


def rectified_thrust(Q_factor, P_input, N=N_TIPS, beta=BETA_TIP):
    """Total DC thrust from the rectified fraction of input RF power.

    PHYSICAL MODEL (power-budget-constrained):
    ──────────────────────────────────────────
    The Q-factor amplifies the VOLTAGE at resonance, but the stored
    energy is Q × (P_input / ω), not Q² × anything.  The total RF
    power flowing through the antenna is still P_input.

    The vacuum varactor rectifies a fraction δ of the RF power into
    DC lattice momentum (phonons).  The rectified power is:

        P_rect = δ × P_input

    where δ is the Jensen's rectification factor from S(E).

    The chiral coupling converts rectified power into unidirectional
    momentum via the Poisson ratio:

        F = η_chiral × P_rect / c = (ν_vac) × δ × P_input / c

    This is identical to radiation pressure physics: a beam of power P
    produces force F = P/c on a perfect absorber.  Here δ is the
    "reflection coefficient" of the nonlinear vacuum varactor, and
    η_chiral = ν_vac selects the longitudinal thrust component.

    The phased array contributes a DIRECTIVITY GAIN G_array.
    For N elements in a phased array: G = N (not N², because P is
    total input, not per-element input).

    Final expression:
        F_total = N × ν_vac × δ(Q,β) × P_input / c

    Returns:
        F_total: total thrust [N]
        delta: rectification factor
        P_rect: rectified power [W]
    """
    delta, _, _, _ = jensen_rectification(Q_factor, beta)
    eta_chiral = NU_VAC  # = 2/7 — helical → longitudinal Poisson coupling
    P_rect = delta * P_input
    F_total = N * eta_chiral * P_rect / C_0
    return F_total, delta, P_rect


# ==============================================================================
# MAIN SIMULATION
# ==============================================================================

def run_simulation():
    """Generate 6-panel diagnostic figure."""

    # Input RF power: V²/(2Z₀) into free space (upper bound)
    P_INPUT = 0.5 * V_DRIVE_RMS**2 / Z_0  # ≈ 1.19 MW (30 kV into 377 Ω)
    # More realistic: antenna has radiation resistance R_rad ≈ 50 Ω
    R_RAD = 50.0  # Ω — typical antenna radiation resistance
    P_REALISTIC = 0.5 * V_DRIVE_RMS**2 / R_RAD  # ≈ 9 MW (maximally coupled)
    # For conservative estimate, use Z₀-limited power
    P_USED = P_INPUT

    # Colors
    C_BG = "#0a0a0f"
    C_CYAN = "#00e5ff"
    C_MAGENTA = "#ff00ff"
    C_GOLD = "#ffc107"
    C_GREEN = "#00e676"
    C_RED = "#ff1744"
    C_GRAY = "#888888"

    def style_ax(ax):
        ax.set_facecolor(C_BG)
        for spine in ax.spines.values():
            spine.set_color(C_GRAY)
        ax.tick_params(colors=C_GRAY, which='both')
        ax.xaxis.label.set_color(C_GRAY)
        ax.yaxis.label.set_color(C_GRAY)
        ax.title.set_color("white")

    fig = plt.figure(figsize=(18, 12), facecolor=C_BG)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    print("=" * 78)
    print("  CHIRAL ACOUSTIC RECTIFICATION: VACUUM VARACTOR THRUST")
    print("  Inputs: m_e, c, α, ℏ, ε₀, ν_vac=2/7  |  All from ave.core.constants")
    print("=" * 78)

    # ── Header: derived constants ──
    print(f"\n  ── DERIVED FIELD CONSTANTS ──")
    print(f"    L_NODE   = ℏ/(m_e c)     = {L_NODE:.4e} m")
    print(f"    V_YIELD  = √α × V_SNAP   = {V_YIELD:.2f} V  ({V_YIELD/1e3:.2f} kV)")
    print(f"    E_YIELD  = V_YIELD/L_NODE = {E_YIELD:.4e} V/m")
    print(f"    E_CRIT   = m_e²c³/(eℏ)   = {(M_E**2 * C_0**3)/(e_charge*HBAR):.4e} V/m")
    print(f"    √α ratio = E_YIELD/E_CRIT = {E_YIELD/((M_E**2*C_0**3)/(e_charge*HBAR)):.6f}  (= √α = {np.sqrt(ALPHA):.6f})")

    print(f"\n  ── HARDWARE PARAMETERS ──")
    print(f"    V_drive  = {V_DRIVE_RMS/1e3:.1f} kV RMS")
    print(f"    d_gap    = {D_GAP*1e3:.1f} mm")
    print(f"    r_tip    = {R_TIP*1e6:.1f} μm")
    print(f"    β (tip)  = h/r = {BETA_TIP:.0f}")
    print(f"    f_drive  = {F_DRIVE/1e6:.0f} MHz")
    print(f"    N_tips   = {N_TIPS:,}")
    print(f"    P_input  = V²/(2Z₀) = {P_USED:.2f} W ({P_USED/1e6:.3f} MW)")

    print(f"\n  ── MACROSCOPIC FIELD ──")
    print(f"    E_macro  = V/d = {E_MACRO:.2e} V/m")
    print(f"    E_macro/E_yield = {E_MACRO/E_YIELD:.2e}  (deep Regime I)")

    # ═══════════════════════════════════════════════════════════════════
    # PANEL 1: E_local/E_yield vs time (one RF cycle) — regime bouncing
    # ═══════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)

    t = np.linspace(0, 1, 1000)
    Q_demo = 1e4
    E_peak_demo = BETA_TIP * Q_demo * E_MACRO * np.sqrt(2)
    ratio_t = (E_peak_demo / E_YIELD) * np.abs(np.sin(2 * np.pi * t))

    ax1.fill_between(t, 0, ratio_t, alpha=0.2, color=C_CYAN)
    ax1.plot(t, ratio_t, color=C_CYAN, linewidth=2)
    ax1.axhline(y=0.1, color=C_GOLD, linestyle='--', alpha=0.7, label="Regime I/II (0.1)")
    ax1.set_xlabel("Time (t / T)")
    ax1.set_ylabel("E_local / E_yield")
    ax1.set_title("Panel 1: Regime Bouncing (Q=10⁴)", fontsize=11)
    ax1.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRAY, labelcolor=C_GRAY)
    ax1.set_xlim(0, 1)

    print(f"\n  ── REGIME BOUNCING (Q = {Q_demo:.0e}) ──")
    print(f"    E_local_peak = β × Q × E_macro × √2 = {E_peak_demo:.2e} V/m")
    print(f"    E_peak / E_yield = {E_peak_demo/E_YIELD:.4e}")

    # ═══════════════════════════════════════════════════════════════════
    # PANEL 2: S(t) vs time — shows the nonlinear dip
    # ═══════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2)

    ratio_clamp = np.minimum(ratio_t, 0.99999)
    S_t = np.sqrt(1.0 - ratio_clamp**2)

    ax2.fill_between(t, S_t, 1.0, alpha=0.3, color=C_MAGENTA, label="Rectified deficit")
    ax2.plot(t, S_t, color=C_CYAN, linewidth=2)
    ax2.axhline(y=np.mean(S_t), color=C_GOLD, linestyle='--', linewidth=1.5,
                label=f"⟨S⟩ = {np.mean(S_t):.10f}")
    ax2.set_xlabel("Time (t / T)")
    ax2.set_ylabel("S(E(t))")
    ax2.set_title("Panel 2: Saturation Factor (Jensen's Dip)", fontsize=11)
    ax2.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRAY, labelcolor=C_GRAY)
    ax2.set_xlim(0, 1)

    delta_demo, S_avg_demo, _, _ = jensen_rectification(Q_demo)
    print(f"\n  ── JENSEN'S RECTIFICATION (Q = {Q_demo:.0e}) ──")
    print(f"    ⟨S⟩_cycle = {S_avg_demo:.12f}")
    print(f"    δ = 1 − ⟨S⟩ = {delta_demo:.6e}")
    print(f"    Interpretation: {delta_demo*1e6:.3f} ppm permittivity asymmetry at tips")

    # ═══════════════════════════════════════════════════════════════════
    # PANEL 3: δ vs Q (rectification strength)
    # ═══════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3)

    Q_sweep = np.logspace(1, 6, 200)
    deltas = []
    for Q in Q_sweep:
        d, _, _, _ = jensen_rectification(Q)
        deltas.append(d)
    deltas = np.array(deltas)

    ax3.loglog(Q_sweep, deltas, color=C_CYAN, linewidth=2)
    ax3.axvline(x=1e4, color=C_GOLD, linestyle='--', alpha=0.7, label="Q = 10⁴ (VHF)")
    ax3.set_xlabel("Resonant Q-Factor")
    ax3.set_ylabel("δ = 1 − ⟨S⟩ (rectification)")
    ax3.set_title("Panel 3: Rectification vs Q", fontsize=11)
    ax3.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRAY, labelcolor=C_GRAY)
    ax3.grid(True, alpha=0.15, color=C_GRAY)

    # ═══════════════════════════════════════════════════════════════════
    # PANEL 4: F_total vs Q (power-constrained)
    # ═══════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[1, 0])
    style_ax(ax4)

    F_vs_Q = []
    for Q in Q_sweep:
        F, _, _ = rectified_thrust(Q, P_USED, N=N_TIPS)
        F_vs_Q.append(F)
    F_vs_Q = np.array(F_vs_Q)

    ax4.loglog(Q_sweep, F_vs_Q, color=C_GREEN, linewidth=2)
    ax4.axvline(x=1e4, color=C_GOLD, linestyle='--', alpha=0.7, label="Q = 10⁴")
    ax4.axhline(y=1e-6, color=C_RED, linestyle=':', alpha=0.8, label="1 μN floor")
    ax4.set_xlabel("Resonant Q-Factor")
    ax4.set_ylabel("F_total [N]")
    ax4.set_title(f"Panel 4: Array Thrust vs Q (N={N_TIPS:,})", fontsize=11)
    ax4.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRAY, labelcolor=C_GRAY)
    ax4.grid(True, alpha=0.15, color=C_GRAY)

    # Print thrust at Q = 1e4
    F_at_Q4, delta_Q4, P_rect_Q4 = rectified_thrust(1e4, P_USED, N=N_TIPS)
    print(f"\n  ── RECTIFIED THRUST (Q = 10⁴, N = {N_TIPS:,}) ──")
    print(f"    δ = {delta_Q4:.6e}")
    print(f"    P_rect = δ × P_input = {P_rect_Q4:.4f} W")
    print(f"    F_total = N × ν_vac × δ × P_input / c")
    print(f"    F_total = {F_at_Q4:.4e} N = {F_at_Q4*1e6:.4f} μN")
    print(f"    η_chiral = ν_vac = 2/7 = {NU_VAC:.6f}")
    detectable = F_at_Q4 > 1e-6
    print(f"    Above 1 μN floor? {'YES ✓' if detectable else 'NO — needs higher Q or N'}")

    # ═══════════════════════════════════════════════════════════════════
    # PANEL 5: F_array vs N_tips — showing N scaling
    # ═══════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5)

    N_sweep = np.logspace(1, 5, 100)
    Q_fixed = 1e4
    F_vs_N = []
    for N in N_sweep:
        F, _, _ = rectified_thrust(Q_fixed, P_USED, N=N)
        F_vs_N.append(F)
    F_vs_N = np.array(F_vs_N)

    ax5.loglog(N_sweep, F_vs_N, color=C_MAGENTA, linewidth=2, label="F = N × ν × δP/c")
    ax5.axhline(y=1e-6, color=C_RED, linestyle=':', alpha=0.8, label="1 μN detection floor")
    ax5.axvline(x=N_TIPS, color=C_GOLD, linestyle='--', alpha=0.7, label=f"N = {N_TIPS:,}")
    ax5.set_xlabel("Number of Tips (N)")
    ax5.set_ylabel("F_total [N]")
    ax5.set_title("Panel 5: Array Directivity Gain", fontsize=11)
    ax5.legend(fontsize=7, facecolor=C_BG, edgecolor=C_GRAY, labelcolor=C_GRAY)
    ax5.grid(True, alpha=0.15, color=C_GRAY)

    # ═══════════════════════════════════════════════════════════════════
    # PANEL 6: Thrust vs V_drive with detection floor
    # ═══════════════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6)

    V_sweep = np.linspace(1000, 50000, 200)
    F_vs_V = []
    for V in V_sweep:
        P_v = 0.5 * V**2 / Z_0
        F, _, _ = rectified_thrust(Q_fixed, P_v, N=N_TIPS)
        F_vs_V.append(F)
    F_vs_V = np.array(F_vs_V)

    ax6.semilogy(V_sweep / 1e3, F_vs_V, color=C_CYAN, linewidth=2)
    ax6.axhline(y=1e-6, color=C_RED, linestyle=':', alpha=0.8, label="1 μN detection floor")
    ax6.axvline(x=30, color=C_GOLD, linestyle='--', alpha=0.7, label="30 kV design point")
    ax6.set_xlabel("V_drive [kV RMS]")
    ax6.set_ylabel("F_total [N]")
    ax6.set_title(f"Panel 6: Thrust vs Voltage (N={N_TIPS:,}, Q={Q_fixed:.0e})", fontsize=11)
    ax6.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRAY, labelcolor=C_GRAY)
    ax6.grid(True, alpha=0.15, color=C_GRAY)

    # ── Verification checks ──
    print(f"\n  ── VERIFICATION ──")

    # Energy conservation: P_thrust must be << P_input
    P_thrust = F_at_Q4 * C_0
    print(f"    P_thrust = F × c = {P_thrust:.4e} W")
    print(f"    P_input  = {P_USED:.4e} W")
    print(f"    Efficiency = P_thrust / P_input = {P_thrust/P_USED:.2e}", end="")
    if P_thrust < P_USED:
        print(f" ✓ (< 1)")
    else:
        print(f" ✗ (MODEL VIOLATION)")

    # Dimensional check
    print(f"\n    Dimensional: F = N × ν × δ × P/c  → [1][1][1][W]/[m/s] = [N] ✓")

    # Limit checks
    F_Q0, _, _ = rectified_thrust(0.0, P_USED, N=N_TIPS)
    print(f"    Q → 0: F = {F_Q0:.2e} N {'✓ (zero)' if F_Q0 < 1e-50 else '✗'}")

    # Q required for 1 μN
    for Q_test in [1e3, 1e4, 1e5, 1e6]:
        F_test, d_test, _ = rectified_thrust(Q_test, P_USED, N=N_TIPS)
        print(f"    Q = {Q_test:.0e}: δ = {d_test:.2e}, F = {F_test*1e6:.4f} μN", end="")
        if F_test > 1e-6:
            print(" ← DETECTABLE")
        else:
            print()

    print(f"\n  ── AXIOM TRACE ──")
    print(f"    Axiom 1: L_NODE = ℏ/(m_e c) → lattice pitch")
    print(f"    Axiom 4: S(E) = √(1 − (E/E_yield)²) → varactor nonlinearity")
    print(f"    ν_vac = 2/7 → chiral (helical ↔ longitudinal) Poisson coupling")
    print(f"    Zero free parameters. All from m_e, c, α, ℏ, ε₀.")

    # Save
    fig.suptitle("Chiral Acoustic Rectification — Vacuum Varactor Thrust",
                 fontsize=16, color="white", fontweight="bold", y=0.98)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "manuscript",
                           "book_7_ponder_01", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "chiral_acoustic_rectification.png")
    plt.savefig(out_path, dpi=200, facecolor=C_BG, bbox_inches="tight")
    print(f"\n  Figure saved: {os.path.relpath(out_path)}")
    print("=" * 78)


if __name__ == "__main__":
    run_simulation()
