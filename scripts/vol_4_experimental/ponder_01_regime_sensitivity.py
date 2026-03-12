#!/usr/bin/env python3
"""
PONDER-01: Regime Sensitivity Analysis
=======================================

Maps the full solution space for the Ponder-01 thrust mechanism
across operating regimes. Sweeps across:

  1. Voltage (V_rms):     1 kV → 50 kV
  2. Aspect ratio:        1:1 → 10000:1  (flat plate → sharp needle)
  3. Cell size (dx):      ℓ_node → 1 cm  (fundamental → macroscopic)
  4. Q-enhanced voltage:  V_cell = Q × V_rms

Per-regime analytical solutions from Chapter 12:
  I.   LINEAR          (V/Vy < 0.1):  Standard Maxwell, ΔF = 0
  II.  WEAKLY NL       (0.1–0.5):     Taylor C_eff ≈ C₀(1 + ½(V/Vy)²)
  III. STRONGLY NL     (0.5–0.99):    Full saturation kernel
  IV.  SATURATED/TVS   (≥ 1.0):       Phase transition, ε → 0

KEY QUESTION: At what physical scale does V_yield apply?
  - FDTD engine uses dx (sim grid cell) → E_yield = V_yield/dx
  - Physical lattice uses ℓ_node → E_yield = V_yield/ℓ_node

This script computes the answer for BOTH interpretations and presents
the regime landscape so the user can assess which is physical.

All constants from ave.core.constants.

Usage:
    PYTHONPATH=src python scripts/vol_4_experimental/ponder_01_regime_sensitivity.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ave.core.constants import (
    C_0, ALPHA, HBAR, e_charge, M_E,
    MU_0, EPSILON_0, Z_0,
    V_SNAP, V_YIELD, XI_TOPO, L_NODE,
)
from ave.core.universal_operators import universal_saturation

# ══════════════════════════════════════════════════════════════════════════════
# REGIME CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def classify_regime(V_local, V_yield=V_YIELD):
    """Classify the operating regime based on V_local/V_yield ratio."""
    ratio = abs(V_local) / V_yield
    if ratio < 0.1:
        return "I-LINEAR", ratio
    elif ratio < 0.5:
        return "II-WEAK-NL", ratio
    elif ratio < 1.0:
        return "III-STRONG-NL", ratio
    else:
        return "IV-SATURATED", ratio

# ══════════════════════════════════════════════════════════════════════════════
# E-FIELD MODELS
# ══════════════════════════════════════════════════════════════════════════════

def E_tip_hyperboloid(V_rms, d_gap, r_tip):
    """Peak E at tip of hyperboloid-plane capacitor."""
    return V_rms / (r_tip * np.log(2.0 * d_gap / r_tip))

def E_flat_plate(V_rms, d_gap):
    """Uniform E in parallel-plate capacitor."""
    return V_rms / d_gap

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_sensitivity():
    print("=" * 78)
    print("  PONDER-01: REGIME SENSITIVITY ANALYSIS")
    print("  Mapping the solution space across operating regimes")
    print("=" * 78)
    
    # ── Fundamental scales ─────────────────────────────────────────────────
    E_yield_lattice = V_YIELD / float(L_NODE)  # Physical yield field at ℓ_node
    
    print(f"\n  CRITICAL SCALES:")
    print(f"    V_YIELD = √α × m_e c²/e = {V_YIELD/1e3:.2f} kV")
    print(f"    ℓ_node  = ℏ/(m_e c)      = {float(L_NODE)*1e13:.2f} × 10⁻¹³ m")
    print(f"    E_yield (at ℓ_node)       = {E_yield_lattice:.3e} V/m")
    print(f"    E_yield (at dx=1cm)       = {V_YIELD/0.01:.3e} V/m")
    print(f"    E_yield (at dx=1mm)       = {V_YIELD/0.001:.3e} V/m")
    print(f"    E_yield (at dx=1µm)       = {V_YIELD/1e-6:.3e} V/m")
    
    # ── Sweep 1: V_rms with different cell sizes ───────────────────────────
    print(f"\n  ═══ SWEEP 1: Regime Classification vs Voltage & Scale ═══")
    V_range = np.array([1e3, 5e3, 10e3, 20e3, 30e3, 40e3, V_YIELD])
    
    # Different physical interpretation scales for dx
    dx_set = {
        "ℓ_node (3.86e-13 m)": float(L_NODE),
        "1 fm (1e-15 m)": 1e-15,
        "1 pm (1e-12 m)": 1e-12,
        "1 µm (1e-6 m)": 1e-6,
        "1 mm (1e-3 m)": 1e-3,
        "1 cm (FDTD default)": 0.01,
    }
    
    print(f"\n  Flat plate (d = 1mm), E = V/d:")
    d_gap = 1e-3
    
    print(f"\n  {'V_rms (kV)':>12}", end="")
    for label in dx_set:
        print(f"  {label[:16]:>18}", end="")
    print()
    print("  " + "-" * (12 + 18 * len(dx_set) + 2 * len(dx_set)))
    
    for V in V_range:
        E = V / d_gap
        print(f"  {V/1e3:12.2f}", end="")
        for label, dx in dx_set.items():
            V_local = E * dx
            regime, ratio = classify_regime(V_local)
            marker = regime.split("-")[0]
            print(f"  {marker:>6} ({ratio:7.1e})", end="")
        print()
    
    # ── Sweep 2: Tip geometry effect ───────────────────────────────────────
    print(f"\n\n  ═══ SWEEP 2: Tip Field Enhancement ═══")
    print(f"  V_rms = 30 kV, d_gap = 1 mm")
    
    V_rms = 30e3
    d_gap = 1e-3
    
    r_tips = np.array([1e-3, 100e-6, 10e-6, 1e-6, 100e-9, 10e-9, 1e-9])
    
    print(f"\n  {'r_tip':>12} {'Aspect':>8} {'E_tip (V/m)':>14} ", end="")
    print(f"{'Regime@ℓ_node':>16} {'Regime@1µm':>14} {'Regime@1cm':>14}")
    print("  " + "-" * 82)
    
    for r_tip in r_tips:
        if r_tip >= d_gap:
            continue
        aspect = d_gap / r_tip
        E_tip = E_tip_hyperboloid(V_rms, d_gap, r_tip)
        
        # Regime at different scales
        reg_l, rat_l = classify_regime(E_tip * float(L_NODE))
        reg_um, rat_um = classify_regime(E_tip * 1e-6)
        reg_cm, rat_cm = classify_regime(E_tip * 0.01)
        
        print(f"  {r_tip*1e6:10.1f} µm {aspect:8.0f}:1 {E_tip:14.3e} "
              f"{reg_l.split('-')[0]:>4}({rat_l:.1e}) "
              f"{reg_um.split('-')[0]:>4}({rat_um:.1e}) "
              f"{reg_cm.split('-')[0]:>4}({rat_cm:.1e})")
    
    # ── Sweep 3: Q-enhanced resonant voltage ───────────────────────────────
    print(f"\n\n  ═══ SWEEP 3: Q-Enhanced Resonant Voltage ═══")
    print(f"  If the HOPF-01 operates at resonance with quality factor Q,")
    print(f"  the voltage across the inductance (wire) is V_wire = Q × V_input.")
    print(f"  The local EM energy density at resonance is Q² × the input level.")
    print()
    
    Q_values = np.array([1, 10, 50, 100, 500, 1000, 5000])
    V_input = np.array([100, 500, 1000, 5000, 10000])  # Volts
    
    print(f"  {'V_in (V)':>10}", end="")
    for Q in Q_values:
        print(f"  Q={Q:<6}", end="")
    print()
    print("  " + "-" * (10 + 10 * len(Q_values)))
    
    for V in V_input:
        print(f"  {V:10.0f}", end="")
        for Q in Q_values:
            V_enhanced = Q * V
            regime, ratio = classify_regime(V_enhanced)
            marker = regime.split("-")[0]
            if V_enhanced > V_YIELD:
                print(f"  {marker:>4}!!", end="")
            elif ratio > 0.1:
                print(f"  {marker:>4}* ", end="")
            else:
                print(f"  {marker:>4}  ", end="")
        print()
    
    print(f"\n  Legend: !! = above V_yield (breakdown), * = nonlinear region")
    print(f"  V_yield = {V_YIELD/1e3:.2f} kV, so Q × V_in > {V_YIELD:.0f} V triggers breakdown")
    
    # ── Sweep 4: The actual HOPF-01 wire near-field ────────────────────────
    print(f"\n\n  ═══ SWEEP 4: HOPF-01 Wire Near-Field Estimate ═══")
    
    # HOPF-01 parameters (from hopf_01_impedance_model.py)
    WIRE_DIA = 0.51e-3  # 24 AWG
    WIRE_HEIGHT = 1.6e-3  # height above ground plane
    L_WIRE = 0.120  # Trefoil wire length
    
    # At 100 MHz, if V_input = 10V into 50Ω, P_in = 1W
    # With Q ≈ 100, the voltage on the resonator is V_wire = Q × V_in = 1000V
    # The E-field near the wire surface at distance r from center:
    # E(r) = V_wire / (r × ln(2h/r_wire))  (wire over ground)
    
    r_wire = WIRE_DIA / 2
    h = WIRE_HEIGHT
    
    print(f"  Wire diameter:  {WIRE_DIA*1e3:.2f} mm (24 AWG)")
    print(f"  Wire height:    {h*1e3:.2f} mm above ground")
    print(f"  Wire length:    {L_WIRE*1e3:.0f} mm (Trefoil)")
    
    print(f"\n  Q = 100, V_input = 10V → V_wire = 1000V")
    
    V_wire = 1000.0
    log_factor = np.log(2 * h / r_wire)
    
    # E at wire surface
    E_surface = V_wire / (r_wire * log_factor)
    print(f"  E at wire surface: {E_surface/1e3:.1f} kV/m")
    print(f"  V_local at ℓ_node: {E_surface * float(L_NODE):.3e} V  (ratio: {E_surface * float(L_NODE)/V_YIELD:.3e})")
    print(f"  V_local at 1µm:    {E_surface * 1e-6:.3e} V  (ratio: {E_surface * 1e-6/V_YIELD:.3e})")
    
    print(f"\n  Q = 100, V_input = 100V → V_wire = 10,000V")
    V_wire = 10000.0
    E_surface = V_wire / (r_wire * log_factor)
    print(f"  E at wire surface: {E_surface/1e3:.1f} kV/m")
    print(f"  V_local at ℓ_node: {E_surface * float(L_NODE):.3e} V  (ratio: {E_surface * float(L_NODE)/V_YIELD:.3e})")
    print(f"  V_local at 1µm:    {E_surface * 1e-6:.3e} V  (ratio: {E_surface * 1e-6/V_YIELD:.3e})")
    
    print(f"\n  Q = 1000, V_input = 100V → V_wire = 100,000V")
    V_wire = 100000.0
    E_surface = V_wire / (r_wire * log_factor)
    print(f"  E at wire surface: {E_surface/1e6:.1f} MV/m")
    print(f"  V_local at ℓ_node: {E_surface * float(L_NODE):.3e} V  (ratio: {E_surface * float(L_NODE)/V_YIELD:.3e})")
    print(f"  V_local at 1µm:    {E_surface * 1e-6:.3e} V  (ratio: {E_surface * 1e-6/V_YIELD:.3e})")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PLOTTING — Regime Map
    # ══════════════════════════════════════════════════════════════════════════
    
    C_BG = "#0a0a1a"
    C_TEXT = "#e0e0e0"
    C_GRID = "#1a2a3a"
    REGIME_COLORS = {
        "I": "#44aaff",      # Linear — blue
        "II": "#44ff88",     # Weak NL — green
        "III": "#ffaa44",    # Strong NL — orange 
        "IV": "#ff4444",     # Saturated — red
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(C_BG)
    
    def style_ax(ax):
        ax.set_facecolor(C_BG)
        ax.tick_params(colors=C_TEXT, labelsize=9)
        ax.grid(True, alpha=0.12, color=C_GRID)
        for spine in ax.spines.values():
            spine.set_color("#333355")
    
    # Panel 1: ε_eff/ε₀ vs V_local/V_yield — the master saturation curve
    ax1 = axes[0, 0]
    style_ax(ax1)
    ratio_range = np.linspace(0, 1.0, 1000)
    S = np.sqrt(1.0 - ratio_range**2)
    ax1.plot(ratio_range, S, color="#ffaa44", linewidth=2.5)
    
    # Shade regimes
    ax1.axvspan(0, 0.1, alpha=0.08, color=REGIME_COLORS["I"], label="I: Linear")
    ax1.axvspan(0.1, 0.5, alpha=0.08, color=REGIME_COLORS["II"], label="II: Weak NL")
    ax1.axvspan(0.5, 1.0, alpha=0.08, color=REGIME_COLORS["III"], label="III: Strong NL")
    ax1.axvline(x=1.0, color=REGIME_COLORS["IV"], linestyle="--", linewidth=2, label="IV: TVS Yield")
    
    ax1.set_xlabel(r"$V_{local} / V_{yield}$", color=C_TEXT, fontsize=11)
    ax1.set_ylabel(r"$\varepsilon_{eff} / \varepsilon_0 = S(V)$", color=C_TEXT, fontsize=11)
    ax1.set_title("Axiom 4 Saturation Curve\n& Operating Regimes", color=C_TEXT, fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, facecolor="#111133", edgecolor="#333355", labelcolor=C_TEXT)
    ax1.set_ylim(0, 1.05)
    
    # Panel 2: Which regime are we in? (Voltage vs Cell size heatmap)
    ax2 = axes[0, 1]
    style_ax(ax2)
    V_sweep = np.logspace(2, 5, 200)  # 100V to 100kV
    dx_sweep = np.logspace(-16, -1, 200)  # sub-ℓ_node to 10cm
    VV, DD = np.meshgrid(V_sweep, dx_sweep)
    
    # For flat plate at 1mm gap
    d_gap = 1e-3
    EE = VV / d_gap   # E-field
    V_local = EE * DD  # voltage across cell
    ratio_map = V_local / V_YIELD
    
    # Map ratio to regime integer
    regime_map = np.ones_like(ratio_map)
    regime_map[ratio_map > 0.1] = 2
    regime_map[ratio_map > 0.5] = 3
    regime_map[ratio_map >= 1.0] = 4
    
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([REGIME_COLORS["I"], REGIME_COLORS["II"],
                           REGIME_COLORS["III"], REGIME_COLORS["IV"]])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    
    im = ax2.pcolormesh(VV/1e3, DD, regime_map, cmap=cmap, norm=norm,
                        shading="auto", alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Applied Voltage (kV)", color=C_TEXT, fontsize=11)
    ax2.set_ylabel("Cell size dx (m)", color=C_TEXT, fontsize=11)
    ax2.set_title("Regime Map: Voltage × Cell Size\n(flat plate, 1mm gap)",
                  color=C_TEXT, fontsize=13, fontweight="bold")
    
    # Mark key scales
    ax2.axhline(y=float(L_NODE), color="#ffffff", linestyle=":", alpha=0.5)
    ax2.text(0.12, float(L_NODE)*1.5, "ℓ_node", color="#ffffff", fontsize=8)
    ax2.axhline(y=0.01, color="#ffffff", linestyle=":", alpha=0.5)
    ax2.text(0.12, 0.012, "FDTD default dx", color="#ffffff", fontsize=8)
    
    # Panel 3: ε_eff deficit (the thrust source) vs V/V_yield
    ax3 = axes[1, 0]
    style_ax(ax3)
    delta_eps = (S - 1.0)  # negative where saturated
    # Taylor leading order: -½(V/Vy)²
    taylor = -0.5 * ratio_range**2
    ax3.plot(ratio_range, delta_eps, color="#ff4444", linewidth=2.5, label=r"Exact: $S - 1$")
    ax3.plot(ratio_range, taylor, color="#44aaff", linewidth=1.5, linestyle="--",
             label=r"Taylor: $-\frac{1}{2}(V/V_y)^2$")
    ax3.axvspan(0, 0.1, alpha=0.08, color=REGIME_COLORS["I"])
    ax3.axvspan(0.1, 0.5, alpha=0.08, color=REGIME_COLORS["II"])
    ax3.axvspan(0.5, 1.0, alpha=0.08, color=REGIME_COLORS["III"])
    ax3.set_xlabel(r"$V_{local} / V_{yield}$", color=C_TEXT, fontsize=11)
    ax3.set_ylabel(r"$\Delta\varepsilon / \varepsilon_0$", color=C_TEXT, fontsize=11)
    ax3.set_title("Saturation Deficit (Thrust Source)\nExact vs Taylor Approximation",
                  color=C_TEXT, fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9, facecolor="#111133", edgecolor="#333355", labelcolor=C_TEXT)
    
    # Panel 4: Q-enhanced voltage map
    ax4 = axes[1, 1]
    style_ax(ax4)
    Q_sweep = np.logspace(0, 4, 200)  # 1 to 10000
    V_in_sweep = np.logspace(0, 4, 200)  # 1V to 10kV
    QQ, VV_in = np.meshgrid(Q_sweep, V_in_sweep)
    V_enhanced = QQ * VV_in
    ratio_q = V_enhanced / V_YIELD
    
    regime_q = np.ones_like(ratio_q)
    regime_q[ratio_q > 0.1] = 2
    regime_q[ratio_q > 0.5] = 3
    regime_q[ratio_q >= 1.0] = 4
    
    im2 = ax4.pcolormesh(QQ, VV_in, regime_q, cmap=cmap, norm=norm,
                         shading="auto", alpha=0.6)
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Quality Factor Q", color=C_TEXT, fontsize=11)
    ax4.set_ylabel("Input Voltage V_in (V)", color=C_TEXT, fontsize=11)
    ax4.set_title("Regime Map: Q × V_input\n(resonant enhancement)",
                  color=C_TEXT, fontsize=13, fontweight="bold")
    
    # Mark the Q × V = V_yield contour
    V_in_contour = V_YIELD / Q_sweep
    ax4.plot(Q_sweep, V_in_contour, color="#ffffff", linewidth=2, linestyle="-",
             label=f"Q × V = V_yield ({V_YIELD/1e3:.1f} kV)")
    ax4.legend(fontsize=8, facecolor="#111133", edgecolor="#333355", labelcolor=C_TEXT,
               loc="upper right")
    
    fig.suptitle(
        "PONDER-01: Regime Sensitivity Analysis\n"
        r"Operating regime depends on $V_{local}/V_{yield}$ — "
        r"which depends on the physical scale of the cell",
        color=C_TEXT, fontsize=14, fontweight="black", y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(os.path.dirname(__file__), "..", "assets", "sim_outputs",
                            "ponder_01_regime_sensitivity.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, facecolor=C_BG, bbox_inches="tight")
    print(f"\n  ✓ Regime map saved → {out_path}")
    
    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n  ═══════════════════════════════════════════════════════════")
    print(f"  CONCLUSIONS:")
    print(f"  ───────────────────────────────────────────────────────────")
    print(f"  1. At ℓ_node scale: E_yield = {E_yield_lattice:.2e} V/m")
    print(f"     → NO LAB FIELD can reach Regime II, let alone III/IV.")
    print(f"     → Bulk saturation thrust = ZERO.")
    print(f"")
    print(f"  2. At FDTD dx=1cm: E_yield = 4.37 MV/m")
    print(f"     → Achievable with sharp electrodes or high-Q resonators.")
    print(f"     → This is the FDTD engine's deliberate coarse-graining.")
    print(f"")
    print(f"  3. Q-enhanced resonant voltage:")
    print(f"     → With Q=100, V_in=437V achieves Q×V = V_yield")
    print(f"     → With Q=1000, V_in=43.7V is sufficient")
    print(f"     → The HOPF-01 operates at resonance — Q matters!")
    print(f"")
    print(f"  4. THE QUESTION IS:")
    print(f"     Does V_yield apply per ℓ_node (= unreachable), or is")
    print(f"     the FDTD's dx-scaling a legitimate coarse-graining of")
    print(f"     collective lattice behavior (= Q-enhanced lab fields)?")
    print(f"  ═══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    run_sensitivity()
