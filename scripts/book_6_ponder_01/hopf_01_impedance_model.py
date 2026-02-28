#!/usr/bin/env python3
r"""
HOPF-01: Full Impedance & Frequency Model
===========================================

Complete electrical model of the wire-stitched torus knot fixture on a
standard 2-layer FR-4 board.  Models:

  1. Wire-above-ground impedance Z₀(h, d, ε_eff) via image-charge theory
  2. Resonant frequencies for all 4 knots (half-wave open resonator)
  3. SMA mismatch reflection (Γ, return loss) at the 50Ω interface
  4. Predicted S₁₁(f) envelope for each knot
  5. Air vs mineral oil comparison
  6. Effect of wire sag / height variation

Board stackup:
  ┌──────────────────────────┐
  │ F.Cu   SMA pads only     │  ← SMA ground tabs, stitching vias
  │ FR-4   1.6mm, ε_r = 4.3  │
  │ B.Cu   ground patches    │  ← 10×10mm zones under SMA connectors only
  └──────────────────────────┘
  ↑ 10mm nylon standoffs ↑
  ═════════════════════════════  (table)

Wire: 24 AWG (d=0.51mm) enameled magnet wire, routed through
unplated holes, sitting in air above and below the board.

Usage:
    PYTHONPATH=src python scripts/book_6_ponder_01/hopf_01_impedance_model.py
"""

import sys
import os
import pathlib
import numpy as np

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

from ave.core.constants import C_0, ALPHA, MU_0, EPSILON_0

# ══════════════════════════════════════════════════════════════
# Physical Parameters — Standard 2-Layer FR-4
# ══════════════════════════════════════════════════════════════

# Board
PCB_THICKNESS = 1.6e-3      # m (FR-4 core)
EPS_R_FR4 = 4.3             # FR-4 relative permittivity
CU_THICKNESS = 35e-6        # m (1 oz copper, both layers)
STANDOFF_HEIGHT = 10e-3     # m (nylon standoff elevation)

# Wire
WIRE_DIA = 0.51e-3          # m (24 AWG)
WIRE_RADIUS = WIRE_DIA / 2
ENAMEL_THICKNESS = 30e-6    # m (polyurethane enamel)
ENAMEL_EPS_R = 3.5          # polyurethane relative permittivity
WIRE_CONDUCTIVITY = 5.8e7   # S/m (copper)

# SMA Connector
Z_SMA = 50.0                # Ω

# Media
EPS_R_AIR = 1.0006          # air at STP
EPS_R_OIL = 2.1             # mineral oil (transformer grade)

# Knot catalog: (p, q, L_wire_m, label)
KNOTS = [
    (2, 3,  0.120, '(2,3) Trefoil'),
    (2, 5,  0.160, '(2,5) Cinquefoil'),
    (3, 7,  0.200, '(3,7)'),
    (3, 11, 0.250, '(3,11)'),
]


# ══════════════════════════════════════════════════════════════
# Impedance Model: Wire Above Ground Plane
# ══════════════════════════════════════════════════════════════

def wire_over_ground_Z0(h, d, eps_eff):
    """Characteristic impedance of a round wire at height h above a
    ground plane, in a medium with effective permittivity eps_eff.

    Uses the image-charge model:
        Z₀ = (60 / √ε_eff) × acosh(2h/d)

    Valid when h >> d (thin wire approximation).

    Parameters
    ----------
    h : float — wire center height above ground plane [m]
    d : float — wire outer diameter [m]
    eps_eff : float — effective relative permittivity of surrounding medium

    Returns
    -------
    Z0 : float — characteristic impedance [Ω]
    """
    ratio = 2 * h / d
    if ratio <= 1:
        ratio = 1.01  # avoid domain error
    return (60.0 / np.sqrt(eps_eff)) * np.arccosh(ratio)


def effective_permittivity(eps_medium, wire_dia, enamel_thickness, enamel_eps_r):
    """Compute effective permittivity accounting for enamel coating.

    The enamel is a thin dielectric shell (ε_r ≈ 3.5) around the conductor.
    Most of the field is in the surrounding medium, so the correction is small.

    Simple volume-fraction model:
        ε_eff ≈ eps_medium × (1 + f_enamel × (ε_enamel/eps_medium - 1))
    where f_enamel ≈ 2×t_enamel / wire_dia (thin shell fraction).
    """
    f_enamel = 2 * enamel_thickness / wire_dia
    eps_eff = eps_medium * (1 + f_enamel * (enamel_eps_r / eps_medium - 1))
    return eps_eff


def skin_depth(freq, sigma=WIRE_CONDUCTIVITY, mu=float(MU_0)):
    """Skin depth in copper at frequency f."""
    return np.sqrt(1 / (np.pi * freq * mu * sigma))


def wire_resistance_per_m(freq, wire_dia):
    """AC resistance per meter for a round wire at frequency f.

    Uses skin-depth limited effective cross section.
    """
    delta = skin_depth(freq)
    r = wire_dia / 2
    if delta >= r:
        # DC resistance
        area = np.pi * r**2
    else:
        # Skin-depth limited
        area = np.pi * (r**2 - (r - delta)**2)
    return 1 / (WIRE_CONDUCTIVITY * area)


# ══════════════════════════════════════════════════════════════
# Resonant Frequency Model
# ══════════════════════════════════════════════════════════════

def f_resonance(L_wire, eps_eff, chiral_factor=0.0):
    """Half-wave open-ended resonator frequency.

    f_res = c / (2 × L × √ε_eff × (1 + χ))

    The factor of 2L (not 2πL) is correct for a half-wave transmission
    line resonator with open ends on both sides.
    """
    n_eff = np.sqrt(eps_eff) * (1 + chiral_factor)
    return float(C_0) / (2 * L_wire * n_eff)


def chiral_factor(p, q):
    """AVE topological coupling: α × pq/(p+q)."""
    return float(ALPHA) * p * q / (p + q)


def quality_factor(freq, L_wire, Z0, R_per_m):
    """Unloaded Q of a half-wave resonator.

    Q = (π × Z₀) / (R_total)
    where R_total = R_per_m × L_wire (total wire resistance).
    """
    R_total = R_per_m * L_wire
    return np.pi * Z0 / R_total


# ══════════════════════════════════════════════════════════════
# S₁₁ Model
# ══════════════════════════════════════════════════════════════

def s11_response(f_array, f_res, Q, Z_ant, Z_source=Z_SMA):
    """Compute S₁₁(f) for a resonant antenna with impedance mismatch.

    Models the antenna as a series RLC resonator seen through an
    impedance transformer (Z_ant != Z_source).

    Returns S₁₁ in dB.
    """
    # Normalized frequency deviation
    delta = (f_array - f_res) / f_res

    # Antenna impedance near resonance (series RLC model)
    Z_ant_f = Z_ant * (1 + 1j * Q * 2 * delta)

    # Reflection coefficient
    gamma = (Z_ant_f - Z_source) / (Z_ant_f + Z_source)

    return 20 * np.log10(np.abs(gamma))


# ══════════════════════════════════════════════════════════════
# Main Analysis
# ══════════════════════════════════════════════════════════════

def analyze_medium(medium_name, eps_r_medium, wire_height):
    """Run complete analysis for one medium (air or oil)."""
    eps_eff = effective_permittivity(eps_r_medium, WIRE_DIA, ENAMEL_THICKNESS, ENAMEL_EPS_R)
    Z0 = wire_over_ground_Z0(wire_height, WIRE_DIA, eps_eff)

    results = []
    for p, q, L_wire, label in KNOTS:
        chi = chiral_factor(p, q)
        pq_ppq = p * q / (p + q)

        f_std = f_resonance(L_wire, eps_eff, 0.0)
        f_ave = f_resonance(L_wire, eps_eff, chi)
        df = f_std - f_ave

        R_per_m = wire_resistance_per_m(f_std, WIRE_DIA)
        Q = quality_factor(f_std, L_wire, Z0, R_per_m)

        # SMA mismatch
        gamma_dc = (Z0 - Z_SMA) / (Z0 + Z_SMA)
        return_loss_dc = -20 * np.log10(abs(gamma_dc))

        results.append({
            'p': p, 'q': q, 'label': label,
            'pq_ppq': pq_ppq, 'chi': chi,
            'L_wire': L_wire,
            'f_std': f_std, 'f_ave': f_ave, 'df': df,
            'df_frac': df / f_std,
            'Z0': Z0, 'Q': Q,
            'R_per_m': R_per_m,
            'gamma_dc': gamma_dc,
            'return_loss_dc': return_loss_dc,
            'eps_eff': eps_eff,
        })

    return results, Z0, eps_eff


def main():
    c0 = float(C_0)
    alpha = float(ALPHA)

    print("=" * 80)
    print("  HOPF-01: Full Impedance & Frequency Model")
    print("  Standard 2-Layer FR-4, Wire-Stitched Fixture")
    print("=" * 80)

    # ── Board Stackup ──
    print(f"\n  ── Board Stackup ──")
    print(f"  PCB:        {PCB_THICKNESS*1e3:.1f}mm FR-4 (ε_r = {EPS_R_FR4})")
    print(f"  Copper:     {CU_THICKNESS*1e6:.0f}μm (1 oz), both layers")
    print(f"  Standoffs:  {STANDOFF_HEIGHT*1e3:.0f}mm nylon (board elevation)")
    print(f"  Wire:       {WIRE_DIA*1e3:.2f}mm (24 AWG), {ENAMEL_THICKNESS*1e6:.0f}μm enamel")

    # Wire height analysis
    # Front side: wire sits on PCB surface + wire radius
    h_front = PCB_THICKNESS + WIRE_RADIUS  # from B.Cu ground patch
    # Back side: wire hangs below board, distance to ground patch above
    # (but no ground on most of B.Cu — wire is in free space)
    # With standoffs: wire hangs in air, nearest ground is SMA patch (~distance varies)
    h_avg = h_front  # Conservative: use front-side height

    print(f"\n  ── Wire-Ground Geometry ──")
    print(f"  Wire height above B.Cu ground:  {h_front*1e3:.2f}mm (front side)")
    print(f"  Note: B.Cu ground only under SMA connectors (10×10mm patches)")
    print(f"  Wire is mostly in FREE SPACE — no continuous ground plane")

    # ── AIR Analysis ──
    air_results, Z0_air, eps_eff_air = analyze_medium("Air", EPS_R_AIR, h_front)

    print(f"\n  ── AIR (ε_eff = {eps_eff_air:.4f}) ──")
    print(f"  Z₀ = {Z0_air:.1f} Ω (wire over ground)")
    print(f"  SMA mismatch: Γ = {air_results[0]['gamma_dc']:.3f}, "
          f"RL = {air_results[0]['return_loss_dc']:.1f} dB")
    print(f"\n  {'Knot':<20} {'f_std':>10} {'f_ave':>10} {'Δf':>10} {'Δf/f':>10} "
          f"{'Q':>8} {'R/m':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")
    for r in air_results:
        print(f"  {r['label']:<20} {r['f_std']/1e9:>8.3f}GHz {r['f_ave']/1e9:>8.3f}GHz "
              f"{r['df']/1e6:>8.2f}MHz {r['df_frac']:>10.6f} "
              f"{r['Q']:>7.0f} {r['R_per_m']:>8.2f}Ω/m")

    # ── MINERAL OIL Analysis ──
    oil_results, Z0_oil, eps_eff_oil = analyze_medium("Mineral Oil", EPS_R_OIL, h_front)

    print(f"\n  ── MINERAL OIL (ε_eff = {eps_eff_oil:.4f}) ──")
    print(f"  Z₀ = {Z0_oil:.1f} Ω")
    print(f"\n  {'Knot':<20} {'f_std':>10} {'f_ave':>10} {'Δf':>10} {'Δf/f':>10} "
          f"{'Q':>8} {'R/m':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")
    for r in oil_results:
        print(f"  {r['label']:<20} {r['f_std']/1e9:>8.3f}GHz {r['f_ave']/1e9:>8.3f}GHz "
              f"{r['df']/1e6:>8.2f}MHz {r['df_frac']:>10.6f} "
              f"{r['Q']:>7.0f} {r['R_per_m']:>8.2f}Ω/m")

    # ── Substrate Independence ──
    print(f"\n  ── Substrate Independence ──")
    print(f"  {'Knot':<20} {'Δf/f (air)':>12} {'Δf/f (oil)':>12} {'Ratio':>8}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*8}")
    for ra, ro in zip(air_results, oil_results):
        ratio = ra['df_frac'] / ro['df_frac'] if ro['df_frac'] > 0 else 0
        print(f"  {ra['label']:<20} {ra['df_frac']:>12.8f} {ro['df_frac']:>12.8f} {ratio:>8.5f}")

    # ── Height Sensitivity ──
    print(f"\n  ── Wire Height Sensitivity ──")
    heights = np.array([0.5, 1.0, 1.6, 2.0, 3.0, 5.0]) * 1e-3
    print(f"  {'h (mm)':<10} {'Z₀ (Ω)':>10} {'ε_eff':>8} {'f_trefoil':>12}")
    print(f"  {'─'*10} {'─'*10} {'─'*8} {'─'*12}")
    for h in heights:
        e = effective_permittivity(EPS_R_AIR, WIRE_DIA, ENAMEL_THICKNESS, ENAMEL_EPS_R)
        z = wire_over_ground_Z0(h, WIRE_DIA, e)
        f = f_resonance(0.120, e)
        print(f"  {h*1e3:<10.1f} {z:>10.1f} {e:>8.4f} {f/1e9:>10.3f}GHz")

    # ── Generate Figure ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(22, 18))
        fig.patch.set_facecolor('#0a0a0a')
        gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.28)
        colors = ['#00ffcc', '#ff6b6b', '#ffd93d', '#6bcaff']

        # ── Panel 1: S₁₁ response (Air) ──
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#111111')
        for i, r in enumerate(air_results):
            f_span = np.linspace(r['f_std'] * 0.85, r['f_std'] * 1.15, 1000)
            s11 = s11_response(f_span, r['f_std'], r['Q'], r['Z0'])
            s11_ave = s11_response(f_span, r['f_ave'], r['Q'], r['Z0'])
            ax1.plot(f_span / 1e9, s11, color=colors[i], lw=1.5, alpha=0.4)
            ax1.plot(f_span / 1e9, s11_ave, color=colors[i], lw=2,
                     label=f"{r['label']}")
            # Mark resonant dips
            ax1.axvline(r['f_std'] / 1e9, color=colors[i], lw=0.5, ls=':', alpha=0.3)
            ax1.axvline(r['f_ave'] / 1e9, color=colors[i], lw=1, ls='--', alpha=0.5)

        ax1.set_xlabel('Frequency (GHz)', color='white', fontsize=11)
        ax1.set_ylabel(r'$S_{11}$ (dB)', color='white', fontsize=11)
        ax1.set_title(f'Predicted S$_{{11}}$ Response (Air)\n'
                      f'Z$_0$ = {Z0_air:.0f}Ω, mismatch to 50Ω SMA',
                      color='#00ffcc', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax1.axhline(-10, color='#ff3366', lw=1, ls=':', alpha=0.5, label='-10 dB')
        ax1.set_ylim(-30, 0)
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.15, color='white')
        for s in ax1.spines.values(): s.set_color('#333')

        # ── Panel 2: S₁₁ response (Oil) ──
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#111111')
        for i, r in enumerate(oil_results):
            f_span = np.linspace(r['f_std'] * 0.85, r['f_std'] * 1.15, 1000)
            s11 = s11_response(f_span, r['f_std'], r['Q'], r['Z0'])
            s11_ave = s11_response(f_span, r['f_ave'], r['Q'], r['Z0'])
            ax2.plot(f_span / 1e9, s11, color=colors[i], lw=1.5, alpha=0.4)
            ax2.plot(f_span / 1e9, s11_ave, color=colors[i], lw=2,
                     label=f"{r['label']}")
            ax2.axvline(r['f_std'] / 1e9, color=colors[i], lw=0.5, ls=':', alpha=0.3)
            ax2.axvline(r['f_ave'] / 1e9, color=colors[i], lw=1, ls='--', alpha=0.5)

        ax2.set_xlabel('Frequency (GHz)', color='white', fontsize=11)
        ax2.set_ylabel(r'$S_{11}$ (dB)', color='white', fontsize=11)
        ax2.set_title(f'Predicted S$_{{11}}$ Response (Mineral Oil)\n'
                      f'Z$_0$ = {Z0_oil:.0f}Ω',
                      color='#ffd93d', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax2.set_ylim(-30, 0)
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.15, color='white')
        for s in ax2.spines.values(): s.set_color('#333')

        # ── Panel 3: Impedance vs wire height ──
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor('#111111')
        h_sweep = np.linspace(0.3e-3, 10e-3, 200)
        for i, (name, eps_r, color) in enumerate([
            ('Air', EPS_R_AIR, '#00ffcc'),
            ('Mineral Oil', EPS_R_OIL, '#ffd93d'),
        ]):
            z_sweep = np.array([wire_over_ground_Z0(h, WIRE_DIA,
                        effective_permittivity(eps_r, WIRE_DIA, ENAMEL_THICKNESS, ENAMEL_EPS_R))
                        for h in h_sweep])
            ax3.plot(h_sweep * 1e3, z_sweep, color=color, lw=2.5, label=name)

        ax3.axhline(Z_SMA, color='#ff3366', lw=2, ls='--', label=r'50Ω SMA')
        ax3.axvline(h_front * 1e3, color='white', lw=1.5, ls=':', alpha=0.5,
                    label=f'Nom. height ({h_front*1e3:.1f}mm)')
        ax3.set_xlabel('Wire Height Above Ground (mm)', color='white', fontsize=11)
        ax3.set_ylabel(r'$Z_0$ (Ω)', color='white', fontsize=11)
        ax3.set_title('Characteristic Impedance vs Wire Height\n'
                      '(Image-charge model: wire above ground plane)',
                      color='white', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.15, color='white')
        ax3.set_ylim(0, 300)
        for s in ax3.spines.values(): s.set_color('#333')

        # ── Panel 4: Q-factor and skin depth ──
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor('#111111')
        f_sweep_ghz = np.linspace(0.1, 2.0, 200)
        f_sweep_hz = f_sweep_ghz * 1e9
        delta_sweep = skin_depth(f_sweep_hz) * 1e6  # μm
        ax4.plot(f_sweep_ghz, delta_sweep, color='#ff6b6b', lw=2.5, label='Skin depth')
        ax4.axhline(WIRE_RADIUS * 1e6, color='#ffd93d', lw=1.5, ls='--',
                    label=f'Wire radius ({WIRE_RADIUS*1e6:.0f}μm)')

        # Mark knot frequencies
        for i, r in enumerate(air_results):
            delta_at_f = skin_depth(r['f_std']) * 1e6
            ax4.scatter([r['f_std'] / 1e9], [delta_at_f], s=100, c=colors[i],
                       edgecolors='white', zorder=5)
            ax4.annotate(r['label'], (r['f_std'] / 1e9, delta_at_f),
                        textcoords='offset points', xytext=(10, 5),
                        color=colors[i], fontsize=8)

        ax4.set_xlabel('Frequency (GHz)', color='white', fontsize=11)
        ax4.set_ylabel('Skin Depth (μm)', color='white', fontsize=11)
        ax4.set_title('Skin Depth in 24 AWG Copper Wire\n'
                      '(AC resistance increases above skin-depth limit)',
                      color='white', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax4.tick_params(colors='white')
        ax4.grid(True, alpha=0.15, color='white')
        for s in ax4.spines.values(): s.set_color('#333')

        # ── Panel 5: Δf/f scaling law ──
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.set_facecolor('#111111')
        pq_vals_air = [r['pq_ppq'] for r in air_results]
        dff_air = [r['df_frac'] * 1e6 for r in air_results]
        pq_vals_oil = [r['pq_ppq'] for r in oil_results]
        dff_oil = [r['df_frac'] * 1e6 for r in oil_results]

        ax5.scatter(pq_vals_air, dff_air, s=150, c='#00ffcc', marker='o',
                   edgecolors='white', lw=1.5, zorder=5, label='Air')
        ax5.scatter(pq_vals_oil, dff_oil, s=150, c='#ffd93d', marker='s',
                   edgecolors='white', lw=1.5, zorder=5, label='Mineral Oil')
        pq_theory = np.linspace(0, 3, 100)
        ax5.plot(pq_theory, alpha * pq_theory * 1e6, 'w--', lw=2, alpha=0.6,
                 label=r'AVE: $\Delta f/f = \alpha \cdot pq/(p+q)$')
        ax5.set_xlabel(r'$pq/(p+q)$', color='white', fontsize=12)
        ax5.set_ylabel(r'$\Delta f / f$ (ppm)', color='white', fontsize=12)
        ax5.set_title('Chiral Scaling Law (Both Media)\n'
                      r'Points must fall on $\alpha \cdot pq/(p+q)$ line',
                      color='white', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=11, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax5.tick_params(colors='white')
        ax5.grid(True, alpha=0.15, color='white')
        for s in ax5.spines.values(): s.set_color('#333')

        # ── Panel 6: Summary table ──
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.set_facecolor('#111111')
        ax6.axis('off')

        summary = (
            "HOPF-01 ELECTRICAL MODEL SUMMARY\n"
            "─" * 38 + "\n\n"
            f"Board:  120×120mm, 2L FR-4, 1.6mm\n"
            f"Wire:   24 AWG enameled Cu (0.51mm)\n"
            f"Mount:  10mm nylon standoffs\n"
            f"Ground: B.Cu patches under SMA only\n\n"
            f"         AIR          OIL\n"
            f"ε_eff:  {eps_eff_air:.4f}       {eps_eff_oil:.4f}\n"
            f"Z₀:     {Z0_air:.0f} Ω        {Z0_oil:.0f} Ω\n"
            f"Γ_SMA:  {air_results[0]['gamma_dc']:.3f}        {oil_results[0]['gamma_dc']:.3f}\n"
            f"RL:     {air_results[0]['return_loss_dc']:.1f} dB      {oil_results[0]['return_loss_dc']:.1f} dB\n\n"
            f"Free-space (no ground plane) note:\n"
            f"  Since B.Cu ground is ONLY under SMA\n"
            f"  connectors, the wire is mostly in\n"
            f"  free space. Z₀ shown above is valid\n"
            f"  only near the feed point. The bulk\n"
            f"  of the resonator is in free space\n"
            f"  (Z → ∞ effectively).\n\n"
            f"  This makes the resonator behave more\n"
            f"  like a LOOP ANTENNA than a microstrip.\n"
            f"  Resonance is set by total wire length."
        )

        ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
                fontsize=10, color='#6bcaff', family='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a1a1a',
                         edgecolor='#6bcaff', alpha=0.9))
        ax6.set_title('Model Summary', color='white', fontsize=13,
                      fontweight='bold', pad=20)

        out_dir = project_root / "assets" / "sim_outputs"
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / "hopf_01_impedance_model.png"
        fig.savefig(out_path, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  📊 Figure saved: {out_path}")
    except ImportError:
        print("\n  ⚠️  matplotlib not available — skipping plots")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
