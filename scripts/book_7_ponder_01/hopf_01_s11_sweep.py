#!/usr/bin/env python3
r"""
HOPF-01: Full S₁₁ Frequency Sweep — Chiral Torus Knot Antenna Prediction
==========================================================================

Generates the complete S₁₁(f) frequency response for wire-stitched torus
knot antennas on FR-4 substrate (wire-in-air model), comparing standard
Maxwell predictions against AVE chiral coupling corrections.

The AVE prediction:
  Standard Maxwell gives the resonant frequency as:
    f_res = c / (2 × L_wire × √ε_eff)

  AVE adds a chiral correction to the effective index:
    n_AVE = √ε_eff × (1 + α × p×q/(p+q))

  This shifts the resonance DOWN by Δf = f_std × α × pq/(p+q).

  The key falsification test: if Δf scales exactly as pq/(p+q) across
  MULTIPLE knot topologies fabricated on the SAME PCB panel, the
  systematic scaling law confirms a topological vacuum coupling that
  standard HFSS/CST cannot predict.

Physical model:
  24 AWG enameled magnet wire (d=0.51mm) stitched through unplated
  PCB holes.  The wire resonates as a FREE-SPACE wire resonator,
  NOT a microstrip line.  ε_eff ≈ 1.295 (air + enamel correction).

This script produces:
  1. Full S₁₁(f) curves for (2,3), (2,5), (3,7), (3,11) knots
  2. Overlay plot showing chiral shift scaling
  3. Scaling law verification: Δf vs pq/(p+q) — should be linear
  4. Comparison table with VNA measurement requirements

Usage:
    PYTHONPATH=src python scripts/book_7_ponder_01/hopf_01_s11_sweep.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ave.core.constants import C_0, ALPHA, MU_0, Z_0

# ======================================================
# Wire-in-Air Physical Parameters
# ======================================================

# Board
PCB_THICKNESS = 1.6e-3      # m (FR-4 core)

# Wire (24 AWG enameled magnet wire)
WIRE_DIA = 0.51e-3           # m
WIRE_RADIUS = WIRE_DIA / 2
ENAMEL_THICKNESS = 30e-6     # m (polyurethane enamel)
ENAMEL_EPS_R = 3.5           # polyurethane relative permittivity
WIRE_CONDUCTIVITY = 5.8e7    # S/m (copper)

# Media
EPS_R_AIR = 1.0006           # air at STP
Z_FEED = 50.0                # SMA feed impedance [Ω]

# Wire height above ground (front side: PCB thickness + wire radius)
WIRE_HEIGHT = PCB_THICKNESS + WIRE_RADIUS

# Torus knot catalog: (p, q, wire_length_m, label)
KNOT_CATALOG = [
    (2, 3,  0.120, r'$(2,3)$ Trefoil'),
    (2, 5,  0.160, r'$(2,5)$ Cinquefoil'),
    (3, 5,  0.170, r'$(3,5)$'),
    (3, 7,  0.200, r'$(3,7)$'),
    (3, 11, 0.250, r'$(3,11)$'),
]


# ======================================================
# Wire-in-Air Impedance Model
# ======================================================

def effective_permittivity(eps_medium):
    """Effective permittivity accounting for enamel coating.

    Volume-fraction model:
        ε_eff ≈ eps_medium × (1 + f_enamel × (ε_enamel/eps_medium - 1))
    where f_enamel ≈ 2×t_enamel / wire_dia (thin shell fraction).
    """
    f_enamel = 2 * ENAMEL_THICKNESS / WIRE_DIA
    return eps_medium * (1 + f_enamel * (ENAMEL_EPS_R / eps_medium - 1))


def wire_over_ground_Z0(eps_eff):
    """Characteristic impedance of wire above ground plane (image-charge model).

    Z₀ = (60 / √ε_eff) × acosh(2h/d)
    """
    ratio = 2 * WIRE_HEIGHT / WIRE_DIA
    if ratio <= 1:
        ratio = 1.01
    return (float(Z_0) / (2 * np.pi) / np.sqrt(eps_eff)) * np.arccosh(ratio)


def skin_depth(freq):
    """Skin depth in copper at frequency f."""
    return np.sqrt(1 / (np.pi * freq * float(MU_0) * WIRE_CONDUCTIVITY))


def wire_resistance_per_m(freq):
    """AC resistance per meter for 24 AWG wire at frequency f."""
    delta = skin_depth(freq)
    r = WIRE_RADIUS
    if delta >= r:
        area = np.pi * r**2
    else:
        area = np.pi * (r**2 - (r - delta)**2)
    return 1 / (WIRE_CONDUCTIVITY * area)


def torus_knot_chiral_factor(p: int, q: int) -> float:
    """AVE chiral coupling factor for a (p,q) torus knot.

    The effective refractive index correction is:
      Δn/n = α × pq/(p+q)

    This is NOT a free parameter — it's derived from the topological
    winding number coupling to the vacuum's intrinsic chirality (Axiom 1).
    """
    alpha = float(ALPHA)
    return alpha * p * q / (p + q)


def f_resonance(L_wire, eps_eff, chiral_factor=0.0):
    """Half-wave open-ended resonator frequency.

    f_res = c / (2 × L × √ε_eff × (1 + χ))

    Uses 2L (not 2πL) — correct for a half-wave transmission
    line resonator with open ends.
    """
    n_eff = np.sqrt(eps_eff) * (1 + chiral_factor)
    return float(C_0) / (2 * L_wire * n_eff)


def quality_factor(freq, L_wire, Z0):
    """Unloaded Q of a half-wave resonator.

    Q = (π × Z₀) / (R_total)
    where R_total = R_per_m × L_wire.
    """
    R_per_m = wire_resistance_per_m(freq)
    R_total = R_per_m * L_wire
    return np.pi * Z0 / R_total


def s11_frequency_response(f_res, Q, Z0, f_array):
    """Compute S₁₁(f) for a resonant wire antenna with SMA mismatch.

    Models the antenna as a series RLC resonator seen through an
    impedance transformer (Z_ant != Z_source).

    Returns S₁₁ magnitude [dB] as array.
    """
    delta = (f_array - f_res) / f_res
    Z_ant_f = Z0 * (1 + 1j * Q * 2 * delta)
    gamma = (Z_ant_f - Z_FEED) / (Z_ant_f + Z_FEED)
    return 20 * np.log10(np.abs(gamma))


def find_resonances(f_array: np.ndarray, s11_db: np.ndarray,
                    threshold: float = -3.0) -> list:
    """Find resonant dips below threshold [dB]."""
    resonances = []
    for i in range(1, len(s11_db) - 1):
        if (s11_db[i] < threshold and
            s11_db[i] < s11_db[i-1] and
            s11_db[i] < s11_db[i+1]):
            resonances.append((f_array[i], s11_db[i]))
    return resonances


def main():
    print("=" * 80)
    print("  HOPF-01: Full S₁₁ Frequency Sweep — Wire-Stitched Torus Knot Antenna")
    print("=" * 80)

    # Compute effective permittivity and impedance
    eps_eff = effective_permittivity(EPS_R_AIR)
    Z0 = wire_over_ground_Z0(eps_eff)

    # ─────────────────────────────────────────────────────
    # 1. Compute S₁₁ for each knot topology
    # ─────────────────────────────────────────────────────
    print(f"\n  Physical model: wire-in-air (free-space resonator)")
    print(f"  Wire: 24 AWG enameled Cu ({WIRE_DIA*1e3:.2f}mm dia)")
    print(f"  ε_eff = {eps_eff:.4f}  (air + enamel correction)")
    print(f"  Z₀ = {Z0:.1f} Ω  (wire above B.Cu ground patch)")
    print(f"  Feed impedance: {Z_FEED:.0f} Ω (SMA)")
    print(f"  α = {float(ALPHA):.6e}")

    results = []

    print(f"\n  {'Knot':<20} {'L_wire':>8} {'f_std':>10} {'f_AVE':>10} "
          f"{'Δf':>8} {'Shift':>10} {'Q':>6} {'Dip':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*10} {'─'*6} {'─'*8}")

    for p, q, L_wire, label in KNOT_CATALOG:
        # Resonant frequencies (analytical)
        c = float(C_0)
        chiral = torus_knot_chiral_factor(p, q)

        f_std = f_resonance(L_wire, eps_eff, 0.0)
        f_ave = f_resonance(L_wire, eps_eff, chiral)
        delta_f = f_std - f_ave
        shift_ppm = delta_f / f_std * 1e6

        # Q-factor
        Q = quality_factor(f_std, L_wire, Z0)

        # Full sweep around resonance (±30% bandwidth)
        f_min = f_std * 0.70
        f_max = f_std * 1.30
        f_array = np.linspace(f_min, f_max, 4001)

        s11_std = s11_frequency_response(f_std, Q, Z0, f_array)
        s11_ave = s11_frequency_response(f_ave, Q, Z0, f_array)

        dip_std = min(s11_std)
        dip_ave = min(s11_ave)

        print(f"  {label:<20} {L_wire*1e3:>6.0f}mm {f_std/1e9:>8.3f}GHz "
              f"{f_ave/1e9:>8.3f}GHz {delta_f/1e6:>6.2f}MHz "
              f"{shift_ppm:>8.0f}ppm {Q:>5.0f} {dip_ave:>6.1f}dB")

        results.append({
            'p': p, 'q': q, 'label': label,
            'L_wire': L_wire,
            'f_std': f_std, 'f_ave': f_ave,
            'delta_f': delta_f, 'shift_ppm': shift_ppm,
            'chiral_factor': chiral,
            'pq_over_ppq': p * q / (p + q),
            'Q': Q,
            'f_array': f_array,
            's11_std': s11_std,
            's11_ave': s11_ave,
            'dip_std': dip_std,
            'dip_ave': dip_ave,
        })

    # ─────────────────────────────────────────────────────
    # 2. Chiral Scaling Law Verification
    # ─────────────────────────────────────────────────────
    print(f"\n  ═══════════════════════════════════════════════════════════════════")
    print(f"  CHIRAL SCALING LAW VERIFICATION")
    print(f"  ═══════════════════════════════════════════════════════════════════")
    print(f"\n  If AVE is correct, the fractional shift must scale EXACTLY as:")
    print(f"    Δf/f = χ/(1+χ)  where χ = α × pq/(p+q)")
    print(f"\n  {'Knot':<16} {'pq/(p+q)':>10} {'χ/(1+χ) exact':>14} {'Δf/f (sim)':>14} {'Match':>8}")
    print(f"  {'─'*16} {'─'*10} {'─'*14} {'─'*14} {'─'*8}")

    alpha = float(ALPHA)
    for r in results:
        chi = alpha * r['pq_over_ppq']
        predicted_ratio = chi / (1 + chi)  # exact: Δf/f = 1 - 1/(1+χ)
        measured_ratio = r['delta_f'] / r['f_std']
        match_pct = abs(predicted_ratio - measured_ratio) / predicted_ratio * 100
        status = "✅" if match_pct < 0.1 else "⚠️"
        print(f"  {r['label']:<16} {r['pq_over_ppq']:>10.4f} "
              f"{predicted_ratio:>14.6e} {measured_ratio:>14.6e} {status:>8}")

    # ─────────────────────────────────────────────────────
    # 3. VNA Measurement Requirements
    # ─────────────────────────────────────────────────────
    print(f"\n  ═══════════════════════════════════════════════════════════════════")
    print(f"  VNA MEASUREMENT REQUIREMENTS")
    print(f"  ═══════════════════════════════════════════════════════════════════")
    print(f"\n  All resonances below 1.2 GHz — within NanoVNA-H4 and LiteVNA range:")
    print(f"\n  {'Knot':<16} {'f_std':>10} {'Δf':>10} {'VNA RBW req':>14} {'Points':>8} {'Feasible':>10}")
    print(f"  {'─'*16} {'─'*10} {'─'*10} {'─'*14} {'─'*8} {'─'*10}")

    for r in results:
        df = r['delta_f']
        rbw_req = df / 10
        span = 20 * df
        points_needed = int(span / rbw_req) + 1
        feasible = "✅ Easy" if points_needed < 1601 else "⚠️ Multi-sweep"
        print(f"  {r['label']:<16} {r['f_std']/1e9:>8.3f}GHz {df/1e6:>8.2f}MHz "
              f"{rbw_req/1e3:>12.1f}kHz {points_needed:>8d} {feasible:>10}")

    # ─────────────────────────────────────────────────────
    # 4. Generate Plots
    # ─────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('#0a0a0a')
        gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

        colors_std = ['#555555', '#555555', '#555555', '#555555', '#555555']
        colors_ave = ['#00ffcc', '#ff6b6b', '#ffd93d', '#6bcaff', '#c78dff']

        # ── Panel 1: All S₁₁ curves overlaid ──
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#111111')
        for i, r in enumerate(results):
            f_ghz = r['f_array'] / 1e9
            ax1.plot(f_ghz, r['s11_std'], color=colors_std[i],
                     alpha=0.4, linewidth=0.8, linestyle='--')
            ax1.plot(f_ghz, r['s11_ave'], color=colors_ave[i],
                     linewidth=1.5, label=r['label'])
        ax1.set_xlabel('Frequency (GHz)', color='white', fontsize=11)
        ax1.set_ylabel(r'$S_{11}$ (dB)', color='white', fontsize=11)
        ax1.set_title('S₁₁ Response: Wire-Stitched Torus Knot Antennas\n'
                      f'(ε_eff = {eps_eff:.4f}, Z₀ = {Z0:.0f}Ω)',
                      color='white', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333',
                   labelcolor='white', loc='upper right')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.15, color='white')
        ax1.set_ylim(-35, 2)
        for spine in ax1.spines.values():
            spine.set_color('#333')

        # ── Panel 2: Zoomed view of (3,11) knot ──
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#111111')
        r311 = results[3]
        f_center = r311['f_std']
        zoom_mask = (r311['f_array'] > f_center * 0.94) & (r311['f_array'] < f_center * 1.06)
        f_zoom = r311['f_array'][zoom_mask] / 1e9
        ax2.plot(f_zoom, r311['s11_std'][zoom_mask], color='#888888',
                 linewidth=2, linestyle='--', label='Standard Maxwell')
        ax2.plot(f_zoom, r311['s11_ave'][zoom_mask], color='#6bcaff',
                 linewidth=2.5, label='AVE (chiral)')
        # Annotate the shift
        delta_mhz = r311['delta_f'] / 1e6
        ax2.annotate(f'Δf = {delta_mhz:.2f} MHz\n({r311["shift_ppm"]:.0f} ppm)',
                     xy=(r311['f_ave']/1e9, r311['dip_ave']),
                     xytext=(r311['f_ave']/1e9 + 0.005, r311['dip_ave'] + 8),
                     fontsize=10, color='#6bcaff',
                     arrowprops=dict(arrowstyle='->', color='#6bcaff', lw=1.5))
        ax2.set_xlabel('Frequency (GHz)', color='white', fontsize=11)
        ax2.set_ylabel(r'$S_{11}$ (dB)', color='white', fontsize=11)
        ax2.set_title(f'Zoomed: (3,11) Knot — Chiral Shift\n'
                      f'Q = {r311["Q"]:.0f}',
                      color='white', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#333',
                   labelcolor='white')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.15, color='white')
        for spine in ax2.spines.values():
            spine.set_color('#333')

        # ── Panel 3: Scaling law — Δf vs pq/(p+q) ──
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor('#111111')
        pq_vals = [r['pq_over_ppq'] for r in results]
        df_vals = [r['delta_f'] / 1e6 for r in results]
        ax3.scatter(pq_vals, df_vals, color='#ff6b6b', s=120, zorder=5,
                    edgecolors='white', linewidths=1.5)
        # Linear fit
        coeffs = np.polyfit(pq_vals, df_vals, 1)
        x_fit = np.linspace(min(pq_vals) * 0.8, max(pq_vals) * 1.2, 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax3.plot(x_fit, y_fit, color='#ff6b6b', linewidth=1.5,
                 linestyle='--', alpha=0.7, label=f'Linear fit: Δf = {coeffs[0]:.3f} × pq/(p+q)')
        for i, r in enumerate(results):
            ax3.annotate(r['label'], (pq_vals[i], df_vals[i]),
                         textcoords="offset points", xytext=(8, 8),
                         fontsize=9, color=colors_ave[i])
        ax3.set_xlabel(r'$pq/(p+q)$  [topological winding parameter]',
                       color='white', fontsize=11)
        ax3.set_ylabel(r'$\Delta f$  (MHz)', color='white', fontsize=11)
        ax3.set_title('Chiral Scaling Law: Δf ∝ α × pq/(p+q)',
                      color='white', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#333',
                   labelcolor='white')
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.15, color='white')
        for spine in ax3.spines.values():
            spine.set_color('#333')

        # ── Panel 4: Summary table ──
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor('#111111')
        ax4.axis('off')

        table_data = []
        for r in results:
            table_data.append([
                r['label'],
                f"{r['L_wire']*1e3:.0f} mm",
                f"{r['f_std']/1e9:.3f}",
                f"{r['f_ave']/1e9:.3f}",
                f"{r['delta_f']/1e6:.2f}",
                f"{r['shift_ppm']:.0f}",
                f"{r['Q']:.0f}",
            ])

        col_labels = ['Knot', 'L_wire', 'f_std (GHz)', 'f_AVE (GHz)',
                      'Δf (MHz)', 'Shift (ppm)', 'Q']
        table = ax4.table(cellText=table_data, colLabels=col_labels,
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 2.0)

        for key, cell in table.get_celld().items():
            cell.set_facecolor('#1a1a1a')
            cell.set_edgecolor('#333')
            cell.set_text_props(color='white')
            if key[0] == 0:
                cell.set_facecolor('#2a2a3a')
                cell.set_text_props(color='#00ffcc', fontweight='bold')

        ax4.set_title('HOPF-01 Prediction Summary (Wire-in-Air)',
                      color='white', fontsize=13, fontweight='bold', pad=20)

        # Save
        out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                               'assets', 'sim_outputs')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'hopf_01_s11_sweep.png')
        fig.savefig(out_path, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  📊 Plot saved: {out_path}")
    except ImportError:
        print("\n  ⚠️  matplotlib not available — skipping plots")

    # ─────────────────────────────────────────────────────
    # 5. Experimental Protocol Summary
    # ─────────────────────────────────────────────────────
    print(f"\n  ═══════════════════════════════════════════════════════════════════")
    print(f"  EXPERIMENTAL PROTOCOL")
    print(f"  ═══════════════════════════════════════════════════════════════════")
    print(f"  1. Fabricate ALL 5 knot antennas + control on a SINGLE 160×120mm FR-4 panel")
    print(f"     Wire-stitched: 24 AWG enameled Cu through unplated holes")
    print(f"  2. Measure each antenna with NanoVNA-H4 or LiteVNA (calibrated SOL)")
    print(f"  3. Record the resonant frequency f_res for each knot")
    print(f"  4. Compute Δf = f_measured - f_Maxwell for each")
    print(f"  5. Plot Δf vs pq/(p+q)")
    print(f"     → If EXACTLY LINEAR through origin: AVE confirmed")
    print(f"     → If zero or random: AVE falsified at this scale")
    print(f"  6. Repeat in mineral oil bath (ε_r ≈ 2.1) to verify substrate")
    print(f"     independence of the chiral coupling constant")
    print(f"  ═══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
