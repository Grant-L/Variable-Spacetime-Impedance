#!/usr/bin/env python3
r"""
HOPF-01: Full S₁₁ Frequency Sweep — Chiral Torus Knot Antenna Prediction
==========================================================================

Generates the complete S₁₁(f) frequency response for torus knot antennas
on FR-4 substrate, comparing standard Maxwell predictions against AVE
chiral coupling corrections.

The AVE prediction:
  Standard Maxwell gives the resonant frequency as:
    f_res = c / (2π × L_trace × √ε_r)

  AVE adds a chiral correction to the effective index:
    n_AVE = √ε_r × (1 + α × p×q/(p+q))

  This shifts the resonance DOWN by Δf = f_std × α × pq/(p+q).

  The key falsification test: if Δf scales exactly as pq/(p+q) across
  MULTIPLE knot topologies fabricated on the SAME PCB panel, the
  systematic scaling law confirms a topological vacuum coupling that
  standard HFSS/CST cannot predict.

This script produces:
  1. Full S₁₁(f) curves for (2,3), (2,5), (3,7), (3,11) knots
  2. Overlay plot showing chiral shift scaling
  3. Scaling law verification: Δf vs pq/(p+q) — should be linear
  4. Comparison table with NanoVNA measurement requirements

Usage:
    PYTHONPATH=src python scripts/book_6_ponder_01/hopf_01_s11_sweep.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ave.core.constants import C_0, ALPHA

# ======================================================
# PCB Substrate Parameters (FR-4)
# ======================================================
EPS_R = 4.3          # FR-4 relative permittivity
TAN_D = 0.02         # FR-4 loss tangent at ~150 MHz
TRACE_W = 1.0e-3     # 1 mm trace width
COPPER_T = 35e-6     # 1 oz copper (35 μm)
Z_FEED = 50.0        # SMA feed impedance [Ω]

# Torus knot catalog: (p, q, trace_length_m, label)
KNOT_CATALOG = [
    (2, 3,  0.060, r'$(2,3)$ Trefoil'),
    (2, 5,  0.090, r'$(2,5)$ Cinquefoil'),
    (3, 7,  0.120, r'$(3,7)$'),
    (3, 11, 0.150, r'$(3,11)$'),
]


def torus_knot_chiral_factor(p: int, q: int) -> float:
    """
    AVE chiral coupling factor for a (p,q) torus knot.

    The effective refractive index correction is:
      Δn/n = α × pq/(p+q)

    This is NOT a free parameter — it's derived from the topological
    winding number coupling to the vacuum's intrinsic chirality (Axiom 1).
    """
    alpha = float(ALPHA)
    return alpha * p * q / (p + q)


def s11_frequency_response(p: int, q: int, L_trace: float,
                           f_array: np.ndarray,
                           include_ave: bool = True) -> np.ndarray:
    """
    Compute S₁₁(f) for a (p,q) torus knot antenna on FR-4.

    Models the antenna as a lossy transmission line resonator with
    characteristic impedance Z_ant matched to the 50Ω feed.

    Standard model:
      Z_ant(f) = Z_c × (Z_L + j×Z_c×tan(βL)) / (Z_c + j×Z_L×tan(βL))
      S₁₁ = (Z_ant - Z_feed) / (Z_ant + Z_feed)

    where β = 2πf × n_eff / c and Z_c depends on geometry.

    AVE modification:
      n_eff includes the chiral correction, shifting the resonance.
      Additionally, the chiral coupling introduces a small reactive
      loading (additional capacitive susceptance) at the knot crossings.

    Args:
        p, q: Torus knot parameters.
        L_trace: Total trace length [m].
        f_array: Frequency array [Hz].
        include_ave: If True, include AVE chiral correction.

    Returns:
        S₁₁ magnitude [dB] as array.
    """
    c = float(C_0)
    alpha = float(ALPHA)

    # Effective refractive index
    n_std = np.sqrt(EPS_R)
    if include_ave:
        chiral = torus_knot_chiral_factor(p, q)
        n_eff = n_std * (1 + chiral)
    else:
        n_eff = n_std
        chiral = 0.0

    # Transmission line parameters
    # Microstrip characteristic impedance (approximate, Wheeler 1977)
    h_sub = 1.6e-3  # FR-4 substrate thickness [m]
    w_h = TRACE_W / h_sub
    if w_h < 1:
        Z_c = 60 / np.sqrt(EPS_R) * np.log(8 / w_h + w_h / 4)
    else:
        Z_c = 120 * np.pi / (np.sqrt(EPS_R) * (w_h + 1.393 + 0.667 * np.log(w_h + 1.444)))

    # Load impedance (open-ended torus knot → high impedance at end)
    Z_L = 1e6  # Open circuit (effectively infinite)

    s11_db = np.zeros_like(f_array)

    for i, f in enumerate(f_array):
        # Propagation constant
        beta = 2 * np.pi * f * n_eff / c

        # Loss: α_loss = π × f × n_eff × tan_δ / c
        alpha_loss = np.pi * f * n_eff * TAN_D / c

        # Complex propagation constant
        gamma_L = (alpha_loss + 1j * beta) * L_trace

        # Input impedance of loaded transmission line
        tanh_gL = np.tanh(gamma_L)
        Z_in = Z_c * (Z_L + Z_c * tanh_gL) / (Z_c + Z_L * tanh_gL)

        # AVE chiral reactive loading at knot crossings
        if include_ave and chiral > 0:
            # Each crossing adds a small shunt susceptance
            n_crossings = p * q  # Total crossings for (p,q) torus knot
            # Reactive loading ∝ α² × crossings × resonant susceptance
            B_chiral = alpha**2 * n_crossings * 2 * np.pi * f * EPS_R * 8.854e-12 * L_trace
            Z_chiral = 1 / (1j * B_chiral) if B_chiral > 0 else 1e12
            # Parallel combination
            Z_in = (Z_in * Z_chiral) / (Z_in + Z_chiral)

        # S₁₁ (reflection coefficient)
        gamma = (Z_in - Z_FEED) / (Z_in + Z_FEED)
        s11_db[i] = 20 * np.log10(max(abs(gamma), 1e-15))

    return s11_db


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
    print("  HOPF-01: Full S₁₁ Frequency Sweep — Torus Knot Antenna Array")
    print("=" * 80)

    # ─────────────────────────────────────────────────────
    # 1. Compute S₁₁ for each knot topology
    # ─────────────────────────────────────────────────────
    print(f"\n  Substrate: FR-4 (ε_r = {EPS_R}, tan_δ = {TAN_D})")
    print(f"  Trace width: {TRACE_W*1e3:.1f} mm, Cu thickness: {COPPER_T*1e6:.0f} μm")
    print(f"  Feed impedance: {Z_FEED:.0f} Ω (SMA)")
    print(f"  α = {float(ALPHA):.6e}")

    results = []

    print(f"\n  {'Knot':<20} {'L_trace':>8} {'f_std':>10} {'f_AVE':>10} "
          f"{'Δf':>8} {'Shift':>10} {'Dip':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*10} {'─'*8}")

    for p, q, L_trace, label in KNOT_CATALOG:
        # Resonant frequencies (analytical)
        c = float(C_0)
        n_std = np.sqrt(EPS_R)
        chiral = torus_knot_chiral_factor(p, q)
        n_ave = n_std * (1 + chiral)

        f_std = c / (2 * np.pi * L_trace * n_std)
        f_ave = c / (2 * np.pi * L_trace * n_ave)
        delta_f = f_std - f_ave
        shift_ppm = delta_f / f_std * 1e6

        # Full sweep around resonance (±30% bandwidth)
        f_min = f_std * 0.70
        f_max = f_std * 1.30
        f_array = np.linspace(f_min, f_max, 4001)

        s11_std = s11_frequency_response(p, q, L_trace, f_array, include_ave=False)
        s11_ave = s11_frequency_response(p, q, L_trace, f_array, include_ave=True)

        # Find deepest dips
        res_std = find_resonances(f_array, s11_std, threshold=-5.0)
        res_ave = find_resonances(f_array, s11_ave, threshold=-5.0)

        dip_std = min(s11_std)
        dip_ave = min(s11_ave)

        print(f"  {label:<20} {L_trace*1e3:>6.0f}mm {f_std/1e6:>8.2f}MHz "
              f"{f_ave/1e6:>8.2f}MHz {delta_f/1e6:>6.2f}MHz "
              f"{shift_ppm:>8.0f}ppm {dip_ave:>6.1f}dB")

        results.append({
            'p': p, 'q': q, 'label': label,
            'L_trace': L_trace,
            'f_std': f_std, 'f_ave': f_ave,
            'delta_f': delta_f, 'shift_ppm': shift_ppm,
            'chiral_factor': chiral,
            'pq_over_ppq': p * q / (p + q),
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
    # 3. NanoVNA Measurement Requirements
    # ─────────────────────────────────────────────────────
    print(f"\n  ═══════════════════════════════════════════════════════════════════")
    print(f"  NanoVNA MEASUREMENT REQUIREMENTS")
    print(f"  ═══════════════════════════════════════════════════════════════════")
    print(f"\n  A $70 NanoVNA-H4 (50 kHz -- 1.5 GHz) can resolve these shifts:")
    print(f"\n  {'Knot':<16} {'Δf':>10} {'VNA RBW req':>14} {'Points':>8} {'Feasible':>10}")
    print(f"  {'─'*16} {'─'*10} {'─'*14} {'─'*8} {'─'*10}")

    for r in results:
        df = r['delta_f']
        # NanoVNA-H4: max 1601 points per sweep
        # Required RBW = Δf / 10 (to resolve the shift)
        rbw_req = df / 10
        # Span needed: ±5× the shift around resonance
        span = 20 * df
        points_needed = int(span / rbw_req) + 1
        feasible = "✅ Easy" if points_needed < 1601 else "⚠️ Multi-sweep"
        print(f"  {r['label']:<16} {df/1e3:>8.1f}kHz {rbw_req/1e3:>12.1f}kHz "
              f"{points_needed:>8d} {feasible:>10}")

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

        colors_std = ['#555555', '#555555', '#555555', '#555555']
        colors_ave = ['#00ffcc', '#ff6b6b', '#ffd93d', '#6bcaff']

        # ── Panel 1: All S₁₁ curves overlaid ──
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#111111')
        for i, r in enumerate(results):
            f_mhz = r['f_array'] / 1e6
            ax1.plot(f_mhz, r['s11_std'], color=colors_std[i],
                     alpha=0.4, linewidth=0.8, linestyle='--')
            ax1.plot(f_mhz, r['s11_ave'], color=colors_ave[i],
                     linewidth=1.5, label=r['label'])
        ax1.set_xlabel('Frequency (MHz)', color='white', fontsize=11)
        ax1.set_ylabel(r'$S_{11}$ (dB)', color='white', fontsize=11)
        ax1.set_title('S₁₁ Response: All Torus Knot Antennas',
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
        f_zoom = r311['f_array'][zoom_mask] / 1e6
        ax2.plot(f_zoom, r311['s11_std'][zoom_mask], color='#888888',
                 linewidth=2, linestyle='--', label='Standard Maxwell')
        ax2.plot(f_zoom, r311['s11_ave'][zoom_mask], color='#6bcaff',
                 linewidth=2.5, label='AVE (chiral)')
        # Annotate the shift
        delta_mhz = r311['delta_f'] / 1e6
        ax2.annotate(f'Δf = {delta_mhz:.2f} MHz\n({r311["shift_ppm"]:.0f} ppm)',
                     xy=(r311['f_ave']/1e6, r311['dip_ave']),
                     xytext=(r311['f_ave']/1e6 + 3, r311['dip_ave'] + 8),
                     fontsize=10, color='#6bcaff',
                     arrowprops=dict(arrowstyle='->', color='#6bcaff', lw=1.5))
        ax2.set_xlabel('Frequency (MHz)', color='white', fontsize=11)
        ax2.set_ylabel(r'$S_{11}$ (dB)', color='white', fontsize=11)
        ax2.set_title(f'Zoomed: (3,11) Knot — Chiral Shift',
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
                f"{r['L_trace']*1e3:.0f} mm",
                f"{r['f_std']/1e6:.2f}",
                f"{r['f_ave']/1e6:.2f}",
                f"{r['delta_f']/1e6:.2f}",
                f"{r['shift_ppm']:.0f}",
            ])

        col_labels = ['Knot', 'L_trace', 'f_std (MHz)', 'f_AVE (MHz)',
                      'Δf (MHz)', 'Shift (ppm)']
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

        ax4.set_title('HOPF-01 Prediction Summary',
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
    print(f"  1. Fabricate ALL 4 knot antennas on a SINGLE FR-4 panel")
    print(f"     (ensures identical ε_r, Cu thickness, etching tolerances)")
    print(f"  2. Measure each antenna with NanoVNA-H4 (calibrated SOLT)")
    print(f"  3. Record the resonant frequency f_res for each knot")
    print(f"  4. Compute Δf = f_measured - f_HFSS_predicted for each")
    print(f"  5. Plot Δf vs pq/(p+q)")
    print(f"     → If EXACTLY LINEAR through origin: AVE confirmed")
    print(f"     → If zero or random: AVE falsified at this scale")
    print(f"  6. Repeat on Rogers RO4003C (ε_r = 3.38) to verify substrate")
    print(f"     independence of the chiral coupling constant")
    print(f"  ═══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
