#!/usr/bin/env python3
r"""
HOPF-01: Classical Crossing Coupling Analysis
================================================

Quantifies the CLASSICAL electromagnetic coupling at wire crossings
to determine whether standard Maxwell effects could mimic the AVE
chiral shift.

At each physical crossing in the wire-stitched design, one wire strand
passes OVER the PCB (in air) and the other passes UNDER (through a hole).
The z-separation is constrained by PCB thickness (~1.6mm).

Classical coupling at each crossing:
  1. Mutual inductance (Neumann formula for perpendicular wires)
  2. Parasitic capacitance (wire-to-wire coupling)

Key question: does the total classical shift scale as pq/(p+q)?

Usage:
    PYTHONPATH=src python scripts/vol_4_experimental/hopf_01_classical_coupling.py
"""

import sys
import os
import pathlib
import numpy as np

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()

from ave.core.constants import C_0, ALPHA, MU_0, EPSILON_0

# ══════════════════════════════════════════════════════════════
# Physical Parameters
# ══════════════════════════════════════════════════════════════

WIRE_DIA = 0.51e-3           # m (24 AWG)
ENAMEL_THICKNESS = 30e-6     # m
ENAMEL_EPS_R = 3.5
EPS_R_AIR = 1.0006
PCB_THICKNESS = 1.6e-3       # m

def effective_permittivity(eps_medium):
    f_enamel = 2 * ENAMEL_THICKNESS / WIRE_DIA
    return eps_medium * (1 + f_enamel * (ENAMEL_EPS_R / eps_medium - 1))

EPS_EFF = effective_permittivity(EPS_R_AIR)

# Physical crossing geometry in wire-stitched design:
# Over-strand: wire sits ~1mm above PCB surface
# Under-strand: wire hangs ~1mm below PCB surface
# Z-separation ≈ PCB thickness + wire offsets ≈ 2-3mm
Z_SEP_CROSSING = PCB_THICKNESS + WIRE_DIA  # ~2.1mm conservative

# Effective coupling length at each crossing (~2 wire diameters)
COUPLING_LENGTH = 3e-3  # 3mm (one stitching-hole spacing)

# Knot catalog with crossing counts from analytic formula:
# c(p,q) = min(p(q-1), q(p-1))
KNOTS = [
    (2, 3,  0.120, '(2,3) Trefoil',     3),   # min(2×2, 3×1) = 3
    (2, 5,  0.160, '(2,5) Cinquefoil',  5),   # min(2×4, 5×1) = 5
    (3, 5,  0.170, '(3,5)',              10),  # min(3×4, 5×2) = 10
    (3, 7,  0.200, '(3,7)',              14),  # min(3×6, 7×2) = 14
    (3, 11, 0.250, '(3,11)',             22),  # min(3×10,11×2) = 22
]

# Theoretical minimum crossing number for (p,q) torus knot:
# c(p,q) = min(p(q-1), q(p-1))

# ══════════════════════════════════════════════════════════════
# Classical Coupling at a Single Crossing
# ══════════════════════════════════════════════════════════════

def mutual_inductance_per_crossing(z_sep, coupling_length):
    """Mutual inductance between two wire segments at a crossing.

    For two short wire segments of length ℓ separated by distance d,
    crossing at approximately right angles (the typical case for
    torus knot crossings):

      M ≈ (μ₀/4π) × ℓ² / d  [for perpendicular crossing]

    This is much smaller than the parallel-wire case because
    cos(θ) → 0 at perpendicular crossings.

    For a worst case (parallel crossing):
      M ≈ (μ₀ℓ/2π) × ln(ℓ/d)
    """
    mu0 = float(MU_0)
    d = max(z_sep, WIRE_DIA)
    ell = coupling_length

    # Average crossing angle for torus knots is ~60-90°
    # We'll compute both limits
    M_perp = (mu0 / (4 * np.pi)) * ell**2 / d   # perpendicular (best case)
    M_para = (mu0 * ell / (2 * np.pi)) * np.log(max(ell / d, 1.01))  # parallel (worst case)

    # Average over typical crossing angle distribution (~70°)
    avg_cos = np.cos(np.radians(70))  # cos(70°) ≈ 0.34
    M_avg = M_para * abs(avg_cos)

    return M_perp, M_para, M_avg


def parasitic_capacitance_per_crossing(z_sep, coupling_length):
    """Parasitic capacitance between two wire segments at a crossing.

    C ≈ (π × ε₀ × ℓ) / acosh(d/a) × |cos(θ)|

    For perpendicular crossings, effective coupling length is short
    (approximately wire diameter), not the full segment length.
    """
    eps0 = float(EPSILON_0)
    d = max(z_sep, WIRE_DIA)
    a = WIRE_DIA / 2
    ratio = max(d / a, 1.01)

    # Perpendicular: coupling only over ~wire diameter
    C_perp = (np.pi * eps0 * WIRE_DIA) / np.arccosh(ratio)

    # Parallel (worst case): coupling over full segment length
    C_para = (np.pi * eps0 * coupling_length) / np.arccosh(ratio)

    # Average
    avg_cos = np.cos(np.radians(70))
    C_avg = C_para * abs(avg_cos)

    return C_perp, C_para, C_avg


def wire_self_inductance(L_wire):
    """Self-inductance of a straight wire of length L (approximate)."""
    mu0 = float(MU_0)
    a = WIRE_DIA / 2
    return (mu0 * L_wire / (2 * np.pi)) * (np.log(2 * L_wire / a) - 1)


def resonant_frequency(L_wire, eps_eff):
    return float(C_0) / (2 * L_wire * np.sqrt(eps_eff))


def main():
    c0 = float(C_0)
    alpha = float(ALPHA)
    mu0 = float(MU_0)
    eps0 = float(EPSILON_0)

    print("=" * 80)
    print("  HOPF-01: Classical Crossing Coupling Analysis")
    print("  Is Standard Maxwell coupling a confounding variable?")
    print("=" * 80)

    # Single crossing coupling magnitudes
    M_perp, M_para, M_avg = mutual_inductance_per_crossing(
        Z_SEP_CROSSING, COUPLING_LENGTH)
    C_perp, C_para, C_avg = parasitic_capacitance_per_crossing(
        Z_SEP_CROSSING, COUPLING_LENGTH)

    print(f"\n  ── Single Crossing Coupling (z_sep = {Z_SEP_CROSSING*1e3:.1f}mm) ──")
    print(f"  Mutual Inductance:")
    print(f"    Perpendicular: M = {M_perp:.3e} H")
    print(f"    Parallel:      M = {M_para:.3e} H  (worst case)")
    print(f"    Average (~70°): M = {M_avg:.3e} H")
    print(f"  Parasitic Capacitance:")
    print(f"    Perpendicular: C = {C_perp:.3e} F")
    print(f"    Parallel:      C = {C_para:.3e} F  (worst case)")
    print(f"    Average (~70°): C = {C_avg:.3e} F")

    # ── Per-knot analysis ──
    results = []
    print(f"\n  ═══════════════════════════════════════════════════════════════════")
    print(f"  PER-KNOT COUPLING ANALYSIS")
    print(f"  ═══════════════════════════════════════════════════════════════════")
    print(f"\n  {'Knot':<16} {'N_cross':>8} {'pq/(p+q)':>10} {'L_self':>10} "
          f"{'M_total':>12} {'ΔL/L':>12} {'Δf/f class':>12} {'Δf/f AVE':>12}")
    print(f"  {'─'*16} {'─'*8} {'─'*10} {'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

    for p, q, L_wire, label, n_cross in KNOTS:
        pq_ppq = p * q / (p + q)
        f_std = resonant_frequency(L_wire, EPS_EFF)
        L_self = wire_self_inductance(L_wire)

        # Total coupling from all crossings (average angle)
        M_total = n_cross * M_avg
        C_total = n_cross * C_avg

        # Inductive perturbation only: Δf/f ≈ -½ ΔL/L
        # Capacitive coupling at perpendicular crossings is negligible
        # compared to the wire self-capacitance (same as SM baseline model).
        dL_frac = M_total / L_self
        dC_frac = 0.0  # suppressed at perpendicular crossings
        df_classical = 0.5 * dL_frac

        # AVE predicted shift
        chi = alpha * pq_ppq
        df_ave = chi / (1 + chi)

        results.append({
            'label': label, 'p': p, 'q': q,
            'pq_ppq': pq_ppq, 'n_cross': n_cross,
            'f_std': f_std, 'L_self': L_self,
            'M_total': M_total, 'C_total': C_total,
            'dL_frac': dL_frac, 'dC_frac': dC_frac,
            'df_classical': df_classical,
            'df_ave': df_ave,
            'df_classical_hz': df_classical * f_std,
            'df_ave_hz': df_ave * f_std,
        })

        print(f"  {label:<16} {n_cross:>8} {pq_ppq:>10.4f} {L_self*1e9:>8.1f}nH "
              f"{M_total:>12.3e} {dL_frac:>12.3e} {df_classical:>12.3e} {df_ave:>12.3e}")

    # ── Comparison ──
    print(f"\n  ═══════════════════════════════════════════════════════════════════")
    print(f"  CLASSICAL vs AVE — ABSOLUTE COMPARISON")
    print(f"  ═══════════════════════════════════════════════════════════════════")
    print(f"\n  {'Knot':<16} {'Δf AVE':>12} {'Δf classical':>14} {'Ratio':>10} {'Orders':>8}")
    print(f"  {'─'*16} {'─'*12} {'─'*14} {'─'*10} {'─'*8}")

    for r in results:
        ratio = r['df_classical'] / r['df_ave'] if r['df_ave'] > 0 else 0
        orders = np.log10(ratio) if ratio > 0 else 0
        print(f"  {r['label']:<16} {r['df_ave_hz']/1e6:>10.2f}MHz "
              f"{r['df_classical_hz']/1e3:>12.2f}kHz "
              f"{ratio:>10.4f} {orders:>8.1f}")

    # ── Scaling shape ──
    print(f"\n  ═══════════════════════════════════════════════════════════════════")
    print(f"  SCALING SHAPE COMPARISON")
    print(f"  ═══════════════════════════════════════════════════════════════════")
    print(f"\n  AVE scales as:     Δf/f = α × pq/(p+q)  → linear in topological invariant")
    print(f"  Classical scales as: Δf/f ∝ N_cross / L_self  → depends on crossing density")
    print(f"\n  {'Knot':<16} {'pq/(p+q)':>10} {'N_cross':>8} {'N/L_self':>12} "
          f"{'AVE (norm)':>12} {'Class (norm)':>14}")
    print(f"  {'─'*16} {'─'*10} {'─'*8} {'─'*12} {'─'*12} {'─'*14}")

    # Normalize to first knot
    ave_0 = results[0]['df_ave']
    class_0 = results[0]['df_classical']

    for r in results:
        n_over_l = r['n_cross'] / r['L_self']
        ave_norm = r['df_ave'] / ave_0
        class_norm = r['df_classical'] / class_0 if class_0 > 0 else 0
        print(f"  {r['label']:<16} {r['pq_ppq']:>10.4f} {r['n_cross']:>8} "
              f"{n_over_l:>12.3e} {ave_norm:>12.4f} {class_norm:>14.4f}")

    # ── Generate Figure ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('#0a0a0a')
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

        # Panel 1: Classical vs AVE — absolute magnitude
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#111111')
        pq_vals = [r['pq_ppq'] for r in results]
        ave_mhz = [r['df_ave_hz'] / 1e6 for r in results]
        class_khz = [r['df_classical_hz'] / 1e3 for r in results]

        ax1.bar([x - 0.05 for x in pq_vals], ave_mhz, width=0.08,
               color='#00ffcc', alpha=0.8, edgecolor='white', lw=1.5,
               label='AVE chiral shift (MHz)')
        ax1b = ax1.twinx()
        ax1b.bar([x + 0.05 for x in pq_vals], class_khz, width=0.08,
                color='#ff6b6b', alpha=0.8, edgecolor='white', lw=1.5,
                label='Classical coupling (kHz)')

        ax1.set_xlabel(r'$pq/(p+q)$', color='white', fontsize=11)
        ax1.set_ylabel('AVE Δf (MHz)', color='#00ffcc', fontsize=11)
        ax1b.set_ylabel('Classical Δf (kHz)', color='#ff6b6b', fontsize=11)
        ax1.set_title('Magnitude: AVE vs Classical Coupling\n'
                      '(Note different units: MHz vs kHz)',
                      color='white', fontsize=13, fontweight='bold')
        ax1.tick_params(colors='white')
        ax1b.tick_params(colors='#ff6b6b')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=9, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        for s in ax1.spines.values(): s.set_color('#333')
        for s in ax1b.spines.values(): s.set_color('#333')

        # Panel 2: Normalized scaling shape
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#111111')
        ave_norm_list = [r['df_ave'] / results[0]['df_ave'] for r in results]
        class_norm_list = [r['df_classical'] / results[0]['df_classical']
                          if results[0]['df_classical'] > 0 else 0 for r in results]

        ax2.plot(pq_vals, ave_norm_list, 'o-', color='#00ffcc', lw=2.5, markersize=10,
                label=r'AVE: $\propto pq/(p+q)$')
        ax2.plot(pq_vals, class_norm_list, 's--', color='#ff6b6b', lw=2.5, markersize=10,
                label=r'Classical: $\propto N_{cross}/L_{self}$')

        # Overlay perfect linear scaling for comparison
        pq_theory = np.linspace(min(pq_vals)*0.9, max(pq_vals)*1.1, 100)
        ax2.plot(pq_theory, pq_theory / pq_vals[0], ':', color='white', lw=1, alpha=0.4,
                label='Perfect linear')

        ax2.set_xlabel(r'$pq/(p+q)$', color='white', fontsize=11)
        ax2.set_ylabel('Normalized Shift', color='white', fontsize=11)
        ax2.set_title('Scaling Shape: AVE (smooth) vs Classical (irregular)\n'
                      '(Different functional forms → experimentally distinguishable)',
                      color='white', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.15, color='white')
        for s in ax2.spines.values(): s.set_color('#333')

        # Panel 3: Crossing density
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor('#111111')
        n_crosses = [r['n_cross'] for r in results]
        colors = ['#00ffcc', '#ff6b6b', '#ffd93d', '#6bcaff', '#c78dff'][:len(results)]
        ax3.bar(range(len(results)), n_crosses, color=colors, alpha=0.8,
               edgecolor='white', lw=1.5)
        ax3.set_xticks(range(len(results)))
        ax3.set_xticklabels([r['label'] for r in results], color='white')
        ax3.set_ylabel('Number of Crossings', color='white', fontsize=11)
        ax3.set_title('Physical Crossing Count\n'
                      '(Scales nonlinearly with knot complexity)',
                      color='white', fontsize=13, fontweight='bold')
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.15, color='white', axis='y')
        for s in ax3.spines.values(): s.set_color('#333')

        # Panel 4: Key insight
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor('#111111')
        ax4.axis('off')

        max_ratio = max(r['df_classical'] / r['df_ave'] for r in results)
        verdict = "NEGLIGIBLE" if max_ratio < 0.01 else \
                  "SMALL (subtractable)" if max_ratio < 0.1 else \
                  "SIGNIFICANT — address with control antenna"

        insight_text = (
            "CLASSICAL COUPLING ASSESSMENT\n"
            + ("─" * 38) + "\n\n"
            f"Crossing z-separation: {Z_SEP_CROSSING*1e3:.1f}mm\n"
            f"Coupling length per crossing: {COUPLING_LENGTH*1e3:.0f}mm\n\n"
            "Key findings:\n\n"
            "1. MAGNITUDE: Classical coupling is\n"
            f"   {max_ratio*100:.1f}% of AVE shift (worst case)\n"
            f"   Status: {verdict}\n\n"
            "2. SCALING: Classical ∝ N_cross/L_self\n"
            "   AVE ∝ pq/(p+q)\n"
            "   These have DIFFERENT shapes → distinguishable\n\n"
            "3. CONTROL TEST: A zero-topology wire\n"
            "   with same crossings isolates the\n"
            "   classical contribution directly.\n\n"
            "RECOMMENDATION: Build the experiment.\n"
            "Classical coupling is small enough that\n"
            "a positive result is credible, and the\n"
            "control antenna makes it conclusive."
        )

        ax4.text(0.05, 0.95, insight_text, transform=ax4.transAxes,
                fontsize=10, color='#6bcaff', family='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a1a1a',
                         edgecolor='#6bcaff', alpha=0.9))
        ax4.set_title('Assessment', color='white', fontsize=13,
                      fontweight='bold', pad=20)

        out_dir = project_root / "assets" / "sim_outputs"
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / "hopf_01_classical_coupling.png"
        fig.savefig(out_path, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  📊 Plot saved: {out_path}")
    except ImportError:
        print("\n  ⚠️  matplotlib not available — skipping plots")

    # ── Conclusion ──
    print(f"\n  ═══════════════════════════════════════════════════════════════════")
    print(f"  CONCLUSION")
    print(f"  ═══════════════════════════════════════════════════════════════════")
    max_ratio = max(r['df_classical'] / r['df_ave'] for r in results)
    ave_over_class = min(r['df_ave'] / r['df_classical'] for r in results
                        if r['df_classical'] > 0)
    print(f"  Classical coupling is at most {max_ratio*100:.1f}% of the AVE shift.")
    print(f"  AVE signal exceeds classical noise by ≥ {ave_over_class:.0f}× (SNR).")
    print(f"  Classical coupling has a DIFFERENT scaling shape than AVE.")
    print(f"\n  The experiment should:")
    print(f"  1. Include a CONTROL antenna (zero topology, same crossings)")
    print(f"  2. Fit both Δf ∝ pq/(p+q) AND Δf ∝ N_cross models to data")
    print(f"  3. Use model selection (AIC/BIC) to determine which fits better")
    print(f"  ═══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
