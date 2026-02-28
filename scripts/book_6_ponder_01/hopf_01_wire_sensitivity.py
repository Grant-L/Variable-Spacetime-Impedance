#!/usr/bin/env python3
r"""
HOPF-01: Wire-Stitched Form Factor — Sensitivity Analysis + BOM
=================================================================

Updated for the ACTUAL hardware: enameled magnet wire stitched through
PCB holes with LONGER trace lengths and MINERAL OIL immersion comparison.

Key update: wire lengths doubled to push resonances into
a practical measurement range, and mineral oil (ε_r ≈ 2.1) analysis
added as a substrate-independence cross-check on the SAME board.

Outputs:
  1. Sensitivity analysis for wire-in-air AND mineral oil
  2. Complete BOM with specific part numbers and costs
  3. 6-panel figure

Usage:
    PYTHONPATH=src python scripts/book_6_ponder_01/hopf_01_wire_sensitivity.py
"""

import sys
import os
import pathlib
import numpy as np

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

from ave.core.constants import C_0, ALPHA

# ══════════════════════════════════════════════════════════════
# Wire-in-Air Parameters
# ══════════════════════════════════════════════════════════════

WIRE_DIA = 0.5e-3          # m, 24 AWG enameled magnet wire
WIRE_HEIGHT = 1.6e-3       # m, PCB thickness (wire top to ground plane)

# Effective permittivity: air medium with enamel coating
EPS_EFF_AIR = 1.15
EPS_EFF_AIR_SIGMA = 0.08

# Mineral oil: ε_r ≈ 2.1 (transformer / USP grade)
# Wire submerged → field lines run through oil instead of air
EPS_R_OIL = 2.1
EPS_EFF_OIL = EPS_R_OIL * 0.85   # ~1.79 — partial coupling, ground plane beneath
EPS_EFF_OIL_SIGMA = 0.10         # Oil temperature coefficient + contamination

# Wire impedance (image charge model)
Z_WIRE_AIR = 60 / np.sqrt(EPS_EFF_AIR) * np.arccosh(2 * WIRE_HEIGHT / WIRE_DIA)
Z_WIRE_OIL = 60 / np.sqrt(EPS_EFF_OIL) * np.arccosh(2 * WIRE_HEIGHT / WIRE_DIA)

# Noise sources
WIRE_LENGTH_SIGMA = 0.5e-3
WIRE_HEIGHT_SIGMA = 0.3e-3
SMA_SIGMA_HZ = 200e3
N_MONTE_CARLO = 5000

# UPDATED trace lengths — ~2× longer to lower resonances
KNOTS = [
    (2, 3,  0.120, r'$(2,3)$ Trefoil'),       # was 60mm → 120mm
    (2, 5,  0.160, r'$(2,5)$ Cinquefoil'),     # was 90mm → 160mm
    (3, 7,  0.200, r'$(3,7)$'),                # was 120mm → 200mm
    (3, 11, 0.250, r'$(3,11)$'),               # was 150mm → 250mm
]

# ══════════════════════════════════════════════════════════════
# BOM
# ══════════════════════════════════════════════════════════════

BOM = [
    (1, 'PCB 120x120mm 2L FR-4 1.6mm ENIG (JLCPCB)',
        'Custom', 'JLCPCB', 5.00),
    (4, 'SMA Edge-Launch 50Ω (Amphenol 132255)',
        '132255', 'DigiKey', 3.50),
    (1, 'Enameled Wire 24AWG 5m spool',
        'MW-24', 'Remington', 6.00),
    (4, 'M3x6mm Pan Head Screw + Nut',
        'B07MFCNV3T', 'Amazon', 0.10),
    (4, 'M3 Nylon Standoff 10mm',
        'B07KWP5HMK', 'Amazon', 0.15),
    (1, 'SMA-SMA Cable 30cm RG316',
        '415-0029', 'DigiKey', 8.00),
    (1, 'Mineral Oil 500mL (transformer grade)',
        'B08V562HBX', 'Amazon', 12.00),
    (1, 'Glass dish 150mm (oil bath)',
        'B000FPKDUW', 'Amazon', 8.00),
]


def f_resonance(eps_eff, L_trace, chiral_factor=0.0):
    """Resonant frequency for wire-in-medium resonator."""
    n_eff = np.sqrt(eps_eff) * (1 + chiral_factor)
    return float(C_0) / (2 * L_trace * n_eff)


def chiral_factor(p, q):
    """AVE chiral coupling: α × pq/(p+q)."""
    return float(ALPHA) * p * q / (p + q)


def run_monte_carlo(eps_eff_nom, eps_eff_sigma, medium_name):
    """Run Monte Carlo sweep for a given medium."""
    rng = np.random.default_rng(42)
    results = {}

    for p, q, L_trace, label in KNOTS:
        chi = chiral_factor(p, q)
        pq_ppq = p * q / (p + q)

        f_std_nom = f_resonance(eps_eff_nom, L_trace, 0.0)
        f_ave_nom = f_resonance(eps_eff_nom, L_trace, chi)
        df_nom = f_std_nom - f_ave_nom

        height_samples = rng.normal(WIRE_HEIGHT, WIRE_HEIGHT_SIGMA, N_MONTE_CARLO)
        eps_samples = eps_eff_nom * (WIRE_HEIGHT / height_samples)**0.15
        L_samples = rng.normal(L_trace, WIRE_LENGTH_SIGMA, N_MONTE_CARLO)
        sma_noise = rng.normal(0, SMA_SIGMA_HZ, N_MONTE_CARLO)

        f_std_mc = np.array([f_resonance(e, L, 0.0) for e, L in
                            zip(eps_samples, L_samples)]) + sma_noise
        f_ave_mc = np.array([f_resonance(e, L, chi) for e, L in
                            zip(eps_samples, L_samples)]) + sma_noise
        df_mc = f_std_mc - f_ave_mc

        results[label] = {
            'p': p, 'q': q, 'pq_ppq': pq_ppq,
            'f_std_nom': f_std_nom, 'f_ave_nom': f_ave_nom,
            'df_nom': df_nom, 'df_mc': df_mc,
            'df_frac_nom': df_nom / f_std_nom,
            'chi': chi, 'L_trace': L_trace,
        }

    return results


def print_results(results, medium, eps_eff, z_wire):
    """Print formatted results table."""
    print(f"\n  ── {medium} (ε_eff = {eps_eff:.2f}, Z = {z_wire:.0f}Ω) ──")
    print(f"  {'Knot':<20} {'pq/(p+q)':>8} {'f_std':>10} {'Δf':>10} {'σ_Δf':>8} {'SNR':>6} {'Δf/f':>10}")
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*6} {'─'*10}")
    for label, r in results.items():
        sigma = np.std(r['df_mc'])
        snr = abs(r['df_nom']) / sigma
        print(f"  {label:<20} {r['pq_ppq']:>8.4f} "
              f"{r['f_std_nom']/1e9:>8.3f}GHz "
              f"{r['df_nom']/1e6:>8.2f}MHz "
              f"{sigma/1e3:>6.0f}kHz "
              f"{snr:>5.0f}σ "
              f"{r['df_frac_nom']:>10.6f}")


def print_bom():
    """Print formatted BOM."""
    print("\n  ╔════════════════════════════════════════════════════════════════╗")
    print("  ║  HOPF-01 WIRE-STITCHED — BOM (Rev 3.0, incl. mineral oil)   ║")
    print("  ╠════════════════════════════════════════════════════════════════╣")
    total = 0
    for qty, desc, pn, supplier, cost in BOM:
        lc = qty * cost
        total += lc
        print(f"  ║  {qty}x {desc:<50}  ${lc:>5.2f} ║")
    print("  ╠════════════════════════════════════════════════════════════════╣")
    print(f"  ║  TOTAL{' '*50}  ${total:>5.2f} ║")
    print("  ╚════════════════════════════════════════════════════════════════╝")
    print(f"  Equipment: LiteVNA (~$100) or NanoVNA-H4 ($60, limited to 1.5 GHz)")
    return total


def main():
    print("=" * 80)
    print("  HOPF-01: Wire-Stitched — Air vs Mineral Oil Comparison")
    print("=" * 80)

    print(f"\n  Wire: 24 AWG enameled, {WIRE_DIA*1e3:.2f}mm dia, {WIRE_HEIGHT*1e3:.1f}mm above ground")
    print(f"  UPDATED trace lengths (2× longer for lower resonances):")
    for p, q, L, label in KNOTS:
        print(f"    {label}: {L*1000:.0f}mm")

    # Run both media
    air = run_monte_carlo(EPS_EFF_AIR, EPS_EFF_AIR_SIGMA, "Air")
    oil = run_monte_carlo(EPS_EFF_OIL, EPS_EFF_OIL_SIGMA, "Mineral Oil")

    print_results(air, "AIR", EPS_EFF_AIR, Z_WIRE_AIR)
    print_results(oil, "MINERAL OIL", EPS_EFF_OIL, Z_WIRE_OIL)

    # Key comparison: Δf/f should be IDENTICAL in both media
    print(f"\n  ── SUBSTRATE INDEPENDENCE CHECK ──")
    print(f"  If AVE is correct: Δf/f = α × pq/(p+q) regardless of medium")
    print(f"  {'Knot':<20} {'Δf/f (air)':>12} {'Δf/f (oil)':>12} {'Ratio':>8} {'Match?':>8}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*8} {'─'*8}")
    for label in air:
        r_a = air[label]['df_frac_nom']
        r_o = oil[label]['df_frac_nom']
        ratio = r_a / r_o if r_o != 0 else float('inf')
        match = "✅" if abs(ratio - 1.0) < 0.01 else "❌"
        print(f"  {label:<20} {r_a:>12.8f} {r_o:>12.8f} {ratio:>8.5f} {match:>6}")

    total = print_bom()

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(22, 18))
        fig.patch.set_facecolor('#0a0a0a')
        gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.28)
        colors = ['#00ffcc', '#ff6b6b', '#ffd93d', '#6bcaff']

        # ── Panel 1: Air — Δf distribution ──
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#111111')
        for i, (label, r) in enumerate(air.items()):
            ax1.boxplot([r['df_mc'] / 1e6], positions=[r['pq_ppq']],
                       widths=0.08, patch_artist=True,
                       boxprops=dict(facecolor=colors[i], alpha=0.3, edgecolor=colors[i]),
                       medianprops=dict(color='white', lw=2),
                       whiskerprops=dict(color=colors[i]),
                       capprops=dict(color=colors[i]),
                       flierprops=dict(markeredgecolor=colors[i], markersize=2))
        pq_x = np.linspace(0, 3, 100)
        avg_f = np.mean([r['f_std_nom'] for r in air.values()])
        ax1.plot(pq_x, avg_f * float(ALPHA) * pq_x / 1e6,
                 'w--', lw=2, alpha=0.5, label=r'$\Delta f \propto \alpha \cdot pq/(p+q)$')
        ax1.set_xlabel(r'$pq/(p+q)$', color='white', fontsize=11)
        ax1.set_ylabel(r'$\Delta f$ (MHz)', color='white', fontsize=11)
        ax1.set_title(f'AIR ($\\varepsilon_{{eff}}$ = {EPS_EFF_AIR})',
                      color='#00ffcc', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.15, color='white')
        for s in ax1.spines.values(): s.set_color('#333')

        # ── Panel 2: Oil — Δf distribution ──
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#111111')
        for i, (label, r) in enumerate(oil.items()):
            ax2.boxplot([r['df_mc'] / 1e6], positions=[r['pq_ppq']],
                       widths=0.08, patch_artist=True,
                       boxprops=dict(facecolor=colors[i], alpha=0.3, edgecolor=colors[i]),
                       medianprops=dict(color='white', lw=2),
                       whiskerprops=dict(color=colors[i]),
                       capprops=dict(color=colors[i]),
                       flierprops=dict(markeredgecolor=colors[i], markersize=2))
        avg_f_oil = np.mean([r['f_std_nom'] for r in oil.values()])
        ax2.plot(pq_x, avg_f_oil * float(ALPHA) * pq_x / 1e6,
                 'w--', lw=2, alpha=0.5, label=r'$\Delta f \propto \alpha \cdot pq/(p+q)$')
        ax2.set_xlabel(r'$pq/(p+q)$', color='white', fontsize=11)
        ax2.set_ylabel(r'$\Delta f$ (MHz)', color='white', fontsize=11)
        ax2.set_title(f'MINERAL OIL ($\\varepsilon_{{eff}}$ = {EPS_EFF_OIL:.2f})',
                      color='#ffd93d', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.15, color='white')
        for s in ax2.spines.values(): s.set_color('#333')

        # ── Panel 3: Δf/f comparison (the kill-shot) ──
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor('#111111')
        pq_vals = [r['pq_ppq'] for r in air.values()]
        dff_air = [r['df_frac_nom'] * 1e6 for r in air.values()]
        dff_oil = [r['df_frac_nom'] * 1e6 for r in oil.values()]
        ax3.scatter(pq_vals, dff_air, s=150, c='#00ffcc', marker='o', zorder=5,
                   edgecolors='white', lw=1.5, label='Air')
        ax3.scatter(pq_vals, dff_oil, s=150, c='#ffd93d', marker='s', zorder=5,
                   edgecolors='white', lw=1.5, label='Mineral Oil')
        # Theory line
        pq_theory = np.linspace(0, 3, 100)
        dff_theory = float(ALPHA) * pq_theory * 1e6
        ax3.plot(pq_theory, dff_theory, 'w--', lw=2, alpha=0.6,
                 label=r'AVE: $\Delta f/f = \alpha \cdot pq/(p+q)$')
        ax3.set_xlabel(r'$pq/(p+q)$', color='white', fontsize=12)
        ax3.set_ylabel(r'$\Delta f / f$ (ppm)', color='white', fontsize=12)
        ax3.set_title('SUBSTRATE INDEPENDENCE: Air vs Oil\n'
                      r'($\Delta f/f$ must overlap if topology is real)',
                      color='white', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=11, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.15, color='white')
        for s in ax3.spines.values(): s.set_color('#333')

        # ── Panel 4: Frequency map (both media, with VNA range) ──
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor('#111111')
        knot_labels = [f"({p},{q})" for p, q, _, _ in KNOTS]
        x_pos = np.arange(len(KNOTS))
        f_air = [r['f_std_nom'] / 1e9 for r in air.values()]
        f_oil = [r['f_std_nom'] / 1e9 for r in oil.values()]
        ax4.bar(x_pos - 0.15, f_air, 0.3, color='#00ffcc', alpha=0.7,
                label='Air', edgecolor='white')
        ax4.bar(x_pos + 0.15, f_oil, 0.3, color='#ffd93d', alpha=0.7,
                label='Oil', edgecolor='white')
        ax4.axhline(1.5, color='#ff3366', lw=1.5, ls=':', label='NanoVNA-H4 max')
        ax4.axhline(6.3, color='#6bcaff', lw=1.5, ls=':', label='LiteVNA max')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(knot_labels, color='white')
        ax4.set_ylabel('f_res (GHz)', color='white', fontsize=11)
        ax4.set_title('Resonant Frequencies (Both Media)\n'
                      'vs VNA Instrument Range',
                      color='white', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax4.tick_params(colors='white')
        ax4.grid(True, alpha=0.15, color='white', axis='y')
        for s in ax4.spines.values(): s.set_color('#333')

        # ── Panel 5: SNR comparison ──
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.set_facecolor('#111111')
        snr_air = [abs(r['df_nom']) / np.std(r['df_mc']) for r in air.values()]
        snr_oil = [abs(r['df_nom']) / np.std(r['df_mc']) for r in oil.values()]
        ax5.bar(x_pos - 0.15, snr_air, 0.3, color='#00ffcc', alpha=0.7,
                label='Air', edgecolor='white')
        ax5.bar(x_pos + 0.15, snr_oil, 0.3, color='#ffd93d', alpha=0.7,
                label='Oil', edgecolor='white')
        ax5.axhline(5, color='#ff3366', lw=2, ls='--', label=r'5$\sigma$ threshold')
        for i in range(len(KNOTS)):
            ax5.text(x_pos[i] - 0.15, snr_air[i] + 2, f'{snr_air[i]:.0f}',
                    ha='center', color='#00ffcc', fontsize=10, fontweight='bold')
            ax5.text(x_pos[i] + 0.15, snr_oil[i] + 2, f'{snr_oil[i]:.0f}',
                    ha='center', color='#ffd93d', fontsize=10, fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(knot_labels, color='white')
        ax5.set_ylabel(r'SNR ($\sigma$)', color='white', fontsize=11)
        ax5.set_title('Detection Confidence (Both Media)',
                      color='white', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax5.tick_params(colors='white')
        ax5.grid(True, alpha=0.15, color='white', axis='y')
        for s in ax5.spines.values(): s.set_color('#333')

        # ── Panel 6: BOM ──
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.set_facecolor('#111111')
        ax6.axis('off')
        bom_text = "BILL OF MATERIALS (Rev 3.0)\n"
        bom_text += "─" * 38 + "\n\n"
        for qty, desc, pn, supplier, cost in BOM:
            short = desc[:36] + '..' if len(desc) > 38 else desc
            bom_text += f" {qty}x {short}\n"
            bom_text += f"    ${qty*cost:.2f} ({supplier})\n"
        bom_text += "\n" + "─" * 38 + "\n"
        bom_text += f"TOTAL: ${total:.2f}\n\n"
        bom_text += "Test Protocol:\n"
        bom_text += " 1. Wire all 4 knots, measure in AIR\n"
        bom_text += " 2. Submerge in mineral oil bath\n"
        bom_text += " 3. Re-measure ALL 4 knots in OIL\n"
        bom_text += " 4. Plot Δf/f(air) vs Δf/f(oil)\n"
        bom_text += " 5. Must overlap → topology real\n"

        ax6.text(0.05, 0.95, bom_text, transform=ax6.transAxes,
                fontsize=10, color='#ffd93d', family='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a1a1a',
                         edgecolor='#ffd93d', alpha=0.9))
        ax6.set_title('BOM + Test Protocol',
                      color='white', fontsize=13, fontweight='bold', pad=20)

        out_dir = project_root / "assets" / "sim_outputs"
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / "hopf_01_wire_sensitivity.png"
        fig.savefig(out_path, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  📊 Plot saved: {out_path}")
    except ImportError:
        print("\n  ⚠️  matplotlib not available — skipping plots")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
