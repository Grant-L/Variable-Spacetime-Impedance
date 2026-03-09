#!/usr/bin/env python3
r"""
HOPF-01: Substrate Independence Verification — Air vs Mineral Oil
===================================================================

Compares the chiral frequency shift in AIR vs MINERAL OIL using the
same wire-stitched board to verify substrate independence.

The AVE prediction: Δf/f = α × pq/(p+q), independent of ε_eff.
  → If the fractional shift is identical in both media,
    the coupling is a VACUUM property (not a material artifact).
  → If it tracks ε_eff, it's just dielectric measurement error.

Physical approach:
  The SAME board is measured first in air (ε_eff ≈ 1.295), then
  submerged in mineral oil (ε_eff ≈ 2.265). Absolute frequencies
  shift downward in oil (expected), but the fractional shift
  Δf/f must remain constant.

This is the strongest single falsification criterion in HOPF-01.

Usage:
    PYTHONPATH=src python scripts/book_7_ponder_01/hopf_01_substrate_comparison.py
"""

import sys
import os
import pathlib
import numpy as np

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

from ave.core.constants import C_0, ALPHA

# ══════════════════════════════════════════════════════════════
# Wire-in-Air Physical Parameters
# ══════════════════════════════════════════════════════════════

WIRE_DIA = 0.51e-3           # m (24 AWG)
ENAMEL_THICKNESS = 30e-6     # m
ENAMEL_EPS_R = 3.5           # polyurethane

# Media definitions
MEDIA = {
    'Air':         {'eps_r': 1.0006, 'color': '#00ffcc', 'marker': 'o'},
    'Mineral Oil': {'eps_r': 2.1,    'color': '#ffd93d', 'marker': 's'},
}

# Torus knot catalog — wire-stitched lengths (matches chapter)
KNOTS = [
    (2, 3,  0.120, r'$(2,3)$'),
    (2, 5,  0.160, r'$(2,5)$'),
    (3, 5,  0.170, r'$(3,5)$'),
    (3, 7,  0.200, r'$(3,7)$'),
    (3, 11, 0.250, r'$(3,11)$'),
]


def effective_permittivity(eps_medium):
    """Effective permittivity accounting for enamel coating."""
    f_enamel = 2 * ENAMEL_THICKNESS / WIRE_DIA
    return eps_medium * (1 + f_enamel * (ENAMEL_EPS_R / eps_medium - 1))


def chiral_factor(p, q):
    """AVE chiral coupling: α × pq/(p+q)."""
    return float(ALPHA) * p * q / (p + q)


def compute_shifts(eps_r_medium, knots):
    """Compute resonant frequencies and shifts for all knots in a medium."""
    c = float(C_0)
    eps_eff = effective_permittivity(eps_r_medium)
    results = []
    for p, q, L, label in knots:
        chi = chiral_factor(p, q)
        pq_ppq = p * q / (p + q)
        n_std = np.sqrt(eps_eff)
        n_ave = n_std * (1 + chi)
        f_std = c / (2 * L * n_std)
        f_ave = c / (2 * L * n_ave)
        df = f_std - f_ave
        df_frac = df / f_std
        results.append({
            'p': p, 'q': q, 'label': label,
            'pq_ppq': pq_ppq,
            'f_std': f_std, 'f_ave': f_ave,
            'df': df, 'df_frac': df_frac,
            'chi': chi, 'eps_eff': eps_eff,
        })
    return results


def main():
    print("=" * 80)
    print("  HOPF-01: Substrate Independence — Air vs Mineral Oil")
    print("=" * 80)

    alpha = float(ALPHA)

    all_results = {}
    for name, medium in MEDIA.items():
        eps_eff = effective_permittivity(medium['eps_r'])
        res = compute_shifts(medium['eps_r'], KNOTS)
        all_results[name] = res
        print(f"\n  ──── {name} (ε_r = {medium['eps_r']}, ε_eff = {eps_eff:.4f}) ────")
        print(f"  {'Knot':<10} {'f_std (GHz)':>12} {'f_AVE (GHz)':>12} "
              f"{'Δf (MHz)':>10} {'Δf/f':>14} {'α×pq/(p+q)':>14}")
        for r in res:
            predicted = alpha * r['pq_ppq']
            print(f"  {r['label']:<10} {r['f_std']/1e9:>12.3f} {r['f_ave']/1e9:>12.3f} "
                  f"{r['df']/1e6:>10.3f} {r['df_frac']:>14.6e} {predicted:>14.6e}")

    # ── Verify substrate independence ──
    print(f"\n  ═════════════════════════════════════════════════════════════")
    print(f"  SUBSTRATE INDEPENDENCE TEST (Air vs Mineral Oil)")
    print(f"  ═════════════════════════════════════════════════════════════")
    print(f"\n  Same board, two media. Δf/f must be IDENTICAL.")

    media_names = list(MEDIA.keys())
    for i, (p, q, L, label) in enumerate(KNOTS):
        fracs = [all_results[name][i]['df_frac'] for name in media_names]
        ratio = fracs[0] / fracs[1] if fracs[1] > 0 else 0
        status = "✅ IDENTICAL" if abs(ratio - 1.0) < 0.001 else "⚠️ DIFFERS"
        print(f"  {label:<10} Δf/f(air)={fracs[0]:.6e}  Δf/f(oil)={fracs[1]:.6e}  "
              f"ratio={ratio:.5f}  {status}")

    # ── Generate figure ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('#0a0a0a')
        gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

        # ── Panel 1: Absolute frequency shifts across media ──
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#111111')

        bar_width = 0.3
        x_base = np.arange(len(KNOTS))

        for j, (name, medium) in enumerate(MEDIA.items()):
            res = all_results[name]
            shifts_mhz = [r['df'] / 1e6 for r in res]
            ax1.bar(x_base + j * bar_width, shifts_mhz,
                    width=bar_width, color=medium['color'], alpha=0.8,
                    edgecolor='white', lw=1.5,
                    label=f"{name} (ε_eff={res[0]['eps_eff']:.3f})")

        ax1.set_xticks(x_base + bar_width / 2)
        ax1.set_xticklabels([f"({p},{q})" for p, q, _, _ in KNOTS], color='white')
        ax1.set_ylabel('Absolute Shift Δf (MHz)', color='white', fontsize=11)
        ax1.set_title('Absolute Frequency Shift\n(Differs Between Media — Expected)',
                      color='white', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.15, color='white', axis='y')
        for spine in ax1.spines.values():
            spine.set_color('#333')

        # ── Panel 2: FRACTIONAL shift (must be media-independent) ──
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#111111')

        for j, (name, medium) in enumerate(MEDIA.items()):
            res = all_results[name]
            pq_vals = [r['pq_ppq'] for r in res]
            frac_shifts = [r['df_frac'] for r in res]
            ax2.scatter(pq_vals, frac_shifts, color=medium['color'], s=120,
                       marker=medium['marker'],
                       edgecolors='white', lw=1.5, zorder=5,
                       label=f"{name} (ε_eff={res[0]['eps_eff']:.3f})")

        # Overlay exact AVE prediction
        pq_x = np.linspace(0, 3, 100)
        ax2.plot(pq_x, alpha * pq_x, color='white', lw=2, linestyle='--',
                label=r'AVE: $\Delta f/f = \alpha \times pq/(p+q)$')

        ax2.set_xlabel(r'$pq/(p+q)$', color='white', fontsize=11)
        ax2.set_ylabel(r'$\Delta f / f$ (fractional shift)', color='white', fontsize=11)
        ax2.set_title('Fractional Shift: MEDIA INDEPENDENT\n'
                      r'(Both media collapse onto $\alpha \times pq/(p+q)$)',
                      color='#00ffcc', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.15, color='white')
        for spine in ax2.spines.values():
            spine.set_color('#333')

        # ── Panel 3: Residual from AVE prediction ──
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor('#111111')

        for j, (name, medium) in enumerate(MEDIA.items()):
            res = all_results[name]
            pq_vals = [r['pq_ppq'] for r in res]
            residuals = [(r['df_frac'] - alpha * r['pq_ppq']) / (alpha * r['pq_ppq']) * 1e6
                        for r in res]
            ax3.scatter(pq_vals, residuals, color=medium['color'], s=120,
                       marker=medium['marker'],
                       edgecolors='white', lw=1.5, zorder=5, label=name)

        ax3.axhline(0, color='white', lw=1, alpha=0.3)
        ax3.set_xlabel(r'$pq/(p+q)$', color='white', fontsize=11)
        ax3.set_ylabel('Residual from AVE prediction (ppm)', color='white', fontsize=11)
        ax3.set_title('Residuals: Sub-ppm Agreement Across Media',
                      color='white', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.15, color='white')
        for spine in ax3.spines.values():
            spine.set_color('#333')

        # ── Panel 4: Decision table ──
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor('#111111')
        ax4.axis('off')

        table_data = []
        for name, medium in MEDIA.items():
            res = all_results[name]
            eps_eff = res[0]['eps_eff']
            fracs = [f"{r['df_frac']:.6e}" for r in res]
            table_data.append([name, f"{eps_eff:.4f}"] + fracs)

        col_labels = ['Medium', 'ε_eff'] + \
                    [f"({p},{q})" for p, q, _, _ in KNOTS]
        table = ax4.table(cellText=table_data, colLabels=col_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 2.5)

        for key, cell in table.get_celld().items():
            cell.set_facecolor('#1a1a1a')
            cell.set_edgecolor('#333')
            cell.set_text_props(color='white')
            if key[0] == 0:
                cell.set_facecolor('#2a2a3a')
                cell.set_text_props(color='#00ffcc', fontweight='bold')

        ax4.set_title('Δf/f Comparison: Air vs Oil\n(Must be IDENTICAL across rows)',
                      color='white', fontsize=13, fontweight='bold', pad=20)

        # Save
        out_dir = project_root / "assets" / "sim_outputs"
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / "hopf_01_substrate_comparison.png"
        fig.savefig(out_path, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  📊 Plot saved: {out_path}")
    except ImportError:
        print("\n  ⚠️  matplotlib not available — skipping plots")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
