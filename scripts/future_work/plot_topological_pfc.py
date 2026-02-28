#!/usr/bin/env python3
r"""
AVE: Topological Power Factor Correction (PFC)
==================================================
Generates 'topological_pfc.png' for future_work Ch. 2.

Top: Active temporal shaping — linear current ramp holding metric grip at 99% of V_sat.
Bottom: Spatial matching — Standard Toroid (k≈0.15) vs Hopf Coil (k≈0.95).
"""
import os, sys, pathlib
import numpy as np
import matplotlib.pyplot as plt

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

def generate():
    print("[*] Generating Topological PFC Figure...")
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.patch.set_facecolor('#0f0f12')

    # --- TOP PANEL: Temporal Shaping ---
    ax1.set_facecolor('#1a1a1f')
    t = np.linspace(0, 4, 1000)

    # Standard sine drive
    V_sine = np.sin(2 * np.pi * t)
    # Optimized linear ramp (flyback)
    V_ramp = np.zeros_like(t)
    period = 1.0
    for i, ti in enumerate(t):
        phase = ti % period
        if phase < period * 0.9:
            V_ramp[i] = 0.99 * (phase / (period * 0.9))
        else:
            frac = (phase - period * 0.9) / (period * 0.1)
            V_ramp[i] = 0.99 * (1.0 - frac)

    ax1.plot(t, V_sine, color='#888899', lw=2, alpha=0.6, label='Standard Sine Drive')
    ax1.plot(t, V_ramp, color='#33ffcc', lw=3, label='Optimized Linear Ramp (99% $V_{sat}$)')
    ax1.axhline(1.0, color='#ff3366', lw=2, linestyle='--', alpha=0.7, label='$V_{sat}$ (Saturation Limit)')
    ax1.fill_between(t, 0, V_ramp, alpha=0.08, color='#33ffcc')

    ax1.set_title("Temporal Optimization: Active Current Shaping\n(Hold Metric Grip at 99% of $V_{sat}$)",
                  color='white', fontsize=14, pad=15)
    ax1.set_xlabel("Time (Duty Cycles)", color='#cccccc', fontsize=12)
    ax1.set_ylabel("Applied Voltage / $V_{sat}$", color='#cccccc', fontsize=12)
    ax1.legend(frameon=False, fontsize=11, loc='upper right')
    ax1.grid(True, color='#333344', alpha=0.3)
    ax1.tick_params(colors='#888899')
    for s in ax1.spines.values(): s.set_color('#444455')

    # --- BOTTOM PANEL: Spatial Matching (Toroid vs Hopf Coil) ---
    ax2.set_facecolor('#1a1a1f')

    categories = ['Standard\nToroid', 'Split\nToroid', 'Bifilar\nCoil', 'HOPF\nTorus Knot']
    coupling_k = [0.15, 0.35, 0.55, 0.95]
    thrust_mult = [k ** 2 for k in coupling_k]  # Thrust scales as k^2

    colors_bar = ['#ff3366', '#ffcc00', '#3399ff', '#33ffcc']
    bars = ax2.bar(categories, coupling_k, color=colors_bar, edgecolor='white',
                   linewidth=1.5, alpha=0.85, width=0.6)

    # Add coupling coefficient labels
    for bar, k in zip(bars, coupling_k):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                 f'$k = {k:.2f}$', ha='center', va='bottom', color='white',
                 fontsize=13, fontweight='bold')

    # Add thrust multiplier labels
    for bar, tm in zip(bars, thrust_mult):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                 f'$F \\propto {tm:.2f}$', ha='center', va='center', color='black',
                 fontsize=11, fontweight='bold')

    ax2.set_title("Spatial Optimization: Chiral Impedance Matching\n($\\mathbf{A} \\parallel \\mathbf{B}$ Helicity Injection)",
                  color='white', fontsize=14, pad=15)
    ax2.set_ylabel("Coupling Coefficient $k$", color='#cccccc', fontsize=12)
    ax2.set_ylim([0, 1.15])
    ax2.axhline(1.0, color='white', lw=0.5, alpha=0.3, linestyle=':')
    ax2.grid(True, color='#333344', alpha=0.2, axis='y')
    ax2.tick_params(colors='#888899')
    for s in ax2.spines.values(): s.set_color('#444455')

    # Annotation arrow
    ax2.annotate('10× Thrust\nMultiplier', xy=(3, 0.95), xytext=(1.5, 0.85),
                 arrowprops=dict(arrowstyle='->', color='#33ffcc', lw=2),
                 color='#33ffcc', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=3.0)

    out_dir = project_root / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "topological_pfc.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[*] Topological PFC Figure Saved: {out_path}")

if __name__ == "__main__":
    generate()
