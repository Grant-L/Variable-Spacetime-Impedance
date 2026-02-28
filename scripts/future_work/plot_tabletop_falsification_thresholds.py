#!/usr/bin/env python3
r"""
AVE: Tabletop Falsification Thresholds
========================================
Generates 'tabletop_falsification_thresholds.png' for future_work Ch. 4.

Left: Scalar strain (RVR) — modulation depth ~10^-26, physically impossible Q.
Right: Mutual inductance (Sagnac-RLVE) — 0.38 m/s drift, 2.07 radian shift.
"""
import os, sys, pathlib
import numpy as np
import matplotlib.pyplot as plt

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

from ave.core.constants import C_0

def generate():
    print("[*] Generating Tabletop Falsification Thresholds Figure...")
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#0f0f12')

    # --- LEFT: Scalar Strain (RVR) — Fails ---
    ax1.set_facecolor('#1a1a1f')

    Q_factors = np.logspace(0, 30, 500)
    modulation_depth = 1e-26  # Scalar gravity modulation

    # Signal = modulation_depth * Q
    signal = modulation_depth * Q_factors
    # Noise floor (thermal Johnson-Nyquist)
    noise = 1e-12 * np.ones_like(Q_factors)

    ax1.loglog(Q_factors, signal, color='#ff3366', lw=3, label='Signal ($\\sim 10^{-26} \\cdot Q$)')
    ax1.loglog(Q_factors, noise, 'w--', lw=2, label='Noise Floor ($\\sim 10^{-12}$)')

    # Shade impossibility zone
    ax1.fill_between(Q_factors, signal, noise, where=(signal < noise), alpha=0.1, color='#ff3366')

    # Q required to detect
    Q_min = noise[0] / modulation_depth
    ax1.axvline(Q_min, color='#ffcc00', lw=2, linestyle=':', label=f'Required Q $\\approx 10^{{{int(np.log10(Q_min))}}}$')

    ax1.set_title("❌ Scalar Metric Strain (RVR)\n(Physically Impossible)", color='#ff3366', fontsize=14, pad=15)
    ax1.set_xlabel("Resonator Q-Factor", color='#cccccc', fontsize=12)
    ax1.set_ylabel("Signal Amplitude (a.u.)", color='#cccccc', fontsize=12)
    ax1.legend(frameon=False, fontsize=10, loc='lower right')
    ax1.grid(True, color='#333344', alpha=0.3, which='both')
    ax1.tick_params(colors='#888899')
    for s in ax1.spines.values(): s.set_color('#444455')

    ax1.text(1e5, 1e-8, "UNDETECTABLE\n($G/c^2$ suppression)",
             color='#ff3366', fontsize=13, fontweight='bold', ha='center')

    # --- RIGHT: Mutual Inductance (Sagnac-RLVE) — Succeeds ---
    ax2.set_facecolor('#1a1a1f')

    L_fiber = np.linspace(10, 500, 500)  # meters of fiber
    v_network = 0.38  # m/s (tungsten at 10k RPM)
    lam = 1550e-9  # 1550nm telecom laser
    c = C_0

    phase_shift = 4 * np.pi * L_fiber * v_network / (lam * c)

    ax2.plot(L_fiber, phase_shift, color='#33ffcc', lw=3, label='$\\Delta\\phi$ (Tungsten, 10k RPM)')
    ax2.axhline(np.pi, color='#ffcc00', lw=2, linestyle='--', label='$\\pi$ (Trivially Detectable)')
    ax2.axhline(0.01, color='#ff3366', lw=1.5, linestyle=':', alpha=0.5, label='$0.01$ rad (Photodetector Limit)')

    # Mark the 200m design point
    design_phi = 4 * np.pi * 200 * v_network / (lam * c)
    ax2.plot(200, design_phi, 'o', color='#ffcc00', markersize=12, zorder=10)
    ax2.annotate(f'200m: {design_phi:.1f} rad', xy=(200, design_phi), xytext=(300, design_phi - 0.3),
                 arrowprops=dict(arrowstyle='->', color='#ffcc00', lw=2),
                 color='#ffcc00', fontsize=12, fontweight='bold')

    # Shade easy detection zone
    ax2.fill_between(L_fiber, 0.01, phase_shift, where=(phase_shift > 0.01), alpha=0.08, color='#33ffcc')

    ax2.set_title("✅ Mutual Inductance (Sagnac-RLVE)\n(Massive, Trivially Detectable)", color='#33ffcc', fontsize=14, pad=15)
    ax2.set_xlabel("Fiber Optic Length (m)", color='#cccccc', fontsize=12)
    ax2.set_ylabel("Phase Shift $\\Delta\\phi$ (Radians)", color='#cccccc', fontsize=12)
    ax2.legend(frameon=False, fontsize=10, loc='upper left')
    ax2.grid(True, color='#333344', alpha=0.3)
    ax2.tick_params(colors='#888899')
    for s in ax2.spines.values(): s.set_color('#444455')

    plt.tight_layout(pad=2.5)
    out_dir = project_root / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "tabletop_falsification_thresholds.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[*] Tabletop Falsification Thresholds Saved: {out_path}")

if __name__ == "__main__":
    generate()
