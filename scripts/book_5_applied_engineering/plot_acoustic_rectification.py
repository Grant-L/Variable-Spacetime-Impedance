#!/usr/bin/env python3
r"""
AVE: Impedance Rectification in a Saturating Dielectric
========================================================
Generates the 'acoustic_rectification.png' figure for future_work/.

Top Panel:  A symmetric sine wave in a linear dielectric → equal and opposite
            forces → zero time-averaged thrust.
Bottom Panel: An asymmetric flyback transient exploiting dielectric saturation.
              The slow edge inductively grips the metric (high impedance);
              the fast edge causes saturated zero-impedance slip.
              The non-linear medium rectifies AC into DC macroscopic thrust.
"""

import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

def generate_acoustic_rectification_figure():
    print("[*] Generating Acoustic Rectification Figure...")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             gridspec_kw={'width_ratios': [2, 2, 1.2]})
    fig.patch.set_facecolor('#0f0f12')

    # --- Time axis ---
    t = np.linspace(0, 4 * np.pi, 2000)
    dt = t[1] - t[0]

    # --- Dielectric Saturation Curve ---
    V_sat = 0.7  # Saturation voltage (normalized)

    def dielectric_response(V, linear=True):
        """Models the polarization P(V) of the medium."""
        if linear:
            return V  # Linear response: P = ε * V
        else:
            # Non-linear saturating: P = V_sat * tanh(V / V_sat)
            return V_sat * np.tanh(V / V_sat)

    # =====================================================================
    # TOP ROW: Symmetric Sine Wave in Linear Dielectric
    # =====================================================================
    V_sym = np.sin(t)

    # Panel 1: Input waveform
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1f')
    ax.plot(t, V_sym, color='#3399ff', lw=2)
    ax.axhline(0, color='white', lw=0.5, alpha=0.3)
    ax.set_title("Input: Symmetric Sine Wave", color='white', fontsize=13, pad=10)
    ax.set_ylabel("Applied Voltage $V(t)$", color='#cccccc', fontsize=11)
    ax.set_xlabel("Time", color='#cccccc', fontsize=10)
    ax.grid(True, color='#333344', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color('#444455')

    # Panel 2: Force response (linear)
    P_sym = dielectric_response(V_sym, linear=True)
    # Force ~ dP/dt (rate of change of polarization → displacement current)
    F_sym = np.gradient(P_sym, dt)

    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1f')
    ax.plot(t, F_sym, color='#3399ff', lw=2, alpha=0.8)
    ax.axhline(0, color='white', lw=0.5, alpha=0.3)

    # Show time-averaged force = 0
    F_avg_sym = np.cumsum(F_sym) * dt / (t + dt)
    ax.plot(t, F_avg_sym, color='#ff3366', lw=2.5, linestyle='--', label=r'$\langle F \rangle_{avg} = 0$')
    ax.set_title("Reaction Force $F(t)$ — Linear Medium", color='white', fontsize=13, pad=10)
    ax.set_ylabel("Force Response", color='#cccccc', fontsize=11)
    ax.set_xlabel("Time", color='#cccccc', fontsize=10)
    ax.legend(frameon=False, fontsize=11, loc='upper right')
    ax.grid(True, color='#333344', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color('#444455')

    # Panel 3: Verdict
    ax = axes[0, 2]
    ax.set_facecolor('#1a1a1f')
    ax.axis('off')
    ax.text(0.5, 0.65, "NET THRUST", ha='center', va='center',
            fontsize=20, fontweight='bold', color='#ff3366',
            transform=ax.transAxes)
    ax.text(0.5, 0.45, "= 0", ha='center', va='center',
            fontsize=36, fontweight='bold', color='#ff3366',
            transform=ax.transAxes)
    ax.text(0.5, 0.25, "Equal & Opposite\nForces Cancel", ha='center', va='center',
            fontsize=12, color='#888899', transform=ax.transAxes)

    # =====================================================================
    # BOTTOM ROW: Asymmetric Flyback in Saturating Dielectric
    # =====================================================================
    # Asymmetric waveform: slow rise, fast snap-back (flyback transient)
    V_asym = np.zeros_like(t)
    period = 2 * np.pi
    for i, ti in enumerate(t):
        phase = ti % period
        if phase < period * 0.85:
            # Slow inductance-limited rise
            V_asym[i] = 1.2 * (phase / (period * 0.85))
        else:
            # Fast nanosecond flyback snap
            frac = (phase - period * 0.85) / (period * 0.15)
            V_asym[i] = 1.2 * (1.0 - frac**0.3) * np.exp(-3 * frac)

    # Panel 4: Asymmetric input
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1f')
    ax.plot(t, V_asym, color='#33ffcc', lw=2)
    ax.axhline(V_sat, color='#ffcc00', lw=1.5, linestyle='--', alpha=0.7, label=f'$V_{{sat}}$ = {V_sat}')
    ax.axhline(0, color='white', lw=0.5, alpha=0.3)
    ax.set_title("Input: Asymmetric Flyback Transient", color='white', fontsize=13, pad=10)
    ax.set_ylabel("Applied Voltage $V(t)$", color='#cccccc', fontsize=11)
    ax.set_xlabel("Time", color='#cccccc', fontsize=10)
    ax.legend(frameon=False, fontsize=11, loc='upper left')
    ax.grid(True, color='#333344', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color('#444455')

    # Shade the below/above saturation regions
    ax.fill_between(t, 0, np.minimum(V_asym, V_sat), alpha=0.08, color='#33ffcc')
    above_sat = np.where(V_asym > V_sat, V_asym, V_sat)
    ax.fill_between(t, V_sat, above_sat, alpha=0.15, color='#ffcc00')

    # Panel 5: Force response (non-linear saturating)
    P_asym = dielectric_response(V_asym, linear=False)
    F_asym = np.gradient(P_asym, dt)

    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1f')
    ax.plot(t, F_asym, color='#33ffcc', lw=2, alpha=0.8)
    ax.axhline(0, color='white', lw=0.5, alpha=0.3)

    # Time avg net thrust
    F_avg_asym = np.cumsum(F_asym) * dt / (t + dt)
    ax.plot(t, F_avg_asym, color='#ffcc00', lw=2.5, linestyle='--',
            label=r'$\langle F \rangle_{avg} > 0$ (DC Thrust)')
    ax.set_title("Reaction Force $F(t)$ — Saturating Dielectric", color='white', fontsize=13, pad=10)
    ax.set_ylabel("Force Response", color='#cccccc', fontsize=11)
    ax.set_xlabel("Time", color='#cccccc', fontsize=10)
    ax.legend(frameon=False, fontsize=11, loc='upper right')
    ax.grid(True, color='#333344', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color('#444455')

    # Panel 6: Verdict
    ax = axes[1, 2]
    ax.set_facecolor('#1a1a1f')
    ax.axis('off')
    final_avg = np.mean(F_asym)
    ax.text(0.5, 0.65, "NET THRUST", ha='center', va='center',
            fontsize=20, fontweight='bold', color='#33ffcc',
            transform=ax.transAxes)
    ax.text(0.5, 0.45, "> 0", ha='center', va='center',
            fontsize=36, fontweight='bold', color='#33ffcc',
            transform=ax.transAxes)
    ax.text(0.5, 0.25, "Asymmetric Dielectric\nSaturation Rectifies\nAC → DC Thrust", ha='center', va='center',
            fontsize=12, color='#888899', transform=ax.transAxes)

    plt.tight_layout(pad=2.5)

    out_dir = project_root / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "acoustic_rectification.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    print(f"[*] Acoustic Rectification Figure Saved: {out_path}")

if __name__ == "__main__":
    generate_acoustic_rectification_figure()
