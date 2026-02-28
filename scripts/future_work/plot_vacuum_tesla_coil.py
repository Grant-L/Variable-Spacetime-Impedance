#!/usr/bin/env python3
r"""
AVE: Autoresonant Dielectric Rupture (Vacuum Tesla Coil)
=========================================================
Generates 'vacuum_tesla_coil.png' for future_work Ch. 2.

Red: Fixed-frequency laser detunes and stalls before Schwinger limit.
Cyan: PLL-tracked autoresonant feedback loop tracks shifting resonance
      and achieves pair production at a fraction of brute-force power.
"""
import os, sys, pathlib
import numpy as np
import matplotlib.pyplot as plt

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

def generate():
    print("[*] Generating Vacuum Tesla Coil (Autoresonant Rupture) Figure...")
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#0f0f12')

    t = np.linspace(0, 10, 1000)

    # Resonant frequency of the vacuum under stress (drops as strain increases)
    def f_resonant(strain):
        """Non-linear varactor: f drops as strain increases (Axiom 4)."""
        return 1.0 / (1.0 + 0.3 * strain ** 2)

    # --- LEFT: Energy Buildup ---
    ax1.set_facecolor('#1a1a1f')

    # Fixed-frequency laser: energy builds then stalls
    strain_fixed = np.zeros_like(t)
    f_drive_fixed = 1.0  # Fixed frequency
    for i in range(1, len(t)):
        dt_val = t[1] - t[0]
        f_res = f_resonant(strain_fixed[i-1])
        mismatch = 1.0 / (1.0 + 10 * (f_drive_fixed - f_res) ** 2)
        strain_fixed[i] = strain_fixed[i-1] + 0.1 * mismatch * dt_val

    # PLL-tracked laser: continuously matches resonance
    strain_pll = np.zeros_like(t)
    for i in range(1, len(t)):
        dt_val = t[1] - t[0]
        # PLL perfectly tracks, coupling = 1.0 always
        strain_pll[i] = strain_pll[i-1] + 0.1 * dt_val

    # Schwinger limit (pair production threshold)
    schwinger = 1.5

    ax1.plot(t, strain_fixed, color='#ff3366', lw=3, label='Fixed-Frequency Laser')
    ax1.plot(t, strain_pll, color='#33ffcc', lw=3, label='PLL Autoresonant Feedback')
    ax1.axhline(schwinger, color='#ffcc00', lw=2, linestyle='--',
                label=f'Schwinger Limit (Pair Production)')
    ax1.fill_between(t, schwinger, np.maximum(strain_pll, schwinger), alpha=0.1, color='#33ffcc')

    # Mark where PLL crosses Schwinger
    cross_idx = np.argmin(np.abs(strain_pll - schwinger))
    ax1.axvline(t[cross_idx], color='#33ffcc', lw=1, linestyle=':', alpha=0.5)
    ax1.text(t[cross_idx] + 0.2, schwinger * 0.5, f'Rupture at\nt = {t[cross_idx]:.1f}',
             color='#33ffcc', fontsize=11)

    ax1.set_title("Metric Strain Buildup\n(Energy Accumulation in Vacuum)", color='white', fontsize=14, pad=15)
    ax1.set_xlabel("Time (normalized)", color='#cccccc', fontsize=12)
    ax1.set_ylabel("Metric Strain $|\\chi_{vol}|$", color='#cccccc', fontsize=12)
    ax1.legend(frameon=False, fontsize=11, loc='upper left')
    ax1.grid(True, color='#333344', alpha=0.3)
    ax1.tick_params(colors='#888899')
    for s in ax1.spines.values(): s.set_color('#444455')

    # --- RIGHT: Frequency Tracking ---
    ax2.set_facecolor('#1a1a1f')

    f_res_fixed = np.array([f_resonant(s) for s in strain_fixed])
    f_res_pll = np.array([f_resonant(s) for s in strain_pll])

    ax2.plot(t, f_res_fixed, color='#ff3366', lw=2, alpha=0.6, label='Vacuum $f_{res}$ (Fixed Drive)')
    ax2.plot(t, f_res_pll, color='#33ffcc', lw=2, alpha=0.6, label='Vacuum $f_{res}$ (PLL Drive)')
    ax2.axhline(f_drive_fixed, color='#ff3366', lw=2, linestyle='--', label='Fixed Drive $f_0 = 1.0$')

    # PLL frequency (tracks resonance)
    ax2.plot(t, f_res_pll, color='#33ffcc', lw=3, linestyle='-.', label='PLL Drive $f(t) = f_{res}(t)$')

    # Shade the mismatch
    ax2.fill_between(t, f_drive_fixed, f_res_fixed, alpha=0.1, color='#ff3366')

    ax2.set_title("Frequency Tracking\n(Impedance Match vs Mismatch)", color='white', fontsize=14, pad=15)
    ax2.set_xlabel("Time (normalized)", color='#cccccc', fontsize=12)
    ax2.set_ylabel("Frequency (normalized)", color='#cccccc', fontsize=12)
    ax2.legend(frameon=False, fontsize=10, loc='upper right')
    ax2.grid(True, color='#333344', alpha=0.3)
    ax2.tick_params(colors='#888899')
    for s in ax2.spines.values(): s.set_color('#444455')

    plt.tight_layout(pad=2.5)
    out_dir = project_root / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "vacuum_tesla_coil.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[*] Vacuum Tesla Coil Figure Saved: {out_path}")

if __name__ == "__main__":
    generate()
