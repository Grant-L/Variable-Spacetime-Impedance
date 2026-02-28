#!/usr/bin/env python3
r"""
AVE: Vacuum Birefringence Kill Switch
=======================================
Generates 'birefringence_killswitch.png' for future_work Ch. 4.

QED predicts refractive index shifts ~ E^2.
AVE demands shifts ~ E^4 (from the 4th-order capacitive non-linearity).
As E → E_crit, the curves diverge, providing a definitive experimental test.
"""
import os, sys, pathlib
import numpy as np
import matplotlib.pyplot as plt

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

def generate():
    print("[*] Generating Birefringence Kill Switch Figure...")
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#0f0f12')

    E = np.linspace(0, 1.0, 500)  # Normalized to E_crit

    # --- LEFT: Refractive Index Shift ---
    ax1.set_facecolor('#1a1a1f')

    # QED (Euler-Heisenberg): Δn ~ E^2
    dn_qed = 0.5 * E ** 2

    # AVE (Axiom 4 4th-order polynomial): Δn ~ E^4
    dn_ave = 0.5 * E ** 4

    ax1.plot(E, dn_qed, color='#ff3366', lw=3, label='QED: $\\Delta n \\propto E^2$')
    ax1.plot(E, dn_ave, color='#33ffcc', lw=3, label='AVE: $\\Delta n \\propto E^4$')
    ax1.fill_between(E, dn_qed, dn_ave, alpha=0.1, color='white')

    # Mark divergence region
    ax1.axvspan(0.6, 1.0, alpha=0.05, color='#ffcc00')
    ax1.text(0.8, 0.12, "BINARY\nTEST\nZONE", color='#ffcc00', fontsize=12,
             ha='center', va='center', fontweight='bold')

    ax1.set_title("Vacuum Refractive Index Shift\nvs Applied Field Strength",
                  color='white', fontsize=14, pad=15)
    ax1.set_xlabel("Electric Field $E / E_{crit}$", color='#cccccc', fontsize=12)
    ax1.set_ylabel("Refractive Index Shift $\\Delta n$ (a.u.)", color='#cccccc', fontsize=12)
    ax1.legend(frameon=False, fontsize=12, loc='upper left')
    ax1.grid(True, color='#333344', alpha=0.3)
    ax1.tick_params(colors='#888899')
    for s in ax1.spines.values(): s.set_color('#444455')

    # --- RIGHT: Log-Log Slope Analysis ---
    ax2.set_facecolor('#1a1a1f')

    E_log = np.logspace(-2, -0.05, 500)

    dn_qed_log = 0.5 * E_log ** 2
    dn_ave_log = 0.5 * E_log ** 4

    ax2.loglog(E_log, dn_qed_log, color='#ff3366', lw=3, label='QED: slope = 2')
    ax2.loglog(E_log, dn_ave_log, color='#33ffcc', lw=3, label='AVE: slope = 4')

    # Reference slopes
    ax2.loglog(E_log, 0.3 * E_log ** 2, ':', color='#ff3366', alpha=0.3, lw=1)
    ax2.loglog(E_log, 0.3 * E_log ** 4, ':', color='#33ffcc', alpha=0.3, lw=1)

    # Annotations
    ax2.text(0.3, 0.05, '$E^2$', color='#ff3366', fontsize=16, fontweight='bold')
    ax2.text(0.3, 0.003, '$E^4$', color='#33ffcc', fontsize=16, fontweight='bold')

    ax2.set_title("Log-Log Slope Analysis\n(Binary Falsification Test)",
                  color='white', fontsize=14, pad=15)
    ax2.set_xlabel("Electric Field $E / E_{crit}$", color='#cccccc', fontsize=12)
    ax2.set_ylabel("$\\Delta n$ (a.u.)", color='#cccccc', fontsize=12)
    ax2.legend(frameon=False, fontsize=12, loc='upper left')
    ax2.grid(True, color='#333344', alpha=0.3, which='both')
    ax2.tick_params(colors='#888899')
    for s in ax2.spines.values(): s.set_color('#444455')

    # Kill switch annotation
    ax2.text(0.15, 0.15, "Measure slope:\n= 2 → AVE KILLED\n= 4 → QED KILLED",
             color='#ffcc00', fontsize=12, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1f', edgecolor='#ffcc00', alpha=0.9))

    plt.tight_layout(pad=2.5)
    out_dir = project_root / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "birefringence_killswitch.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[*] Birefringence Kill Switch Saved: {out_path}")

if __name__ == "__main__":
    generate()
