#!/usr/bin/env python3
r"""
AVE: Nuclear Fusion Crisis Audit
=================================
Generates 'fusion_crisis_audit.png' for Book 4, Chapter 14.

4-Panel comparison showing why existing fusion approaches hit fundamental
AVE hardware limits, and how Active Metric Compression solves them:
  1. Tokamak: 15 keV ions generate 60.3 kV strain → shatters 43.65 kV yield limit
  2. Laser ICF (NIF): implosion causes Zero-Impedance Phase Slip → RT failure
  3. Pulsed FRC: shatters 511 kV Dielectric Snap → pair production drain
  4. AVE Solution: metric compression shrinks Bohr radii safely below danger zone
"""
import os, sys, pathlib
import numpy as np
import matplotlib.pyplot as plt

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

from ave.core.constants import C_0

def generate():
    print("[*] Generating Fusion Crisis Audit Figure...")
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.patch.set_facecolor('#0f0f12')

    V_sat = 43.65  # kV, Dielectric Saturation limit
    V_snap = 511.0  # kV, Dielectric Snap (pair production)

    # --- Panel 1: Tokamak Crisis ---
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1f')
    T = np.linspace(0, 20, 500)  # keV
    V_strain = T * 4.02  # Topological strain (kV) per keV for D-T collision
    ax.plot(T, V_strain, color='#ff3366', lw=3, label='Ion Collision Strain (kV)')
    ax.axhline(V_sat, color='#ffcc00', lw=2, linestyle='--', label=f'$V_{{sat}}$ = {V_sat} kV')
    ax.fill_between(T, V_sat, np.maximum(V_strain, V_sat), alpha=0.15, color='#ff3366')
    ax.axvline(15, color='white', linestyle=':', alpha=0.5)
    ax.text(15.3, 20, '15 keV\n(Ignition)', color='white', fontsize=10)
    ax.set_title("Tokamak Crisis\n(Thermal Strain Exceeds Yield)", color='#ff3366', fontsize=13, pad=10)
    ax.set_xlabel("Ion Temperature (keV)", color='#cccccc')
    ax.set_ylabel("Topological Strain (kV)", color='#cccccc')
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color='#333344', alpha=0.3)
    ax.tick_params(colors='#888899')
    for s in ax.spines.values(): s.set_color('#444455')

    # --- Panel 2: Laser ICF (NIF) Crisis ---
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1f')
    r = np.linspace(0.01, 1.0, 500)
    P_implosion = 100.0 / r**2  # Pressure scaling
    Z_impedance = np.where(P_implosion > 200, 0.01, 1.0 - (P_implosion / 200) ** 2)
    ax.semilogy(r, P_implosion, color='#3399ff', lw=3, label='Implosion Pressure')
    ax2 = ax.twinx()
    ax2.plot(r, Z_impedance, color='#ffcc00', lw=2.5, linestyle='--', label='$Z_{vacuum}$ (Impedance)')
    ax2.set_ylabel("Vacuum Impedance $Z$", color='#ffcc00', fontsize=11)
    ax2.tick_params(colors='#ffcc00')
    ax.set_title("Laser ICF (NIF) Crisis\n(Zero-Impedance Phase Slip)", color='#3399ff', fontsize=13, pad=10)
    ax.set_xlabel("Capsule Radius (normalized)", color='#cccccc')
    ax.set_ylabel("Implosion Pressure (arb.)", color='#cccccc')
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    ax2.legend(frameon=False, fontsize=10, loc='center right')
    ax.grid(True, color='#333344', alpha=0.3)
    ax.tick_params(colors='#888899')
    for s in ax.spines.values(): s.set_color('#444455')

    # --- Panel 3: Pulsed FRC Crisis ---
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1f')
    B_field = np.linspace(0, 30, 500)  # Tesla
    V_frc = B_field ** 2 * 0.6  # kV scaling with B^2
    ax.plot(B_field, V_frc, color='#cc33ff', lw=3, label='FRC Peak Strain (kV)')
    ax.axhline(V_snap, color='#ff3366', lw=2, linestyle='--', label=f'$V_{{snap}}$ = {V_snap} kV (Pair Production)')
    ax.axhline(V_sat, color='#ffcc00', lw=2, linestyle=':', label=f'$V_{{sat}}$ = {V_sat} kV')
    ax.fill_between(B_field, V_snap, np.maximum(V_frc, V_snap), alpha=0.1, color='#cc33ff')
    ax.set_title("Pulsed FRC Crisis\n(Pair Production Energy Drain)", color='#cc33ff', fontsize=13, pad=10)
    ax.set_xlabel("Magnetic Field (T)", color='#cccccc')
    ax.set_ylabel("Peak Topological Strain (kV)", color='#cccccc')
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color='#333344', alpha=0.3)
    ax.tick_params(colors='#888899')
    for s in ax.spines.values(): s.set_color('#444455')

    # --- Panel 4: AVE Solution (Metric Compression) ---
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1f')
    n_metric = np.linspace(1.0, 5.0, 500)
    T_required = 15.0 / n_metric**2  # Required temperature drops with n^2
    V_strain_ave = T_required * 4.02
    ax.plot(n_metric, T_required, color='#33ffcc', lw=3, label='Required $T_{ignition}$ (keV)')
    ax3 = ax.twinx()
    ax3.plot(n_metric, V_strain_ave, color='#ffcc00', lw=2.5, linestyle='--', label='Peak Strain (kV)')
    ax3.axhline(V_sat, color='#ff3366', lw=1.5, linestyle=':', alpha=0.5)
    ax3.set_ylabel("Peak Strain (kV)", color='#ffcc00', fontsize=11)
    ax3.tick_params(colors='#ffcc00')
    ax.set_title("AVE Solution\n(Active Metric Compression)", color='#33ffcc', fontsize=13, pad=10)
    ax.set_xlabel("Metric Refractive Index $n_{scalar}$", color='#cccccc')
    ax.set_ylabel("Required Ignition Temperature (keV)", color='#cccccc')
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    ax3.legend(frameon=False, fontsize=10, loc='center right')
    ax.grid(True, color='#333344', alpha=0.3)
    ax.tick_params(colors='#888899')
    for s in ax.spines.values(): s.set_color('#444455')

    # Safe zone annotation
    ax.axhspan(0, 2, alpha=0.05, color='#33ffcc')
    ax.text(3.5, 1.0, "SAFE\nOPERATING\nZONE", color='#33ffcc', fontsize=11,
            ha='center', va='center', fontweight='bold')

    plt.tight_layout(pad=2.5)

    out_dir = project_root / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "fusion_crisis_audit.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[*] Fusion Crisis Audit Figure Saved: {out_path}")

if __name__ == "__main__":
    generate()
