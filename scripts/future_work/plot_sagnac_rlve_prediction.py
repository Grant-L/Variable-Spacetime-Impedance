#!/usr/bin/env python3
r"""
AVE: Sagnac-RLVE Exact Prediction
===================================
Generates 'sagnac_rlve_prediction.png' for future_work Ch. 4.

Shows the exact parameter-free phase shift prediction for the
Sagnac-RLVE experiment: 200m fiber around a tungsten rotor at 10k RPM
→ ~2.07 radians, plus the GR near-zero prediction for comparison.
"""
import os, sys, pathlib
import numpy as np
import matplotlib.pyplot as plt

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

from ave.core.constants import C_0

def generate():
    print("[*] Generating Sagnac-RLVE Prediction Figure...")
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#0f0f12')

    # --- Physical parameters ---
    rho_W = 19300       # Tungsten density (kg/m^3)
    rho_Al = 2700       # Aluminum density (kg/m^3)
    rho_bulk = 7.916e6  # Bulk vacuum mass-energy density (kg/m^3)
    R_rotor = 0.15      # 15 cm radius
    RPM = 10000
    v_tan = 2 * np.pi * R_rotor * RPM / 60  # ~157 m/s
    lam = 1550e-9       # 1550nm telecom laser

    kappa_W = rho_W / rho_bulk   # ~0.00244
    kappa_Al = rho_Al / rho_bulk

    v_net_W = v_tan * kappa_W    # ~0.38 m/s
    v_net_Al = v_tan * kappa_Al

    # --- LEFT: Phase Shift vs Fiber Length ---
    ax1.set_facecolor('#1a1a1f')

    L_fiber = np.linspace(0, 500, 500)

    phi_W = 4 * np.pi * L_fiber * v_net_W / (lam * C_0)
    phi_Al = 4 * np.pi * L_fiber * v_net_Al / (lam * C_0)
    phi_GR = 1e-20 * np.ones_like(L_fiber)  # GR: near-zero at lab scale

    ax1.semilogy(L_fiber, phi_W, color='#33ffcc', lw=3, label=f'AVE: Tungsten ($\\rho$ = {rho_W} kg/m³)')
    ax1.semilogy(L_fiber, phi_Al, color='#ffcc00', lw=2.5, label=f'AVE: Aluminum ($\\rho$ = {rho_Al} kg/m³)')
    ax1.semilogy(L_fiber, phi_GR, 'w--', lw=2, alpha=0.5, label='GR: Frame-Dragging ($\\sim 10^{-20}$ rad)')

    ax1.axhline(np.pi, color='#ff3366', lw=1.5, linestyle=':', alpha=0.7, label='$\\pi$ rad')

    # Mark design point
    design_phi = 4 * np.pi * 200 * v_net_W / (lam * C_0)
    ax1.plot(200, design_phi, 'o', color='#33ffcc', markersize=12, zorder=10)
    ax1.annotate(f'  200m: {design_phi:.2f} rad', xy=(200, design_phi),
                 color='#33ffcc', fontsize=12, fontweight='bold', va='center')

    ax1.set_title("Sagnac-RLVE: Phase Shift vs Fiber Length",
                  color='white', fontsize=14, pad=15)
    ax1.set_xlabel("Fiber Optic Spool Length (m)", color='#cccccc', fontsize=12)
    ax1.set_ylabel("$|\\Delta\\phi|$ (Radians)", color='#cccccc', fontsize=12)
    ax1.legend(frameon=False, fontsize=10, loc='lower right')
    ax1.grid(True, color='#333344', alpha=0.3, which='both')
    ax1.set_ylim([1e-22, 1e2])
    ax1.tick_params(colors='#888899')
    for s in ax1.spines.values(): s.set_color('#444455')

    # --- RIGHT: Material Density Ratio Test ---
    ax2.set_facecolor('#1a1a1f')

    materials = ['Tungsten\n($\\rho$ = 19,300)', 'Steel\n($\\rho$ = 7,800)',
                 'Aluminum\n($\\rho$ = 2,700)', 'GR Prediction\n(Density-Independent)']
    densities = [19300, 7800, 2700, 0]
    phi_vals = [4 * np.pi * 200 * (d / rho_bulk) * v_tan / (lam * C_0) if d > 0 else 1e-20
                for d in densities]
    colors = ['#33ffcc', '#3399ff', '#ffcc00', '#888899']

    bars = ax2.bar(materials, phi_vals, color=colors, edgecolor='white',
                   linewidth=1.5, alpha=0.85, width=0.6)

    for bar, phi_val in zip(bars, phi_vals):
        if phi_val > 0.01:
            ax2.text(bar.get_x() + bar.get_width() / 2, phi_val + 0.05,
                     f'{phi_val:.2f} rad', ha='center', va='bottom', color='white',
                     fontsize=11, fontweight='bold')

    ax2.set_title("Material Density Ratio ($\\Psi$) Kill-Switch\n(200m Fiber, 10k RPM)",
                  color='white', fontsize=14, pad=15)
    ax2.set_ylabel("Phase Shift $|\\Delta\\phi|$ (Radians)", color='#cccccc', fontsize=12)
    ax2.grid(True, color='#333344', alpha=0.2, axis='y')
    ax2.tick_params(colors='#888899')
    for s in ax2.spines.values(): s.set_color('#444455')

    # Psi ratio annotation
    psi = rho_W / rho_Al
    ax2.text(0.5, max(phi_vals) * 0.7,
             f'$\\Psi = \\rho_W / \\rho_{{Al}} \\approx {psi:.2f}$',
             color='#33ffcc', fontsize=14, fontweight='bold', ha='center',
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1f', edgecolor='#33ffcc', alpha=0.9))

    plt.tight_layout(pad=2.5)
    out_dir = project_root / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "sagnac_rlve_prediction.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[*] Sagnac-RLVE Prediction Saved: {out_path}")

if __name__ == "__main__":
    generate()
