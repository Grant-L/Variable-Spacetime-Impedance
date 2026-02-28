#!/usr/bin/env python3
r"""
AVE: Vacuum Aerodynamics — CFD Streamlining
=============================================
Generates 'vacuum_aerodynamics.png' for future_work Ch. 2.

Shows how active metric saturation ahead of a vessel eliminates
the inductive bow shock, creating laminar vacuum flow and
collapsing inertial resistance.
"""
import os, sys, pathlib
import numpy as np
import matplotlib.pyplot as plt

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

def generate():
    print("[*] Generating Vacuum Aerodynamics Figure...")
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#0f0f12')

    N = 150
    x = np.linspace(-3, 3, N)
    y = np.linspace(-2, 2, N)
    X, Y = np.meshgrid(x, y)

    # Vessel shape (simple ellipse at origin)
    a, b = 0.8, 0.3
    vessel_mask = ((X / a) ** 2 + (Y / b) ** 2) < 1.0

    # --- LEFT: Standard Inertial Flight (Bow Shock) ---
    ax1.set_facecolor('#1a1a1f')

    # Uniform flow field + point source repulsion (bow shock)
    U0 = 1.0
    R = np.sqrt(X ** 2 + Y ** 2) + 0.01
    Vx = U0 + 0.5 / R ** 2 * X / R
    Vy = 0.5 / R ** 2 * Y / R

    # Impedance field (high at bow = shock)
    Z_field = 1.0 + 2.0 * np.exp(-((X + 0.8) ** 2 + Y ** 2) / 0.3)

    im1 = ax1.pcolormesh(X, Y, Z_field, cmap='hot', shading='auto', alpha=0.7, vmin=0.5, vmax=3.5)
    ax1.streamplot(X, Y, Vx, Vy, color='white', linewidth=0.8, density=2.0, arrowsize=1)
    # Draw vessel
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.fill(a * np.cos(theta), b * np.sin(theta), color='#3399ff', alpha=0.8)
    ax1.plot(a * np.cos(theta), b * np.sin(theta), color='white', lw=2)

    ax1.set_title("Standard Flight\n(Inductive Bow Shock)", color='#ff3366', fontsize=14, pad=15)
    ax1.set_xlabel("$x / \\ell_{node}$", color='#cccccc')
    ax1.set_ylabel("$y / \\ell_{node}$", color='#cccccc')
    ax1.set_xlim([-2.5, 2.5])
    ax1.set_ylim([-1.8, 1.8])
    ax1.set_aspect('equal')
    ax1.tick_params(colors='#888899')
    cb1 = plt.colorbar(im1, ax=ax1, shrink=0.7)
    cb1.set_label('Impedance $Z$', color='#cccccc')
    cb1.ax.tick_params(colors='#cccccc')
    for s in ax1.spines.values(): s.set_color('#444455')

    # Label
    ax1.annotate('BOW\nSHOCK', xy=(-0.8, 0), xytext=(-2, 1),
                 arrowprops=dict(arrowstyle='->', color='#ff3366', lw=2),
                 color='#ff3366', fontsize=12, fontweight='bold')

    # --- RIGHT: AVE Metric Streamlining (No Bow Shock) ---
    ax2.set_facecolor('#1a1a1f')

    # Saturated metric ahead of vessel: impedance drops → laminar flow
    Z_saturated = np.where(X < 0,
                           0.3 + 0.2 * np.exp(-((X + 1.5) ** 2 + Y ** 2) / 0.8),
                           1.0 + 0.3 * np.exp(-Y ** 2 / 0.5))

    # Smooth laminar flow (no shock divergence)
    Vx_lam = U0 * np.ones_like(X)
    Vy_lam = -0.3 * Y * np.exp(-X ** 2 / 1.5)

    im2 = ax2.pcolormesh(X, Y, Z_saturated, cmap='cool', shading='auto', alpha=0.7, vmin=0.1, vmax=1.5)
    ax2.streamplot(X, Y, Vx_lam, Vy_lam, color='white', linewidth=0.8, density=2.0, arrowsize=1)
    ax2.fill(a * np.cos(theta), b * np.sin(theta), color='#33ffcc', alpha=0.8)
    ax2.plot(a * np.cos(theta), b * np.sin(theta), color='white', lw=2)

    ax2.set_title("AVE Metric Streamlining\n(Saturated Laminar Flow)", color='#33ffcc', fontsize=14, pad=15)
    ax2.set_xlabel("$x / \\ell_{node}$", color='#cccccc')
    ax2.set_ylabel("$y / \\ell_{node}$", color='#cccccc')
    ax2.set_xlim([-2.5, 2.5])
    ax2.set_ylim([-1.8, 1.8])
    ax2.set_aspect('equal')
    ax2.tick_params(colors='#888899')
    cb2 = plt.colorbar(im2, ax=ax2, shrink=0.7)
    cb2.set_label('Impedance $Z$ (Saturated)', color='#cccccc')
    cb2.ax.tick_params(colors='#cccccc')
    for s in ax2.spines.values(): s.set_color('#444455')

    ax2.text(-2, -1.4, "$Z_d \\ll Z_c$\nLaminar Vacuum Flow",
             color='#33ffcc', fontsize=11, fontweight='bold')

    plt.tight_layout(pad=2.5)
    out_dir = project_root / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "vacuum_aerodynamics.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[*] Vacuum Aerodynamics Figure Saved: {out_path}")

if __name__ == "__main__":
    generate()
