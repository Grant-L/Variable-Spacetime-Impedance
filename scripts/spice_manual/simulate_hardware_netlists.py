#!/usr/bin/env python3
r"""
SPICE Manual Ch.4: EE Bench Dielectric Yield Plateau
======================================================

Generates a figure showing the AVE-predicted non-linear capacitance
plateau as the gap voltage approaches V_yield = 43.65 kV, alongside
the PONDER-01 cascaded transmission-line thrust prediction.

Outputs to spice_manual/assets/sim_outputs/ for the SPICE manual PDF.

Usage:
    PYTHONPATH=src python scripts/spice_manual/simulate_hardware_netlists.py
"""

import sys
import os
import pathlib
import numpy as np

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

from ave.core.constants import ALPHA, M_E, C_0, e_charge, Z_0

OUT_DIR = project_root / "spice_manual" / "assets" / "sim_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def gen_ee_bench():
    """EE Bench: Dielectric plateau and PONDER-01 transmission line."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── Derived constants ──
    V_yield = float((M_E * C_0**2 / e_charge) * np.sqrt(ALPHA))
    l_node = 3.86e-13
    E_yield = V_yield / l_node

    # ── Panel 1: C_eff / C_0 vs Gap Voltage ──
    V_sweep = np.linspace(0, V_yield * 1.02, 1000)
    safe_V = np.clip(V_sweep, 0, V_yield * 0.999)
    C_ratio = np.sqrt(1 - (safe_V / V_yield)**2)
    C_ratio[V_sweep >= V_yield] = 0.05

    # Standard physics: flat
    C_std = np.ones_like(V_sweep)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.patch.set_facecolor('#0a0a0a')

    # ── Panel 1: Capacitance Plateau ──
    ax1 = axes[0]
    ax1.set_facecolor('#111111')
    ax1.plot(V_sweep / 1e3, C_std, color='#ff3333', lw=2, linestyle='--',
            label='Standard EM (Linear)')
    ax1.plot(V_sweep / 1e3, C_ratio, color='#00ffcc', lw=3,
            label=r'AVE: $C_{eff}/C_0 = \sqrt{1-(V/V_{yield})^2}$')
    ax1.axvline(V_yield / 1e3, color='white', lw=2, linestyle=':',
               label=f'$V_{{yield}}$ = {V_yield/1e3:.2f} kV')
    ax1.axvspan(V_yield * 0.85 / 1e3, V_yield / 1e3, alpha=0.15,
               color='#ffff99', label='Detectable Anomaly Window')
    ax1.set_xlabel('Gap Voltage (kV)', color='white', fontsize=11)
    ax1.set_ylabel(r'$C_{eff}/C_0$', color='white', fontsize=11)
    ax1.set_title('EE Bench: Capacitance Plateau\n(LCR Meter Measurement)',
                  color='white', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#333',
              labelcolor='white', loc='lower left')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15, color='white')
    ax1.set_ylim([0, 1.15])
    for s in ax1.spines.values(): s.set_color('#333')

    # ── Panel 2: PONDER-01 Cascaded Transmission Line ──
    ax2 = axes[1]
    ax2.set_facecolor('#111111')

    # Model the PONDER-01 as cascaded lumped LC sections
    # 10 air layers + 10 FR4 layers
    N_layers = 20
    layer_types = ['air', 'fr4'] * 10
    layer_thickness = 100e-6  # 100 μm per layer

    z_pos = np.cumsum([layer_thickness] * N_layers) * 1e3  # mm
    V_phase = []
    Z_local = []

    for lt in layer_types:
        if lt == 'air':
            eps_r = 1.0
            Z_l = float(Z_0)
        else:
            eps_r = 4.3
            Z_l = float(Z_0) / np.sqrt(eps_r)
        V_phase.append(30000 / np.sqrt(eps_r))  # V_rms normalized
        Z_local.append(Z_l)

    ax2.bar(range(N_layers), Z_local, color=['#33ffcc' if lt == 'air' else '#ff6b6b'
            for lt in layer_types], alpha=0.8, edgecolor='white', lw=0.5)
    ax2.set_xlabel('Layer Index', color='white', fontsize=11)
    ax2.set_ylabel(r'Local Impedance $Z$ ($\Omega$)', color='white', fontsize=11)
    ax2.set_title('PONDER-01: Cascaded LC Stack\n(Air/FR4 Impedance Mismatch)',
                  color='white', fontsize=13, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.15, color='white', axis='y')
    for s in ax2.spines.values(): s.set_color('#333')

    # Legend for layer types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#33ffcc', edgecolor='white', label=f'Air (Z = {float(Z_0):.0f} Ω)'),
                      Patch(facecolor='#ff6b6b', edgecolor='white', label=f'FR4 (Z = {float(Z_0)/np.sqrt(4.3):.0f} Ω)')]
    ax2.legend(handles=legend_elements, fontsize=9, facecolor='#1a1a1a',
              edgecolor='#333', labelcolor='white')

    # ── Panel 3: Ponderomotive Force Gradient ──
    ax3 = axes[2]
    ax3.set_facecolor('#111111')

    # ∇|E|² across the stack boundary
    z = np.linspace(0, 2.0, 500)  # mm
    # Sharp gradient at each air/FR4 boundary
    grad_E2 = np.zeros_like(z)
    for b in range(10):
        boundary = (2 * b + 1) * layer_thickness * 1e3  # mm
        grad_E2 += 1e6 * np.exp(-((z - boundary) / 0.02)**2) * (1 if b % 2 == 0 else -0.3)

    ax3.plot(z, grad_E2, color='#ffcc00', lw=2)
    ax3.fill_between(z, 0, grad_E2, where=grad_E2 > 0, alpha=0.2, color='#ffcc00')
    ax3.fill_between(z, 0, grad_E2, where=grad_E2 < 0, alpha=0.2, color='#ff3366')
    ax3.set_xlabel('Position along stack (mm)', color='white', fontsize=11)
    ax3.set_ylabel(r'$\nabla |E|^2$ (a.u.)', color='white', fontsize=11)
    ax3.set_title('PONDER-01: Ponderomotive Gradient\n(Asymmetric Force at Boundaries)',
                  color='white', fontsize=13, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.15, color='white')
    for s in ax3.spines.values(): s.set_color('#333')

    plt.tight_layout(pad=2)
    out_path = OUT_DIR / "hardware_netlist_overview.png"
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[*] Saved: {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  SPICE Manual Ch.4: Hardware Netlist Figures")
    print("=" * 60)
    gen_ee_bench()
    print("=" * 60)
