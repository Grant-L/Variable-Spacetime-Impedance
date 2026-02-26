#!/usr/bin/env python3
r"""
Water Anomaly: The 4°C Density Maximum from Impedance Matching
================================================================

Demonstrates that water's anomalous density maximum at 3.98°C arises
from the H-bond network reaching peak impedance matching (maximum Q)
at that temperature.

Generates a comparison plot: AVE model vs. empirical Kell (1975) data.

Usage:
    PYTHONPATH=src python scripts/future_work/simulate_water_anomaly.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ave.fluids.water import (
    WaterMolecule, water_density, dielectric_constant_water,
    ave_density_model, find_density_maximum, impedance_crossing_temperature,
    hbond_network_q_factor
)


def main():
    print("=" * 60)
    print("  WATER ANOMALY: 4°C Density Maximum")
    print("  from H-Bond Impedance Matching")
    print("=" * 60)

    mol = WaterMolecule()
    print(f"\n  H₂O Molecular Properties (AVE)")
    print(f"  ────────────────────────────────")
    print(f"  O-H bond length:     {mol.oh_bond_length*1e10:.4f} Å")
    print(f"  H-O-H angle:         {mol.hoh_angle:.2f}°")
    print(f"  O-H stretch freq:    {mol.oh_resonant_frequency/1e12:.2f} THz")
    print(f"  AVE inductance:      {mol.inductance_ave:.3e} H")
    print(f"  AVE capacitance:     {mol.capacitance_ave:.3e} F")
    print(f"  AVE impedance:       {mol.impedance_ave:.3e} Ω-equiv")
    print(f"  H-bond energy:       {mol.hbond_energy/1.602e-19:.3f} eV")

    # ─────────────────────────────────────────────────────────
    # Temperature sweep
    # ─────────────────────────────────────────────────────────
    temps = np.linspace(-2, 30, 1000)
    rho_empirical = np.array([water_density(T) for T in temps])
    rho_ave = np.array([ave_density_model(T) for T in temps])
    eps_r = np.array([dielectric_constant_water(T) for T in temps])
    Q_hb = np.array([hbond_network_q_factor(T) for T in temps])

    # Find maxima
    T_max_emp = temps[np.argmax(rho_empirical)]
    T_max_ave, rho_max_ave = find_density_maximum()
    T_cross = impedance_crossing_temperature()

    print(f"\n  Temperature of Maximum Density")
    print(f"  ──────────────────────────────")
    print(f"  Empirical (Kell 1975):   {T_max_emp:.2f}°C")
    print(f"  AVE model:               {T_max_ave:.2f}°C")
    print(f"  Impedance crossing:      {T_cross:.2f}°C")
    print(f"  Known value:             3.98°C")
    print(f"  AVE deviation:           {abs(T_max_ave - 3.98):.2f}°C")

    print(f"\n  Density at Maximum")
    print(f"  ──────────────────")
    print(f"  Empirical:  {water_density(3.98):.4f} kg/m³")
    print(f"  AVE model:  {rho_max_ave:.4f} kg/m³")

    print(f"\n  Dielectric Constant ε_r(T)")
    print(f"  ──────────────────────────")
    print(f"  ε_r(0°C)  = {dielectric_constant_water(0):.1f}")
    print(f"  ε_r(25°C) = {dielectric_constant_water(25):.1f}")
    print(f"  ε_r(100°C)= {dielectric_constant_water(100):.1f}")

    # ─────────────────────────────────────────────────────────
    # AVE interpretation
    # ─────────────────────────────────────────────────────────
    print(f"\n  ═══════════════════════════════════════════════")
    print(f"  AVE INTERPRETATION")
    print(f"  ═══════════════════════════════════════════════")
    print(f"  At 4°C, the thermal phonon frequency matches")
    print(f"  the H-bond network's fundamental resonance.")
    print(f"  → Maximum impedance matching (Q peaks)")
    print(f"  → Maximum energy transmission efficiency")
    print(f"  → Minimum molecular volume (tightest packing)")
    print(f"  → DENSITY MAXIMUM")
    print(f"")
    print(f"  Above 4°C: thermal expansion dominates")
    print(f"  Below 4°C: tetrahedral ice ordering expands")
    print(f"  At exactly 4°C: perfect impedance balance")
    print(f"  ═══════════════════════════════════════════════")

    # ─────────────────────────────────────────────────────────
    # Generate visualization
    # ─────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('#08081a')

        # [0,0] Density: Empirical vs AVE
        ax = axes[0, 0]
        ax.set_facecolor('#08081a')
        ax.plot(temps, rho_empirical, color='#44aaff', linewidth=2,
                label='Empirical (Kell 1975)')
        ax.plot(temps, rho_ave, color='#ff6644', linewidth=2, linestyle='--',
                label='AVE Model')
        ax.axvline(x=3.98, color='#ffcc44', linewidth=1, alpha=0.5, linestyle=':')
        ax.axvline(x=T_max_ave, color='#ff4444', linewidth=1, alpha=0.5, linestyle=':')
        ax.set_title('Density vs Temperature', color='#ccccee', fontweight='bold')
        ax.set_xlabel('Temperature (°C)', color='#888888')
        ax.set_ylabel('Density (kg/m³)', color='#888888')
        ax.legend(facecolor='#111122', edgecolor='#333355', labelcolor='#ccccdd')
        ax.tick_params(colors='#666688')
        for spine in ax.spines.values():
            spine.set_color('#222244')
        ax.grid(True, color='#151530', alpha=0.5)

        # [0,1] Q factor
        ax = axes[0, 1]
        ax.set_facecolor('#08081a')
        ax.fill_between(temps, 0, Q_hb, color='#44ff88', alpha=0.2)
        ax.plot(temps, Q_hb, color='#44ff88', linewidth=2)
        ax.axvline(x=3.98, color='#ffcc44', linewidth=1, alpha=0.5, linestyle=':',
                  label='3.98°C')
        ax.set_title('H-Bond Network Q Factor', color='#44ff88', fontweight='bold')
        ax.set_xlabel('Temperature (°C)', color='#888888')
        ax.set_ylabel('Relative Q', color='#888888')
        ax.legend(facecolor='#111122', edgecolor='#333355', labelcolor='#ccccdd')
        ax.tick_params(colors='#666688')
        for spine in ax.spines.values():
            spine.set_color('#222244')
        ax.grid(True, color='#151530', alpha=0.5)

        # [1,0] Dielectric constant
        ax = axes[1, 0]
        ax.set_facecolor('#08081a')
        ax.plot(temps, eps_r, color='#ffcc44', linewidth=2)
        ax.set_title('Dielectric Constant ε_r(T)', color='#ffcc44', fontweight='bold')
        ax.set_xlabel('Temperature (°C)', color='#888888')
        ax.set_ylabel('ε_r', color='#888888')
        ax.tick_params(colors='#666688')
        for spine in ax.spines.values():
            spine.set_color('#222244')
        ax.grid(True, color='#151530', alpha=0.5)

        # [1,1] Density residual (AVE - Empirical)
        ax = axes[1, 1]
        ax.set_facecolor('#08081a')
        residual = rho_ave - rho_empirical
        ax.plot(temps, residual, color='#ff88ff', linewidth=2)
        ax.axhline(y=0, color='#444466', linewidth=0.5)
        ax.set_title('Density Residual (AVE − Empirical)', color='#ff88ff',
                     fontweight='bold')
        ax.set_xlabel('Temperature (°C)', color='#888888')
        ax.set_ylabel('Δρ (kg/m³)', color='#888888')
        ax.tick_params(colors='#666688')
        for spine in ax.spines.values():
            spine.set_color('#222244')
        ax.grid(True, color='#151530', alpha=0.5)

        fig.suptitle("Water Anomaly: 4°C Density Maximum from Impedance Matching",
                    color='#ccccee', fontsize=14, fontweight='bold', y=0.98)
        fig.tight_layout()

        out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'water_anomaly.png')
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  ✓ Water anomaly plot saved to: {out_path}")

    except ImportError:
        print("\n  (matplotlib not available, skipping visualization)")


if __name__ == "__main__":
    main()
