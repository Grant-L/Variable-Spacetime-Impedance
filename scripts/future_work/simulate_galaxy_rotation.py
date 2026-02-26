#!/usr/bin/env python3
r"""
Galaxy Rotation Curves: No Dark Matter, Zero Free Parameters
==============================================================

Demonstrates that flat galaxy rotation curves emerge naturally from
the AVE vacuum's Bingham-plastic rheology. The lattice's mutual
inductance provides additional gravitational acceleration when the
Newtonian field is weak (g_N < a₀), producing the same effect as
a dark matter halo.

The script computes:
  1. Newtonian (baryons only) rotation curves for 5 galaxies
  2. AVE (with lattice drag) rotation curves — NO dark matter
  3. The Radial Acceleration Relation (McGaugh et al. 2016)
  4. Derives a₀ from cosmology: a₀ = cH₀/(2π)

Key result: The AVE model reproduces flat rotation curves across
5 orders of magnitude in galaxy mass — from DDO 154 (dwarf) to
UGC 2885 (giant) — using only baryonic mass and one universal
constant a₀ derived from the Hubble parameter.

Usage:
    PYTHONPATH=src python scripts/future_work/simulate_galaxy_rotation.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ave.gravity.galactic_rotation import (
    GalaxyModel, GALAXY_CATALOG, A0_MOND, KPC, M_SUN,
    ave_rotation_velocity, ave_effective_acceleration,
    radial_acceleration_relation, derive_a0_from_cosmology,
)
from ave.core.constants import G


def main():
    print("=" * 75)
    print("  GALAXY ROTATION CURVES: No Dark Matter, Zero Free Parameters")
    print("=" * 75)

    # 1. Derive a₀ from cosmology
    a0_derived = derive_a0_from_cosmology()
    print(f"\n  a₀ derivation:")
    print(f"    Empirical (McGaugh):   a₀ = {A0_MOND:.2e} m/s²")
    print(f"    AVE (cH₀/2π):         a₀ = {a0_derived:.2e} m/s²")
    print(f"    Deviation:             {abs(a0_derived - A0_MOND)/A0_MOND*100:.1f}%")

    # 2. Compute rotation curves for all galaxies
    print(f"\n  {'Galaxy':<20} {'M_bar (M☉)':>12} {'V_max(N)':>10} {'V_max(AVE)':>10} {'V_flat':>10}")
    print("  " + "─" * 65)

    galaxy_data = {}
    for name, gal in GALAXY_CATALOG.items():
        r_max = 15 * gal.R_d
        radii = np.linspace(0.1 * gal.R_d, r_max, 200)
        v_newton = np.array([gal.newtonian_velocity(r) for r in radii])
        v_ave = np.array([ave_rotation_velocity(gal, r) for r in radii])

        v_max_n = np.max(v_newton) / 1e3  # km/s
        v_max_a = np.max(v_ave) / 1e3
        v_flat = v_ave[-1] / 1e3  # velocity at outer edge

        M_total = (gal.M_disk + gal.M_bulge) / M_SUN
        print(f"  {name:<20} {M_total:>12.2e} {v_max_n:>8.1f}   {v_max_a:>8.1f}   {v_flat:>8.1f}")

        galaxy_data[name] = (radii, v_newton, v_ave)

    # 3. Key insight: why rotation curves flatten
    print(f"\n  ═══════════════════════════════════════════════════════════════════════")
    print(f"  WHY CURVES FLATTEN (AVE EXPLANATION)")
    print(f"  ═══════════════════════════════════════════════════════════════════════")
    print(f"  At large r, Newtonian g_N = GM/r² drops below a₀ = {A0_MOND:.1e}")
    print(f"  The vacuum lattice is still INTACT at these low shear rates.")
    print(f"  Its mutual inductance resists differential rotation → extra 'gravity'")
    print(f"  The effective acceleration transitions to:")
    print(f"     g_eff ≈ √(g_N × a₀)    (deep MOND limit)")
    print(f"  This gives v ∝ (GM × a₀)^(1/4) = constant → FLAT ROTATION CURVE")
    print(f"")
    print(f"  At small r, g_N >> a₀ → lattice is saturated → no extra drag")
    print(f"  → pure Newtonian behavior (as observed)")
    print(f"")
    print(f"  The crossover radius r_c where g_N(r_c) = a₀:")
    for name, gal in GALAXY_CATALOG.items():
        M = gal.M_disk + gal.M_bulge
        r_c = np.sqrt(G * M / A0_MOND)
        print(f"    {name:<20}: r_c = {r_c/KPC:.1f} kpc")
    print(f"  ═══════════════════════════════════════════════════════════════════════")

    # 4. Baryonic Tully-Fisher Relation
    print(f"\n  BARYONIC TULLY-FISHER RELATION (BTFR)")
    print(f"  v_flat⁴ = G × M_bar × a₀")
    print(f"  {'Galaxy':<20} {'M_bar':>12} {'v_flat⁴/(GM)':>14} {'a₀ derived':>12}")
    print("  " + "─" * 60)
    for name, (radii, v_n, v_a) in galaxy_data.items():
        gal = GALAXY_CATALOG[name]
        M = gal.M_disk + gal.M_bulge
        v_flat = v_a[-1]
        a0_btfr = v_flat**4 / (G * M)
        print(f"  {name:<20} {M/M_SUN:>12.2e} {v_flat**4/(G*M):>14.3e} {a0_btfr:>12.3e}")

    # 5. Generate visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.patch.set_facecolor('#08081a')

        colors_n = ['#444466', '#444466', '#444466', '#444466', '#444466']
        colors_a = ['#ff6644', '#44aaff', '#44ff88', '#ffcc44', '#ff88ff']

        for idx, (name, (radii, v_n, v_a)) in enumerate(galaxy_data.items()):
            ax = axes.flat[idx]
            ax.set_facecolor('#08081a')
            r_kpc = radii / KPC

            ax.plot(r_kpc, v_n / 1e3, color=colors_n[idx], linewidth=2,
                    linestyle='--', label='Newtonian (baryons only)')
            ax.fill_between(r_kpc, 0, v_n / 1e3, color=colors_n[idx], alpha=0.1)

            ax.plot(r_kpc, v_a / 1e3, color=colors_a[idx], linewidth=2.5,
                    label='AVE (lattice drag)')

            ax.set_title(name, color=colors_a[idx], fontsize=13, fontweight='bold')
            ax.set_xlabel('r (kpc)', color='#888888')
            ax.set_ylabel('v (km/s)', color='#888888')
            ax.legend(fontsize=7, facecolor='#111122', edgecolor='#333355',
                     labelcolor='#ccccdd', loc='lower right')
            ax.tick_params(colors='#666688', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#222244')
            ax.grid(True, color='#151530', alpha=0.5)
            ax.set_ylim(bottom=0)

        # 6th panel: Radial Acceleration Relation
        ax = axes.flat[5]
        ax.set_facecolor('#08081a')
        g_bar = np.logspace(-13, -8, 500)
        g_obs = radial_acceleration_relation(g_bar)
        ax.loglog(g_bar, g_obs, color='#ff6644', linewidth=2.5, label='AVE prediction')
        ax.loglog(g_bar, g_bar, color='#444466', linewidth=1, linestyle='--',
                 label='g_obs = g_bar (no DM)')
        ax.loglog(g_bar, np.sqrt(g_bar * A0_MOND), color='#ffcc44', linewidth=1,
                 linestyle=':', label='deep MOND: √(g·a₀)')
        ax.axhline(y=A0_MOND, color='#44ff88', linewidth=0.8, alpha=0.5, linestyle=':')
        ax.axvline(x=A0_MOND, color='#44ff88', linewidth=0.8, alpha=0.5, linestyle=':')
        ax.set_title('Radial Acceleration Relation', color='#ccccee',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('g_bar (m/s²)', color='#888888')
        ax.set_ylabel('g_obs (m/s²)', color='#888888')
        ax.legend(fontsize=7, facecolor='#111122', edgecolor='#333355',
                 labelcolor='#ccccdd')
        ax.tick_params(colors='#666688', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#222244')
        ax.grid(True, color='#151530', alpha=0.3)
        ax.text(3e-13, 5e-11, f'a₀ = cH₀/2π\n= {derive_a0_from_cosmology():.2e}',
               color='#44ff88', fontsize=8, alpha=0.8)

        fig.suptitle("Galaxy Rotation Curves: No Dark Matter (AVE Lattice Drag)",
                    color='#ccccee', fontsize=15, fontweight='bold', y=0.98)
        fig.tight_layout()

        out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'galaxy_rotation_curves.png')
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  ✓ Galaxy rotation curves saved to: {out_path}")

    except ImportError:
        print("\n  (matplotlib not available, skipping visualization)")


if __name__ == "__main__":
    main()
