#!/usr/bin/env python3
r"""
AVE Seismology: Earth Interior as Impedance Gradient
=====================================================

Demonstrates that seismic wave propagation through the Earth is equivalent
to electromagnetic wave propagation through a layered impedance medium.

The script:
  1. Builds the PREM radial impedance profile
  2. Computes reflection coefficients at all major boundaries
  3. Calculates P-wave and S-wave vertical travel times
  4. Compares AVE predictions against PREM empirical values
  5. Generates a visualization of the Earth's impedance structure

Key AVE insight: The Moho discontinuity has Γ = 0.17 — identical to
what a microwave engineer would calculate for an impedance step from
18.9 to 26.7 MRayl. Seismology IS RF engineering at geological scale.

Usage:
    PYTHONPATH=src python scripts/future_work/simulate_seismic_ave.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ave.geophysics.seismic import (
    PREM_LAYERS, SeismicLayer,
    reflection_coefficient, transmission_coefficient,
    travel_time, all_reflections, build_1d_impedance_profile
)


def main():
    print("=" * 70)
    print("  AVE SEISMOLOGY: Earth Interior as Impedance Gradient")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────
    # 1. Layer properties
    # ─────────────────────────────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  PREM Layer Properties (AVE Impedance Mapping)                    │")
    print("├───────────────┬───────┬───────┬───────┬──────────┬────────┬───────┤")
    print("│ Layer         │ ρ     │ V_p   │ V_s   │ Z_p      │ ε_r    │ μ_r   │")
    print("│               │ kg/m³ │ km/s  │ km/s  │ MRayl    │ (AVE)  │ (AVE) │")
    print("├───────────────┼───────┼───────┼───────┼──────────┼────────┼───────┤")
    for layer in PREM_LAYERS:
        z_p = layer.acoustic_impedance_p / 1e6
        print(f"│ {layer.name:<13} │ {layer.rho:>5} │ {layer.v_p/1e3:>5.1f} │ "
              f"{layer.v_s/1e3:>5.1f} │ {z_p:>8.1f} │ {layer.eps_r_ave:>6.3f} │ "
              f"{layer.mu_r_ave:>5.2f}  │")
    print("└───────────────┴───────┴───────┴───────┴──────────┴────────┴───────┘")

    # ─────────────────────────────────────────────────────────────
    # 2. Reflection coefficients at all boundaries
    # ─────────────────────────────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  Reflection Coefficients at Major Boundaries                      │")
    print("├──────────────────────────────────┬──────────┬──────────┬──────────┤")
    print("│ Boundary                         │ Γ_p      │ Γ_s      │ |Γ_p|²  │")
    print("├──────────────────────────────────┼──────────┼──────────┼──────────┤")

    for i in range(len(PREM_LAYERS) - 1):
        l1 = PREM_LAYERS[i]
        l2 = PREM_LAYERS[i + 1]
        name = f"{l1.name} → {l2.name}"
        gamma_p = reflection_coefficient(l1, l2, 'p')
        gamma_s = reflection_coefficient(l1, l2, 's')
        power_p = gamma_p**2
        print(f"│ {name:<32} │ {gamma_p:>+8.4f} │ {gamma_s:>+8.4f} │ {power_p:>8.4f} │")
    print("└──────────────────────────────────┴──────────┴──────────┴──────────┘")

    # ─────────────────────────────────────────────────────────────
    # 3. Travel times
    # ─────────────────────────────────────────────────────────────
    t_p = travel_time(PREM_LAYERS, 'p')
    t_s = travel_time(PREM_LAYERS, 's')

    # PREM empirical values for vertical travel time (surface → center)
    t_p_prem = 1212.0  # seconds (approximately)
    t_s_prem = 1680.0  # seconds (S-waves stop at outer core)

    print(f"\n  P-wave vertical travel time (surface → center):")
    print(f"    AVE computed:  {t_p:.1f} s")
    print(f"    PREM expected: ~{t_p_prem:.0f} s")
    print(f"    Deviation:     {abs(t_p - t_p_prem)/t_p_prem * 100:.1f}%")

    # S-wave only through solid layers
    solid_layers = [l for l in PREM_LAYERS if l.v_s > 0]
    t_s_solid = travel_time(solid_layers, 's')
    print(f"\n  S-wave travel time (solid layers only):")
    print(f"    AVE computed:  {t_s_solid:.1f} s")

    # ─────────────────────────────────────────────────────────────
    # 4. Key AVE insights
    # ─────────────────────────────────────────────────────────────
    moho_gamma = reflection_coefficient(PREM_LAYERS[1], PREM_LAYERS[2], 'p')
    cmb_gamma = reflection_coefficient(PREM_LAYERS[4], PREM_LAYERS[5], 'p')

    print(f"\n  ═══════════════════════════════════════════════════════════")
    print(f"  AVE INTERPRETATION")
    print(f"  ═══════════════════════════════════════════════════════════")
    print(f"  Moho:     Γ = {moho_gamma:+.4f} (17% amplitude reflection)")
    print(f"            → Identical to a 19→27 MRayl impedance step")
    print(f"            → RF equivalent: coax cable with connector mismatch")
    print(f"")
    print(f"  CMB:      Γ = {cmb_gamma:+.4f} ({abs(cmb_gamma)*100:.0f}% amplitude)")
    print(f"            → μ_r jumps to ∞ (liquid core: zero shear)")
    print(f"            → S-waves TOTALLY REFLECTED (shadow zone)")
    print(f"            → RF equivalent: open circuit (broken inductor)")
    print(f"")
    print(f"  The Earth is a spherical waveguide with 6 impedance layers.")
    print(f"  Seismology = RF engineering at geological frequencies.")
    print(f"  ═══════════════════════════════════════════════════════════")

    # ─────────────────────────────────────────────────────────────
    # 5. 1D impedance profile
    # ─────────────────────────────────────────────────────────────
    profile = build_1d_impedance_profile(dx_km=5.0)
    print(f"\n  1D impedance profile: {len(profile['depth_km'])} cells @ 5 km resolution")
    print(f"  ε_r range: [{np.min(profile['eps_r']):.3f}, {np.max(profile['eps_r']):.3f}]")
    print(f"  μ_r range: [{np.min(profile['mu_r']):.3f}, {np.max(profile['mu_r']):.3f}]")

    # ─────────────────────────────────────────────────────────────
    # 6. Generate visualization
    # ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=True)
        fig.patch.set_facecolor('#08081a')

        depth = profile['depth_km']

        params = [
            ('V_p (km/s)', profile['v_p'] / 1e3, '#ff6644', axes[0]),
            ('V_s (km/s)', profile['v_s'] / 1e3, '#44aaff', axes[1]),
            ('ε_r (AVE)',  profile['eps_r'],      '#ffcc44', axes[2]),
            ('μ_r (AVE)',  profile['mu_r'],        '#44ff88', axes[3]),
        ]

        for title, data, color, ax in params:
            ax.set_facecolor('#08081a')
            ax.fill_betweenx(depth, 0, data, color=color, alpha=0.3)
            ax.plot(data, depth, color=color, linewidth=2)
            ax.set_title(title, color=color, fontsize=12, fontweight='bold')
            ax.tick_params(colors='#666688', labelsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(title, color='#888888', fontsize=9)
            for spine in ax.spines.values():
                spine.set_color('#222244')
            ax.grid(True, color='#151530', alpha=0.5)

        # Mark boundaries
        boundaries = [15, 35, 410, 660, 2891, 5150]
        boundary_names = ['Moho', 'L.Crust', '410', '660', 'CMB', 'ICB']
        for ax in axes:
            for bd, bn in zip(boundaries, boundary_names):
                ax.axhline(y=bd, color='#ff4444', linewidth=0.8, alpha=0.4, linestyle='--')
            # Label boundaries on first axis only
        for bd, bn in zip(boundaries, boundary_names):
            axes[0].text(0.5, bd + 30, bn, color='#ff6666', fontsize=7, alpha=0.7)

        axes[0].set_ylabel('Depth (km)', color='#888888', fontsize=10)

        fig.suptitle("Earth Interior: AVE Impedance Profile (PREM)",
                    color='#ccccee', fontsize=14, fontweight='bold', y=0.98)

        out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'earth_impedance_profile.png')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  ✓ Earth impedance profile saved to: {out_path}")

    except ImportError:
        print("\n  (matplotlib not available, skipping visualization)")


if __name__ == "__main__":
    main()
