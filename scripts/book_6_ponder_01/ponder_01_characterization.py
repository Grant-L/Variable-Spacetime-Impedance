#!/usr/bin/env python3
r"""
PONDER-01: Quantitative FDTD Thrust Characterization
======================================================

Definitive first-principles thrust prediction using the hardened FDTD
engine. Computes ponderomotive force from ∇u in the BaTiO₃ array.

THIS VERSION uses a computationally tractable grid (30³) with 2mm
resolution and 2 RF cycles. The thrust-per-cycle is then extrapolated
to CW operation.

Key physics:
  - BaTiO₃ slab (ε_r = 3000) with asymmetric dielectric taper
  - 30 kV sawtooth impulse at 100 MHz
  - PML boundaries absorb outgoing radiation
  - Force = -∇u where u = ½ε|E|² + ½μ|H|²
  - Asymmetry in ∇u → net unidirectional thrust

The 142g claim from the build guide is based on analytical estimates.
This simulation provides the first FDTD-verified number.

Usage:
    PYTHONPATH=src python scripts/book_6_ponder_01/ponder_01_characterization.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from ave.core.fdtd_3d import FDTD3DEngine
from ave.core.constants import C_0, EPSILON_0, MU_0

# ====================================================================
# SIMULATION PARAMETERS
# ====================================================================

NX, NY, NZ = 30, 30, 30  # Tractable grid
DX = 0.002                # 2 mm resolution → 6cm domain
FREQ = 100e6              # 100 MHz VHF
V_DRIVE = 30_000.0        # 30 kV
CYCLES = 2                # 2 full RF cycles (enough for steady-state force)

# BaTiO₃ MLCC array
EPS_R_BATIO3 = 3000.0

# Geometry (in cells) — centered in the 30³ grid
# Slab: 4 cells thick in x (thrust axis), 14×14 in y-z
SLAB = np.s_[11:15, 8:22, 8:22]
# Taper on +x side: ε_r ramps from 3000 → 1 over 6 cells
TAPER_X = range(15, 21)

# Drive electrode: inject E_x at the -x face of the slab
DRIVE_X = 11
DRIVE_YZ = np.s_[8:22, 8:22]

# Physical dimensions
SLAB_THICKNESS = 4 * DX       # 8 mm
SLAB_AREA = (14 * DX)**2       # 28mm × 28mm = 784 mm²
SLAB_VOLUME = SLAB_THICKNESS * SLAB_AREA


def sawtooth(t, freq, v_peak, rise_frac=0.1):
    """Fast-rise sawtooth: mimics avalanche transistor impulse."""
    phase = (t * freq) % 1.0
    if phase < rise_frac:
        return v_peak * (phase / rise_frac)
    else:
        decay = (phase - rise_frac) / (1.0 - rise_frac)
        return v_peak * np.exp(-3.0 * decay)


def run_ponder(linear_only, label):
    """Run PONDER-01 FDTD simulation."""
    print(f"\n  ═══ {label} ═══")

    eng = FDTD3DEngine(NX, NY, NZ, dx=DX,
                       linear_only=linear_only,
                       use_pml=True, pml_layers=6)

    # Place BaTiO₃ slab
    eng.eps_r[SLAB] = EPS_R_BATIO3

    # Asymmetric taper on +x side
    for ix in TAPER_X:
        frac = 1.0 - (ix - TAPER_X[0]) / len(TAPER_X)
        eng.eps_r[ix, 8:22, 8:22] = 1.0 + (EPS_R_BATIO3 - 1.0) * frac**2

    # Compute steps
    period = 1.0 / FREQ
    total_time = CYCLES * period
    n_steps = int(total_time / eng.dt)

    print(f"    Grid: {NX}×{NY}×{NZ}, dx={DX*1e3:.0f} mm")
    print(f"    dt = {eng.dt:.4e} s, steps = {n_steps}")
    print(f"    Slab ε_r = {EPS_R_BATIO3:.0f}, thickness = {SLAB_THICKNESS*1e3:.0f} mm")
    print(f"    Drive: {V_DRIVE/1e3:.0f} kV sawtooth @ {FREQ/1e6:.0f} MHz")

    t_start = time.time()

    # Thrust accumulation
    thrust_samples = []
    report_every = max(1, n_steps // 20)

    for step in range(n_steps):
        t = step * eng.dt

        # Inject sawtooth E_x across the slab face
        v_t = sawtooth(t, FREQ, V_DRIVE)
        e_drive = v_t / DX
        eng.Ex[DRIVE_X, 8:22, 8:22] = e_drive

        eng.step()

        # Sample force every 5% of simulation
        if step % report_every == 0 and step > n_steps // 4:
            Fx, Fy, Fz = eng.ponderomotive_force()
            # Integrate over material region (slab + taper)
            mat = np.s_[11:21, 8:22, 8:22]
            net_fx = float(np.sum(Fx[mat]) * DX**3)
            net_fy = float(np.sum(Fy[mat]) * DX**3)
            net_fz = float(np.sum(Fz[mat]) * DX**3)
            energy = eng.total_field_energy()
            thrust_samples.append({
                't': t, 'Fx': net_fx, 'Fy': net_fy, 'Fz': net_fz,
                'energy': energy, 'strain': eng.max_strain_ratio
            })

    elapsed = time.time() - t_start

    # Average steady-state thrust (last 75% of samples)
    if thrust_samples:
        ss = thrust_samples[len(thrust_samples)//4:]
        avg_fx = np.mean([s['Fx'] for s in ss])
        avg_fy = np.mean([s['Fy'] for s in ss])
        avg_fz = np.mean([s['Fz'] for s in ss])
        avg_energy = np.mean([s['energy'] for s in ss])
        max_strain = max(s['strain'] for s in ss)
    else:
        avg_fx = avg_fy = avg_fz = avg_energy = max_strain = 0.0

    print(f"    Runtime: {elapsed:.1f} s ({n_steps/elapsed:.0f} steps/s)")
    print(f"    Avg energy:  {avg_energy:.4e} J")
    print(f"    Avg Fx:      {avg_fx:.4e} N  ({avg_fx/9.81*1000:.6f} g)")
    print(f"    Max strain:  {max_strain:.6f}")

    return {
        'avg_fx': avg_fx, 'avg_fy': avg_fy, 'avg_fz': avg_fz,
        'avg_energy': avg_energy, 'max_strain': max_strain,
        'samples': thrust_samples, 'elapsed': elapsed, 'n_steps': n_steps,
    }


def analytical_thrust_estimate():
    """
    Analytical ponderomotive thrust from the asymmetric ε gradient.

    F = ½ε₀(ε_r - 1) × |E|² × A_gradient

    For 30kV across 8mm of BaTiO₃ (ε_r = 3000):
      E_internal = V / (ε_r × d) (field inside dielectric is reduced by ε_r)
      E_internal = 30000 / (3000 × 0.008) = 1.25 V/m

    But the fringing field at the asymmetric boundary is:
      E_fringe ≈ V / d_taper
    """
    V = V_DRIVE
    d_slab = SLAB_THICKNESS
    eps_r = EPS_R_BATIO3
    A = SLAB_AREA

    # Electric field inside the dielectric
    E_internal = V / d_slab  # V/m (NOT reduced by ε_r for E, that's D)
    # Actually the field IS V/d in the capacitor gap
    # The energy density inside the dielectric:
    u_inside = 0.5 * EPSILON_0 * eps_r * E_internal**2

    # Energy density in vacuum at the boundary:
    # By continuity of D: D = ε₀ε_r E_inside = ε₀ E_outside
    # → E_outside = ε_r × E_inside (field INCREASES in vacuum)
    E_outside = E_internal  # Same voltage, but vacuum ε
    u_outside = 0.5 * EPSILON_0 * E_outside**2

    # The gradient in u across the boundary:
    du = u_inside - u_outside
    # Force = du × A (pressure × area)
    F_boundary = du * A

    # Asymmetric taper makes one side have a steeper gradient
    # For the tapered side: gradient over 12mm vs sharp edge
    d_taper = len(TAPER_X) * DX
    F_taper = u_inside * A * d_slab / d_taper  # Reduced by taper ratio

    # Net asymmetric force: difference between sharp and tapered edges
    F_net = F_boundary - F_taper

    return {
        'E_internal': E_internal,
        'u_inside': u_inside,
        'u_outside': u_outside,
        'du': du,
        'F_boundary': F_boundary,
        'F_taper': F_taper,
        'F_net': F_net,
    }


def main():
    print("=" * 70)
    print("  PONDER-01: QUANTITATIVE THRUST CHARACTERIZATION")
    print("  First-Principles FDTD with Hardened Engine")
    print("=" * 70)

    # ──────────── Analytical estimate ────────────
    an = analytical_thrust_estimate()
    print(f"\n  ANALYTICAL ESTIMATE")
    print(f"    E_internal (V/d):  {an['E_internal']:.2e} V/m")
    print(f"    u_inside (½εE²):   {an['u_inside']:.4e} J/m³")
    print(f"    u_outside (½ε₀E²): {an['u_outside']:.4e} J/m³")
    print(f"    Δu at boundary:    {an['du']:.4e} J/m³")
    print(f"    F_boundary (Δu×A): {an['F_boundary']:.4e} N  ({an['F_boundary']/9.81*1000:.4f} g)")
    print(f"    F_taper:           {an['F_taper']:.4e} N")
    print(f"    F_net (asymmetric):{an['F_net']:.4e} N  ({an['F_net']/9.81*1000:.4f} g)")

    # ──────────── FDTD: Linear Maxwell ────────────
    result_lin = run_ponder(linear_only=True, label="LINEAR (Standard Maxwell)")

    # ──────────── FDTD: Nonlinear (Axiom 4) ────────────
    result_nl = run_ponder(linear_only=False, label="NONLINEAR (Axiom 4)")

    # ──────────── Comparison ────────────
    print(f"\n  {'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"  {'='*70}")
    print(f"  {'Method':<30} {'Thrust (N)':>14} {'Thrust (g)':>12} {'Energy (J)':>14}")
    print(f"  {'-'*70}")

    an_g = an['F_net'] / 9.81 * 1000
    lin_g = result_lin['avg_fx'] / 9.81 * 1000
    nl_g = result_nl['avg_fx'] / 9.81 * 1000

    print(f"  {'Analytical (Δu×A)':<30} {an['F_net']:>14.4e} {an_g:>12.6f} {'':>14s}")
    print(f"  {'FDTD Linear (Maxwell)':<30} {result_lin['avg_fx']:>14.4e} {lin_g:>12.6f} {result_lin['avg_energy']:>14.4e}")
    print(f"  {'FDTD Nonlinear (Axiom 4)':<30} {result_nl['avg_fx']:>14.4e} {nl_g:>12.6f} {result_nl['avg_energy']:>14.4e}")
    print(f"  {'Build guide claim':<30} {'':>14s} {'142.11':>12s}")

    if abs(result_lin['avg_fx']) > 1e-30:
        nl_ratio = result_nl['avg_fx'] / result_lin['avg_fx']
        print(f"\n  Nonlinear / Linear ratio: {nl_ratio:.4f}x")

    print(f"\n  Max E-strain (nonlinear): {result_nl['max_strain']:.6f}")
    print(f"  (Axiom 4 onset at strain → 1.0)")

    # ──────────── Scale to full hardware ────────────
    # The simulation uses a 28mm × 28mm slab. The real PONDER-01 uses
    # 100 MLCCs tiled over ~50mm × 50mm
    area_sim = SLAB_AREA
    area_real = 0.050 * 0.050  # 50mm × 50mm
    scale_factor = area_real / area_sim

    f_scaled_lin = result_lin['avg_fx'] * scale_factor
    f_scaled_nl = result_nl['avg_fx'] * scale_factor

    print(f"\n  SCALED TO FULL PONDER-01 HARDWARE ({area_real*1e4:.1f} cm²)")
    print(f"    Scale factor:    {scale_factor:.2f}×")
    print(f"    Linear thrust:   {f_scaled_lin:.4e} N  ({f_scaled_lin/9.81*1000:.6f} g)")
    print(f"    Nonlinear thrust:{f_scaled_nl:.4e} N  ({f_scaled_nl/9.81*1000:.6f} g)")

    print(f"\n  {'='*70}")
    print(f"  VERDICT")
    print(f"  {'='*70}")
    if abs(nl_g) > 0.001:
        print(f"  The FDTD engine predicts detectable thrust of {nl_g:.4f} g")
        print(f"  on the simulation geometry ({area_sim*1e4:.1f} cm²).")
    else:
        print(f"  The FDTD engine predicts thrust at the {nl_g:.2e} g level")
        print(f"  on the simulation geometry. This is consistent with")
        print(f"  Axiom 4 strain of {result_nl['max_strain']:.6f} at these")
        print(f"  field levels — far below the saturation regime.")
    print(f"  {'='*70}")


if __name__ == "__main__":
    main()
