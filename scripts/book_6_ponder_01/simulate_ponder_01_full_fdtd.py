#!/usr/bin/env python3
r"""
PONDER-01: Full 3D FDTD Ponderomotive Thrust Simulation
=========================================================

This is the definitive PONDER-01 simulation using the upgraded FDTD engine
with dual nonlinearity (ε + μ), spatial materials, PML boundaries, and
ponderomotive force extraction.

The geometry:
    - BaTiO₃ MLCC array (ε_r = 3000) in the center
    - Asymmetric ground plane: flat on one side, tapered on the other
    - 30kV sawtooth drive at 100 MHz VHF
    - PML absorbing boundaries (cosmological horizon)

Outputs:
    - Energy density ∇u map
    - Net ponderomotive force vector (thrust)
    - Comparison: linear vs Axiom 4 nonlinear

Usage:
    PYTHONPATH=src python scripts/book_6_ponder_01/simulate_ponder_01_full_fdtd.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

try:
    from ave.core.fdtd_3d_jax import FDTD3DEngineJAX as FDTD3DEngine
except ImportError:
    from ave.core.fdtd_3d import FDTD3DEngine
from ave.core.constants import C_0, EPSILON_0

# ====================================================================
# SIMULATION PARAMETERS
# ====================================================================

GRID_SIZE = 60          # 60×60×60 cells
DX = 0.001              # 1 mm resolution (60mm = 6cm total domain)
FREQ = 100e6            # 100 MHz VHF
V_DRIVE = 30_000.0      # 30 kV drive voltage
TOTAL_CYCLES = 10       # Number of RF cycles to simulate
SAWTOOTH_RISE = 0.1     # Rise fraction (10% of period = fast rise)

# BaTiO₃ parameters
EPS_R_BATIO3 = 3000.0   # Relative permittivity

# Geometry (in cells)
# The MLCC array is a slab in the center
SLAB_X = (25, 35)       # 10 cells thick in x (the thrust axis)
SLAB_Y = (15, 45)       # 30 cells wide in y
SLAB_Z = (15, 45)       # 30 cells wide in z

# Asymmetric ground plane: tapered electrode on the +x side
# Creates the ∇n gradient that rectifies into thrust
TAPER_START = 35        # Starts right after the slab
TAPER_END = 45          # Extends 10 cells beyond


def build_sawtooth(t, freq, v_peak, rise_frac):
    """Generate a sawtooth waveform: fast rise, slow decay."""
    phase = (t * freq) % 1.0
    if phase < rise_frac:
        # Fast rise
        return v_peak * (phase / rise_frac)
    else:
        # Slow exponential decay
        decay = (phase - rise_frac) / (1.0 - rise_frac)
        return v_peak * np.exp(-3.0 * decay)


def run_simulation(linear_only, label):
    """Run a full PONDER-01 simulation and return thrust data."""
    print(f"\n{'='*60}")
    print(f"  PONDER-01 FDTD: {label}")
    print(f"{'='*60}")

    eng = FDTD3DEngine(
        GRID_SIZE, GRID_SIZE, GRID_SIZE,
        dx=DX,
        linear_only=linear_only,
        use_pml=True,
        pml_layers=8,
    )

    # Place BaTiO₃ slab
    eng.eps_r[SLAB_X[0]:SLAB_X[1], SLAB_Y[0]:SLAB_Y[1], SLAB_Z[0]:SLAB_Z[1]] = EPS_R_BATIO3
    print(f"  BaTiO₃ slab: x=[{SLAB_X[0]}:{SLAB_X[1]}], ε_r={EPS_R_BATIO3}")

    # Asymmetric taper: linearly decreasing ε_r from slab edge to vacuum
    for ix in range(TAPER_START, TAPER_END):
        frac = 1.0 - (ix - TAPER_START) / (TAPER_END - TAPER_START)
        eps_taper = 1.0 + (EPS_R_BATIO3 - 1.0) * frac**2
        eng.eps_r[ix, SLAB_Y[0]:SLAB_Y[1], SLAB_Z[0]:SLAB_Z[1]] = eps_taper
    print(f"  Taper: x=[{TAPER_START}:{TAPER_END}], ε_r ramps to 1.0")

    # Drive parameters
    period = 1.0 / FREQ
    total_time = TOTAL_CYCLES * period
    n_steps = int(total_time / eng.dt)
    print(f"  Drive: {V_DRIVE/1000:.0f} kV sawtooth @ {FREQ/1e6:.0f} MHz")
    print(f"  Steps: {n_steps}, dt = {eng.dt:.4e} s")
    print(f"  Total time: {total_time*1e9:.2f} ns ({TOTAL_CYCLES} cycles)")

    # Drive electrode positions (inject E_x across the slab)
    drive_y = slice(SLAB_Y[0], SLAB_Y[1])
    drive_z = slice(SLAB_Z[0], SLAB_Z[1])
    drive_x = SLAB_X[0]  # Drive from the flat side

    # Run simulation
    thrust_history = []
    energy_history = []

    report_interval = max(1, n_steps // 10)

    for step in range(n_steps):
        t = step * eng.dt

        # Inject sawtooth drive: E_x = V / dx across the slab entrance
        v_t = build_sawtooth(t, FREQ, V_DRIVE, SAWTOOTH_RISE)
        e_drive = v_t / DX  # V/m
        eng.Ex = eng.Ex.at[drive_x, drive_y, drive_z].set(e_drive)

        eng.step()

        # Periodically extract force
        if step % report_interval == 0 and step > 0:
            Fx, Fy, Fz = eng.ponderomotive_force()

            # Net force on the material region (slab + taper)
            mat_region = np.s_[SLAB_X[0]:TAPER_END, SLAB_Y[0]:SLAB_Y[1], SLAB_Z[0]:SLAB_Z[1]]
            net_fx = float(np.sum(Fx[mat_region]) * DX**3)
            net_fy = float(np.sum(Fy[mat_region]) * DX**3)
            net_fz = float(np.sum(Fz[mat_region]) * DX**3)

            energy = eng.total_field_energy()
            thrust_history.append((t, net_fx, net_fy, net_fz))
            energy_history.append((t, energy))

            pct = 100.0 * step / n_steps
            print(f"  {pct:5.1f}%  E={energy:.4e} J  Fx={net_fx:.4e} N  strain={eng.max_strain_ratio:.4e}")

    # Final force measurement
    Fx, Fy, Fz = eng.ponderomotive_force()
    mat_region = np.s_[SLAB_X[0]:TAPER_END, SLAB_Y[0]:SLAB_Y[1], SLAB_Z[0]:SLAB_Z[1]]
    final_fx = float(np.sum(Fx[mat_region]) * DX**3)
    final_fy = float(np.sum(Fy[mat_region]) * DX**3)
    final_fz = float(np.sum(Fz[mat_region]) * DX**3)
    final_energy = eng.total_field_energy()

    print(f"\n  FINAL STATE:")
    print(f"    Energy: {final_energy:.4e} J")
    print(f"    Net Fx: {final_fx:.4e} N  ({final_fx/9.81*1000:.4f} g-force)")
    print(f"    Net Fy: {final_fy:.4e} N")
    print(f"    Net Fz: {final_fz:.4e} N")
    print(f"    Max E-strain: {eng.max_strain_ratio:.6f}")
    print(f"    Max B-strain: {eng.max_mag_strain:.6f}")

    return {
        'thrust': thrust_history,
        'energy': energy_history,
        'final_fx': final_fx,
        'final_energy': final_energy,
        'max_strain': eng.max_strain_ratio,
    }


def main():
    print("=" * 60)
    print("  PROJECT PONDER-01: FULL 3D FDTD THRUST SIMULATION")
    print("=" * 60)
    print(f"  Grid: {GRID_SIZE}³ @ {DX*1000:.1f} mm resolution")
    print(f"  Domain: {GRID_SIZE*DX*100:.1f} cm × {GRID_SIZE*DX*100:.1f} cm × {GRID_SIZE*DX*100:.1f} cm")
    print(f"  BaTiO₃: ε_r = {EPS_R_BATIO3:.0f}")
    print(f"  Drive: {V_DRIVE/1000:.0f} kV sawtooth @ {FREQ/1e6:.0f} MHz")
    print(f"  Boundaries: PML (8 layers)")

    # Run linear (standard Maxwell)
    result_lin = run_simulation(linear_only=True, label="LINEAR (Standard Maxwell)")

    # Run nonlinear (Axiom 4)
    result_nl = run_simulation(linear_only=False, label="NONLINEAR (Axiom 4: ε + μ)")

    # Compare
    print("\n" + "=" * 60)
    print("  COMPARISON: LINEAR vs NONLINEAR")
    print("=" * 60)
    print(f"  {'Quantity':<25s} {'Linear':>15s} {'Nonlinear':>15s} {'Ratio':>10s}")
    print(f"  {'-'*65}")

    fx_l = result_lin['final_fx']
    fx_nl = result_nl['final_fx']
    e_l = result_lin['final_energy']
    e_nl = result_nl['final_energy']

    ratio_fx = fx_nl / fx_l if abs(fx_l) > 1e-30 else float('inf')
    ratio_e = e_nl / e_l if abs(e_l) > 1e-30 else float('inf')

    print(f"  {'Net Thrust Fx (N)':<25s} {fx_l:>15.4e} {fx_nl:>15.4e} {ratio_fx:>10.4f}")
    print(f"  {'Field Energy (J)':<25s} {e_l:>15.4e} {e_nl:>15.4e} {ratio_e:>10.4f}")
    print(f"  {'Max E-strain':<25s} {result_lin['max_strain']:>15.6f} {result_nl['max_strain']:>15.6f}")

    # Thrust in grams
    thrust_g = fx_nl / 9.81 * 1000
    print(f"\n  PONDER-01 predicted thrust: {thrust_g:.4f} grams")
    print(f"  (Build guide predicts: 142.11 grams)")


if __name__ == "__main__":
    main()
