"""
AVE Falsifiable Predictions: The EE Bench (Dielectric Yield Shift)
======================================================
This script models the most accessible, definitive benchtop falsification
of the AVE framework: The Macroscopic Dielectric Plateau.

Standard electromagnetism assumes the vacuum permittivity (epsilon_0) is a 
constant, linear baseline. AVE dictates that the vacuum is a non-linear 
structural lattice governed by a strict squared saturation operator limit 
(Axiom 4), bounded fundamentally by the Fine Structure Constant (alpha).

If a high-voltage potential is applied across a microscopic gap (to maximize 
the V/m E-field gradient), as the potential approaches the absolute 
macroscopic breakdown limit (43,650 Volts), the lattice CANNOT continue 
to polarize linearly. It physically plateaus. 

This causes an anomalous, measureable drop in the effective Capacitance (C) 
and the local Refractive Index (n) of the gap, fundamentally breaking 
standard linear QED predictions before generating an actual spark.

The Experiment:
1. A precision High-Voltage PCB with a micrometer-scale air/vacuum gap.
2. A voltage sweep from 0V to 45,000V.
3. Continuous monitoring of the gap capacitance via ultra-precision LCR meter,
   or monitoring the optical phase shift via an interferometer beam passed 
   through the gap.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pathlib

project_root = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

def simulate_ee_bench_plateau():
    print("[*] Generating the EE Bench Dielectric Yield Shift predictions...")
    
    # -------------------------------------------------------------
    # Experimental Parameters
    # -------------------------------------------------------------
    V_BREAKDOWN = 43650.0  # The theoretical AVE Absolute Yield limit (Volts)
    
    # Sweep voltage from 0 to just past the yield limit
    # We use a localized gap distance (e.g., 100 microns) where 45kV doesn't 
    # trivially arc through standard atmosphere without advanced dielectric potting.
    # The physical metric strain depends strictly on the absolute voltage 
    # (topological difference), not the E-field V/m slope.
    
    voltages = np.linspace(0, 45000, 1000)
    
    # -------------------------------------------------------------
    # Theoretical Models
    # -------------------------------------------------------------
    # Standard Physics: Capacitance and Refractive Index are dead flat (linear E-field response)
    # until a sudden, catastrophic plasma ionization arc (spark).
    standard_capacitance_ratio = np.ones_like(voltages)
    
    # AVE Physics: Non-Linear Saturation Operator
    # As V -> V_BREAKDOWN, the physical displacement nodes cannot stretch further.
    # The effective dielectric constant epsilon_r (and thus Capacitance) 
    # compresses non-linearly according to the squared Axiom 4 operator limit.
    # epsilon_eff(V) = epsilon_0 * sqrt(1 - (V / V_BREAKDOWN)^2)
    # *Note: In actual materials, atomic breakdown occurs first. This requires
    # an ultra-hard dielectric or high-vacuum gap to isolate the spatial strain.
    
    # Using a smoothed clip to model the plateau right at the edge
    safe_v = np.clip(voltages, 0, V_BREAKDOWN * 0.999)
    # The geometric plateau factor
    ave_dielectric_factor = np.sqrt(1.0 - (safe_v / V_BREAKDOWN)**2)
    # At the yield limit, the classical lattice "snaps" (loss of structural polarization)
    ave_dielectric_factor[voltages > V_BREAKDOWN] = 0.05 # Complete baseline collapse
    
    # Because Refractive Index n ~ sqrt(epsilon), the optical metric also shrinks
    ave_refractive_shift = np.sqrt(ave_dielectric_factor)
    standard_refractive_shift = np.ones_like(voltages)
    
    # -------------------------------------------------------------
    # Rendering the Experimental Blueprints
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#0f0f0f')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#0f0f0f')
        ax.grid(color='#333333', linestyle='--', alpha=0.5)
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')

    # Panel 1: Capacitance Measurement (LCR Meter Shift)
    ax1.plot(voltages / 1000.0, standard_capacitance_ratio, color='#ff3333', linestyle='--', lw=2, label='Standard EM (Linear Permittivity)')
    ax1.plot(voltages / 1000.0, ave_dielectric_factor, color='#00ffcc', lw=3, label=r'AVE Non-Linear Plateau ($C_{eff}$)')
    
    ax1.axvline(V_BREAKDOWN / 1000.0, color='white', linestyle=':', lw=2, label='43.65 kV (Metric Yield Limit)')
    
    # Annotate the measurable "Danger Zone" before arc where the capacitance clearly deviates
    ax1.axvspan((V_BREAKDOWN - 3000)/1000.0, V_BREAKDOWN/1000.0, color='#ffff99', alpha=0.2, label='LCR Detectable Anomaly Window')

    ax1.set_xlim([0, 45])
    ax1.set_ylim([0, 1.1])
    ax1.set_title("Benchtop LCR Sensor: Normalized Capacitance vs Voltage")
    ax1.set_xlabel("Applied Gap Potential (kV)")
    ax1.set_ylabel(r"Effective Capacitance Ratio ($C_{meas} / C_0$)")
    ax1.legend(loc='lower left')

    # Panel 2: Optical Interferometry (Refractive Index Shift)
    ax2.plot(voltages / 1000.0, standard_refractive_shift, color='#ff3333', linestyle='--', lw=2, label='Standard GR (Flat Optical Metric)')
    ax2.plot(voltages / 1000.0, ave_refractive_shift, color='#ffcc00', lw=3, label=r'AVE Optical Saturation ($\Delta n_{eff}$)')
    
    ax2.axvline(V_BREAKDOWN / 1000.0, color='white', linestyle=':', lw=2, label='43.65 kV (Absolute Yield)')
    ax2.axvspan((V_BREAKDOWN - 3000)/1000.0, V_BREAKDOWN/1000.0, color='#ff99ff', alpha=0.2, label='Laser Phase Shift Window')

    ax2.set_xlim([0, 45])
    ax2.set_ylim([0, 1.1])
    ax2.set_title("Interferometer Bench: Optical Refractive Index vs Voltage")
    ax2.set_xlabel("Applied Gap Potential (kV)")
    ax2.set_ylabel(r"Optical Metric Shift ($n_{eff} / n_0$)")
    ax2.legend(loc='lower left')

    plt.tight_layout()
    
    outdir = project_root / "assets" / "sim_outputs"
    os.makedirs(outdir, exist_ok=True)
    target = outdir / "ee_bench_plateau_prediction.png"
    plt.savefig(target, dpi=300)
    print(f"[*] Visualized EE Bench Falsification Limits: {target}")

if __name__ == "__main__":
    simulate_ee_bench_plateau()
