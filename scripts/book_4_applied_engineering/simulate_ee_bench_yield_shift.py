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
    # Experimental Parameters (Zero-Parameter Foundation)
    # -------------------------------------------------------------
    from ave.core.constants import ALPHA, M_E, C_0, e_charge
    
    # Fundamental Node Coherence Length
    l_node = 3.86e-13  
    
    # Absolute Localized Node Voltage Limit (derived dynamically)
    # V_node = (m_e * c^2) / e * sqrt(alpha) -> 43,653 Volts
    V_NODE_LIMIT = (M_E * C_0**2 / e_charge) * np.sqrt(ALPHA)
    
    # Macroscopic E-Field Saturation Limit (V/m)
    # This is the physical strain limit normalized over the macroscopic geometry
    E_BREAKDOWN = V_NODE_LIMIT / l_node  # Approx 1.13e17 V/m
    
    # Sweep E-field from 0 to just past the yield limit
    # We sweep by fraction of the breakdown absolute limit for clean charting
    e_fields = np.linspace(0, E_BREAKDOWN, 1000)
    
    # -------------------------------------------------------------
    # Theoretical Models
    # -------------------------------------------------------------
    # Standard Physics: Capacitance and Refractive Index are dead flat (linear E-field response)
    # until a sudden, catastrophic plasma ionization arc (spark).
    standard_capacitance_ratio = np.ones_like(e_fields)
    
    # AVE Physics: Non-Linear Saturation Operator
    # As E -> E_BREAKDOWN, the physical displacement nodes cannot stretch further.
    # The effective dielectric constant epsilon_r (and thus Capacitance) 
    # compresses non-linearly according to the squared Axiom 4 operator limit.
    # epsilon_eff(E) = epsilon_0 * sqrt(1 - (E / E_BREAKDOWN)^2)
    # *Note: In actual materials, atomic breakdown occurs first. This requires
    # an ultra-hard dielectric or high-vacuum gap to isolate the spatial strain.
    
    # Using a smoothed clip to model the plateau right at the edge
    safe_e = np.clip(e_fields, 0, E_BREAKDOWN * 0.999)
    # The geometric plateau factor
    ave_dielectric_factor = np.sqrt(1.0 - (safe_e / E_BREAKDOWN)**2)
    # At the yield limit, the classical lattice "snaps" (loss of structural polarization)
    ave_dielectric_factor[e_fields >= E_BREAKDOWN] = 0.05 # Complete baseline collapse
    
    # Because Refractive Index n ~ sqrt(epsilon), the optical metric also shrinks
    ave_refractive_shift = np.sqrt(ave_dielectric_factor)
    standard_refractive_shift = np.ones_like(e_fields)
    
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

    # Convert x-axis to 10^17 for readable plotting
    plot_e_fields = e_fields / 1e17
    plot_breakdown = E_BREAKDOWN / 1e17

    # Panel 1: Capacitance Measurement (LCR Meter Shift)
    ax1.plot(plot_e_fields, standard_capacitance_ratio, color='#ff3333', linestyle='--', lw=2, label='Standard EM (Linear Permittivity)')
    ax1.plot(plot_e_fields, ave_dielectric_factor, color='#00ffcc', lw=3, label=r'AVE Non-Linear Plateau ($C_{eff}$)')
    
    ax1.axvline(plot_breakdown, color='white', linestyle=':', lw=2, label=r'$E_{yield} \approx 1.13 \times 10^{17}$ V/m')
    
    # Annotate the measurable "Danger Zone" before arc where the capacitance clearly deviates
    ax1.axvspan(plot_breakdown * 0.85, plot_breakdown, color='#ffff99', alpha=0.2, label='LCR Detectable Anomaly Window')

    ax1.set_xlim([0, plot_breakdown * 1.05])
    ax1.set_ylim([0, 1.1])
    ax1.set_title("Benchtop LCR Sensor: Normalized Capacitance vs E-Field")
    ax1.set_xlabel(r"Applied Macroscopic E-Field ($\times 10^{17}$ V/m)")
    ax1.set_ylabel(r"Effective Capacitance Ratio ($C_{meas} / C_0$)")
    ax1.legend(loc='lower left')

    # Panel 2: Optical Interferometry (Refractive Index Shift)
    ax2.plot(plot_e_fields, standard_refractive_shift, color='#ff3333', linestyle='--', lw=2, label='Standard GR (Flat Optical Metric)')
    ax2.plot(plot_e_fields, ave_refractive_shift, color='#ffcc00', lw=3, label=r'AVE Optical Saturation ($\Delta n_{eff}$)')
    
    ax2.axvline(plot_breakdown, color='white', linestyle=':', lw=2, label='Absolute Yield Limit')
    ax2.axvspan(plot_breakdown * 0.85, plot_breakdown, color='#ff99ff', alpha=0.2, label='Laser Phase Shift Window')

    ax2.set_xlim([0, plot_breakdown * 1.05])
    ax2.set_ylim([0, 1.1])
    ax2.set_title("Interferometer Bench: Optical Refractive Index vs E-Field")
    ax2.set_xlabel(r"Applied Macroscopic E-Field ($\times 10^{17}$ V/m)")
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
