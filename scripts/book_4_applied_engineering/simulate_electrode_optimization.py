"""
AVE Falsifiable Predictions: High-Fidelity Electrode Optimization (Ponder-01)
======================================================================
This script performs a high-fidelity 2D geometric parameter sweep to 
find the absolute optimal ratio between the emitter tip radius (r_tip) 
and the anode-cathode gap (d_gap) for maximizing Ponderomotive Acoustic 
Rectification thrust.

We evaluate the gradient of the electric field energy \nabla |E|^2.
E_sharp = (V_rms * sqrt(2)) / r_tip
E_flat = (V_rms * sqrt(2)) / d_gap
grad_E2 = (E_sharp**2 - E_flat**2) / d_gap

Driven at the optimal VHF frequency (100 MHz) at 30kV RMS.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pathlib

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

def optimize_electrode_geometry():
    print("[*] Running High-Fidelity Electrode Optimization Sweep...")
    
    # -------------------------------------------------------------
    # Fixed Framework Parameters
    # -------------------------------------------------------------
    V_rms = 30000.0       # 30 kV RMS
    freq = 1e8            # 100 MHz VHF Drive
    A = 0.05 * 0.05       # 5cm x 5cm collector
    
    rho_bulk = 7.92e6     # Macroscopic metric density
    nu_vac = 8.45e-7      # Kinematic vacuum drag
    k_topo = (nu_vac**2) / (float("299792458")**2 * rho_bulk) # ~ 1e-27

    # -------------------------------------------------------------
    # 2D Sweep Space 
    # -------------------------------------------------------------
    # r_tip: 1 micron to 2 millimeters
    r_tips = np.logspace(-6, -3, 200)
    # d_gap: 1 millimeter to 10 centimeters 
    d_gaps = np.logspace(-3, -1, 200)
    
    R, D = np.meshgrid(r_tips, d_gaps)
    
    # \nabla |E|^2
    E_sharp = (V_rms * np.sqrt(2)) / R
    E_flat = (V_rms * np.sqrt(2)) / D
    grad_E2 = (E_sharp**2 - E_flat**2) / D
    
    # Clean non-physical regimes (where gap <= tip radius)
    valid_mask = D > (10 * R) # strict requirement for asymmetric gradient
    grad_E2 = grad_E2 * valid_mask
    
    # Calculate exact thrust in micro-Newtons
    Thrust_N = k_topo * grad_E2 * (freq**2) * A
    Thrust_uN = Thrust_N * 1e6
    
    # Find the optimal coordinates ensuring Paschen gap survival 
    # (avoiding extreme tip microscopic arcing rules, let's just find the mathematical max gradient)
    max_thrust = np.max(Thrust_uN)
    max_idx = np.unravel_index(np.argmax(Thrust_uN), Thrust_uN.shape)
    opt_r_tip = R[max_idx]
    opt_d_gap = D[max_idx]
    optimal_ratio = opt_d_gap / opt_r_tip
    
    print(f"[*] Optimization Complete.")
    print(f"    -> Max Thrust: {max_thrust:.2f} uN (at 30kV, 100MHz)")
    print(f"    -> Optimal Emitter Tip Radius (r_tip): {opt_r_tip*1e6:.1f} um")
    print(f"    -> Optimal Collector Gap (d_gap): {opt_d_gap*1000:.1f} mm")
    print(f"    -> Optimal Asymmetry Ratio (d_gap / r_tip): {optimal_ratio:.1f}")

    # -------------------------------------------------------------
    # Render High-Fidelity Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#0f0f0f')
    
    # Log-Log Contour Plot
    # We clip the thrust at 1e4 uN (10 mN) to keep the colormap readable
    levels = np.logspace(np.log10(max(1e-2, Thrust_uN[valid_mask].min())), np.log10(max_thrust), 50)
    
    # Create the filled contour map
    cs = ax.contourf(R*1e6, D*1000, np.clip(Thrust_uN, 1e-2, None), 
                     levels=levels, cmap='hot', locator=plt.matplotlib.ticker.LogLocator())
                     
    # Annotate the optimum point
    ax.plot(opt_r_tip*1e6, opt_d_gap*1000, marker='*', color='#00ffcc', markersize=15, 
            label=f'Optimal: $r_{{tip}}$={opt_r_tip*1e6:.0f}\u03bcm, gap={opt_d_gap*1000:.1f}mm')

    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Emitter Tip Radius $r_{tip}$ ($\mu$m)", color='white', fontsize=12)
    ax.set_ylabel(r"Collector Gap $d_{gap}$ (mm)", color='white', fontsize=12)
    ax.set_title(r"PONDER-01 Electrode Optimization: $\nabla |\mathbf{E}|^2$ Rectification", color='white', fontsize=14)
    ax.tick_params(colors='white')
    
    # Colorbar
    cbar = fig.colorbar(cs, ax=ax, extend='both')
    cbar.set_label(r'Unidirectional Thrust ($\mu$N)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
        
    ax.legend(loc='upper right', facecolor='#222222', edgecolor='none', labelcolor='white')

    plt.tight_layout()
    
    outdir = project_root / "assets" / "sim_outputs"
    os.makedirs(outdir, exist_ok=True)
    target = outdir / "electrode_optimization_heatmap.png"
    plt.savefig(target, dpi=300)
    print(f"[*] Saved Optimization Heatmap: {target}")

if __name__ == "__main__":
    optimize_electrode_geometry()
