#!/usr/bin/env python3
r"""
PONDER-01: Open Air Convective Thermal Limits
=============================================

This script models the thermal accumulation of the PONDER-01 phased array
in an open-air environment (1 ATM, 20 C) instead of a hard vacuum.

In the vacuum simulator, the substrate suffered catastrophic delamination 
within ~2.0 seconds because $100\%$ of the massive reactive $V^2 \\omega C \\tan(\\delta)$ 
heat had nowhere to go except radiative emission.

In standard pressure, Newton's Law of Cooling (convection) significantly alters 
the steady-state temperature limit. This script plots the new runtime allowable
before reaching the critical $130^{\\circ}\\text{C}$ FR-4 delamination limit.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def simulate_open_air_thermal():
    print("[*] Generating Open-Air Convective Thermal Model...")
    
    # Time Domain (0 to 600 Seconds / 10 Minutes)
    time_steps = np.linspace(0, 600, 2000)
    dt = time_steps[1] - time_steps[0]
    
    # ---------------------------------------------------------
    # Electrical Heating Parameters (from previous sims)
    # Note: Because Paschen limits open air to ~0.5 kV (down from 30 kV),
    # the sheer power dumped into the PCBA drops by V^2 = (1/60)^2
    # For this simulation, we'll assume we push it right to 3.0 kV peak 
    # to see a worst-case RF heating scenario before arcing overrides it.
    # ---------------------------------------------------------
    V_RMS = 3000.0 # Volts
    FREQ = 100.0e6 # Hz
    C_MUTUAL = 100.0e-12 # 100 pF
    TAN_DELTA_FR4 = 0.02 # High frequency loss tangent for simple FR4
    
    # P_heat = V^2 * omega * C * tan(delta)
    OMEGA = 2.0 * np.pi * FREQ
    power_dielectric_FR4 = (V_RMS**2) * OMEGA * C_MUTUAL * TAN_DELTA_FR4
    
    print(f"[*] Assumed Forced P_loss into FR-4 Substrate: {power_dielectric_FR4:.1f} W")
    
    # ---------------------------------------------------------
    # Thermal Mass and Convective Properties
    # ---------------------------------------------------------
    # PCBA Volume Approx (10cm x 10cm x 1.6mm)
    mass_pcb = 0.05 # 50 grams
    cp_fr4 = 1100.0 # Specific heat J/(kg*K)
    thermal_mass = mass_pcb * cp_fr4
    
    surface_area = 2.0 * (0.1 * 0.1) # 2 sides of a 10cm square
    
    # Convection coefficient (h) for free natural convection in air is roughly 10 W/(m^2*K)
    # Forced convection (via a standard muffin fan) can push it to 40 W/(m^2*K)
    h_free = 10.0
    h_forced = 40.0
    
    T_ambient = 20.0 # Celsius
    T_FAIL_FR4 = 130.0 # TG limit
    
    # ---------------------------------------------------------
    # Simulation Loop
    # ---------------------------------------------------------
    temp_vac = np.ones_like(time_steps) * T_ambient
    temp_air_free = np.ones_like(time_steps) * T_ambient
    temp_air_forced = np.ones_like(time_steps) * T_ambient
    
    for i in range(1, len(time_steps)):
        
        # 1. Vacuum (No Convection, trivial radiation ignored for short times)
        dT_vac = (power_dielectric_FR4 / thermal_mass) * dt
        temp_vac[i] = temp_vac[i-1] + dT_vac
        
        # 2. Open Air (Free Convection)
        q_conv_free = h_free * surface_area * (temp_air_free[i-1] - T_ambient)
        dT_free = ((power_dielectric_FR4 - q_conv_free) / thermal_mass) * dt
        temp_air_free[i] = temp_air_free[i-1] + dT_free
        
        # 3. Open Air (Forced Fan Convection)
        q_conv_forced = h_forced * surface_area * (temp_air_forced[i-1] - T_ambient)
        dT_forced = ((power_dielectric_FR4 - q_conv_forced) / thermal_mass) * dt
        temp_air_forced[i] = temp_air_forced[i-1] + dT_forced

    # -------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(time_steps, temp_vac, label='Hard Vacuum (Radiative Only)', color='purple', linewidth=3, linestyle='-.')
    ax.plot(time_steps, temp_air_free, label='Open Air (Free Convection, $h=10$)', color='orange', linewidth=3)
    ax.plot(time_steps, temp_air_forced, label='Forced Air (Cooling Fan, $h=40$)', color='cyan', linewidth=3)
    
    # Delamination limit
    ax.axhline(y=T_FAIL_FR4, color='red', linestyle='--', linewidth=2, label='FR-4 Delamination ($T_{g}=130^{\\circ}\\text{C}$)')
    
    # Find interception times
    idx_vac = np.argmax(temp_vac >= T_FAIL_FR4) if np.any(temp_vac >= T_FAIL_FR4) else -1
    if idx_vac != -1:
        t_vac = time_steps[idx_vac]
        ax.scatter([t_vac], [T_FAIL_FR4], color='red', zorder=5)
        ax.annotate(f"{t_vac:.1f} s", xy=(t_vac, T_FAIL_FR4), xytext=(t_vac + 10, T_FAIL_FR4 + 10))

    idx_free = np.argmax(temp_air_free >= T_FAIL_FR4) if np.any(temp_air_free >= T_FAIL_FR4) else -1
    if idx_free != -1:
        t_free = time_steps[idx_free]
        ax.scatter([t_free], [T_FAIL_FR4], color='red', zorder=5)
        ax.annotate(f"{t_free:.1f} s", xy=(t_free, T_FAIL_FR4), xytext=(t_free + 10, T_FAIL_FR4 + 10))
        
    idx_forced = np.argmax(temp_air_forced >= T_FAIL_FR4) if np.any(temp_air_forced >= T_FAIL_FR4) else -1
    if idx_forced == -1:
        steady_state = temp_air_forced[-1]
        ax.annotate(f"Steady State: {steady_state:.1f}$^{{\\circ}}$C\nConstant Testing Allowed", xy=(time_steps[-1], steady_state), xytext=(time_steps[-1]-150, steady_state + 20), arrowprops={'facecolor': 'black', 'shrink': 0.05, 'width': 1.5, 'headwidth': 6})
        
    ax.set_title("PONDER-01: Open Air Convective Thermal Limits\nFR-4 Substrate Heating @ 3.0 kV RMS / 100 MHz", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Continuous Active RF Runtime (Seconds)", fontsize=12)
    ax.set_ylabel(r"Substrate Core Temperature ($^{\circ}$C)", fontsize=12)
    
    ax.set_ylim([0, 250])
    ax.set_xlim([0, 600])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Export
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ponder_01_open_air_thermal.png')
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Simulation complete. Output saved to: {out_path}")

if __name__ == "__main__":
    simulate_open_air_thermal()
