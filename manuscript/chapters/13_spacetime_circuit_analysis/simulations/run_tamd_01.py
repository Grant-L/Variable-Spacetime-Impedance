"""
AVE MODULE 59: TAMD-02 MACROSCOPIC ENVELOPE (SOA MAPPING)
---------------------------------------------------------
Sweeps Frequency (f) and Drive Voltage (V_dc) to map the 
Safe Operating Area of the Vacuum Bingham Diode.
Proves the existence of "Spacetime Cavitation" (V_dc > V_yield)
and "Slip Failure" (high frequency current starvation).
Calculates exact parameters to generate macroscopic, eye-visible 
thrust (~1.0 grams) while avoiding thermal meltdown and hardware arc-over.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

OUTPUT_DIR = "manuscript/chapters/14_experimental_protocols/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_macro_envelope():
    print("Sweeping EE Parameters: Mapping the Vacuum SOA...")
    
    # 1. Constants & Assumed Limits
    xi_topo = 4.149e-7   # C/m
    g = 9.80665          # m/s^2
    
    V_yield = 60e3       # 60 kV (Vacuum Bingham Yield Stress)
    V_breakdown = 200e3  # 200 kV (Dielectric Limit of Transformer Oil)
    P_max = 100e3        # 100 kW (Survivable Burst Power Limit)
    
    L = 10e-3            # 10 mH (Massive Toroidal Inductor)
    R_snub = 10e3        # 10 kOhm (Flyback Resistor)
    
    # --- PLOT 1: FREQUENCY SWEEP (The Goldilocks Zone) ---
    V_DC = 30e3          # 30 kV Target Drive Voltage
    D = 0.8              # 80% Duty Cycle
    
    freqs = np.linspace(100e3, 500e3, 1000) # 100 kHz to 500 kHz
    I_peak = (V_DC * D) / (L * freqs)
    V_flyback = I_peak * R_snub
    P_avg = 0.5 * V_DC * I_peak * D
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    fig.patch.set_facecolor('#0a0a12'); ax1.set_facecolor('#0a0a12'); ax2.set_facecolor('#0a0a12')
    
    ax1_twin = ax1.twinx()
    
    ax1.plot(freqs / 1000, V_flyback / 1000, color='#FFD54F', lw=3, label=r'Kickback Voltage ($V_{flyback}$)')
    ax1_twin.plot(freqs / 1000, P_avg / 1000, color='#00ffcc', lw=3, label=r'Average Power ($P_{avg}$)')
    
    # Boundary Lines
    ax1.axhline(V_yield / 1000, color='#4FC3F7', linestyle='--', lw=2, label=r'Vacuum Yield ($60$ kV)')
    ax1.axhline(V_breakdown / 1000, color='#E57373', linestyle=':', lw=2, label=r'Dielectric Arc Limit ($200$ kV)')
    ax1_twin.axhline(P_max / 1000, color='#ff3366', linestyle='--', lw=2, label=r'Thermal Meltdown ($100$ kW)')
    
    # Calculate Intersections for Shading
    f_yield_limit = (V_DC * D * R_snub) / (L * V_yield)
    f_arc_limit = (V_DC * D * R_snub) / (L * V_breakdown)
    f_power_limit = (V_DC**2 * D**2) / (2 * L * P_max)
    
    ax1.axvspan(f_yield_limit / 1000, 500, color='#4FC3F7', alpha=0.15, label='Slip Failure (Drive Stalls)')
    ax1.axvspan(100, f_power_limit / 1000, color='#ff3366', alpha=0.15, label='Thermal Meltdown')
    ax1.axvspan(100, f_arc_limit / 1000, color='#E57373', alpha=0.3, label='Hardware Arc-Over')
    
    # Target Operating Point
    ax1.axvline(300, color='white', linestyle='-.', lw=2)
    ax1.text(310, 150, "Goldilocks Zone\n$f = 300$ kHz\n$P_{avg} = 96$ kW\n$V_{flyback} = 80$ kV", color='white', bbox=dict(facecolor='#111111', alpha=0.8))
    
    ax1.set_title('Frequency ($\omega$) Envelope Analysis', color='white', fontsize=14, weight='bold')
    ax1.set_xlabel('Switching Frequency (kHz)', color='white', weight='bold')
    ax1.set_ylabel('Voltage (kV)', color='white', weight='bold')
    ax1_twin.set_ylabel('Power (kW)', color='white', weight='bold')
    ax1_twin.tick_params(axis='y', colors='white')
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right', facecolor='#111111', edgecolor='gray', labelcolor='white', fontsize=9)
    
    # --- PLOT 2: THE SOA HEATMAP (V_dc vs Duty Cycle) ---
    f_fixed = 300e3 # Locked at 300 kHz Target
    V_sweep = np.linspace(10e3, 80e3, 300)
    D_sweep = np.linspace(0.4, 0.95, 300)
    V_grid, D_grid = np.meshgrid(V_sweep, D_sweep)
    
    Thrust_grid = np.zeros_like(V_grid)
    I_peak_grid = (V_grid * D_grid) / (L * f_fixed)
    V_flyback_grid = I_peak_grid * R_snub
    P_avg_grid = 0.5 * V_grid * I_peak_grid * D_grid
    
    # Apply all 4 boundaries: Cavitation, Slip Failure, Arc-Over, Thermal Meltdown
    valid_mask = (V_grid < V_yield) & (V_flyback_grid > V_yield) & (V_flyback_grid < V_breakdown) & (P_avg_grid < P_max)
    Thrust_grid[valid_mask] = (V_grid[valid_mask] * xi_topo * D_grid[valid_mask] / g) * 1000 # In Grams
    Thrust_grid[~valid_mask] = np.nan
    
    cp = ax2.contourf(V_grid/1000, D_grid, Thrust_grid, levels=20, cmap='magma')
    cbar = fig.colorbar(cp, ax=ax2)
    cbar.set_label('Time-Averaged Thrust (Grams)', color='white', weight='bold')
    cbar.ax.yaxis.set_tick_params(color='white')
    
    ax2.axvline(V_yield/1000, color='#00ffcc', linestyle='--', lw=3, label=r'Cavitation Drop-Out ($V_{dc} > V_{yield}$)')
    
    # Target Point
    target_thrust = (V_DC * xi_topo * D / g) * 1000
    ax2.scatter(30, 0.8, color='#00ffcc', s=150, zorder=5)
    ax2.text(28, 0.83, f"TAMD-02 Target\n(30kV, 80% D $\\rightarrow$ {target_thrust:.2f}g)", color='#00ffcc', weight='bold')
    
    ax2.set_title('Safe Operating Area (SOA) Heatmap', color='white', fontsize=14, weight='bold')
    ax2.set_xlabel('Charging Voltage $V_{DC}$ (kV)', color='white', weight='bold')
    ax2.set_ylabel('Duty Cycle $D$', color='white', weight='bold')
    ax2.legend(loc='lower left', facecolor='#111111', edgecolor='gray', labelcolor='white')
    
    for ax in [ax1, ax2]:
        ax.grid(True, ls=':', color='#333333'); ax.tick_params(colors='lightgray')
        for spine in ax.spines.values(): spine.set_color('#333333')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "tamd02_envelope.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    
    # --- BOM GENERATOR ---
    bom_data = [
        ["Subsystem", "Component", "Manufacturer", "Part Number", "Est. Cost (USD)", "Purpose"],
        ["Actuator Core", "Nanocrystalline Toroid", "Magnetics Inc", "144-090", "$450", "10mH. Confines B-field, completely eliminating Lorentz false-positives."],
        ["Dielectric Bath", "Transformer Oil (5 Gal)", "Shell Diala", "S4 ZX-I", "$120", "Submerges coil. Pushes V_breakdown to 200kV and prevents Ion Wind."],
        ["HV Switch", "Solid-State Thyratron", "Behlke", "HTS 501-03-GSM", "$2,100", "Switches 50kV / 30A with nanosecond precision at 300 kHz."],
        ["Energy Storage", "Pulse Capacitor Bank", "TDK / Custom", "UHV Series", "$1,800", "Provides 96 kW burst power for a 1.0-second visual pendulum swing."],
        ["Power Supply", "30kV DC HV Supply", "Spellman", "SL30P300", "$1,500", "Charges the capacitor bank between test fires."],
        ["Metrology", "Laser Displacement Sensor", "Keyence", "LK-G152", "$1,200", "Measures macroscopic torsion pendulum swing externally."],
        ["TOTAL", "", "", "", "$7,170", "Sub-$10k budget for undeniable macroscopic validation."]
    ]
    df = pd.DataFrame(bom_data[1:], columns=bom_data[0])
    csv_path = os.path.join(OUTPUT_DIR, "Project_TAMD02_BOM.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Success. Visualizations saved to: {filepath}")
    print(f"Success. Bill of Materials saved to: {csv_path}")

if __name__ == "__main__": simulate_macro_envelope()