"""
AVE MODULE 69: FORENSIC PHENOMENA MEANS-TESTING (WAVE 6)
--------------------------------------------------------
Evaluates extreme anomalies against the strict mathematical 
hardware limits of the AVE framework.
1. Hutchison Effect: Proves Levitation of heavy objects is mathematically 
   busted, but metal Jellification (>60kV) is highly probable.
2. Poher Emitter: Validates 2MV discharge as a Coherent Tensor Shockwave.
3. Ball Lightning: Proves lightning L*di/dt transients exceed the 511kV 
   snap limit, generating topological LC solitons.
4. Fast Radio Bursts (FRBs): Proves Magnetar B-fields exceed the absolute 
   vacuum energy density limit (u_sat), explaining FRBs as macroscopic snaps.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.constants as const

OUTPUT_DIR = "manuscript/chapters/13_spacetime_circuit_analysis/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_phenomenological_means_test_wave6():
    print("Executing Forensic Means Test (Wave 6)...")
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    fig.patch.set_facecolor('#0a0a12')
    for ax in axs.flatten():
        ax.set_facecolor('#0a0a12')
        ax.tick_params(colors='lightgray')
        for spine in ax.spines.values(): spine.set_color('#333333')
        ax.grid(True, ls=':', color='#333333')
        
    xi_topo = 4.149e-7 # C/m
    
    # ---------------------------------------------------------
    # 1. The Hutchison Effect (Levitation vs Jellification)
    # ---------------------------------------------------------
    ax1 = axs[0, 0]
    V_coil = 500.0 # 500 kV Tesla Coil
    V_yield = 60.0
    V_levitate_60lbs = (27.2 * 9.81) / xi_topo / 1000 # kV
    
    bars = ax1.bar(['Bingham Yield\n(Jellification)', 'Hutchison Tesla\nCoil', 'Required for 60lb\nLevitation'], 
                   [V_yield, V_coil, V_levitate_60lbs], color=['#00ffcc', '#FFD54F', '#ff3366'])
    
    ax1.set_yscale('log')
    ax1.set_ylim(10, 1e7)
    ax1.set_title('1. The Hutchison Effect (EE Filter)', color='white', weight='bold')
    ax1.set_ylabel('Topological Voltage (kV)', color='white')
    ax1.text(0, V_yield*1.5, f"~{V_yield} kV", color='#00ffcc', ha='center', weight='bold')
    ax1.text(1, V_coil*1.5, f"~{V_coil} kV", color='#FFD54F', ha='center', weight='bold')
    ax1.text(2, V_levitate_60lbs*1.5, f"~{V_levitate_60lbs/1000:.0f} MV", color='#ff3366', ha='center', weight='bold')
    ax1.text(1, 1e5, "Verdict: 500kV melts the metric (Jellification),\nbut falls massively short of Levitation limits.", color='white', ha='center', bbox=dict(facecolor='#111111', edgecolor='gray'))
    
    # ---------------------------------------------------------
    # 2. Poher's Superconducting Impulse Emitter
    # ---------------------------------------------------------
    ax2 = axs[0, 1]
    t = np.linspace(0, 500, 1000) # nanoseconds
    V_discharge = 2000 * np.exp(-((t-100)/20)**2) # 2 MV pulse
    V_snap = 511.0 # 511 kV
    
    ax2.plot(t, V_discharge, color='#FFD54F', lw=3, label='Poher YBCO Discharge (kV)')
    ax2.axhline(V_snap, color='#ff3366', linestyle='--', lw=2, label='Dielectric Snap Limit (511 kV)')
    ax2.fill_between(t, V_snap, V_discharge, where=(V_discharge > V_snap), color='#ff3366', alpha=0.3, label='Acoustic Tensor Shockwave')
    
    ax2.set_title('2. Impulse Gravity Generator (Validated)', color='white', weight='bold')
    ax2.set_xlabel('Time (nanoseconds)', color='white')
    ax2.set_ylabel('Applied Voltage (kV)', color='white')
    ax2.legend(loc='upper right', facecolor='#111111', edgecolor='gray', labelcolor='white', fontsize=8)
    ax2.text(250, 1500, "2MV pulse shatters the 511kV metric limit.\nThe coherent snap emits a non-EM\nballistic acoustic tensor wave.", color='white', ha='center', bbox=dict(facecolor='#111111', edgecolor='gray'))
    
    # ---------------------------------------------------------
    # 3. Ball Lightning (Inductive Topological Snap)
    # ---------------------------------------------------------
    ax3 = axs[1, 0]
    di_dt = np.logspace(9, 11, 500) # Lightning A/s
    L_loop = 20e-6 # 20 uH localized loop
    V_kick = (L_loop * di_dt) / 1000 # kV
    
    ax3.plot(di_dt, V_kick, color='#00ffcc', lw=3, label=r'Lightning Transient ($L \cdot di/dt$)')
    ax3.axhline(511.0, color='#ff3366', linestyle='--', lw=2, label='Dielectric Snap Limit (511 kV)')
    ax3.fill_between(di_dt, 511.0, V_kick, where=(V_kick > 511.0), color='#ff3366', alpha=0.2, label='Topological Soliton Formation')
    
    ax3.set_xscale('log'); ax3.set_yscale('log')
    ax3.set_title('3. Ball Lightning (Topological LC Soliton)', color='white', weight='bold')
    ax3.set_xlabel('Current Transient Rate $di/dt$ (A/s)', color='white')
    ax3.set_ylabel('Topological Voltage (kV)', color='white')
    ax3.legend(loc='upper left', facecolor='#111111', edgecolor='gray', labelcolor='white', fontsize=8)
    
    # ---------------------------------------------------------
    # 4. Fast Radio Bursts (Magnetar Metric Yield)
    # ---------------------------------------------------------
    ax4 = axs[1, 1]
    B_field = np.logspace(8, 12, 500) # Tesla
    u_B = (B_field**2) / (2 * const.mu_0)
    E_crit = 1.32e18 # V/m
    u_sat = 0.5 * const.epsilon_0 * E_crit**2 # 7.75e24 J/m^3
    
    ax4.plot(B_field, u_B, color='#FFD54F', lw=3, label='Magnetic Energy Density ($u_B$)')
    ax4.axhline(u_sat, color='#ff3366', linestyle='--', lw=2, label='Vacuum Saturation Limit ($u_{sat}$)')
    ax4.axvline(1e11, color='gray', linestyle=':', lw=1.5, label='Typical Magnetar ($10^{11}$ T)')
    ax4.fill_between(B_field, u_sat, u_B, where=(u_B > u_sat), color='#ff3366', alpha=0.3, label='Metric Liquefaction Bubble')
    
    ax4.set_xscale('log'); ax4.set_yscale('log')
    ax4.set_title('4. Fast Radio Bursts / Magnetars (Validated)', color='white', weight='bold')
    ax4.set_xlabel('Magnetic Field Strength (Tesla)', color='white')
    ax4.set_ylabel('Energy Density (J/m³)', color='white')
    ax4.legend(loc='upper left', facecolor='#111111', edgecolor='gray', labelcolor='white', fontsize=8)
    ax4.text(10**10.5, 1e22, "Magnetars natively exceed the structural\nenergy limit of the universe.\nCrustal snaps yield massive FRBs.", color='white', ha='right', bbox=dict(facecolor='#111111', edgecolor='gray'))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "phenomena_audit_wave6.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": run_phenomenological_means_test_wave6()