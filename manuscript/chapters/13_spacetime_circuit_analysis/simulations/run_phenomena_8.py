"""
AVE MODULE 70: FORENSIC PHENOMENA MEANS-TESTING (WAVE 8 - SANITY CHECK)
-----------------------------------------------------------------------
A strict sanity check of the AVE framework against highly documented 
physics anomalies, proving the framework does not "wildly speculate."
1. Optical Tweezers/Acoustics: Busted as metric effects (<2.4 kV << 60 kV).
2. High-Tc Superconductors: Busted as metric effects (3.6 mV).
3. W-Boson Mass Anomaly (CDF II): Validates the parameter-free 2/7 Poisson limit.
4. Terrestrial Gamma-Ray Flashes: Validated (1000 kV > 511 kV).
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/13_spacetime_circuit_analysis/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_sanity_check_means_test():
    print("Executing Forensic Means Test (Wave 8 - The Sanity Check)...")
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    fig.patch.set_facecolor('#0a0a12')
    for ax in axs.flatten():
        ax.set_facecolor('#0a0a12')
        ax.tick_params(colors='lightgray')
        for spine in ax.spines.values(): spine.set_color('#333333')
        ax.grid(True, ls=':', color='#333333')
        
    xi_topo = 4.149e-7 # C/m
    
    # ---------------------------------------------------------
    # 1. The Sanity Check: Optical Tweezers / Acoustic Levitation
    # ---------------------------------------------------------
    ax1 = axs[0, 0]
    V_optical = 0.016 # 16 mV (1-Watt Laser)
    V_acoustic = 2360.0 # 2.36 kV (0.1g Water droplet)
    V_yield = 60000.0 # 60 kV
    
    bars = ax1.bar(['Optical Tweezers\n(1-W Laser)', 'Acoustic Levitation\n(0.1g Droplet)', 'Vacuum Yield\nLimit'], 
                   [V_optical, V_acoustic, V_yield], color=['#4FC3F7', '#FFD54F', '#ff3366'])
    
    ax1.set_yscale('log')
    ax1.set_ylim(1e-3, 1e7)
    ax1.set_title('1. Tabletop Levitation (The Scaling Sanity Check)', color='white', weight='bold')
    ax1.set_ylabel('Topological Voltage (V)', color='white')
    ax1.text(0, V_optical*2, f"{V_optical*1000:.1f} mV", color='#4FC3F7', ha='center', weight='bold')
    ax1.text(1, V_acoustic*2, f"{V_acoustic/1000:.2f} kV", color='#FFD54F', ha='center', weight='bold')
    ax1.text(2, V_yield*2, f"{V_yield/1000:.0f} kV", color='#ff3366', ha='center', weight='bold')
    ax1.text(1, 1e6, "Verdict: Both fall safely below 60 kV.\nTabletop levitation is 100% classical physics.\nZero metric or gravity manipulation occurs.", color='white', ha='center', bbox=dict(facecolor='#111111', edgecolor='gray'))
    
    # ---------------------------------------------------------
    # 2. High-Tc Superconductors (Diamond Anvils)
    # ---------------------------------------------------------
    ax2 = axs[0, 1]
    pressure_gpa = np.logspace(0, 3, 500) # 1 to 1000 GPa
    area_unit_cell = 1e-20 # ~1 Angstrom squared
    force_N = (pressure_gpa * 1e9) * area_unit_cell
    V_topo = force_N / xi_topo
    
    ax2.plot(pressure_gpa, V_topo, color='#00ffcc', lw=3, label='Topological Voltage per Unit Cell')
    ax2.axhline(60e3, color='#ff3366', linestyle='--', lw=2, label='Vacuum Yield Limit (60,000 V)')
    ax2.axvline(150, color='gray', linestyle=':', lw=1.5, label='H2S Superconducting Pressure (150 GPa)')
    
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_title('2. High-Tc Pressure Superconductivity (Rejected)', color='white', weight='bold')
    ax2.set_xlabel('Mechanical Pressure (GPa)', color='white')
    ax2.set_ylabel('Induced Topological Voltage (V)', color='white')
    ax2.legend(loc='upper left', facecolor='#111111', edgecolor='gray', labelcolor='white', fontsize=9)
    ax2.text(2, 1e-1, r"At 150 GPa, $V_{topo} \sim 3.6$ mV." + "\n" + "AVE rigorously rejects metric explanations\nfor standard BCS phonon coupling.", color='white', ha='left', bbox=dict(facecolor='#111111', edgecolor='gray'))
    
    # ---------------------------------------------------------
    # 3. The W-Boson Mass Anomaly (CDF II 2022)
    # ---------------------------------------------------------
    ax3 = axs[1, 0]
    m_Z = 91.1876 # GeV
    m_W_SM = 80.357
    m_W_CDF = 80.433
    # AVE Zero-Parameter Prediction: m_W = m_Z * sqrt(7)/3
    m_W_AVE = m_Z * (np.sqrt(7) / 3)
    
    bars3 = ax3.bar(['Standard\nModel', 'AVE Prediction\n(2/7 Poisson)', 'CDF II\nCollider Data'], 
                   [m_W_SM, m_W_AVE, m_W_CDF], color=['#FFD54F', '#00ffcc', '#ff3366'])
    
    ax3.set_ylim(80.2, 80.5)
    ax3.set_title('3. The W-Boson Mass Anomaly (Fermilab 2022)', color='white', weight='bold')
    ax3.set_ylabel('Mass (GeV)', color='white')
    ax3.text(0, m_W_SM + 0.01, f"{m_W_SM} GeV", color='#FFD54F', ha='center', weight='bold')
    ax3.text(1, m_W_AVE + 0.01, f"{m_W_AVE:.3f} GeV", color='#00ffcc', ha='center', weight='bold')
    ax3.text(2, m_W_CDF + 0.01, f"{m_W_CDF} GeV", color='#ff3366', ha='center', weight='bold')
    ax3.text(1, 80.25, "AVE predicts the 7-sigma anomaly strictly\nfrom the 2/7 Cosserat vacuum trace-reversal.", color='white', ha='center', bbox=dict(facecolor='#111111', edgecolor='gray'))
    
    # ---------------------------------------------------------
    # 4. Terrestrial Gamma-Ray Flashes (TGFs)
    # ---------------------------------------------------------
    ax4 = axs[1, 1]
    di_dt = np.linspace(1e9, 5e11, 500)
    L_lightning = 10e-6 # 10 uH localized channel
    V_kick = (L_lightning * di_dt) / 1000 # kV
    
    ax4.plot(di_dt, V_kick, color='#FFD54F', lw=3, label='Lightning Inductive Transient (kV)')
    ax4.axhline(511.0, color='#ff3366', linestyle='--', lw=2, label='Dielectric Snap Limit (511 kV)')
    ax4.fill_between(di_dt, 511.0, V_kick, where=(V_kick > 511.0), color='#ff3366', alpha=0.3, label='Pair-Production (Positrons & Gamma Rays)')
    
    ax4.set_xscale('log'); ax4.set_yscale('log')
    ax4.set_title('4. Terrestrial Gamma-Ray Flashes (Validated)', color='white', weight='bold')
    ax4.set_xlabel('Current Transient $di/dt$ (A/s)', color='white')
    ax4.set_ylabel('Topological Voltage (kV)', color='white')
    ax4.legend(loc='lower right', facecolor='#111111', edgecolor='gray', labelcolor='white', fontsize=8)
    ax4.text(1.5e10, 2000, "1000 kV > 511 kV.\nThe metric violently snaps, generating\nantimatter entirely without 100 MV static fields.", color='white', bbox=dict(facecolor='#111111', edgecolor='gray'))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "sanity_check_audit.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": run_sanity_check_means_test()