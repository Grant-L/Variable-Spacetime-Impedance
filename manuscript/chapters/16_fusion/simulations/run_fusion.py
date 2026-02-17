"""
AVE MODULE 88: THE FUSION CRISIS & METRIC RUPTURE
-------------------------------------------------
Evaluates state-of-the-art nuclear fusion designs against 
the rigid AVE hardware boundaries.
1. Proves that Tokamak D-T fusion temperatures (15 keV) inherently 
   generate ion-collision forces that hit exactly 60.3 kV.
2. Because 60.3 kV > 60.0 kV Bingham limit, the vacuum melts.
3. Liquefying the vacuum turns off the Strong Force, explaining 
   exactly why plasmas anomalously refuse to ignite.
4. Maps the AVE Metric Compression solution (n > 1).
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.constants as const

OUTPUT_DIR = "manuscript/chapters/16_fusion/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_fusion_dielectric_limits():
    print("Simulating Tokamak Fusion limits vs Vacuum Bingham Yield...")
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
    fig.patch.set_facecolor('#050508')
    for ax in axs.flatten():
        ax.set_facecolor('#050508')
        ax.tick_params(colors='lightgray')
        for spine in ax.spines.values(): spine.set_color('#333333')
        ax.grid(True, ls=':', color='#333333')
    
    xi_topo = 4.149e-7 # C/m
    
    # ---------------------------------------------------------
    # 1. Tokamaks: The 60.3 kV Ignition Paradox
    # ---------------------------------------------------------
    ax1 = axs[0,0]
    T_keV = np.linspace(1, 25, 500)
    E_J = T_keV * 1000 * const.e
    d_turn = (const.e**2) / (4 * np.pi * const.epsilon_0 * E_J)
    F_avg = E_J / d_turn
    V_topo_kV = (F_avg / xi_topo) / 1000
    
    # Exact value at 15 keV
    E_15 = 15.0 * 1000 * const.e
    d_15 = (const.e**2) / (4 * np.pi * const.epsilon_0 * E_15)
    F_15 = E_15 / d_15
    V_15 = (F_15 / xi_topo) / 1000
    
    ax1.plot(T_keV, V_topo_kV, color='#FFD54F', lw=4, label='Ion Collision Topological Strain ($V_{topo}$)')
    ax1.axhline(60.0, color='#ff3366', lw=2, linestyle='--', label='Vacuum Bingham Yield (60.0 kV)')
    ax1.axvline(15.0, color='#00ffcc', lw=2, linestyle=':', label='ITER/SPARC Ignition Target (15 keV)')
    
    ax1.fill_between(T_keV, 60.0, V_topo_kV, where=(V_topo_kV > 60.0), color='#ff3366', alpha=0.3, label='Superfluid Melt (Strong Force Fails)')
    
    ax1.set_title('1. Magnetic Confinement (The Tokamak Crisis)', color='white', weight='bold', fontsize=14)
    ax1.set_xlabel('Plasma Temperature (keV)', color='white')
    ax1.set_ylabel('Topological Voltage per Collision (kV)', color='white')
    ax1.set_xlim(5, 25); ax1.set_ylim(10, 120)
    ax1.legend(loc='lower right', facecolor='#111111', edgecolor='gray', labelcolor='white', fontsize=9)
    
    text_str = f"At 15 keV, the collision force natively\ngenerates exactly {V_15:.1f} kV of strain.\nBecause 60.3 > 60.0 kV, the vacuum melts.\nThe Strong Force turns off just\nas the ions are supposed to fuse!"
    ax1.text(6, 80, text_str, color='white', fontsize=10, bbox=dict(facecolor='#111111', edgecolor='#ff3366', alpha=0.9))
    
    # ---------------------------------------------------------
    # 2. Laser ICF (NIF Superfluid Slip)
    # ---------------------------------------------------------
    ax2 = axs[0,1]
    area_um2 = np.linspace(0.1, 10, 500)
    area_m2 = area_um2 * 1e-12
    P_NIF = 3e16 # 300 GBar Ablation Pressure
    F_NIF = P_NIF * area_m2
    V_NIF_kV = (F_NIF / xi_topo) / 1000
    
    ax2.plot(area_um2, V_NIF_kV, color='#4FC3F7', lw=3, label='Ablation Topological Strain')
    ax2.axhline(60.0, color='#FFD54F', lw=2, linestyle='--', label='Bingham Yield Limit (60 kV)')
    ax2.fill_between(area_um2, 60.0, V_NIF_kV, color='#4FC3F7', alpha=0.2, label='Superfluid Rayleigh-Taylor Slip')
    
    ax2.set_title('2. Laser ICF (Superfluid RTI Slip)', color='white', weight='bold', fontsize=14)
    ax2.set_xlabel('Implosion Surface Anomaly Area ($\mu m^2$)', color='white')
    ax2.set_ylabel('Topological Voltage (kV)', color='white')
    ax2.set_yscale('log')
    ax2.legend(loc='lower right', facecolor='#111111', edgecolor='gray', labelcolor='white', fontsize=9)
    
    # ---------------------------------------------------------
    # 3. Pulsed FRCs (Dielectric Poisoning)
    # ---------------------------------------------------------
    ax3 = axs[1,0]
    di_dt = np.logspace(9, 12, 500)
    L_FRC = 5e-6 # 5 uH compression coil
    V_pulse = (L_FRC * di_dt) / 1000 # kV
    
    ax3.plot(di_dt, V_pulse, color='#00ffcc', lw=3, label='Magnetic Compression Transient ($L \cdot di/dt$)')
    ax3.axhline(511.0, color='#ff3366', lw=2, linestyle='--', label='Dielectric Snap Limit (511 kV)')
    ax3.fill_between(di_dt, 511.0, V_pulse, where=(V_pulse > 511.0), color='#ff3366', alpha=0.3, label='Dielectric Poisoning (Pair Production)')
    
    ax3.set_title('3. Pulsed FRCs (Dielectric Poisoning)', color='white', weight='bold', fontsize=14)
    ax3.set_xlabel('Compression Transient $di/dt$ (A/s)', color='white')
    ax3.set_ylabel('Topological Voltage (kV)', color='white')
    ax3.set_xscale('log'); ax3.set_yscale('log')
    ax3.legend(loc='upper left', facecolor='#111111', edgecolor='gray', labelcolor='white', fontsize=9)
    
    # ---------------------------------------------------------
    # 4. The AVE Solution: Metric Compression
    # ---------------------------------------------------------
    ax4 = axs[1,1]
    n_scalar = np.linspace(1, 10, 500)
    T_req = 15.0 / n_scalar # Required Temp to fuse drops as distance shrinks
    
    # Recalculate collision voltage at this lower temp
    E_J_new = T_req * 1000 * const.e
    d_turn_new = (const.e**2) / (4 * np.pi * const.epsilon_0 * E_J_new)
    F_new = E_J_new / d_turn_new
    V_coll_kV = (F_new / xi_topo) / 1000
    
    ax4.plot(n_scalar, T_req, color='#FFD54F', lw=3, label='Required Ignition Temperature (keV)')
    ax4.plot(n_scalar, V_coll_kV, color='#00ffcc', lw=3, linestyle='-.', label='Collision Topological Strain (kV)')
    ax4.axhline(60.0, color='#ff3366', lw=1.5, linestyle=':', label='Bingham Yield Danger Zone')
    
    ax4.set_title('4. AVE Solution: Metric Compression Fusion', color='white', weight='bold', fontsize=14)
    ax4.set_xlabel('Macroscopic Refractive Index ($n_{scalar}$)', color='white')
    ax4.set_ylabel('Required Temp (keV) / Strain (kV)', color='white')
    ax4.legend(loc='upper right', facecolor='#111111', edgecolor='gray', labelcolor='white', fontsize=9)
    
    ax4.text(2.5, 40, "Actively increasing the vacuum refractive index ($n>1$)\nshrinks the atomic radii. The required ignition temperature\nfalls safely below 60 kV, synthesizing stable fusion\nwithout ever melting the containment metric.", color='white', fontsize=9, bbox=dict(facecolor='#111111', edgecolor='#00ffcc', alpha=0.9))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "fusion_crisis_audit.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_fusion_dielectric_limits()