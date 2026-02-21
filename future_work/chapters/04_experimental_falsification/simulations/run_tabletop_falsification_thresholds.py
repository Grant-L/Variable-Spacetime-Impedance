"""
AVE MODULE 46: THE TABLETOP FALSIFICATION THRESHOLDS
----------------------------------------------------
Strict mathematical evaluation comparing the RVR (Scalar Parametric Amplifier) 
against the Sagnac-RLVE (Kinematic Optical Interferometer).
Proves that scalar metric shifts (RVR) are suppressed by G/c^2 (~10^-26), 
making them physically undetectable even with infinite Q-factors.
Proves the Sagnac-RLVE bypasses this via first-order kinematics (v/c), 
yielding a massive ~2 Radian signal.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/12_experimental_falsification/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_experimental_thresholds():
    print("Simulating Experimental Thresholds (RVR vs Sagnac)...")
    
    # Constants
    G = 6.674e-11
    c = 299792458.0
    
    # 1. RVR SCALAR MODULATION (Tungsten Lobe, 1kg at 1cm)
    M_lobe = 1.0
    r_coil = 0.01
    chi_vol = (7 * G * M_lobe) / (c**2 * r_coil)
    delta_L = (1.0 / 7.0) * chi_vol # ~7.4e-26
    
    # Parametric Q required: Q * delta_L > 2 => Q = 2/delta_L
    Q_required = 2.0 / delta_L
    Q_max_humanity = 1e11 # Niobium SRF Cavity
    
    # 2. SAGNAC-RLVE KINEMATIC DRAG (Tungsten Rotor, 15cm at 10k RPM)
    rho_vac = 7.9159e6
    rho_W = 19300.0
    v_tan = 10000 * (2 * np.pi / 60) * 0.15
    v_fluid = v_tan * (rho_W / rho_vac)
    
    L_fiber = 200.0
    lambda_laser = 1550e-9
    phase_shift_rads = (4 * np.pi * L_fiber * v_fluid) / (lambda_laser * c)
    
    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor('#050508'); ax1.set_facecolor('#050508'); ax2.set_facecolor('#050508')
    
    # Left: The RVR Q-Factor Crisis
    bars1 = ax1.bar(['Max Human Q-Factor\n(Superconducting)', 'RVR Required Q-Factor\n(Parametric Threshold)'], 
                    [Q_max_humanity, Q_required], color=['#ff3366', '#444444'])
    ax1.set_yscale('log')
    ax1.set_ylabel('Quality Factor (Q)', color='white', fontsize=12, weight='bold')
    ax1.set_title('RVR: The Scalar Gap ($G/c^2$)', color='white', fontsize=14, weight='bold')
    ax1.grid(True, axis='y', ls=":", color='#333333')
    ax1.tick_params(colors='white')
    ax1.text(1, Q_required*1.5, f"Required: ~10^{int(np.log10(Q_required))}", color='white', ha='center', weight='bold')
    ax1.text(0, Q_max_humanity*1.5, f"Max limit: ~10^{int(np.log10(Q_max_humanity))}", color='#ff3366', ha='center', weight='bold')
    
    # Right: The Sagnac-RLVE Triumph
    bars2 = ax2.bar(['Interferometer\nNoise Floor', 'Sagnac-RLVE\nPredicted Signal'], 
                    [1e-7, phase_shift_rads], color=['#444444', '#00ffcc'])
    ax2.set_yscale('log')
    ax2.set_ylabel('Optical Phase Shift (Radians)', color='white', fontsize=12, weight='bold')
    ax2.set_title('Sagnac-RLVE: The Kinematic Vector Signal', color='white', fontsize=14, weight='bold')
    ax2.grid(True, axis='y', ls=":", color='#333333')
    ax2.tick_params(colors='white')
    ax2.text(1, phase_shift_rads*1.5, f"Signal: {phase_shift_rads:.2f} Rads", color='#00ffcc', ha='center', weight='bold')
    
    for spine in ax1.spines.values(): spine.set_color('#333333')
    for spine in ax2.spines.values(): spine.set_color('#333333')
    
    plt.suptitle("Tabletop Falsification: Scalar Strain vs Kinematic Advection", color='white', fontsize=16, weight='bold', y=0.98)
    
    textstr = (
        r"$\mathbf{Conclusion:}$ The RVR fails because scalar gravity ($GM/rc^2$) requires" + "\n" +
        r"planetary masses to be detectable. The Sagnac-RLVE succeeds because" + "\n" +
        r"it leverages a massive 200m optical lever and measures first-order fluid" + "\n" +
        r"drift velocity ($v_{fluid} \approx 0.38$ m/s), completely bypassing the $G/c^2$ gap."
    )
    plt.figtext(0.5, 0.02, textstr, ha='center', color='white', fontsize=12, bbox=dict(facecolor='#111111', edgecolor='#00ffcc', alpha=0.9, pad=10))

    filepath = os.path.join(OUTPUT_DIR, "tabletop_falsification_thresholds.png")
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_experimental_thresholds()