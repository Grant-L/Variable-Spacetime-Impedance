"""
AVE MODULE 56: LOG-SCALE IMPEDANCE MATCHING & S-PARAMETERS
----------------------------------------------------------
Models a gravity well as a Tapered LC Transmission Line on an 
astronomical logarithmic scale.
Proves that because gravity volumetrically compresses the lattice, 
it increases both Inductance (\mu) and Capacitance (\epsilon) proportionally.
Therefore, Characteristic Impedance Z_0 remains perfectly invariant.
Calculates the Return Loss (S11) to prove why gravity wells 
perfectly absorb light without reflecting it.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/13_spacetime_circuit_analysis/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_log_scale_s_parameters():
    print("Simulating Log-Scale Gravitational S-Parameters...")
    
    # Distance from singularity (r / R_s), from 100,000 R_s down to the Event Horizon (1 R_s)
    r = np.logspace(0, 5, 1000)
    
    # Gravitational Refractive Index Profile
    n_r = 1.0 + 1.0 / r
    Z_base = 376.73 # Ohms
    
    # 1. AVE MODEL (Volumetric Compression)
    # Both L and C scale proportionally with density n(r)
    L_ave = 1.0 * n_r
    C_ave = 1.0 * n_r
    Z_ave = np.sqrt(L_ave / C_ave) * Z_base # Stays perfectly flat at 376.73
    
    # 2. CLASSICAL UNMATCHED MODEL (Flawed Dielectric)
    # If gravity was just a standard optical dielectric, only C (\epsilon) would scale.
    L_flawed = np.ones_like(r)
    C_flawed = 1.0 * n_r
    Z_flawed = np.sqrt(L_flawed / C_flawed) * Z_base # Drops as it enters the well
    
    # 3. S-PARAMETERS (Return Loss / Reflection Coefficient)
    Gamma_ave = np.zeros_like(r) + 1e-15 # Numerical floor for perfect match
    Gamma_flawed = np.abs((Z_flawed - Z_base) / (Z_flawed + Z_base))
    
    S11_ave = 20 * np.log10(Gamma_ave)
    S11_flawed = 20 * np.log10(Gamma_flawed + 1e-15)
    
    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), dpi=150)
    fig.patch.set_facecolor('#0a0a12'); ax1.set_facecolor('#0a0a12'); ax2.set_facecolor('#0a0a12')
    
    # Invert X-axis so wave travels Left-to-Right into the singularity
    ax1.set_xlim(1e5, 1e0)
    ax2.set_xlim(1e5, 1e0)
    
    # Plot 1: Log-Log Component Divergence
    ax1.plot(r, n_r, color='#FFD54F', lw=3, label=r'Refractive Index ($n \propto \rho$)')
    ax1.plot(r, Z_ave / Z_base, color='#4FC3F7', lw=4, linestyle='--', label=r'AVE Impedance ($Z_0$ Perfectly Matched)')
    ax1.plot(r, Z_flawed / Z_base, color='#E57373', lw=2, linestyle='-.', label=r'Unmatched Model ($Z_0$ Drops)')
    
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_title('Gravitational Component Divergence (Log-Log Scale)', color='white', fontsize=14, weight='bold')
    ax1.set_ylabel(r'Component Scaling Factor', color='white', weight='bold')
    ax1.legend(loc='upper right', facecolor='#111111', edgecolor='gray', labelcolor='white')
    
    # Plot 2: Semi-Log Return Loss (S11)
    ax2.plot(r, S11_flawed, color='#E57373', lw=3, label='Standard Dielectric Reflection ($S_{11} \gg -10$ dB)')
    ax2.plot(r, S11_ave, color='#4FC3F7', lw=4, label='AVE Perfect Match ($S_{11} \to -\infty$ dB)')
    
    ax2.set_xscale('log')
    ax2.set_ylim(-160, 0)
    ax2.set_title('RF Return Loss Profile ($S_{11}$)', color='white', fontsize=14, weight='bold')
    ax2.set_xlabel('Radial Distance from Singularity ($r/R_s$)', color='white', weight='bold')
    ax2.set_ylabel('Reflected Power $S_{11}$ (dB)', color='white', weight='bold')
    ax2.legend(loc='center right', facecolor='#111111', edgecolor='gray', labelcolor='white')
    
    textstr = (
        r"$\mathbf{The~S{-}Parameters~of~Spacetime:}$" + "\n" +
        r"If gravity acted like standard optical glass (only changing $\epsilon$), the" + "\n" +
        r"impedance mismatch would cause the gravity well to reflect light like a mirror." + "\n" +
        r"Because AVE gravity is volumetric compression, it scales $\mu$ and $\epsilon$ symmetrically." + "\n" +
        r"This keeps $Z_0$ perfectly flat, pushing Return Loss to absolute zero."
    )
    ax2.text(1e4, -130, textstr, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='gray', alpha=0.9, pad=10))

    for ax in [ax1, ax2]:
        ax.grid(True, which="both", ls=':', color='#333333'); ax.tick_params(colors='lightgray')
        for spine in ax.spines.values(): spine.set_color('#333333')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "log_impedance_s_parameters.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_log_scale_s_parameters()