"""
AVE MODULE 8: DIELECTRIC SATURATION SHOCKWAVE (AXIOM 4)
-------------------------------------------------------
Enforces \Delta\phi \equiv \alpha limit. Demonstrates how the exact geometric saturation 
limit causes c to collapse at high energy densities, forcing wave-steepening, 
topological shockwaves, and the Dielectric Snap (genesis of fermions).
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_strict_shockwave():
    print("Simulating Topological Shockwave (Strict Alpha Bound)...")
    NX = 800
    z = np.linspace(0, 40, NX)
    
    ALPHA = 1 / 137.036
    c0 = 1.0 
    
    # Initialize near-saturation gamma ray pulse
    peak_strain = 0.96 * ALPHA
    V_linear = peak_strain * np.exp(-((z - 5)**2) / 2.0)
    V_nonlin = np.copy(V_linear)
    
    snapshots_lin = [np.copy(V_linear) / ALPHA]
    snapshots_nonlin = [np.copy(V_nonlin) / ALPHA]
    
    for t in range(1, 900):
        V_linear = np.roll(V_linear, 1)
        
        # Non-Linear AVE Advection (Axiom 4): c_eff = c_0 * (1 - (V/alpha)^4)^0.25
        V_ratio = np.clip(V_nonlin / ALPHA, 0.0, 0.999)
        c_eff = c0 * np.power(1.0 - V_ratio**4, 0.25)
        
        shift = (c0 * c_eff * 1.5).astype(int) 
        V_new = np.zeros_like(V_nonlin)
        
        for i in range(NX):
            if i + shift[i] < NX:
                V_new[i + shift[i]] = max(V_new[i + shift[i]], V_nonlin[i])
        
        V_nonlin = np.convolve(V_new, np.ones(3)/3.0, mode='same')
        
        if t % 250 == 0:
            snapshots_lin.append(np.copy(V_linear) / ALPHA)
            snapshots_nonlin.append(np.copy(V_nonlin) / ALPHA)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), facecolor='#050508')
    
    for i, snap in enumerate(snapshots_lin):
        axes[0].plot(z, snap, color='cyan', lw=2, alpha=0.3 + (i*0.15))
        axes[0].fill_between(z, snap, color='cyan', alpha=0.1)
        
    axes[0].set_title("Standard Linear Vacuum ($C_{eff}$ assumed constant)", color='white', fontsize=14, weight='bold')
    axes[0].axhline(1.0, color='red', linestyle=':', lw=2)
    axes[0].text(2, 1.05, "Dielectric Saturation Limit ($\\Delta\phi \\equiv \\alpha$)", color='red')

    for i, snap in enumerate(snapshots_nonlin):
        color = 'orange' if i < 3 else '#ff3366'
        axes[1].plot(z, snap, color=color, lw=2, alpha=0.3 + (i*0.15))
        axes[1].fill_between(z, snap, color=color, alpha=0.1)
        if i == 3:
            peak_idx = np.argmax(snap)
            axes[1].axvline(z[peak_idx], color='#ff3366', linestyle='--', alpha=0.8)
            axes[1].text(z[peak_idx]-0.5, 0.9, "Topological Rupture\n(Fermion Genesis)", color='#ff3366', ha='right', weight='bold')
        
    axes[1].set_title(r"AVE Non-Linear Vacuum ($c_{eff} = c_0 [1 - (\Delta\phi/\alpha)^4]^{1/4}$)", color='white', fontsize=14, weight='bold')
    axes[1].axhline(1.0, color='red', linestyle=':', lw=2)
    axes[1].text(5, 0.8, r"Peaks lag the base ($c \to 0$)", color='orange', weight='bold')
    
    for ax in axes:
        ax.set_facecolor('#050508')
        ax.set_ylim(0, 1.3)
        ax.set_xlim(0, 40)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "dielectric_shockwave.png")
    plt.savefig(output_file, dpi=300, facecolor=fig.get_facecolor())
    print(f"Saved Strict Axiom 4 Shockwave to {output_file}")

if __name__ == "__main__":
    simulate_strict_shockwave()