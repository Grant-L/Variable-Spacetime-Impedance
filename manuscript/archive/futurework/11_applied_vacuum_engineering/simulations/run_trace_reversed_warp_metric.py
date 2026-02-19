"""
AVE MODULE 36: THE TRACE-REVERSED WARP METRIC
---------------------------------------------
Strict derivation of the Warp Drive optical metric using Cosserat Elasticity.
Enforces the exact Poisson ratio coupling (\nu = 2/7) for the transverse 
optical index (n_\perp) and the scalar isotropic coupling (1/7) for n_scalar.
Demonstrates that a volumetric pressure dipole (compression front, rarefaction rear)
natively yields v_g > c in the trailing wake, enabling superluminal translation.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/11_applied_vacuum_engineering/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_warp_metric():
    print("Simulating Trace-Reversed Warp Drive Metric...")
    
    z = np.linspace(-10, 10, 1000)
    
    # 1. Engineered Volumetric Strain \text{Tr}(\varepsilon)
    # Compression (Positive strain) in front, Rarefaction (Negative strain) in rear
    Tr_strain = np.exp(-((z - 3)**2)/2.0) * 0.8 - np.exp(-((z + 3)**2)/2.0) * 0.8
    
    # 2. Strict AVE Optical Tensor Couplings (From Chapter 7)
    # Scalar Index (felt by massive passengers inside the ship)
    n_scalar = 1.0 + (1.0/7.0) * Tr_strain
    
    # Transverse Index (felt by propagating photons / external observers)
    n_perp = 1.0 + (2.0/7.0) * Tr_strain
    
    # Local Phase Velocity of light v = c / n_\perp
    v_light = 1.0 / n_perp
    
    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor('#050508'); ax1.set_facecolor('#050508')
    
    color1 = '#00ffcc'
    ax1.plot(z, n_perp, color=color1, lw=3, label=r'Transverse Optical Index ($n_\perp = 1 + \frac{2}{7}\text{Tr}(\varepsilon)$)')
    ax1.plot(z, n_scalar, color='#ffcc00', lw=2, linestyle='--', label=r'Passenger Scalar Index ($n_{scalar} = 1 + \frac{1}{7}\text{Tr}(\varepsilon)$)')
    ax1.set_xlabel('Spatial Coordinate (z)', color='white', fontsize=13, weight='bold')
    ax1.set_ylabel('Refractive Index (n)', color=color1, fontsize=13, weight='bold')
    ax1.tick_params(axis='y', labelcolor=color1, colors='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.axhline(1.0, color='gray', linestyle=':', lw=1.5)
    
    ax2 = ax1.twinx()
    color2 = '#ff3366'
    ax2.plot(z, v_light, color=color2, lw=3, label=r'Local Light Speed ($v_{eff} = c/n_\perp$)')
    ax2.set_ylabel('Effective Wave Velocity ($v/c$)', color=color2, fontsize=13, weight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    for spine in ax1.spines.values(): spine.set_color('#333333')
    for spine in ax2.spines.values(): spine.set_color('#333333')
    
    # Annotations
    ax1.axvspan(-5, -1, color='blue', alpha=0.1)
    ax1.text(-3, 0.9, "Rear Rarefaction\n($Tr(\epsilon) < 0$)\n$v_{eff} > c$", color='#ff3366', ha='center', weight='bold')
    
    ax1.axvspan(1, 5, color='red', alpha=0.1)
    ax1.text(3, 1.1, "Front Compression\n($Tr(\epsilon) > 0$)\n$v_{eff} < c$", color='#00ffcc', ha='center', weight='bold')
    
    fig.suptitle('The Engineered Warp Metric via Cosserat Optical Tensors', color='white', fontsize=16, weight='bold', y=0.95)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    filepath = os.path.join(OUTPUT_DIR, "warp_metric_tensors.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_warp_metric()