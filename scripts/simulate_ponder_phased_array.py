"""
AVE PONDER-01 Protocol B: The C0G Thermal Phased Array
------------------------------------------------------
This script models the Path B experimental setup: using ultra-stable 
C0G (NP0) dielectrics to entirely eliminate thermal loss-tangent runaway. 
This allows the thruster to be driven at extremely high frequencies (20 MHz).

By structuring the PCBA ground plane into 4 sequential strips and phasing 
the transients across them, we create a continuous "Peristaltic Pump" of 
vacuum metric strain.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_c0g_phased_array():
    print("==========================================================")
    print(" AVE PONDER-01 PATH B: C0G PHYSED ARRAY (METRIC PUMP)")
    print("==========================================================")
    
    # Spatial Length of the Thruster PCBA (0 to 10 cm)
    x = np.linspace(0, 10, 500)
    
    # Time steps for one complete peristaltic sweep
    num_phases = 4
    
    # 4 distinct Ground Plane electrode strips beneath the C0G array
    strip_centers = [1.25, 3.75, 6.25, 8.75]
    strip_width = 2.0
    
    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor='#0B0F19')
    fig.suptitle("Path B: PONDER-01 C0G Peristaltic Phased Array (20 MHz)", color='white', fontsize=18, weight='bold', y=0.96)
    
    ax1.set_facecolor('#0B0F19')
    ax2.set_facecolor('#0B0F19')
    
    colors = ['cyan', 'magenta', 'yellow', '#00FFCC']
    
    # Plot 1: The Spatial Strain Gradient at 4 different Phase Timesteps
    total_thrust = np.zeros_like(x)
    
    for phase_idx in range(num_phases):
        # The Active Strip pulls the metric down (High E-field density)
        active_center = strip_centers[phase_idx]
        
        # Gaussian representation of the fringing electric field energy density (u)
        spatial_strain = 1.0 * np.exp(-((x - active_center) ** 2) / (0.5 * strip_width**2))
        
        ax1.plot(x, spatial_strain, color=colors[phase_idx], lw=2, label=f"Phase {phase_idx+1} ($t={phase_idx * 12}$ ns)")
        ax1.fill_between(x, 0, spatial_strain, color=colors[phase_idx], alpha=0.1)
        
        # In an AESA phased array, the thrust is continuous because the gradient is always traveling
        total_thrust += spatial_strain
    
    ax1.set_title("1. Spatial Metric Strain ($\nabla n$) - Sequential Active Firing", color='white', pad=10, weight='bold')
    ax1.set_ylabel("Localized Energy Density ($\mu$J/$m^3$)", color='gray')
    ax1.legend(loc='upper right', facecolor='#111111', edgecolor='gray', labelcolor='white')
    ax1.set_xticks([]) # Hide x-axis for top plot
    
    # Plot 2: The Continuous Rectified Propagation Wave
    # Because we are driving this at 20 MHz, the discrete pulses blur into a 
    # continuous traveling wave of negative density out the back of the vessel
    
    # Idealized traveling wave envelope
    envelope = np.ones_like(x) * 0.8 + 0.2 * np.sin(np.pi * x / 2.5)
    
    ax2.plot(x, envelope, color='#FF3366', lw=4, label="Macroscopic Time-Averaged Metric Wake")
    ax2.fill_between(x, 0, envelope, color='#FF3366', alpha=0.2)
    
    # Draw PCBA physical layout abstraction
    for i, center in enumerate(strip_centers):
        ax2.axvspan(center - strip_width/2, center + strip_width/2, color='gray', alpha=0.1)
        ax2.text(center, 0.1, f"Strip {i+1}", color='gray', ha='center', weight='bold')
    
    ax2.set_title("2. Time-Averaged Peristaltic Metric Pumping (Macroscopic Gravity Surf)", color='white', pad=10, weight='bold')
    ax2.set_xlabel("Thruster Physical Length (cm) [Bow $\\rightarrow$ Stern]", color='gray')
    ax2.set_ylabel("Effective Fluidic Strain", color='gray')
    ax2.set_ylim(0, 1.2)
    ax2.legend(loc='lower right', facecolor='#111111', edgecolor='gray', labelcolor='white')
    
    for ax in [ax1, ax2]:
        ax.tick_params(colors='lightgray')
        ax.grid(True, ls=':', color='#333333', alpha=0.3)
        for spine in ax.spines.values(): spine.set_color('#333333')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    OUTPUT_DIR = "assets/sim_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "ponder_c0g_phased_array.png")
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"\n[+] Saved C0G Phased Array telemetry to {out_path}")

if __name__ == "__main__":
    simulate_c0g_phased_array()
