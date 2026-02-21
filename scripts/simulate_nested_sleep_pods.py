"""
AVE Falsification Protocol: Nested Subluminal Sleep Pods
--------------------------------------------------------
Demonstrates the engineering principle of nesting topological
impedance boundaries. 

Because time dilation and the speed of light are strictly 
local functions of the vacuum's refractive index (n = sqrt(mu*epsilon)),
we can engineer a ship that actively rarefies the exterior vacuum 
(superluminal low-impedance slipstream) while simultaneously compressing 
the internal vacuum of a sleep pod (subluminal high-impedance cavity).

This allows the ship's mainframe to experience months of transit time 
at v > c, while the biological occupant experiences only hours, 
trapped inside an artificial gravitational time-dilation field.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_nested_sleep_pods():
    print("==========================================================")
    print(" AVE TIME DILATION: NESTED SUBLUMINAL SLEEP PODS")
    print("==========================================================")
    
    # Spatial X-axis representing a cross section:
    # Deep Space -> Ship Hull -> Sleep Pod -> Ship Hull -> Deep Space
    x = np.linspace(-15, 15, 2000)
    
    # 1. Background deep-space baseline (n = 1.0)
    n_space = np.ones_like(x)
    
    # 2. Outer Hull Superluminal Slipstream (Rarefaction, n < 1.0)
    # The ship's main drive creates a low-impedance bubble to exceed c_0
    hull_mask = (np.abs(x) < 10)
    n_hull = np.ones_like(x)
    n_hull[hull_mask] = 0.1 # 10x faster local speed of light (Superluminal)
    
    # Smooth the hull boundary transition
    n_hull_smooth = 1.0 - 0.9 * np.exp(-0.5 * (np.abs(x) - 10)**2)
    n_hull_smooth = np.where(np.abs(x) < 10, 0.1, n_hull_smooth)
    
    # 3. Inner Sleep Pod Subluminal Cavity (Compression, n >> 1.0)
    # The bunker creates a localized, artificial high-gravity field
    pod_mask = (np.abs(x) < 2)
    # n drops massively into a high-impedance spike
    n_pod_spike = 40.0 * np.exp(-1.5 * x**2)
    
    # Total Nested Refractive Profile
    n_total = n_hull_smooth + n_pod_spike
    
    # Local Speed of Light (c_local = c_0 / n)
    # Normalized to c_0 = 1.0
    c_local = 1.0 / n_total
    
    # Biological Clock Tick Rate (Relative to Deep Space = 1.0)
    # Tick rate scales linearly with local c (slower c = slower clocks)
    tick_rate = c_local 
    
    # --- Visualization Suite ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor='#0B0F19')
    fig.suptitle("AVE Nested Impedance: Superluminal Transit with Subluminal Sleep Pods", color='white', fontsize=18, weight='bold', y=0.96)
    
    # Panel 1: The Local Refractive Index (Vacuum Density)
    ax1.set_facecolor('#0B0F19')
    ax1.plot(x, n_total, color='cyan', lw=3, label="Local Refractive Index $n_\perp(x)$")
    ax1.axhline(1.0, color='gray', ls='--', label="Deep Space Baseline ($n=1$)")
    ax1.fill_between(x, 0, n_total, where=pod_mask, color='magenta', alpha=0.3, label="Pod (High Impedance)")
    ax1.fill_between(x, 0, n_total, where=(hull_mask & ~pod_mask), color='blue', alpha=0.3, label="Hull (Low Impedance)")
    
    ax1.set_title("1. Topological Density Map (Nested Macroscopic Boundaries)", color='white', pad=10, weight='bold')
    ax1.set_ylabel("Refractive Index / Metric Density", color='gray')
    ax1.set_yscale('log')
    ax1.set_ylim(0.05, 100)
    ax1.legend(loc='upper right', facecolor='#111111', edgecolor='gray', labelcolor='white')
    
    # Panel 2: Local Time Dilation (Biological Tick Rate)
    ax2.set_facecolor('#0B0F19')
    ax2.plot(x, tick_rate, color='#00FFCC', lw=3, label="Local Clock Frame ($c_{local} / c_0$)")
    ax2.axhline(1.0, color='gray', ls='--', label="Deep Space Baseline (1s / s)")
    
    # Highlight the different time frames
    ax2.axhline(10.0, color='blue', ls=':', lw=2)
    ax2.text(-14, 12, "Ship Mainframe Rate\n(10x Accelerated)", color='blue', weight='bold')
    
    ax2.axhline(1/40.0, color='magenta', ls=':', lw=2)
    ax2.text(-4, 1/30.0, "Biological Pod Rate\n(1/40th baseline)", color='magenta', weight='bold')
    
    ax2.set_title("2. Local Biological Time Dilation (Speed of Light Frame)", color='white', pad=10, weight='bold')
    ax2.set_xlabel("Cross-Sectional Distance (meters)", color='gray')
    ax2.set_ylabel("Relative Clock Rate ($dt_{local} / dt_0$)", color='gray')
    ax2.set_yscale('log')
    ax2.set_ylim(0.01, 20)
    ax2.legend(loc='upper right', facecolor='#111111', edgecolor='gray', labelcolor='white')
    
    for ax in [ax1, ax2]:
        ax.tick_params(colors='lightgray')
        ax.grid(True, ls=':', color='#333333', alpha=0.5)
        for spine in ax.spines.values(): spine.set_color('#333333')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    OUTPUT_DIR = "assets/sim_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "simulate_nested_sleep_pods.png")
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"\n[+] Saved Time Dilation Nested Plot to {out_path}")

if __name__ == "__main__":
    simulate_nested_sleep_pods()
