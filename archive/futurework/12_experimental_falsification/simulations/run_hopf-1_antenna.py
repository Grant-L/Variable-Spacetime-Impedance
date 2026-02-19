"""
AVE MODULE 120: OPEN-SOURCE PCBA IMPLEMENTATION
-----------------------------------------------
1. Generates the exact Cartesian routing coordinates for the 
   Top and Bottom copper layers of the HOPF-01 PCBA Antenna.
2. Simulates the electrical telemetry of the PONDER-01 BaTiO3 
   SiC MOSFET metric thruster on an analytical micro-balance.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/12_experimental_falsification/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_pcba_eda_and_telemetry():
    print("Generating EDA PCBA Routing and Solid-State Telemetry...")
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    fig.patch.set_facecolor('#050508')
    for ax in axs:
        ax.set_facecolor('#050508')
        ax.tick_params(colors='lightgray')
        for spine in ax.spines.values(): spine.set_color('#333333')
        ax.grid(True, ls=':', color='#333333')
    
    # ---------------------------------------------------------
    # 1. HOPF-01: KiCad/Altium Torus Knot Trace Generator
    # ---------------------------------------------------------
    ax1 = axs[0]
    R_inner = 5.0  # mm
    R_outer = 15.0 # mm
    turns = 24     # Number of Vias/Traces
    
    # Twist angle to generate Helicity (A parallel B)
    twist_angle = np.pi / 2  # 90 degree twist per stroke
    
    for i in range(turns):
        theta_start = i * (2 * np.pi / turns)
        
        # TOP LAYER TRACE (Red): Spirals Out and Clockwise
        r_top = np.linspace(R_inner, R_outer, 50)
        theta_top = np.linspace(theta_start, theta_start + twist_angle, 50)
        x_top = r_top * np.cos(theta_top)
        y_top = r_top * np.sin(theta_top)
        
        # BOTTOM LAYER TRACE (Blue): Spirals In and Clockwise
        theta_next = (i + 1) * (2 * np.pi / turns)
        r_bot = np.linspace(R_outer, R_inner, 50)
        theta_bot = np.linspace(theta_start + twist_angle, theta_next, 50)
        x_bot = r_bot * np.cos(theta_bot)
        y_bot = r_bot * np.sin(theta_bot)
        
        ax1.plot(x_top, y_top, color='#ff3366', lw=2.0, alpha=0.9, label='Top Copper' if i==0 else "")
        ax1.plot(x_bot, y_bot, color='#00ffcc', lw=2.0, alpha=0.9, linestyle='--', label='Bottom Copper' if i==0 else "")
        
        # Vias
        ax1.scatter(x_top[0], y_top[0], color='white', s=20, zorder=5) 
        ax1.scatter(x_top[-1], y_top[-1], color='white', s=20, zorder=5)

    ax1.set_aspect('equal')
    ax1.set_title('1. HOPF-01: Chiral PCB Routing Paths', color='white', weight='bold', fontsize=14)
    ax1.legend(loc='upper right', facecolor='#111111', edgecolor='gray', labelcolor='white')
    ax1.axis('off')
    ax1.text(0, 0, 'Void Inner\nLayers', color='gray', ha='center', va='center', weight='bold')
    
    # ---------------------------------------------------------
    # 2. PONDER-01: Analytical Balance Telemetry
    # ---------------------------------------------------------
    ax2 = axs[1]
    t = np.linspace(0, 10, 1000) # 10 seconds of bench testing
    
    # Bluetooth trigger at t=3s, off at t=7s
    mosfet_active = (t >= 3) & (t <= 7)
    
    # Base weight of PCBA = 50.000 grams
    base_weight = np.full_like(t, 50000.0) 
    
    # Thrust = 47 uN = ~4.8 milligrams of weight reduction
    thrust_mg = 4.8
    scale_readout = base_weight - (thrust_mg * mosfet_active)
    
    # Add +/- 0.2 mg of typical laboratory scale noise
    noise = np.random.normal(0, 0.2, len(t))
    scale_readout += noise
    
    ax2.plot(t, scale_readout, color='#00ffcc', lw=2, label='Digital Scale Readout (mg)')
    ax2.axvspan(3, 7, color='#FFD54F', alpha=0.15, label='SiC MOSFET 1kV PWM Active')
    
    ax2.set_title('2. PONDER-01: Macroscopic Thrust Readout', color='white', weight='bold', fontsize=14)
    ax2.set_xlabel('Time (seconds)', color='white')
    ax2.set_ylabel('Total Scale Weight (milligrams)', color='white')
    
    # Zoom in on the shift
    ax2.set_ylim(49990, 50005)
    # Format y-axis to look like a precise scale (e.g., 50.0000 g)
    ax2.set_yticks([49990, 49995, 50000, 50005])
    ax2.set_yticklabels(['49,990 mg', '49,995 mg', '50,000 mg', '50,005 mg'])
    
    ax2.legend(loc='upper right', facecolor='#111111', edgecolor='gray', labelcolor='white', fontsize=9)
    ax2.text(5, 49993, r"$\Delta = -4.8$ mg" + "\n" + r"(47 $\mu$N Ponderomotive Thrust)", color='white', ha='center', weight='bold', bbox=dict(facecolor='#111111', edgecolor='#FFD54F'))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "pcba_design_blueprints.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": generate_pcba_eda_and_telemetry()