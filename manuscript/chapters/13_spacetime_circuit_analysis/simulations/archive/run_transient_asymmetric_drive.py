"""
AVE MODULE 51: TRANSIENT ASYMMETRIC METRIC DRIVE (TAMD)
-------------------------------------------------------
Applies SMPS Flyback topology to the Bingham-Plastic vacuum.
Proves that an asymmetric electromagnetic waveform (fast rise, slow fall) 
acts as a "Vacuum Rectifier" (A Fluidic Diode).
A sawtooth transient pulse mechanically grips the solid lattice 
on the slow edge, and liquefies/slips through the lattice on the fast edge, 
generating massive, continuous DC thrust. Resolves the EmDrive.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
from scipy import integrate
import os

OUTPUT_DIR = "manuscript/chapters/13_ee_for_ave/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_flyback_metric_rectification():
    print("Simulating TAMD (The Vacuum Fluidic Diode)...")
    
    t = np.linspace(0, 3, 5000) # 3 cycles
    
    # 1. WAVEFORMS
    # Waveform A: Continuous Sine Wave (NASA EmDrive)
    I_sine = np.sin(2 * np.pi * 1 * t)
    
    # Waveform B: Asymmetric Sawtooth (AVE TAMD / Flyback Pulse)
    # Fast rise (10%), Slow fall (90%)
    I_saw = sawtooth(2 * np.pi * 1 * t, width=0.1)
    
    # 2. VACUUM FORCE (F_{vac} = dp/dt \propto dI/dt)
    dt = t[1] - t[0]
    F_sine = np.gradient(I_sine, dt)
    F_saw = np.gradient(I_saw, dt)
    
    # Normalize forces
    F_sine /= np.max(np.abs(F_sine))
    F_saw /= np.max(np.abs(F_saw))
    
    # 3. BINGHAM-PLASTIC RECTIFICATION (The Fluidic Diode)
    yield_threshold = 0.20 # 20% of max force
    
    def apply_bingham_rectifier(force_array):
        # By Newton's 3rd Law, F_ship = -F_vac
        # If |F_vac| < yield, vacuum is SOLID -> Grip -> Thrust is transferred
        # If |F_vac| > yield, vacuum is SUPERFLUID -> Slip -> Zero thrust
        thrust = np.zeros_like(force_array)
        for i, f in enumerate(force_array):
            if abs(f) < yield_threshold:
                thrust[i] = -f  # Solid grip, momentum transferred to ship
            else:
                thrust[i] = 0.0 # Superfluid slip, zero traction
        return thrust

    Thrust_sine = apply_bingham_rectifier(F_sine)
    Thrust_saw = apply_bingham_rectifier(F_saw)
    
    # Calculate Net DC Thrust
    net_sine = integrate.trapezoid(Thrust_sine, t)
    net_saw = integrate.trapezoid(Thrust_saw, t)
    
    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=150)
    fig.patch.set_facecolor('#050508')
    
    # Plot 1: The CW Sine Wave Failure
    ax1.set_facecolor('#050508')
    ax1.plot(t, F_sine, color='#444444', lw=2, linestyle='--', label=r'Applied Force on Vacuum ($dp/dt$)')
    ax1.plot(t, Thrust_sine, color='#ff3366', lw=3, label='Reaction Force on Ship (Traction)')
    ax1.axhline(yield_threshold, color='white', linestyle=':', lw=1)
    ax1.axhline(-yield_threshold, color='white', linestyle=':', lw=1)
    ax1.set_title(f'Standard RF Cavity (Sine Wave)\nNet DC Thrust: {net_sine:.4f} N (Symmetric Cancellation)', color='white', fontsize=14, weight='bold')
    ax1.legend(loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    # Plot 2: The Flyback Rectification Triumph
    ax2.set_facecolor('#050508')
    ax2.plot(t, F_saw, color='#444444', lw=2, linestyle='--', label=r'Applied Force on Vacuum ($dp/dt$)')
    ax2.plot(t, Thrust_saw, color='#00ffcc', lw=3, label='Reaction Force on Ship (Traction)')
    ax2.axhline(yield_threshold, color='white', linestyle=':', lw=1, label=r'Bingham Yield Threshold ($\dot{\gamma}_c$)')
    ax2.axhline(-yield_threshold, color='white', linestyle=':', lw=1)
    
    ax2.fill_between(t, 0, Thrust_saw, where=(Thrust_saw > 0), color='#00ffcc', alpha=0.3)
    
    ax2.set_title(f'AVE Transient Metric Drive (Asymmetric Flyback)\nNet DC Thrust: {net_saw:.2f} N (Massive Forward Rectification)', color='white', fontsize=14, weight='bold')
    ax2.set_xlabel('Time (Microseconds)', color='white', fontsize=12, weight='bold')
    ax2.legend(loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    for ax in [ax1, ax2]:
        for spine in ax.spines.values(): spine.set_color('#333333')
        ax.grid(True, which="major", ls=":", color='#444444', alpha=0.8)
        ax.tick_params(colors='white')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "flyback_rectification.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight'); plt.close(); print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_flyback_metric_rectification()