"""
AVE MODULE 5: DIELECTRIC SATURATION (THE KERR EFFECT)
Simulates a 1D pulse. Demonstrates how the Axiom 4 saturation limit 
causes wave-steepening and topological shockwaves at extreme energy densities.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_shockwave():
    print("Simulating Topological Shockwave (Axiom 4)...")
    NX = 800
    z = np.linspace(0, 40, NX)
    
    # Initialize Gaussian Pulse
    V_linear = np.exp(-((z - 5)**2) / 2.0)
    V_nonlin = np.copy(V_linear)
    
    # Physics parameters
    c0 = 1.0        # Base speed of light
    V_max = 1.05    # The Breakdown Voltage threshold (V_0)
    
    # Store snapshots
    snapshots_lin = [np.copy(V_linear)]
    snapshots_nonlin = [np.copy(V_nonlin)]
    
    for t in range(1, 800):
        # 1. Linear Propagation (Constant c)
        V_linear = np.roll(V_linear, 1) # simple advection
        
        # 2. Non-Linear Propagation (Axiom 4 Kerr Effect)
        # As V approaches V_max, Capacitance spikes -> speed c drops
        c_eff = c0 * np.power(np.clip(1.0 - (V_nonlin / V_max)**4, 0.01, 1.0), 0.25)
        
        # Non-linear advection: Peaks travel slower than the base
        shift = (c_eff * 1.5).astype(int) 
        V_new = np.zeros_like(V_nonlin)
        for i in range(NX):
            if i + shift[i] < NX:
                V_new[i + shift[i]] = max(V_new[i + shift[i]], V_nonlin[i])
        
        V_nonlin = np.convolve(V_new, np.ones(3)/3.0, mode='same')
        
        if t % 250 == 0:
            snapshots_lin.append(np.copy(V_linear))
            snapshots_nonlin.append(np.copy(V_nonlin))

    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='black')
    colors = ['cyan', 'dodgerblue', 'blue', 'purple']
    
    # Plot Linear
    for i, snap in enumerate(snapshots_lin):
        axes[0].plot(z, snap, color=colors[i], lw=2, alpha=0.8)
        axes[0].fill_between(z, snap, color=colors[i], alpha=0.1)
        
    axes[0].set_title("Standard Linear Vacuum (Constant Capacitance)", color='white', fontsize=14)
    axes[0].text(35, 0.8, "Pulse maintains shape\n(Constant c)", color='cyan', ha='right')

    # Plot Non-Linear
    for i, snap in enumerate(snapshots_nonlin):
        axes[1].plot(z, snap, color='orange' if i < 3 else 'red', lw=2, alpha=0.8)
        axes[1].fill_between(z, snap, color='orange' if i < 3 else 'red', alpha=0.1)
        if i == 3:
            peak_idx = np.argmax(snap)
            axes[1].axvline(z[peak_idx], color='red', linestyle='--', alpha=0.7)
            axes[1].text(z[peak_idx]-0.5, 0.9, "Topological Rupture\n(Pair Production)", color='red', ha='right')
        
    axes[1].set_title("AVE Non-Linear Vacuum (Axiom 4: Dielectric Saturation)", color='white', fontsize=14)
    axes[1].text(5, 0.8, "Peaks lag the base ($c \propto V^{-3}$)\ncausing violent steepening.", color='orange')
    
    for ax in axes:
        ax.set_facecolor('black')
        ax.axhline(0, color='gray', lw=1)
        ax.axis('off')
        
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "dielectric_shockwave.png")
    plt.savefig(output_file, dpi=300, facecolor='black')
    print(f"Saved Dielectric Shockwave to {output_file}")

if __name__ == "__main__":
    simulate_shockwave()