"""
AVE MODULE 55: REACTIVE METRIC POWER (THE ORBITAL LC TANK)
----------------------------------------------------------
Resolves the "Vacuum Friction" paradox of classical fluid aethers.
Applies AC Circuit Power Analysis to orbital mechanics.
Proves that because the Topo-Kinematic Voltage (Gravity) is exactly 90 degrees 
out of phase with the Current (Velocity), the Real Power (Watts) dissipated 
by the vacuum is absolute zero. 
The orbit is a purely Reactive circuit exchanging Volt-Amperes Reactive (VARs).
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/13_spacetime_circuit_analysis/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_orbital_ac_power():
    print("Simulating Orbital Reactive Power (AC Circuit Analogy)...")
    
    # Time vector for one complete orbit
    t = np.linspace(0, 2 * np.pi, 1000)
    
    # 1. Orbital Kinematics (Circular Orbit)
    # Velocity (Current: I = xi * v)
    I_x = -np.sin(t)
    I_y = np.cos(t)
    
    # Centripetal Force (Voltage: V = F / xi)
    V_x = -np.cos(t)
    V_y = -np.sin(t)
    
    # 2. AC Power Analysis
    # Real Power: P = V_x*I_x + V_y*I_y (Dot Product)
    Real_Power = (V_x * I_x) + (V_y * I_y)
    
    # Reactive Power: Q = |V x I| (Cross Product Magnitude)
    Reactive_Power = np.abs((V_x * I_y) - (V_y * I_x))
    
    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor('#0a0a12')
    ax1.set_facecolor('#0a0a12'); ax2.set_facecolor('#0a0a12')
    
    # Plot 1: The Phase Angle
    ax1.plot(t, I_y, color='#00ffcc', lw=3, label=r'Tangential Current ($I_{vac} \propto v$)')
    ax1.plot(t, V_y, color='#ff3366', lw=3, linestyle='--', label=r'Radial Voltage ($V_{vac} \propto F_g$)')
    ax1.set_title(r'Orbital AC Phase Shift ($\theta = 90^\circ$)', color='white', fontsize=14, weight='bold')
    ax1.set_xlabel('Orbital Phase (Radians)', color='white')
    ax1.set_ylabel('Amplitude (Normalized)', color='white')
    ax1.legend(loc='lower right', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    # Plot 2: Real vs Reactive Power
    ax2.plot(t, Real_Power, color='#ffcc00', lw=4, label='Real Power $P$ (Dissipated Watts)')
    ax2.plot(t, Reactive_Power, color='#4FC3F7', lw=3, linestyle='--', label='Reactive Power $Q$ (Conserved VARs)')
    ax2.fill_between(t, 0, Reactive_Power, color='#4FC3F7', alpha=0.15)
    ax2.set_title('Spacetime Power Dissipation', color='white', fontsize=14, weight='bold')
    ax2.set_xlabel('Orbital Phase (Radians)', color='white')
    ax2.set_ylabel('Power Amplitude', color='white')
    ax2.legend(loc='center', facecolor='#111111', edgecolor='white', labelcolor='white')
    ax2.set_ylim(-0.5, 1.5)
    
    textstr = (
        r"$\mathbf{The~Friction~Paradox~Resolved:}$" + "\n" +
        r"$P_{real} = V_{rms} I_{rms} \cos(90^\circ) \equiv \mathbf{0}$" + "\n" +
        r"Because force and velocity are orthogonal, the Earth dissipates" + "\n" +
        r"exactly zero Real Power into the vacuum. The orbit is a lossless" + "\n" +
        r"reactive LC tank circuit maintaining constant energy."
    )
    ax2.text(0.4, -0.3, textstr, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='white', alpha=0.8, pad=8))

    for ax in [ax1, ax2]:
        ax.grid(True, ls=':', color='#333333'); ax.tick_params(colors='lightgray')
        for spine in ax.spines.values(): spine.set_color('#333333')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "orbital_reactive_power.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_orbital_ac_power()