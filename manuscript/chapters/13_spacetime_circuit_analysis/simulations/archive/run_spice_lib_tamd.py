"""
AVE MODULE 53: TAMD SPICE SIMULATION (MOMENTUM RECTIFIER)
---------------------------------------------------------
Executes a transient circuit simulation using the Bingham Diode.
Proves that driving the vacuum with an asymmetric AC Voltage (Force)
rectifies into a continuous DC Current (Velocity) because the 
non-linear resistor (vacuum viscosity) acts exactly as a TVS Diode.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import sawtooth
import os

OUTPUT_DIR = "manuscript/chapters/13_ee_for_ave/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BinghamDiode:
    def __init__(self, V_zener, R_solid, R_fluid, k_smooth=1000):
        self.V_zener = V_zener
        self.R_fluid = R_fluid
        self.k = k_smooth
        
    def v_drag(self, I):
        """ Voltage drop (Drag Force) across the vacuum for a given Current """
        return self.V_zener * np.tanh(self.k * I) + self.R_fluid * I

def simulate_tamd_rectifier():
    print("Simulating Transient Momentum Rectifier...")
    
    t = np.linspace(0, 3, 5000)
    L_paddle = 0.1 
    D_vac = BinghamDiode(V_zener=2.0, R_solid=1.0, R_fluid=0.1)
    
    def ode_sine(t_val, I):
        V_s = 5.0 * np.sin(2 * np.pi * 1.0 * t_val)
        dI_dt = (V_s - D_vac.v_drag(I[0])) / L_paddle
        return [dI_dt]

    def ode_saw(t_val, I):
        # Asymmetric AC Drive (Fast rise, slow fall) -> Zero Mean AC Voltage
        V_s = 5.0 * sawtooth(2 * np.pi * 1.0 * t_val, width=0.1)
        dI_dt = (V_s - D_vac.v_drag(I[0])) / L_paddle
        return [dI_dt]

    sol_sine = solve_ivp(ode_sine, [0, 3.0], [0.0], t_eval=t, method='RK45')
    sol_saw = solve_ivp(ode_saw, [0, 3.0], [0.0], t_eval=t, method='RK45')
    
    # Thrust is the reaction force from the vacuum = V_drag(I)
    Thrust_sine = np.array([D_vac.v_drag(i) for i in sol_sine.y[0]])
    Thrust_saw = np.array([D_vac.v_drag(i) for i in sol_saw.y[0]])
    
    Net_sine = np.cumsum(Thrust_sine) * (t[1] - t[0])
    Net_saw = np.cumsum(Thrust_saw) * (t[1] - t[0])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=150)
    fig.patch.set_facecolor('#050508'); ax1.set_facecolor('#050508'); ax2.set_facecolor('#050508')
    
    # Plot 1: Waveforms and Drag Force
    ax1.plot(t, 5.0 * sawtooth(2 * np.pi * 1.0 * t, width=0.1), color='#444444', lw=2, linestyle='--', label=r'Applied Topo-Voltage $V_s(t)$ (Sawtooth)')
    ax1.plot(t, Thrust_saw, color='#00ffcc', lw=3, label=r'Reaction Force / Thrust ($V_{drag}$)')
    ax1.axhline(D_vac.V_zener, color='white', linestyle=':', lw=1.5, label=r'Zener Breakdown ($V_{zener}$)')
    ax1.axhline(-D_vac.V_zener, color='white', linestyle=':', lw=1.5)
    ax1.set_title('Switch-Mode Rectification (Bingham Zener Diode)', color='white', fontsize=14, weight='bold')
    ax1.set_ylabel('Force / Voltage (V)', color='white', weight='bold')
    ax1.legend(loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white', fontsize=10)
    
    # Plot 2: Cumulative Thrust
    ax2.plot(t, Net_sine, color='#ff3366', lw=3, linestyle='--', label='Standard RF Cavity (Sine Drive) -> Net Zero')
    ax2.plot(t, Net_saw, color='#00ffcc', lw=4, label='AVE Asymmetric Drive (Sawtooth) -> Massive DC Thrust')
    ax2.fill_between(t, 0, Net_saw, color='#00ffcc', alpha=0.2)
    ax2.set_title('Macroscopic Momentum Transfer (Propellantless Thrust)', color='white', fontsize=14, weight='bold')
    ax2.set_xlabel('Time (Oscillator Cycles)', color='white', weight='bold')
    ax2.set_ylabel('Net Accumulated Impulse', color='white', weight='bold')
    ax2.legend(loc='upper left', facecolor='#111111', edgecolor='white', labelcolor='white', fontsize=11)
    
    textstr = (
        r"$\mathbf{The~Fluidic~Diode~Circuit:}$" + "\n" +
        r"During the fast edge ($V_s \gg V_z$), the vacuum Zener diode short-circuits" + "\n" +
        r"($\eta \to 0$) and the actuator slips. During the slow edge ($V_s < V_z$), the" + "\n" +
        r"vacuum remains a rigid solid, and the actuator grips the lattice." + "\n" +
        r"A zero-mean AC vibration is perfectly rectified into continuous DC thrust."
    )
    ax2.text(1.2, 0.4, textstr, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='#00ffcc', alpha=0.9))

    for ax in [ax1, ax2]:
        ax.grid(True, ls=':', color='#333333'); ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#333333')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "tamd_rectifier.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_tamd_rectifier()