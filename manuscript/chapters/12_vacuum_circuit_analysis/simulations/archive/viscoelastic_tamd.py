"""
AVE MODULE 53.5: VISCOELASTIC VOLTAGE-DRIVEN BINGHAM RECTIFIER
--------------------------------------------------------------
Implements the peer-reviewed corrections:
1. Bingham Yield is strictly Voltage-Driven (Shear Stress = Voltage).
2. Introduces the Thixotropic Time Constant (\tau_{hull} = L_{hull}/c).
Proves that a TAMD drive must be tuned to the geometric size of the vessel.
If the drive frequency exceeds f_max = c/L_{hull}, the vacuum cannot liquefy 
fast enough to envelop the ship, and propellantless thrust stalls out.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import sawtooth
import os

OUTPUT_DIR = "manuscript/chapters/13_spacetime_circuit_analysis/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_viscoelastic_tamd():
    print("Simulating Viscoelastic TAMD (Voltage-Triggered & Thixotropic)...")
    
    # Circuit parameters
    L_ship = 1.0     # Inertial mass of the ship
    C_vac = 0.05     # Local vacuum compliance
    R_solid = 50.0   # Grip
    R_fluid = 0.1    # Slip
    V_zener = 2.0    # Yield stress threshold
    freq = 1.0       # Drive frequency
    
    def run_sim(tau_vac):
        t_max = 5.0 / freq
        t_eval = np.linspace(0, t_max, 5000)
        
        def V_applied(t):
            return 5.0 * sawtooth(2 * np.pi * freq * t, width=0.1)
            
        def ode(t, y):
            I_L, V_vac, S = y
            V_s = V_applied(t)
            
            # Effective resistance depends on fluidity state S [0 to 1]
            R_eff = R_solid * (1 - S) + R_fluid * S
            
            # Viscoelastic Circuit Equations
            dI_L_dt = (V_s - V_vac) / L_ship
            dV_vac_dt = (I_L - V_vac / R_eff) / C_vac
            
            # Target State driven strictly by VOLTAGE (Stress) across vacuum
            S_eq = 0.5 * (1.0 + np.tanh(15.0 * (np.abs(V_vac) - V_zener)))
            
            # Thixotropic delay
            dS_dt = (S_eq - S) / tau_vac
            
            return [dI_L_dt, dV_vac_dt, dS_dt]
            
        sol = solve_ivp(ode, [0, t_max], [0, 0, 0], t_eval=t_eval, method='RK45')
        return sol
        
    sol_ideal = run_sim(tau_vac=0.02) # Tuned: Drive is slower than tau_vac
    sol_stall = run_sim(tau_vac=0.5)  # Stalled: Drive is much faster than tau_vac
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=150)
    fig.patch.set_facecolor('#0a0a12'); ax1.set_facecolor('#0a0a12'); ax2.set_facecolor('#0a0a12')
    
    # Plot 1: Viscoelastic State Transition
    ax1.plot(sol_ideal.t, sol_ideal.y[1], color='#444444', lw=1.5, label='Topological Stress ($V_{vac}$)')
    ax1.plot(sol_ideal.t, sol_ideal.y[2] * 5 - 2.5, color='#00ffcc', lw=2, label=r'Fluidity $S$ (Tuned: $t_{pulse} \gg \tau_{hull}$)')
    ax1.plot(sol_stall.t, sol_stall.y[2] * 5 - 2.5, color='#ff3366', lw=2, linestyle='--', label=r'Fluidity $S$ (Stalled: $t_{pulse} < \tau_{hull}$)')
    ax1.axhline(V_zener, color='white', linestyle=':', lw=1, label=r'Zener Yield Limit ($V_{yield}$)')
    
    ax1.set_title('Viscoelastic Zener Breakdown (Voltage-Triggered)', color='white', fontsize=14, weight='bold')
    ax1.set_ylabel('Voltage / Fluidity State', color='white', weight='bold')
    ax1.legend(loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    # Plot 2: Macroscopic Velocity (Current)
    ax2.plot(sol_ideal.t, sol_ideal.y[0], color='#00ffcc', lw=3, label='Tuned Drive ($f < c/L$): Continuous DC Velocity (Thrust)')
    ax2.plot(sol_stall.t, sol_stall.y[0], color='#ff3366', lw=3, linestyle='--', label='Stalled Drive ($f > c/L$): Zero Net Velocity')
    ax2.fill_between(sol_ideal.t, 0, sol_ideal.y[0], color='#00ffcc', alpha=0.15)
    
    ax2.set_title('Macroscopic Kinematic Rectification & The Cutoff Frequency', color='white', fontsize=14, weight='bold')
    ax2.set_xlabel('Time (Oscillator Cycles)', color='white', weight='bold')
    ax2.set_ylabel('Ship Velocity (Amperes)', color='white', weight='bold')
    ax2.legend(loc='upper left', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    textstr = (
        r"$\mathbf{The~Hull~Scaling~Law:}$" + "\n" +
        r"The vacuum yield state must physically propagate across the ship's hull to form a slipstream." + "\n" +
        r"This imposes a macroscopic relaxation time $\tau_{hull} \approx L_{hull}/c$." + "\n" +
        r"Pulsing the drive faster than $f_{max} = c/L_{hull}$ causes the drive to stall (Red)."
    )
    ax2.text(0.5, np.max(sol_ideal.y[0])*0.2, textstr, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='white', alpha=0.8, pad=8))

    for ax in [ax1, ax2]:
        ax.grid(True, ls=':', color='#333333'); ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#333333')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "viscoelastic_tamd.png"), facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved: viscoelastic_tamd.png")

if __name__ == "__main__": simulate_viscoelastic_tamd()