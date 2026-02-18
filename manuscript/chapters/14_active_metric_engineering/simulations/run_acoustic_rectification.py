import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import sawtooth
import os

OUTPUT_DIR = "manuscript/chapters/14_active_metric_engineering/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BinghamResistor:
    def __init__(self, V_yield, R_base, R_fluid, k_smooth=1000):
        self.V_yield = V_yield
        self.R_fluid = R_fluid
        self.k = k_smooth
        
    def v_drop(self, I):
        return self.V_yield * np.tanh(self.k * I) + self.R_fluid * I

def simulate_acoustic_rectification():
    t = np.linspace(0, 3, 5000)
    L_sys = 0.1 
    R_nonlin = BinghamResistor(V_yield=2.0, R_base=10.0, R_fluid=0.1)
    
    def ode_sine(t_val, I):
        V_s = 5.0 * np.sin(2 * np.pi * 1.0 * t_val)
        return [(V_s - R_nonlin.v_drop(I[0])) / L_sys]

    def ode_saw(t_val, I):
        V_s = 5.0 * sawtooth(2 * np.pi * 1.0 * t_val, width=0.1)
        return [(V_s - R_nonlin.v_drop(I[0])) / L_sys]

    sol_sine = solve_ivp(ode_sine, [0, 3.0], [0.0], t_eval=t, method='RK45')
    sol_saw = solve_ivp(ode_saw, [0, 3.0], [0.0], t_eval=t, method='RK45')
    
    Reaction_sine = np.array([R_nonlin.v_drop(i) for i in sol_sine.y[0]])
    Reaction_saw = np.array([R_nonlin.v_drop(i) for i in sol_saw.y[0]])
    
    Net_sine = np.cumsum(Reaction_sine) * (t[1] - t[0])
    Net_saw = np.cumsum(Reaction_saw) * (t[1] - t[0])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=150)
    fig.patch.set_facecolor('#0a0a12'); ax1.set_facecolor('#0a0a12'); ax2.set_facecolor('#0a0a12')
    
    ax1.plot(t, 5.0 * sawtooth(2 * np.pi * 1.0 * t, width=0.1), color='#444444', lw=2, linestyle='--', label=r'Applied Acoustic Stress $V_s(t)$')
    ax1.plot(t, Reaction_saw, color='#00ffcc', lw=3, label=r'Reaction Voltage ($V_{drop}$)')
    ax1.axhline(R_nonlin.V_yield, color='white', linestyle=':', lw=1.5, label=r'Bingham Yield Threshold ($\tau_{yield}$)')
    ax1.axhline(-R_nonlin.V_yield, color='white', linestyle=':', lw=1.5)
    ax1.set_title('Acoustic Rectification via Bingham-Plastic Yielding', color='white', fontsize=14, weight='bold')
    ax1.legend(loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    ax2.plot(t, Net_sine, color='#ff3366', lw=3, linestyle='--', label='Symmetric Sine Drive (Zero Net Momentum)')
    ax2.plot(t, Net_saw, color='#00ffcc', lw=4, label='Asymmetric Drive (Continuous DC Rectification)')
    ax2.fill_between(t, 0, Net_saw, color='#00ffcc', alpha=0.2)
    ax2.set_title('Time-Averaged Macroscopic Momentum Transfer', color='white', fontsize=14, weight='bold')
    ax2.set_xlabel('Time (Oscillator Cycles)', color='white', weight='bold')
    ax2.legend(loc='upper left', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    for ax in [ax1, ax2]:
        ax.grid(True, ls=':', color='#333333'); ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#333333')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "acoustic_rectification.png"), facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

if __name__ == "__main__": simulate_acoustic_rectification()