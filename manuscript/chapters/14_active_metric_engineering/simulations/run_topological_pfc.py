import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import os

OUTPUT_DIR = "manuscript/chapters/14_active_metric_engineering/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_active_metric_pfc():
    xi_topo, V_yield = 4.149e-7, 60e3       
    V_source, L, R_parasitic = 60e3, 10e-3, 50.0   
    t = np.linspace(0, 15e-6, 1000)
    
    I_passive = (V_source / R_parasitic) * (1 - np.exp(-R_parasitic * t / L))
    V_L_passive = L * np.gradient(I_passive, t)
    V_L_active = np.full_like(t, V_yield * 0.99)
    I_active = (V_L_active[0] / L) * t
    
    k_standard = 0.15 # H = 0
    k_hopf = 0.95     # H != 0 (Beltrami)
    
    Thrust_dumb = V_L_passive * xi_topo * k_standard
    Thrust_smart = V_L_active * xi_topo * k_hopf
    Thrust_Multiplier = trapezoid(Thrust_smart, t) / trapezoid(Thrust_dumb, t)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), dpi=150)
    fig.patch.set_facecolor('#0a0a12'); ax1.set_facecolor('#0a0a12'); ax2.set_facecolor('#0a0a12')
    
    ax1.plot(t * 1e6, I_passive, color='#E57373', lw=3, linestyle='--', label='Passive Actuator (Exponential Decay)')
    ax1.plot(t * 1e6, I_active, color='#4FC3F7', lw=4, label='Active Waveform Shaping (Linear Ramp)')
    ax1.set_title('Temporal Shaping: Active Metric PFC Profile', color='white', fontsize=14, weight='bold')
    ax1.set_ylabel('Current (Amperes)', color='white', weight='bold')
    ax1.legend(loc='upper left', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    ax2.plot(t * 1e6, Thrust_dumb * 1e3, color='#E57373', lw=3, linestyle='--', label=r'Standard Toroid ($\mathbf{A} \perp \mathbf{B}$, $k=0.15$)')
    ax2.plot(t * 1e6, Thrust_smart * 1e3, color='#00ffcc', lw=4, label=r'Hopf Coil ($\mathbf{A} \parallel \mathbf{B}$, $k=0.95$)')
    ax2.fill_between(t * 1e6, 0, Thrust_smart * 1e3, color='#00ffcc', alpha=0.15)
    ax2.fill_between(t * 1e6, 0, Thrust_dumb * 1e3, color='#E57373', alpha=0.3)
    ax2.axhline((V_yield * xi_topo) * 1e3, color='#FFD54F', linestyle=':', lw=2, label='Bingham Slip Boundary')
    
    ax2.set_title(f'Chiral Impedance Matching (Coupling Multiplier: {Thrust_Multiplier:.1f}x)', color='white', fontsize=14, weight='bold')
    ax2.set_xlabel(r'Time ($\mu$s)', color='white', weight='bold')
    ax2.set_ylabel('Net Coupled Topological Force (mN)', color='white', weight='bold')
    ax2.legend(loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    textstr = (
        r"$\mathbf{Topological~Power~Factor~Correction~({TPFC}):}$" + "\n" +
        r"1. $\mathbf{Temporal:}$ Active PFC forces a linear ramp ($di/dt = const$)." + "\n" +
        r"This holds the stress flat at exactly $99\%$ of the condensate yield limit." + "\n" +
        r"2. $\mathbf{Spatial:}$ The Beltrami geometry aligns $\mathbf{A} \parallel \mathbf{B}$, injecting Kinetic Helicity." + "\n" +
        r"This matches the chiral Cosserat eigenmode, eliminating Reactive Power bounce."
    )
    ax2.text(0.5, 4, textstr, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='#4FC3F7', alpha=0.9, pad=10))

    for ax in [ax1, ax2]:
        ax.grid(True, ls=':', color='#333333'); ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#333333')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "topological_pfc.png"), facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

if __name__ == "__main__": simulate_active_metric_pfc()