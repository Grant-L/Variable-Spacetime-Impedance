import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/14_active_metric_engineering/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_inertial_cancellation():
    xi_topo = 4.149e-7 
    t = np.linspace(0, 1.0, 5000); dt = t[1] - t[0]
    
    a_shock = np.zeros_like(t)
    in_maneuver = (t >= 0.4) & (t <= 0.6)
    a_shock[in_maneuver] = -500.0 * 9.81 * np.sin(np.pi * (t[in_maneuver] - 0.4) / 0.2)
    v_kinematic = 4000.0 + np.cumsum(a_shock) * dt 
    v_kinematic[v_kinematic < 0] = 0.0
    
    mass_test = 80.0 
    F_unmitigated = mass_test * a_shock
    G_unmitigated = np.abs(F_unmitigated / (mass_test * 9.81))
    
    V_vacuum_spike = -(1.0 / xi_topo) * F_unmitigated 
    
    lag_steps = int(0.002 / dt)
    V_snubber = np.zeros_like(V_vacuum_spike)
    V_snubber[lag_steps:] = -V_vacuum_spike[:-lag_steps] 
    V_snubber[in_maneuver] *= (1.0 + 0.05 * np.sin(2 * np.pi * 200 * t[in_maneuver]))
    
    V_net = V_vacuum_spike + V_snubber
    F_mitigated = -V_net * xi_topo
    G_mitigated = np.abs(F_mitigated / (mass_test * 9.81))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), dpi=150, gridspec_kw={'height_ratios': [1, 1.5, 1.5]})
    fig.patch.set_facecolor('#050508')
    
    ax1.set_facecolor('#050508')
    ax1.plot(t, v_kinematic / 1000.0, color='#00ffcc', lw=3)
    ax1.set_ylabel('Boundary Velocity\n(km/s)', color='white', weight='bold')
    ax1.set_title('Active Inertial Cancellation via Transient Metric Snubbers', color='white', fontsize=16, weight='bold')
    
    ax2.set_facecolor('#050508')
    ax2.plot(t, V_vacuum_spike / 1e9, color='#ff3366', lw=3, label='Condensate Inductive Spike (G-Force Mechanism)')
    ax2.plot(t, V_snubber / 1e9, color='#ffff00', lw=2, linestyle='--', label='Active Metric Snubber (CEMF Injection)')
    ax2.plot(t, V_net / 1e9, color='white', lw=1.5, label='Net Topological Voltage across Internal Mass')
    ax2.set_ylabel('Topological Potential\n(GigaVolts)', color='white', weight='bold')
    ax2.legend(loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    ax3.set_facecolor('#050508')
    ax3.fill_between(t, 0, G_unmitigated, color='#ff3366', alpha=0.3, label='Unmitigated Structural Shock')
    ax3.plot(t, G_unmitigated, color='#ff3366', lw=2)
    ax3.fill_between(t, 0, G_mitigated, color='#00ffcc', alpha=0.7, label='Mitigated Internal Field (Safe)')
    ax3.plot(t, G_mitigated, color='#00ffcc', lw=2)
    ax3.axhline(9.0, color='gray', linestyle=':', lw=2, label='Biological Blackout Limit (9 Gs)')
    
    ax3.set_xlabel('Time (seconds)', color='white', weight='bold', fontsize=12)
    ax3.set_ylabel('Internal\nAcceleration (Gs)', color='white', weight='bold')
    ax3.legend(loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    textstr = (
        r"$\mathbf{The~Circuit~Analogy:}$" + "\n" +
        r"$V_{spike} = -L_m \frac{di_m}{dt} \equiv -\xi_{topo}^{-1} (m a_{kinematic})$" + "\n" +
        r"An active array acts as an electrical Snubber, injecting a" + "\n" +
        r"destructive CEMF transient that shunts the inductive" + "\n" +
        r"vacuum wake, dropping internal G-forces to near-zero."
    )
    ax2.text(0.02, 0.45, textstr, transform=ax2.transAxes, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='#00ffcc', alpha=0.9))

    for ax in [ax1, ax2, ax3]: 
        ax.grid(True, ls=':', color='#333333')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#333333')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "inertial_damping_snubber.png"), facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

if __name__ == "__main__": simulate_inertial_cancellation()