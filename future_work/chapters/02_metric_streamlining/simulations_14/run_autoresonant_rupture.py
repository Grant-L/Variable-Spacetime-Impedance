import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp

OUTPUT_DIR = "manuscript/chapters/14_active_metric_engineering/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_autoresonance():
    w0, Q, F0 = 2 * np.pi * 1.0, 40.0, 0.08  
    
    def fixed_drive(t, y):
        x_clip = np.clip(y[0], -0.999, 0.999)
        restoring = w0**2 * y[0] * np.sqrt(1 - x_clip**4) 
        return [y[1], F0 * np.cos(w0 * t) - (w0 / Q) * y[1] - restoring]

    def auto_drive(t, y):
        x_clip = np.clip(y[0], -0.999, 0.999)
        restoring = w0**2 * y[0] * np.sqrt(1 - x_clip**4)
        return [y[1], F0 * np.tanh(y[1] * 1000) - (w0 / Q) * y[1] - restoring]

    t_span = [0, 50.0]; t_eval = np.linspace(0, 50, 5000)
    sol_fixed = solve_ivp(fixed_drive, t_span, [0.0, 0.0], t_eval=t_eval, rtol=1e-6)
    
    def hit_breakdown(t, y): return np.abs(y[0]) - 0.995
    hit_breakdown.terminal = True
    sol_auto = solve_ivp(auto_drive, t_span, [0.0, 0.0], t_eval=t_eval, events=hit_breakdown, rtol=1e-6)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor('#050508'); ax.set_facecolor('#050508')
    
    ax.plot(sol_fixed.t, np.abs(sol_fixed.y[0]), color='#ff3366', lw=2, label='Fixed Freq Laser (Detunes & Reflects)')
    ax.plot(sol_auto.t, np.abs(sol_auto.y[0]), color='#00ffcc', lw=3, label='Autoresonant Feedback Loop (Phase-Locked)')
    ax.axhline(1.0, color='white', lw=2, linestyle=':', label='Dielectric Snap (Schwinger Breakdown Limit)')
    ax.axhline(F0, color='gray', lw=1.5, linestyle='--', label=f'Raw Linear Input Power ({F0*100}%)')
    
    if len(sol_auto.t_events[0]) > 0:
        ax.scatter([sol_auto.t_events[0][0]], [1.0], color='yellow', s=150, zorder=10)
        ax.text(sol_auto.t_events[0][0]-7, 1.05, "DIELECTRIC RUPTURE", color='yellow', weight='bold')

    ax.set_ylim(0, 1.2); ax.set_xlim(0, 50)
    ax.set_xlabel('Time (Optical Cycles)', color='white', weight='bold', fontsize=12)
    ax.set_ylabel(r'Local Metric Strain Envelope ($\Delta\phi / \alpha$)', color='white', weight='bold', fontsize=12)
    ax.set_title('Autoresonant Dielectric Rupture (Non-Linear Vacuum Optics)', color='white', fontsize=15, weight='bold')
    
    textstr = (
        r"$\mathbf{The~Non{-}Linear~Detuning~Paradox:}$" + "\n" +
        r"Because the vacuum is a 4th-order non-linear capacitor (Axiom 4), its" + "\n" +
        r"resonant frequency drops under extreme optical stress. A fixed-frequency laser" + "\n" +
        r"detunes and is phase-rejected. By actively placing the vacuum in a" + "\n" +
        r"regenerative feedback loop, it acts as a non-linear Duffing Oscillator, reaching" + "\n" +
        r"the breakdown limit at a fraction of the traditionally calculated brute-force power."
    )
    ax.text(1.0, 0.4, textstr, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='#00ffcc', alpha=0.9, pad=10))

    ax.grid(True, ls=":", color='#444444'); ax.tick_params(colors='white')
    ax.legend(loc='lower right', facecolor='black', edgecolor='white', labelcolor='white')
    for spine in ax.spines.values(): spine.set_color('#333333')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "vacuum_tesla_coil.png"), facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

if __name__ == "__main__": simulate_autoresonance()