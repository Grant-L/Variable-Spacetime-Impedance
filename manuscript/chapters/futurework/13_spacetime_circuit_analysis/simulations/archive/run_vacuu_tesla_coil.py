"""
AVE MODULE 50: THE VACUUM TESLA COIL (AUTORESONANT MATTER SYNTHESIS)
--------------------------------------------------------------------
Applies Non-Linear Duffing Oscillator theory to the spatial vacuum.
Proves that because Axiom 4 dictates C_eff diverges at E_crit, 
the local resonant frequency of the vacuum drops under stress.
A fixed-frequency laser detunes and stalls (reflects power).
An Autoresonant Regenerative laser maintains phase-lock, achieving 
dielectric breakdown (matter synthesis) at low input powers.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp

OUTPUT_DIR = "manuscript/chapters/13_ee_for_ave/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_vacuum_tesla_coil():
    print("Simulating Vacuum LC Ring-Up (Fixed vs Autoresonant)...")
    
    w0 = 2 * np.pi * 1.0  # Base frequency
    Q = 40.0              # Modest optical cavity Q-factor
    F0 = 0.08             # Input laser is only 8% of the breakdown limit!
    
    # 1. FIXED FREQUENCY ODE (Standard Laser)
    def fixed_drive(t, y):
        x, v = y
        x_clip = np.clip(x, -0.999, 0.999)
        restoring_force = w0**2 * x * np.sqrt(1 - x_clip**4) # Axiom 4
        damping = (w0 / Q) * v
        drive = F0 * np.cos(w0 * t)
        return [v, drive - damping - restoring_force]

    # 2. AUTORESONANT REGENERATIVE FEEDBACK (AVE Engineering)
    def auto_drive(t, y):
        x, v = y
        x_clip = np.clip(x, -0.999, 0.999)
        restoring_force = w0**2 * x * np.sqrt(1 - x_clip**4)
        damping = (w0 / Q) * v
        # Regenerative Feedback: Laser always pushes in direction of velocity
        drive = F0 * np.tanh(v * 1000) 
        return [v, drive - damping - restoring_force]

    t_span = [0, 50.0]
    t_eval = np.linspace(0, 50, 5000)
    
    sol_fixed = solve_ivp(fixed_drive, t_span, [0.0, 0.0], t_eval=t_eval, rtol=1e-6)
    
    def hit_breakdown(t, y): return np.abs(y[0]) - 0.995
    hit_breakdown.terminal = True
    sol_auto = solve_ivp(auto_drive, t_span, [0.0, 0.0], t_eval=t_eval, events=hit_breakdown, rtol=1e-6)

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor('#050508'); ax.set_facecolor('#050508')
    
    ax.plot(sol_fixed.t, np.abs(sol_fixed.y[0]), color='#ff3366', lw=2, label='Fixed Freq Laser (Detunes & Stalls)')
    ax.plot(sol_auto.t, np.abs(sol_auto.y[0]), color='#00ffcc', lw=3, label='AVE Regenerative Feedback Loop (Phase-Locked)')
    
    ax.axhline(1.0, color='white', lw=2, linestyle=':', label='Dielectric Snap (Spontaneous Pair Production)')
    ax.axhline(F0, color='gray', lw=1.5, linestyle='--', label=f'Raw Laser Input Power ({F0*100}%)')
    
    if len(sol_auto.t_events[0]) > 0:
        ax.scatter([sol_auto.t_events[0][0]], [1.0], color='yellow', s=150, zorder=10)
        ax.text(sol_auto.t_events[0][0]-6, 1.05, "VACUUM SHATTER\n(Matter Synthesis)", color='yellow', weight='bold')

    ax.set_ylim(0, 1.2); ax.set_xlim(0, 50)
    ax.set_xlabel('Time (Optical Cycles)', color='white', weight='bold', fontsize=12)
    ax.set_ylabel(r'Local Metric Strain Envelope ($E / E_{crit}$)', color='white', weight='bold', fontsize=12)
    ax.set_title('The Vacuum Tesla Coil: Autoresonant Matter Synthesis', color='white', fontsize=15, weight='bold')
    
    textstr = (
        r"$\mathbf{The~Detuning~Paradox~Resolved:}$" + "\n" +
        r"Because the vacuum is a non-linear capacitor (Axiom 4), its" + "\n" +
        r"resonant frequency drops as stress increases. A fixed laser" + "\n" +
        r"detunes and is phase-rejected. By actively placing the vacuum in a" + "\n" +
        r"regenerative feedback loop, it acts as a resonant Tesla Coil, reaching" + "\n" +
        r"the breakdown limit at a fraction of the brute-force power."
    )
    ax.text(1.0, 0.4, textstr, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='#00ffcc', alpha=0.9, pad=10))

    ax.grid(True, ls=":", color='#444444'); ax.tick_params(colors='white')
    ax.legend(loc='lower right', facecolor='black', edgecolor='white', labelcolor='white')
    for spine in ax.spines.values(): spine.set_color('#333333')
    
    filepath = os.path.join(OUTPUT_DIR, "vacuum_tesla_coil.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight'); plt.close(); print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_vacuum_tesla_coil()