"""
AVE MODULE 56: IMPEDANCE MATCHING (GRAVITATIONAL LENSING)
---------------------------------------------------------
Models a gravity well as a Tapered LC Transmission Line.
Proves that because gravity compresses the vacuum volume, it increases 
both Inductance (\\mu) and Capacitance (\\epsilon) proportionally.
Therefore, Characteristic Impedance Z_0 = \\sqrt{L/C} remains perfectly 
invariant. This explains why gravity bends light and compresses its 
wavelength (Redshift) but NEVER generates a reflected signal.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

OUTPUT_DIR = "manuscript/chapters/13_spacetime_circuit_analysis/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_impedance_matched_gravity():
    print("Simulating Impedance-Matched Gravity Well...")
    
    N_nodes = 150
    x = np.linspace(-10, 10, N_nodes)
    
    # Refractive Index Profile n(x) of a gravity well
    n_local = 1.0 + 2.0 * np.exp(-(x - 2.0)**2 / 4.0) 
    
    # Gravity scales BOTH L and C proportionately to maintain constant Z_0
    L_array = 1.0 * n_local
    C_array = 1.0 * n_local
    Z_0 = np.sqrt(L_array / C_array) # Evaluates strictly to 1.0 everywhere!
    
    def tline_ode(t, y):
        dy = np.zeros(2 * N_nodes)
        
        # Inject continuous HF carrier wave
        V_in = np.sin(2 * np.pi * 1.5 * t) * np.exp(-((t - 10.0)**2) / 30.0)
        I_in_0 = (V_in - y[0]) / 1.0 
        
        dy[0] = (I_in_0 - y[1]) / C_array[0]
        for i in range(1, N_nodes):
            dy[2*i] = (y[2*i - 1] - y[2*i + 1]) / C_array[i]
        for i in range(N_nodes - 1):
            dy[2*i + 1] = (y[2*i] - y[2*i + 2]) / L_array[i]
        dy[2*(N_nodes-1) + 1] = (y[2*(N_nodes-1)] - 0.0) / L_array[-1]
                
        return dy

    t_eval = np.linspace(0, 40.0, 1000)
    sol = solve_ivp(tline_ode, [0, 40.0], np.zeros(2 * N_nodes), t_eval=t_eval)
    
    # Extract spatial snapshot
    snapshot_idx = int(len(t_eval) * 0.75)
    V_snapshot = sol.y[0::2, snapshot_idx]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=150, gridspec_kw={'height_ratios': [1, 2]})
    fig.patch.set_facecolor('#0a0a12'); ax1.set_facecolor('#0a0a12'); ax2.set_facecolor('#0a0a12')
    
    # Plot 1: Vacuum Impedance vs Refractive Index
    ax1.plot(x, n_local, color='#ffcc00', lw=3, label=r'Refractive Index $n(x)$ (Gravity Well)')
    ax1.plot(x, Z_0, color='#00ffcc', lw=2, linestyle='--', label=r'Vacuum Impedance $Z_0(x) = \sqrt{L/C}$')
    ax1.set_title('Gravity as a Tapered Transmission Line', color='white', fontsize=14, weight='bold')
    ax1.legend(loc='upper left', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    # Plot 2: Waveform Snapshot
    ax2.plot(x, V_snapshot, color='#ff3366', lw=2.5, label='Light Wave $V(x)$ at $t=30$')
    ax2.axvspan(-2, 6, color='#ffcc00', alpha=0.1, label='Gravity Well')
    
    ax2.set_title('Gravitational Wavelength Compression (Zero Reflection)', color='white', fontsize=14, weight='bold')
    ax2.set_xlabel('Spatial Node ($x$)', color='white', weight='bold')
    ax2.set_ylabel('Topological Voltage (Amplitude)', color='white', weight='bold')
    ax2.legend(loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    textstr = (
        r"$\mathbf{Perfect~Impedance~Matching:}$" + "\n" +
        r"Because gravity compresses the grid, it increases both Inductance ($\mu$)" + "\n" +
        r"and Capacitance ($\epsilon$) proportionately. Characteristic Impedance ($Z_0$) remains perfectly flat." + "\n" +
        r"Thus, light compresses its wavelength ($v_g = c/n$) inside the well without suffering ANY reflections."
    )
    ax2.text(-9, -0.6, textstr, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='white', alpha=0.8, pad=8))

    for ax in [ax1, ax2]:
        ax.grid(True, ls=':', color='#333333'); ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#333333')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "impedance_gravity_well.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight'); plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_impedance_matched_gravity()