"""
AVE MODULE 57: VACUUM IMD SPECTROSCOPY
--------------------------------------
Simulates a tabletop dual-tone intermodulation test.
Drives the Axiom 4 Metric Varactor C(V) with two pure frequencies (f1, f2).
Because AVE strictly mandates a 4th-order dielectric saturation boundary,
the non-linear mixing generates a highly specific, mathematically rigorous
"Harmonic Fingerprint" (e.g., 3*f1 - 2*f2) in the local spatial metric.
This provides a binary falsification signature detectable by interferometry.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal.windows import blackmanharris
import os

OUTPUT_DIR = "manuscript/chapters/13_spacetime_circuit_analysis/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_vacuum_imd_spectroscopy():
    print("Simulating Vacuum Intermodulation Distortion (IMD) Spectrum...")
    
    # Dual-Tone Input
    f1, f2 = 1000.0, 1300.0  
    w1, w2 = 2 * np.pi * f1, 2 * np.pi * f2
    
    # ODE Solver for Non-Linear Varactor: dV/dt = I(t) / C(V)
    # Axiom 4: C(V) = C0 / sqrt(1 - (V/V_crit)^4)
    C0 = 1.0
    V_crit = 10.0 
    I_amp = 4.0   
    
    def varactor_ode(t, V):
        V_ratio = np.clip(V[0] / V_crit, -0.99, 0.99)
        C_eff = C0 / np.sqrt(1.0 - V_ratio**4)
        I_t = I_amp * (np.sin(w1 * t) + np.sin(w2 * t))
        return [I_t / C_eff]

    Fs = 20000.0 
    T_total = 2.0
    t_eval = np.linspace(0, T_total, int(Fs * T_total))
    
    sol = solve_ivp(varactor_ode, [0, T_total], [0.0], t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
    V_out = sol.y[0]
    
    # FFT Spectral Analysis
    window = blackmanharris(len(V_out))
    fft_vals = np.fft.rfft(V_out * window)
    fft_freqs = np.fft.rfftfreq(len(V_out), 1/Fs)
    
    fft_mag = np.abs(fft_vals)
    fft_mag_db = 20 * np.log10(fft_mag / np.max(fft_mag) + 1e-12)
    
    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    fig.patch.set_facecolor('#0a0a12'); ax.set_facecolor('#0a0a12')
    
    ax.plot(fft_freqs, fft_mag_db, color='#00ffcc', lw=1.5)
    
    # Annotate Fundamentals
    ax.scatter([f1, f2], [0, 0], color='#FFD54F', s=50, zorder=5)
    ax.text(f1-100, 5, "f1", color='#FFD54F', weight='bold')
    ax.text(f2+50, 5, "f2", color='#FFD54F', weight='bold')
    
    # AVE 5th-Order Intermodulation Products (3f1-2f2, etc.)
    imd1 = abs(3*f1 - 2*f2)  # 400 Hz
    imd2 = abs(3*f2 - 2*f1)  # 1900 Hz
    ax.axvline(imd1, color='#E57373', linestyle=':', lw=1.5)
    ax.axvline(imd2, color='#E57373', linestyle=':', lw=1.5)
    ax.text(imd1-350, -40, "AVE Signature\n(3f1 - 2f2)", color='#E57373')
    ax.text(imd2+50, -40, "AVE Signature\n(3f2 - 2f1)", color='#E57373')
    
    ax.set_xlim(0, 3000); ax.set_ylim(-140, 10)
    ax.set_xlabel('Frequency (Hz)', color='white', weight='bold')
    ax.set_ylabel('Magnitude (dBc)', color='white', weight='bold')
    ax.set_title('Vacuum IMD Spectroscopy: The Axiom 4 Harmonic Fingerprint', color='white', fontsize=14, weight='bold')
    
    textstr = (
        r"$\mathbf{The~Non{-}Linear~Fingerprint:}$" + "\n" +
        r"Because standard mechanical materials feature 2nd or 3rd-order elasticity," + "\n" +
        r"and AVE mandates an exact 4th-order vacuum dielectric limit ($1 - V^4$)," + "\n" +
        r"the vacuum acts as a unique RF mixer. It mathematically generates" + "\n" +
        r"distinct 5th-order sidebands, completely isolated from normal material noise."
    )
    ax.text(1400, -20, textstr, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='#4FC3F7', alpha=0.9, pad=10))

    ax.grid(True, ls=':', color='#333333'); ax.tick_params(colors='lightgray')
    for spine in ax.spines.values(): spine.set_color('#333333')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "vacuum_imd_spectroscopy.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_vacuum_imd_spectroscopy()