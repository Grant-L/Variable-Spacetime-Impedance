import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_rlve_exact():
    print("Simulating Exact Parameter-Free RLVE Phase Shift...")
    rpm = np.linspace(0, 100000, 500)
    v_tan = (rpm * (2 * np.pi / 60)) * 0.1 # 10cm rotor
    c = 299792458.0
    rho_sat, rho_w, rho_al = 2.3e17, 19300.0, 2700.0
    
    # Exact AVE Prediction: v_fluid = v_tan * (rho / rho_sat)
    v_fluid_w = v_tan * (rho_w / rho_sat)
    v_fluid_al = v_tan * (rho_al / rho_sat)
    
    wavelength, L_cavity, Finesse = 1064e-9, 0.2, 10000
    L_eff = L_cavity * (2 * Finesse / np.pi) 
    phase_factor = (4 * np.pi * L_eff) / (wavelength * c)
    phase_shift_w = phase_factor * v_fluid_w
    phase_shift_al = phase_factor * v_fluid_al
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150); fig.patch.set_facecolor('#050508'); ax.set_facecolor('#050508')
    scale = 1e9 # nano-radians
    ax.plot(rpm, phase_shift_w * scale, color='#ff3366', lw=3, label=r'Tungsten ($\rho = 19.3$ g/cc): Signal')
    ax.plot(rpm, phase_shift_al * scale, color='#00ffcc', lw=3, linestyle='-.', label=r'Aluminum ($\rho = 2.7$ g/cc): Control')
    ax.plot(rpm, np.zeros_like(rpm), color='white', lw=2, linestyle='--', label=r'General Relativity (Null Frame-Dragging)')
    ax.text(80000, max(phase_shift_w * scale)*0.9, r"$\Psi \approx 7.1$ (AVE Signal)", color='#ff3366', weight='bold')
    
    ax.set_xlabel('Rotor Speed (RPM)', fontsize=12, color='white'); ax.set_ylabel(r'Phase Shift $\Delta\phi$ (nano-radians)', fontsize=12, color='white')
    ax.set_title('RLVE Exact Prediction: Fresnel-Fizeau Entrainment', fontsize=14, pad=15, color='white', weight='bold')
    ax.grid(True, which="both", ls="--", color='#333333', alpha=0.7); ax.tick_params(colors='white'); ax.legend(loc='upper left')
    
    filepath = os.path.join(OUTPUT_DIR, "fresnel_fizeau_rlve.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight'); plt.close(); print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_rlve_exact()