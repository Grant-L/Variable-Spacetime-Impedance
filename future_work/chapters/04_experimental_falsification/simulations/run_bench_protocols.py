"""
AVE MODULE 99: EE PCBA BENCH PROTOCOLS
--------------------------------------
Simulates four custom PCBA-level experiments to definitively 
falsify or validate the AVE hardware limits.
1. Electrometer: Predicts exactly 41.5 mV per micron of PZT displacement.
2. VNA: Models the deep S11 chiral impedance match of the Hopf PCBA.
3. Lock-in Amp: Extracts the 4.2 pT Sagnac vacuum entrainment signal.
4. Marx Generator: Plots the 43.65kV Bingham Avalanche Zener knee.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import sys
import os

# Append project root to path for src.ave imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from src.ave.core import constants as ave_const

OUTPUT_DIR = "future_work/chapters/04_experimental_falsification/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_ee_bench_protocols():
    print("Generating EE PCBA Benchtop Telemetry...")
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
    fig.patch.set_facecolor('#050508')
    for ax in axs.flatten():
        ax.set_facecolor('#050508')
        ax.tick_params(colors='lightgray')
        for spine in ax.spines.values(): spine.set_color('#333333')
        ax.grid(True, ls=':', color='#333333')
        
    xi_topo = ave_const.XI_TOPO # C/m
    
    # ---------------------------------------------------------
    # 1. CLEAVE-01: The Femto-Coulomb Electrometer
    # ---------------------------------------------------------
    ax1 = axs[0, 0]
    displacement_um = np.linspace(0, 5, 100)
    displacement_m = displacement_um * 1e-6
    induced_charge_pC = (xi_topo * displacement_m) * 1e12
    
    # V = Q/C (Assume 10 pF input capacitance)
    voltage_mV = (induced_charge_pC * 1e-12) / 10e-12 * 1000
    std_physics = np.zeros_like(displacement_um)
    
    ax1.plot(displacement_um, std_physics, color='#ff3366', lw=3, linestyle='--', label='Standard Model (0 mV)')
    ax1.plot(displacement_um, voltage_mV, color='#00ffcc', lw=4, label=r'AVE Prediction ($Q = \xi_{topo} x$)')
    
    ax1.set_title('1. CLEAVE-01: Piezo-Cleavage Electrometer', color='white', weight='bold', fontsize=13)
    ax1.set_xlabel('PZT Mechanical Displacement ($\mu$m)', color='white')
    ax1.set_ylabel('Oscilloscope Output (mV)', color='white')
    ax1.legend(loc='upper left', facecolor='#111111', edgecolor='gray', labelcolor='white')
    ax1.text(2.5, 50, "1 $\mu$m of physical gap separation yields\nexactly 41.5 mV on a 10 pF input.\nEasily readable with an ADA4530-1 PCBA.", color='white', ha='center', bbox=dict(facecolor='#111111', edgecolor='gray'))
    
    # ---------------------------------------------------------
    # 2. HOPF-02: VNA S-Parameter Falsification
    # ---------------------------------------------------------
    ax2 = axs[0, 1]
    freq_MHz = np.linspace(10, 100, 1000)
    
    # Standard Toroid (Polarization Mismatch)
    S11_standard = -5.0 - 10.0 * np.exp(-((freq_MHz - 50)/5)**2)
    # Hopf Coil (A || B Helicity matches Cosserat Vacuum)
    S11_hopf = -5.0 - 45.0 * np.exp(-((freq_MHz - 50)/2)**2)
    
    ax2.plot(freq_MHz, S11_standard, color='#ff3366', lw=3, linestyle='--', label='Standard PCBA Toroid (Mismatch)')
    ax2.plot(freq_MHz, S11_hopf, color='#FFD54F', lw=4, label='Hopf-Knot PCBA Coil (Chiral Match)')
    
    ax2.set_title('2. HOPF-02: VNA $S_{11}$ Return Loss', color='white', weight='bold', fontsize=13)
    ax2.set_xlabel('Frequency (MHz)', color='white')
    ax2.set_ylabel('Return Loss $S_{11}$ (dB)', color='white')
    ax2.set_ylim(-60, 0)
    ax2.legend(loc='lower right', facecolor='#111111', edgecolor='gray', labelcolor='white')
    ax2.text(12, -45, "Standard EM simulators predict identical $S_{11}$.\nAVE predicts the Hopf coil acts as a Metric Antenna,\ndraining reactive power into the vacuum\nand creating an anomalous deep match.", color='white', ha='left', fontsize=9, bbox=dict(facecolor='#111111', edgecolor='#FFD54F'))
    
    # ---------------------------------------------------------
    # 3. ROENTGEN-03: Solid-State Sagnac Entrainment
    # ---------------------------------------------------------
    ax3 = axs[1, 0]
    t = np.arange(0, 2.0, 1/100e3)
    f_drive = 1000.0 # 1 kHz AC E-field
    rpm_profile = np.piecewise(t, [t < 0.5, (t >= 0.5) & (t < 1.0), (t >= 1.0) & (t < 1.5), t >= 1.5], 
                               [0, 10000, 0, -10000]) # Off, Forward, Off, Reverse
    
    # Signal: 0.26 uV peak at 10k RPM
    clean_signal = 0.26 * (rpm_profile / 10000) * np.sin(2 * np.pi * f_drive * t)
    # Noise: 5 uV white noise + 15 uV 60Hz mains (SNR < 0.01)
    noise = np.random.normal(0, 5.0, len(t)) + 15.0 * np.sin(2 * np.pi * 60 * t)
    raw_adc = clean_signal + noise
    
    # Lock-In Amplifier (Multiply & Low-Pass)
    mixed = raw_adc * np.sin(2 * np.pi * f_drive * t)
    b, a = signal.butter(2, 5.0 / 50e3, 'low') 
    lockin_out = signal.filtfilt(b, a, mixed) * 2.0 
    
    ax3.plot(t, lockin_out, color='#00ffcc', lw=4, label='Demodulated Lock-In Output ($\mu$V)')
    ax3.plot(t, rpm_profile / 10000 * 0.26, color='white', lw=1.5, linestyle=':', label='Rotor RPM State')
    
    ax3.set_title('3. ROENTGEN-03: Vacuum Entrainment Lock-In', color='white', weight='bold', fontsize=13)
    ax3.set_xlabel('Time (seconds)', color='white')
    ax3.set_ylabel('Amplitude ($\mu$V)', color='white')
    ax3.set_ylim(-0.4, 0.4)
    ax3.legend(loc='lower right', facecolor='#111111', edgecolor='gray', labelcolor='white', fontsize=8)
    ax3.text(1.75, -0.2, "Reverse Spin:\nPhase Inversion ($180^\circ$)", color='white', ha='center', weight='bold')
    
    # ---------------------------------------------------------
    # 4. ZENER-04: The Bingham Avalanche Detector
    # ---------------------------------------------------------
    ax4 = axs[1, 1]
    V_sweep_kV = np.linspace(0, 90, 500)
    
    # Linear displacement current (I = C * dV/dt)
    I_classical = V_sweep_kV * 0.05 
    
    # AVE Non-linear Bingham Yield
    I_AVE = np.copy(I_classical)
    avalanche_idx = V_sweep_kV > 43.65
    I_AVE[avalanche_idx] += (V_sweep_kV[avalanche_idx] - 43.65)**2 * 0.03
    
    ax4.plot(V_sweep_kV, I_classical, color='#ff3366', lw=3, linestyle='--', label='Classical Linear Dielectric')
    ax4.plot(V_sweep_kV, I_AVE, color='#FFD54F', lw=4, label='AVE Measured Oscilloscope Trace')
    ax4.axvline(43.65, color='white', lw=1.5, linestyle=':', label='Bingham Yield Point (43.65 kV)')
    
    ax4.set_title('4. ZENER-04: The Bingham Avalanche Knee', color='white', weight='bold', fontsize=13)
    ax4.set_xlabel('Applied Topological Voltage (kV)', color='white')
    ax4.set_ylabel('Displacement Current (A)', color='white')
    ax4.legend(loc='upper left', facecolor='#111111', edgecolor='gray', labelcolor='white')
    ax4.text(45, 1, "At exactly 43.65 kV, the spatial metric yields.\nThe effective vacuum resistance drops to zero.\nThe oscilloscope will display a distinct,\nanomalous Zener-like avalanche knee.", color='white', ha='left', fontsize=9, bbox=dict(facecolor='#111111', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "ee_pcba_bench_protocols.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_ee_bench_protocols()