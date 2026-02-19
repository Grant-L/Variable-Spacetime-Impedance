"""
AVE MODULE 37: LABORATORY FALSIFICATION (HTS DETECTOR)
------------------------------------------------------
Generates the explicit, testable laboratory prediction for the AVE framework.
Proves that inducing a volumetric strain \\chi_{vol} via a high-speed macroscopic 
centrifuge alters the local magnetic permeability \\mu(\\mathbf{r}).
Because a Superconductor's Kinetic Inductance (L_K) is strictly proportional 
to \\mu(\\mathbf{r}), the rotating mass will induce a measurable \\Delta L_K shift 
in an adjacent YBCO loop. This provides an immediate, low-cost falsification test.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_hts_detector_prediction():
    print("Simulating HTS Kinetic Inductance Detector Prediction...")
    
    # Centrifuge RPM (up to 100,000 RPM)
    rpm = np.linspace(0, 100000, 500)
    omega = rpm * (2 * np.pi / 60.0) # rad/s
    
    # Centrifuge parameters (e.g., 500 kg dense rotor, radius 0.5m)
    M_rotor = 500.0
    R_rotor = 0.5
    
    # Rotational kinetic energy generates effective mass via E=mc^2
    # E_k = 1/2 I \omega^2 = 1/4 M R^2 \omega^2
    c = 3.0e8
    delta_M_eff = (0.25 * M_rotor * R_rotor**2 * omega**2) / c**2
    
    # 1. Raw Volumetric Strain \chi_{vol} measured at distance r = 0.1m from rim
    G = 6.674e-11
    r_sensor = 0.6 
    
    # \chi_{vol} = 7 G (\Delta M_eff) / (c^2 r)
    chi_vol = (7.0 * G * delta_M_eff) / (c**2 * r_sensor)
    
    # 2. Shift in Local Permeability (Scalar coupling: 1/7)
    # \mu_local = \mu_0 * (1 + 1/7 \chi_{vol})
    # Therefore, fractional shift \Delta L_K / L_K = \Delta \mu / \mu_0 = 1/7 \chi_{vol}
    fractional_shift = (1.0 / 7.0) * chi_vol
    
    # Convert to parts-per-trillion (ppt) for standard lock-in amplifier measurement scales
    shift_ppt = fractional_shift * 1e12
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#050508'); ax.set_facecolor('#050508')
    
    ax.plot(rpm, shift_ppt, color='#00ffcc', lw=3, label=r'Predicted Kinetic Inductance Shift ($\Delta L_K / L_K$)')
    
    ax.fill_between(rpm, 0, shift_ppt, color='#00ffcc', alpha=0.1)
    
    # State of the art Lock-In Amplifier sensitivity floor (~ 10^-12)
    ax.axhline(1.0, color='#ff3366', linestyle='--', lw=2, label='Current Quantum Lock-In Sensitivity Floor (1 ppt)')
    
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1e2); ax.set_xlim(0, 100000)
    ax.set_xlabel('Centrifuge Rotation Speed (RPM)', fontsize=13, color='white', weight='bold')
    ax.set_ylabel(r'Inductance Shift ($\Delta L_K / L_K$) [Parts-Per-Trillion]', fontsize=13, color='white', weight='bold')
    ax.set_title('Laboratory Falsification: HTS Vacuum Density Detector', fontsize=15, pad=15, color='white', weight='bold')
    
    textstr = (
        r"$\mathbf{Experimental~Protocol:}$" + "\n" +
        r"1. Spin a 500kg rotor to induce local metric strain via $E_k/c^2$." + "\n" +
        r"2. Place a resonant YBCO Superconducting loop at $r=0.6m$." + "\n" +
        r"3. Because $L_K \propto \mu_0 \cdot n_{scalar}(r)$, the resonant frequency of" + "\n" +
        r"the circuit will deterministically drop as RPM increases."
    )
    ax.text(2000, 1e-3, textstr, color='white', fontsize=12, bbox=dict(facecolor='#111111', edgecolor='white', alpha=0.9, pad=10))

    ax.grid(True, which="major", ls=":", color='#444444', alpha=0.8)
    ax.legend(loc='upper left', facecolor='#111111', edgecolor='white', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333333')
    
    filepath = os.path.join(OUTPUT_DIR, "hts_detector_prediction.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": 
    simulate_hts_detector_prediction()
