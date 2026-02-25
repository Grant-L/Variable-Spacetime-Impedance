import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

# Theoretical Constants
ALPHA = 1.0 / float("137.036") # Fine-structure constant
V_YIELD = 43650.0 # Volts, theoretical yield
Z_0 = float("376.73") # Impedance of free spaceuum impedance
EPSILON_0 = 8.854e-12

def calculate_effective_permittivity(V_applied):
    """Calculate the non-linear plateau of effective permittivity based on AVE Axiom 4."""
    # Prevent complex numbers past the yield limit, capping it
    V = np.clip(V_applied, 0, V_YIELD * 0.9999)
    return EPSILON_0 * np.sqrt(1 - (V / V_YIELD)**2)

def calculate_impedance(V_applied):
    """If epsilon drops, Z local spikes."""
    eps_eff = calculate_effective_permittivity(V_applied)
    return np.sqrt(4 * np.pi * 1e-7 / eps_eff)

def paschen_curve(p, d, gas='Air'):
    """
    Standard Paschen curve modeling for DC breakdown voltage.
    p = pressure in Torr
    d = gap distance in cm
    V_b = (B * p * d) / (ln(A * p * d) - ln(ln(1 + 1/gamma)))
    """
    # Rough parameters for air/nitrogen
    A = 15.0 # cm^-1 Torr^-1
    B = 365.0 # V cm^-1 Torr^-1
    gamma = 0.01
    
    pd = p * d
    # Avoid negative log domain
    valid_mask = (A * pd) > np.exp(np.log(1 + 1/gamma))
    
    V_breakdown = np.full_like(pd, np.inf)
    pd_valid = pd[valid_mask]
    
    # Paschen equation
    V_breakdown[valid_mask] = (B * pd_valid) / (np.log(A * pd_valid) - np.log(np.log(1 + 1/gamma)))
    return V_breakdown

def run_sensitivity_sweeps():
    print("[*] Running Sensitivity Studies for Induced Vacuum Impedance Mirror...")
    
    # 1. Vacuum Pressure Sensitivity (Paschen vs AVE Yield)
    pressures = np.logspace(-9, 3, 500) # 10^-9 Torr up to atmospheric
    gap_cm = 0.01 # 100 micron (0.1 mm) gap for extreme E-fields
    
    V_breakdown = paschen_curve(pressures, gap_cm)
    
    # We need to operate below Paschen breakdown, but ABOVE the AVE yield curve to see the anomaly.
    # The anomaly starts getting significant > 30kV.
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0f0f0f')
    
    for ax in axes:
        ax.set_facecolor('#0f0f0f')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('#333333')
            
    # Plot 1: Pressure
    axes[0].loglog(pressures, V_breakdown, color='#ff3333', linewidth=2, label='Paschen Curve (Air)')
    axes[0].axhline(V_YIELD, color='#00ffcc', linestyle='--', linewidth=2, label='AVE Absolute Yield ($43.65$ kV)')
    axes[0].axhline(30000, color='#ffff00', linestyle=':', label='Minimum SNR Threshold (30 kV)')
    axes[0].fill_between(pressures, 30000, V_YIELD, where=(V_breakdown > V_YIELD), color='#00ffcc', alpha=0.2, label='Viable Falsification Zone')
    
    axes[0].set_ylim(1e2, 1e6)
    axes[0].set_xlabel("Chamber Pressure (Torr)")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("Required Vacuum bounds ($100\\mu$m Gap)")
    leg = axes[0].legend(facecolor='#0f0f0f', edgecolor='none')
    for text in leg.get_texts(): text.set_color('white')
    
    # Plot 2: Local Impedance Spike
    v_sweep = np.linspace(0, 43600, 500)
    z_sweep = calculate_impedance(v_sweep)
    
    axes[1].plot(v_sweep/1000, z_sweep, color='#00ffcc', linewidth=2)
    axes[1].axhline(Z_0, color='#555555', linestyle='--', label='Standard Model $Z_0$')
    axes[1].set_xlabel("Applied Gap Voltage (kV)")
    axes[1].set_ylabel("Local Spacetime Impedance (Ohms)")
    axes[1].set_title("Metric Impedance vs HV Stress")
    
    # Plot 3: Laser Wavelength vs Diffraction Limit
    # To hit a 100 micron gap cleanly without scattering off the electrodes:
    wavelengths = np.linspace(200, 1500, 200) # nm, Deep UV to IR
    # Rayleigh criterion / Gaussian beam waist constraint
    # We want waist w_0 << gap_size
    gap_microns = gap_cm * 10000
    w_0 = 10.0 # 10 micron focal spot
    divergence_angle = (wavelengths * 1e-9) / (np.pi * w_0 * 1e-6) * 1000 # mrad
    
    axes[2].plot(wavelengths, divergence_angle, color='#ff00ff', linewidth=2)
    axes[2].axhline(gap_microns/100, color='#ff3333', linestyle='--', label='Electrode Interference Limit')
    axes[2].set_xlabel("Probe Wavelength (nm)")
    axes[2].set_ylabel("Beam Divergence (mrad)")
    axes[2].set_title("Probe Laser Constraint ($10\\mu$m Waist)")
    
    plt.tight_layout()
    
    # Save output
    project_root = pathlib.Path('/Users/grantlindblom/Variable-Spacetime-Impedance/Variable-Spacetime-Impedance')
    outdir = project_root / 'assets' / 'sim_outputs'
    os.makedirs(outdir, exist_ok=True)
    target = outdir / 'vacuum_mirror_sensitivities.png'
    
    plt.savefig(target, dpi=150, facecolor='#0f0f0f')
    print(f"[*] Visualized Vacuum Mirror Sensitivity Bounds: {target}")

if __name__ == "__main__":
    run_sensitivity_sweeps()
