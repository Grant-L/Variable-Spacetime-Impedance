import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
OUTPUT_DIR = "manuscript/chapters/12_experimental_falsification/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Physics Constants ---
C = 299792458.0           # Speed of light (m/s)
ALPHA = 1.0 / 137.035999  # Fine Structure Constant (Max Viscosity Limit)

def calculate_viscous_shift(rpm, density, radius, length, wavelength, finesse):
    """
    Calculates the VSI Viscous Drag Phase Shift with Density Scaling.
    
    Formula: delta_n = alpha * (rho / rho_sat) * (v/c)^2
    
    This introduces the 'Mass Saturation Factor' required to distinguish
    gravitational drag from aerodynamic noise.
    """
    # Reference Density (Saturation Limit ~ Tungsten)
    RHO_SAT = 19.3 
    
    # 1. Kinematics
    omega = rpm * (2 * np.pi / 60.0) 
    v_tan = omega * radius
    
    # 2. Lattice Viscosity (Density Corrected)
    # The coupling efficiency scales with mass density relative to saturation
    # eta = alpha * (rho_rotor / rho_sat)
    coupling_efficiency = density / RHO_SAT
    
    # Clamp efficiency at 1.0 (Saturation)
    if coupling_efficiency > 1.0: 
        coupling_efficiency = 1.0 
    
    eta = ALPHA * coupling_efficiency
    delta_n = eta * (v_tan / C)**2
    
    # 3. Optical Phase Shift
    # phi = F * (2*pi*L / lambda) * delta_n
    phi_single = (2 * np.pi * length / wavelength) * delta_n
    
    # 4. Cavity Amplification
    phi_total = phi_single * finesse
    
    return phi_total * 1000.0  # Convert to milli-radians

def run_simulation():
    print("--- VSI RLVE Prediction Model (v2 - Density Corrected) ---")
    
    # --- Experimental Parameters ---
    R_WHEEL = 0.1         # 10 cm Radius
    L_PATH = 0.2          # 20 cm Interaction Length
    LAMBDA = 1.55e-6      # 1550 nm Laser
    FINESSE = 10000       # Cavity Finesse
    TARGET_RPM = 100000   # Max Speed
    
    # Materials (g/cm^3)
    RHO_TUNGSTEN = 19.3
    RHO_ALUMINUM = 2.7
    
    # Calculate Points at Target Speed
    pred_W = calculate_viscous_shift(TARGET_RPM, RHO_TUNGSTEN, R_WHEEL, L_PATH, LAMBDA, FINESSE)
    pred_Al = calculate_viscous_shift(TARGET_RPM, RHO_ALUMINUM, R_WHEEL, L_PATH, LAMBDA, FINESSE)
    
    print(f"Target RPM: {TARGET_RPM}")
    print(f"Tungsten Signal (Signal): {pred_W:.4f} mrad")
    print(f"Aluminum Signal (Control): {pred_Al:.4f} mrad")
    print(f"Signal Ratio: {pred_W/pred_Al:.2f} (Target ~7.1)")
    
    # Generate Plot Curves
    rpms = np.linspace(0, 120000, 100)
    shifts_W = [calculate_viscous_shift(r, RHO_TUNGSTEN, R_WHEEL, L_PATH, LAMBDA, FINESSE) for r in rpms]
    shifts_Al = [calculate_viscous_shift(r, RHO_ALUMINUM, R_WHEEL, L_PATH, LAMBDA, FINESSE) for r in rpms]
    
    plt.figure(figsize=(10, 6))
    
    # Plot Tungsten (Signal)
    plt.plot(rpms, shifts_W, color='#E76F51', linewidth=3, label=r'Tungsten ($\rho=19.3$): Signal')
    
    # Plot Aluminum (Control)
    plt.plot(rpms, shifts_Al, color='#2A9D8F', linewidth=2, linestyle='-.', label=r'Aluminum ($\rho=2.7$): Control')
    
    # Plot GR (Null Result)
    plt.plot(rpms, np.zeros_like(rpms), 'k--', label='General Relativity (Null)')
    
    # Markers
    plt.axvline(x=TARGET_RPM, color='b', linestyle=':', label='100k RPM')
    plt.plot(TARGET_RPM, pred_W, 'ro')
    plt.plot(TARGET_RPM, pred_Al, 'go')
    
    plt.title(f'RLVE Sensitivity Prediction (Density Dependent)', fontsize=14)
    plt.xlabel('Rotor Speed (RPM)', fontsize=12)
    plt.ylabel('Phase Shift (milli-radians)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    output_path = OUTPUT_DIR + '/rlve_prediction.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    run_simulation()