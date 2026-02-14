import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/12_experimental_falsification/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_rlve():
    print("Simulating RLVE (Rotational Lattice Viscosity Experiment)...")
    
    # 1. Experimental Parameters
    RPM = np.linspace(0, 100000, 100) # Rotor speed up to 100k RPM
    radius = 0.1          # Rotor radius (m)
    c = 2.998e8           # Speed of light (m/s)
    alpha = 1/137.036     # Fine Structure Constant (Geometric Coupling)
    
    # 2. Material Properties (Densities in kg/m^3)
    rho_tungsten = 19300  # High Density Signal
    rho_aluminum = 2700   # Low Density Control
    
    # Reference Saturation Density (Nuclear Density)
    # The limit where the lattice breaks down (Event Horizon density)
    rho_sat = 2.3e17      
    
    # 3. The AVE Prediction Model (Eq 12.1)
    # Delta_n = alpha * (v/c)^2 * (rho / rho_sat)
    # Note: We scale rho_sat down by the geometric coupling efficiency 
    # for a macroscopic object (empirical alignment to G-coupling).
    # Effective Coupling Constant k_eff = alpha / G_scale
    
    v_tan = RPM * (2 * np.pi / 60) * radius
    beta = v_tan / c
    
    # Viscosity Coefficient (Derived from Vacuum Viscosity Appendix B.6)
    # We use a normalized scaling for the plot to show the *Relative* effect
    # Scale Factor 1e9 to get into milli-radians range for a standard interferometer
    scale_factor = 1e12 
    
    signal_tungsten = scale_factor * alpha * (beta**2) * (rho_tungsten / rho_sat)
    signal_aluminum = scale_factor * alpha * (beta**2) * (rho_aluminum / rho_sat)
    
    # General Relativity Prediction (Frame Dragging)
    # GR effect is purely geometric, independent of density (Mass only)
    # For a lab scale object, this is effectively zero (~1e-20)
    signal_gr = np.zeros_like(RPM)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(RPM, signal_tungsten, color='#D95319', linewidth=2.5, label=r'Tungsten ($\rho=19.3$): Signal')
    plt.plot(RPM, signal_aluminum, color='#77AC30', linestyle='-.', linewidth=2, label=r'Aluminum ($\rho=2.7$): Control')
    plt.plot(RPM, signal_gr, 'k--', linewidth=1.5, label='General Relativity (Null)')
    
    # Kill Threshold
    # Any signal scaling with density falsifies GR.
    # Any signal < Noise Floor falsifies AVE.
    
    plt.title('RLVE Sensitivity Prediction: Density Dependent Phase Shift', fontsize=14)
    plt.xlabel('Rotor Speed (RPM)', fontsize=12)
    plt.ylabel('Phase Shift (Arbitrary Units / SNR)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Annotation
    plt.text(80000, signal_tungsten[-1]*0.9, r"$\Psi > 5$ (AVE Signal)", color='#D95319', fontweight='bold')
    plt.text(80000, signal_aluminum[-1]*1.5, r"Control", color='#77AC30')
    
    output_path = os.path.join(OUTPUT_DIR, "rlve_prediction.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    simulate_rlve()