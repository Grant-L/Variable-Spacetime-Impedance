import numpy as np
import matplotlib.pyplot as plt

def simulate_sagnac_rlve():
    print("==========================================================")
    print(" AVE GRAND AUDIT: SAGNAC-RLVE KINEMATIC DRAG SIMULATION")
    print("==========================================================")

    # 1. EMPIRICAL CONSTANTS
    C = 299792458.0  # Speed of light (m/s)
    LAMBDA_LASER = 1550e-9  # Telecom laser wavelength (m)
    L_FIBER = 200.0  # Length of fiber optic spool (m)
    R_ROTOR = 0.15  # Radius of the rotor (m)
    
    # Material Densities (kg/m^3)
    RHO_BULK_VACUUM = 7.916e6
    RHO_TUNGSTEN = 19300.0
    RHO_ALUMINUM = 2700.0

    print(f"\n[HARDWARE SETUP]")
    print(f"Fiber Length: {L_FIBER} m")
    print(f"Laser Wavelength: {LAMBDA_LASER*1e9} nm")
    print(f"Rotor Radius: {R_ROTOR} m")
    
    # 2. THEORETICAL ENTRAINMENT COUPLING K
    # AVE dictates that the vacuum fluid is entrained precisely proportional to local mass density
    k_W = RHO_TUNGSTEN / RHO_BULK_VACUUM
    k_Al = RHO_ALUMINUM / RHO_BULK_VACUUM
    
    print(f"\n[AVE ENTRAINMENT COUPLING]")
    print(f"Tungsten (W) Coupling:  {k_W:.6f}")
    print(f"Aluminum (Al) Coupling: {k_Al:.6f}")
    
    # 3. KINEMATIC SIMULATION OVER RPM
    rpms = np.linspace(0, 20000, 500)
    omega = rpms * (2 * np.pi / 60.0)  # Rad/s
    v_tan = omega * R_ROTOR  # Tangential velocity at the rim (m/s)
    
    # Fluid Drift Velocity (m/s)
    v_fluid_W = v_tan * k_W
    v_fluid_Al = v_tan * k_Al
    
    # 4. OPTICAL PHASE SHIFT DERIVATION
    # 1st-order phase shift for Sagnac loop with drifting medium
    # Delta Phi = (4 * pi * L * v_fluid) / (lambda * c)
    phase_shift_modifier = (4 * np.pi * L_FIBER) / (LAMBDA_LASER * C)
    
    delta_phi_W = phase_shift_modifier * v_fluid_W
    delta_phi_Al = phase_shift_modifier * v_fluid_Al
    
    # 5. GENERAL RELATIVITY (LENSE-THIRRING EFFECT) PREDICTION
    # Frame dragging for a 15cm, ~20kg cylinder at 10k RPM is virtually zero (approx 10^-20 radians)
    # Rendering it as a flat zero line for comparative baseline.
    delta_phi_GR = np.zeros_like(rpms)
    
    # Output the exact 10k RPM validation point from the manuscript
    rpm_10k_idx = np.argmin(np.abs(rpms - 10000))
    print(f"\n[10,000 RPM VALIDATION CHECK]")
    print(f"v_tan at 10k RPM: {v_tan[rpm_10k_idx]:.2f} m/s")
    print(f"v_fluid_W:        {v_fluid_W[rpm_10k_idx]:.4f} m/s")
    print(f"Delta Phi (W):    {delta_phi_W[rpm_10k_idx]:.4f} Radians")
    print(f"Delta Phi (Al):   {delta_phi_Al[rpm_10k_idx]:.4f} Radians")
    
    # 6. VISUALIZATION
    plt.figure(figsize=(12, 8), facecolor='#0B0F19')
    ax = plt.gca()
    ax.set_facecolor('#0B0F19')
    ax.grid(color='#1E293B', linestyle='--', linewidth=1)
    
    plt.plot(rpms, delta_phi_W, color='#FF3366', linewidth=3, label='AVE Prediction (Tungsten Rotor)')
    plt.plot(rpms, delta_phi_Al, color='#00FFCC', linewidth=3, label='AVE Prediction (Aluminum Rotor)')
    plt.plot(rpms, delta_phi_GR, color='#FFFFFF', linewidth=2, linestyle=':', label='General Relativity (Null Frame-Dragging)')
    
    # 10k RPM Marker
    plt.scatter([10000], [delta_phi_W[rpm_10k_idx]], color='white', zorder=5, s=80)
    plt.annotate(f"{delta_phi_W[rpm_10k_idx]:.2f} Rad", 
                 xy=(10000, delta_phi_W[rpm_10k_idx]), 
                 xytext=(10500, delta_phi_W[rpm_10k_idx] - 0.2),
                 color='white', fontsize=12, weight='bold')
    
    plt.title("Sagnac-RLVE Optical Phase Shift vs Rotor RPM\nKinematic Vacuum Entrainment Falsification (200m Fiber Loop)", 
              color='white', fontsize=16, pad=15, weight='bold')
    plt.xlabel("Rotor Speed (RPM)", color='white', fontsize=14)
    plt.ylabel("Interferometer Phase Shift $\Delta\Phi$ (Radians)", color='white', fontsize=14)
    
    ax.tick_params(colors='white', labelsize=12)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend = plt.legend(facecolor='#0B0F19', edgecolor='white', fontsize=12, loc='upper left')
    for text in legend.get_texts():
        text.set_color("white")
        
    plt.tight_layout()
    plt.savefig("sagnac_rlve_verification.png", dpi=300, facecolor='#0B0F19')
    print("\nSaved visualization to 'sagnac_rlve_verification.png'")
    # plt.show() # Uncomment if running interactively

if __name__ == "__main__":
    simulate_sagnac_rlve()
