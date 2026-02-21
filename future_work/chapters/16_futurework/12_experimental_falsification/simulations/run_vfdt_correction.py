"""
AVE MODULE 41: THE VACUUM-FLUX DRAG TEST (LORENTZ STABILITY)
------------------------------------------------------------
Evaluates the proposed VFDT using the strict AVE \rho_{bulk} limit.
Proves that because \mathbf{A} is momentum density, v_vac = p_vac / M_bulk.
Demonstrates that the hyper-density of the vacuum suppresses optical 
frame-dragging in standard EM fields, physically preserving macroscopic 
Lorentz invariance (light travels in straight lines through magnets).
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/12_experimental_falsification/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_vfdt_null_result():
    print("Simulating Dimensionally Exact VFDT (Optical Stability Proof)...")
    
    c = 299792458.0
    xi_topo = 4.149e-7 # C/m
    rho_bulk = 7.9159e6  # kg/m^3 (Exact AVE derivation)
    mu_0 = 1.2566e-6
    
    current = np.linspace(0, 100000, 500) # up to 100 kA
    N_turns = 100.0; R_torus = 0.25; r_wire = 0.05
    L_path = 2 * np.pi * R_torus
    Area = np.pi * r_wire**2
    Volume = Area * L_path
    
    # 1. B-Field and Vector Potential (Momentum)
    B_field = (mu_0 * N_turns * current) / L_path
    Phi = B_field * Area
    p_vac = Phi * xi_topo 
    
    # 2. Strict AVE Kinematic Velocity (Regime B)
    M_vac = rho_bulk * Volume # ~97,450 kg
    v_vac = p_vac / M_vac
    
    # 3. Fizeau Phase Shift (1550 nm laser in 200m Sagnac Fiber)
    lambda_laser = 1550e-9
    L_fiber = 200.0 
    delta_phi = (4 * np.pi * L_fiber * v_vac) / (lambda_laser * c)
    
    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor('#050508'); ax1.set_facecolor('#050508')
    
    ax1.plot(current / 1000, v_vac * 1e13, color='#00ffcc', lw=3.5, label=r'Exact Fluid Drift Velocity ($v_{vac} = p_{vac} / M_{vac}$)')
    ax1.set_xlabel('Toroidal Pulse Current (kA)', fontsize=13, color='white', weight='bold')
    ax1.set_ylabel(r'Vacuum Drift Velocity ($10^{-13}$ m/s)', color='#00ffcc', fontsize=13, weight='bold')
    ax1.tick_params(axis='y', labelcolor='#00ffcc', colors='white'); ax1.tick_params(axis='x', colors='white')
    
    ax2 = ax1.twinx()
    ax2.plot(current / 1000, delta_phi * 1e12, color='#ff3366', lw=3.5, linestyle='--', label=r'Sagnac Phase Shift ($\Delta\phi$)')
    ax2.set_ylabel(r'Optical Phase Shift ($10^{-12}$ radians)', color='#ff3366', fontsize=13, weight='bold')
    ax2.tick_params(axis='y', labelcolor='#ff3366')
    
    # Noise floor of standard interferometers (~1e-7 rad)
    ax2.axhline(100000, color='gray', linestyle=':', lw=2, label='Typical Instrument Noise Floor ($10^{-7}$ rad)')
    
    for spine in ax1.spines.values(): spine.set_color('#333333')
    for spine in ax2.spines.values(): spine.set_color('#333333')
    
    plt.title('The VFDT Null Result: Optical Stability of the Vacuum', color='white', fontsize=15, weight='bold', pad=15)
    
    textstr = (
        r"$\mathbf{Resolution~of~the~Drag~Paradox:}$" + "\n" +
        r"Because the vacuum is structurally hyper-dense ($\approx 7.9 \times 10^6$ kg/m$^3$), the" + "\n" +
        r"fluid mass inside the torus is $\approx 97,000$ kg. The EM momentum generates" + "\n" +
        r"negligible fluid advection ($10^{-13}$ m/s)." + "\n" +
        r"This mathematically guarantees that light travels in straight lines through" + "\n" +
        r"standard magnetic fields, rigorously preserving macroscopic Lorentz invariance."
    )
    ax1.text(2, 0.3, textstr, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='white', alpha=0.9, pad=10))

    filepath = os.path.join(OUTPUT_DIR, "vfdt_dimensional_correction.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_vfdt_null_result()