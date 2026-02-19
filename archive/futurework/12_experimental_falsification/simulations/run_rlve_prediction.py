"""
AVE MODULE 47: SAGNAC-RLVE MACROSCOPIC PREDICTION
-------------------------------------------------
Strict mathematical evaluation of the Fiber-Optic Sagnac-RLVE.
Applies the exact \rho_{bulk} = 7.9159e6 kg/m^3 vacuum density.
Proves that the kinematic entrainment of a spinning Tungsten rotor generates 
a 2.07 Radian Fresnel-Fizeau phase shift, rendering AVE safely and cheaply 
falsifiable on a standard optical bench.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/12_experimental_falsification/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_sagnac_rlve():
    print("Simulating Sagnac-RLVE (Tabletop Falsification)...")
    
    c = 299792458.0
    rho_vac = 7.9159e6  
    RPM = np.linspace(0, 15000, 500) 
    r_rotor = 0.15 
    v_tan = RPM * (2 * np.pi / 60) * r_rotor
    
    rho_W = 19300.0  # Tungsten
    rho_Al = 2700.0  # Aluminum
    
    # Kinematic Entrainment Velocity
    v_fluid_W = v_tan * (rho_W / rho_vac)
    v_fluid_Al = v_tan * (rho_Al / rho_vac)
    
    # Sagnac Phase Shift (\Delta\phi)
    L_fiber = 200.0 # 200 meter wound SMF-28
    lambda_laser = 1550e-9 
    
    shift_W = (4 * np.pi * L_fiber * v_fluid_W) / (lambda_laser * c)
    shift_Al = (4 * np.pi * L_fiber * v_fluid_Al) / (lambda_laser * c)
    
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    fig.patch.set_facecolor('#050508'); ax.set_facecolor('#050508')
    
    ax.plot(RPM, shift_W, color='#00ffcc', lw=4, label=r'Tungsten Rotor ($\rho = 19.3$ g/cc): Signal')
    ax.plot(RPM, shift_Al, color='#ff3366', lw=3, linestyle='--', label=r'Aluminum Rotor ($\rho = 2.7$ g/cc): Control')
    ax.axhline(0, color='white', lw=2, linestyle=':', label='General Relativity (Null Frame Dragging)')
    
    ax.set_ylim(0, 3.0); ax.set_xlim(0, 15000)
    ax.set_xlabel('Rotor Speed (RPM)', fontsize=13, color='white', weight='bold')
    ax.set_ylabel(r'Sagnac Phase Shift $\Delta\phi$ (Radians)', fontsize=13, color='white', weight='bold')
    ax.set_title('The Sagnac-RLVE: Tabletop Optical Falsification', fontsize=15, pad=15, color='white', weight='bold')
    
    textstr = (
        r"$\mathbf{The~Macroscopic~Breakthrough:}$" + "\n" +
        r"Because the vacuum density is $\rho_{vac} \approx 7.9 \times 10^6 \text{ kg/m}^3$," + "\n" +
        r"a spinning Tungsten mass mechanically drags the vacuum fluid" + "\n" +
        r"at $\approx 0.24\%$ of its tangential velocity." + "\n\n" +
        r"Coupled with a 200m fiber coil, this produces a massive Fizeau" + "\n" +
        r"optical phase shift of $\mathbf{\sim 2.07~Rads}$ at 10,000 RPM. This is easily" + "\n" +
        r"detectable by a sub-\$5,000 standard fiber interferometer."
    )
    ax.text(500, 1.8, textstr, color='white', fontsize=12, bbox=dict(facecolor='#111111', edgecolor='#00ffcc', alpha=0.9, pad=10))

    ax.grid(True, which="major", ls=":", color='#444444', alpha=0.8)
    ax.legend(loc='upper left', facecolor='#111111', edgecolor='white', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333333')
    
    filepath = os.path.join(OUTPUT_DIR, "sagnac_rlve_prediction.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_sagnac_rlve()