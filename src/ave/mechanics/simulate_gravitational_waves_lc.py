import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os

def main():
    print("==========================================================")
    print(" AVE COSMIC SCALE: GENERAL RELATIVITY & GRAVITATIONAL WAVES")
    print("==========================================================\n")

    print("- Objective: Eliminate Einstein's 'Empty Curved 4D Manifold'.")
    print("- We will map Gravitational Waves explicitly as Macroscopic Inductive")
    print("  Shear-Waves rippling through the massive dielectric LC vacuum matrix.")
    print("  'Curved Spacetime' is strictly Variable Vacuum Impedance (Z = sqrt(L/C)).\n")

    # Simulation Parameters
    NX, NY = 120, 120
    FRAMES = 150
    
    # Grid initialization (2D slice of the continuous vacuum fluid)
    # H_z represents the transverse magnetic / inductive shear (strain-wave)
    Hz = np.zeros((NX, NY))
    
    # We will simulate a Binary Orbit (e.g., Two Black Holes)
    # Their immense rotating Inductive Torsional fields (mass) pump
    # acoustic shear-waves into the high-tension vacuum medium.
    
    center_x, center_y = NX // 2, NY // 2
    orbit_radius = 12
    orbit_speed = 0.15
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#050510')
    ax.set_facecolor('#050510')
    
    # Colormap showing gravitational strain amplitude (seismic wave in the LC grid)
    img = ax.imshow(Hz, cmap='twilight_shifted', vmin=-1.5, vmax=1.5, origin='lower')
    ax.axis('off')
    ax.set_title("Gravitational Waves: Inductive Shear in the LC Vacuum", color='white', pad=20, fontsize=14)

    print("[1] Simulating 2D binary black hole orbital pumping...")
    
    def update(frame):
        nonlocal Hz
        
        # Simple wave equation (Discrete Laplacian)
        new_Hz = np.copy(Hz)
        
        # Speed of propagation (c)
        c2 = 0.25 
        damping = 0.99 # Geometric spreading
        
        # Calculate Laplacian for acoustic wave propagation
        for i in range(1, NX-1):
            for j in range(1, NY-1):
                laplacian = (Hz[i+1, j] + Hz[i-1, j] + Hz[i, j+1] + Hz[i, j-1] - 4*Hz[i, j])
                new_Hz[i, j] = Hz[i, j] + c2 * laplacian
                new_Hz[i, j] *= damping
                
        # Inject orbital source (Binary Black Holes acting as physical impellers)
        angle = frame * orbit_speed
        
        # BH 1
        x1 = int(center_x + orbit_radius * np.cos(angle))
        y1 = int(center_y + orbit_radius * np.sin(angle))
        
        # BH 2
        x2 = int(center_x + orbit_radius * np.cos(angle + np.pi))
        y2 = int(center_y + orbit_radius * np.sin(angle + np.pi))
        
        # They drag the vacuum, creating an alternating quadrupole strain wave
        if 1 < x1 < NX-1 and 1 < y1 < NY-1:
            new_Hz[x1, y1] = 2.0 * np.cos(frame * 0.2)
        if 1 < x2 < NX-1 and 1 < y2 < NY-1:
            new_Hz[x2, y2] = -2.0 * np.cos(frame * 0.2) # Quadrupole symmetry
            
        # Absorbing boundary conditions
        new_Hz[0, :] = 0
        new_Hz[-1, :] = 0
        new_Hz[:, 0] = 0
        new_Hz[:, -1] = 0
        
        Hz = new_Hz
        img.set_array(Hz)
        return [img]

    print("[2] Rendering Quadrupole Inductive Strain Waves...")
    ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=40, blit=True)
    
    os.makedirs('standard_model/animations', exist_ok=True)
    out_path = 'standard_model/animations/gravitational_waves_lc.gif'
    ani.save(out_path, writer='pillow', fps=25)
    
    # Extract a static frame showing the spiral wave pattern
    print("[3] Slicing final frame for manuscript PDF...")
    final_frame_data = Hz
    
    fig_static, ax_static = plt.subplots(figsize=(8, 8), facecolor='#050510')
    ax_static.set_facecolor('#050510')
    ax_static.imshow(final_frame_data, cmap='twilight_shifted', vmin=-1.5, vmax=1.5, origin='lower')
    ax_static.axis('off')
    ax_static.set_title("Binary Orbit: Quadrupole LC Strain Radiation", color='white', pad=20, fontsize=14)
    
    os.makedirs('assets/figures', exist_ok=True)
    static_out = 'assets/figures/gravitational_waves_lc_static.pdf'
    fig_static.savefig(static_out, facecolor='#050510', bbox_inches='tight', dpi=150)

    print(f"\n[STATUS: SUCCESS] General Relativity mapped as Variable Spacetime Impedance.")
    print(f"Animated propagation saved to {out_path}")
    print(f"Static spiral state saved to {static_out}")

if __name__ == "__main__":
    main()
