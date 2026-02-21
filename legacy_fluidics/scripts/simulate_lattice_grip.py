"""
AVE MODULE: Microscopic Lattice Grip Animation
----------------------------------------------
Models a 2D slice of the discrete M_A generic Cosserat lattice.
Visualizes how the macroscopic PONDER-01 asymmetric gradient (del_u)
induces a nonreciprocal phonon flow, exerting a net reaction force 
(thrust) against the dielectric hardware via Newton's 3rd Law.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def simulate_lattice_grip():
    print("==========================================================")
    print(" AVE GRAND AUDIT: EXECUTING MICROSCOPIC LATTICE GRIP SIM")
    print("==========================================================")
    
    # 1. Initialize Discrete Grid
    grid_size = 20
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    
    # Base node coordinates (x, y)
    nodes_x = x.flatten().astype(float)
    nodes_y = y.flatten().astype(float)
    
    # Node orientational state (angles internal to the Cosserat solid)
    # 0 implies a perfectly relaxed, unpolarized metric
    angles = np.zeros_like(nodes_x)
    
    # 2. Define the PONDER-01 Asymmetric Gradient field shape
    # We model the scalar energy density (del_u) as a concentrated cone pointing down-left
    gradient_center_x = grid_size * 0.7
    gradient_center_y = grid_size * 0.7
    
    # Calculate geometric strain from the central high-voltage transient
    distances = np.sqrt((nodes_x - gradient_center_x)**2 + (nodes_y - gradient_center_y)**2)
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0B0F19')
    ax.set_facecolor('#0B0F19')
    ax.set_xlim(-2, grid_size+2)
    ax.set_ylim(-2, grid_size+2)
    
    # Customize aesthetics
    ax.tick_params(colors='lightgray')
    for spine in ax.spines.values(): spine.set_color('#333333')
    ax.set_title("AVE Microscopic Lattice Grip\n(Phonon Momentum Transfer Sequence)", color='white', weight='bold', fontsize=14, pad=15)
    
    # Scatter points for the lattice nodes
    scat = ax.scatter(nodes_x, nodes_y, c='#00FFCC', s=20, alpha=0.6, edgecolors='white', linewidths=0.5)
    
    # Quiver plot for internal orientation (chiral twist/polarization)
    Q = ax.quiver(nodes_x, nodes_y, np.cos(angles)*0.4, np.sin(angles)*0.4, 
                  color='#FF3366', scale=25, alpha=0.7, headwidth=4)

    # Background gradient shading overlay 
    # (Visualizes the macroscopic asymmetric Wedge field sliding over the lattice)
    bg_img = ax.imshow(np.zeros((grid_size, grid_size)), cmap='magma', extent=[0, grid_size-1, 0, grid_size-1], origin='lower', alpha=0.3, zorder=0)
    
    # Add Hardware Graphic (The PCBA pad producing the field)
    hw_patch = plt.Polygon([[12, 18], [18, 18], [15, 12]], color='#E0E0E0', alpha=0.8, zorder=10)
    ax.add_patch(hw_patch)
    ax.text(15, 18.5, "BaTiO3 Array & Asymmetric Wedge", color='white', ha='center', weight='bold', fontsize=10)

    # Macroscopic Reaction Force Arrow (Thrust)
    thrust_arrow = ax.arrow(15, 15, 2, 2, head_width=1, head_length=1.5, fc='#FFD54F', ec='white', linewidth=3, zorder=12, alpha=0)
    thrust_text = ax.text(18, 18.5, "Net Ponderomotive Thrust", color='#FFD54F', ha='left', weight='bold', fontsize=12, alpha=0)

    # 3. Animation Logic
    def update(frame):
        # We simulate a 60-frame sequence: 
        # 0-10: Idle
        # 10-30: The 1ns dV/dt transient strikes (Violent Polarization & Displacement)
        # 30-60: Adiabatic Relaxation (Phonon Avalanche & Reaction Force)
        
        current_x = np.copy(nodes_x)
        current_y = np.copy(nodes_y)
        current_angles = np.zeros_like(angles)
        
        # The dynamic magnitude of the SiC MOSFET pulse
        pulse_mag = 0.0
        if 10 < frame <= 30:
            # Steep pulse 
            pulse_mag = (frame - 10) / 20.0
        elif frame > 30:
            # Slow relaxation
            pulse_mag = max(0, 1.0 - (frame - 30) / 20.0)
            
        # The intense spatial gradient is asymmetrical, pointing bottom-left
        grad_vx = -0.06 * pulse_mag
        grad_vy = -0.06 * pulse_mag
        
        # Calculate lattice distortion based on proximity to the asymmetric wedge tip
        strain_profile = np.exp(-distances / 4.0)
        
        # 1. Translational Displacement ("The Squeeze")
        current_x += grad_vx * strain_profile * 15.0
        current_y += grad_vy * strain_profile * 15.0
        
        # 2. Orientational Polarization ("The Twist")
        # High E-fields torque the chiral Cosserat nodes
        current_angles = np.pi/2 * strain_profile * pulse_mag
        
        # Update Visuals
        scat.set_offsets(np.c_[current_x, current_y])
        Q.set_UVC(np.cos(current_angles)*0.4, np.sin(current_angles)*0.4)
        
        # Update Background scalar energy density field (del_u)
        bg_intensity = np.outer(np.exp(-np.linspace(0, 1, grid_size)), np.exp(-np.linspace(0, 1, grid_size)))
        bg_intensity = bg_intensity * pulse_mag
        bg_img.set_data(bg_intensity)
        bg_img.set_clim(0, 1)

        # Update Hardware displacement (Newton's 3rd Law -> Momentum Conservation)
        # If the lattice is forced down/left, the hardware is thrust up/right.
        hw_disp_x = np.sum(grad_vx * strain_profile) * 0.02
        hw_disp_y = np.sum(grad_vy * strain_profile) * 0.02
        
        base_hw = np.array([[12, 18], [18, 18], [15, 12]])
        # Apply inverse displacement to the hardware 
        new_hw = base_hw - np.array([hw_disp_x, hw_disp_y])
        hw_patch.set_xy(new_hw)
        
        # Show Thrust Vector and Background color shift during peak pulse
        if frame > 15 and pulse_mag > 0.1:
            thrust_arrow.set_alpha(1.0)
            thrust_text.set_alpha(1.0)
            thrust_arrow.set_data(x=new_hw[2,0], y=new_hw[2,1], dx=-hw_disp_x*20, dy=-hw_disp_y*20)
        else:
            thrust_arrow.set_alpha(0)
            thrust_text.set_alpha(0)
            
        return scat, Q, bg_img, hw_patch, thrust_arrow, thrust_text

    print("Generating High-Fidelity Lattice Animation...")
    ani = animation.FuncAnimation(fig, update, frames=60, interval=50, blit=False)
    
    OUTPUT_DIR = "assets/sim_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "microscopic_lattice_grip.gif")
    
    try:
        # Save as a looping GIF
        ani.save(out_path, writer='pillow', fps=15, savefig_kwargs={'facecolor': fig.get_facecolor()})
        print(f"Saved Microscopic Animation: {out_path}")
    except Exception as e:
        print(f"Failed to save GIF: {e}")

if __name__ == "__main__":
    simulate_lattice_grip()
