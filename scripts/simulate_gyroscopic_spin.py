"""
AVE MODULE: Quantum Spin as Classical Gyroscopic Precession
-----------------------------------------------------------
This script mathematically maps the abstract 20th-century quantum "Spin 1/2" 
state transition (Bloch Sphere) directly onto the 19th-century classical 
mechanical Larmor precession of a physical gyroscope.

If the electron is a macroscopic 3_1 Topological Flywheel, Nuclear Magnetic 
Resonance (NMR) and Electron Paramagnetic Resonance (EPR) are not abstract
"transition probabilities," but continuous mechanical projections of the 
flywheel's tilt angle (geometric displacement) when subjected to an external
magnetic torque.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from mpl_toolkits.mplot3d import Axes3D

def simulate_gyroscopic_spin():
    print("==========================================================")
    print(" AVE GRAND AUDIT: QUANTUM SPIN = GYROSCOPIC PRECESSION")
    print("==========================================================")
    
    # ---------------------------------------------------------
    # 1. MATHEMATICAL SETUP
    # ---------------------------------------------------------
    frames = 150
    t = np.linspace(0, 4*np.pi, frames) # 2 full precession cycles
    
    # The external static magnetic field (B_z) pointing straight up
    B_z = 1.0 
    
    # The applied RF resonant magnetic pulse (B_1) pointing along the X axis
    # In NMR, the B_1 field applies the torque that tilts the spin from +Z (Up) to -Z (Down)
    # The tilt angle theta transitions continuously from 0 (North Pole) to pi (South Pole)
    theta = np.linspace(0, np.pi, frames)
    
    # The Larmor Precession ( azimuthal rotation omega_L around the Z axis)
    # omega_L = gamma * B_z
    omega_L = 2.0  # Fast precession
    phi = omega_L * t
    
    # ---------------------------------------------------------
    # 2. CALCULATE 3D TRAJECTORY
    # ---------------------------------------------------------
    # Spherical coordinates to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # ---------------------------------------------------------
    # 3. VISUALIZATION AND ANIMATION
    # ---------------------------------------------------------
    print("Generating side-by-side 3D visualization...")
    fig = plt.figure(figsize=(16, 8), facecolor='#0B0F19')
    
    # --- PANEL 1: ABSTRACT QUANTUM BLOCH SPHERE ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d', facecolor='#0B0F19')
    ax1.set_axis_off()
    ax1.set_title("1. Abstract Quantum Mechanics\n(The Bloch Sphere Spinor)", color='white', weight='bold', pad=20)
    
    # Draw the transparent Bloch Sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    S_x = np.outer(np.cos(u), np.sin(v))
    S_y = np.outer(np.sin(u), np.sin(v))
    S_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(S_x, S_y, S_z, color='cyan', alpha=0.1, edgecolor='none')
    
    # Draw Axes
    ax1.plot([-1.5, 1.5], [0, 0], [0, 0], color='#444444', lw=1, zorder=0)
    ax1.plot([0, 0], [-1.5, 1.5], [0, 0], color='#444444', lw=1, zorder=0)
    ax1.plot([0, 0], [0, 0], [-1.5, 1.5], color='#444444', lw=1, zorder=0)
    ax1.text(0, 0, 1.6, r"$|+\rangle$ (Spin Up)", color='cyan', ha='center', weight='bold')
    ax1.text(0, 0, -1.7, r"$|-\rangle$ (Spin Down)", color='magenta', ha='center', weight='bold')
    
    # The abstract state vector (initialized at +Z)
    quantum_vector, = ax1.plot([0, 0], [0, 0], [0, 1], color='cyan', lw=4, marker='o', markersize=8)
    quantum_trail, = ax1.plot([], [], [], color='white', alpha=0.5, lw=1)
    
    # --- PANEL 2: LITERAL CLASSICAL FLYWHEEL ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d', facecolor='#0B0F19')
    ax2.set_axis_off()
    ax2.set_title("2. Applied Vacuum Engineering\n(Classical Gyroscopic Precession)", color='#00FFCC', weight='bold', pad=20)
    
    # Draw the static B_z field lines acting as gravity/torque
    for offset in [-0.8, 0, 0.8]:
        ax2.plot([offset, offset], [0, 0], [-1.5, 1.5], color='yellow', alpha=0.3, ls='--')
    ax2.text(0, 0, 1.7, "Static Magnetic Torque ($B_z$)", color='yellow', ha='center', weight='bold')
    
    # Draw a visual disk to represent the physical "flywheel"
    disk_theta = np.linspace(0, 2*np.pi, 50)
    disk_x_base = 0.6 * np.cos(disk_theta)
    disk_y_base = 0.6 * np.sin(disk_theta)
    disk_z_base = np.zeros_like(disk_theta)
    
    flywheel_edge, = ax2.plot([], [], [], color='#00FFCC', lw=3)
    flywheel_axle, = ax2.plot([], [], [], color='white', lw=4)
    classical_trail, = ax2.plot([], [], [], color='#00FFCC', alpha=0.5, lw=1, ls=':')

    # Set consistent bounds
    for ax in [ax1, ax2]:
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.view_init(elev=20, azim=45)

    OUTPUT_DIR = "assets/sim_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Render and save a static mid-transition frame for the LaTeX PDF
    mid_frame = 75
    quantum_vector.set_data_3d([0, x[mid_frame]], [0, y[mid_frame]], [0, z[mid_frame]])
    quantum_vector.set_color('magenta') # Halfway down
    quantum_trail.set_data_3d(x[:mid_frame], y[:mid_frame], z[:mid_frame])
    
    classical_trail.set_data_3d(x[:mid_frame], y[:mid_frame], z[:mid_frame])
    flywheel_axle.set_data_3d([-x[mid_frame]*0.5, x[mid_frame]], [-y[mid_frame]*0.5, y[mid_frame]], [-z[mid_frame]*0.5, z[mid_frame]])
    
    Ry_static = np.array([
        [np.cos(theta[mid_frame]), 0, np.sin(theta[mid_frame])],
        [0, 1, 0],
        [-np.sin(theta[mid_frame]), 0, np.cos(theta[mid_frame])]
    ])
    Rz_static = np.array([
        [np.cos(phi[mid_frame]), -np.sin(phi[mid_frame]), 0],
        [np.sin(phi[mid_frame]), np.cos(phi[mid_frame]), 0],
        [0, 0, 1]
    ])
    R_static = Rz_static @ Ry_static
    disk_rotated_static = np.zeros((3, 50))
    for i in range(50):
        vec = np.array([disk_x_base[i], disk_y_base[i], disk_z_base[i]])
        disk_rotated_static[:, i] = R_static @ vec
        
    flywheel_edge.set_data_3d(disk_rotated_static[0, :], disk_rotated_static[1, :], disk_rotated_static[2, :])
    
    plt.tight_layout()
    static_out_path = os.path.join(OUTPUT_DIR, "quantum_spin_gyroscopic_precession.png")
    plt.savefig(static_out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Saved static LaTeX plot: {static_out_path}")

    # ---------------------------------------------------------
    # 4. ANIMATION UPDATE LOOP
    # ---------------------------------------------------------
    def update(frame):
        # Current vector head position
        curr_x = x[frame]
        curr_y = y[frame]
        curr_z = z[frame]
        
        # 1. Update Quantum Abstraction
        quantum_vector.set_data_3d([0, curr_x], [0, curr_y], [0, curr_z])
        quantum_trail.set_data_3d(x[:frame], y[:frame], z[:frame])
        
        # Color shifting (Cyan +Z to Magenta -Z) to represent probability transition
        if curr_z >= 0:
            quantum_vector.set_color('cyan')
        else:
            quantum_vector.set_color('magenta')
            
        # 2. Update Classical Mechanics
        classical_trail.set_data_3d(x[:frame], y[:frame], z[:frame])
        
        # The physical axle points along the angular momentum vector
        flywheel_axle.set_data_3d([-curr_x*0.5, curr_x], [-curr_y*0.5, curr_y], [-curr_z*0.5, curr_z])
        
        # To draw the tilted flywheel disk perpendicular to the axle, we use Euler angles
        # Rotation around Y by theta, then around Z by phi
        Ry = np.array([
            [np.cos(theta[frame]), 0, np.sin(theta[frame])],
            [0, 1, 0],
            [-np.sin(theta[frame]), 0, np.cos(theta[frame])]
        ])
        Rz = np.array([
            [np.cos(phi[frame]), -np.sin(phi[frame]), 0],
            [np.sin(phi[frame]), np.cos(phi[frame]), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry
        
        # Apply rotation matrix manually to the base disk coords
        disk_rotated = np.zeros((3, 50))
        for i in range(50):
            vec = np.array([disk_x_base[i], disk_y_base[i], disk_z_base[i]])
            disk_rotated[:, i] = R @ vec
            
        flywheel_edge.set_data_3d(disk_rotated[0, :], disk_rotated[1, :], disk_rotated[2, :])
        
        return quantum_vector, quantum_trail, flywheel_edge, flywheel_axle, classical_trail

    print("Rendering mechanical sequence...")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=30, blit=False)
    
    OUTPUT_DIR = "assets/sim_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "quantum_spin_gyroscopic_precession.gif")
    
    try:
        ani.save(out_path, writer='pillow', fps=20, savefig_kwargs={'facecolor': fig.get_facecolor()})
        print(f"Saved Larmor Precession Animation: {out_path}")
    except Exception as e:
        print(f"Failed to generate GIF: {e}")

if __name__ == "__main__":
    simulate_gyroscopic_spin()
