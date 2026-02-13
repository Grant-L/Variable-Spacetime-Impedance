import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# Directory setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================================================================
# AVE GIZA SIMULATION SUITE
# Generates GIF animations + annotated static multi-panel PNGs
# ===================================================================

# 1. Viscous Flow: GIF + Static 2x2 views
def generate_viscous():
    print("\n=== Generating Viscous Flow Visualizations ===")
    num_shafts = 8
    shaft_radius = 20.0
    height = 648.0
    ring_radius = 150.0
    num_particles_per_shaft = 80  # More for clearer statics
    
    # Shaft centers
    centers = [(ring_radius * np.cos(2*np.pi*i/num_shafts), ring_radius * np.sin(2*np.pi*i/num_shafts)) for i in range(num_shafts)]
    
    # Particles
    particles = []
    for cx, cy in centers:
        r_off = np.random.uniform(0, shaft_radius, num_particles_per_shaft)
        th_off = np.random.uniform(0, 2*np.pi, num_particles_per_shaft)
        z_start = np.random.uniform(0, height, num_particles_per_shaft)
        speeds = (1 - (r_off / shaft_radius)**2)
        particles.append((r_off, th_off, z_start, speeds))
    
    # Static multi-panel (2x2 views at different "times"/rotations)
    fig_static = plt.figure(figsize=(14, 10))
    views = [(20, 45, 0), (20, 135, 50), (20, 225, 100), (20, 315, 150)]  # elev, azim, frame offset
    for idx, (elev, azim, frame_offset) in enumerate(views):
        ax = fig_static.add_subplot(2, 2, idx+1, projection='3d')
        
        # Shaft surfaces
        for cx, cy in centers:
            theta = np.linspace(0, 2*np.pi, 30)
            z = np.linspace(0, height, 40)
            theta, z = np.meshgrid(theta, z)
            x = shaft_radius * np.cos(theta) + cx
            y = shaft_radius * np.sin(theta) + cy
            ax.plot_surface(x, y, z, alpha=0.15, color='gray')
        
        # Particles at "time" frame_offset
        xs, ys, zs = [], [], []
        for i, (r_off, th_off, z_start, speeds) in enumerate(particles):
            cx, cy = centers[i]
            z_pos = (z_start - frame_offset * 3 * speeds) % height
            x_pos = r_off * np.cos(th_off) + cx
            y_pos = r_off * np.sin(th_off) + cy
            xs.extend(x_pos)
            ys.extend(y_pos)
            zs.extend(z_pos)
        ax.scatter(xs, ys, zs, c='red', s=20, alpha=0.8)
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'View {idx+1}: Downward Wake Flow')
        ax.text2D(0.05, 0.95, "Faster center flow ->", transform=ax.transAxes, color='blue', fontsize=12)
    
    plt.suptitle('Viscous Vacuum Flow in 8-Shaft Network (Parabolic Profiles)')
    static_path = os.path.join(SCRIPT_DIR, 'viscous_static_evolution.png')
    fig_static.savefig(static_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved static: {static_path}")
    
    # GIF animation
    fig_ani = plt.figure(figsize=(12,10))
    ax_ani = fig_ani.add_subplot(111, projection='3d')
    
    # Setup static shafts
    for cx, cy in centers:
        theta = np.linspace(0, 2*np.pi, 40)
        z = np.linspace(0, height, 50)
        theta, z = np.meshgrid(theta, z)
        x = shaft_radius * np.cos(theta) + cx
        y = shaft_radius * np.sin(theta) + cy
        ax_ani.plot_surface(x, y, z, alpha=0.15, color='gray')
    
    scat = ax_ani.scatter([], [], [], c='red', s=30)
    ax_ani.text2D(0.05, 0.05, "v Hubble Wake Direction\nFaster at centers", transform=ax_ani.transAxes, color='white', fontsize=14, bbox=dict(facecolor='black', alpha=0.6))
    
    def update(frame):
        xs, ys, zs = [], [], []
        for i, (r_off, th_off, z_start, speeds) in enumerate(particles):
            cx, cy = centers[i]
            z_pos = (z_start - frame * 3 * speeds) % height
            x_pos = r_off * np.cos(th_off) + cx
            y_pos = r_off * np.sin(th_off) + cy
            xs.extend(x_pos)
            ys.extend(y_pos)
            zs.extend(z_pos)
        scat.set_offsets(np.c_[xs, ys])
        scat.set_3d_properties(zs, 'z')
        ax_ani.view_init(elev=20, azim=frame * 0.5)
        return scat,
    
    ani = FuncAnimation(fig_ani, update, frames=200, interval=50, blit=False)
    gif_path = os.path.join(SCRIPT_DIR, 'viscous_flow_motion.gif')
    ani.save(gif_path, writer=PillowWriter(fps=20))
    plt.close()
    print(f"Saved GIF: {gif_path}")

# 2. Helical Wave: GIF + Static 1x3 phases
def generate_helical():
    print("\n=== Generating Helical Wave Visualizations ===")
    a = 20.0
    height = 648.0
    p = 60.0
    psi = np.arctan(p / (2*np.pi*a))
    vp_ratio = np.tan(psi)
    wavelength = 100.0
    
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(0, height, 100)
    theta, z = np.meshgrid(theta, z)
    x_shaft = a * np.cos(theta)
    y_shaft = a * np.sin(theta)
    
    # Static 1x3
    fig_static = plt.figure(figsize=(15, 5))
    phases = [0, 25, 50]
    for idx, frame in enumerate(phases):
        ax = fig_static.add_subplot(1, 3, idx+1, projection='3d')
        ax.plot_surface(x_shaft, y_shaft, z, alpha=0.2, color='gray')
        
        phase = 2*np.pi * frame / 50
        phi = (2*np.pi * z / wavelength - vp_ratio * phase) % (2*np.pi)
        amp = np.cos(phi)
        X = (a * 0.9) * np.cos(theta) * (1 + 0.3*amp)
        Y = (a * 0.9) * np.sin(theta) * (1 + 0.3*amp)
        ax.plot_surface(X, Y, z, alpha=0.7, cmap='viridis')
        
        ax.set_title(f'Phase {idx+1}: Pulse Propagation')
        ax.text2D(0.5, 0.9, f'v_p ~ {vp_ratio:.2f}c', transform=ax.transAxes, ha='center', color='white', bbox=dict(facecolor='black', alpha=0.7))
    
    plt.suptitle('Chiral Rifled Pulse Evolution in Single Shaft')
    static_path = os.path.join(SCRIPT_DIR, 'helical_static_propagation.png')
    fig_static.savefig(static_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved static: {static_path}")
    
    # GIF animation
    fig_ani = plt.figure(figsize=(12, 10))
    ax_ani = fig_ani.add_subplot(111, projection='3d')
    ax_ani.plot_surface(x_shaft, y_shaft, z, alpha=0.2, color='gray')
    
    # Initialize wave surface
    phase_init = 0
    phi_init = (2*np.pi * z / wavelength - vp_ratio * phase_init) % (2*np.pi)
    amp_init = np.cos(phi_init)
    X_init = (a * 0.9) * np.cos(theta) * (1 + 0.3*amp_init)
    Y_init = (a * 0.9) * np.sin(theta) * (1 + 0.3*amp_init)
    wave = ax_ani.plot_surface(X_init, Y_init, z, alpha=0.7, cmap='viridis', cstride=1, rstride=1)
    
    ax_ani.text2D(0.5, 0.9, f'v_p ~ {vp_ratio:.2f}c', transform=ax_ani.transAxes, ha='center', color='white', fontsize=14, bbox=dict(facecolor='black', alpha=0.7))
    
    def update(frame):
        nonlocal wave
        phase = 2*np.pi * frame / 50
        phi = (2*np.pi * z / wavelength - vp_ratio * phase) % (2*np.pi)
        amp = np.cos(phi)
        X = (a * 0.9) * np.cos(theta) * (1 + 0.3*amp)
        Y = (a * 0.9) * np.sin(theta) * (1 + 0.3*amp)
        wave.remove()
        wave = ax_ani.plot_surface(X, Y, z, alpha=0.7, cmap='viridis', cstride=1, rstride=1)
        ax_ani.set_title(f'Helical Rifled Pulse Propagation (v_p ~ {vp_ratio:.2f}c)')
        return wave,
    
    ani = FuncAnimation(fig_ani, update, frames=200, interval=50, blit=False)
    gif_path = os.path.join(SCRIPT_DIR, 'helical_wave_propagation.gif')
    ani.save(gif_path, writer=PillowWriter(fps=15))
    plt.close()
    print(f"Saved GIF: {gif_path}")

# 3. Non-Linear Saturation: GIF + Static 2x2 ramp
def generate_nonlinear():
    print("\n=== Generating Non-Linear Saturation Visualizations ===")
    r = np.linspace(0, 20, 200)
    v_base = 1 - (r/20)**2
    
    def velocity_profile(sat_level):
        if sat_level < 1.0:
            return v_base / (1 - sat_level * v_base**2)
        else:
            return np.zeros_like(r)
    
    # Static 2x2
    fig_static = plt.figure(figsize=(12, 8))
    levels = [0.0, 0.5, 0.9, 1.1]
    titles = ['Linear Regime', 'Mild Softening', 'Near Saturation (Amplification)', 'Breakdown Quench']
    for idx, sat in enumerate(levels):
        ax = fig_static.add_subplot(2, 2, idx+1)
        v = velocity_profile(sat)
        ax.plot(r, v_base, '--', label='Linear Baseline', alpha=0.6)
        ax.plot(r, v, lw=3, label=f'Sat = {sat:.1f}')
        ax.set_ylim(0, max(2, v.max()+0.2))
        ax.set_xlabel('Radial position (m)')
        ax.set_ylabel('Normalized velocity')
        ax.set_title(titles[idx])
        ax.legend()
        ax.grid(True)
        if idx == 2:
            ax.annotate('Shock formation ->', xy=(10, v[100]), xytext=(12, v[100]+0.5),
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.suptitle('Non-Linear Dielectric Saturation Ramp in Shaft Flow Profile')
    static_path = os.path.join(SCRIPT_DIR, 'nonlinear_static_ramp.png')
    fig_static.savefig(static_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved static: {static_path}")
    
    # GIF animation
    fig_ani = plt.figure(figsize=(10, 6))
    ax_ani = fig_ani.add_subplot(111)
    
    line_base, = ax_ani.plot(r, v_base, '--', label='Linear Baseline', alpha=0.6)
    line_sat, = ax_ani.plot(r, velocity_profile(0.0), lw=3, label='Saturation Level', color='red')
    
    ax_ani.set_ylim(0, 2.5)
    ax_ani.set_xlabel('Radial position (m)')
    ax_ani.set_ylabel('Normalized velocity')
    ax_ani.set_title('Non-Linear Dielectric Saturation Evolution')
    ax_ani.legend()
    ax_ani.grid(True)
    
    def update(frame):
        sat_level = frame / 100.0  # Ramp from 0 to 2.0
        v = velocity_profile(sat_level)
        line_sat.set_ydata(v)
        line_sat.set_label(f'Sat = {sat_level:.2f}')
        ax_ani.legend()
        if sat_level >= 1.0:
            ax_ani.set_title('Non-Linear Dielectric Saturation Evolution (Breakdown)')
        else:
            ax_ani.set_title('Non-Linear Dielectric Saturation Evolution')
        return line_sat,
    
    ani = FuncAnimation(fig_ani, update, frames=200, interval=50, blit=False)
    gif_path = os.path.join(SCRIPT_DIR, 'non_linear_saturation.gif')
    ani.save(gif_path, writer=PillowWriter(fps=15))
    plt.close()
    print(f"Saved GIF: {gif_path}")

if __name__ == "__main__":
    generate_viscous()
    generate_helical()
    generate_nonlinear()
    print("\nAll visualizations complete! Static PNGs for print + GIFs for digital/supplementary.")