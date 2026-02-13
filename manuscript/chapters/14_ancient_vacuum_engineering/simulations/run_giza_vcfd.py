import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import sympy as sp
from sympy import symbols, cos, pi, atan

# ===================================================================
# UPDATED AVE SIMULATION SUITE WITH ANIMATIONS
# Generates static PNGs + animated GIFs (flow motion + helical propagation)
# Focus: "Field in motion" via particle tracers in viscous flow & wave in helical shaft
# ===================================================================

# Existing static simulations unchanged (saved as PNG)
# ... [lepton_hierarchy, rotation_curve, poiseuille_profile unchanged for brevity]

# 1. Animated Viscous Flow: Particle Tracers in Multi-Shaft Network
def animate_viscous_flow():
    print("\n=== Generating Animated Viscous Flow (flow.gif) ===")
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    
    num_shafts = 8
    shaft_radius = 20.0
    height = 648.0
    ring_radius = 150.0
    num_particles_per_shaft = 50
    frames = 200
    
    # Pre-compute shaft centers
    centers = []
    for i in range(num_shafts):
        angle = 2*np.pi * i / num_shafts
        centers.append((ring_radius * np.cos(angle), ring_radius * np.sin(angle)))
    
    # Particle positions (radial offset determines speed: parabolic profile)
    particles = []
    for cx, cy in centers:
        r_offsets = np.random.uniform(0, shaft_radius, num_particles_per_shaft)
        theta_offsets = np.random.uniform(0, 2*np.pi, num_particles_per_shaft)
        z_start = np.random.uniform(0, height, num_particles_per_shaft)
        speeds = (1 - (r_offsets / shaft_radius)**2)  # Normalized parabolic velocity
        particles.append((r_offsets, theta_offsets, z_start, speeds))
    
    # Setup static shafts
    for cx, cy in centers:
        theta = np.linspace(0, 2*np.pi, 40)
        z = np.linspace(0, height, 50)
        theta, z = np.meshgrid(theta, z)
        x = shaft_radius * np.cos(theta) + cx
        y = shaft_radius * np.sin(theta) + cy
        ax.plot_surface(x, y, z, alpha=0.15, color='gray')
    
    scat = ax.scatter([], [], [], c='red', s=30)
    
    def update(frame):
        xs, ys, zs = [], [], []
        for i, (r_off, th_off, z_start, speeds) in enumerate(particles):
            cx, cy = centers[i]
            # Downward motion (wrap around for loop)
            z_pos = (z_start - frame * 3 * speeds) % height
            x_pos = r_off * np.cos(th_off) + cx
            y_pos = r_off * np.sin(th_off) + cy
            xs.extend(x_pos)
            ys.extend(y_pos)
            zs.extend(z_pos)
        scat.set_offsets(np.c_[xs, ys])
        scat.set_3d_properties(zs, 'z')
        ax.view_init(elev=20, azim=frame * 0.5)  # Slow rotation
        return scat,
    
    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    ani.save('viscous_flow_motion.gif', writer=PillowWriter(fps=20))
    plt.close()
    print("Saved: viscous_flow_motion.gif (particles moving faster in shaft centers)")

# 2. Animated Helical Wave Propagation in Single Shaft
def animate_helical_wave():
    print("\n=== Generating Animated Helical Wave (helical_wave.gif) ===")
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    a = 20.0
    height = 648.0
    p = 60.0
    psi = np.arctan(p / (2*np.pi*a))
    vp_ratio = np.tan(psi)
    wavelength = 100.0  # Illustrative slowed wave
    freq = 1e6  # Hz (below cutoff)
    
    # Static shaft
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(0, height, 100)
    theta, z = np.meshgrid(theta, z)
    x = a * np.cos(theta)
    y = a * np.sin(theta)
    ax.plot_surface(x, y, z, alpha=0.2, color='gray')
    
    # Wave surface (E-field amplitude) - initialize with first frame
    phase_init = 0
    phi_init = (2*np.pi * z / wavelength - vp_ratio * phase_init) % (2*np.pi)
    amp_init = np.cos(phi_init)
    X_init = (a * 0.9) * np.cos(theta) * (1 + 0.3*amp_init)
    Y_init = (a * 0.9) * np.sin(theta) * (1 + 0.3*amp_init)
    wave = ax.plot_surface(X_init, Y_init, z, alpha=0.7, cmap='viridis', cstride=1, rstride=1)
    
    def update(frame):
        phase = 2*np.pi * frame / 50
        phi = (2*np.pi * z / wavelength - vp_ratio * phase) % (2*np.pi)
        amp = np.cos(phi)
        X = (a * 0.9) * np.cos(theta) * (1 + 0.3*amp)
        Y = (a * 0.9) * np.sin(theta) * (1 + 0.3*amp)
        wave.remove()
        wave = ax.plot_surface(X, Y, z, alpha=0.7, cmap='viridis', cstride=1, rstride=1)
        ax.set_title(f'Helical Rifled Pulse Propagation (v_p ≈ {vp_ratio:.2f}c)')
        return wave,
    
    ani = FuncAnimation(fig, update, frames=100, interval=100, blit=False)
    ani.save('helical_wave_propagation.gif', writer=PillowWriter(fps=15))
    plt.close()
    print("Saved: helical_wave_propagation.gif (slowed chiral wave along shaft)")

# Run animations + statics
if __name__ == "__main__":
    # Static PNGs (unchanged)
    # simulate_lepton_hierarchy() etc. if desired
    
    animate_viscous_flow()
    animate_helical_wave()
    
    print("\nAnimations complete! GIFs show fields in motion:")
    print("- viscous_flow_motion.gif: Particle tracers with parabolic velocity (faster center)")
    print("- helical_wave_propagation.gif: Slowed rifled pulse propagating down shaft")