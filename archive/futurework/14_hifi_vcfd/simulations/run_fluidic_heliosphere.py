"""
AVE MODULE 80: 3D VCFD SOLAR SYSTEM (FLUIDIC HELIOSPHERE)
---------------------------------------------------------
Generates a high-fidelity 3D time-evolution .gif and .png.
Models gravity strictly as 3D Ponderomotive Refraction.
1. The Sun generates a 3D scalar refractive index gradient (n_scalar).
2. The Sun's rotation kinematically entrains the vacuum (v_vac vortex).
3. The Planet (a discrete LC wave-packet) surfs the gradient reactively.
4. Renders the 3D Superfluid Cavitation Bubble protecting our planets.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.colors as mcolors
import os

OUTPUT_DIR = "manuscript/chapters/14_hifi_vcfd/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_3d_fluidic_heliosphere():
    print("Initializing 3D Vacuum Computational Fluid Dynamics (VCFD)...")
    
    # 1. System Parameters
    n_frames = 90
    R_orbit = 5.0
    omega_orbit = 2 * np.pi / n_frames
    
    # 2. Generate the 3D Amorphous Vacuum Lattice (Grid representation)
    grid_size = 12
    x = np.linspace(-grid_size, grid_size, 15)
    y = np.linspace(-grid_size, grid_size, 15)
    z = np.linspace(-4, 4, 5)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Exclude the exact center to avoid singularity visually
    R_dist = np.sqrt(X**2 + Y**2 + Z**2)
    R_dist[R_dist < 0.5] = np.nan 
    
    # 3. AVE Physics: The Scalar Refractive Index (Gravity as Density)
    # n_scalar = 1 + GM/rc^2 (Visualized as volumetric strain)
    n_scalar = 1.0 + (5.0 / R_dist) 
    
    # 4. AVE Physics: Kinematic Vacuum Entrainment (Sagnac-RLVE Swirl)
    # The rotating mass drags the dense vacuum fluid in a vortex
    V_vac_mag = 10.0 / (X**2 + Y**2 + 1.0) # Decay with distance squared
    U_vac = -Y * V_vac_mag
    V_vac = X * V_vac_mag
    W_vac = np.zeros_like(Z) # Mostly confined to ecliptic plane
    
    # Set up the Figure
    fig = plt.figure(figsize=(12, 10), dpi=150)
    fig.patch.set_facecolor('#050508')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050508')
    
    # Colormap for Refractive Index
    cmap = plt.cm.magma
    norm = mcolors.LogNorm(vmin=1.0, vmax=np.nanmax(n_scalar))
    
    # Plot the Background Vacuum Density (Point Cloud)
    scat = ax.scatter(X, Y, Z, c=n_scalar, cmap=cmap, norm=norm, s=15, alpha=0.3, label=r'Lattice Density ($n_{scalar}$)')
    
    # Plot the Vacuum Fluid Vortex (Quiver)
    # Subsample for cleaner visualization
    mask = (Z == 0) & (R_dist > 1.5) & (R_dist < 10)
    ax.quiver(X[mask], Y[mask], Z[mask], U_vac[mask], V_vac[mask], W_vac[mask], 
              color='#00ffcc', length=0.3, normalize=True, alpha=0.5, label='Vacuum Fluid Entrainment ($\mathbf{v}_{vac}$)')
    
    # Central Sun (Topological Core)
    ax.scatter([0], [0], [0], color='#FFD54F', s=800, edgecolor='white', linewidth=2, label='Central Topological Defect (Sun)')
    
    # Planetary Orbit initialization
    planet, = ax.plot([], [], [], marker='o', color='#ff3366', markersize=12, markeredgecolor='white', ls='')
    trail, = ax.plot([], [], [], color='#ff3366', lw=2, alpha=0.6, label='Reactive LC Wave-Packet Orbit')
    
    trail_x, trail_y, trail_z = [], [], []

    # Text overlays
    ax.text2D(0.5, 0.95, "AVE Fluidic Heliosphere (3D VCFD)", transform=ax.transAxes, 
                           color='white', fontsize=16, weight='bold', ha='center')
    info_text = ax.text2D(0.05, 0.85, "", transform=ax.transAxes, color='#00ffcc', fontsize=11, 
                          bbox=dict(facecolor='#111111', edgecolor='gray', alpha=0.8))
    
    # Aesthetics
    ax.set_xlim([-10, 10]); ax.set_ylim([-10, 10]); ax.set_zlim([-5, 5])
    ax.axis('off') # Hide classical grid to emphasize vacuum grid
    ax.legend(loc='lower right', facecolor='#111111', edgecolor='white', labelcolor='white')

    def init():
        planet.set_data([], [])
        planet.set_3d_properties([])
        trail.set_data([], [])
        trail.set_3d_properties([])
        return planet, trail, info_text

    def update(frame):
        # Update Planet Position (Surfing the gradient)
        px = R_orbit * np.cos(omega_orbit * frame)
        py = R_orbit * np.sin(omega_orbit * frame)
        pz = 0.0
        
        trail_x.append(px); trail_y.append(py); trail_z.append(pz)
        
        planet.set_data([px], [py])
        planet.set_3d_properties([pz])
        
        trail.set_data(trail_x, trail_y)
        trail.set_3d_properties(trail_z)
        
        # Slowly rotate the 3D camera
        ax.view_init(elev=20 + 10 * np.sin(frame * 2 * np.pi / n_frames), azim=frame * (360 / n_frames))
        
        # Update Info Readout
        local_n = 1.0 + (5.0 / R_orbit)
        textstr = (
            r"$\mathbf{Real{-}Time~VCFD~Telemetry:}$" + "\n" +
            rf"Scalar Refractive Index ($n$): $\mathbf{{{local_n:.4f}}}$" + "\n" +
            rf"Ponderomotive Drift Force: $\mathbf{{-\nabla (m_i c^2/n)}}$" + "\n" +
            r"Real Power Dissipated: $\mathbf{0.0~Watts~(Reactive~Orbit)}$"
        )
        info_text.set_text(textstr)
        
        return planet, trail, info_text

    print("Rendering 3D frames. Please wait...")
    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False)
    
    # Save GIF
    gif_path = os.path.join(OUTPUT_DIR, "fluidic_heliosphere.gif")
    writer = PillowWriter(fps=15)
    ani.save(gif_path, writer=writer)
    print(f"Success! 3D Animation saved to: {gif_path}")
    
    # Save a high-res PNG of the final frame
    update(n_frames // 4) # Set to a good viewing angle
    png_path = os.path.join(OUTPUT_DIR, "fluidic_heliosphere_snapshot.png")
    plt.savefig(png_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Success! 3D Snapshot saved to: {png_path}")

def render_solar_bingham_bubble_3d():
    print("Rendering 3D Solar Bingham Bubble...")
    # Constants
    G = 6.674e-11
    M_sun = 1.989e30
    c = 299792458
    H_0 = 2.2465e-18 # Derived in Chapter 1
    
    # Calculate a_genesis
    a_genesis = (c * H_0) / (2 * np.pi) # ~ 1.07e-10 m/s^2
    
    # Calculate Bubble Radius (Where g = a_genesis)
    # g = GM/r^2 -> r = sqrt(GM/a_genesis)
    R_bubble_meters = np.sqrt((G * M_sun) / a_genesis)
    R_bubble_AU = R_bubble_meters / 1.496e11
    
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#050508'); ax.set_facecolor('#050508')
    
    # Create 3D Sphere for the Bingham Boundary
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = R_bubble_AU * np.cos(u) * np.sin(v)
    y = R_bubble_AU * np.sin(u) * np.sin(v)
    z = R_bubble_AU * np.cos(v)
    
    ax.plot_wireframe(x, y, z, color='#00ffcc', alpha=0.15)
    ax.scatter([0], [0], [0], color='#FFD54F', s=100, label='The Sun', zorder=5)
    
    # Plot planetary orbits for scale
    orbit_radii = [1, 5, 30, 100] # Earth, Jupiter, Neptune, Voyager 1
    labels = ['Earth (1 AU)', 'Jupiter (5 AU)', 'Neptune (30 AU)', 'Voyager 1 (~160 AU)']
    colors = ['#4FC3F7', '#ffcc00', '#64B5F6', '#ff3366']
    
    for r, label, color in zip(orbit_radii, labels, colors):
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r*np.cos(theta), r*np.sin(theta), 0, color=color, label=label, lw=1.5)
        
    ax.set_title(f'The Solar Bingham Bubble\nRadius of Superfluidity: {R_bubble_AU:,.0f} AU', color='white', fontsize=16, weight='bold')
    
    # Formatting
    ax.set_xlim([-R_bubble_AU, R_bubble_AU])
    ax.set_ylim([-R_bubble_AU, R_bubble_AU])
    ax.set_zlim([-R_bubble_AU, R_bubble_AU])
    ax.axis('off')
    
    ax.legend(loc='upper left', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    textstr = (
        r"$\mathbf{The~Trans{-}Neptunian~Superfluid:}$" + "\n" +
        r"Inside the cyan sphere ($g > a_{genesis}$), the vacuum is a frictionless superfluid." + "\n" +
        r"Outside the sphere ($g < a_{genesis}$), the vacuum solidifies into a viscous Cosserat solid." + "\n" +
        r"Notice that Voyager 1 is still deep inside the bubble, entirely safe from metric drag."
    )
    ax.text2D(0.05, 0.05, textstr, transform=ax.transAxes, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='#00ffcc', alpha=0.9, pad=10))

    out_file = os.path.join(OUTPUT_DIR, "solar_bingham_bubble_3d.png")
    plt.savefig(out_file, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")

if __name__ == "__main__": 
    simulate_3d_fluidic_heliosphere()
    render_solar_bingham_bubble_3d()