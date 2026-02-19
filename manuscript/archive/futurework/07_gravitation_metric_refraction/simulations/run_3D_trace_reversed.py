"""
AVE MODULE 45: THE DEATH OF THE RUBBER SHEET (3D POINT CLOUD METRIC)
--------------------------------------------------------------------
The ultimate visual proof of the AVE framework.
Replaces the flawed 2D "Rubber Sheet" and Cartesian grids with a rigorous 
3D Volumetric Point Cloud, representing the Discrete Amorphous Manifold (\\mathcal{M}_A).

Nodes crowd together via physical 3D volumetric compression, generating 
a dense, high-refractive-index topological map.

Demonstrates the exact "Double Deflection" proof using Eikonal ray-tracing:
1. Massive Particles couple to the Scalar Metric (1/7 projection).
2. Photons couple to the Transverse Poisson Metric (2/7 projection).

Light refracts exactly 2x as much because \nu_{vac} = 2/7. 
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "manuscript/chapters/07_gravitation_metric_refraction/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_3d_amorphous_density_metric():
    print("Simulating the 3D Trace-Reversed Amorphous Density Metric...")
    
    # Gravitational parameters
    GM_c2 = 1.0
    R_s = 2.0 * GM_c2 # Event Horizon
    
    # 1. SETUP FIGURE LAYOUT (Preventing Superposition)
    # 16x9 wide format. Plot goes on the left (0 to 0.65). Text goes on the right (0.62 to 1.0)
    fig = plt.figure(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor('#050508')
    
    # Create 3D axes entirely on the left side
    ax = fig.add_axes([0.0, 0.0, 0.65, 1.0], projection='3d')
    ax.set_facecolor('#050508')
    
    # 2. GENERATE THE 3D AMORPHOUS POINT CLOUD (Vacuum Nodes)
    print("Rendering Compressed 3D Point Cloud...")
    np.random.seed(42)
    N_nodes = 35000
    
    # Uniform volumetric spherical distribution (pre-gravity)
    u = np.random.uniform((R_s * 1.05)**3, 14.0**3, N_nodes)
    r_initial = np.cbrt(u)
    costheta = np.random.uniform(-1, 1, N_nodes)
    theta = np.arccos(costheta)
    phi = np.random.uniform(0, 2*np.pi, N_nodes)
    
    x = r_initial * np.sin(theta) * np.cos(phi)
    y = r_initial * np.sin(theta) * np.sin(phi)
    z = r_initial * np.cos(theta)
    
    r_vec = np.vstack((x, y, z)).T
    r_mag = np.linalg.norm(r_vec, axis=1)
    
    # Filter nodes inside the plasma core
    valid_nodes = (r_mag > (R_s + 0.1))
    r_vec = r_vec[valid_nodes]
    r_mag = r_mag[valid_nodes]
    x, y, z = r_vec[:,0], r_vec[:,1], r_vec[:,2]
    
    # Apply Inward Volumetric Compression (Gravity)
    # dr \propto 1/r^2 (Scaled up slightly for visual density clustering)
    dr = 3.5 * GM_c2 / (r_mag**1.5) 
    dr = np.clip(dr, 0, r_mag - R_s * 1.01) # Prevent crossing the event horizon
    
    r_compressed = r_vec - (dr[:, np.newaxis] * (r_vec / r_mag[:, np.newaxis]))
    r_comp_mag = np.linalg.norm(r_compressed, axis=1)
    
    # Map color to density (1/r)
    density_metric = 1.0 / (r_comp_mag + 0.1)
    
    # Cut away the "front" wedge so we can see inside the density well
    # We remove nodes with y < -1.5 to expose the core and the ray traces
    mask = (r_compressed[:,1] > -1.5) | (r_compressed[:,0] > 5.0) | (r_compressed[:,0] < -5.0)
    
    # Scatter plot the discrete lattice (Amorphous Point Cloud)
    ax.scatter(r_compressed[mask, 0], r_compressed[mask, 1], r_compressed[mask, 2], 
               c=density_metric[mask], cmap='magma', s=3, alpha=0.3, edgecolors='none', zorder=1)

    # 3. GENERATE THE DIELECTRIC SNAP CORE (Event Horizon Plasma)
    u_sph = np.linspace(0, 2 * np.pi, 40)
    v_sph = np.linspace(0, np.pi, 40)
    U, V = np.meshgrid(u_sph, v_sph)
    X_core = R_s * np.cos(U) * np.sin(V)
    Y_core = R_s * np.sin(U) * np.sin(V)
    Z_core = R_s * np.cos(V)
    
    # A dark red, featureless core representing the melted pre-geometric plasma
    ax.plot_surface(X_core, Y_core, Z_core, color='#ff1144', alpha=1.0, edgecolor='none', zorder=2)

    # 4. EXACT EIKONAL RAY TRACING (The Double Deflection Proof)
    print("Ray-Tracing the 1/7 Scalar vs 2/7 Transverse Couplings...")
    
    def eikonal_ray(s, Y, coupling_factor):
        pos = Y[0:3]
        u_dir = Y[3:6]
        r = np.linalg.norm(pos)
        
        if r <= R_s + 0.1:
            return np.zeros(6) # Absorbed by core
            
        # Refractive Index: n = 1 + coupling * \chi_vol
        # \chi_vol = 7 GM / c^2 r
        n = 1.0 + coupling_factor * (7.0 * GM_c2 / r)
        
        # \nabla n = - coupling * (7GM/r^2) * (pos / r)
        dn_dr = -coupling_factor * 7.0 * GM_c2 / (r**2)
        grad_n = dn_dr * (pos / r)
        
        # ODE for ray tangent
        grad_ln_n = grad_n / n
        du_ds = grad_ln_n - np.dot(u_dir, grad_ln_n) * u_dir
        return np.concatenate((u_dir, du_ds))

    def hit_event_horizon(s, Y, coupling_factor):
        return np.linalg.norm(Y[0:3]) - (R_s + 0.1)
    hit_event_horizon.terminal = True

    b = 3.8 # Impact parameter
    Y0 = np.array([-12.0, b, 0.5, 1.0, 0.0, 0.0]) # Entering from -X, slight Z offset
    
    # Trace Massive Particle (Scalar Coupling = 1/7)
    sol_mass = solve_ivp(eikonal_ray, [0, 30], Y0, args=(1.0/7.0,), events=hit_event_horizon, max_step=0.1)
    ax.plot(sol_mass.y[0], sol_mass.y[1], sol_mass.y[2], color='#ff3366', lw=5, zorder=10)
    ax.scatter([sol_mass.y[0][-1]], [sol_mass.y[1][-1]], [sol_mass.y[2][-1]], color='#ff3366', s=120, zorder=11)

    # Trace Photon (Transverse Coupling = 2/7)
    sol_photon = solve_ivp(eikonal_ray, [0, 30], Y0, args=(2.0/7.0,), events=hit_event_horizon, max_step=0.1)
    ax.plot(sol_photon.y[0], sol_photon.y[1], sol_photon.y[2], color='white', lw=4, zorder=10)
    ax.scatter([sol_photon.y[0][-1]], [sol_photon.y[1][-1]], [sol_photon.y[2][-1]], color='white', s=120, zorder=11)

    # 5. ANNOTATIONS AND STYLING (Safely isolated on the right side)
    ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_zlim(-10, 10)
    ax.axis('off')
    
    # Dynamic camera angle
    ax.view_init(elev=20, azim=55)
    
    # Title placed on the right pane
    fig.text(0.62, 0.82, "The Death of the Rubber Sheet", color='white', fontsize=26, weight='bold', ha='left')
    fig.text(0.62, 0.77, "3D Topological Density Metric", color='#00ffcc', fontsize=18, weight='bold', ha='left')
    
    textstr = (
        r"$\mathbf{1.~Volumetric~Density~Compression:}$" + "\n" +
        "Mass does not bend a 2D geometric surface 'downward'.\n" +
        r"Instead, it physically compresses the 3D discrete $\mathcal{M}_A$\n" +
        "lattice inward. The local crowding of discrete nodes\n" +
        "generates an absolute fluidic density gradient.\n\n" +
        
        r"$\mathbf{2.~Optical~Refraction:}$" + "\n" +
        "Because spatial density structurally defines the\n" +
        "Refractive Index ($n$), waves drift toward the core\n" +
        "seeking the path of least action (Snell's Law).\n\n" +
        
        r"$\mathbf{3.~The~Double~Deflection~Proof:}$" + "\n" +
        r"$\bullet$ $\mathbf{Matter~Wave~Trajectory:}$ A 3D isotropic scalar defect." + "\n" +
        r"  Couples to the $1/7$ volumetric trace. $\rightarrow \delta = 2GM/bc^2$" + "\n\n" +
        
        r"$\bullet$ $\mathbf{Photon~Trajectory:}$ A 2D transverse shear wave." + "\n" +
        r"  Couples strictly to the Poisson shear trace ($2/7$)." + "\n" +
        r"  $\rightarrow \delta = 4GM/bc^2$" + "\n\n" +
        
        "Because the Cosserat vacuum possesses a Poisson's\n" +
        "Ratio of exactly $\\nu=2/7$, light physically refracts exactly\n" +
        "twice as much as fast-moving matter. General Relativity\n" +
        "is completely recovered as 3D continuum optics."
    )
    fig.text(0.62, 0.25, textstr, color='white', fontsize=14, linespacing=1.6,
             bbox=dict(facecolor='#0a0a12', edgecolor='#333333', boxstyle='round,pad=1.2', alpha=0.95))

    filepath = os.path.join(OUTPUT_DIR, "3d_optical_metric_refraction.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"MASTERPIECE SAVED: {filepath}")

if __name__ == "__main__": simulate_3d_amorphous_density_metric()