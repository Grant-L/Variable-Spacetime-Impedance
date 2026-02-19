import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for batch generation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import os

# --- OUTPUT CONFIGURATION ---
OUTPUT_DIR = "assets/figures/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- PHYSICAL CONSTANTS (From Verification Phase) ---
ALPHA_INV = 137.036
V_CRIT = 1.0  # Normalized breakdown voltage (alpha)
TAU_YIELD = 1.0 # Normalized Bingham yield stress

def plot_dielectric_saturation():
    """
    Figure 1: The 4th-Order Dielectric Saturation Curve (Axiom 4).
    Standard QED assumes linear response. AVE predicts a 'Wall'.
    """
    print("Generating Figure 1: Dielectric Saturation Curve...")
    
    # Normalized Electric Field / Potential (V / V_crit)
    v = np.linspace(0, 0.99, 1000)
    
    # Linear QED (Standard Model)
    c_linear = np.ones_like(v)
    
    # AVE 4th-Order Non-Linearity (Axiom 4)
    # C_eff = C_0 / sqrt(1 - (V/V_crit)^4)
    c_ave = 1.0 / np.sqrt(1 - v**4)
    
    plt.figure(figsize=(10, 6))
    plt.plot(v, c_linear, 'k--', linewidth=2, label="Standard QED (Linear Vacuum)")
    plt.plot(v, c_ave, 'r-', linewidth=3, label="AVE Condensate (4th-Order Saturation)")
    
    plt.axvline(x=1.0, color='r', linestyle=':', label="Dielectric Breakdown Limit")
    plt.fill_between(v, c_ave, 0, color='red', alpha=0.1)
    
    plt.title("Axiom 4: Vacuum Dielectric Saturation Profile", fontsize=14)
    plt.xlabel("Topological Stress ($V / V_{crit}$)", fontsize=12)
    plt.ylabel(r"Relative Permittivity ($\epsilon_r$)", fontsize=12)
    plt.ylim(0, 10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(f"{OUTPUT_DIR}vacuum_dielectric_saturation.png", dpi=300)
    plt.close()

def plot_golden_torus():
    """
    Figure 2: The Electron as a Golden Torus (3_1 Knot).
    Visualizes the R*r = 1/4 geometry.
    """
    print("Generating Figure 2: Electron Geometry...")
    
    # Golden Ratio Parameters
    phi = (1 + np.sqrt(5)) / 2
    R = (1 + np.sqrt(5)) / 4  # ~0.809
    r = (np.sqrt(5) - 1) / 4  # ~0.309
    
    # Parametric Torus Cross-Section
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Major Circle (The Guide)
    x_main = R * np.cos(theta)
    y_main = R * np.sin(theta)
    
    plt.figure(figsize=(8, 8))
    
    # Plot Major Radius
    plt.plot(x_main, y_main, 'b--', label=f"Major Radius R={R:.3f}")
    
    # Plot Minor Radius (Tube Thickness) at cardinal points
    # Representing the flux tube cross-section
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        cx = R * np.cos(angle)
        cy = R * np.sin(angle)
        circle_x = cx + r * np.cos(theta)
        circle_y = cy + r * np.sin(theta)
        plt.plot(circle_x, circle_y, 'r-', linewidth=2)
        if angle == 0: # Label one
            plt.text(cx, cy, f"  r={r:.3f}", color='red', fontsize=12)

    plt.axis('equal')
    plt.title(r"The Electron: Golden Torus Geometry\n($R \cdot r = 1/4$)", fontsize=14)
    plt.xlabel("Normalized Distance ($l_{node}$)")
    plt.ylabel("Normalized Distance ($l_{node}$)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center')
    
    plt.savefig(f"{OUTPUT_DIR}electron_golden_torus.png", dpi=300)
    plt.close()

def plot_bingham_rheology():
    """
    Figure 3: The Dark Sector Rheology (Bingham Plastic).
    Shows the transition from Solid (Dark Matter) to Superfluid.
    """
    print("Generating Figure 3: Bingham Rheology...")
    
    stress = np.linspace(0, 2.0, 1000)
    viscosity = np.zeros_like(stress)
    
    # Bingham Model: High viscosity below yield, drops to near-zero above
    mask_solid = stress < TAU_YIELD
    mask_fluid = stress >= TAU_YIELD
    
    # Solid Phase (High Viscosity - "Dark Matter" Halo)
    viscosity[mask_solid] = 1.0 
    
    # Fluid Phase (Shear Thinning - Superfluid Slip)
    # Simple decay model for post-yield viscosity
    viscosity[mask_fluid] = 1.0 * np.exp(-5 * (stress[mask_fluid] - TAU_YIELD))
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(stress, viscosity, 'g-', linewidth=3)
    plt.axvline(x=TAU_YIELD, color='k', linestyle='--', label="Bingham Yield Stress ($\tau_{yield}$)")
    
    # Annotations
    plt.text(0.2, 0.5, "Solid Phase\n(Galactic Halo)", fontsize=12, color='green', ha='center')
    plt.text(1.5, 0.5, "Superfluid Phase\n(Planetary Orbit)", fontsize=12, color='green', ha='center')
    
    plt.title("Dark Sector: Vacuum Rheology Profile", fontsize=14)
    plt.xlabel("Applied Gravitational Shear Stress ($\tau$)", fontsize=12)
    plt.ylabel("Effective Kinematic Viscosity ($\\nu_{vac}$)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(f"{OUTPUT_DIR}bingham_rheology.png", dpi=300)
    plt.close()


def set_axes_equal(ax):
    """Make 3D axes have equal scale so spheres look like spheres."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_electron_3d():
    """Figure 4: 3D Electron as Trefoil Knot (3_1)."""
    print("Generating Figure 4: 3D Electron (Golden Torus)...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    t = np.linspace(0, 2*np.pi, 200)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    ax.plot(x, y, z, color='cyan', linewidth=8, alpha=0.8, label=r'Flux Tube ($d=l_{node}$)')
    ax.plot(x, y, z, color='blue', linewidth=1)
    ax.set_title(r"The Electron: 3D Golden Torus ($3_1$)", fontsize=16)
    set_axes_equal(ax)
    ax.axis('off')
    plt.savefig(f"{OUTPUT_DIR}electron_3d_knot.png", dpi=300, transparent=True)
    plt.close()


def plot_proton_3d():
    """Figure 5: 3D Proton as Borromean Linkage (6^3_2)."""
    print("Generating Figure 5: 3D Proton (Borromean Linkage)...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(t), np.sin(t), np.zeros_like(t)+0.2*np.sin(3*t), 'r', linewidth=5, label='Loop 1')
    ax.plot(np.zeros_like(t)+0.2*np.sin(3*t), np.cos(t), np.sin(t), 'g', linewidth=5, label='Loop 2')
    ax.plot(np.sin(t), np.zeros_like(t)+0.2*np.sin(3*t), np.cos(t), 'b', linewidth=5, label='Loop 3')
    ax.set_title(r"The Proton: Borromean Linkage ($6^3_2$)\n(Source of Tensor Mass Deficit)", fontsize=16)
    set_axes_equal(ax)
    ax.axis('off')
    plt.savefig(f"{OUTPUT_DIR}proton_borromean_3d.png", dpi=300, transparent=True)
    plt.close()


def plot_lattice_3d():
    """Figure 6: 3D Vacuum Lattice (Discrete Amorphous Condensate)."""
    print("Generating Figure 6: 3D Vacuum Lattice (Poisson-Disk)...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    np.random.seed(42)
    num_pts = 300
    x = np.random.rand(num_pts)
    y = np.random.rand(num_pts)
    z = np.random.rand(num_pts)
    ax.scatter(x, y, z, c=z, cmap='plasma', s=40, alpha=0.8)
    points = np.column_stack((x, y, z))
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=0.15)
    for i, j in list(pairs)[:200]:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'k-', alpha=0.2, linewidth=0.5)
    ax.set_title(r"The Vacuum: Discrete Amorphous Condensate ($\mathcal{M}_A$)", fontsize=16)
    ax.set_xlabel(r'X ($l_{node}$)')
    ax.set_ylabel(r'Y ($l_{node}$)')
    ax.set_zlabel(r'Z ($l_{node}$)')
    plt.savefig(f"{OUTPUT_DIR}lattice_structure_3d.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_dielectric_saturation()
    plot_golden_torus()
    plot_bingham_rheology()
    plot_electron_3d()
    plot_proton_3d()
    plot_lattice_3d()
    print("Figures generated in assets/figures/")