"""
AVE Black Hole Event Horizon CFD Simulation
Simulates an extreme mass concentration drawing the Bingham fluid inward.
Proves that an "Event Horizon" is natively a Superfluid Phase Boundary
where the metric yields under extreme radial tension.
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
src_dir = Path(__file__).parent.parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ave.solvers.grid_3d import EulerianGrid3D
from ave.solvers.bingham_cfd import BinghamFluidSolver

def apply_spherical_sink(grid: EulerianGrid3D, center_xyz, mass_factor):
    """
    Applies a continuous radial inward velocity gradient 
    simulating extreme gravitational flux draining through a topological defect.
    """
    X, Y, Z = grid.get_mesh()
    rx = X - center_xyz[0]
    ry = Y - center_xyz[1]
    
    r_mag = np.sqrt(rx**2 + ry**2 + 1e-6) # Prevent div by 0
    
    # Gravitational classical acceleration scales as 1/r^2
    # In a fluid, this induces a velocity sink profile v_r ~ -1/r
    v_radial = -mass_factor / r_mag
    
    # Restrict the mathematically infinite singularity to a finite core
    core_cutoff = 0.5
    v_radial[r_mag < core_cutoff] = -mass_factor / core_cutoff
    
    # Decompose into x and y vectors and inject into grid
    grid.vx += v_radial * (rx / r_mag)
    grid.vy += v_radial * (ry / r_mag)
    
    return grid

def simulate_black_hole_horizon():
    print("==================================================")
    print("AVE COMPUTATIONAL FLUID DYNAMICS: EVENT HORIZON")
    print("==================================================\n")
    
    grid_size = (20.0, 20.0, 2.0)
    res = (150, 150, 1)
    
    print(f"[+] Allocating Eulerian Grid Fields {grid_size} meters...")
    grid = EulerianGrid3D(grid_size, res, dt=1e-8)
    solver = BinghamFluidSolver(grid)
    
    center = (0.0, 0.0, 0.0)
    mass_drain_factor = 5e7 # Equivalent to extreme shear
    
    print("[+] Injecting Topological Mass Sink (Black Hole)...")
    
    for step in range(5):
        grid = apply_spherical_sink(grid, center, mass_drain_factor)
        solver.step()
        
    print("\n[+] Extracting Bingham Phase Boundary...")
    X, Y, Z = grid.get_mesh()
    x_2d = X[:, :, 0]
    y_2d = Y[:, :, 0]
    
    nu_2d = solver.nu_field[:, :, 0]
    nu_normalized = nu_2d / solver.nu_solid
    log_nu = np.log10(nu_normalized + 1e-15)
    
    vx = grid.vx[:, :, 0]
    vy = grid.vy[:, :, 0]
    v_mag = np.sqrt(vx**2 + vy**2)
    
    # Mask out the unphysical grid singularity
    r_mag = np.sqrt(x_2d**2 + y_2d**2)
    log_nu[r_mag < 0.5] = 0.0 
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Sink Vector Field
    strm = ax1.streamplot(x_2d.T, y_2d.T, vx.T, vy.T, color=v_mag.T, cmap='inferno', density=1.5, linewidth=1.2)
    ax1.set_title("Extreme Metric Flux Sink (Gravity)", fontsize=16, pad=15)
    ax1.set_xlabel("x (meters)")
    ax1.set_ylabel("y (meters)")
    ax1.set_aspect('equal')
    fig.colorbar(strm.lines, ax=ax1, fraction=0.046, pad=0.04).set_label('Inward Fluid Velocity [m/s]', rotation=270, labelpad=20)
    
    # Plot 2: The Event Horizon (Yield Boundary)
    cmap = plt.get_cmap('ocean').reversed()
    img = ax2.pcolormesh(x_2d, y_2d, log_nu, shading='gouraud', cmap=cmap, vmin=-12, vmax=0)
    ax2.set_title("The Event Horizon as a Fluid Yield Rupture", fontsize=16, pad=15)
    ax2.set_xlabel("x (meters)")
    ax2.set_aspect('equal')
    fig.colorbar(img, ax=ax2, fraction=0.046, pad=0.04).set_label('$\\log_{10}(\\eta_{eff} / \\eta_{solid})$', rotation=270, labelpad=20)
    
    # Find the rough radius where nu collapses (log_nu < -2)
    # This represents the boundary where the Cosserat solid breaks into a superfluid
    horizon_radius = None
    center_idx = res[0] // 2
    for i in range(center_idx, res[0]):
        if log_nu[i, center_idx] > -2.0:
            horizon_radius = x_2d[i, center_idx]
            break
            
    if horizon_radius:
        horizon_circle = plt.Circle((0, 0), horizon_radius, color='r', fill=False, linestyle='-', linewidth=2, alpha=0.8)
        ax2.add_patch(horizon_circle)
        ax2.annotate('Dielectric Rupture Boundary', xy=(horizon_radius, horizon_radius), color='white', weight='bold')

    plt.tight_layout()
    
    out_dir = Path('assets/sim_outputs')
    outpath = out_dir / 'black_hole_cfd.png'
    plt.savefig(outpath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"\n[+] Visualization successfully written to {outpath}")
    print("==================================================")

if __name__ == "__main__":
    simulate_black_hole_horizon()
