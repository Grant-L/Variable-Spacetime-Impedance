"""
AVE Metric Streamlining (Warp) Validation Script
Simulates a high-velocity localized mass propagating through the Bingham vacuum tensor.
Visualizes the spontaneous collapse of local viscosity (Superfluid Avalanche).
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
from ave.solvers.topological_coupling import apply_macroscopic_rotor_to_grid

def simulate_warp_wake():
    print("==================================================")
    print("AVE COMPUTATIONAL FLUID DYNAMICS: METRIC STREAMLINING")
    print("==================================================\n")
    
    # 1. Initialize a 3D Macroscopic Vacuum Volume
    # Size: 10m x 10m x 2m slice
    grid_size = (10.0, 10.0, 2.0)
    res = (100, 100, 1) # 2D slice approximation for speed
    
    print(f"[+] Allocating Eulerian Grid Fields {grid_size} meters...")
    grid = EulerianGrid3D(grid_size, res, dt=1e-8)
    
    # 2. Instantiate the Bingham Non-Linear Fluid Solver
    print("[+] Instantiating Bingham Phase-Transition Engine...")
    solver = BinghamFluidSolver(grid)
    
    # 3. Inject a spinning topological mass 
    # (High angular momentum forcibly triggers the K=2G yield rupture radially)
    rotor_center = (0.0, 0.0, 0.0)
    rotor_radius = 1.5 # meters
    
    # Needs EXTREME angular velocity to shear the vacuum metric
    # Let's say omega = [0, 0, 10^8 rad/s]
    omega = np.array([0.0, 0.0, 5e7]) 
    
    print("[+] Injecting Macroscopic Torsional Rotor...")
    grid = apply_macroscopic_rotor_to_grid(grid, rotor_center, rotor_radius, omega)
    
    # 4. Step the fluid PDE to reach steady state buckling
    print(f"[+] Advancing Navier-Stokes Topology Equations...")
    for step in range(10):
        # We hold the rotor constant, and let the fluid stress propagate
        grid = apply_macroscopic_rotor_to_grid(grid, rotor_center, rotor_radius, omega)
        solver.step()
    
    # 5. Extract and Visualize the Phase Transition
    print("\n[+] Extracting Phase Transition Cross-Section...")
    
    X, Y, Z = grid.get_mesh()
    slice_idx = 0 # It's effectively 2D
    
    x_2d = X[:, :, slice_idx]
    y_2d = Y[:, :, slice_idx]
    
    # Extract the kinematic viscosity map
    nu_2d = solver.nu_field[:, :, slice_idx]
    
    # Normalize viscosity against the rigid asymptote (Dark Matter limit)
    nu_normalized = nu_2d / solver.nu_solid
    
    # Calculate Velocity magnitude
    vx = grid.vx[:, :, slice_idx]
    vy = grid.vy[:, :, slice_idx]
    v_mag = np.sqrt(vx**2 + vy**2)
    
    # Plotting
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Velocity / Shear Vector Field
    strm = ax1.streamplot(x_2d.T, y_2d.T, vx.T, vy.T, color=v_mag.T, cmap='magma', density=1.5, linewidth=1.2)
    ax1.set_title("Macroscopic Trefoil (Warp) Fluid Strain", fontsize=16, pad=15)
    ax1.set_xlabel("x (meters)")
    ax1.set_ylabel("y (meters)")
    ax1.set_aspect('equal')
    cbar1 = fig.colorbar(strm.lines, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Fluid Velocity [m/s]', rotation=270, labelpad=20)
    
    # Plot 2: The Bingham Superfluid Avalanche
    # Because nu can span many orders of magnitude, we plot log10(nu_eff / nu_solid)
    # 0 = Completely rigid (Cosserat Solid / Machian limit)
    # Negative values = Collapsing toward frictionless superfluid
    log_nu = np.log10(nu_normalized + 1e-15)
    
    # Mask out the inside of the rotor (it's the rigid mass)
    r_mag = np.sqrt(x_2d**2 + y_2d**2)
    log_nu[r_mag <= rotor_radius] = 0.0 # Define interior as solid
    
    cmap = plt.get_cmap('ocean').reversed()
    img = ax2.pcolormesh(x_2d, y_2d, log_nu, shading='gouraud', cmap=cmap, vmin=-12, vmax=0)
    ax2.set_title("Superfluid Slipstream / Bingham Phase Transition", fontsize=16, pad=15)
    ax2.set_xlabel("x (meters)")
    ax2.set_aspect('equal')
    
    cbar2 = fig.colorbar(img, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('$\\log_{10}(\\eta_{eff} / \\eta_{solid})$ (Viscosity Collapse)', rotation=270, labelpad=20)
    
    # Aesthetics
    rotor_circle = plt.Circle((0, 0), rotor_radius, color='w', fill=False, linestyle='--', alpha=0.5)
    ax2.add_patch(rotor_circle)
    ax2.annotate('Rigid Rotor Mass', xy=(0, rotor_radius/2), ha='center', color='black', alpha=0.7, weight='bold')

    plt.tight_layout()
    
    out_dir = Path('assets/sim_outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    outpath = out_dir / 'metric_streamlining_cfd.png'
    
    plt.savefig(outpath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"\n[+] Visualization successfully written to {outpath}")
    print("==================================================")

if __name__ == "__main__":
    simulate_warp_wake()
