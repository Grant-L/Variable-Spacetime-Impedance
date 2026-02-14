import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
OUTPUT_DIR = "manuscript/chapters/10_vacuum_cfd/simulations"
NX = 41            # Lattice Nodes (X)
NY = 41            # Lattice Nodes (Y)
NT = 500           # Time Steps (Lattice Updates)
NIT = 50           # Pressure Solver Iterations
C = 1              # Speed of Light (Normalized Acoustic Limit)
DX = 2 / (NX - 1)  # Lattice Pitch (Normalized)
DY = 2 / (NY - 1)
RHO = 1            # Vacuum Density (mu_0)
NU = 0.1           # Vacuum Viscosity (eta_vac / rho) -> Inverse Reynolds
DT = 0.001         # Time Step

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def solve_vacuum_cavity():
    print("Initializing VCFD Lattice (Lid-Driven Cavity)...")
    
    # Field Arrays
    # u: Flux Velocity X, v: Flux Velocity Y, p: Vacuum Potential (Pressure)
    u = np.zeros((NY, NX))
    v = np.zeros((NY, NX))
    p = np.zeros((NY, NX)) 
    b = np.zeros((NY, NX))
    
    # Time Stepping (The Universal Clock)
    for n in range(NT):
        # 1. Source Term for Pressure Poisson (Divergence of intermediate velocity)
        b[1:-1, 1:-1] = (RHO * (1 / DT * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * DX) + 
                     (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * DY)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * DX))**2 -
                    2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * DY) *
                         (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * DX)) -
                    ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * DY))**2))

        # 2. Pressure Correction (Iterative Relaxation)
        # Solving the Vacuum Potential Field
        for it in range(NIT):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * DY**2 + 
                              (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * DX**2) /
                              (2 * (DX**2 + DY**2)) -
                              DX**2 * DY**2 / (2 * (DX**2 + DY**2)) * b[1:-1, 1:-1])

            # Boundary Conditions (Pressure)
            p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
            p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
            p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
            p[-1, :] = 0        # p = 0 at y = 2 (Top Lid reference)

        # 3. Velocity Update (Navier-Stokes Momentum)
        # Advection + Diffusion + Pressure Gradient
        un = u.copy()
        vn = v.copy()
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * DT / DX *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * DT / DY *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         DT / (2 * RHO * DX) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         NU * (DT / DX**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         DT / DY**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * DT / DX *
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * DT / DY *
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         DT / (2 * RHO * DY) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         NU * (DT / DX**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                         DT / DY**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # 4. Boundary Conditions (The Lid)
        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = 1    # The "Lid" moves at v = 1 (Driving the cavity)
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0
        
    return u, v, p

def plot_vcfd_results(u, v, p):
    x = np.linspace(0, 2, NX)
    y = np.linspace(0, 2, NY)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(11, 7), dpi=100)
    
    # Plot Streamlines (Flux Lines)
    plt.streamplot(X, Y, u, v, density=1.5, linewidth=1, arrowsize=1.5, arrowstyle='->', color='w')
    
    # Plot Pressure (Vacuum Potential)
    plt.contourf(X, Y, p, alpha=0.8, cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label('Vacuum Potential (Pressure)')
    
    # Styling
    plt.title('VCFD Benchmark: Lid-Driven Cavity ($Re=10$)')
    plt.xlabel('Lattice X ($l_P$)')
    plt.ylabel('Lattice Y ($l_P$)')
    
    # Add text annotation
    plt.text(1.0, 1.0, "Stable Vortex Core\n(Matter Formation)", 
             ha='center', va='center', color='white', fontweight='bold', 
             bbox=dict(facecolor='black', alpha=0.5))

    # Background fix for dark theme plots
    plt.gca().set_facecolor('#222222')
    
    output_path = os.path.join(OUTPUT_DIR, "lid_driven_cavity.png")
    plt.savefig(output_path)
    print(f"Simulation Complete. Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    ensure_output_dir()
    u, v, p = solve_vacuum_cavity()
    plot_vcfd_results(u, v, p)