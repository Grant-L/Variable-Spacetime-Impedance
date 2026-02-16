import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def simulate_vcfd_cavity():
    print("Simulating VCFD Cavity with Exact nu_vac...")
    NX, NY, NT, NIT = 41, 41, 500, 50
    C = 1.0; DX = 2.0 / (NX - 1); DY = 2.0 / (NY - 1)
    RHO = 1.0
    
    # EXACT DERIVATION: nu_vac = alpha * c * l_node
    ALPHA = 1.0 / 137.036
    L_NODE = DX # Normalized to grid
    NU = ALPHA * C * L_NODE
    DT = 0.001

    u = np.zeros((NY, NX)); v = np.zeros((NY, NX))
    p = np.zeros((NY, NX)); b = np.zeros((NY, NX))

    for n in range(NT):
        b[1:-1, 1:-1] = (RHO * (1 / DT * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * DX) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * DY)) -
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * DX))**2 -
                        2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * DY) * (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * DX)) -
                        ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * DY))**2))

        for it in range(NIT):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * DY**2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * DX**2) /
                            (2 * (DX**2 + DY**2)) - DX**2 * DY**2 / (2 * (DX**2 + DY**2)) * b[1:-1, 1:-1])
            p[:, -1] = p[:, -2]; p[0, :] = p[1, :]; p[:, 0] = p[:, 1]; p[-1, :] = 0

        un = u.copy(); vn = v.copy()
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] - un[1:-1, 1:-1] * DT / DX * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * DT / DY * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         DT / (2 * RHO * DX) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         NU * (DT / DX**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         DT / DY**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - un[1:-1, 1:-1] * DT / DX * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * DT / DY * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         DT / (2 * RHO * DY) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         NU * (DT / DX**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                         DT / DY**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :] = 0; u[:, 0] = 0; u[:, -1] = 0; u[-1, :] = 1.0 
        v[0, :] = 0; v[-1, :] = 0; v[:, 0] = 0; v[:, -1] = 0

    x = np.linspace(0, 2, NX); y = np.linspace(0, 2, NY); X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(11, 7), dpi=150); fig.patch.set_facecolor('#050508'); plt.gca().set_facecolor('#050508')
    plt.streamplot(X, Y, u, v, density=1.5, linewidth=1, arrowsize=1.5, arrowstyle='->', color='w')
    contour = plt.contourf(X, Y, p, alpha=0.8, cmap='viridis'); cbar = plt.colorbar(contour)
    cbar.set_label('Vacuum Potential (Pressure)', color='white'); cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    plt.title(r'VCFD Benchmark: Kinematic Viscosity $\nu_{vac} = \alpha \cdot c \cdot l_{node}$', fontsize=14, color='white')
    plt.xlabel('Lattice X', color='white'); plt.ylabel('Lattice Y', color='white'); plt.tick_params(colors='white')
    plt.text(1.0, 1.0, "Stable Vortex Core\n(Topological Matter)", ha='center', va='center', color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))
    filepath = os.path.join(OUTPUT_DIR, "lid_driven_cavity.png"); plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight'); plt.close(); print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_vcfd_cavity()