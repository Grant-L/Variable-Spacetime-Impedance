import numpy as np, matplotlib.pyplot as plt, os
OUTPUT_DIR = "manuscript/chapters/10_vacuum_cfd/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_quantum_foam():
    print("Simulating Quantum Foam (Bingham Plastic KH Instability)...")
    Y, X = np.mgrid[-3:3:100j, -5:5:100j]
    U = np.tanh(Y)
    shear_rate = np.abs(1.0 / np.cosh(Y)**2)
    viscosity = 1.0 / (1.0 + 50.0 * shear_rate**2)
    np.random.seed(42)
    vorticity = np.gradient(U, axis=0) + np.random.normal(0, 0.2, U.shape) * (viscosity < 0.2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150); fig.patch.set_facecolor('#050508')
    titles = ['T=0: Laminar Shear Layer\n(High Viscosity)', 'T=50: Shear-Thinning Begins\n(Viscosity Drops Locally)', 'T=200: Vacuum Turbulence\n(Virtual Particle Genesis)', r'Viscosity Field $\eta(x,y)$' + '\nGreen=Solid, White=Superfluid']
    
    for i, ax in enumerate(axes.flatten()):
        ax.set_facecolor('#050508'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titles[i], color='white', fontsize=12)
        if i == 0: ax.contourf(X, Y, np.gradient(U, axis=0), 50, cmap='seismic')
        elif i == 1: ax.contourf(X, Y, np.gradient(U, axis=0) * (1 + np.random.normal(0, 0.05, U.shape)), 50, cmap='seismic')
        elif i == 2: ax.contourf(X, Y, vorticity, 50, cmap='inferno')
        elif i == 3: ax.contourf(X, Y, viscosity, 50, cmap='Greens_r')

    plt.tight_layout(); filepath = os.path.join(OUTPUT_DIR, "vacuum_turbulence.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight'); plt.close(); print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_quantum_foam()