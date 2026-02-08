import numpy as np
import matplotlib.pyplot as plt

def simulate_proton_triplet():
    print("Initializing Vacuum Lattice...")
    
    # 1. Setup Grid
    N = 200
    L = 20.0
    dx = L / N
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)

    # 2. Initialize 3 Vortices (Quarks) in a Triangle
    # Radius of the "molecule"
    r = 4.0
    
    # Coordinates for the triplet (Top, Bottom-Left, Bottom-Right)
    # We use 3 specific angles to form an equilateral triangle
    angles = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
    points = [(r * np.cos(a), r * np.sin(a)) for a in angles]
    
    # Superpose phase windings
    theta = np.zeros_like(X)
    for (px, py) in points:
        theta += np.arctan2(Y - py, X - px)
            
    # Create the Order Parameter (Psi)
    psi = np.ones((N, N)) * np.exp(1j * theta)
    
    # 3. Time Evolution (Stable Ginzburg-Landau)
    # FIX: Lower dt to ensure numerical stability (dt < dx^2/4)
    # dx = 0.1, dx^2 = 0.01, so dt must be < 0.0025
    dt = 0.001 
    steps = 2000 # Increased steps since dt is smaller
    
    print(f"Relaxing Vacuum Field for {steps} steps (Stable dt={dt})...")
    
    for i in range(steps):
        # 5-point Laplacian Stencil
        # np.roll allows us to calculate neighbors efficiently
        lap = (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) + 
               np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4*psi) / (dx**2)
        
        # Ginzburg-Landau Equation
        psi += dt * (lap + psi * (1.0 - np.abs(psi)**2))

    # 4. Generate Output Plots
    print("Generating Plots...")
    plt.figure(figsize=(12, 5))

    # Plot 1: Vacuum Density (|Psi|^2)
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(psi)**2, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.title("Vacuum Density $|\\psi|^2$\n(Dark Spots = Quark Cores)")
    plt.colorbar(label="Superfluid Density")
    plt.xlabel("Lattice Units")
    plt.ylabel("Lattice Units")

    # Plot 2: Phase Topology (Angle)
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(psi), extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='twilight')
    plt.title("Phase Topology $\\theta$\n(Color Gradients = Phase Tension/Gluons)")
    plt.colorbar(label="Phase (Radians)")
    plt.xlabel("Lattice Units")
    plt.ylabel("Lattice Units")

    plt.suptitle(f"The Lindblom 'Proton': Stable Triplet (Step {steps})", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_proton_triplet()