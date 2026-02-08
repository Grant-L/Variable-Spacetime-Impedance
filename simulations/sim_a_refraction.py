import numpy as np
import matplotlib.pyplot as plt

def run_refraction_sim():
    print("LCT Simulation A: Gravitational Refraction")
    # Grid Setup
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Mass at center creates Impedance Gradient
    R = np.sqrt(X**2 + Y**2)
    Z0_vacuum = 377.0
    # Impedance increases near mass (loading)
    Z_local = Z0_vacuum * (1 + 5.0 * np.exp(-R/2.0))
    
    # Refractive Index n ~ Z_local
    n = Z_local / Z0_vacuum
    
    plt.figure(figsize=(8,6))
    plt.pcolormesh(X, Y, n, shading='auto', cmap='plasma')
    plt.colorbar(label='Refractive Index $n_{eff}$')
    plt.title("Effective Refractive Geometry (Gravity)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    run_refraction_sim()