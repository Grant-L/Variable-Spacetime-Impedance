import os

sim_a_code = r"""import numpy as np
import matplotlib.pyplot as plt

def run_strain_sim():
    print("LCT Simulation A: Metric Elasticity (Gravity as Strain)")
    
    # 1. Grid Setup
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    
    # 2. Define Mass Source
    # Mass at (0,0) creates a stress load on the lattice
    R = np.sqrt(X**2 + Y**2)
    
    # 3. Calculate Strain Field (Epsilon)
    # Strain decays as 1/R from the source. 
    # We add a small regularization term (+1.0) to avoid infinity at R=0
    Strain = 1.0 / (R + 1.0)
    
    # 4. Plot
    plt.figure(figsize=(8, 6))
    
    # Plot the Strain Intensity (Heatmap)
    # Viridis colormap: Yellow = High Strain (Time Dilation), Purple = Low Strain (Flat Space)
    plt.pcolormesh(X, Y, Strain, shading='auto', cmap='viridis')
    plt.colorbar(label=r'Metric Strain $\epsilon$ (Effective Shapiro Delay)')
    
    # Add contour lines to visualize "Equipotential" surfaces
    plt.contour(X, Y, Strain, colors='white', alpha=0.3)
    
    plt.title("Effective Metric Elasticity (Gravity)")
    plt.xlabel("x (Lattice Nodes)")
    plt.ylabel("y (Lattice Nodes)")
    
    # Annotate
    plt.text(0, 0, "Mass Load", color='white', ha='center', fontweight='bold')
    plt.arrow(5, 5, -2, -2, color='white', head_width=0.3)
    # Use raw string for newlines to avoid syntax errors
    plt.text(5.5, 5.5, "Lattice Dilation\n(Signal Slows Down)", color='white')
    
    plt.show()

if __name__ == "__main__":
    run_strain_sim()
"""

# Write the file
file_path = os.path.join('simulations', 'sim_a_metric_strain.py')
with open(file_path, 'w') as f:
    f.write(sim_a_code)

print(f"Successfully fixed and created: {file_path}")