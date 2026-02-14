import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = "manuscript/chapters/09_viscous_dynamics_dark_matter/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_3d_shield():
    print("Generating 3D Rheological Shield Visualization...")
    
    # 1. Grid Setup
    N = 50
    L = 10.0 # Scale length (AU approx)
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    z = np.linspace(-L/2, L/2, N//2) # Flat slice for visibility
    X, Y, Z = np.meshgrid(x, y, z)
    
    # 2. Field Calculation
    # r = Distance from Star
    R = np.sqrt(X**2 + Y**2 + Z**2)
    R[R < 0.5] = 0.5 # Avoid singularity
    
    # Shear Rate ~ 1/r^1.5 (Keplerian derivative)
    # We normalize constants so that Critical Shear is at R=5
    gamma_dot = 1.0 / (R**1.5)
    gamma_c = 1.0 / (5.0**1.5) # Transition at R=5 units
    
    # Viscosity Field (Bingham Plastic)
    # Eta = 1 / (1 + (shear/critical)^2)
    Eta = 1.0 / (1.0 + (gamma_dot / gamma_c)**4) # Steep transition for visual clarity
    
    # 3. Visualization (Slice Plot)
    plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')
    
    # We plot the viscosity as a color map on the Z=0 plane
    # But represented as a surface height to show the "Well"
    
    Z_plane = X[:,:,N//4]
    Y_plane = Y[:,:,N//4]
    Eta_plane = Eta[:,:,N//4]
    
    # Plot Surface
    surf = ax.plot_surface(Z_plane, Y_plane, Eta_plane, cmap='viridis', 
                           linewidth=0, antialiased=False, alpha=0.9)
    
    # Add the "Star"
    ax.scatter([0], [0], [0], color='yellow', s=500, label='Star (High Shear Source)')
    
    # Add "Planet" Orbit
    theta = np.linspace(0, 2*np.pi, 100)
    orbit_r = 3.0
    ax.plot(orbit_r*np.cos(theta), orbit_r*np.sin(theta), np.zeros_like(theta), 
            color='white', linestyle='--', linewidth=2, label='Planetary Orbit (Superfluid Zone)')
    
    # Labels
    ax.set_zlabel(r'Vacuum Viscosity ($\eta$)')
    ax.set_title('The Rheological Shield: Viscosity Well around a Star', fontsize=16)
    
    # Custom Legend
    # We add text annotations instead of a messy 3D legend
    ax.text2D(0.05, 0.95, "Yellow: High Viscosity (Dark Matter)", transform=ax.transAxes, color='yellow')
    ax.text2D(0.05, 0.90, "Blue: Zero Viscosity (Superfluid)", transform=ax.transAxes, color='darkblue')
    
    # Dark Theme
    ax.set_facecolor('#111111')
    ax.xaxis.pane.set_color((0.1, 0.1, 0.1, 1.0))
    ax.yaxis.pane.set_color((0.1, 0.1, 0.1, 1.0))
    ax.zaxis.pane.set_color((0.1, 0.1, 0.1, 1.0))
    
    # Camera Angle
    ax.view_init(elev=45, azim=45)
    
    output_path = os.path.join(OUTPUT_DIR, "rheology_3d_well.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    simulate_3d_shield()