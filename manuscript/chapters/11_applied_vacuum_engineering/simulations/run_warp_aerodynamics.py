import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/11_applied_vacuum_engineering/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_metric_streamlining():
    print("Simulating Vacuum Aerodynamics (Relativistic Flow)...")
    
    # 1. SETUP GRID
    # We simulate a 2D cross-section of the vacuum
    NX, NY = 200, 100
    X, Y = np.meshgrid(np.linspace(-2, 6, NX), np.linspace(-2, 2, NY))
    
    # 2. DEFINE THE PROJECTILE (The "Hull")
    # A simple ellipse at the origin
    R_hull = np.sqrt(X**2 + (2*Y)**2)
    hull_mask = R_hull < 0.5
    
    # 3. DEFINE THE FLOW (Relativistic Frame)
    # In the ship's frame, the vacuum flows past at velocity v
    v_flow = 0.9 # 0.9c (Relativistic)
    
    # 4. CALCULATE VACUUM PRESSURE (P_vac)
    # High velocity impact creates a "Bow Shock" of compressed lattice density
    # P ~ v^2 (Bernoulli Stagnation Pressure)
    
    # Radial distance from nose (X=-0.5, Y=0)
    R_nose = np.sqrt((X+0.5)**2 + Y**2)
    
    # Standard Shock (Blunt Body)
    # Pressure piles up in front
    shock_envelope = np.exp(-R_nose*2) * np.cos(3*np.arctan2(Y, X+0.5))
    P_standard = v_flow**2 * np.clip(shock_envelope, 0, 1)
    
    # 5. CALCULATE METRIC STREAMLINING (The "Dimple")
    # We apply a "Metric Actuator" beam ahead of the ship
    # This beam lowers the local viscosity (Shear Thinning)
    
    # Beam Geometry
    beam_mask = (X < -0.6) & (np.abs(Y) < 0.2)
    
    # Modified Pressure Field
    # The beam "liquefies" the shock, reducing stagnation pressure
    P_streamlined = P_standard.copy()
    P_streamlined[beam_mask] *= 0.1 # 90% Viscosity Reduction
    
    # 6. VISUALIZATION
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='#111111')
    
    # Plot A: Standard Relativistic Drag (The Wall)
    ax1 = axes[0]
    c1 = ax1.contourf(X, Y, P_standard, 50, cmap='inferno')
    ax1.add_patch(plt.Circle((0,0), 0.5, color='gray', label='Ship Hull'))
    ax1.set_title("Standard Flight (v = 0.9c)\nMassive Vacuum Bow Shock (High Inertia)", color='white')
    ax1.axis('off')
    
    # Plot B: Metric Streamlining (The Bubble)
    ax2 = axes[1]
    c2 = ax2.contourf(X, Y, P_streamlined, 50, cmap='inferno')
    ax2.add_patch(plt.Circle((0,0), 0.5, color='gray'))
    
    # Draw the Actuator Beam
    ax2.plot([-2, -0.6], [0, 0], 'c--', linewidth=2, label='Metric Actuator Beam')
    
    ax2.set_title("Metric Streamlining (Active Flow Control)\nViscosity Reduced by Shear Beam (Low Inertia)", color='white')
    ax2.axis('off')
    ax2.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "warp_aerodynamics.png")
    plt.savefig(output_path, dpi=300, facecolor='#111111')
    print(f"Simulation saved to {output_path}")

if __name__ == "__main__":
    simulate_metric_streamlining()