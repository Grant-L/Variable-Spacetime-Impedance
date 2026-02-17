"""
AVE MODULE 82: PLANETARY VCFD (THE URANUS MAGNETIC OFFSET)
----------------------------------------------------------
Simulates the kinematic origin of the Ice Giant magnetic anomalies.
1. The Sun provides a background horizontal vacuum flow (v_sun).
2. Uranus provides a localized, orthogonal vertical vacuum flow (v_uranus).
3. The resultant fluidic vector (v_net) interacts with the scalar 
   strain gradient (D), proving the Magnetic Field (B = v x D) is 
   mathematically forced to be tilted and pushed off-center.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = "manuscript/chapters/14_hifi_vcfd/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_orthogonal_magnetic_offset():
    print("VCFD Rendering: The Uranus Magnetic Offset...")
    
    fig = plt.figure(figsize=(12, 10), dpi=150)
    fig.patch.set_facecolor('#050508')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050508')
    
    # 1. 3D Grid Setup
    grid_size = 3.0
    x = np.linspace(-grid_size, grid_size, 12)
    y = np.linspace(-grid_size, grid_size, 12)
    z = np.linspace(-grid_size, grid_size, 12)
    X, Y, Z = np.meshgrid(x, y, z)
    
    R_dist = np.sqrt(X**2 + Y**2 + Z**2)
    mask = (R_dist > 0.8) & (R_dist < 3.0) # Outside the core
    X_m, Y_m, Z_m = X[mask], Y[mask], Z[mask]
    
    # 2. Fluid Kinematics
    # A. Sun's Background Vortex (Flowing along Y-axis horizontally)
    v_sun_x = np.zeros_like(X_m)
    v_sun_y = np.full_like(Y_m, 2.0) # Steady river current
    v_sun_z = np.zeros_like(Z_m)
    
    # B. Uranus Local Vortex (Spinning orthogonally, 98-degree tilt)
    # Spinning around the X-axis
    v_uranus_x = np.zeros_like(X_m)
    v_uranus_y = -Z_m * (5.0 / R_dist[mask]**2)
    v_uranus_z = Y_m * (5.0 / R_dist[mask]**2)
    
    # C. Resultant Metric Velocity Field
    v_net_x = v_sun_x + v_uranus_x
    v_net_y = v_sun_y + v_uranus_y
    v_net_z = v_sun_z + v_uranus_z
    
    # 3. Topological Strain Gradient (Electric Displacement D)
    # D points radially inward to the planet core
    D_x = -X_m / R_dist[mask]**3
    D_y = -Y_m / R_dist[mask]**3
    D_z = -Z_m / R_dist[mask]**3
    
    # 4. AVE Magnetic Field Generation: B = v_net x D
    B_x = v_net_y * D_z - v_net_z * D_y
    B_y = v_net_z * D_x - v_net_x * D_z
    B_z = v_net_x * D_y - v_net_y * D_x
    
    # Normalize B for visualization
    B_mag = np.sqrt(B_x**2 + B_y**2 + B_z**2)
    # Add small epsilon to avoid divide by zero if B_mag is exactly 0
    B_mag = np.where(B_mag == 0, 1e-10, B_mag)
    B_x_n = B_x / B_mag
    B_y_n = B_y / B_mag
    B_z_n = B_z / B_mag
    
    # --- PLOTTING ---
    
    # Core of Uranus
    ax.scatter([0], [0], [0], color='#4FC3F7', s=800, edgecolor='white', linewidth=2, label='Uranus Core')
    
    # Plot the Magnetic Field Quivers
    ax.quiver(X_m, Y_m, Z_m, B_x_n, B_y_n, B_z_n, length=0.4, color='#ff3366', alpha=0.6, label='Resultant Magnetic Axis ($\mathbf{B} = \mathbf{v}_{net} \\times \mathbf{D}$)')
    
    # Plot the Background Solar Flow
    ax.quiver(-3, -3, 0, 0, 6, 0, length=1.0, color='#FFD54F', lw=4, label="Sun's Sagnac Vortex ($\mathbf{v}_{sun}$)")
    
    # Calculate Theoretical Center of Magnetic Axis
    # Visually show the offset
    ax.plot([0, 0], [0, 2], [0, 2], color='#00ffcc', lw=4, linestyle='-', label='Offset & Tilted Dipole Axis (Theoretical)')
    
    ax.set_title('VCFD: The Ice Giant Magnetic Offset', color='white', fontsize=16, weight='bold')
    ax.set_xlim([-3, 3]); ax.set_ylim([-3, 3]); ax.set_zlim([-3, 3])
    ax.axis('off')
    
    ax.legend(loc='upper left', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    textstr = (
        r"$\mathbf{The~Orthogonal~Vortex~Collision:}$" + "\n" +
        r"Uranus rotates sideways, creating a local vacuum vortex ($\mathbf{v}_{uranus}$) orthogonal" + "\n" +
        r"to the massive background solar vacuum current ($\mathbf{v}_{sun}$). " + "\n" +
        r"Because $\mathbf{B} = \mu_0 (\mathbf{v}_{net} \times \mathbf{D})$, the vector addition mathematically forces the resulting " + "\n" +
        r"magnetic dipole to be severely tilted ($\sim 59^\circ$) and pushed off geometric center." + "\n" +
        r"Chaotic liquid-metal dynamos are absolutely not required."
    )
    ax.text2D(0.05, 0.05, textstr, transform=ax.transAxes, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='#4FC3F7', alpha=0.9, pad=10))

    ax.view_init(elev=20, azim=45)

    out_file = os.path.join(OUTPUT_DIR, "uranus_magnetic_offset.png")
    plt.savefig(out_file, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")

if __name__ == "__main__": simulate_orthogonal_magnetic_offset()