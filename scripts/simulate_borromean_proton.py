import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import traceback

def main():
    try:
        # Ensure output directory exists
        os.makedirs('../assets/sim_outputs', exist_ok=True)
        
        fig = plt.figure(figsize=(10, 8), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        
        # Hide standard axes for a clean spatial look
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.axis('off')
        
        t = np.linspace(0, 2*np.pi, 2000)
        
        # Parametric equations for Borromean Rings 
        # Using a 3-fold symmetric set of skewed ellipses
        R1 = 1.2
        R2 = 0.5
        Z_amp = 0.4
        
        # Ring 1 (Red) - Up Quark loop
        x1 = R1 * np.cos(t)
        y1 = R2 * np.sin(t)
        z1 = Z_amp * np.sin(3*t)
        
        # Ring 2 (Green) - Up Quark loop
        x2 = Z_amp * np.sin(3*t)
        y2 = R1 * np.cos(t)
        z2 = R2 * np.sin(t)
        
        # Ring 3 (Blue) - Down Quark loop
        x3 = R2 * np.sin(t)
        y3 = Z_amp * np.sin(3*t)
        z3 = R1 * np.cos(t)
        
        # Function to plot tubes with a glowing neon effect
        def plot_glowing_tube(x, y, z, color, core_color='white'):
            for lw, alpha in zip(np.linspace(18, 2, 6), np.linspace(0.05, 0.8, 6)):
                ax.plot(x, y, z, color=color, linewidth=lw, alpha=alpha)
            ax.plot(x, y, z, color=core_color, linewidth=1.5, alpha=1.0)
            
        # Draw the topological flux tubes
        plot_glowing_tube(x1, y1, z1, '#ff1111') # Red
        plot_glowing_tube(x2, y2, z2, '#11ff11') # Green
        plot_glowing_tube(x3, y3, z3, '#1155ff') # Blue
        
        # The Tensor Halo at the geometric center
        # Saturated structural gradient generating the proton mass
        N_points = 5000
        r = np.random.normal(0, 0.25, N_points)
        theta = np.random.uniform(0, 2*np.pi, N_points)
        phi = np.random.uniform(0, np.pi, N_points)
        
        hx = r * np.sin(phi) * np.cos(theta)
        hy = r * np.sin(phi) * np.sin(theta)
        hz = r * np.cos(phi)
        
        # Determine distance from center for color mapping
        dist = np.sqrt(hx**2 + hy**2 + hz**2)
        # Apply a sharp cutoff
        mask = dist < 0.55
        hx, hy, hz, dist = hx[mask], hy[mask], hz[mask], dist[mask]
        
        # Plasma colormap for the extreme tensor strain
        colors = plt.cm.plasma(1.0 - (dist / np.max(dist)))
        # Make the center extremely bright and opaque, edges transparent
        colors[:, 3] = np.clip(1.0 - (dist / 0.35), 0, 1) * 0.7 
        
        ax.scatter(hx, hy, hz, c=colors, s=20, edgecolor='none', alpha=0.5)
        
        # The Macroscopic Gravitational Projection Envelope (1/7 Isotropic Trace)
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        sx = 1.35 * np.outer(np.cos(u), np.sin(v))
        sy = 1.35 * np.outer(np.sin(u), np.sin(v))
        sz = 1.35 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(sx, sy, sz, color='cyan', alpha=0.04, edgecolor='none')
        
        # Set isometric viewpoint
        ax.set_box_aspect([1,1,1])
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_zlim([-1.3, 1.3])
        ax.view_init(elev=25, azim=55)
        
        plt.title("The Borromean Proton ($6^3_2$)\nTopological Tensor Halo & Quark Confinement", 
                  color='white', pad=20, fontsize=16, fontweight='bold')
        
        output_path = '../assets/sim_outputs/borromean_proton_3d.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Successfully saved 3D Borromean Proton visualization to {output_path}")

    except Exception as e:
        print(f"Simulation failed with error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
