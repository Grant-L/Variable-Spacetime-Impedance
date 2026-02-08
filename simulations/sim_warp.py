import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

def simulate_animated_gravity_well():
    print("Animating Gravitational Collapse...")
    
    # 1. Setup Grid (Smaller for animation speed)
    grid_size = 300
    N = 1000 
    x = np.linspace(-grid_size, grid_size, N)
    y = np.linspace(-grid_size, grid_size, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # 2. Setup Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Remove panes for "Void" look
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.axis('off')
    
    def update(frame):
        ax.clear()
        ax.axis('off')
        
        # Dynamics: Mass (Rs) grows from 0.0 to 3.0
        Rs = 3.0 * (frame / 40.0)
        
        # Calculate Impedance (with safety clamp)
        # Avoid Singularity: R must be > Rs
        R_safe = R.copy()
        R_safe[R_safe < Rs + 0.2] = Rs + 0.2
        
        # Refractive Index n = 1 / sqrt(1 - Rs/r)
        # We handle the 'imaginary' region inside horizon by clamping
        val = 1.0 - Rs / R_safe
        val[val < 0.001] = 0.001
        n = 1.0 / np.sqrt(val)
        
        # Visual Depth
        Z_visual = -5.0 * (n - 1.0)
        Z_visual[Z_visual < -15] = -15 # Clamp bottom for visual clarity
        
        # Plot Surface
        # Color mapped to depth
        norm = (Z_visual - Z_visual.max()) / (Z_visual.min() - Z_visual.max() + 1e-6)
        ax.plot_surface(X, Y, Z_visual, 
                        facecolors=plt.cm.inferno(norm), 
                        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=True)
        
        # Draw Event Horizon Ring
        if Rs > 0.1:
            theta = np.linspace(0, 2*np.pi, 60)
            xh = Rs * np.cos(theta)
            yh = Rs * np.sin(theta)
            zh = np.ones_like(theta) * np.min(Z_visual)
            ax.plot(xh, yh, zh, color='cyan', linewidth=2)
        
        # Camera Rotation
        ax.view_init(elev=70, azim=-60 + frame)
        ax.set_title(f"Vacuum Stress: $R_s = {Rs:.2f}$", color='white', fontsize=15)
        
    ani = FuncAnimation(fig, update, frames=600, interval=50)
    ani.save("gravity_well_collapse.gif", writer=PillowWriter(fps=15))
    print("Animation saved.")

if __name__ == "__main__":
    simulate_animated_gravity_well()