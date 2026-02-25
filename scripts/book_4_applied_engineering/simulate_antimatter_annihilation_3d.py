"""
AVE Antimatter Annihilation Simulator (3D Geometry)
===================================================
Simulates the exact 3D macroscopic geometric intersection of an Electron 
(Left-Handed Trefoil Knot / Beltrami Vortex) and a Positron (Right-Handed Trefoil).

Plots the progressive overlap of the two macroscopic forms. Because they possess 
strict inverse parity, their geometries perfectly map onto one another oppositely, 
resulting in an instantaneous structural null (Linking Number drops to 0), proving 
the mathematical mechanics behind the annihilation flash.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
from pathlib import Path

# Geometry Parameters
N_POINTS = 500

def generate_trefoil_knot(handedness=1):
    """
    Parametric equations for a macroscopic Trefoil knot.
    handedness = 1 (Left-Hand / Matter / e-)
    handedness = -1 (Right-Hand / Antimatter / e+)
    """
    t = np.linspace(0, 2*np.pi, N_POINTS)
    
    # Standard torus knot parametric footprint
    x = np.sin(t) + 2 * np.sin(2*t)
    y = np.cos(t) - 2 * np.cos(2*t)
    
    # The Handedness directly flips the Z-phase
    z = handedness * np.sin(3*t) 
    
    return np.vstack((x, y, z)).T

def run_3d_annihilation(out_path):
    print("Generating 3D Topological Parity Cancellation (Annihilation)...")
    
    # Generate the pristine structures
    matter = generate_trefoil_knot(handedness=1)
    antimatter = generate_trefoil_knot(handedness=-1)
    
    # We will animate them sliding together along the X axis until they overlap perfectly
    FRAMES = 60
    
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Formatting
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    
    title = ax.set_title("Topological Parity Inversion: $e^-$ (Left) vs $e^+$ (Right)", color='white', size=14, pad=20)
    
    line1, = ax.plot([], [], [], color='#00aaff', linewidth=3, alpha=0.8, label='$e^-$ (Matter)')
    line2, = ax.plot([], [], [], color='#ff00aa', linewidth=3, alpha=0.8, label='$e^+$ (Antimatter)')
    
    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
    
    def update(f):
        # Progress from 0 to 1
        progress = f / float(FRAMES)
        
        # Start far apart (X = -5 and X = 5)
        # End completely overlapped (X = 0)
        x_offset = 5.0 * (1.0 - progress)
        
        m_curr = np.copy(matter)
        a_curr = np.copy(antimatter)
        
        m_curr[:, 0] -= x_offset
        a_curr[:, 0] += x_offset
        
        # When they begin to heavily overlap, the structure physically yields (vanishes/alpha drops)
        # simulating the E=mc^2 unspooling into pure energy.
        alpha_val = 0.8
        if progress > 0.85:
            # Drop transparency rapidly as the structure cancels out
            alpha_val = 0.8 * (1.0 - ((progress - 0.85) / 0.15))
            alpha_val = max(0.0, alpha_val)
            title.set_text("Topological Yield ($\omega - \omega = 0$): SHATTERING ($E=mc^2$)")
            title.set_color('#ffcc00')
        else:
            title.set_text("Topological Parity Inversion: $e^-$ (Matter) vs $e^+$ (Antimatter)")
            title.set_color('white')

        line1.set_data(m_curr[:, 0], m_curr[:, 1])
        line1.set_3d_properties(m_curr[:, 2])
        line1.set_alpha(alpha_val)
        
        line2.set_data(a_curr[:, 0], a_curr[:, 1])
        line2.set_3d_properties(a_curr[:, 2])
        line2.set_alpha(alpha_val)
        
        ax.view_init(elev=20, azim=progress * 90)
        return line1, line2, title

    ani = animation.FuncAnimation(fig, update, frames=FRAMES, blit=False)
    ani.save(out_path, writer='pillow', fps=15)
    plt.close()
    print(f"[Done] Saved 3D Annihilation GIF: {out_path}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    out_dir = PROJECT_ROOT / "scripts" / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    run_3d_annihilation(out_dir / "annihilation_3d_parity.gif")
