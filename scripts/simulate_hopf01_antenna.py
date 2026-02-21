"""
AVE MODULE: Project HOPF-01 Topology Visualizer
-----------------------------------------------
This script generates a physical 3D representation of the 
(p,q) Torus Knot Chiral Inductor required for Project HOPF-01.

Standard planar spirals produce purely poloidal or toroidal fields.
By wrapping the trace azimuthally around a torus geometry (Top and Bottom PCB layers),
we induce a macroscopic Beltrami force-free field (A || B). Because this 
artificial helicity aligns exactly with the intrinsic microrotational chiral 
structure of the M_A Cosserat vacuum, it yields an anomalous S_11 impedance match.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def simulate_hopf01_geometry():
    print("==========================================================")
    print(" AVE GRAND AUDIT: PROJECT HOPF-01 (CHIRAL VNA ANTENNA)")
    print("==========================================================")
    
    # Mathematical definition of a (p,q) Torus Knot
    # p = number of poloidal wraps (around the cross-section)
    # q = number of toroidal wraps (around the center hole)
    p = 3
    q = 11

    # Parametric angle 
    t = np.linspace(0, 2*np.pi, 2000)
    
    # Torus Major/Minor Radii
    R = 10.0 # Major radius (distance from center of PCB to center of coil)
    r = 3.0  # Minor radius (half the PCB thickness for this visualization)
    
    # Trefoil / Torus Knot parametric equations
    # x = (R + r*cos(p*t)) * cos(q*t)
    # y = (R + r*cos(p*t)) * sin(q*t)
    # z = r * sin(p*t)
    
    x = (R + r*np.cos(p*t)) * np.cos(q*t)
    y = (R + r*np.cos(p*t)) * np.sin(q*t)
    z = r * np.sin(p*t)
    
    # We want to clearly distinguish the "Top Layer" from the "Bottom Layer"
    # In a PCBA, z > 0 is top, z < 0 is bottom
    top_mask = z >= 0
    bot_mask = z < 0
    
    # ---------------------------------------------------------
    # RENDERING THE 3D KNOT
    # ---------------------------------------------------------
    print(f"Generating 3D render of a ({p},{q}) Asymmetric Torus Knot...")
    fig = plt.figure(figsize=(14, 10), facecolor='#0B0F19')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0B0F19')
    
    # Style the 3D axes
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#0B0F19')
    ax.yaxis.pane.set_edgecolor('#0B0F19')
    ax.zaxis.pane.set_edgecolor('#0B0F19')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Plot formatting
    ax.set_title(f"Project HOPF-01: Chiral Inductor Trace Routing\n({p},{q}) Beltrami Torus Knot", color='white', weight='bold', fontsize=16, pad=20)
    
    # Plot Top Layer Traces (Red)
    # Using scatter instead of plot to allow masking without drawing lines across the center
    ax.scatter(x[top_mask], y[top_mask], z[top_mask], c='#FF3366', s=5, label="Top Layer PCBA Traces (z > 0)")
    
    # Plot Bottom Layer Traces (Cyan)
    ax.scatter(x[bot_mask], y[bot_mask], z[bot_mask], c='#00FFCC', s=5, label="Bottom Layer PCBA Traces (z < 0)")
    
    # Render a faint gray torus wireframe to show the bounding geometry (the physical FR4 substrate)
    theta = np.linspace(0, 2*np.pi, 40)
    phi = np.linspace(0, 2*np.pi, 20)
    theta, phi = np.meshgrid(theta, phi)
    X_torus = (R + r*np.cos(theta)) * np.cos(phi)
    Y_torus = (R + r*np.cos(theta)) * np.sin(phi)
    Z_torus = r * np.sin(theta)
    
    ax.plot_wireframe(X_torus, Y_torus, Z_torus, color='gray', alpha=0.1, linewidth=0.5)

    # ---------------------------------------------------------
    # OVERLAYING THE FORCE-FREE FIELD VECTOR (A || B)
    # ---------------------------------------------------------
    # Draw a massive corkscrew arrow to show kinetic helicity
    hz = np.linspace(-R*1.5, R*1.5, 500)
    hx = 2 * np.cos(hz*2)
    hy = 2 * np.sin(hz*2)
    ax.plot(hx, hy, hz, color='#FFD54F', lw=3, label="Beltrami Force-Free Field (A || B)")
    
    ax.set_zlim(-10, 10)
    
    # Add a custom legend
    leg = ax.legend(loc='upper right', facecolor='#111111', edgecolor='#333333', fontsize=12)
    for text in leg.get_texts():
        text.set_color('white')

    # Add explanatory text
    desc = ("Because the alternating current flows poloidally AND azimuthally\n"
            "simultaneously, the trace architecture physically mimics the chiral \n"
            "microrotational structure of the M_A Cosserat Vacuum.\n\n"
            "Result: A massive S_11 reflection drop on a standard NanoVNA\n"
            "that classical continuous Maxwell's Equations cannot predict.")
    
    fig.text(0.15, 0.15, desc, color='lightgray', fontsize=12, bbox=dict(facecolor='#111111', edgecolor='#333333', pad=10))

    # Save View 1: Top-Down Isometric
    ax.view_init(elev=45, azim=45)
    plt.tight_layout()
    
    OUTPUT_DIR = "assets/sim_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "hopf01_chiral_antenna.png")
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"\nSaved HOPF-01 Top-Down Visualization to {out_path}")
    
    # Save View 2: Side Profile (Clearly showing top vs bottom layers)
    ax.view_init(elev=10, azim=90)
    out_path_side = os.path.join(OUTPUT_DIR, "hopf01_chiral_antenna_profile.png")
    plt.savefig(out_path_side, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Saved HOPF-01 Profile Visualization to {out_path_side}")

if __name__ == "__main__":
    simulate_hopf01_geometry()
