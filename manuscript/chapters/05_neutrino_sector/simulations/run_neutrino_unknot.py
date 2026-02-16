import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/05_neutrino_sector/simulations/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_neutrino_unknot():
    print("Simulating Neutrino Torsional Unknot (0_1)...")
    
    R = 1.0  # Major Radius
    r = 0.2  # Minor Radius (No crowding)
    
    u = np.linspace(0, 2 * np.pi, 500)
    v = np.linspace(0, 2 * np.pi, 100)
    U, V = np.meshgrid(u, v)
    
    # Mobius twist
    twists = 1
    X = (R + r * np.cos(V + twists * U / 2)) * np.cos(U)
    Y = (R + r * np.cos(V + twists * U / 2)) * np.sin(U)
    Z = r * np.sin(V + twists * U / 2)
    
    # Internal phase twist (Torsion without crossing)
    phase = np.sin(3 * U + V)
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.cool(phase/2 + 0.5), 
                    alpha=0.8, rstride=1, cstride=1, linewidth=0, antialiased=False)
    
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')
    
    ax.text2D(0.05, 0.90, "AVE: Neutrino Soliton ($0_1$ Twisted Unknot)", transform=ax.transAxes, color='#00ffff', fontsize=16, weight='bold')
    
    textstr = (
        r"Topology: No crossings ($C=0$). Skyrme term evaluates to zero." + "\n" +
        r"Dielectric Saturation: None. Denominator remains $1.0$." + "\n" +
        r"Mass: Pure linear torsional strain (Escapes the exponential mass spike)."
    )
    ax.text2D(0.05, 0.75, textstr, transform=ax.transAxes, color='white', fontsize=12, bbox=dict(facecolor='black', edgecolor='#00ffff', alpha=0.7, pad=8))

    ax.view_init(elev=30, azim=45)
    filepath = os.path.join(OUTPUT_DIR, "neutrino_unknot.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    simulate_neutrino_unknot()