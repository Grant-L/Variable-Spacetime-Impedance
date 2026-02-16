import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/04_baryon_sector/simulations/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_neutron_threaded():
    print("Simulating Neutron Threaded Topology (6^3_2 U 3_1)...")
    
    t = np.linspace(0, 2*np.pi, 1000)
    R, r, d = 1.0, 0.4, 0.3
    x1, y1, z1 = R * np.cos(t), r * np.sin(t), d * np.cos(2*t)
    x2, y2, z2 = d * np.cos(2*t), R * np.cos(t), r * np.sin(t)
    x3, y3, z3 = r * np.sin(t), d * np.cos(2*t), R * np.cos(t)
    
    # Golden Torus (Electron), scaled to fit in the core
    Phi = (1 + np.sqrt(5)) / 2
    scale = 0.35
    R_t, r_t = (Phi / 2) * scale, ((Phi - 1) / 2) * scale
    p, q = 3, 2
    
    x_t = (R_t + r_t * np.cos(q * t)) * np.cos(p * t)
    y_t = (R_t + r_t * np.cos(q * t)) * np.sin(p * t)
    z_t = r_t * np.sin(q * t)
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    ax.plot(x1, y1, z1, color='#00FFFF', linewidth=2, alpha=0.3)
    ax.plot(x2, y2, z2, color='#FF00FF', linewidth=2, alpha=0.3)
    ax.plot(x3, y3, z3, color='#FFFF00', linewidth=2, alpha=0.3)
    
    strain = np.abs(np.gradient(z_t))
    ax.scatter(x_t, y_t, z_t, c=strain, cmap='magma', s=20, alpha=1.0, edgecolors='none')
    ax.plot(x_t, y_t, z_t, color='white', linewidth=1.5, alpha=0.8, label='Trapped $3_1$ Electron')
    
    ax.axis('off')
    ax.text2D(0.05, 0.90, r"AVE: Neutron Soliton ($6^3_2 \cup 3_1$) Metastability", transform=ax.transAxes, color='#ff3333', fontsize=16, weight='bold')
    
    textstr = (
        r"Topology: Trefoil electron trapped within Proton core void." + "\n" +
        r"Beta Decay: Electron tunnels out of topological link ($\cup$), snapping" + "\n" +
        r"the localized lattice tension (released as antineutrino $\bar{\nu}_e$)."
    )
    ax.text2D(0.05, 0.75, textstr, transform=ax.transAxes, color='white', fontsize=12, bbox=dict(facecolor='black', edgecolor='#ff3333', alpha=0.7, pad=8))

    ax.view_init(elev=20, azim=60)
    filepath = os.path.join(OUTPUT_DIR, "neutron_threaded.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    simulate_neutron_threaded()