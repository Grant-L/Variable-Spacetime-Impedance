import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/04_baryon_sector/simulations/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_borromean_proton():
    print("Simulating Proton Borromean Linkage (6^3_2)...")
    
    t = np.linspace(0, 2*np.pi, 1000)
    R, r, d = 1.0, 0.4, 0.3
    
    # Z_3 Symmetric permutation
    x1, y1, z1 = R * np.cos(t), r * np.sin(t), d * np.cos(2*t)
    x2, y2, z2 = d * np.cos(2*t), R * np.cos(t), r * np.sin(t)
    x3, y3, z3 = r * np.sin(t), d * np.cos(2*t), R * np.cos(t)
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    # SU(3) Colors
    ax.plot(x1, y1, z1, color='#00FFFF', linewidth=6, alpha=0.9, label='Quark Loop 1 (Cyan)')
    ax.plot(x2, y2, z2, color='#FF00FF', linewidth=6, alpha=0.9, label='Quark Loop 2 (Magenta)')
    ax.plot(x3, y3, z3, color='#FFFF00', linewidth=6, alpha=0.9, label='Quark Loop 3 (Yellow)')
    
    ax.axis('off')
    ax.text2D(0.05, 0.90, r"AVE: Proton Soliton ($6^3_2$ Borromean Linkage)", transform=ax.transAxes, color='white', fontsize=16, weight='bold')
    
    textstr = (
        r"Strict $\mathbb{Z}_3$ Permutation Symmetry $\rightarrow$ Color Charge ($SU(3)$)" + "\n" +
        r"Topological Confinement: Cutting one loop unlinks all." + "\n" +
        r"Fractionalization: Witten Effect on $\theta$-vacuum yields $\pm 1/3 e, \pm 2/3 e$."
    )
    ax.text2D(0.05, 0.75, textstr, transform=ax.transAxes, color='white', fontsize=12, bbox=dict(facecolor='black', edgecolor='white', alpha=0.7, pad=8))

    filepath = os.path.join(OUTPUT_DIR, "proton_borromean.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    simulate_borromean_proton()