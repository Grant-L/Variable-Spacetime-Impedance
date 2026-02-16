import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def simulate_trefoil_golden_torus():
    print("Simulating Trefoil Geometric Q-Factor (Golden Torus Limit)...")
    
    # Exact Hardware Saturation Limits (Golden Ratio)
    Phi = (1 + np.sqrt(5)) / 2
    R = Phi / 2  # Major Radius (~0.809)
    r = (Phi - 1) / 2  # Minor Radius (~0.309)
    
    p = 3
    q = 2
    t = np.linspace(0, 2 * np.pi, 2000)
    
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)
    strain = np.sqrt(dx**2 + dy**2 + dz**2)
    
    ax.scatter(x, y, z, c=strain, cmap='magma', s=80, alpha=0.9, edgecolors='none')
    ax.plot(x, y, z, color='white', linewidth=1, alpha=0.3)
    
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')
    
    ax.text2D(0.05, 0.90, "AVE: Electron Soliton ($3_1$) at Dielectric Ropelength", transform=ax.transAxes, color='#00ffcc', fontsize=16, weight='bold')
    ax.text2D(0.05, 0.85, r"Geometric Q-Factor ($\alpha^{-1}_{ideal}$) $\approx 137.036$", transform=ax.transAxes, color='white', fontsize=14)
    
    textstr = (
        r"Golden Torus Limits: $R = \Phi/2$, $r = (\Phi-1)/2$, $d = 1$" + "\n" +
        r"$\Lambda_{vol} = 4\pi^3 \approx 124.025$ (Volumetric Bulk Inductance)" + "\n" +
        r"$\Lambda_{surf} = \pi^2 \approx 9.870$ (Cross-Sectional Screening)" + "\n" +
        r"$\Lambda_{line} = \pi \approx 3.142$ (Linear Flux Moment)"
    )
    ax.text2D(0.05, 0.65, textstr, transform=ax.transAxes, color='white', fontsize=12, 
              bbox=dict(facecolor='black', edgecolor='#00ffcc', alpha=0.7, pad=8))

    filepath = os.path.join(OUTPUT_DIR, "trefoil_alpha_qfactor.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Simulation Complete. Saved: {filepath}")
    plt.close()

if __name__ == "__main__":
    ensure_output_dir()
    simulate_trefoil_golden_torus()