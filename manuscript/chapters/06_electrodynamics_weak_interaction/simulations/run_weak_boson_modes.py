import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def simulate_weak_modes():
    print("Simulating W and Z Boson Cosserat Modes...")
    z = np.linspace(0, 10, 100)
    theta_w = np.sin(z)
    
    # Mode 1: W Boson (Longitudinal Torsion / Twist)
    x_W = 0.5 * np.cos(theta_w)
    y_W = 0.5 * np.sin(theta_w)
    
    # Mode 2: Z Boson (Transverse Bending / Flexure)
    x_Z = 0.5 * np.sin(z)
    y_Z = np.zeros_like(z)
    
    fig = plt.figure(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor('#050508')
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('#050508')
    ax1.plot(np.zeros_like(z), np.zeros_like(z), z, color='gray', lw=1, linestyle='--')
    for i in range(0, len(z), 2):
        u, v = np.cos(theta_w[i]) * 0.5, np.sin(theta_w[i]) * 0.5
        ax1.quiver(0, 0, z[i], u, v, 0, color='#00ffcc', linewidth=1.5, arrow_length_ratio=0.3)
    ax1.set_title(r'$W^\pm$ Boson: Pure Torsional Mode (Shear Modulus $G$)', color='white', fontsize=12)
    ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1); ax1.set_zlim(0, 10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('#050508')
    ax2.plot(np.zeros_like(z), np.zeros_like(z), z, color='gray', lw=1, linestyle='--')
    ax2.plot(x_Z, y_Z, z, color='white', lw=1, alpha=0.5)
    for i in range(0, len(z), 2):
        u, w = 0.5, np.sin(z[i]) * 0.5
        ax2.quiver(x_Z[i], 0, z[i], u, 0, w, color='#ff3366', linewidth=1.5, arrow_length_ratio=0.3)
    ax2.set_title(r'$Z^0$ Boson: Bending Mode (Young\'s Modulus $E$)', color='white', fontsize=12)
    ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_zlim(0, 10)
    ax2.axis('off')
    
    plt.suptitle("Weak Force Gauge Bosons as Cosserat Acoustic Cutoff Modes", color='white', fontsize=16, weight='bold', y=0.95)
    textstr = r"$\frac{m_W}{m_Z} = \sqrt{\frac{k_{torsion}}{k_{bending}}} = \frac{1}{\sqrt{1+\nu}} \Rightarrow \nu_{vac} \approx 0.287$"
    fig.text(0.5, 0.05, textstr, ha='center', color='white', fontsize=14, bbox=dict(facecolor='black', edgecolor='#00ffcc', pad=10))

    filepath = os.path.join(OUTPUT_DIR, "weak_boson_modes.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    ensure_output_dir()
    simulate_weak_modes()