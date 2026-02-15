import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import qmc
import os

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_rifled_photon():
    print("Simulating Rifled Photon on Strict Lattice...")
    L = 20.0
    engine = qmc.PoissonDisk(d=3, radius=0.4/L, seed=42)
    points = engine.fill_space() * L
    
    k, m, z_center, sigma = 3.0, 1.0, 10.0, 3.0
    active_x, active_y, active_z, active_c = [], [], [], []
    
    for pos in points:
        dz = pos[2] - z_center
        r = np.sqrt((pos[0]-L/2)**2 + (pos[1]-L/2)**2)
        envelope = np.exp(-(dz**2)/(2*sigma**2)) * np.exp(-(r**2)/(2*(sigma/2)**2))
        if envelope < 0.05: continue
        
        theta = np.arctan2(pos[1]-L/2, pos[0]-L/2)
        phase = k * dz + m * theta
        active_x.append(pos[0]); active_y.append(pos[1]); active_z.append(pos[2]); active_c.append(envelope * np.cos(phase))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(active_x, active_y, active_z, c=active_c, cmap='coolwarm', s=25, alpha=0.9)
    ax.plot([L/2, L/2], [L/2, L/2], [0, L], 'w--', linewidth=1.5, alpha=0.6)
    
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_title("Photon Propagation (Rifling m=1)\nBlue/Red = +/- Lattice Phase Twist", color='white', fontsize=14)
    ax.text2D(0.05, 0.05, "The Spiral Phase geometrically averages the random nodes\ninto a deterministic straight trajectory.", 
              transform=ax.transAxes, color='cyan', fontsize=12)

    plt.savefig(os.path.join(OUTPUT_DIR, "photon_rifling.png"), dpi=300, facecolor='black', bbox_inches='tight')

if __name__ == "__main__": simulate_rifled_photon()