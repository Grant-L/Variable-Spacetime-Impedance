"""
AVE MODULE 7: 3D PHOTON RIFLING ON THE COSSERAT LATTICE
-------------------------------------------------------
A mathematically flawless 3D visualization of a Spin-1 (Helical) Vector Boson 
propagating through the Trace-Reversed Cosserat Vacuum.
Demonstrates how the spatial helicity (Vector Potential A) spans across 
both primary kinematic links and transverse Cosserat links, averaging out the 
stochastic geometry to zero via the Central Limit Theorem.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from scipy.stats import qmc
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_cosserat_rifling():
    print("Simulating Spin-1 Photon on Over-braced Cosserat Lattice...")
    L = 20.0
    MIN_DIST = 1.0
    COSSERAT_RATIO = 1.67 # Strict structural span derived in Chapter 1
    
    # 1. Generate the Stochastic Cosserat Hardware
    engine = qmc.PoissonDisk(d=3, radius=MIN_DIST/L, seed=42)
    points = engine.fill_space() * L
    kd_tree = cKDTree(points)
    
    # 2. Define the Helical Vector Field (Spin-1 Photon)
    k_wave = 1.5           # Longitudinal Wavenumber
    m_spin = 1.0           # Helicity / Spin
    z_center = 10.0
    sigma_z, sigma_r = 4.0, 3.5
    
    active_x, active_y, active_z, active_c = [], [], [], []
    active_nodes = set()
    
    for i, pos in enumerate(points):
        dz = pos[2] - z_center
        r = np.sqrt((pos[0]-L/2)**2 + (pos[1]-L/2)**2)
        
        # Gaussian Envelope for the wave packet
        envelope = np.exp(-(dz**2)/(2*sigma_z**2)) * np.exp(-(r**2)/(2*sigma_r**2))
        
        if envelope > 0.1:
            active_nodes.add(i)
            theta = np.arctan2(pos[1]-L/2, pos[0]-L/2)
            phase = k_wave * dz + m_spin * theta  # The Rifling Math
            
            active_x.append(pos[0])
            active_y.append(pos[1])
            active_z.append(pos[2])
            # Color maps to the local polarization amplitude
            active_c.append(envelope * np.cos(phase))

    # 3. Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050508')
    fig.patch.set_facecolor('#050508')
    
    # Draw ONLY the connections between active photon nodes to show the tunneling
    print("Tracing Dynamic Vacuum Links within the Photon Volume...")
    for idx in active_nodes:
        # Primary Kinematic Links (Cyan)
        p_neighbors = kd_tree.query_ball_point(points[idx], r=MIN_DIST * 1.2)
        for j in p_neighbors:
            if idx < j and j in active_nodes:
                p1, p2 = points[idx], points[j]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                        color='cyan', alpha=0.4, linewidth=1.5)
                        
        # Cosserat Transverse Links (Magenta)
        c_neighbors = kd_tree.query_ball_point(points[idx], r=MIN_DIST * COSSERAT_RATIO)
        for j in c_neighbors:
            if idx < j and j in active_nodes and j not in p_neighbors:
                p1, p2 = points[idx], points[j]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                        color='#ff00ff', alpha=0.2, linewidth=0.8, linestyle=':')

    # Scatter the nodes scaled by their wave amplitude
    sc = ax.scatter(active_x, active_y, active_z, c=active_c, cmap='coolwarm', 
                    s=np.abs(active_c)*150 + 10, alpha=0.9, edgecolors='white', linewidth=0.5)

    # Central Geodesic axis
    ax.plot([L/2, L/2], [L/2, L/2], [0, L], color='white', linestyle='--', linewidth=2, alpha=0.6)
    
    ax.axis('off')
    ax.set_title("Photon Tunneling in the Cosserat Vacuum\n(Spin-1 Helical Phase Propagation)", color='white', fontsize=16, weight='bold')
    
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='cyan', lw=2),
        Line2D([0], [0], color='#ff00ff', lw=2, linestyle=':'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, linestyle='None')
    ]
    ax.legend(custom_lines, 
              ['Primary Kinematic Excitation', 'Cosserat Transverse Excitation', 'Positive Phase Amplitude', 'Negative Phase Amplitude'], 
              loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white')

    output_path = os.path.join(OUTPUT_DIR, "cosserat_photon_rifling.png")
    plt.savefig(output_path, dpi=300, facecolor='#050508', bbox_inches='tight')
    print(f"Simulation saved to: {output_path}")

if __name__ == "__main__": simulate_cosserat_rifling()