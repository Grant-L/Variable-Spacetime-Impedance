import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import os

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_rifled_photon():
    print("Simulating Rifled Photon (Helicity m=1)...")
    
    # Setup Lattice
    np.random.seed(42) 
    n_nodes = 8000
    L = 20.0
    points = np.random.rand(n_nodes, 3) * L
    
    # Signal Parameters
    k = 3.0   # Momentum (Wave Number)
    m = 1.0   # Helicity (Spin)
    z_center = 10.0
    sigma = 3.0
    
    # We visualize the STRESS on the nodes directly
    active_x, active_y, active_z, active_c = [], [], [], []
    
    for i in range(n_nodes):
        pos = points[i]
        
        # Spatial Filter (The Packet Envelope)
        dz = pos[2] - z_center
        r = np.sqrt((pos[0]-L/2)**2 + (pos[1]-L/2)**2)
        envelope = np.exp(- (dz**2) / (2*sigma**2)) * np.exp(- (r**2) / (2*(sigma/2)**2))
        
        if envelope < 0.05: continue
        
        # RIFLING PHYSICS (Axiom II)
        # Phase = k*z + m*theta
        theta = np.arctan2(pos[1]-L/2, pos[0]-L/2)
        phase = k * dz + m * theta
        
        # Real Amplitude (The twist stress)
        amplitude = envelope * np.cos(phase)
        
        active_x.append(pos[0])
        active_y.append(pos[1])
        active_z.append(pos[2])
        active_c.append(amplitude)

    # Visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the Photon
    p = ax.scatter(active_x, active_y, active_z, c=active_c, cmap='coolwarm', s=15, alpha=0.8)
    
    # Add "Flight Path" line
    ax.plot([L/2, L/2], [L/2, L/2], [0, L], 'w--', linewidth=1, alpha=0.5)
    
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_title("Photon Propagation (Rifling m=1)\nBlue/Red = +/- Lattice Twist", color='white')
    
    # Annotation
    ax.text2D(0.05, 0.05, "The Spiral Phase averages the random nodes\ninto a straight trajectory.", 
              transform=ax.transAxes, color='white')

    output_path = os.path.join(OUTPUT_DIR, "photon_rifling.png")
    plt.savefig(output_path, dpi=300, facecolor='black')
    print(f"Rifling simulation saved to {output_path}")

if __name__ == "__main__":
    simulate_rifled_photon()