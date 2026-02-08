import numpy as np
import matplotlib.pyplot as plt

def run_genesis_sim():
    print("LCT Simulation B: Kibble-Zurek Genesis")
    N = 100
    # Random Phase Field (Hot Universe)
    phase = np.random.uniform(0, 2*np.pi, (N, N))
    
    # Cooling / Relaxation Step (Cellular Automaton approximation)
    for _ in range(50):
        # Average neighbors to simulate energy minimization
        phase_new = (np.roll(phase, 1, 0) + np.roll(phase, -1, 0) + 
                     np.roll(phase, 1, 1) + np.roll(phase, -1, 1)) / 4.0
        phase = phase_new

    # Detect Vortices (Topological Defects)
    # Curl calculation (simplified)
    curl = np.roll(phase, 1, 0) - np.roll(phase, -1, 0)
    
    plt.figure(figsize=(8,6))
    plt.imshow(np.sin(phase), cmap='twilight')
    plt.title("Topological Defects (Matter) in Cooling Lattice")
    plt.colorbar(label='Vacuum Phase')
    plt.show()

if __name__ == "__main__":
    run_genesis_sim()