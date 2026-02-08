import numpy as np
import matplotlib.pyplot as plt

def simulate_casimir_effect():
    print("Initializing Vacuum Substrate...")
    
    # 1. Setup 1D Lattice
    Nx = 400
    u = np.zeros(Nx)       # Voltage/Displacement
    u_prev = np.zeros(Nx)
    
    # 2. Define Plates (Conducting Boundaries -> V=0)
    # Plate 1 at 100, Plate 2 at 140 (Gap = 40 nodes)
    p1_idx = 100
    p2_idx = 140
    
    # 3. Simulation Parameters
    c = 1.0
    dt = 0.5 # CFL stability < 1.0
    steps = 4000
    
    # Accumulators for RMS measurement
    energy_sum = np.zeros(Nx)
    
    print(f"Simulating Quantum Vacuum Noise for {steps} steps...")
    
    for t in range(steps):
        # A. Wave Equation (1D FDTD)
        # u_next = 2u - u_prev + c^2 * dt^2 * Laplacian
        lap = np.roll(u, 1) + np.roll(u, -1) - 2*u
        
        # Add Damping (to stabilize the noise equilibrium)
        damping = 0.99 
        u_next = (2.0*u - u_prev + (c*dt)**2 * lap) * damping
        
        # B. Inject Quantum Foam (ZPE)
        # Add small random noise to EVERY node
        noise = np.random.normal(0, 0.05, Nx)
        u_next += noise
        
        # C. Apply Boundary Conditions (The Plates)
        # Conducting plates short the vacuum to ground (V=0)
        u_next[p1_idx] = 0.0
        u_next[p2_idx] = 0.0
        
        # Cycle buffers
        u_prev = u.copy()
        u = u_next.copy()
        
        # D. Accumulate Energy (after settling time)
        if t > 500:
            energy_sum += u**2

    # 4. Calculate Average Energy Density
    avg_energy = energy_sum / (steps - 500)
    
    # 5. Visualization
    plt.figure(figsize=(10, 6))
    
    # Plot Energy Density
    plt.plot(avg_energy, 'b-', linewidth=1.5, label="Vacuum Energy Density ($V_{rms}^2$)")
    
    # Draw Plates
    plt.axvline(x=p1_idx, color='k', linewidth=3, linestyle='-', label="Conducting Plate")
    plt.axvline(x=p2_idx, color='k', linewidth=3, linestyle='-')
    
    # Highlight the "Casimir Gap"
    plt.axvspan(p1_idx, p2_idx, color='yellow', alpha=0.2, label="Excluded Modes (Casimir Gap)")
    
    # Theoretical Baseline (External Vacuum)
    baseline = np.mean(avg_energy[:50])
    plt.axhline(y=baseline, color='r', linestyle='--', alpha=0.5, label="Free Space ZPE")
    
    plt.title("Simulation D.10: The Casimir Effect (Vacuum Filtration)")
    plt.xlabel("Lattice Position")
    plt.ylabel("Vacuum Energy Density")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Annotation
    gap_energy = np.mean(avg_energy[p1_idx:p2_idx])
    percent_drop = 100 * (1 - gap_energy/baseline)
    plt.text((p1_idx+p2_idx)/2, gap_energy/2, f"-{percent_drop:.1f}% Energy", 
             ha='center', fontweight='bold', color='darkblue')
    
    plt.show()

if __name__ == "__main__":
    simulate_casimir_effect()