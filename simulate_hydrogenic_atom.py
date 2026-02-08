import numpy as np
import matplotlib.pyplot as plt

def simulate_hydrogenic_atom():
    print("Initializing Hydrogenic Simulation...")
    
    # 1. Setup the Vacuum Domain
    N = 400
    L = 40.0 # Angstroms (scaled)
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    
    # 2. The Proton (Central Potential)
    # Modeled as a refractive index gradient (Gravity/Coulomb Well)
    # "The Proton pulls the vacuum density"
    R = np.sqrt(X**2 + Y**2)
    R[R==0] = 0.01 # Avoid singularity
    Pot = -10.0 / R # Coulomb-like potential
    
    # 3. The Electron (Walker)
    # Starts at random position with random velocity
    px, py = 12.0, 0.0 
    vx, vy = 0.0, 0.8 
    
    # 4. The Pilot Wave Field (Memory)
    wave_field = np.zeros((N, N))
    
    # 5. Simulation Loop
    dt = 0.1
    steps = 4000
    trajectory_x = []
    trajectory_y = []
    
    print(f"Electron orbiting for {steps} steps...")
    
    for t in range(steps):
        # A. Wave Equation (The Vacuum Responds)
        # The wave propagates and decays
        lap = (np.roll(wave_field, 1, axis=0) + np.roll(wave_field, -1, axis=0) + 
               np.roll(wave_field, 1, axis=1) + np.roll(wave_field, -1, axis=1) - 4*wave_field)
        
        # Damping (Energy Loss) + Restoration
        wave_field = 0.9*wave_field + 0.1*lap
        
        # B. Electron Impact (Source)
        # The electron hits the vacuum, creating a ripple
        ix, iy = int((px + L/2)/L * N), int((py + L/2)/L * N)
        if 0 <= ix < N and 0 <= iy < N:
            wave_field[iy, ix] += 1.0 * np.sin(0.5 * t)
            
        # C. Gradient Force (Pilot Wave Guidance)
        # Calculate local wave slope
        if 1 < ix < N-1 and 1 < iy < N-1:
            grad_x = (wave_field[iy, ix+1] - wave_field[iy, ix-1])
            grad_y = (wave_field[iy+1, ix] - wave_field[iy-1, ix])
        else:
            grad_x, grad_y = 0, 0
            
        # D. Proton Force (Coulomb Attraction)
        # F = qE (Standard Physics)
        dist = np.sqrt(px**2 + py**2)
        fx_coulomb = -15.0 * px / (dist**3 + 0.1)
        fy_coulomb = -15.0 * py / (dist**3 + 0.1)
        
        # E. Update Position (Newton's Laws + Pilot Wave)
        # "The electron surfs the wave AND falls toward the proton"
        vx += dt * (fx_coulomb - 0.5*grad_x - 0.05*vx) # Drag term simulates radiation
        vy += dt * (fy_coulomb - 0.5*grad_y - 0.05*vy)
        
        px += vx * dt
        py += vy * dt
        
        trajectory_x.append(px)
        trajectory_y.append(py)

    # 6. Visualization
    plt.figure(figsize=(10, 8))
    
    # Plot the Wave Field (The "Probabilistic" Cloud)
    plt.imshow(wave_field, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='Blues', alpha=0.6)
    
    # Plot the Trajectory (The "Deterministic" Path)
    plt.plot(trajectory_x, trajectory_y, 'r-', linewidth=0.5, alpha=0.8, label="Electron Path")
    plt.plot(trajectory_x[-1], trajectory_y[-1], 'ro', label="Current Position")
    plt.plot(0, 0, 'k+', markersize=12, markeredgewidth=2, label="Proton")
    
    # Draw Bohr Radii for comparison (n=1, n=2)
    # These are the "Magic Circles" where the wave interferes constructively
    circle1 = plt.Circle((0, 0), 4.0, color='g', fill=False, linestyle='--', label='n=1 Stable Orbit')
    circle2 = plt.Circle((0, 0), 16.0, color='m', fill=False, linestyle='--', label='n=2 Stable Orbit')
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)

    plt.title("The Emergent Atom: Electron Quantization via Fluid Memory")
    plt.xlabel("Angstroms"); plt.ylabel("Angstroms")
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    simulate_hydrogenic_atom()