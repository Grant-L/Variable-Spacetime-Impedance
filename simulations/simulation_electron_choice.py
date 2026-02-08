import numpy as np
import matplotlib.pyplot as plt

def simulate_observer_effect():
    # 1. Setup Vacuum Domain
    Nx, Ny = 300, 200
    # We need CURRENT and PREVIOUS wave states for a true wave equation
    u = np.zeros((Ny, Nx))       # Current
    u_prev = np.zeros((Ny, Nx))  # Previous
    
    wall_x = 100
    
    # 2. Define Slits
    slit_w = 8; slit_sep = 15; cy = Ny // 2
    s1_top = cy + slit_sep + slit_w; s1_bot = cy + slit_sep  # Top Slit
    s2_top = cy - slit_sep; s2_bot = cy - slit_sep - slit_w  # Bottom Slit
    
    # 3. The Observer (Switch)
    # CHANGE THIS to True to see the "Collapse"
    OBSERVER_ON = False
    
    # Create Damping Map (1.0 = Vacuum, <1.0 = Viscosity/Absorber)
    damping = np.ones((Ny, Nx))
    
    if OBSERVER_ON:
        print(">> OBSERVER ACTIVE: Absorbing Slit 2 Wave...")
        # Create a "Soft Absorber" gradient behind Slit 2
        for x in range(wall_x, Nx):
            for y in range(0, cy): # Bottom Half only
                # Smoothly ramp up viscosity to avoid reflection artifacts
                dist = (x - wall_x) / 50.0
                factor = max(0.85, 1.0 - 0.05 * dist) 
                damping[y, x] = factor
    else:
        print(">> OBSERVER OFF: Full Quantum Interference.")

    # 4. The Electron (Walker)
    # Start near Top Slit (Slit 1)
    px, py = 50.0, s1_bot + 4.0 
    vx, vy = 1.5, 0.0
    
    # STABILITY FIX: c*dt/dx must be <= 0.707
    # We use c=1, dx=1, so we choose dt=0.5
    dt = 0.5 
    c2_dt2 = (1.0 * dt)**2 
    
    steps = 800
    traj_x, traj_y = [], [] # FIX: Initialized correctly as two lists

    print("Firing Electron...")
    
    for t in range(steps):
        # A. True Wave Equation (u_tt = c^2 * Laplacian)
        # Lap = (Up + Down + Left + Right - 4*Center)
        lap = (np.roll(u, 1, 0) + np.roll(u, -1, 0) + 
               np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4*u)
        
        # Verlet Integration for Waves:
        # u_next = 2*u - u_prev + (c*dt)^2 * Laplacian
        # We also apply a global drag (0.999) to prevent infinite buildup
        u_next = (2.0*u - u_prev + c2_dt2 * lap) * 0.999
        
        # Apply Observer Damping (Localized Viscosity)
        u_next *= damping
        
        # Wall Boundary Conditions (Reflective)
        mask = np.zeros_like(u)
        mask[:, wall_x:wall_x+5] = 1 # The Wall
        mask[s1_bot:s1_top, wall_x:wall_x+5] = 0 # Hole 1
        mask[s2_bot:s2_top, wall_x:wall_x+5] = 0 # Hole 2
        u_next[mask==1] = 0
        
        # B. Electron Source (The "Walker")
        # The electron PUSHES the vacuum at its current position
        ix, iy = int(px), int(py)
        if 0 < ix < Nx and 0 < iy < Ny:
            u_next[iy, ix] += 2.0 * np.sin(0.4 * t)
            
        # Cycle Buffers
        u_prev = u.copy()
        u = u_next.copy()
            
        # C. Guidance Force (Pilot Wave)
        # Calculate gradient (slope) of the wave field at electron position
        if ix < Nx-1 and iy < Ny-1 and ix > 1 and iy > 1:
            grad_y = (u[iy+1, ix] - u[iy-1, ix])
        else:
            grad_y = 0
            
        # D. Newton's Law
        # Wall Collision check
        if wall_x <= ix <= wall_x+5:
            if not ((s1_bot < iy < s1_top) or (s2_bot < iy < s2_top)):
                vx, vy = 0, 0 # Stopped by wall
        
        # The Force: "Surf the slope"
        # Increased sensitivity to make the wiggle visible
        vy += dt * (-0.1 * grad_y) 
        
        px += vx * dt
        py += vy * dt
        
        traj_x.append(px)
        traj_y.append(py)

    # 5. Visualization
    plt.figure(figsize=(10, 6))
    # Using vmin/vmax prevents color scaler from crashing on singularities
    plt.imshow(u, extent=[0, Nx, 0, Ny], origin='lower', cmap='RdBu', vmin=-1, vmax=1)
    
    # Draw Walls
    plt.axvline(x=wall_x, color='black', linewidth=3)
    plt.plot([wall_x, wall_x], [s1_bot, s1_top], 'w-', linewidth=3) 
    plt.plot([wall_x, wall_x], [s2_bot, s2_top], 'w-', linewidth=3) 
    
    # Draw Trajectory
    style = 'r-' if OBSERVER_ON else 'g-'
    label = 'Particle Mode (Observer ON)' if OBSERVER_ON else 'Wave Mode (Observer OFF)'
    plt.plot(traj_x, traj_y, style, linewidth=2, label=label)
    
    plt.legend(loc='upper right')
    plt.title(f"Simulation D.8: Observer Effect is {'ON' if OBSERVER_ON else 'OFF'}")
    plt.xlabel("Distance")
    plt.ylabel("Lateral Position")
    plt.show()

if __name__ == "__main__":
    simulate_observer_effect()