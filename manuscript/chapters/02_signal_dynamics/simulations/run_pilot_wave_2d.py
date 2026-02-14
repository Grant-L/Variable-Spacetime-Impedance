import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_pilot_wave_comparison():
    print("Simulating High-Fidelity Pilot Wave Mechanism...")
    
    # 1. SETUP HIGH-RES VACUUM TANK
    NX, NY = 400, 200
    L_x, L_y = 8.0, 4.0
    dx = L_x / NX
    dy = L_y / NY
    
    # Wave Grids
    h = np.zeros((NY, NX))
    v = np.zeros((NY, NX))
    
    # 2. SETUP BARRIER
    wall_x = int(NX * 0.25)
    slit_w = 8    # Slit width (pixels)
    slit_sep = 20 # Separation (pixels)
    cy = NY // 2
    
    barrier = np.zeros((NY, NX))
    barrier[:, wall_x-3:wall_x+3] = 1.0 
    
    # Cut Slits
    # Top Slit
    barrier[cy + slit_sep//2 : cy + slit_sep//2 + slit_w, wall_x-3:wall_x+3] = 0.0
    # Bottom Slit
    barrier[cy - slit_sep//2 - slit_w : cy - slit_sep//2, wall_x-3:wall_x+3] = 0.0
    
    # 3. PARTICLE SETUP (Top Slit Trajectory)
    # Start aligned with top slit center
    px = 0.5
    py_start = (cy + slit_sep//2 + slit_w/2) * dy
    py = py_start
    vx, vy = 1.8, 0.0 # Moving right
    
    # 4. PHYSICS ENGINE
    c = 4.0        # Wave Speed
    dt = 0.004     # Fine time step
    steps = 800
    
    traj_x, traj_y = [], []
    
    for t in range(steps):
        # -- Wave Equation (Finite Difference) --
        # Laplacian
        lap_h = (np.roll(h, 1, axis=0) + np.roll(h, -1, axis=0) + 
                 np.roll(h, 1, axis=1) + np.roll(h, -1, axis=1) - 4*h) / (dx**2)
        
        # Update Velocity & Height (Damped Wave Eq)
        v += (c**2 * lap_h) * dt * 0.995 
        h += v * dt
        
        # Apply Barrier (Reflect)
        h[barrier > 0.5] = 0
        v[barrier > 0.5] = 0
        
        # -- Coupling --
        # 1. Particle Excites Vacuum (The Source)
        pi, pj = int(px/dx), int(py/dy)
        if 0 <= pi < NX and 0 <= pj < NY:
            # Oscillating source term at particle location
            h[pj, pi] += 10.0 * np.cos(1.5 * t * dt) * dt 
            
        # 2. Vacuum Guides Particle (The Pilot)
        # Calculate local gradient
        if 0 < pi < NX-1 and 0 < pj < NY-1:
            grad_x = (h[pj, pi+1] - h[pj, pi-1]) / (2*dx)
            grad_y = (h[pj+1, pi] - h[pj-1, pi]) / (2*dy)
            
            # Gradient Force (Surfing)
            # The particle accelerates DOWN the pressure gradient
            vx += -grad_x * 12.0 * dt
            vy += -grad_y * 12.0 * dt
            
        # Update Position
        px += vx * dt
        py += vy * dt
        traj_x.append(px)
        traj_y.append(py)
        
        if px > L_x: break

    # 5. VISUALIZATION
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='black')
    
    # Plot A: The Non-Local Wave
    ax1 = axes[0]
    # Enhance contrast for wave visualization
    im1 = ax1.imshow(h, extent=[0, L_x, 0, L_y], origin='lower', cmap='RdBu', vmin=-0.2, vmax=0.2)
    ax1.imshow(barrier, extent=[0, L_x, 0, L_y], origin='lower', cmap='binary', alpha=0.4)
    ax1.set_title("1. The Vacuum Wave\n(Non-Local: Passes through BOTH slits)", color='white', fontsize=14)
    ax1.axis('off')
    
    # Plot B: The Local Particle
    ax2 = axes[1]
    # Faint background wave
    ax2.imshow(h, extent=[0, L_x, 0, L_y], origin='lower', cmap='gray', vmin=-0.5, vmax=0.5, alpha=0.3)
    ax2.imshow(barrier, extent=[0, L_x, 0, L_y], origin='lower', cmap='binary', alpha=0.4)
    
    # Trajectory
    ax2.plot(traj_x, traj_y, color='cyan', linewidth=2.5, label='Particle Path')
    # Final Position
    ax2.scatter([traj_x[-1]], [traj_y[-1]], color='yellow', s=150, zorder=5, edgecolors='black', label='Particle ($N=3$)')
    
    # Annotations
    ax2.set_title("2. The Particle Trajectory\n(Local: Passes through ONE slit, Surfs the Wake)", color='white', fontsize=14)
    
    # Arrow indicating the interference kick
    kick_x = L_x * 0.45
    kick_y = L_y * 0.6
    ax2.annotate("Interference Kick", xy=(kick_x, kick_y), xytext=(kick_x, kick_y+0.8),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 color='red', fontsize=12, ha='center')

    ax2.legend(loc='lower right')
    ax2.axis('off')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "pilot_wave_comparison.png")
    plt.savefig(output_path, dpi=300, facecolor='black')
    print(f"Comparison simulation saved to {output_path}")

if __name__ == "__main__":
    simulate_pilot_wave_comparison()