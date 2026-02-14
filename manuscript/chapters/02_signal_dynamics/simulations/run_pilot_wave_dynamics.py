import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_pilot_wave():
    print("Simulating High-Fidelity Lattice Memory...")
    
    # 1. SETUP
    NX, NY = 400, 200 # High Res
    L_x, L_y = 8.0, 4.0
    dx = L_x / NX
    dy = L_y / NY
    
    h = np.zeros((NY, NX))
    v = np.zeros((NY, NX))
    
    # 2. BARRIER
    wall_x = int(NX * 0.25)
    slit_w = 8
    slit_sep = 20
    cy = NY // 2
    
    barrier = np.zeros((NY, NX))
    barrier[:, wall_x-3:wall_x+3] = 1.0
    
    # Slits
    barrier[cy + slit_sep//2 : cy + slit_sep//2 + slit_w, wall_x-3:wall_x+3] = 0.0
    barrier[cy - slit_sep//2 - slit_w : cy - slit_sep//2, wall_x-3:wall_x+3] = 0.0
    
    # 3. PARTICLE
    px = 0.5
    py = (cy + slit_sep//2 + slit_w/2) * dy # Top slit
    vx, vy = 1.8, 0.0
    
    c_wave = 4.0
    dt = 0.004
    steps = 850
    
    traj_x, traj_y = [], []
    
    # 4. RUN
    for t in range(steps):
        # Wave
        lap_h = (np.roll(h, 1, axis=0) + np.roll(h, -1, axis=0) + 
                 np.roll(h, 1, axis=1) + np.roll(h, -1, axis=1) - 4*h) / (dx**2)
        v += (c_wave**2 * lap_h) * dt * 0.995
        h += v * dt
        h[barrier > 0.5] = 0
        v[barrier > 0.5] = 0
        
        # Particle Source
        pi, pj = int(px/dx), int(py/dy)
        if 0 <= pi < NX and 0 <= pj < NY:
            h[pj, pi] += 10.0 * np.cos(1.5 * t * dt) * dt
            
        # Particle Guide
        if 0 < pi < NX-1 and 0 < pj < NY-1:
            grad_x = (h[pj, pi+1] - h[pj, pi-1]) / (2*dx)
            grad_y = (h[pj+1, pi] - h[pj-1, pi]) / (2*dy)
            vx += -grad_x * 12.0 * dt
            vy += -grad_y * 12.0 * dt
            
        px += vx * dt
        py += vy * dt
        traj_x.append(px)
        traj_y.append(py)
        if px > L_x: break

    # 5. RENDER
    plt.figure(figsize=(12, 8), facecolor='black')
    
    # Main Field
    plt.imshow(h, extent=[0, L_x, 0, L_y], origin='lower', cmap='inferno', vmin=-0.15, vmax=0.15)
    plt.imshow(barrier, extent=[0, L_x, 0, L_y], origin='lower', cmap='binary', alpha=0.5)
    
    # Path
    plt.plot(traj_x, traj_y, 'w-', linewidth=2, label='Deterministic Path')
    plt.scatter([traj_x[-1]], [traj_y[-1]], c='cyan', s=100, edgecolors='white', label='Particle')
    
    # Decorations
    plt.title("Lattice Memory: The Hydrodynamic Origin of Quantization", color='white', fontsize=16)
    plt.xlabel("Propagation Distance ($l_0$)", color='white')
    plt.ylabel("Transverse Position", color='white')
    plt.legend(loc='lower right')
    
    # Colorbar
    cbar = plt.colorbar(orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label("Vacuum Potential Pressure ($P_{vac}$)", color='white')
    cbar.ax.xaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')
    
    plt.gca().tick_params(axis='x', colors='white')
    plt.gca().tick_params(axis='y', colors='white')
    
    output_path = os.path.join(OUTPUT_DIR, "pilot_wave_interference.png")
    plt.savefig(output_path, dpi=300, facecolor='black')
    print(f"Interference simulation saved to {output_path}")

if __name__ == "__main__":
    simulate_pilot_wave()