"""
AVE MODULE 4: LATTICE MEMORY (DOUBLE SLIT FDTD)
Strict discrete FDTD simulation demonstrating deterministic pilot-wave interference.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_lattice_double_slit():
    NX, NY = 200, 200
    P, V, wall = np.zeros((NX, NY)), np.zeros((NX, NY)), np.zeros((NX, NY))
    
    slit_width, slit_spacing, barrier_x = 8, 30, 60
    wall[barrier_x:barrier_x+4, :] = 1.0  
    wall[barrier_x:barrier_x+4, NY//2 - slit_spacing//2 - slit_width : NY//2 - slit_spacing//2] = 0.0
    wall[barrier_x:barrier_x+4, NY//2 + slit_spacing//2 : NY//2 + slit_spacing//2 + slit_width] = 0.0

    c2, steps, intensity = 0.25, 400, np.zeros((NX, NY)) 

    for t in range(steps):
        if t < 100: P[10, NY//2] += np.sin(0.4 * t) 
        laplacian = (np.roll(P, 1, axis=0) + np.roll(P, -1, axis=0) + np.roll(P, 1, axis=1) + np.roll(P, -1, axis=1) - 4 * P)
        V += c2 * laplacian
        P += V
        P[wall == 1.0] = 0.0; V[wall == 1.0] = 0.0
        
        if t > 150: intensity += P**2

    # Deterministic Trajectory
    path_x, path_y = [barrier_x + 4], [NY//2 - slit_spacing//2 - slit_width//2]
    curr_x, curr_y = path_x[0], path_y[0]
    
    for _ in range(120):
        if curr_x >= NX-2: break
        grad_y = intensity[curr_x, curr_y+1] - intensity[curr_x, curr_y-1]
        curr_x += 1 
        curr_y += int(np.sign(grad_y)) 
        path_x.append(curr_x); path_y.append(curr_y)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
    ax.imshow(intensity.T, cmap='inferno', origin='lower', alpha=0.9)
    ax.axvline(barrier_x, color='gray', lw=4, alpha=0.5)
    ax.plot(path_x, path_y, color='cyan', lw=2, label="Deterministic Particle Path\n(Passed through Slit A ONLY)")
    ax.scatter([path_x[0]], [path_y[0]], color='white', s=50, zorder=5)
    
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_title("Lattice Memory: The Deterministic Double Slit\n(Discrete Hardware Pressure Field)", color='white', fontsize=14)
    legend = ax.legend(loc='upper right', facecolor='black', edgecolor='white')
    for text in legend.get_texts(): text.set_color('white')

    plt.savefig(os.path.join(OUTPUT_DIR, "pilot_wave_interference.png"), dpi=300, bbox_inches='tight', facecolor='black')

if __name__ == "__main__": simulate_lattice_double_slit()