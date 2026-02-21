"""
AVE MODULE 9: OHMIC DECOHERENCE (DOUBLE SLIT FDTD)
--------------------------------------------------
Strict discrete FDTD simulation demonstrating deterministic pilot-wave interference.
Shows the LIVE pressure ripples diffracting through the slits using Gamma Correction.
Proves that a "Detector" is physically just an Ohmic Impedance Load ($Z_{EM}$) 
that mechanically dampens the pressure wave via viscous drag ($\\Omega \\equiv kg/s$).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_double_slit(use_detector=False):
    NX, NY = 400, 250
    P = np.zeros((NX, NY))  
    V = np.zeros((NX, NY))  
    intensity = np.zeros((NX, NY)) 
    
    barrier_x = 100
    slit_w = 12
    slit_s = 50
    wall = np.zeros((NX, NY), dtype=bool)
    wall[barrier_x:barrier_x+4, :] = True
    
    slit_A_start, slit_A_end = NY//2 - slit_s//2 - slit_w, NY//2 - slit_s//2
    slit_B_start, slit_B_end = NY//2 + slit_s//2, NY//2 + slit_s//2 + slit_w
    wall[barrier_x:barrier_x+4, slit_A_start:slit_A_end] = False
    wall[barrier_x:barrier_x+4, slit_B_start:slit_B_end] = False

    c2 = 0.4
    damping_field = np.zeros((NX, NY))
    
    for i in range(20):
        sponge = (20 - i) / 20.0 * 0.1
        damping_field[i, :] += sponge; damping_field[-i-1, :] += sponge
        damping_field[:, i] += sponge; damping_field[:, -i-1] += sponge
    
    if use_detector:
        # The Detector: A physical Impedance Load (Viscous Drag in kg/s)
        damping_field[barrier_x+4:barrier_x+30, slit_B_start-6:slit_B_end+6] = 0.5 

    steps = 1000
    print(f"Executing Lattice FDTD (Ohmic Measurement: {'ON' if use_detector else 'OFF'})...")
    for t in range(steps):
        if t < 900: 
            P[20, 20:NY-20] += np.sin(0.3 * t) * np.hanning(NY-40)
            
        laplacian = (np.roll(P, 1, axis=0) + np.roll(P, -1, axis=0) + 
                     np.roll(P, 1, axis=1) + np.roll(P, -1, axis=1) - 4 * P)
                     
        V += c2 * laplacian - damping_field * V
        P += V
        P[wall] = 0.0; V[wall] = 0.0
        
        if t > 600: intensity += P**2

    intensity = scipy.ndimage.gaussian_filter(intensity, sigma=1.5)
    post_barrier_max = np.max(intensity[barrier_x+10:, :])
    if post_barrier_max > 0: intensity /= post_barrier_max

    paths = []
    start_ys = np.linspace(slit_A_start + 2, slit_A_end - 2, 13)
    
    for sy in start_ys:
        px, py = [barrier_x + 5], [sy]
        curr_x, curr_y = float(px[0]), float(py[0])
        
        vy = (sy - (slit_A_start + slit_w/2)) * 0.12 
        
        for _ in range(NX - barrier_x - 15):
            if int(curr_x) >= NX-2 or int(curr_y) >= NY-2 or int(curr_y) <= 2: break
            
            grad_y = intensity[int(curr_x), int(curr_y)+1] - intensity[int(curr_x), int(curr_y)-1]
            vy += 3.5 * grad_y - 0.05 * vy 
            
            curr_x += 1.0; curr_y += vy
            px.append(curr_x); py.append(curr_y)
            
        paths.append((px, py))

    wave_snapshot = np.abs(P)
    return wave_snapshot, paths, slit_A_start, slit_A_end, slit_B_start, slit_B_end, barrier_x

def plot_comparison():
    wave_coh, paths_coh, sA_s, sA_e, sB_s, sB_e, bx = simulate_double_slit(use_detector=False)
    wave_det, paths_det, _, _, _, _, _ = simulate_double_slit(use_detector=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), facecolor='#050508')
    
    vis_coh = np.power(wave_coh, 0.4)
    vis_det = np.power(wave_det, 0.4)

    # --- Subplot 1: Coherent ---
    ax1 = axes[0]
    ax1.imshow(vis_coh.T, cmap='inferno', origin='lower', alpha=0.9, vmin=0, vmax=np.max(vis_coh)*0.6)
    ax1.axvline(bx, color='gray', lw=4, alpha=0.6)
    
    ax1.add_patch(patches.Rectangle((bx-2, 0), 4, sA_s, facecolor='silver', alpha=1.0))
    ax1.add_patch(patches.Rectangle((bx-2, sA_e), 4, sB_s - sA_e, facecolor='silver', alpha=1.0))
    ax1.add_patch(patches.Rectangle((bx-2, sB_e), 4, 250 - sB_e, facecolor='silver', alpha=1.0))
    
    for (px, py) in paths_coh:
        ax1.plot(px, py, color='#00ffcc', lw=2.0, alpha=0.9)
        ax1.scatter([px[0]], [py[0]], color='white', s=20, zorder=5) 
    
    ax1.set_title("COHERENT MODE: The Quantized Pilot Wave", color='white', fontsize=16, pad=15, weight='bold')
    ax1.text(bx+15, sA_e+20, "1. Vacuum Wake clearly passes through both slits, creating distinct interference ripples.\n2. Particles from Slit A are deterministically steered into quantized bands by the pressure gradients.", color='#00ffcc', fontsize=12, weight='bold')

    # --- Subplot 2: Measured (Damped) ---
    ax2 = axes[1]
    ax2.imshow(vis_det.T, cmap='inferno', origin='lower', alpha=0.9, vmin=0, vmax=np.max(vis_det)*0.6)
    ax2.axvline(bx, color='gray', lw=4, alpha=0.6)
    
    ax2.add_patch(patches.Rectangle((bx-2, 0), 4, sA_s, facecolor='silver', alpha=1.0))
    ax2.add_patch(patches.Rectangle((bx-2, sA_e), 4, sB_s - sA_e, facecolor='silver', alpha=1.0))
    ax2.add_patch(patches.Rectangle((bx-2, sB_e), 4, 250 - sB_e, facecolor='silver', alpha=1.0))
    
    detector_rect = patches.Rectangle((bx+4, sB_s-5), 25, (sB_e-sB_s)+10, linewidth=2, edgecolor='#ff3366', facecolor='#ff3366', alpha=0.4, hatch='//')
    ax2.add_patch(detector_rect)
    ax2.text(bx+16, sB_e+15, r"Impedance Load\n($1~\Omega \equiv \xi_{topo}^{-2}$ kg/s)", color='#ff3366', ha='center', fontsize=12, fontweight='bold')

    for (px, py) in paths_det:
        ax2.plot(px, py, color='orange', lw=2.0, alpha=0.9)
        ax2.scatter([px[0]], [py[0]], color='white', s=20, zorder=5)
    
    ax2.set_title("MEASURED MODE: Ohmic Decoherence", color='white', fontsize=16, pad=15, weight='bold')
    ax2.text(bx+15, sA_e+20, "1. Detector physically dissipates the Slit B wave via Joule heating (Mechanical Drag).\n2. With the interference ridges destroyed, particles revert to a classical ballistic spray.", color='orange', fontsize=12, weight='bold')

    for ax in axes:
        ax.set_facecolor('#050508')
        ax.axis('off')
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 250)
        
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "impedance_decoherence.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050508')
    print(f"Simulation saved to {output_path}")

if __name__ == "__main__":
    plot_comparison()