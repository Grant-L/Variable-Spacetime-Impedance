#!/usr/bin/env python3
r"""
PONDER-01: Slow-Motion Topological Back-Reaction
================================================

Generates a highly readable, slow-motion (5+ seconds) 3D topographical
animation of the LC lattice density. The camera is fixed, the colors are 
high-contrast, and the array explicitly drives the continuum pressure.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

# Bind into the AVE framework
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.ave.core.fdtd_3d import FDTD3DEngine

def generate_slow_motion_density_animation():
    print("[*] Initializing Slow-Motion Topological Animator...")
    
    # Large grid to clearly see wave propagation outward
    GRID_SIZE = 80
    RESOLUTION_M = 0.05
    engine = FDTD3DEngine(nx=GRID_SIZE, ny=GRID_SIZE, nz=GRID_SIZE, dx=RESOLUTION_M)
    
    FREQUENCY = 100.0e6 
    
    center_x = GRID_SIZE // 2
    center_y = GRID_SIZE // 2
    
    num_elements = 8
    radius = 12
    angles = np.linspace(0, 2 * np.pi, num_elements, endpoint=False)
    
    dipole_z_start = GRID_SIZE // 4
    dipole_z_end = 3 * (GRID_SIZE // 4)
    
    antennas = []
    for i, angle in enumerate(angles):
        sx = int(center_x + radius * np.cos(angle))
        sy = int(center_y + radius * np.sin(angle))
        phase_shift = i * (np.pi / 4.0)
        antennas.append({'x': sx, 'y': sy, 'phase': phase_shift})
        
    # We want a very smooth, long animation.
    # Total physical time = frames * steps * dt
    # 120 frames at 15fps = 8 seconds of animation.
    TOTAL_FRAMES = 120
    STEPS_PER_FRAME = 2 # Very fine temporal resolution
    
    print(f"[*] Computing {TOTAL_FRAMES} high-resolution Maxwell frames...")
    
    z_slice_idx = GRID_SIZE // 2
    frames_data = []
    
    for frame in range(TOTAL_FRAMES):
        for _ in range(STEPS_PER_FRAME):
            t = engine.dt * (frame * STEPS_PER_FRAME + _)
            for src in antennas:
                signal = np.sin(2.0 * np.pi * FREQUENCY * t - src['phase'])
                for z in range(dipole_z_start, dipole_z_end):
                    engine.inject_soft_source('Ez', src['x'], src['y'], z, signal * 200.0)
            engine.step()
            
        # Extract the 2D plane and smooth it slightly for rendering aesthetics
        raw_slice = engine.Ez[:, :, z_slice_idx].copy()
        frames_data.append(raw_slice.T) # Transpose for proper X/Y alignment
        sys.stdout.write(f"\r  -> Computed frame {frame+1}/{TOTAL_FRAMES}")
        sys.stdout.flush()
        
    print("\n[*] FDTD Matrix computation complete. Rendering Slow-Motion 3D Surface...")
    
    # ---------------------------------------------------------------------
    # Render High-Contrast 3D Surface Animation
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    # We use a dark background for maximum contrast with the "luminous" topological waves
    plt.style.use('dark_background')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    ax.set_box_aspect([1, 1, 0.5]) # Flatter Z-axis to emphasize the horizontal ripples
    
    X, Y = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
    
    # Calculate global max for stable Z-axis limits
    v_max = np.max(np.abs(frames_data[-1])) / 1.5
    
    # Initial Plot Setup
    # Using 'magma' colormap as it creates bright, distinct ripples over a dark floor
    surf = [ax.plot_surface(X, Y, frames_data[0], cmap='magma', vmin=-v_max, vmax=v_max, rstride=1, cstride=1, antialiased=True, alpha=0.9)]
    
    # Draw strictly rigid physical antenna pillars
    for src in antennas:
        ax.plot([src['x'], src['x']], [src['y'], src['y']], [-v_max, v_max], color='cyan', linewidth=3, linestyle='-', zorder=10)
    
    ax.set_zlim(-v_max * 1.5, v_max * 1.5)
    
    ax.set_title(r"PONDER-01: Slow-Motion Topological Acoustic Rectification" + "\n" + r"$\Delta\phi = 45^{\circ}$ Phased Standing-Wave Formation", fontsize=16, fontweight='bold', color='white', pad=20)
    
    # We turn off the axis panes to make the wave float cleanly in space
    ax.set_axis_off()
    
    # Set a dramatic, fixed camera angle looking slightly downward at the array center
    ax.view_init(elev=35, azim=45)
    
    # Timing overlay text
    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, color='cyan', fontsize=12)

    def update(frame):
        surf[0].remove()
        
        # Adding a light source for dramatic shadowing on the surface waves
        ls = LightSource(azdeg=180, altdeg=45)
        rgb = ls.shade(frames_data[frame], cmap=plt.get_cmap('magma'), vert_exag=0.5, blend_mode='soft')
        
        surf[0] = ax.plot_surface(X, Y, frames_data[frame], facecolors=rgb, rstride=1, cstride=1, antialiased=True, linewidth=0)
        
        t_ns = frame * STEPS_PER_FRAME * engine.dt * 1e9
        time_text.set_text(f"Evolution Time: {t_ns:.2f} ns")
        return surf[0], time_text
        
    print("[*] Generating GIF (This will take a minute for 120 shaded 3D frames)...")
    ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=False)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    anim_path = os.path.join(out_dir, 'ponder_01_lattice_density_slow.gif')
    
    # 15 fps for 120 frames = exactly 8.0 seconds duration!
    ani.save(anim_path, writer='pillow', fps=15)
    plt.close(fig)
    
    # Restore style so we don't accidentally break future plots
    plt.style.use('default')
    
    print(f"[*] Awesome Slow-Motion Animation complete -> {anim_path}")

if __name__ == "__main__":
    generate_slow_motion_density_animation()
