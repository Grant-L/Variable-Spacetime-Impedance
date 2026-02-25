#!/usr/bin/env python3
r"""
PONDER-01: True Volumetric OAM 3D Animation (Omnidirectional)
=============================================================

Renders a TRUE 3D volumetric visualization of the Orbital Angular Momentum (OAM) 
wave expanding spherically outward in all directions.

By centering the array in the absolute middle of the grid, we show that 
the acoustic topological wave is not a one-directional "laser beam", but 
rather an expanding spherical wavefront with helical/twisted arms.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Bind into the AVE framework
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.ave.core.fdtd_3d import FDTD3DEngine

def generate_volumetric_animation():
    print("[*] Initializing PONDER-01 True Volumetric Omnidirectional Animator...")
    
    # We want a high-res cubic grid to see the helical rotation
    GRID_SIZE = 60
    RESOLUTION_M = 0.05
    engine = FDTD3DEngine(nx=GRID_SIZE, ny=GRID_SIZE, nz=GRID_SIZE, dx=RESOLUTION_M)
    
    FREQUENCY = 100.0e6 
    
    center_x = GRID_SIZE // 2
    center_y = GRID_SIZE // 2
    center_z = GRID_SIZE // 2
    
    num_elements = 8
    radius = 10
    angles = np.linspace(0, 2 * np.pi, num_elements, endpoint=False)
    
    # Place array directly in the center to allow full 3D spherical expansion
    dipole_z_start = center_z - 4
    dipole_z_end = center_z + 4
    
    antennas = []
    for i, angle in enumerate(angles):
        sx = int(center_x + radius * np.cos(angle))
        sy = int(center_y + radius * np.sin(angle))
        phase_shift = i * (np.pi / 4.0)
        antennas.append({'x': sx, 'y': sy, 'phase': phase_shift})
        
    # We want a smooth, long animation.
    STEPS_PER_FRAME = 2
    TOTAL_FRAMES = 80 # ~5.3 seconds at 15fps
    
    print(f"[*] Computing {TOTAL_FRAMES} full-volume Maxwell frames. Array centered for 3D propagation...")
    
    # We will pre-calculate the X,Y,Z mesh grids for scatter lookups
    X, Y, Z = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing='ij')
    
    # Create an exclusion mask to keep the source nodes from blinding the render
    r_squared = (X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2
    core_mask = r_squared <= (radius - 2)**2
    
    frames_coords = []
    frames_colors = []
    
    for frame in range(TOTAL_FRAMES):
        for _ in range(STEPS_PER_FRAME):
            t = engine.dt * (frame * STEPS_PER_FRAME + _)
            for src in antennas:
                signal = np.sin(2.0 * np.pi * FREQUENCY * t - src['phase'])
                for z in range(dipole_z_start, dipole_z_end):
                    engine.inject_soft_source('Ez', src['x'], src['y'], z, signal * 300.0)
            engine.step()
            
        # Extract the entire 3D volume
        full_vol = engine.Ez.copy()
        
        # Suppress the extremely loud center elements so we can see the propagating wave
        full_vol[core_mask] = 0.0
        
        # Lower threshold to capture more of the fainter spherical expansion
        threshold = np.max(np.abs(full_vol)) * 0.18
        
        mask = np.abs(full_vol) > threshold
        
        frames_coords.append((X[mask], Y[mask], Z[mask]))
        frames_colors.append(full_vol[mask])
        
        sys.stdout.write(f"\r  -> Computed frame {frame+1}/{TOTAL_FRAMES}")
        sys.stdout.flush()
        
    print("\n[*] FDTD Matrix computation complete. Rendering 3D Volumetric Scatter GIF...")
    
    fig = plt.figure(figsize=(10, 10))
    # Dark background for neon wave contrast
    plt.style.use('dark_background')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.set_box_aspect([1, 1, 1])
    
    v_max = np.max(np.abs(frames_colors[-1])) if len(frames_colors[-1]) > 0 else 1.0
    
    # Start blank 
    scat = [ax.scatter([], [], [], c=[], cmap='bwr', vmin=-v_max, vmax=v_max, s=40, alpha=0.4, edgecolors='none')]
    
    # Draw physical antennas
    for src in antennas:
        ax.plot([src['x'], src['x']], [src['y'], src['y']], [dipole_z_start, dipole_z_end], 
                color='cyan', linewidth=4, linestyle='-', zorder=10)
    
    # Add a phantom center marker
    ax.scatter([center_x], [center_y], [center_z], color='white', marker='x', s=100, zorder=15)
    
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_zlim(0, GRID_SIZE)
    
    ax.set_title(r"True 3D Volumetric OAM ($E_z$)" + "\n" + r"Omnidirectional Spherical Helical Propagation", fontsize=16, fontweight='bold', color='white', pad=20)
    ax.set_axis_off()
    
    # Elevate the camera so we can see the upward and downward expansion
    ax.view_init(elev=30, azim=45)
    
    def update(frame):
        # We must completely remove the old scatter and plot a new one
        # because scatter points change size/count every frame
        scat[0].remove()
        cx, cy, cz = frames_coords[frame]
        cval = frames_colors[frame]
        
        if len(cx) > 0:
            scat[0] = ax.scatter(cx, cy, cz, c=cval, cmap='bwr', vmin=-v_max, vmax=v_max, s=20, alpha=0.4, edgecolors='none')
        else:
            scat[0] = ax.scatter([], [], [], c=[], cmap='bwr')
            
        # Rotate camera slowly to fully grasp the 3D volumetric depth
        ax.view_init(elev=30, azim=45 + (frame * 0.8))
        return scat[0],
        
    print("[*] Generating GIF (This will take a minute for thresholded 3D scatter)...")
    ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=False)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    anim_path = os.path.join(out_dir, 'ponder_01_fdtd_volumetric.gif')
    
    # Smooth 15 fps
    ani.save(anim_path, writer='pillow', fps=15)
    plt.close(fig)
    plt.style.use('default')
    
    print(f"[*] True 3D Volumetric Animation complete -> {anim_path}")

if __name__ == "__main__":
    generate_volumetric_animation()
