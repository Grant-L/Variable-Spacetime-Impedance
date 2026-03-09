#!/usr/bin/env python3
r"""
PONDER-01: Phased Array vs Lattice Density Animation
====================================================

Generates a 2D Heatmap and 3D surface plot animation showing the exact acoustic 
pressure ("Lattice Density") ripples accumulating from the phased array. 
Exports an animated .gif and 4 sequential static frames for the manuscript.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Bind into the AVE framework
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from src.ave.core.fdtd_3d_jax import FDTD3DEngineJAX as FDTD3DEngine
except ImportError:
    from src.ave.core.fdtd_3d import FDTD3DEngine

def generate_density_animation():
    print("[*] Initializing PONDER-01 Lattice Density Animator...")
    
    GRID_SIZE = 80 # slightly larger for better centering visibility
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
        
    print("[*] Computing 60 keyframes of Maxwell evolution...")
    
    STEPS_PER_FRAME = 4 # slightly slower evolution over more time
    TOTAL_FRAMES = 60
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
        frames_data.append(np.array(engine.Ez[:, :, z_slice_idx]))
        sys.stdout.write(f"\r  -> Computed frame {frame+1}/{TOTAL_FRAMES}")
        sys.stdout.flush()
        
    print("\n[*] FDTD Matrix computation complete. Rendering Graphics...")
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    
    v_max = np.nanmax(np.abs(frames_data[-1]))
    v_max = max(float(v_max) / 1.5, 1e-6) if np.isfinite(v_max) else 1e-6
    
    # ---------------------------------------------------------------------
    # 1. Export 4 Static Snapshots for Manuscript
    # ---------------------------------------------------------------------
    snapshot_frames = [14, 29, 44, 59]
    for idx, f_idx in enumerate(snapshot_frames):
        fig_snap, ax_snap = plt.subplots(figsize=(8, 8))
        im_snap = ax_snap.imshow(frames_data[f_idx].T, cmap='plasma', vmin=-v_max, vmax=v_max, origin='lower')
        # Draw antennas
        for src in antennas:
            ax_snap.scatter(src['x'], src['y'], color='yellow', edgecolor='black', s=100, zorder=5)
            
        ax_snap.set_title(rf"PONDER-01: Acoustic Lattice Density (t={f_idx * STEPS_PER_FRAME * engine.dt * 1e9:.1f} ns)", fontsize=14, fontweight='bold', pad=15)
        ax_snap.set_xlabel("Grid X (meters)")
        ax_snap.set_ylabel("Grid Y (meters)")
        
        # Grid marks in meters
        ticks = np.arange(0, GRID_SIZE+1, 20)
        ax_snap.set_xticks(ticks)
        ax_snap.set_yticks(ticks)
        ax_snap.set_xticklabels([f"{t*RESOLUTION_M:.1f}" for t in ticks])
        ax_snap.set_yticklabels([f"{t*RESOLUTION_M:.1f}" for t in ticks])
        
        plt.colorbar(im_snap, ax=ax_snap, fraction=0.046, pad=0.04, label="Continuum Pressure")
        plt.tight_layout()
        snap_path = os.path.join(out_dir, f'ponder_01_density_frame_{idx+1}.png')
        plt.savefig(snap_path, dpi=300, bbox_inches='tight')
        plt.close(fig_snap)
        print(f"[*] Saved static frame {idx+1} to {snap_path}")
        
    # ---------------------------------------------------------------------
    # 2. Render Dual-Panel Animated GIF
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 7))
    
    # Left: 2D Heatmap
    ax1 = fig.add_subplot(121)
    im1 = ax1.imshow(frames_data[0].T, cmap='plasma', vmin=-v_max, vmax=v_max, origin='lower')
    for src in antennas:
        ax1.scatter(src['x'], src['y'], color='yellow', edgecolor='black', s=100, zorder=5)
    
    ax1.set_title(r"2D Heatmap (Top-Down)", fontsize=14, fontweight='bold')
    ax1.set_xlabel(f"Grid X (width = {GRID_SIZE*RESOLUTION_M:.1f}m)")
    ax1.set_ylabel(f"Grid Y (width = {GRID_SIZE*RESOLUTION_M:.1f}m)")
    
    ticks = np.arange(0, GRID_SIZE+1, 20)
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_xticklabels([f"{t*RESOLUTION_M:.1f}" for t in ticks])
    ax1.set_yticklabels([f"{t*RESOLUTION_M:.1f}" for t in ticks])
    
    # Right: 3D Surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_box_aspect([1, 1, 0.4])
    X, Y = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
    surf = [ax2.plot_surface(X, Y, frames_data[0].T, cmap='plasma', vmin=-v_max, vmax=v_max, antialiased=False)]
    
    ax2.set_zlim(-v_max*1.5, v_max*1.5)
    ax2.set_title(r"3D Topographical Surface", fontsize=14, fontweight='bold')
    ax2.view_init(elev=40, azim=45) # Fixed camera
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    
    fig.suptitle(r"PONDER-01: Time-Evolved Lattice Density ($\Delta\rho_{LC}$)" + "\n" + "Acoustic Standing Wave Back-Reaction", fontsize=16, fontweight='black', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    def update(frame):
        # Update Heatmap
        im1.set_data(frames_data[frame].T)
        # Update Surface
        surf[0].remove()
        surf[0] = ax2.plot_surface(X, Y, frames_data[frame].T, cmap='plasma', vmin=-v_max, vmax=v_max, antialiased=False)
        return [im1, surf[0]]
        
    ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=False)
    anim_path = os.path.join(out_dir, 'ponder_01_lattice_density.gif')
    # Slow down animation to 8 fps for human readability
    ani.save(anim_path, writer='pillow', fps=8)
    plt.close(fig)
    
    print(f"[*] Animation Render Complete. Output saved to: {anim_path}")

if __name__ == "__main__":
    generate_density_animation()
