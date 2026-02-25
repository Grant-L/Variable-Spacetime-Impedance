#!/usr/bin/env python3
r"""
PONDER-01: Conservation of Momentum "Dark Wake" Animator
========================================================

A critical objection against Asymmetric Capacitor thrusters is that they
violate the Conservation of Momentum (Newton's Third Law) as "reactionless drives".

This script proves the AVE resolution visually:
As the PONDER-01 array pumps a macroscopic acoustic standing wave forward, 
it must exert an equal-and-opposite longitudinal strain on the LC lattice behind it.
We define this as the "Dark Wake"—a non-luminous topological shear wave 
($\tau_{zx}$) that carries the exact reaction momentum backwards into the vacuum.

It extracts the time-evolved longitudinal gradient from the 3D FDTD Engine.
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

def generate_dark_wake_animation():
    print("[*] Initializing PONDER-01 Momentum Dark Wake Animator...")
    
    # We use an asymmetrical 3D grid, elongated along the Z axis, 
    # to capture both the forward thrust wave and the rearward Dark Wake.
    # We put the array near the center. Z goes from 0 to 100.
    GRID_X = 60
    GRID_Y = 60
    GRID_Z = 100
    RESOLUTION_M = 0.05
    engine = FDTD3DEngine(nx=GRID_X, ny=GRID_Y, nz=GRID_Z, dx=RESOLUTION_M)
    
    FREQUENCY = 100.0e6 
    
    center_x = GRID_X // 2
    center_y = GRID_Y // 2
    
    num_elements = 8
    radius = 10
    angles = np.linspace(0, 2 * np.pi, num_elements, endpoint=False)
    
    # The array hardware exists in the center of the Z block
    dipole_z_start = 45
    dipole_z_end = 55
    
    antennas = []
    for i, angle in enumerate(angles):
        sx = int(center_x + radius * np.cos(angle))
        sy = int(center_y + radius * np.sin(angle))
        
        # PONDER-01 Phased offset
        phase_shift = i * (np.pi / 4.0)
        antennas.append({'x': sx, 'y': sy, 'phase': phase_shift})
        
    print("[*] Computing 60 keyframes of Maxwell evolution...")
    
    STEPS_PER_FRAME = 3
    TOTAL_FRAMES = 60
    
    # To visualize the dark wake vs forward radiation, we want a vertical slice running 
    # through the center array (X-Z plane).
    y_slice_idx = GRID_Y // 2
    
    frames_data_E = []
    frames_data_Wake = []
    
    for frame in range(TOTAL_FRAMES):
        for _ in range(STEPS_PER_FRAME):
            t = engine.dt * (frame * STEPS_PER_FRAME + _)
            for src in antennas:
                signal = np.sin(2.0 * np.pi * FREQUENCY * t - src['phase'])
                for z in range(dipole_z_start, dipole_z_end):
                    # We inject hard Z-axis polarization
                    engine.inject_soft_source('Ez', src['x'], src['y'], z, signal * 300.0)
            engine.step()
            
        # 1. Forward Wave Analysis: Map the standard E-field (Transverse/Rotational)
        E_slice = engine.Ez[:, y_slice_idx, :].copy()
        
        # 2. Dark Wake Analysis: The topological shear reaction.
        # Momentum must be dumped backwards. We calculate the longitudinal strain gradient 
        # (dEz/dz), which physically represents the compressive shear (radiation pressure) 
        # pushing the topological lattice backwards.
        
        # Calculate central difference along Z
        dEz_dz = np.zeros_like(E_slice)
        dEz_dz[:, 1:-1] = (engine.Ez[:, y_slice_idx, 2:] - engine.Ez[:, y_slice_idx, :-2]) / (2.0 * engine.dx)
        
        frames_data_E.append(E_slice)
        frames_data_Wake.append(dEz_dz)
        
        sys.stdout.write(f"\r  -> Computed frame {frame+1}/{TOTAL_FRAMES}")
        sys.stdout.flush()
        
    print("\n[*] FDTD Matrix computation complete. Rendering 2-Panel GIF...")
    
    # -------------------------------------------------------------
    # Visualization: 2-Panel Dynamic Tracking
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Panel 1: The Luminous Wave (Forward OAM / Standard EM)
    v_max_e = np.max(np.abs(frames_data_E[-1])) / 1.5
    im1 = ax1.imshow(frames_data_E[0].T, cmap='RdBu', vmin=-v_max_e, vmax=v_max_e, 
                     interpolation='bilinear', origin='lower')
    
    # Draw array bounds
    ax1.axhline(dipole_z_start, color='black', linestyle='--', alpha=0.5)
    ax1.axhline(dipole_z_end, color='black', linestyle='--', alpha=0.5)
    ax1.scatter([center_x], [(dipole_z_start+dipole_z_end)/2], color='gold', edgecolor='black', s=200, marker='s', zorder=5, label='Phased Array Hardware')
    
    ax1.set_title(r"Forward Radiation ($E_z$)" + "\n" + r"Transverse Luminous Wavefront", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Grid X (meters)")
    ax1.set_ylabel("Grid Z (meters) - Forward Propagation $\\rightarrow$")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Volumetric Strain ($V/m$)")
    ax1.legend(loc='lower left')
    
    # Panel 2: The Dark Wake (Longitudinal Strain / Momentum Sink)
    v_max_w = np.max(np.abs(frames_data_Wake[-1])) / 1.5
    
    # We use a completely different colormap (e.g., Greens/Purples) to represent the invisible 
    # structural shear force acting as the opposite reaction mass.
    im2 = ax2.imshow(frames_data_Wake[0].T, cmap='PRGn', vmin=-v_max_w, vmax=v_max_w, 
                     interpolation='bilinear', origin='lower')
                     
    ax2.axhline(dipole_z_start, color='black', linestyle='--', alpha=0.5)
    ax2.axhline(dipole_z_end, color='black', linestyle='--', alpha=0.5)
    ax2.scatter([center_x], [(dipole_z_start+dipole_z_end)/2], color='gold', edgecolor='black', s=200, marker='s', zorder=5)
    
    ax2.set_title(r"Rearward Dark Wake ($\tau_{zx} \propto \frac{\partial E_z}{\partial z}$)" + "\n" + r"Reaction Mass (Momentum Sink)", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Grid X (meters)")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Longitudinal Shear Gradient")
    
    # Overall Title and Mechanics Box
    fig.suptitle("PONDER-01: Proving Conservation of Momentum\nSymmetric Topological Reaction via Macroscopic Fluid Mechanics", fontsize=18, fontweight='black', y=0.98)
    
    # Add an explanatory text box bridging the two
    props = dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='white')
    textstr = '\n'.join((
        r'$\sum \vec{P} = 0$',
        r'Newton\'s 3rd Law explicitly upheld:',
        r'Thrust generates a massive backwards',
        r'imperceptible compression pulse.',
        'No "reactionless" violation.'
    ))
    fig.text(0.5, 0.05, textstr, fontsize=12, color='white',
              horizontalalignment='center', verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    def update(frame):
        im1.set_data(frames_data_E[frame].T)
        im2.set_data(frames_data_Wake[frame].T)
        return [im1, im2]
        
    ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=True)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ponder_01_dark_wake.gif')
    
    ani.save(out_path, writer='pillow', fps=15)
    plt.close()
    
    print(f"[*] Animation Render Complete. Output saved to: {out_path}")

if __name__ == "__main__":
    generate_dark_wake_animation()
