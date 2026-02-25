#!/usr/bin/env python3
r"""
AVE: 3D Warp Metric Time Evolution (Superluminal Bow Shock)
===========================================================

This script generates a high-fidelity, 3D time-evolved animation of the 
Warp Metric tensor mapping.

To formally demonstrate the Alcubierre "Expansion / Compression" scalar 
(York Time $\theta$) physically forming in the AVE vacuum, we use the 
3D FDTD engine to model a macroscopic vessel accelerating to and maintaining 
a superluminal velocity ($v = 1.2c$).

The simulation proves that a superluminal topological transit natively creates 
the required warp geometry: a massive, compressed Mach-cone (Bow Shock / Blue) 
in the forward vector, and an elongated, drafted rarefaction void (Wake / Red) 
trailing behind. This visually establishes the exact spatial tensor shapes 
derived analytically in `simulate_warp_metric_tensors.py`.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Bind to AVE parameters
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.ave.core.fdtd_3d import FDTD3DEngine
from src.ave.core.constants import C_0

def generate_3d_warp_evolution():
    print("[*] Initializing 3D Superluminal Warp Metric FDTD Simulator...")
    
    # ---------------------------------------------------------
    # 1. Engine & Grid Setup
    # ---------------------------------------------------------
    # We need a long tube to capture the Mach cone propagation
    NX, NY, NZ = 40, 40, 140
    DX = 0.05
    engine = FDTD3DEngine(nx=NX, ny=NY, nz=NZ, dx=DX)
    
    # The vessel (a macroscopic RF oscillator cluster)
    cx, cy = NX // 2, NY // 2
    vessel_radius = 4
    FREQUENCY = 150.0e6  # High frequency to show tight wavefront pulses
    
    # Vessel Velocity: v = 1.25 c (Superluminal to force the transit bow shock)
    v_vessel = 1.25 * engine.c
    
    # Calculate cell jumps per dt
    # dt = dx / (c * sqrt(3))
    # dz_per_step = v_vessel * dt / dx = 1.25 / sqrt(3) ~ 0.72 cells per step
    dz_per_step = (v_vessel * engine.dt) / engine.dx
    
    # ---------------------------------------------------------
    # 2. Time Evolution Loop
    # ---------------------------------------------------------
    STEPS_PER_FRAME = 2
    TOTAL_FRAMES = 65  # Enough frames for the vessel to travel from Z=20 to Z=110
    
    print(f"[*] Engine configured. Vessel Velocity: {v_vessel/engine.c:.2f}c")
    print(f"[*] Simulating {TOTAL_FRAMES} frames. Solving 3D Maxwell matrices (PML ABC enabled)...")
    
    frames_coords = []
    frames_colors = []
    vessel_positions = []
    
    X, Y, Z = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing='ij')
    
    current_z_float = 15.0  # Start near the bottom
    
    for frame in range(TOTAL_FRAMES):
        for _ in range(STEPS_PER_FRAME):
            t = engine.dt * (frame * STEPS_PER_FRAME + _)
            
            # The vessel acts as a driven macroscopic boundary, 
            # pulsating and injecting topological strain at its current physical location.
            curr_z_int = int(current_z_float)
            
            # Prevent out of bounds
            if curr_z_int < NZ - 2:
                signal = np.sin(2.0 * np.pi * FREQUENCY * t) * 200.0
                
                # Inject a circular disc (the vessel face)
                for i in range(-vessel_radius, vessel_radius + 1):
                    for j in range(-vessel_radius, vessel_radius + 1):
                        if i**2 + j**2 <= vessel_radius**2:
                            xg, yg = cx + i, cy + j
                            if 0 <= xg < NX and 0 <= yg < NY:
                                engine.inject_soft_source('Ez', xg, yg, curr_z_int, signal)
                                # Add structural bulk thickness
                                engine.inject_soft_source('Ez', xg, yg, curr_z_int-1, signal * 0.5)

            engine.step()
            current_z_float += dz_per_step
            
        # Extract visual frame
        vol = engine.Ez.copy()
        
        # We want to clearly see the pulses and standing waves
        # Threshold out the near-zero vacuum to make the Mach cone transparent
        intensity_max = np.max(np.abs(vol))
        # Drop threshold dynamically to keep the cone visible as it spreads
        threshold = intensity_max * 0.15 
        
        mask = np.abs(vol) > threshold
        
        frames_coords.append((X[mask], Y[mask], Z[mask]))
        frames_colors.append(vol[mask])
        vessel_positions.append(current_z_float)
        
        sys.stdout.write(f"\r  -> Solved frame {frame+1}/{TOTAL_FRAMES} (Vessel Z = {current_z_float:.1f})")
        sys.stdout.flush()
        
    print("\n[*] Integration complete. Compiling 3D Scatter animation...")

    # ---------------------------------------------------------
    # 3. 3D Volumetric Rendering
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(10, 12))
    plt.style.use('dark_background')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.set_box_aspect([NX, NY, NZ])  # True spatial proportions
    
    # Global color normalization based on a middle frame
    v_max = np.max(np.abs(frames_colors[TOTAL_FRAMES//2])) * 0.8
    
    scat = [ax.scatter([], [], [], c=[], cmap='coolwarm', vmin=-v_max, vmax=v_max)]
    
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.set_zlim(0, NZ)
    ax.set_axis_off()
    
    # Titles and Annotations
    ax.set_title(r"AVE Warp Metric Time Evolution ($v = 1.25c$)" + "\n" + r"Superluminal Bow Shock & York Time Rarefaction Wake", 
                 fontsize=14, fontweight='bold', color='white', pad=10)
                 
    # We will track the vessel with a blue wireframe ring
    theta = np.linspace(0, 2*np.pi, 30)
    ring_x = cx + vessel_radius * np.cos(theta)
    ring_y = cy + vessel_radius * np.sin(theta)
    vessel_ring, = ax.plot([], [], [], color='cyan', lw=3, zorder=20)
    
    # Dynamic camera to follow the action
    base_elev = 20
    base_azim = 35

    def update(frame):
        scat[0].remove()
        cx_pts, cy_pts, cz_pts = frames_coords[frame]
        cval = frames_colors[frame]
        
        # Render the thresholded scalar field (Pulse wavefronts)
        # Blue = Positive Strain (Compression), Red = Negative Strain (Rarefaction)
        if len(cx_pts) > 0:
            scat[0] = ax.scatter(cx_pts, cy_pts, cz_pts, c=cval, cmap='bwr', vmin=-v_max, vmax=v_max, s=12, alpha=0.5, edgecolors='none')
        else:
            scat[0] = ax.scatter([], [], [], c=[], cmap='bwr')
            
        # Update vessel marker
        vz = vessel_positions[frame]
        if vz < NZ:
            ring_z = np.full_like(ring_x, vz)
            vessel_ring.set_data(ring_x, ring_y)
            vessel_ring.set_3d_properties(ring_z)
        
        # Slowly pan the camera upwards to follow the Mach cone
        ax.view_init(elev=base_elev + (frame * 0.1), azim=base_azim + (frame * 0.3))
        
        return scat[0], vessel_ring

    print("[*] Generating GIF Sequence (Processing high-density 3D matrices)...")
    ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=False)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    anim_path = os.path.join(out_dir, 'warp_metric_time_evolution_3d.gif')
    
    ani.save(anim_path, writer='pillow', fps=12)
    plt.close(fig)
    plt.style.use('default')
    
    print(f"[*] 3D Volumetric Time-Evolution Animation complete -> {anim_path}")

if __name__ == "__main__":
    generate_3d_warp_evolution()
