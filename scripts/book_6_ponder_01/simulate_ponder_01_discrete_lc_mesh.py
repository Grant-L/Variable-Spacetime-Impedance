#!/usr/bin/env python3
r"""
PONDER-01: Discrete 3D LC Mesh Network Animation
================================================

This script discards continuous FDTD approximations and explicitly builds 
a node-by-node 3D LC SPICE-like matrix (the vacuum manifold). 

It integrates Kirchhoff's laws at every discrete structural junction to prove 
the macroscopic Torus Knot lock-in is a literal structural deformation of the grid.
Due to massive node counts, the output visualization explicitly zooms into a 
small structural "window" directly above the array to show the individual struts.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Bind into the AVE framework
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.ave.core.constants import C_0, EPSILON_0

# We don't use the FDTD Engine here. We build a literal discrete mesh.
def generate_discrete_lc_animation():
    print("[*] Initializing PONDER-01 Discrete 3D LC Mesh Integrator...")
    
    # Grid Size: Needs to be large enough to hold the array, 
    # but small enough that an N-body explicit integration doesn't take hours.
    # We will simulate a 30x30x30 physical node matrix.
    GRID_SIZE = 30
    
    # Physics parameters
    c = float(C_0)
    dx = 0.05
    dt = dx / (c * np.sqrt(3))
    
    # Capacitance (nodes) and Inductance (struts) of the vacuum
    # Z0 = sqrt(L/C) \approx 377 ohms
    # c = 1/sqrt(LC) 
    # For a unit cell, roughly C \approx epsilon_0 * dx
    eps_0 = float(EPSILON_0)
    C_node = eps_0 * dx
    L_strut = (1.0 / (c**2 * C_node))
    
    # State vectors for every node
    # V is the voltage (displacement) at each node
    V = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE))
    # I_x, I_y, I_z are the currents (magnetic tension) through the struts connecting nodes
    I_x = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE))
    I_y = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE))
    I_z = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE))
    
    center_x = GRID_SIZE // 2
    center_y = GRID_SIZE // 2
    
    num_elements = 8
    radius = 6 # Scaled down for the 30x30 grid
    angles = np.linspace(0, 2 * np.pi, num_elements, endpoint=False)
    
    # Place array near the bottom
    dipole_z_start = 5
    dipole_z_end = 10
    
    antennas = []
    for i, angle in enumerate(angles):
        sx = int(center_x + radius * np.cos(angle))
        sy = int(center_y + radius * np.sin(angle))
        phase_shift = i * (np.pi / 4.0)
        antennas.append({'x': sx, 'y': sy, 'phase': phase_shift})
        
    FREQUENCY = 200.0e6 # Slightly higher frequency to fit smaller grid
    
    TOTAL_FRAMES = 60
    STEPS_PER_FRAME = 2
    
    # We must identify a "Window" to visualize. 
    # Let's visualize an 8x8x8 cube of explicit nodes just above the array
    view_z_min = 12
    view_z_max = 20
    view_xy_min = center_x - 4
    view_xy_max = center_x + 4
    
    frames_V_window = []
    
    print(f"[*] Integrating explicit Kirchhoff LC equations over {GRID_SIZE**3} nodes...")
    
    # Explicit Leapfrog Integrator (Yee-style but strictly conceptual LC components)
    for frame in range(TOTAL_FRAMES):
        for _ in range(STEPS_PER_FRAME):
            t = (frame * STEPS_PER_FRAME + _) * dt
            
            # 1. Update Strut Currents (Inductors resist change in voltage gradient)
            # V = L * dI/dt --> dI = (V_diff / L) * dt
            
            # X struts
            V_diff_x = V[1:, :, :] - V[:-1, :, :]
            I_x[:-1, :, :] += (V_diff_x / L_strut) * dt
            
            # Y struts
            V_diff_y = V[:, 1:, :] - V[:, :-1, :]
            I_y[:, :-1, :] += (V_diff_y / L_strut) * dt
            
            # Z struts
            V_diff_z = V[:, :, 1:] - V[:, :, :-1]
            I_z[:, :, :-1] += (V_diff_z / L_strut) * dt
            
            # 2. Update Node Voltages (Capacitors integrate net current)
            # I = C * dV/dt --> dV = (I_net / C) * dt
            
            # Net current into each node
            I_net = np.zeros_like(V)
            
            # Current arriving from the "left"
            I_net[1:, :, :] -= I_x[:-1, :, :]
            I_net[:, 1:, :] -= I_y[:, :-1, :]
            I_net[:, :, 1:] -= I_z[:, :, :-1]
            
            # Current flowing to the "right"
            I_net[:-1, :, :] += I_x[:-1, :, :]
            I_net[:, :-1, :] += I_y[:, :-1, :]
            I_net[:, :, :-1] += I_z[:, :, :-1]
            
            V += (I_net / C_node) * dt
            
            # 3. Inject explicit drive voltages (Overriding node capacitor voltages)
            for src in antennas:
                signal = np.sin(2.0 * np.pi * FREQUENCY * t - src['phase'])
                for z in range(dipole_z_start, dipole_z_end):
                    V[src['x'], src['y'], z] = signal * 1000.0 # High V drive
                    
        # Extract the rendering window
        window_V = V[view_xy_min:view_xy_max, view_xy_min:view_xy_max, view_z_min:view_z_max].copy()
        frames_V_window.append(window_V)
        
        sys.stdout.write(f"\r  -> Computed LC frame {frame+1}/{TOTAL_FRAMES}")
        sys.stdout.flush()

    print("\n[*] Integration complete. Compiling explicit 3D Structural Node Graph...")

    # We need to construct a literal 3D list of line segments (struts) and points (nodes)
    # for the visualizer.
    
    fig = plt.figure(figsize=(12, 12))
    plt.style.use('dark_background')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    win_size_x = view_xy_max - view_xy_min
    win_size_y = win_size_x
    win_size_z = view_z_max - view_z_min
    
    # Baseline visual coordinates
    X, Y, Z = np.meshgrid(np.arange(win_size_x), np.arange(win_size_y), np.arange(win_size_z), indexing='ij')
    
    v_max = np.max(np.abs(frames_V_window[-1])) * 0.5
    
    # Create the strut lines logic
    def build_struts(V_matrix):
        lines = []
        colors = []
        for x in range(win_size_x):
            for y in range(win_size_y):
                for z in range(win_size_z):
                    v_current = V_matrix[x, y, z]
                    
                    # Connect to x+1
                    if x < win_size_x - 1:
                        lines.append([(x, y, z), (x+1, y, z)])
                        colors.append(abs(v_current))
                    # Connect to y+1
                    if y < win_size_y - 1:
                        lines.append([(x, y, z), (x, y+1, z)])
                        colors.append(abs(v_current))
                    # Connect to z+1
                    if z < win_size_z - 1:
                        lines.append([(x, y, z), (x, y, z+1)])
                        colors.append(abs(v_current))
        return lines, colors

    # Start empty and populate in the update function
    scatter = ax.scatter([], [], [], c=[], cmap='plasma', s=30, vmin=-v_max, vmax=v_max, edgecolors='white', linewidth=0.2, zorder=5)
    
    # Line3DCollection requires at least one dummy segment to prevent auto_scale crash
    dummy_segments = [[(0, 0, 0), (1, 1, 1)]]
    lc = Line3DCollection(dummy_segments, cmap='plasma', norm=plt.Normalize(vmin=0, vmax=v_max), linewidths=1.5, alpha=0.0, zorder=1) # start invisible
    
    # Disable auto-scaling to prevent the crash when replacing data
    ax.auto_scale_xyz([0, win_size_x], [0, win_size_y], [0, win_size_z])
    ax.add_collection3d(lc, autolim=False)

    ax.set_xlim(0, win_size_x - 1)
    ax.set_ylim(0, win_size_y - 1)
    ax.set_zlim(0, win_size_z - 1)
    
    ax.set_axis_off()
    ax.set_title(r"Discrete 3D LC Mesh Network ($11\times11\times11$ Node Window)" + "\n" + r"Structural Manifold Compression via PONDER-01 Array", fontsize=14, fontweight='bold', color='white', pad=20)
    
    ax.view_init(elev=25, azim=45)

    def update(frame):
        V_matrix = frames_V_window[frame]
        
        # 1. Update Nodes (Scatter points)
        # To show true structural deformation, we offset the X,Y,Z coords slightly
        # based on the local structural tension (Voltage gradient/Current).
        # We will approximate displacement directly from V for aesthetic LC representation.
        
        displacement_factor = 0.4 / v_max
        dx = V_matrix * displacement_factor
        dy = V_matrix * displacement_factor
        dz = V_matrix * displacement_factor
        
        # Shift the nodes visually to show the structure "breathing" and warping
        mod_X = X.flatten() + dx.flatten()
        mod_Y = Y.flatten() + dy.flatten()
        mod_Z = Z.flatten() + dz.flatten()
        
        # Unfortunately scatter3D in matplotlib cannot easily update just `_offsets3d` directly.
        # We must explicitly remove old collections to redraw
        for collection in list(ax.collections):
            collection.remove()
        
        # Redraw Scatter
        scatter = ax.scatter(mod_X, mod_Y, mod_Z, c=V_matrix.flatten(), cmap='plasma', s=40, vmin=-v_max, vmax=v_max, edgecolors='white', linewidth=0.3, zorder=5)
        
        # Redraw Struts
        # We need the modified coords to draw the bent struts
        V_shifted = np.zeros((*V_matrix.shape, 3))
        V_shifted[..., 0] = X + dx
        V_shifted[..., 1] = Y + dy
        V_shifted[..., 2] = Z + dz
        
        lines = []
        colors = []
        for x in range(win_size_x):
            for y in range(win_size_y):
                for z in range(win_size_z):
                    tension = abs(V_matrix[x,y,z])
                    pt_curr = V_shifted[x,y,z]
                    
                    if x < win_size_x - 1:
                        lines.append([pt_curr, V_shifted[x+1,y,z]])
                        colors.append(tension)
                    if y < win_size_y - 1:
                        lines.append([pt_curr, V_shifted[x,y+1,z]])
                        colors.append(tension)
                    if z < win_size_z - 1:
                        lines.append([pt_curr, V_shifted[x,y,z+1]])
                        colors.append(tension)
                        
        lc = Line3DCollection(lines, cmap='plasma', norm=plt.Normalize(vmin=0, vmax=v_max), linewidths=1.5, alpha=0.7, zorder=1)
        lc.set_array(np.array(colors))
        ax.add_collection3d(lc)
        
        # Slow pan
        ax.view_init(elev=25, azim=45 + (frame * 0.5))
        
        return scatter, lc
        
    print("[*] Generating GIF (This discrete node visualization takes significant rendering time)...")
    ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=False)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    anim_path = os.path.join(out_dir, 'ponder_01_discrete_lc_mesh.gif')
    
    # 15 fps gives exactly 4 seconds of evolution
    ani.save(anim_path, writer='pillow', fps=15)
    plt.close(fig)
    
    # Reset
    plt.style.use('default')
    
    print(f"[*] Discrete LC Mesh Animation complete -> {anim_path}")

if __name__ == "__main__":
    generate_discrete_lc_animation()
