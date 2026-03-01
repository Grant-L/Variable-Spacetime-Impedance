#!/usr/bin/env python3
r"""
AVE: Metric Streamlining CFD Heatmap (Schlieren Simulation)
===========================================================

This script models the discrete vacuum LC matrix explicitly as a 
macroscopic compressible fluid (an acoustic continuum) to meet the user's 
request for a CFD (Computational Fluid Dynamics) style visualization.

By numerically integrating the 2D scalar wave equation:
    \frac{\partial^2 \rho}{\partial t^2} = c^2 \nabla^2 \rho + S(x,t)
where \rho is the topological node density and c is 1/\sqrt{LC}.

We inject a macroscopic boundary ("The Vessel") moving at $v > c$ (Mach 1.5).
The resulting "Schlieren photography" style heatmap (magma colormap) tracks 
the physical fluid pressure dynamics, exposing the massive Cherenkov Mach-cone 
compressing the vacuum ahead (The Bow Shock) and the low-pressure draft 
filling the vacuum behind (The York Time Expansion Wake).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Bind to AVE Core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.ave.core.constants import C_0, EPSILON_0

# JAX GPU acceleration (graceful fallback to numpy)
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

def generate_cfd_heatmap():
    print("[*] Initializing Metric Streamlining 2D CFD Solver...")
    
    # Grid Resolution
    NX, NY = 400, 200
    TARGET_ASPECT = NX/NY
    dx = 0.02
    dy = dx
    
    # AVE Acoustic lattice speed of light
    c = float(C_0)
    
    # Courant stability limit for 2D scalar wave: dt <= dx / (c * sqrt(2))
    dt = dx / (c * np.sqrt(2.0)) * 0.99
    
    # Fluid scalar density matrices (past, present, future)
    rho_prev = np.zeros((NY, NX))
    rho_curr = np.zeros((NY, NX))
    rho_next = np.zeros((NY, NX))
    
    # Superluminal Vessel Parameters
    MACH_NUMBER = 1.5
    v_ship = MACH_NUMBER * c
    
    # Number of steps
    TOTAL_STEPS = 550
    # Steps per frame
    SPF = 4
    FRAMES = TOTAL_STEPS // SPF
    
    # Arrays to store the extracted Schlieren layers
    heatmap_frames = []
    ship_coords = []
    
    # Precompute damping layer for Absorbing Boundary Conditions (ABC)
    # This prevents the shockwave from violently bouncing off the rectangular grid walls
    damping = np.ones((NY, NX))
    DAMP_WIDTH = 20
    # Vectorized damping computation (replaces nested Python loop)
    j_idx = np.arange(NX)
    i_idx = np.arange(NY)
    dist_x = np.minimum(j_idx, NX - 1 - j_idx)  # shape: (NX,)
    dist_y = np.minimum(i_idx, NY - 1 - i_idx)  # shape: (NY,)
    min_dist = np.minimum(dist_y[:, np.newaxis], dist_x[np.newaxis, :])  # shape: (NY, NX)
    mask = min_dist < DAMP_WIDTH
    damping[mask] = (min_dist[mask] / DAMP_WIDTH)**2

    print(f"[*] Solving non-linear 2D LC density PDE up to {TOTAL_STEPS} dt. (Mach {MACH_NUMBER})")
    
    for step in range(TOTAL_STEPS):
        t = step * dt
        
        # Ship's current position
        ship_x = int((v_ship * t) / dx) + DAMP_WIDTH + 10  # Start slightly inside
        ship_y = NY // 2
        
        # 2D Finite Difference Laplacian (Central Difference)
        # d^2p/dt^2 = c^2 (d^2p/dx^2 + d^2p/dy^2)
        laplacian = (
            np.roll(rho_curr, 1, axis=1) + np.roll(rho_curr, -1, axis=1) +
            np.roll(rho_curr, 1, axis=0) + np.roll(rho_curr, -1, axis=0) -
            4 * rho_curr
        ) / (dx**2)
        
        # explicit time integration
        rho_next = 2 * rho_curr - rho_prev + (dt**2 * c**2) * laplacian
        
        # Apply strict Absorbing Boundary Damping
        rho_next *= damping
        
        # Ship Forcing Function (The Metric Streamlining Source)
        # Using a dipole equivalent: compression in front (positive density injection), 
        # rarefaction in back (negative density injection).
        # This matches the PONDER-01 asymmetric dielectric gradient.
        if 0 < ship_x < NX - 10:
            # We inject a soft volumetric pressure gradient
            R = 4
            for dy_p in range(-R, R+1):
                for dx_p in range(-R, R+1):
                    dist = np.sqrt(dx_p**2 + dy_p**2)
                    if dist <= R:
                        # Asymmetric gradient: Push front, Pull back
                        pressure_mag = 100.0 * (1.0 - dist/R)
                        # Front compression
                        rho_next[ship_y + dy_p, ship_x + dx_p + 1] += pressure_mag * dt
                        # Rear rarefaction
                        rho_next[ship_y + dy_p, ship_x + dx_p - 1] -= pressure_mag * dt
                        
        # Step time
        rho_prev[:, :] = rho_curr
        rho_curr[:, :] = rho_next
        
        if step % SPF == 0:
            # To get a "Schlieren" effect, we want the magnitude of the density gradient
            # This highlights the sharp shockwaves (the Mach cone) vividly against dark backgrounds
            grad_y, grad_x = np.gradient(rho_curr)
            schlieren_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            heatmap_frames.append(schlieren_mag.copy())
            ship_coords.append((ship_x, ship_y))
            
            sys.stdout.write(f"\r  -> Integrated PDE Step {step}/{TOTAL_STEPS}")
            sys.stdout.flush()

    print("\n[*] PDE Integration complete. Plotting high-contrast Schlieren CFD GIF...")

    fig, ax = plt.subplots(figsize=(12, 12 / TARGET_ASPECT))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0a0a2e')
    ax.set_facecolor('#0a0a2e')
    
    # Calculate a global max for normalization but clip extreme peaks to make the wake brighter
    global_max = np.nanmax(heatmap_frames[-1])
    vmax = max(float(global_max) * 0.4, 1e-6) if np.isfinite(global_max) else 1e-6
    
    # YlOrRd: white at 0, hot red at max — wake clearly visible on dark backgrounds
    im = ax.imshow(heatmap_frames[0], cmap='hot', origin='lower', vmin=0, vmax=vmax, extent=[0, NX, 0, NY])
    ax.set_title(rf"AVE Metric Streamlining CFD (Mach {MACH_NUMBER})" + "\n" + r"Schlieren Topology LC Density Gradient ($\nabla \rho_{LC}$)", color='white', pad=15, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Ship geometry (a sleek diamond/teardrop)
    ship_poly, = ax.plot([], [], color='white', lw=1.5, zorder=10)
    
    def update(frame):
        im.set_array(heatmap_frames[frame])
        sx, sy = ship_coords[frame]
        
        # Nose, Top, Tail, Bottom
        px = [sx + 6, sx, sx - 8, sx, sx + 6]
        py = [sy, sy + 3, sy, sy - 3, sy]
        if sx < NX:
            ship_poly.set_data(px, py)
        else:
            ship_poly.set_data([], [])
            
        return [im, ship_poly]

    print(f"[*] Compiling {FRAMES} animation frames...")
    ani = FuncAnimation(fig, update, frames=FRAMES, blit=True)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'warp_metric_cfd.gif')
    
    ani.save(out_path, writer='pillow', fps=15)
    plt.close()
    
    print(f"[*] Schlieren CFD Animation generated -> {out_path}")

if __name__ == "__main__":
    generate_cfd_heatmap()
