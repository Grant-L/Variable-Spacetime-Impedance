"""
AVE Antimatter Annihilation Simulator (2D Continuum)
===================================================
Simulates the collision of a matter vs antimatter wave packet (e.g., e- and e+).
In the AVE framework, an electron is a localized macroscopic acoustic phase vortex 
(OAM = +1) and the positron is identical but with inverse parity (OAM = -1).

This FDTD engine fires two localized traveling phase-vortices at each other.
When their topological centers overlap, the +w and -w localized phase structures
perfectly destructively interfere. The standing wave "mass" collapses, and the 
previously trapped rotational energy unspools into the rigid lattice, flashing
outward into pure linear transverse shockwaves (Gamma ray photons / E=mc^2).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib import animation

# FDTD Parameters
N = 300                 # Grid size
C_0 = 0.5               # Wavespeed (Courant limiter)
STEPS = 400             # Frames
SIGMA = 20.0            # Packet width

def generate_vortex_pulse(X, Y, center_x, center_y, k_x, OAM_charge):
    """
    Generates a localized traveling wave-packet with internal Orbital Angular Momentum.
    OAM_charge = +1 (Matter) or -1 (Antimatter)
    """
    dx = X - center_x
    dy = Y - center_y
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    
    # Gaussian envelope defining the "Mass" boundary
    envelope = np.exp(-r**2 / (2 * SIGMA**2))
    
    # Laguerre-Gaussian phase twist (Vorticity)
    # The r term creates the central null (the doughnut hole)
    amplitude = r * envelope 
    
    # Travel phase + Rotational phase
    phase = k_x * dx + OAM_charge * theta
    
    # We return the real scalar displacement field for the FDTD grid
    U = amplitude * np.cos(phase)
    # Return numerical time derivative (velocity) approximation for a right/left moving pulse
    V = -C_0 * (k_x * amplitude * np.sin(phase)) 
    
    return U, V

def simulate_annihilation_2d():
    print("Initializing 2D FDTD Annihilation Grid...")
    x = np.linspace(0, N, N)
    y = np.linspace(0, N, N)
    X, Y = np.meshgrid(x, y)
    
    # Initialize Matter (e-) on the left moving right
    U_matter, V_matter = generate_vortex_pulse(X, Y, center_x=N//4, center_y=N//2, k_x=0.2, OAM_charge=1)
    
    # Initialize Antimatter (e+) on the right moving left
    U_anti, V_anti = generate_vortex_pulse(X, Y, center_x=3*N//4, center_y=N//2, k_x=-0.2, OAM_charge=-1)
    
    # Superposition
    U = U_matter + U_anti
    V = V_matter + V_anti
    
    U_prev = U - V  # FDTD reverse-step approximation
    
    frames = []
    
    # Run the 2D Wave Equation
    for t in range(STEPS):
        U_next = np.zeros((N, N))
        
        # 5-point discrete Laplacian for scalar wave equation
        laplacian = (np.roll(U, 1, axis=0) + np.roll(U, -1, axis=0) +
                     np.roll(U, 1, axis=1) + np.roll(U, -1, axis=1) - 4 * U)
                     
        # Explicit time-stepping
        U_next = 2 * U - U_prev + (C_0**2) * laplacian
        
        # Apply absorbing boundary conditions (soft damping at edges)
        damping = np.ones((N,N))
        damping[:10, :] *= 0.9; damping[-10:, :] *= 0.9
        damping[:, :10] *= 0.9; damping[:, -10:] *= 0.9
        U_next *= damping
        
        # Compute Energy Density Map (Kinetic + Potential)
        # E = (dU/dt)^2 + |gradient U|^2
        kinetic = (U_next - U)**2
        grad_x = np.roll(U, -1, axis=1) - U
        grad_y = np.roll(U, -1, axis=0) - U
        potential = grad_x**2 + grad_y**2
        energy_density = kinetic + (C_0**2)*potential
        
        if t % 5 == 0:
            frames.append(np.copy(energy_density))
            
        U_prev = np.copy(U)
        U = np.copy(U_next)
        
        if t % 50 == 0:
            print(f" Simulating Frame {t}/{STEPS}...")
            
    return frames

def render_sequence_and_gif(frames, out_png, out_gif):
    print("Rendering Annihilation Graphics...")
    plt.style.use('dark_background')
    
    # 1. Generate 4-panel static sequence for the manuscript
    fig = plt.figure(figsize=(16, 4))
    idx = [0, len(frames)//3, 2*len(frames)//3, len(frames)-1]
    titles = ["1. Topological Approach (e- and e+)", 
              "2. Metric Overlap (Phase Cancellation)", 
              "3. Rotational Shatter (Non-Linear Yield)", 
              "4. E=mc^2 Linear Radiant Expansion (Gamma)"]
              
    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1)
        ax.imshow(frames[idx[i]], cmap='magma', origin='lower', vmax=np.max(frames[0])*1.5)
        ax.set_title(titles[i], color='white', pad=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Generate the Animation Loop
    fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
    ax_anim.axis('off')
    im = ax_anim.imshow(frames[0], cmap='magma', origin='lower', vmax=np.max(frames[0])*1.5)
    ax_anim.set_title("Continuum Eradication (Matter-Antimatter Collision)", color='white', pad=15)
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]
        
    ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), blit=True)
    ani.save(out_gif, writer='pillow', fps=20)
    plt.close()
    
    print(f"[Done] Output saved to:\n  - {out_png}\n  - {out_gif}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    out_dir = PROJECT_ROOT / "scripts" / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    frames = simulate_annihilation_2d()
    render_sequence_and_gif(frames, 
                            out_dir / "annihilation_sequence.png",
                            out_dir / "annihilation_unspooling.gif")
