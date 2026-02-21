import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import os

def run_simulation(N=1000, max_steps=2000, save_path="lc_gravity_refraction.gif"):
    # 1D FDTD Simulation of an LC Transmission Line
    # Represents light/matter surfing through the variable impedance vacuum
    
    # Grid parameters
    dx = 1.0
    dt = 0.5 # Courant condition (dt < dx/c)
    
    # Baseline Vacuum LC Parameters
    C0 = 1.0  # Vacuum Permittivity equivalent
    L0 = 1.0  # Vacuum Permeability equivalent
    c_0 = 1.0 / np.sqrt(L0 * C0)
    
    # Create the Spacetime Impedance Gradient (A "Mass")
    # Refractive index n(x) increases near the center
    x = np.linspace(0, N, N)
    center = N // 2
    width = N // 10
    
    # n(x) = 1 + Potential well
    # We square the well to make it a deep sink
    well_depth = 3.0
    n_x = 1.0 + well_depth * np.exp(-((x - center)**2) / (width**2))
    
    # Spatially variable L and C
    C = C0 * n_x
    L = L0 * n_x
    
    # Local Speed of Light
    c_local = 1.0 / np.sqrt(L * C)
    
    # Initialize Voltage (V) and Current (I) Arrays
    V = np.zeros(N)
    I = np.zeros(N)
    
    # Data storage for animation
    V_history = []
    
    # Inject a Gaussian wave packet (Photon / Matter Wave)
    wave_center = N // 10
    wave_width = N // 40
    # In order to launch a right-traveling wave smoothly, we could force the boundary,
    # or just initialize the V and I fields to represent a moving Gaussian.
    # V(x,0) = f(x), I(x,0) = f(x)/Z0
    V = np.exp(-((x - wave_center)**2) / (wave_width**2))
    I = V / np.sqrt(L0 / C0) # Initially in pure vacuum matching Z0
    
    for step in range(max_steps):
        # Update Current (I) using Voltage gradient
        # L * dI/dt = - dV/dx
        I[:-1] = I[:-1] - (dt / L[:-1]) * (V[1:] - V[:-1])
        
        # Inject boundary condition to keep the wave clean, or just let it go
        # Update Voltage (V) using Current gradient
        # C * dV/dt = - dI/dx
        V[1:] = V[1:] - (dt / C[1:]) * (I[1:] - I[:-1])
        
        # Store for animation every 10 steps
        if step % 10 == 0:
            V_history.append(V.copy())

    # --- Plotting & Animation ---
    print(f"Generating animation ({len(V_history)} frames)...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top plot: Wave propagation
    line_v, = ax1.plot(x, V_history[0], color='cyan', lw=2)
    ax1.set_xlim(0, N)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_title("Electromagnetic Refraction in an LC Gravitational Well", fontsize=14, color='white')
    ax1.set_ylabel("Wave Amplitude (V)", color='white')
    ax1.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')
    ax1.tick_params(colors='white')
    ax1.grid(color='#333333', linestyle='--', alpha=0.5)
    
    # Bottom plot: Spacetime Impedance
    ax2.plot(x, c_local, color='red', lw=2)
    ax2.fill_between(x, c_local, 1.0, color='red', alpha=0.3)
    ax2.set_xlim(0, N)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Local $c = 1/\sqrt{LC}$", color='white')
    ax2.set_xlabel("Spatial LC Nodes", color='white')
    ax2.set_facecolor('#111111')
    ax2.tick_params(colors='white')
    ax2.grid(color='#333333', linestyle='--', alpha=0.5)
    
    def update(frame):
        line_v.set_ydata(V_history[frame])
        return line_v,
    
    ani = animation.FuncAnimation(fig, update, frames=len(V_history), interval=30, blit=True)
    
    ani.save(save_path, writer='pillow', fps=30)
    print(f"Animation saved to {save_path}")

if __name__ == "__main__":
    out_dir = "/Users/grantlindblom/.gemini/antigravity/brain/52427d99-ccfc-4262-a754-f488d18e15bb"
    out_path = os.path.join(out_dir, "gravity_lc_refraction.gif")
    run_simulation(N=1000, max_steps=1800, save_path=out_path)
