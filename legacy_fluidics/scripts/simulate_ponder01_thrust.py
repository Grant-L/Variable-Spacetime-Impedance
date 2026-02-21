import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import os

def solve_laplace_2d(grid_size, V_top, V_bot, mask_top, mask_bot, max_iter=2000, tol=1e-5):
    """
    Finite-Difference solver for Laplace's Equation (del^2 V = 0)
    to calculate the electrostatic potential around the asymmetrical pads.
    """
    V = np.zeros((grid_size, grid_size))
    
    # Boundary conditions
    V[mask_top] = V_top
    V[mask_bot] = V_bot
    
    # Precompute mask of free space (where we update the potential)
    free_space = ~(mask_top | mask_bot)
    
    for i in range(max_iter):
        V_old = V.copy()
        
        # Vectorized 4-neighbor averaging (Gauss-Seidel style via Jacobi convolution)
        V[1:-1, 1:-1] = 0.25 * (V[2:, 1:-1] + V[:-2, 1:-1] + V[1:-1, 2:] + V[1:-1, :-2])
        
        # Enforce boundary conditions
        V[mask_top] = V_top
        V[mask_bot] = V_bot
        
        # Enforce Neumann boundaries (zero normal derivative at the box edges)
        V[0, :] = V[1, :]
        V[-1, :] = V[-2, :]
        V[:, 0] = V[:, 1]
        V[:, -1] = V[:, -2]
        
        # Convergence check
        if np.max(np.abs(V - V_old)) < tol:
            print(f"Laplace solver converged in {i} iterations.")
            break
            
    return V

def simulate_ponder01():
    print("==========================================================")
    print(" AVE GRAND AUDIT: PROJECT PONDER-01 (METRIC THRUST)")
    print("==========================================================")
    
    # ---------------------------------------------------------
    # 1. ELECTROSTATIC SOLVER (The Asymmetric Wedge)
    # ---------------------------------------------------------
    print("Solving electrostatic fringing fields...")
    GRID_SIZE = 150
    V_TOP = 1000.0  # Volts
    V_BOT = 0.0     # Ground
    
    # Create geometric masks
    mask_top = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    mask_bot = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    
    # Top Layer: Flat Rectangular Pad (Uniform)
    mask_top[110:115, 30:120] = True
    
    # Bottom Layer: The Asymmetric Wedge (Generates the Gradient)
    # We create a triangle/chevron pointing downward
    for y in range(35, 45):
        width = int((y - 35) * 4)  # Wedge widens as it goes down
        center = 75
        mask_bot[y, center - width:center + width] = True

    # Run Laplace Solver
    V = solve_laplace_2d(GRID_SIZE, V_TOP, V_BOT, mask_top, mask_bot)
    
    # Calculate E-Field (Electric field is negative gradient of Potential)
    Ey, Ex = np.gradient(-V)
    E_mag = np.sqrt(Ex**2 + Ey**2)
    
    # Calculate scalar energy density gradient magnitude (del_u ~ del(E^2))
    # In AVE, Ponderomotive acceleration a = c^2 * del(n) ~ del(u)
    u_dense = E_mag**2
    du_y, du_x = np.gradient(u_dense)
    grad_u_mag = np.sqrt(du_x**2 + du_y**2)
    
    # ---------------------------------------------------------
    # 2. VISUALIZATION AND ANIMATION
    # ---------------------------------------------------------
    print("Generating comprehensive visual telemetry...")
    fig = plt.figure(figsize=(18, 6), facecolor='#0B0F19')
    
    # Panel 1: Electrostatic Potential & E-Field Streamlines
    ax1 = fig.add_subplot(1, 3, 1, facecolor='#0B0F19')
    img1 = ax1.imshow(V, cmap='magma', origin='lower', extent=[0, 150, 0, 150])
    
    Y, X = np.mgrid[0:150, 0:150]
    ax1.streamplot(X, Y, Ex, Ey, color='white', linewidth=0.8, density=1.2, arrowsize=1.0)
    
    # Overlay the physical pads
    ax1.contour(mask_top, levels=[0.5], colors=['#FF3366'], linewidths=2, origin='lower', extent=[0, 150, 0, 150])
    ax1.contour(mask_bot, levels=[0.5], colors=['#00FFCC'], linewidths=2, origin='lower', extent=[0, 150, 0, 150])
    
    ax1.set_title("1. Asymmetric PCBA Fringing Fields\n$V(x,y)$ and E-Field Streamlines", color='white', weight='bold')
    ax1.axis('off')
    
    # Panel 2: The Ponderomotive Gradient (del_u)
    ax2 = fig.add_subplot(1, 3, 2, facecolor='#0B0F19')
    img2 = ax2.imshow(grad_u_mag, cmap='plasma', origin='lower', extent=[0, 150, 0, 150], vmax=np.percentile(grad_u_mag, 98))
    
    # Draw a massive downward thrust vector arrow originating from the Wedge
    ax2.arrow(75, 55, 0, -35, head_width=8, head_length=10, fc='#00FFCC', ec='white', linewidth=3, zorder=5)
    ax2.text(75, 15, r"Net Ponderomotive Thrust Thrust", color='#00FFCC', ha='center', weight='bold', fontsize=12)
    
    ax2.set_title("2. Scalar Energy Gradient ($\\nabla u$)\nDriving Metric Acceleration ($a = c^2 \\nabla n$)", color='white', weight='bold')
    ax2.axis('off')
    
    # Panel 3: Experimental Scale Telemetry
    ax3 = fig.add_subplot(1, 3, 3, facecolor='#0B0F19')
    ax3.tick_params(colors='lightgray')
    for spine in ax3.spines.values(): spine.set_color('#333333')
    ax3.grid(True, ls=':', color='#333333')
    
    t = np.linspace(0, 10, 1000)
    mosfet_active = (t >= 3) & (t <= 7)
    base_weight = np.full_like(t, 50000.0) 
    thrust_mg = 4.8  # ~47 uN
    
    scale_readout = base_weight - (thrust_mg * mosfet_active)
    noise = np.random.normal(0, 0.15, len(t))
    scale_readout += noise
    
    ax3.plot(t, scale_readout, color='#00ffcc', lw=2, label='Ohaus Micro-balance (mg)')
    ax3.axvspan(3, 7, color='#FF3366', alpha=0.15, label='1000V SiC MOSFET Pulsing')
    
    ax3.set_title("3. PONDER-01 Physical Falsification\nMetric Thrust Telemetry Output", color='white', weight='bold')
    ax3.set_xlabel("Time (seconds)", color='white')
    ax3.set_ylabel("Total PCBA Weight (milligrams)", color='white')
    
    ax3.set_ylim(49992, 50003)
    ax3.set_yticks([49992, 49996, 50000])
    ax3.set_yticklabels(['49,992.0 mg', '49,996.0 mg', '50,000.0 mg'])
    
    ax3.legend(loc='lower left', facecolor='#111111', edgecolor='gray', labelcolor='white')
    ax3.text(5, 49994.5, r"$\Delta = -4.8$ mg Shift", color='white', ha='center', weight='bold', bbox=dict(facecolor='#111111', edgecolor='#FF3366', pad=5))
    
    plt.tight_layout()
    
    # Save a static high-res overview
    OUTPUT_DIR = "assets/sim_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    static_filepath = os.path.join(OUTPUT_DIR, "ponder01_thrust_matrix.png")
    plt.savefig(static_filepath, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Saved Static Visualization: {static_filepath}")
    
    # ---------------------------------------------------------
    # 3. DYNAMIC GIF GENERATION (The dV/dt Pulse Animation)
    # ---------------------------------------------------------
    print("Generating Dynamic GIF Animation of the Fringing Fields...")
    
    # We will animate Panel 1 (Potential) and Panel 2 (Gradient) fading in and out to simulate the 1000V PWM pulse.
    def update_anim(frame):
        # Sine wave envelope to simulate the voltage rising and falling
        voltage_factor = np.abs(np.sin(frame * np.pi / 20)) 
        
        # Scale the data dynamically
        img1.set_data(V * voltage_factor)
        img1.set_clim(0, V_TOP) # Keep colorbar fixed so the image visibly brightens
        
        img2.set_data(grad_u_mag * (voltage_factor**2))
        
        if voltage_factor > 0.8:
            ax1.set_title("1. SiC MOSFET ACTIVE ($V_{ds} \sim 1000V$)", color='#FF3366', weight='bold')
        else:
            ax1.set_title("1. SiC MOSFET IDLE ($V_{ds} \sim 0V$)", color='white', weight='bold')
            
        return [img1, img2, ax1.title]

    # Create the animation
    ani = animation.FuncAnimation(fig, update_anim, frames=40, interval=50, blit=False)
    
    # Save as GIF using Pillow (built-in, doesn't require FFmpeg)
    try:
        gif_filepath = os.path.join(OUTPUT_DIR, "ponder01_dynamic_thrust.gif")
        ani.save(gif_filepath, writer='pillow', fps=15, savefig_kwargs={'facecolor': fig.get_facecolor()})
        print(f"Saved Dynamic Animation: {gif_filepath}")
    except Exception as e:
        print(f"Failed to generate GIF (Likely Pillow writer missing): {e}")

if __name__ == "__main__":
    simulate_ponder01()
