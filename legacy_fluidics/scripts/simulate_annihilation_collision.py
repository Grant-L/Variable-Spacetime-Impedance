"""
AVE MODULE: Matter-Antimatter Annihilation as Wave-Packet Collision
-------------------------------------------------------------------
This script uses a 2D scalar wave equation (representing the elastic
Cosserat vacuum) to model the literal mechanical shattering of two 
counter-rotating topological vortices (an Electron and a Positron).

Because they possess identical rotational inertia (mass) but exactly 
opposite angular momentum (spin = +omega vs -omega), their head-on 
collision geometrically nullifies their structural forms.

The massive rotational potential energy stored in their inertial cores 
is violently geometrically dumped into the ambient metric as an 
unspooling transverse shockwave (Gamma Ray Photons).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

global U, V, cont
U = None
V = None
cont = None

def simulate_annihilation_collision():
    global U, V, cont
    print("==========================================================")
    print(" AVE GRAND AUDIT: ANNIHILATION = MECHANICAL SHATTERING")
    print("==========================================================")
    
    # ---------------------------------------------------------
    # 1. HYDRODYNAMIC GRID SETUP
    # ---------------------------------------------------------
    N = 200
    L = 30.0
    dx = L / N
    X, Y = np.meshgrid(np.linspace(-L/2, L/2, N), np.linspace(-L/2, L/2, N))
    
    # Numerical stability
    dt = 0.05
    c2 = 1.0     # Baseline wave speed in the metric
    nu = 0.02    # Slight viscous damping for numerical stability
    
    # The Scalar Field U (Displacement) and its velocity V
    U = np.zeros((N, N))
    V = np.zeros((N, N))
    
    # ---------------------------------------------------------
    # 2. VORTEX INITIALIZATION (Electron + Positron)
    # ---------------------------------------------------------
    # We construct two localized Gaussian wave-packets carrying opposite angular momentum.
    # U is the static bump (mass), V is the rotational velocity (spin).
    
    r_core = 2.5
    amp = 3.0
    
    # Electron (Left, Spin UP: +omega)
    x1, y1 = -8.0, 0.0
    r1_sq = (X - x1)**2 + (Y - y1)**2
    U1 = amp * np.exp(-r1_sq / r_core**2)
    # Rotational velocity field (V = omega x r)
    Vx1 = -0.5 * (Y - y1) * U1
    Vy1 =  0.5 * (X - x1) * U1
    
    # Positron (Right, Spin DOWN: -omega)
    x2, y2 = 8.0, 0.0
    r2_sq = (X - x2)**2 + (Y - y2)**2
    U2 = amp * np.exp(-r2_sq / r_core**2)
    # Opposite rotational velocity
    Vx2 =  0.5 * (Y - y2) * U2
    Vy2 = -0.5 * (X - x2) * U2
    
    # Combine the structures into the scalar displacement field
    U = U1 + U2
    # For a scalar wave equation, we map the divergence of the rotational fields to V
    # to represent the internal tension of the spinning knot.
    # Div(V1 + V2) ~ The structural stress tensor.
    # Simplified here to purely contra-rotating initial kinetic energy kicks.
    V = (U1 - U2) * 1.5
    
    # Give them an initial linear momentum so they crash into each other
    # Electron moves right, Positron moves left
    momentum_kick = 0.8
    U_x_gradient = np.gradient(U, axis=1)
    
    # Velocity field is V: we add a +kx kick to the left mass, -kx to the right mass
    mask_left = X < 0
    mask_right = X > 0
    V[mask_left] -= momentum_kick * U_x_gradient[mask_left]
    V[mask_right] += momentum_kick * U_x_gradient[mask_right]

    # ---------------------------------------------------------
    # 3. VISUALIZATION AND SOLVER
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0B0F19')
    fig.suptitle("Applied Vacuum Engineering: Matter-Antimatter Annihilation\n(Mechanical Fusing of Contra-Rotating Vortices)", color='white', weight='bold', y=0.96)
    ax.set_facecolor('#0B0F19')
    ax.set_axis_off()
    
    img = ax.imshow(U, cmap='magma', extent=[-L/2, L/2, -L/2, L/2], origin='lower', vmin=-1.0, vmax=amp*1.5)
    
    # Define cont at the function scope so update can see it
    global cont
    cont = None
        
    time_text = ax.text(-14, 13, "", color='white', weight='bold')
    energy_text = ax.text(-14, 11, "", color='yellow', weight='bold')
    
    fig.text(0.5, 0.05, "The Electron ($+\\omega$) and Positron ($-\\omega$) mechanically cancel their structural chirality upon impact.\nThe localized rotational inertia (Mass) snaps, violently unspooling into transverse ripples (Gamma Ray Photons).", 
             color='lightgray', ha='center', fontsize=11, style='italic')

    # Simulation array
    frames = 200
    steps_per_frame = 4
    
    # Helper for the Laplacian (Curvature in the elastic metric)
    def laplacian(Z):
        Z_top = np.roll(Z, 1, axis=0)
        Z_bottom = np.roll(Z, -1, axis=0)
        Z_left = np.roll(Z, 1, axis=1)
        Z_right = np.roll(Z, -1, axis=1)
        
        # Hard bounce boundaries
        Z_top[0,:] = Z[0,:]
        Z_bottom[-1,:] = Z[-1,:]
        Z_left[:,0] = Z[:,0]
        Z_right[:,-1] = Z[:,-1]
        
        return (Z_top + Z_bottom + Z_left + Z_right - 4.0 * Z) / (dx**2)

    def update(frame):
        global U, V
            
        for _ in range(steps_per_frame):
            dU2 = c2 * laplacian(U) - 0.2 * U**3 - nu * V
            U_grad_x, U_grad_y = np.gradient(U, dx)
            V -= 0.05 * U_grad_x * np.sign(X) * dt 
            V += dU2 * dt
            U += V * dt
            
        img.set_data(U)
        img.set_data(U)
            
        # Calculate metric values
        total_structural_mass = np.sum(U[U > 1.0])
        total_kinetic_energy = np.sum(V**2)
        
        time_text.set_text(f"Time: {frame*dt*steps_per_frame:.2f} s")
        
        if total_structural_mass < 5.0 and frame > 50:
             energy_text.set_text(f"Structure: DESTROYED\nRadiating Shockwave Energy: High")
             energy_text.set_color('#FF3366')
        else:
             energy_text.set_text(f"Structural Mass Integrity: {total_structural_mass:.0f}\nInternal Kinematics: Bound")
             energy_text.set_color('yellow')

        # Static frame for LaTeX mid-explosion
        if frame == 110:
            OUTPUT_DIR = "assets/sim_outputs"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            static_out = os.path.join(OUTPUT_DIR, "annihilation_unspooling.png")
            fig.savefig(static_out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
            print(f"Saved static LaTeX plot: {static_out}")
             
        return img, time_text, energy_text

    print("Executing non-linear hydrodynamic structural shatter...")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=30, blit=False)
    
    out_path = "assets/sim_outputs/annihilation_collision.gif"
    try:
        ani.save(out_path, writer='pillow', fps=25, savefig_kwargs={'facecolor': fig.get_facecolor()})
        print(f"Saved Annihilation Animation: {out_path}")
    except Exception as e:
        print(f"Failed to generate GIF: {e}")

if __name__ == "__main__":
    simulate_annihilation_collision()
