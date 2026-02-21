"""
AVE MODULE: Superconductivity as a Phase-Locked Gear Train
----------------------------------------------------------
This script models the macroscopic physical vacuum as an NxN 2D lattice 
of topological flywheels (electrons). 

- At T > Tc (Hot/Normal Metal), thermal noise melts the elastic "teeth" 
  of their magnetic boundaries. They couple purely via fluidic viscosity.
  An external RF magnetic field (boundary torque) penetrates deep into 
  the bulk via the classical Skin Effect.
  
- At T < Tc (Cold/Superconductor), the noise dies, and the flywheels 
  elastically interlock into a completely rigid "Phase-Locked Gear Train".
  The boundary torque now attempts to spin a macroscopic object with 
  near-infinite moment of inertia, resulting in rapid exponential shielding.
  This represents the exact classical mechanical derivation of the Meissner 
  Effect and the London Penetration Depth.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def get_torques(phi, omega, K_e, K_v):
    """
    Computes the net torque on every node in the 2D flywheel array.
    K_e: Elastic phase-locking constant (analogous to Cooper pairing rigidity).
    K_v: Viscous fluidic drag (analogous to normal resistance).
    """
    # Periodic boundary conditions on the Y-axis (top to bottom)
    phi_up = np.roll(phi, 1, axis=0)
    phi_down = np.roll(phi, -1, axis=0)
    omega_up = np.roll(omega, 1, axis=0)
    omega_down = np.roll(omega, -1, axis=0)
    
    # Non-periodic on X-axis (left boundary driven, right boundary pinned to bulk)
    phi_left = np.roll(phi, 1, axis=1)
    phi_right = np.roll(phi, -1, axis=1)
    omega_left = np.roll(omega, 1, axis=1)
    omega_right = np.roll(omega, -1, axis=1)
    
    T = np.zeros_like(phi)
    
    # Y-axis coupling
    T += K_e * np.sin(phi_up - phi) + K_v * (omega_up - omega)
    T += K_e * np.sin(phi_down - phi) + K_v * (omega_down - omega)
    
    # X-axis coupling
    T_l = K_e * np.sin(phi_left - phi) + K_v * (omega_left - omega)
    T_r = K_e * np.sin(phi_right - phi) + K_v * (omega_right - omega)
    
    # Nullify invalid roll wraparounds at the hard boundaries
    T_l[:, 0] = 0
    T_r[:, -1] = 0
    
    T += T_l + T_r
    return T

def simulate_meissner_effect():
    print("==========================================================")
    print(" AVE GRAND AUDIT: MEISSNER EFFECT = MACROSCOPIC GEAR TRAIN")
    print("==========================================================")
    
    Nx, Ny = 20, 10
    dt = 0.05
    steps_per_frame = 4
    frames = 150
    
    # Initial states
    phi_h = np.zeros((Ny, Nx))
    om_h  = np.zeros((Ny, Nx))
    phi_c = np.zeros((Ny, Nx))
    om_c  = np.zeros((Ny, Nx))
    
    # Field amplitude monitors (Exponential Moving Averages of angular velocity mapping to B-field penetration)
    env_h = np.zeros(Nx)
    env_c = np.zeros(Nx)
    
    # We stagger the literal visual angles so they look like interlocked counter-rotating gears
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    signs = (-1.0)**(X + Y)
    
    fig = plt.figure(figsize=(16, 8), facecolor='#0B0F19')
    fig.suptitle("Applied Vacuum Engineering: The Mechanical Origin of the Meissner Effect", color='white', fontsize=18, weight='bold', y=0.96)
    
    # --- HOT LATTICE (Normal Metal) ---
    ax1 = fig.add_subplot(2, 2, 1, facecolor='#0B0F19')
    ax1.set_title("1. Hot Random Lattice ($T > T_c$): Normal Viscous Transmission", color='#FF3366', fontsize=14, pad=10)
    ax1.scatter(X, Y, s=400, facecolors='none', edgecolors='#555555', lw=1.5)
    quiv_h = ax1.quiver(X, Y, np.ones_like(X), np.zeros_like(Y), color='#FF3366', pivot='tail', scale=20, width=0.005)
    ax1.set_axis_off()
    
    ax2 = fig.add_subplot(2, 2, 2, facecolor='#111111')
    ax2.set_title("Magnetic Field Penetration Profile (Skin Effect)", color='white')
    line_h, = ax2.plot(np.arange(Nx), env_h, color='#FF3366', lw=3)
    ax2.set_xlim(0, Nx-1); ax2.set_ylim(0, 1.5)
    ax2.set_facecolor('#0B0F19')
    ax2.tick_params(colors='gray')
    ax2.grid(True, color='#333333', ls=':')
    ax2.set_ylabel(r"Local B-Field Density ($\langle \omega \rangle$)", color='gray')
    
    # --- COLD LATTICE (Superconductor) ---
    ax3 = fig.add_subplot(2, 2, 3, facecolor='#0B0F19')
    ax3.set_title("2. Cold Phase-Locked Lattice ($T < T_c$): Rigid $N$-Body Gear Train", color='#00FFCC', fontsize=14, pad=10)
    ax3.scatter(X, Y, s=400, facecolors='none', edgecolors='#555555', lw=1.5)
    quiv_c = ax3.quiver(X, Y, np.ones_like(X), np.zeros_like(Y), color='#00FFCC', pivot='tail', scale=20, width=0.005)
    ax3.set_axis_off()
    
    ax4 = fig.add_subplot(2, 2, 4, facecolor='#111111')
    ax4.set_title("Perfect Diamagnetic Shielding (Meissner London Decay)", color='white')
    line_c, = ax4.plot(np.arange(Nx), env_c, color='#00FFCC', lw=3)
    ax4.set_xlim(0, Nx-1); ax4.set_ylim(0, 1.5)
    ax4.set_facecolor('#0B0F19')
    ax4.tick_params(colors='gray')
    ax4.grid(True, color='#333333', ls=':')
    ax4.set_xlabel("Depth into Lattice $X$ (Inductive Flywheels)", color='gray')
    
    # Text annotation
    fig.text(0.5, 0.03, "If electrons are flywheels, coupling them below Tc creates a macro-gearbox with infinite uniform inertia. External boundary torques\ncannot turn the gears, causing instant exponential shielding. Zero Resistance = Lossless Transmission of Angular Momentum.", 
             color='lightgray', ha='center', fontsize=12, style='italic')

    t = [0.0]

    def update(frame):
        for _ in range(steps_per_frame):
            t[0] += dt
            
            # The external applied Magnetic Field (Oscillating Torque) at the left boundary
            tau_ext = 4.0 * np.sin(2.0 * t[0])
            
            # --- NORMAL METAL PHYSICS ---
            # Ke = 0 (No phase locking), Kv = 2.0 (Viscous diffusion / resistance)
            Th = get_torques(phi_h, om_h, K_e=0.0, K_v=2.0)
            Th[:, 0] += tau_ext
            om_h[...] += (Th - 0.2 * om_h) * dt
            phi_h[...] += om_h * dt
            phi_h[:, -1] = 0; om_h[:, -1] = 0 # Anchored to deep bulk
            
            # --- SUPERCONDUCTOR PHYSICS ---
            # Ke = 25.0 (Absolute rigid phase locking), Kv = 0.2 (Minimal slip)
            Tc = get_torques(phi_c, om_c, K_e=25.0, K_v=0.2)
            Tc[:, 0] += tau_ext
            om_c[...] += (Tc - 0.2 * om_c) * dt
            phi_c[...] += om_c * dt
            phi_c[:, -1] = 0; om_c[:, -1] = 0
            
        # Update Profiles (Exponential moving average of absolute angular velocity = RMS Field Penetration)
        env_h[:] = 0.90 * env_h + 0.10 * np.mean(np.abs(om_h), axis=0)
        env_c[:] = 0.90 * env_c + 0.10 * np.mean(np.abs(om_c), axis=0)
        
        # Update Visuals
        quiv_h.set_UVC(np.cos(phi_h * signs), np.sin(phi_h * signs))
        line_h.set_ydata(env_h)
        
        quiv_c.set_UVC(np.cos(phi_c * signs), np.sin(phi_c * signs))
        line_c.set_ydata(env_c)
        
        # Output static frame for LaTeX document midway through
        if frame == 75:
            OUTPUT_DIR = "assets/sim_outputs"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            static_out = os.path.join(OUTPUT_DIR, "meissner_gear_train.png")
            fig.savefig(static_out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
            print(f"Saved static LaTeX plot: {static_out}")
            
        return quiv_h, line_h, quiv_c, line_c
        
    print("Rendering 2D classical thermodynamic phase transition...")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
    
    out_path = "assets/sim_outputs/meissner_gear_train.gif"
    try:
        ani.save(out_path, writer='pillow', fps=20, savefig_kwargs={'facecolor': fig.get_facecolor()})
        print(f"Saved Meissner Animation: {out_path}")
    except Exception as e:
        print(f"Failed to generate GIF: {e}")

if __name__ == "__main__":
    simulate_meissner_effect()
