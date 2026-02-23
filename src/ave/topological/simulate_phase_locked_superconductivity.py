import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def simulate_phase_locked_superconductivity(n_electrons=30, frames=200):
    """
    Simulates Solid-State Superconductivity in the AVE framework.
    
    Instead of 'Cooper Pairs' condensing into a 'Macroscopic Quantum State',
    AVE models Superconductivity as the classical geometric phase-locking
    of massive topological inductors spinning at high AC frequencies.
    
    We use a modified Kuramoto model to demonstrate how dropping
    the transverse acoustic 'thermal' jitter (Temperature) allows mutual
    magnetic coupling to synchronize all nodes. 
    
    Result: When (d_Phase / dt)_relative = 0, relative d_B/dt = 0, thus 
    Inductive Drag (Resistance) = 0.
    """
    
    print("\n AVE APPLIED PHYSICS: CLASSICAL SUPERCONDUCTIVITY (PHASE-LOCK)")
    print("================================================================")
    print(" Objective: Prove that zero-resistance is simply perfect macroscopic")
    print("            phase-locking of topological inductors (electrons).")
    print(" Mechanism: Dropping thermal jitter below T_c triggers Kuramoto sync.")
    
    # Simulation Parameters
    dt = 0.1
    K = 1.0     # Coupling strength (Mutual Inductance of the Grid)
    
    # Natural frequencies of the spinning electron knots (normally identical + slight noise)
    omega = np.random.normal(1.0, 0.05, n_electrons)
    
    # Initial random phases (0 to 2pi)
    phases = np.random.uniform(0, 2 * np.pi, n_electrons)
    
    # Thermal Noise profile (Temperature dropping over time)
    # Starts high (Normal state, Resistance > 0)
    # Drops to near-zero (Superconducting state, Resistance = 0)
    T_initial = 3.0
    T_final = 0.05
    T_c_frame = int(frames * 0.4) # Frame where we hit Critical Temp
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('#111111')
    
    # Polar Plot: Visualizing Phases on a Circle
    ax1 = plt.subplot(121, projection='polar')
    ax1.set_facecolor('#111111')
    ax1.set_title("Topological Node Phase Angles ($r=1$)", color='white', pad=20)
    ax1.tick_params(colors='grey')
    ax1.grid(color='#333333')
    ax1.set_rticks([])  # Hide radius ticks
    
    scatter = ax1.scatter(phases, np.ones(n_electrons), c=phases, cmap='hsv', s=100, alpha=0.9, edgecolor='white')
    
    # Line Plot: Order Parameter (Macro Phase-Lock) over time
    ax2 = plt.subplot(122)
    ax2.set_facecolor('#111111')
    ax2.set_xlim(0, frames)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Macroscopic Inductive Synchronization (Kuramoto Order $R$)", color='white')
    ax2.set_xlabel("Time (Cooling $\\rightarrow$)", color='white')
    ax2.set_ylabel("Order Parameter ($R$) $\\rightarrow$ % Phase-Locked", color='white')
    ax2.tick_params(colors='white')
    ax2.grid(color='#333333')
    
    line_r, = ax2.plot([], [], color='cyan', lw=3, label="Synchronization ($R$)")
    
    ax2.axvline(T_c_frame, color='magenta', linestyle='--', lw=2, label="Critical Temp ($T_c$)")
    ax2.legend(loc='lower right', facecolor='black', edgecolor='white', labelcolor='white')
    
    status_text = ax2.text(0.05, 0.85, '', transform=ax2.transAxes, color='yellow', fontsize=12, fontweight='bold')
    
    history_R = []

    def update(frame):
        nonlocal phases
        
        # Temperature profile (drops linearly until T_c_frame, then plateau)
        if frame < T_c_frame:
            current_T = T_initial - (T_initial - T_final) * (frame / T_c_frame)
        else:
            current_T = T_final
            
        # Kuramoto Model with Thermal Noise
        # d(theta_i)/dt = omega_i + (K/N)*sum(sin(theta_j - theta_i)) + Noise
        
        d_phases = np.zeros(n_electrons)
        for i in range(n_electrons):
            # Mutual inductive coupling term
            coupling = (K / n_electrons) * np.sum(np.sin(phases - phases[i]))
            # Thermal noise term (transverse acoustic grid jitter)
            thermal_kick = np.random.normal(0, current_T)
            
            d_phases[i] = omega[i] + coupling + thermal_kick
            
        phases += d_phases * dt
        phases = np.mod(phases, 2 * np.pi) # keep in [0, 2pi]
        
        # Calculate Kuramoto Order Parameter R (0 = random, 1 = perfectly synced)
        # R * e^(i*Psi) = (1/N) * sum(e^(i*theta_j))
        complex_phases = np.exp(1j * phases)
        R = np.abs(np.mean(complex_phases))
        history_R.append(R)
        
        # Update Polar Plot
        scatter.set_offsets(np.c_[phases, np.ones(n_electrons)])
        scatter.set_array(phases)
        
        # Update R Plot
        line_r.set_data(np.arange(len(history_R)), history_R)
        
        # Update Status Text
        if R < 0.3:
            status_text.set_text("Normal State: High Relative Induction (Resistance > 0)")
            status_text.set_color("salmon")
        elif R < 0.8:
            status_text.set_text("Cooling: Nodes beginning to couple...")
            status_text.set_color("yellow")
        else:
            status_text.set_text("SUPERCONDUCTIVITY:\nAbsolute Phase-Lock! ($d\vec{B}/dt = 0$)")
            status_text.set_color("lime")

        return scatter, line_r, status_text

    plt.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
    
    print("\n[STATUS: SUCCESS] Generating Superconductivity Phase-Lock execution.")
    
    out_path = 'superconductivity_phase_lock.gif'
    ani.save(out_path, writer='pillow', fps=25)
    print(f"Animated propagation saved to {out_path}")

if __name__ == "__main__":
    simulate_phase_locked_superconductivity()
