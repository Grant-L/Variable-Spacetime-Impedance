import numpy as np
import matplotlib.pyplot as plt

def run_bullet_cluster_sim():
    print("LCT Simulation J: The Bullet Cluster (Superfluid Dark Matter)")
    
    # --- Setup Clusters ---
    # Cluster A (Moving Right) and Cluster B (Moving Left)
    num_particles = 200
    
    # 1. Initialize Positions (random clouds)
    # Gas (Baryonic - Red)
    gas_A_x = np.random.normal(-5, 1, num_particles)
    gas_B_x = np.random.normal(5, 1, num_particles)
    gas_y = np.random.normal(0, 1, num_particles)
    
    # Halo (Superfluid Vortex - Blue)
    halo_A_x = np.random.normal(-5, 1, num_particles)
    halo_B_x = np.random.normal(5, 1, num_particles)
    halo_y = np.random.normal(0, 1, num_particles)
    
    # 2. Velocity Setup (Collision Course)
    v_A = 2.0  # Moving Right
    v_B = -2.0 # Moving Left
    
    # 3. Simulate Collision (Time Evolution)
    dt = 0.1
    steps = 40
    
    # Lists to store "final" state for plotting
    final_gas_A = []
    final_gas_B = []
    final_halo_A = []
    final_halo_B = []
    
    # Physics Loop
    for t in range(steps):
        # Update Halos (Superfluid = No Viscosity/Friction)
        halo_A_x += v_A * dt
        halo_B_x += v_B * dt
        
        # Update Gas (Viscous Fluid = Friction/Ram Pressure)
        # Interaction: As they approach x=0, friction increases
        friction_A = 0.0
        friction_B = 0.0
        
        # Simple drag model: if clouds overlap (near x=0), slow down
        center_A = np.mean(gas_A_x)
        center_B = np.mean(gas_B_x)
        dist = abs(center_A - center_B)
        
        if dist < 4.0: # Interaction zone
            drag_factor = 0.15 # Viscosity coeff
            friction_A = -v_A * drag_factor
            friction_B = -v_B * drag_factor
            
        gas_A_x += (v_A + friction_A) * dt
        gas_B_x += (v_B + friction_B) * dt

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Plot Halos (The "Dark Matter" - Separated)
    plt.scatter(halo_A_x, halo_y, color='blue', alpha=0.3, label='Superfluid Vortex Halo (Dark Matter)')
    plt.scatter(halo_B_x, halo_y, color='blue', alpha=0.3)
    
    # Plot Gas (The "Visible Matter" - Lagging/Shocked)
    plt.scatter(gas_A_x, gas_y, color='red', marker='^', alpha=0.5, label='Viscous Baryonic Gas (X-Ray)')
    plt.scatter(gas_B_x, gas_y, color='red', marker='^', alpha=0.5)
    
    plt.title("LCT Solution to the Bullet Cluster")
    plt.xlabel("Position (Mpc)")
    plt.yticks([])
    plt.axvline(0, color='black', linestyle='--', alpha=0.2)
    plt.text(0, -3, "Collision Center", ha='center')
    plt.legend()
    
    # Annotations
    plt.arrow(np.mean(halo_A_x), 2, 0, -0.5, color='blue', head_width=0.2)
    plt.text(np.mean(halo_A_x), 2.2, "Gravity Center", color='blue', ha='center')
    
    plt.arrow(np.mean(gas_A_x), -2, 0, 0.5, color='red', head_width=0.2)
    plt.text(np.mean(gas_A_x), -2.4, "Mass Center", color='red', ha='center')
    
    print("Simulation J Complete: Generated bullet_cluster.png")
    plt.savefig('bullet_cluster.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_bullet_cluster_sim()