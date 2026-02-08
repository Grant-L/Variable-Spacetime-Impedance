import numpy as np
import matplotlib.pyplot as plt

def simulate_black_hole_lensing():
    print("Initializing Schwarzschild Lattice...")
    
    # 1. Setup Space
    # Range: -20 to 20 Schwarzschild Radii (Rs)
    L = 20.0 
    
    # 2. The Black Hole (Event Horizon at Rs = 1.0)
    Rs = 1.0 
    
    # Define Refractive Index Function n(r)
    # n(r) = 1 / (1 - Rs/r) is the exact Schwarzschild optical density
    # As r -> Rs, n -> Infinity (Light stops)
    def get_n(x, y):
        r = np.sqrt(x**2 + y**2)
        if r < Rs + 0.1: return 50.0 # Clamp near horizon to avoid Infinity
        return 1.0 / (1.0 - Rs/r)

    # Gradient of n (The "Force" pulling the photon)
    def get_grad_n(x, y):
        r = np.sqrt(x**2 + y**2)
        if r < Rs + 0.2: return 0, 0
        
        # Analytic gradient of n = 1/(1-1/r)
        # dn/dr = -1 / (r-1)^2
        dn_dr = -1.0 / ((r - Rs)**2) 
        
        # Components
        nx = dn_dr * (x/r)
        ny = dn_dr * (y/r)
        return nx, ny

    # 3. Launch Photons (A Beam)
    # Launch 10 photons at different impact parameters (y-heights)
    photons_y = np.linspace(0.5, 8.0, 12)
    start_x = 15.0 # Start far right
    
    dt = 0.05
    steps = 1500
    
    plt.figure(figsize=(10, 8))
    
    print(f"Tracing {len(photons_y)} Photon Geodesics...")
    
    for y_init in photons_y:
        px, py = start_x, y_init
        
        # Initial Velocity (Moving Left at c=1)
        # We normalize speed to local c/n
        n_local = get_n(px, py)
        v = 1.0 / n_local
        vx, vy = -v, 0.0 
        
        traj_x, traj_y = [px], [py]
        captured = False
        
        for t in range(steps):
            r_sq = px**2 + py**2
            
            # Event Horizon Check
            if r_sq < Rs**2 + 0.1:
                captured = True
                break
                
            # Update Velocity (Snell's Law / Optical Geodesic Equation)
            # Acceleration is proportional to Gradient of Index
            # F = grad(n)
            gx, gy = get_grad_n(px, py)
            
            # This 'force' turns the velocity vector toward higher n
            vx += -gx * dt 
            vy += -gy * dt
            
            # Renormalize speed (Speed of light must be 1/n)
            # This ensures we are simulating LIGHT, not a massive rock
            v_curr = np.sqrt(vx**2 + vy**2)
            n_curr = get_n(px, py)
            v_target = 1.0 / n_curr
            
            vx = (vx / v_curr) * v_target
            vy = (vy / v_curr) * v_target
            
            px += vx * dt
            py += vy * dt
            
            traj_x.append(px)
            traj_y.append(py)
            
        # Plot Path
        color = 'r-' if captured else 'g-'
        alpha = 0.3 if captured else 0.8
        width = 1 if captured else 1.5
        plt.plot(traj_x, traj_y, color, linewidth=width, alpha=alpha)

    # Draw Black Hole
    circle = plt.Circle((0, 0), Rs, color='black', zorder=10, label="Event Horizon ($R_s$)")
    plt.gca().add_patch(circle)
    
    # Draw Photon Sphere (1.5 Rs) - Theoretical Light Orbit
    circle_ph = plt.Circle((0, 0), 1.5*Rs, color='orange', fill=False, linestyle='--', label="Photon Sphere")
    plt.gca().add_patch(circle_ph)

    plt.title("Simulation D.9: Black Hole Geodesics (Variable Vacuum Index)")
    plt.xlabel("Distance ($R_s$)"); plt.ylabel("Distance ($R_s$)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    simulate_black_hole_lensing()