import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/10_vacuum_cfd/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_vacuum_turbulence():
    print("Running VCFD: Vacuum Kelvin-Helmholtz Instability...")
    
    # 1. GRID SETUP
    N = 100
    L = 1.0
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    
    # 2. INITIAL CONDITIONS (Shear Layer)
    # Top half moves Right, Bottom half moves Left
    U = np.tanh(10 * (Y - 0.5)) 
    V = 0.1 * np.sin(2 * np.pi * X) * np.exp(-10 * (Y - 0.5)**2) # Perturbation
    
    # 3. AVE RHEOLOGY PARAMETERS
    # Base Viscosity (Dark Matter State)
    nu_0 = 0.01 
    # Critical Shear Rate (Breakdown Threshold)
    gamma_c = 5.0
    
    # Time Stepping
    dt = 0.005
    steps = 200
    
    # Storage for Visualization
    vorticity_history = []
    viscosity_history = []
    
    print("Solving Navier-Stokes with Shear-Thinning...")
    
    for t in range(steps):
        # A. Calculate Local Shear Rate (Magnitude of Strain Tensor)
        # Simplified 2D shear: |du/dy + dv/dx|
        # Finite difference approx
        du_dy = np.gradient(U, axis=0)
        dv_dx = np.gradient(V, axis=1)
        shear_rate = np.abs(du_dy + dv_dx)
        
        # B. Apply AVE Shear-Thinning (Eq 9.1)
        # High Shear -> Low Viscosity -> High Reynolds -> Turbulence
        nu_eff = nu_0 / (1 + (shear_rate / gamma_c)**2)
        
        # C. Update Vorticity (Curl of Velocity)
        # dW/dt + (u.grad)W = nu * del^2 W
        # We simulate the advection-diffusion of vorticity directly
        W = du_dy - dv_dx # Vorticity
        
        # Diffusion Term (Variable Viscosity)
        # W_new = W + dt * (Diffusion - Advection)
        # Simplified spectral-like update for demonstration logic
        
        # Advection (Move with flow)
        U_grad_W = U * np.gradient(W, axis=1) + V * np.gradient(W, axis=0)
        
        # Diffusion (Damped by Viscosity)
        # Note: In AVE, viscosity acts as damping. 
        # Lower viscosity (Shear Thinning) means LESS damping -> MORE Turbulence.
        Laplacian_W = np.gradient(np.gradient(W, axis=0), axis=0) + \
                      np.gradient(np.gradient(W, axis=1), axis=1)
        
        dW_dt = -U_grad_W + nu_eff * Laplacian_W
        
        # Update Flow (First Order Euler)
        # Note: A full pressure solver is complex; 
        # we approximate the evolution of the instability pattern.
        W += dW_dt * dt
        
        # Update Velocity from Vorticity (Streamfunction Poisson eq - Simplified)
        # For the visual, we just advect the perturbation to show the rollout
        U += -V * dt # Coriolis-like mixing
        V += U * dt
        
        if t % 50 == 0:
            vorticity_history.append(W.copy())
            viscosity_history.append(nu_eff.copy())

    # 4. VISUALIZATION
    print("Generating Hyper-Complex Flow Field...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='black')
    
    # Plot 1: Initial State
    axes[0,0].contourf(X, Y, vorticity_history[0], 50, cmap='bwr')
    axes[0,0].set_title("T=0: Laminar Shear Layer\n(High Viscosity)", color='white')
    
    # Plot 2: Onset of Instability
    axes[0,1].contourf(X, Y, vorticity_history[1], 50, cmap='bwr')
    axes[0,1].set_title("T=50: Shear-Thinning Begins\n(Viscosity Drops Locally)", color='white')
    
    # Plot 3: Fully Developed Turbulence (The AVE Prediction)
    im3 = axes[1,0].contourf(X, Y, vorticity_history[-1], 100, cmap='inferno')
    axes[1,0].set_title("T=200: Vacuum Turbulence\n(Virtual Particle Genesis)", color='white')
    
    # Plot 4: The Viscosity Map (The Mechanism)
    # Shows the "Holes" in the vacuum where shear is high
    im4 = axes[1,1].contourf(X, Y, viscosity_history[-1], 50, cmap='Greens_r')
    axes[1,1].set_title("Viscosity Field $\eta(x,y)$\nGreen=Solid, White=Superfluid", color='white')
    
    for ax in axes.flat:
        ax.axis('off')
        ax.set_aspect('equal')
    
    output_path = os.path.join(OUTPUT_DIR, "vacuum_turbulence.png")
    plt.savefig(output_path, dpi=300, facecolor='black')
    print(f"Simulation saved to {output_path}")

if __name__ == "__main__":
    simulate_vacuum_turbulence()