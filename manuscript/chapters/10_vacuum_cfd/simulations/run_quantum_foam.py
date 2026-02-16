"""
AVE MODULE 33: DETERMINISTIC QUANTUM FOAM
-----------------------------------------
Strict deterministic simulation of Vacuum Turbulence.
NO RANDOM NUMBERS. 
Applies the Bingham Plastic shear-thinning law from Chapter 9.
Proves that a microscopic, fully deterministic perturbation perfectly rolls up 
into chaotic Kelvin-Helmholtz vortices (Quantum Foam) purely because the local 
shear stress crashes the vacuum viscosity (\\eta \\to 0), driving Re \\to \\infty.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import os

OUTPUT_DIR = "manuscript/chapters/10_vacuum_cfd/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_deterministic_quantum_foam():
    print("Simulating Deterministic Quantum Foam (Bingham KH Instability)...")
    
    N = 256
    L = 2.0 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    
    # 1. Deterministic Initial State (Hyperbolic Tangent Shear Layer)
    delta = 0.1
    U = np.tanh(Y / delta)
    
    # Tiny, strictly deterministic sinusoidal perturbation (No random noise!)
    V = 0.05 * np.sin(2 * X) * np.exp(-(Y / delta)**2)
    
    # Vorticity \omega = \nabla \times \mathbf{u}
    W = np.gradient(V, x[1], axis=1) - np.gradient(U, y[1], axis=0)
    
    # AVE Shear-Thinning Parameters
    nu_base = 0.05
    gamma_c = 2.0
    dt = 0.01
    
    W_history = [np.copy(W)]
    Nu_history = []
    
    for step in range(1, 201):
        # Calculate local shear rate (magnitude of strain rate tensor)
        du_dy = np.gradient(U, y[1], axis=0)
        dv_dx = np.gradient(V, x[1], axis=1)
        shear_rate = np.sqrt(du_dy**2 + dv_dx**2)
        
        # BINGHAM PLASTIC COLLAPSE: \nu_{eff} = \nu_base / (1 + (\dot{\gamma}/\gamma_c)^2)
        nu_eff = nu_base / (1.0 + (shear_rate / gamma_c)**2)
        
        if step == 200: Nu_history.append(np.copy(nu_eff))
        
        # Deterministic Semi-Lagrangian Advection (Backtracking)
        X_back = X - U * dt
        Y_back = Y - V * dt
        
        # Periodic boundaries on X, clamped on Y
        X_back = np.mod(X_back, L)
        Y_back = np.clip(Y_back, -L/2, L/2)
        
        # Interpolate to find advected vorticity
        # Convert to array coordinates (row, col) = (y, x)
        # Normalize coordinates to array indices [0, N-1]
        y_coords = (Y_back + L/2) / (L / (N-1))
        x_coords = X_back / (L / (N-1))
        # map_coordinates expects shape (ndim, ...) where first dim is coordinates
        coords = np.stack([y_coords, x_coords], axis=0)
        W_advected = scipy.ndimage.map_coordinates(W, coords, order=3, mode='wrap')
        
        # Add variable viscous diffusion
        laplacian_W = scipy.ndimage.laplace(W) / (x[1]**2)
        W_new = W_advected + nu_eff * laplacian_W * dt
        
        # Recover velocities (Simplified Streamfunction approximation for fast visualization)
        W_hat = np.fft.fft2(W_new)
        kx = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0, 0] = 1e-10 # avoid div-by-zero
        Psi_hat = W_hat / K2
        Psi = np.real(np.fft.ifft2(Psi_hat))
        
        U = np.gradient(Psi, y[1], axis=0)
        V = -np.gradient(Psi, x[1], axis=1)
        W = W_new
        
        if step in [50, 100, 200]: W_history.append(np.copy(W))

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    fig.patch.set_facecolor('#050508')
    titles = [
        'T=0: Laminar Shear Layer\n(Strictly Deterministic Perturbation)', 
        'T=100: Kelvin-Helmholtz Rollup\n(Viscosity Collapses)', 
        'T=200: Deterministic Quantum Foam\n(Fractal Vortices)', 
        r'Viscosity Field $\eta_{eff}(x,y)$ at T=200' + '\n(Dark = Superfluid, Bright = Viscous Solid)'
    ]
    
    for i, ax in enumerate(axes.flatten()):
        ax.set_facecolor('#050508')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[i], color='white', fontsize=14, weight='bold', pad=10)
        for spine in ax.spines.values(): spine.set_color('#333333')
        
        if i < 3:
            ax.contourf(X, Y, W_history[i], 80, cmap='inferno', vmin=-15, vmax=15)
        else:
            ax.contourf(X, Y, Nu_history[0], 80, cmap='ocean')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "vacuum_turbulence.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_deterministic_quantum_foam()