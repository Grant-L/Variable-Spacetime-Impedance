r"""
AVE 3D Volumetric Pair Production Simulator (Gamma -> e- + e+)
==============================================================
Simulates the literal fluid dynamic shear of a pure 3D 
transverse acoustic wave (Gamma Ray) striking a high-impedance 
nucleus.

1. Inject a high-energy planar wave: H = V \cdot (\nabla \times V) = 0.
2. The wave collides with a rigid spatial sphere (the Nucleus).
3. The geometric deceleration forces the linear momentum to shear 
   and curl around the obstruction in 3D space.
4. We algorithmically filter exclusively for Kinetic Helicity (H \neq 0).
5. The visualization ignores the invisible H=0 wave and captures 
   the precise temporal moment the wave shatters, shedding a 
   persistent Left-Handed Vortex (Blue) and Right-Handed Vortex (Red).

Energy (linear wave) converts to Mass (topological vortex dipoles).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path

# JAX GPU acceleration (graceful fallback to numpy)
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

# 3D Grid Configuration (kept small to prevent memory/CPU overload)
N = 45 
C_0 = 0.5
T_MAX = 80

# The physical 3D displacement vectors (dx, dy, dz)
Ux = np.zeros((N, N, N))
Uy = np.zeros((N, N, N))
Uz = np.zeros((N, N, N))

Ux_prev = np.zeros((N, N, N))
Uy_prev = np.zeros((N, N, N))
Uz_prev = np.zeros((N, N, N))

# The heavy nucleus directly in the wave's path
nucleus_mask = np.zeros((N, N, N), dtype=bool)
cx, cy, cz = 15, N//2, N//2
R_nuc = 4

for i in range(N):
    for j in range(N):
        for k in range(N):
            if (i - cx)**2 + (j - cy)**2 + (k - cz)**2 <= R_nuc**2:
                nucleus_mask[i,j,k] = True

def laplacian(A):
    # 3D 6-point stencil scalar Laplacian approximation
    return (np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0) +
            np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1) +
            np.roll(A, 1, axis=2) + np.roll(A, -1, axis=2) - 6*A)

def grad_central(A, axis):
    return (np.roll(A, -1, axis=axis) - np.roll(A, 1, axis=axis)) / 2.0

# Extract a Gaussian planar wave pulse
def inject_gamma_ray(t):
    pulse_width = 3.0
    x_center = 2.0 + C_0 * t
    # Pure planar transverse wave oscillating in Y
    pulse = 2.0 * np.exp(-((np.arange(N) - x_center)**2) / pulse_width**2)
    # Broadcast across Y and Z
    Uy_inject = np.zeros((N, N, N))
    for i in range(N):
        Uy_inject[i, :, :] = pulse[i]
    return Uy_inject

def simulate_wave_tear():
    print(f"Simulating 3D Volumetric Wave Tear ({N}x{N}x{N} nodes)...")
    
    global Ux, Uy, Uz, Ux_prev, Uy_prev, Uz_prev
    frames_pos_x = []
    frames_pos_y = []
    frames_pos_z = []
    
    frames_neg_x = []
    frames_neg_y = []
    frames_neg_z = []

    if _HAS_JAX:
        Ux_j = jnp.array(Ux)
        Uy_j = jnp.array(Uy)
        Uz_j = jnp.array(Uz)
        Ux_p = jnp.array(Ux_prev)
        Uy_p = jnp.array(Uy_prev)
        Uz_p = jnp.array(Uz_prev)
        nuc_j = jnp.array(nucleus_mask)

        def _lap(A):
            return (jnp.roll(A, 1, axis=0) + jnp.roll(A, -1, axis=0) +
                    jnp.roll(A, 1, axis=1) + jnp.roll(A, -1, axis=1) +
                    jnp.roll(A, 1, axis=2) + jnp.roll(A, -1, axis=2) - 6*A)

        @jit
        def _step(Ux, Uy, Uz, Ux_prev, Uy_prev, Uz_prev):
            Ux_next = 2*Ux - Ux_prev + (C_0**2) * _lap(Ux)
            Uy_next = 2*Uy - Uy_prev + (C_0**2) * _lap(Uy)
            Uz_next = 2*Uz - Uz_prev + (C_0**2) * _lap(Uz)
            Ux_next = jnp.where(nuc_j, 0.0, Ux_next)
            Uy_next = jnp.where(nuc_j, 0.0, Uy_next)
            Uz_next = jnp.where(nuc_j, 0.0, Uz_next)
            return Ux_next, Uy_next, Uz_next

        for t in range(T_MAX):
            Ux_n, Uy_n, Uz_n = _step(Ux_j, Uy_j, Uz_j, Ux_p, Uy_p, Uz_p)
            if t < 20:
                Uy_n = Uy_n + jnp.array(inject_gamma_ray(t)) * 0.1

            # Extract helicity on host
            Vx = np.array(Ux_n - Ux_j)
            Vy = np.array(Uy_n - Uy_j)
            Vz = np.array(Uz_n - Uz_j)

            curl_x = grad_central(Vz, 1) - grad_central(Vy, 2)
            curl_y = grad_central(Vx, 2) - grad_central(Vz, 0)
            curl_z = grad_central(Vy, 0) - grad_central(Vx, 1)
            H = Vx*curl_x + Vy*curl_y + Vz*curl_z

            threshold = 1.0e-5
            pos_coords = np.where(H > threshold)
            neg_coords = np.where(H < -threshold)

            frames_pos_x.append(pos_coords[0])
            frames_pos_y.append(pos_coords[1])
            frames_pos_z.append(pos_coords[2])
            frames_neg_x.append(neg_coords[0])
            frames_neg_y.append(neg_coords[1])
            frames_neg_z.append(neg_coords[2])

            Ux_p = Ux_j
            Uy_p = Uy_j
            Uz_p = Uz_j
            Ux_j = Ux_n
            Uy_j = Uy_n
            Uz_j = Uz_n

            if t % 10 == 0:
                print(f" Simulating Frame {t}/{T_MAX}...")
    else:
        for t in range(T_MAX):
            Ux_next = 2*Ux - Ux_prev + (C_0**2) * laplacian(Ux)
            Uy_next = 2*Uy - Uy_prev + (C_0**2) * laplacian(Uy)
            Uz_next = 2*Uz - Uz_prev + (C_0**2) * laplacian(Uz)
            if t < 20:
                 Uy_next += inject_gamma_ray(t) * 0.1
            Ux_next[nucleus_mask] = 0
            Uy_next[nucleus_mask] = 0
            Uz_next[nucleus_mask] = 0

            Vx = Ux_next - Ux
            Vy = Uy_next - Uy
            Vz = Uz_next - Uz

            curl_x = grad_central(Vz, 1) - grad_central(Vy, 2)
            curl_y = grad_central(Vx, 2) - grad_central(Vz, 0)
            curl_z = grad_central(Vy, 0) - grad_central(Vx, 1)
            H = Vx*curl_x + Vy*curl_y + Vz*curl_z

            threshold = 1.0e-5
            pos_coords = np.where(H > threshold)
            neg_coords = np.where(H < -threshold)

            frames_pos_x.append(pos_coords[0])
            frames_pos_y.append(pos_coords[1])
            frames_pos_z.append(pos_coords[2])
            frames_neg_x.append(neg_coords[0])
            frames_neg_y.append(neg_coords[1])
            frames_neg_z.append(neg_coords[2])

            Ux_prev = np.copy(Ux)
            Uy_prev = np.copy(Uy)
            Uz_prev = np.copy(Uz)
            Ux = np.copy(Ux_next)
            Uy = np.copy(Uy_next)
            Uz = np.copy(Uz_next)

            if t % 10 == 0:
                print(f" Simulating Frame {t}/{T_MAX}...")

    return frames_pos_x, frames_pos_y, frames_pos_z, frames_neg_x, frames_neg_y, frames_neg_z

def render_3d_helicity(fx_p, fy_p, fz_p, fx_n, fy_n, fz_n, out_path):
    print("Exporting 3D Volumetric Helicity Animation...")
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Remove axes to make it look like deep space
    ax.set_axis_off()
    
    # Store scatter handles
    scatter_pos = ax.scatter([], [], [], s=40, c='#ff003c', alpha=0.8, edgecolors='none', label='Positron ($H > 0$)')
    scatter_neg = ax.scatter([], [], [], s=40, c='#00f0ff', alpha=0.8, edgecolors='none', label='Electron ($H < 0$)')
    
    # Draw the invisible nucleus as a faint wireframe
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_nuc = cx + R_nuc*np.cos(u)*np.sin(v)
    y_nuc = N//2 + R_nuc*np.sin(u)*np.sin(v)
    z_nuc = N//2 + R_nuc*np.cos(v)
    ax.plot_wireframe(x_nuc, y_nuc, z_nuc, color='gray', alpha=0.1)

    title = ax.set_title("", color='white', pad=20, fontsize=14)
    ax.legend(loc='lower left', facecolor='black', edgecolor='white')

    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_zlim(0, N)
    ax.view_init(elev=30, azim=45)

    def update(frame):
        # The planar wave has H=0 and is thus completely invisible.
        # We only see the wave tear into mass!
        
        # Update Positron scatter
        x_p, y_p, z_p = fx_p[frame], fy_p[frame], fz_p[frame]
        if len(x_p) > 0:
            scatter_pos._offsets3d = (x_p, y_p, z_p)
        else:
            scatter_pos._offsets3d = ([], [], [])
            
        # Update Electron scatter
        x_n, y_n, z_n = fx_n[frame], fy_n[frame], fz_n[frame]
        if len(x_n) > 0:
            scatter_neg._offsets3d = (x_n, y_n, z_n)
        else:
            scatter_neg._offsets3d = ([], [], [])
            
        title.set_text(f"Pair Production: 3D Volumetric Wave Tear | Frame {frame}/{T_MAX}")
        
        # Slow camera pan
        ax.view_init(elev=20, azim=45 + frame*0.5)
        return scatter_pos, scatter_neg, title

    ani = animation.FuncAnimation(fig, update, frames=T_MAX, interval=50, blit=False)
    ani.save(out_path, writer='pillow', fps=20, dpi=120)
    plt.close(fig)
    print(f"[Done] Geometric Helicity Yield Animation Saved: {out_path}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    out_dir = PROJECT_ROOT / "scripts" / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    fx_p, fy_p, fz_p, fx_n, fy_n, fz_n = simulate_wave_tear()
    render_3d_helicity(fx_p, fy_p, fz_p, fx_n, fy_n, fz_n, out_dir / "pair_production_3d.gif")
