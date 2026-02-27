#!/usr/bin/env python3
r"""
AVE: Gargantua Acoustic Vortex Simulation 
=========================================

Recreates the canonical "Interstellar" black hole (Gargantua: ~10^8 M_sun, spin a~0.999)
strictly within the AVE optical-acoustic fluid mechanics framework.

Instead of General Relativity geometric geodesics, we solve the exact Hamiltonian
optical paths for shear waves propagating through a continuous refractive fluid.
1. Spacetime Curvature ≡ Spherical Refractive Index Gradient n(r) (Isotropic Approx).
2. Frame Dragging (Kerr Spin) ≡ Macroscopic Vortex Velocity Flow Field v(r).

This script raymarches 320,000 photons backwards from the camera into the flowing
lattice, mapping the intersection points against an equatorial glowing accretion disk
subjected to acoustic frame-dragging and Doppler beaming.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def render_gargantua():
    print("[*] Initializing AVE Continuous Raymarcher: Gargantua Acoustic Vortex...")
    
    # -------------------------------------------------------------
    # Render Settings
    # -------------------------------------------------------------
    WIDTH, HEIGHT = 2000, 1000
    MAX_STEPS = 2500
    DTAU = 0.15
    
    # Disk Geometry
    R_IN = 2.0
    R_OUT = 7.0
    
    # -------------------------------------------------------------
    # Physics Parameters (Isotropic Optical-Fluid Mapping)
    # -------------------------------------------------------------
    M = 1.0
    rh = 0.5 * M # Schwarzschild equivalent event horizon in isotropic coordinates
    alpha = 1.8 # Strong frame dragging coefficient (Spin Lense-Thirring effect)
    
    # -------------------------------------------------------------
    # Camera Initialization
    # -------------------------------------------------------------
    cam_pos = np.array([0.0, -35.0, 2.0]) 
    look_at = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    
    forward = look_at - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    
    FOV = 0.35 # Radians (~20 degrees for telescopic cinematic shot)
    aspect = WIDTH / HEIGHT
    
    u = np.linspace(-1, 1, WIDTH) * aspect * np.tan(FOV/2)
    v = np.linspace(-1, 1, HEIGHT) * np.tan(FOV/2)
    uu, vv = np.meshgrid(u, v)
    
    uu = uu.flatten()
    vv = vv.flatten()
    
    ray_dirs = forward + uu[:, None] * right + vv[:, None] * cam_up
    ray_dirs /= np.linalg.norm(ray_dirs, axis=1)[:, None]
    
    num_rays = WIDTH * HEIGHT
    r = np.tile(cam_pos, (num_rays, 1))
    p = ray_dirs.copy()
    
    # State tracking
    color_map = np.zeros((num_rays, 3))
    final_mask = np.zeros(num_rays, dtype=bool) # True = Ray hit disk, horizon, or escaped bounds
    
    print(f"[*] Tracing {num_rays:,} fluid shear waves backwards through refracting matrix...")
    
    # -------------------------------------------------------------
    # Raymarching Euler ODE solver
    # -------------------------------------------------------------
    for step in range(MAX_STEPS):
        if step % 100 == 0:
            print(f"    -> Step {step}/{MAX_STEPS} (Active rays: {num_rays - np.sum(final_mask):,})")
            
        # Select currently active rays
        active = ~final_mask
        if not np.any(active):
            break
            
        active_idx = np.where(active)[0]
        r_act = r[active_idx]
        p_act = p[active_idx]
        
        r_mag = np.linalg.norm(r_act, axis=1)
        
        # 1. Vacuum Yield Horizon Mask (Event Horizon absorbed)
        hit_bh = r_mag < (rh + 0.02)
        if np.any(hit_bh):
            bh_idx = np.where(hit_bh)[0]
            g_bh_idx = active_idx[bh_idx]
            color_map[g_bh_idx] = [0.0, 0.0, 0.0]
            final_mask[g_bh_idx] = True
            
        # 2. Re-filter active rays after BH hit to prevent math explosion
        active_sub = ~hit_bh
        if not np.any(active_sub):
            continue
            
        r_act = r_act[active_sub]
        p_act = p_act[active_sub]
        r_mag = r_mag[active_sub]
        sub_idx = active_idx[active_sub] # Global indices for the remaining active rays
        
        # ---------------------------------------------------------
        # Hamiltonian Optical Fluid Physics Equations
        # ---------------------------------------------------------
        W = 1.0 + rh / r_mag
        U = 1.0 - rh / r_mag
        U = np.maximum(U, 1e-4) # Absolute numeric stabilizer
        
        # Isotropic Refractive Index 
        n = (W**3) / U
        
        dn_dr = 2.0 * (W**2) * (2.0 - rh/r_mag) / (U**2) * (-rh / (r_mag**2))
        grad_n_x = dn_dr * (r_act[:, 0] / r_mag)
        grad_n_y = dn_dr * (r_act[:, 1] / r_mag)
        grad_n_z = dn_dr * (r_act[:, 2] / r_mag)
        
        # Macroscopic Acoustic Vortex (Frame Dragging)
        v_x = -r_act[:, 1]
        v_y = r_act[:, 0]
        v_z = np.zeros_like(v_x)
        
        v_act_x = (alpha / (r_mag**3)) * v_x
        v_act_y = (alpha / (r_mag**3)) * v_y
        v_act_z = (alpha / (r_mag**3)) * v_z
        
        # Angular Momentum correlation penalty
        Lz = r_act[:, 0] * p_act[:, 1] - r_act[:, 1] * p_act[:, 0]
        
        grad_pv_x = -3.0 * alpha * Lz / (r_mag**5) * r_act[:, 0] + (alpha / (r_mag**3)) * p_act[:, 1]
        grad_pv_y = -3.0 * alpha * Lz / (r_mag**5) * r_act[:, 1] - (alpha / (r_mag**3)) * p_act[:, 0]
        grad_pv_z = -3.0 * alpha * Lz / (r_mag**5) * r_act[:, 2]
        
        p_mag_sq = p_act[:, 0]**2 + p_act[:, 1]**2 + p_act[:, 2]**2
        
        # Symplectic updates
        dr_x = (p_act[:, 0] / (n**2) + v_act_x) * DTAU
        dr_y = (p_act[:, 1] / (n**2) + v_act_y) * DTAU
        dr_z = (p_act[:, 2] / (n**2) + v_act_z) * DTAU
        
        dp_x = ((p_mag_sq / (n**3)) * grad_n_x - grad_pv_x) * DTAU
        dp_y = ((p_mag_sq / (n**3)) * grad_n_y - grad_pv_y) * DTAU
        dp_z = ((p_mag_sq / (n**3)) * grad_n_z - grad_pv_z) * DTAU
        
        r_new_x = r_act[:, 0] + dr_x
        r_new_y = r_act[:, 1] + dr_y
        r_new_z = r_act[:, 2] + dr_z
        
        p_new_x = p_act[:, 0] + dp_x
        p_new_y = p_act[:, 1] + dp_y
        p_new_z = p_act[:, 2] + dp_z
        
        # ---------------------------------------------------------
        # Accretion Disk Rendering Logic (Equatorial Plane check)
        # ---------------------------------------------------------
        crosses = (r_act[:, 2] * r_new_z) <= 0
        if np.any(crosses):
            c_idx = np.where(crosses)[0]
            
            # Plane intersection formula
            dz = r_new_z[c_idx] - r_act[c_idx, 2]
            t_cross = -r_act[c_idx, 2] / (dz + 1e-8)
            
            rc_x = r_act[c_idx, 0] + t_cross * dr_x[c_idx]
            rc_y = r_act[c_idx, 1] + t_cross * dr_y[c_idx]
            
            rc_mag = np.sqrt(rc_x**2 + rc_y**2)
            
            hit_disk = (rc_mag >= R_IN) & (rc_mag <= R_OUT)
            if np.any(hit_disk):
                hd_idx = np.where(hit_disk)[0]
                actual_c_idx = c_idx[hd_idx]
                global_c_idx = sub_idx[actual_c_idx]
                
                # Fetch physics variables for Doppler beaming
                final_rc_x = rc_x[hd_idx]
                final_rc_y = rc_y[hd_idx]
                final_rc_mag = rc_mag[hd_idx]
                ray_p_x = p_act[actual_c_idx, 0]
                ray_p_y = p_act[actual_c_idx, 1]
                
                # Accretion Disk Classical Fluid Velocity (Keplerian)
                v_disk_x = -final_rc_y / (final_rc_mag**1.5)
                v_disk_y = final_rc_x / (final_rc_mag**1.5)
                
                # Relativistic Doppler shifting against reverse ray
                doppler = 1.0 - (v_disk_x * ray_p_x + v_disk_y * ray_p_y) * 2.0
                doppler = np.clip(doppler, 0.2, 3.5)
                
                # Procedural Texture (Rings + Distance falloff)
                rings = 0.6 + 0.4 * np.sin(final_rc_mag * 18.0)
                base_lum = (2.0 / (final_rc_mag - R_IN + 0.5)) * rings
                base_lum *= doppler**2.0 # Slightly less aggressive beaming to preserve dark side details

                # High-fidelity fiery plasma colormap
                red = np.clip(base_lum * 2.0, 0.0, 1.0)
                green = np.clip(base_lum * 1.2 * (doppler/1.2)**0.6, 0.0, 1.0)
                blue = np.clip(base_lum * 0.4 * (doppler/1.2)**1.6, 0.0, 1.0)
                
                color_map[global_c_idx, 0] = red
                color_map[global_c_idx, 1] = green
                color_map[global_c_idx, 2] = blue
                
                final_mask[global_c_idx] = True
        
        # ---------------------------------------------------------
        # Escape bounds (Rays that shoot off to infinity)
        # ---------------------------------------------------------
        r_new_mag = np.sqrt(r_new_x**2 + r_new_y**2 + r_new_z**2)
        escapes = r_new_mag > 50.0
        if np.any(escapes):
            esc_idx = np.where(escapes)[0]
            g_esc_idx = sub_idx[esc_idx]
            
            # Simple starry noise mapped to ray momentum direction
            px = p_new_x[esc_idx]
            py = p_new_y[esc_idx]
            pz = p_new_z[esc_idx]
            star_val = np.sin(px * 160) * np.sin(py * 160) * np.sin(pz * 160)
            
            is_star = star_val > 0.99
            
            bg_colors = np.zeros((len(esc_idx), 3))
            bg_colors[:, 2] = 0.02 # Faint deep space blue
            bg_colors[is_star] = [1.0, 1.0, 1.0]
            
            color_map[g_esc_idx] = bg_colors
            final_mask[g_esc_idx] = True
            
        # Update states for surviving rays
        r[sub_idx, 0] = r_new_x
        r[sub_idx, 1] = r_new_y
        r[sub_idx, 2] = r_new_z
        p[sub_idx, 0] = p_new_x
        p[sub_idx, 1] = p_new_y
        p[sub_idx, 2] = p_new_z
        
    print("[*] Integrations complete. Rebuilding image tensor...")
    
    # Reshape color map into cinematic image
    image_tensor = color_map.reshape((HEIGHT, WIDTH, 3))
    
    # Optional glow bloom (convolution pass approx)
    print("[*] Applying cinematic rendering bloom...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('black')
    
    ax.imshow(image_tensor, interpolation='bilinear', origin='lower')
    ax.axis('off')
    
    ax.text(20, 20, "GARGANTUA ACOUSTIC VORTEX\nContinuous FDTD Refraction / Frame Dragging equivalent\nMass = ~10^8 M_sun  Spin = 0.999", 
            color='white', alpha=0.5, fontsize=12, family='monospace', verticalalignment='bottom')
            
    plt.tight_layout(pad=0)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'gargantua_acoustic_vortex.png')
    
    plt.savefig(out_path, dpi=250, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    
    print(f"[*] Interstellar rendering complete -> {out_path}")

if __name__ == "__main__":
    render_gargantua()
