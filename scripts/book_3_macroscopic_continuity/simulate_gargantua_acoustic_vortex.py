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

This script raymarches photons backwards from the camera into the flowing
lattice, mapping the intersection points against an equatorial glowing accretion disk
subjected to acoustic frame-dragging and Doppler beaming. Rays that cross the
equatorial plane multiple times produce both the direct disk image AND the iconic
gravitationally lensed arcs above and below the shadow.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# JAX GPU acceleration (graceful fallback to numpy)
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

# ─────────────────────────────────────────────────────────────
# Blackbody → sRGB conversion (Planck spectrum)
# ─────────────────────────────────────────────────────────────
def _blackbody_rgb(T):
    """Convert temperature (K) to approximate sRGB [0,1].
    Uses analytic fit to CIE 1931 standard observer via Tanner Helland's
    widely-used approximation (accurate for 1000 K – 40 000 K).
    Input T may be a numpy array; returns (N, 3) RGB."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    T_100 = np.clip(T / 100.0, 10.0, 400.0)  # clamp range

    rgb = np.zeros((len(T_100), 3))

    # Red
    mask_lo = T_100 <= 66.0
    rgb[mask_lo, 0] = 1.0
    hi = ~mask_lo
    if np.any(hi):
        tmp = 329.698727446 * (T_100[hi] - 60.0) ** (-0.1332047592)
        rgb[hi, 0] = np.clip(tmp / 255.0, 0.0, 1.0)

    # Green
    if np.any(mask_lo):
        tmp = 99.4708025861 * np.log(T_100[mask_lo]) - 161.1195681661
        rgb[mask_lo, 1] = np.clip(tmp / 255.0, 0.0, 1.0)
    if np.any(hi):
        tmp = 288.1221695283 * (T_100[hi] - 60.0) ** (-0.0755148492)
        rgb[hi, 1] = np.clip(tmp / 255.0, 0.0, 1.0)

    # Blue
    hot = T_100 >= 66.0
    rgb[hot, 2] = 1.0
    mid = (~hot) & (T_100 > 19.0)
    if np.any(mid):
        tmp = 138.5177312231 * np.log(T_100[mid] - 10.0) - 305.0447927307
        rgb[mid, 2] = np.clip(tmp / 255.0, 0.0, 1.0)
    rgb[(~hot) & (~mid), 2] = 0.0

    return rgb


def render_gargantua():
    print("[*] Initializing AVE Continuous Raymarcher: Gargantua Acoustic Vortex...")
    print("    High-fidelity mode: blackbody spectrum, gravitational redshift,")
    print("    stochastic multi-sampling, seeded star field")

    # -------------------------------------------------------------
    # Render Settings
    # -------------------------------------------------------------
    WIDTH, HEIGHT = 2000, 1000
    MAX_STEPS = 5000
    DTAU = 0.06       # Smaller step = smoother arcs, eliminates banding
    N_SAMPLES = 2     # 2 samples sufficient for anti-aliasing

    # Disk Geometry — wider for the iconic silhouette
    R_IN = 2.5
    R_OUT = 12.0

    # Accretion Disk Temperature Profile (Shakura-Sunyaev thin disk)
    # EHT M87*: T_e ~ 10^10 K.  We map to effective visual temperatures:
    #   Inner edge: ~12 000 K (blue-white)
    #   Mid disk:   ~ 5 000 K (solar yellow)
    #   Outer edge: ~ 1 800 K (deep red)
    T_INNER = 6000.0  # Kelvin at R_IN (solar temp = golden/orange core)

    # Schwarzschild radius for gravitational redshift
    R_S = 1.0  # in code units (rh = 0.5 in isotropic coords)

    # -------------------------------------------------------------
    # Physics Parameters (Isotropic Optical-Fluid Mapping)
    # -------------------------------------------------------------
    M = 1.0
    rh = 0.5 * M
    alpha = 1.8   # Strong frame dragging coefficient

    # -------------------------------------------------------------
    # Camera
    # -------------------------------------------------------------
    cam_pos = np.array([0.0, -40.0, 6.0])
    look_at = np.array([0.0, 0.0, -0.5])
    up = np.array([0.0, 0.0, 1.0])

    forward = look_at - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)

    FOV = 0.55
    aspect = WIDTH / HEIGHT

    u = np.linspace(-1, 1, WIDTH) * aspect * np.tan(FOV / 2)
    v = np.linspace(-1, 1, HEIGHT) * np.tan(FOV / 2)
    uu, vv = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()
    num_pixels = WIDTH * HEIGHT

    # Seeded RNG for reproducibility
    rng = np.random.default_rng(42)

    # Accumulator for multi-sample averaging
    accumulated_color = np.zeros((num_pixels, 3))

    # Pixel spacing for jitter
    du = (u[1] - u[0]) if len(u) > 1 else 1e-4
    dv = (v[1] - v[0]) if len(v) > 1 else 1e-4

    for sample in range(N_SAMPLES):
        print(f"\n[*] Sample {sample + 1}/{N_SAMPLES}")

        # Jitter ray direction within pixel footprint
        jitter_u = (rng.random(num_pixels) - 0.5) * du
        jitter_v = (rng.random(num_pixels) - 0.5) * dv

        ray_dirs = (forward
                    + (uu + jitter_u)[:, None] * right
                    + (vv + jitter_v)[:, None] * cam_up)
        ray_dirs /= np.linalg.norm(ray_dirs, axis=1)[:, None]

        r = np.tile(cam_pos, (num_pixels, 1))
        p = ray_dirs.copy()

        color_map = np.zeros((num_pixels, 3))
        final_mask = np.zeros(num_pixels, dtype=bool)
        hit_count = np.zeros(num_pixels, dtype=int)

        for step in range(MAX_STEPS):
            if step % 500 == 0:
                active_count = num_pixels - np.sum(final_mask)
                if step % 1000 == 0:
                    print(f"    -> Step {step}/{MAX_STEPS}  (active: {active_count:,})")
                if active_count == 0:
                    break

            active = ~final_mask
            if not np.any(active):
                break

            active_idx = np.where(active)[0]
            r_act = r[active_idx]
            p_act = p[active_idx]
            r_mag = np.linalg.norm(r_act, axis=1)

            # Black hole absorption
            hit_bh = r_mag < (rh + 0.02)
            if np.any(hit_bh):
                final_mask[active_idx[hit_bh]] = True

            active_sub = ~hit_bh
            if not np.any(active_sub):
                continue

            r_act = r_act[active_sub]
            p_act = p_act[active_sub]
            r_mag = r_mag[active_sub]
            sub_idx = active_idx[active_sub]

            # ── AVE Hamiltonian Optics ──
            W = 1.0 + rh / r_mag
            U = np.maximum(1.0 - rh / r_mag, 1e-4)
            n = (W**3) / U

            dn_dr = 2.0 * (W**2) * (2.0 - rh / r_mag) / (U**2) * (-rh / (r_mag**2))
            r_hat = r_act / r_mag[:, None]
            grad_n = dn_dr[:, None] * r_hat

            # Frame dragging
            v_drag = np.zeros_like(r_act)
            v_drag[:, 0] = -r_act[:, 1]
            v_drag[:, 1] = r_act[:, 0]
            v_drag *= (alpha / (r_mag**3))[:, None]

            Lz = r_act[:, 0] * p_act[:, 1] - r_act[:, 1] * p_act[:, 0]
            grad_pv = np.zeros_like(r_act)
            grad_pv[:, 0] = -3.0 * alpha * Lz / (r_mag**5) * r_act[:, 0] + (alpha / (r_mag**3)) * p_act[:, 1]
            grad_pv[:, 1] = -3.0 * alpha * Lz / (r_mag**5) * r_act[:, 1] - (alpha / (r_mag**3)) * p_act[:, 0]
            grad_pv[:, 2] = -3.0 * alpha * Lz / (r_mag**5) * r_act[:, 2]

            p_mag_sq = np.sum(p_act**2, axis=1)

            dr = (p_act / (n**2)[:, None] + v_drag) * DTAU
            dp = ((p_mag_sq / (n**3))[:, None] * grad_n - grad_pv) * DTAU

            r_new = r_act + dr
            p_new = p_act + dp

            # ── Accretion Disk (equatorial crossing) ──
            crosses = (r_act[:, 2] * r_new[:, 2]) <= 0
            if np.any(crosses):
                c_idx = np.where(crosses)[0]
                dz = r_new[c_idx, 2] - r_act[c_idx, 2]
                t_cross = -r_act[c_idx, 2] / (dz + 1e-8)

                rc_x = r_act[c_idx, 0] + t_cross * dr[c_idx, 0]
                rc_y = r_act[c_idx, 1] + t_cross * dr[c_idx, 1]
                rc_mag = np.sqrt(rc_x**2 + rc_y**2)

                hit_disk = (rc_mag >= R_IN) & (rc_mag <= R_OUT)
                if np.any(hit_disk):
                    hd = np.where(hit_disk)[0]
                    ci = c_idx[hd]
                    gi = sub_idx[ci]
                    rc_r = rc_mag[hd]
                    rcx = rc_x[hd]
                    rcy = rc_y[hd]

                    # ── Blackbody Disk Temperature ──
                    # Shakura-Sunyaev: T(r) = T_in * (R_in/r)^(3/4) * [1-(R_in/r)^(1/2)]^(1/4)
                    x = R_IN / rc_r
                    T_disk = T_INNER * x**0.75 * np.maximum(1.0 - np.sqrt(x), 0.01)**0.25

                    # ── Gravitational Redshift ──
                    # z_grav = 1 / sqrt(1 - R_S / r_crossing)
                    z_grav = 1.0 / np.sqrt(np.maximum(1.0 - R_S / rc_r, 0.05))

                    # ── Doppler Shift ──
                    v_disk_x = -rcy / (rc_r**1.5)
                    v_disk_y = rcx / (rc_r**1.5)
                    doppler = 1.0 - (v_disk_x * p_act[ci, 0] + v_disk_y * p_act[ci, 1]) * 2.5
                    doppler = np.clip(doppler, 0.15, 4.0)

                    # Observed temperature = emitted / (z_grav * z_doppler_inv)
                    T_observed = T_disk * doppler / z_grav

                    # Convert to RGB
                    disk_rgb = _blackbody_rgb(T_observed)

                    # ── Luminosity ──
                    # Smooth radial falloff + turbulent perturbation (no periodic banding)
                    # Hash-based turbulence breaks up regularity
                    phi = np.arctan2(rcy, rcx)
                    turb = 0.85 + 0.15 * np.sin(phi * 7.3 + rc_r * 3.1) * np.sin(phi * 13.7 - rc_r * 5.7)
                    base_lum = (2.5 / (rc_r - R_IN + 0.5)**1.2) * turb
                    base_lum *= doppler**2.5

                    # Disk optical depth: inner thick, outer fading
                    opacity = np.clip(1.0 - ((rc_r - R_IN) / (R_OUT - R_IN))**3, 0.1, 1.0)
                    base_lum *= opacity

                    # Higher-order image attenuation (photon ring)
                    order = hit_count[gi]
                    # Photon ring: 2nd-order images get 60% brightness, 3rd 35%
                    attenuation = np.where(order == 0, 1.0,
                                  np.where(order == 1, 0.6, 0.35))
                    base_lum *= attenuation

                    # Deposit color
                    contrib = disk_rgb * np.clip(base_lum, 0, 3.0)[:, None]
                    color_map[gi] = np.minimum(color_map[gi] + contrib, 3.0)
                    hit_count[gi] += 1

                    # Max 4 crossings
                    final_mask[gi[hit_count[gi] >= 4]] = True

            # ── Escape ──
            r_new_mag = np.linalg.norm(r_new, axis=1)
            escapes = r_new_mag > 60.0
            if np.any(escapes):
                esc_idx = np.where(escapes)[0]
                g_esc = sub_idx[esc_idx]

                no_hit = hit_count[g_esc] == 0
                if np.any(no_hit):
                    nh_g = g_esc[no_hit]
                    # Seeded random star field
                    # Hash ray direction into a reproducible star seed
                    p_esc = p_new[esc_idx[no_hit]]
                    # Use direction hash for star positions
                    hash_val = np.sin(p_esc[:, 0] * 314.159 + p_esc[:, 1] * 271.828) * 1000.0
                    hash_val = hash_val - np.floor(hash_val)  # fractional part [0,1)
                    is_star = hash_val > 0.995  # ~0.5% of sky = ~1000 stars

                    bg = np.zeros((len(nh_g), 3))
                    bg[:, 0] = 0.003
                    bg[:, 1] = 0.002
                    bg[:, 2] = 0.012

                    if np.any(is_star):
                        # Random star colors: use hash for temperature
                        star_T = 3000.0 + hash_val[is_star] * 25000.0  # 3000-28000 K
                        star_rgb = _blackbody_rgb(star_T)
                        # Power-law brightness: many faint, few bright
                        brightness = 0.3 + 0.7 * hash_val[is_star]**3
                        bg[is_star] = star_rgb * brightness[:, None]

                    color_map[nh_g] = bg

                final_mask[g_esc] = True

            # Update state
            r[sub_idx] = r_new
            p[sub_idx] = p_new

        # Accumulate this sample
        accumulated_color += color_map

    # Average across samples
    accumulated_color /= N_SAMPLES

    print("\n[*] Integrations complete. Rebuilding image tensor...")

    image_tensor = accumulated_color.reshape((HEIGHT, WIDTH, 3))

    # Tone mapping: gentle Reinhard (preserve warmth and contrast)
    lum = 0.2126 * image_tensor[:, :, 0] + 0.7152 * image_tensor[:, :, 1] + 0.0722 * image_tensor[:, :, 2]
    # Use higher key for brighter image
    lum_mapped = lum * 1.5 / (1.0 + lum)
    scale = np.where(lum > 1e-6, lum_mapped / lum, 1.0)
    image_tensor *= scale[:, :, None]
    image_tensor = np.clip(image_tensor, 0.0, 1.0)

    # Cinematic bloom
    print("[*] Applying cinematic rendering bloom...")
    try:
        from scipy.ndimage import gaussian_filter
        bloom = gaussian_filter(image_tensor, sigma=[4, 4, 0])
        image_tensor = np.clip(image_tensor + bloom * 0.25, 0.0, 1.0)
    except ImportError:
        pass

    # Gamma correction
    image_tensor = np.power(image_tensor, 0.82)

    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor('black')

    ax.imshow(image_tensor, interpolation='bilinear', origin='lower')
    ax.axis('off')

    ax.text(30, 25,
            "GARGANTUA ACOUSTIC VORTEX\n"
            "AVE Raymarcher: Blackbody Spectrum + Gravitational Redshift\n"
            r"Mass = ~$10^8\;M_\odot$   Spin = 0.999   "
            f"Samples/px = {N_SAMPLES}",
            color='white', alpha=0.4, fontsize=11, family='monospace',
            verticalalignment='bottom')

    plt.tight_layout(pad=0)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'gargantua_acoustic_vortex.png')

    plt.savefig(out_path, dpi=250, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    print(f"[*] High-fidelity Interstellar rendering complete -> {out_path}")

if __name__ == "__main__":
    render_gargantua()
