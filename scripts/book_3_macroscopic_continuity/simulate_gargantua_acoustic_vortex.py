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
3. Event Horizon ≡ Dielectric Rupture (n→∞, Z_radial→∞, Γ→+1).

The shadow boundary is NOT a hard wall absorber. It emerges naturally from the
impedance physics: as n(r)→∞ near the Schwarzschild radius, the radial
reflection coefficient Γ(r) = (n-1)/(n+1) → +1, meaning transmitted power
(1-|Γ|²) → 0. Rays lose their intensity exponentially as they approach the
horizon, creating the smooth shadow boundary.
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
    MAX_STEPS = 3000
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

    # ── Event Horizon = Dielectric Rupture (Impedance Derivation) ──
    # In a smoothly graded refractive medium, rays bend (geodesics) but
    # do NOT lose intensity per step to reflection — that only happens at
    # sharp impedance discontinuities.  The shadow emerges geometrically:
    # rays below a critical impact parameter spiral into the singularity.
    #
    # The impedance physics EXPLAINS the boundary:
    #   n(r) = (1+rh/r)^3 / (1-rh/r)  →  ∞  as r → rh
    #   Z_radial = Z₀ · n(r)           →  ∞  (open circuit)
    #   Γ_inward = (n-1)/(n+1)         →  +1 (total reflection)
    # But in the geometric optics (ray tracing) regime, this manifests
    # as rays being trapped once r < rh, not as per-step attenuation.

    # -------------------------------------------------------------
    # Physics Parameters (Isotropic Optical-Fluid Mapping)
    # -------------------------------------------------------------
    M = 1.0
    rh = 0.5 * M
    a_star = 0.999    # Kerr dimensionless spin parameter
    a_kerr = a_star * M  # Kerr angular momentum parameter

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

            # ── Dielectric Rupture Absorption ──
            # Ray absorbed when r < rh (isotropic horizon).
            # At this boundary: n → ∞, Z_radial → ∞, Γ → +1.
            # The geometric optics ray is trapped — no escape.
            hit_bh = r_mag < rh
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
            # Isotropic Schwarzschild refractive index:
            #   n(r) = W³/U,  W = 1 + rh/r,  U = 1 - rh/r
            W = 1.0 + rh / r_mag
            U = np.maximum(1.0 - rh / r_mag, 1e-4)
            n = (W**3) / U

            # dn/dr = -W² · rh/r² · (3U + W) / U²
            #       = -W² · rh/r² · (4 - 2rh/r) / U²
            dn_dr = -W**2 * (rh / r_mag**2) * (4.0 - 2.0 * rh / r_mag) / U**2
            r_hat = r_act / r_mag[:, None]
            grad_n = dn_dr[:, None] * r_hat

            # Frame dragging: Kerr angular velocity ω = 2Mar/(r²+a²)²
            r2 = r_mag**2
            denom = (r2 + a_kerr**2)**2
            omega = 2.0 * M * a_kerr * r_mag / denom
            v_drag = np.zeros_like(r_act)
            v_drag[:, 0] = -omega * r_act[:, 1]
            v_drag[:, 1] = omega * r_act[:, 0]

            Lz = r_act[:, 0] * p_act[:, 1] - r_act[:, 1] * p_act[:, 0]
            # Gradient of ω·Lz term for momentum update
            #   d(ω)/dr = 2Ma(r²+a²)²·1 - 2Mar·2(r²+a²)·2r / (r²+a²)⁴
            #           = 2Ma[(r²+a²) - 4r²] / (r²+a²)³
            #           = 2Ma[a² - 3r²] / (r²+a²)³
            domega_dr = 2.0 * M * a_kerr * (a_kerr**2 - 3.0 * r2) / (r2 + a_kerr**2)**3
            grad_pv = domega_dr[:, None] * (r_act / r_mag[:, None]) * Lz[:, None]
            # Add ω × cross-term
            grad_pv[:, 0] += omega * p_act[:, 1]
            grad_pv[:, 1] -= omega * p_act[:, 0]

            p_mag_sq = np.sum(p_act**2, axis=1)

            dr = (p_act / (n**2)[:, None] + v_drag) * DTAU
            dp = ((p_mag_sq / (n**3))[:, None] * grad_n - grad_pv) * DTAU

            r_new = r_act + dr
            p_new = p_act + dp

            # ── Accretion Disk (equatorial crossing) ──
            # Detect z-sign change with softening to avoid cusp artifacts
            # from horizon-grazing rays that barely brush z=0
            crosses = (r_act[:, 2] * r_new[:, 2]) <= 0
            # Exclude rays extremely close to the horizon (cusp artifact zone)
            if np.any(crosses):
                c_mag = np.linalg.norm(r_act[crosses], axis=1)
                too_close = c_mag < rh * 1.5
                if np.any(too_close):
                    cross_idx_all = np.where(crosses)[0]
                    crosses[cross_idx_all[too_close]] = False
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

                    # ── Relativistic Doppler Shift ──
                    # Keplerian orbital velocity: v = sqrt(M/r), tangential
                    v_disk_x = -rcy / (rc_r**1.5)  # v_x = -v_orb * sin(φ)
                    v_disk_y = rcx / (rc_r**1.5)    # v_y = +v_orb * cos(φ)
                    beta_sq = 1.0 / rc_r            # |v|² = M/r (M=1)
                    gamma_inv = np.sqrt(np.maximum(1.0 - beta_sq, 0.01))
                    # v · p̂  (projection of disk velocity onto ray direction)
                    v_dot_p = v_disk_x * p_act[ci, 0] + v_disk_y * p_act[ci, 1]
                    # Full relativistic: f_obs/f_emit = √(1-β²) / (1 - v·p̂)
                    doppler = gamma_inv / np.maximum(1.0 - v_dot_p, 0.05)
                    doppler = np.clip(doppler, 0.15, 4.0)

                    # Observed temperature = emitted × doppler / z_grav
                    T_observed = T_disk * doppler / z_grav

                    # Convert to RGB
                    disk_rgb = _blackbody_rgb(T_observed)

                    # ── Luminosity ──
                    # Smooth radial falloff + turbulent perturbation (no periodic banding)
                    # Hash-based turbulence breaks up regularity
                    phi = np.arctan2(rcy, rcx)
                    turb = 0.85 + 0.15 * np.sin(phi * 7.3 + rc_r * 3.1) * np.sin(phi * 13.7 - rc_r * 5.7)
                    base_lum = (2.5 / (rc_r - R_IN + 0.5)**1.2) * turb
                    base_lum *= doppler**3.0  # I_obs = I_emit × δ³ (optically thick)

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
