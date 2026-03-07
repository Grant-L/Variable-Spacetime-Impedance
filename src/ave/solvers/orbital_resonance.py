"""
Macroscopic Orbital Resonance Solver (Applied Vacuum Engineering)
=================================================================

Models black holes as macroscopic electron orbitals.  The same 1/d mutual
impedance topology that quantises electron shells and carves Saturn ring
gaps produces discrete standing-wave resonance bands around a Kerr black
hole.

Key insight: the electron is a self-trapped photon at ℓ_node scale with
Γ = −1 total reflection.  The black hole event horizon is a Γ = +1
dielectric rupture boundary at r_s scale.  From the exterior, both
present quantised orbital structure radiating outward.

Physical outputs:
    • Quantised impedance-band radii in the accretion disk
    • QPO (Quasi-Periodic Oscillation) frequency predictions
    • Scale-invariance mapping table (electron ↔ black hole)

All constants imported from ave.core.constants — zero free parameters.
"""

import numpy as np
from ave.core.constants import (
    C_0, G, M_E, HBAR, ALPHA, L_NODE, Z_0,
    NU_VAC, ISOTROPIC_PROJECTION, P_C, T_EM,
)

# Alias for readability
G_NEWTON = G

# ---------------------------------------------------------------------------
# Physical constants for astrophysical targets
# ---------------------------------------------------------------------------
M_SUN = 1.989e30          # Solar mass [kg]

# ---------------------------------------------------------------------------
# 1.  Isotropic Schwarzschild Refractive Index
# ---------------------------------------------------------------------------

def refractive_index(r, M):
    """
    Isotropic Schwarzschild refractive index  n(r).

    In isotropic coordinates the half-Schwarzschild radius is
        r_h = G M / (2 c²)

    and the refractive index governing photon propagation is
        n(r) = W³ / U,   W = 1 + r_h/r,   U = 1 − r_h/r

    Parameters
    ----------
    r : array_like   Isotropic radial coordinate [m]
    M : float        Central mass [kg]

    Returns
    -------
    n : ndarray       Local refractive index (dimensionless)
    """
    r = np.asarray(r, dtype=float)
    rh = G_NEWTON * M / (2.0 * C_0**2)
    ratio = rh / np.maximum(r, rh * 1.01)          # clamp at horizon
    W = 1.0 + ratio
    U = np.maximum(1.0 - ratio, 1e-6)
    return W**3 / U


def reflection_coefficient(r, M):
    """
    Radial impedance reflection coefficient  Γ(r) = (n − 1) / (n + 1).

    Γ → 0  in flat space,  Γ → +1  at the event horizon.
    Analogous to the electron's Γ = −1 self-confinement boundary, but
    with inverted sign (outward impedance wall vs. inward confinement).

    Parameters
    ----------
    r : array_like   Isotropic radial coordinate [m]
    M : float        Central mass [kg]

    Returns
    -------
    Gamma : ndarray  Reflection coefficient (dimensionless, 0 ≤ Γ ≤ 1)
    """
    n = refractive_index(r, M)
    return (n - 1.0) / (n + 1.0)


# ---------------------------------------------------------------------------
# 2.  Orbital Keplerian Frequency
# ---------------------------------------------------------------------------

def keplerian_frequency(r, M):
    """
    Keplerian orbital frequency at Schwarzschild coordinate radius r.

    ν_K = (1 / 2π) √(G M / r³)

    Parameters
    ----------
    r : array_like   Schwarzschild radial coordinate [m]
    M : float        Central mass [kg]

    Returns
    -------
    nu : ndarray     Orbital frequency [Hz]
    """
    r = np.asarray(r, dtype=float)
    return (1.0 / (2.0 * np.pi)) * np.sqrt(G_NEWTON * M / r**3)


# ---------------------------------------------------------------------------
# 3.  Characteristic Radii
# ---------------------------------------------------------------------------

def schwarzschild_radius(M):
    """Event horizon radius  r_s = 2 G M / c²."""
    return 2.0 * G_NEWTON * M / C_0**2


def photon_sphere_radius(M):
    """
    Photon sphere radius  r_ph = 3 G M / c².

    This is the "1s orbital" of the black hole — the innermost
    radius where photons can orbit in a circular standing wave.
    """
    return 3.0 * G_NEWTON * M / C_0**2


def isco_radius(M, a_star=0.0):
    """
    Innermost Stable Circular Orbit (ISCO) for a Kerr black hole.

    For Schwarzschild (a_star=0):  r_ISCO = 6 G M / c²  =  3 r_s
    For prograde extremal Kerr (a_star=1):  r_ISCO → r_s/2

    Parameters
    ----------
    M : float       Central mass [kg]
    a_star : float  Dimensionless Kerr spin parameter (0 ≤ a_star ≤ 1)

    Returns
    -------
    r_isco : float  ISCO radius [m]
    """
    rs = schwarzschild_radius(M)
    if abs(a_star) < 1e-10:
        return 3.0 * rs   # 6 GM/c²

    # Bardeen, Press, Teukolsky (1972) exact formulae
    a = a_star
    Z1 = 1.0 + (1.0 - a**2)**(1.0/3.0) * ((1.0 + a)**(1.0/3.0) + (1.0 - a)**(1.0/3.0))
    Z2 = np.sqrt(3.0 * a**2 + Z1**2)
    # Prograde orbit
    r_isco = rs / 2.0 * (3.0 + Z2 - np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)))
    return r_isco


# ---------------------------------------------------------------------------
# 4.  Impedance Band Quantisation  (Standing-Wave Resonance)
# ---------------------------------------------------------------------------

def impedance_orbital_radii(M, a_star=0.0, n_modes=8):
    """
    Quantised orbital radii from standing-wave resonance in the
    refractive gradient n(r).

    The standing-wave condition is:
        ∫_{r_in}^{r_n}  n(r) dr  =  n × λ_fundamental / 2

    where the fundamental wavelength λ₀ is set by the photon sphere
    circumference (the "ground-state" orbital):
        λ₀ = 2π r_ph

    This is the direct macroscopic analogue of electron orbital
    quantisation where standing de Broglie waves fit integer
    wavelengths around the nucleus.

    Parameters
    ----------
    M : float       Central mass [kg]
    a_star : float  Kerr spin parameter
    n_modes : int   Number of resonance modes to compute

    Returns
    -------
    radii : ndarray  Quantised orbital radii [m], shape (n_modes,)
    mode_numbers : ndarray  Integer mode numbers, shape (n_modes,)
    """
    rs = schwarzschild_radius(M)
    r_ph = photon_sphere_radius(M)
    r_isco = isco_radius(M, a_star)

    # Fundamental wavelength: photon sphere circumference
    lambda_0 = 2.0 * np.pi * r_ph

    # Solve standing-wave condition numerically
    # ∫_{r_ph}^{r_n} n(r') dr' = n × λ₀/2
    # Use fine radial grid for numerical integration
    r_max = 30.0 * rs
    N_grid = 10000
    r_grid = np.linspace(r_ph * 1.001, r_max, N_grid)
    n_grid = refractive_index(r_grid, M)
    dr = r_grid[1] - r_grid[0]

    # Cumulative optical path length from photon sphere outward
    optical_path = np.cumsum(n_grid * dr)

    radii = []
    mode_numbers = []
    for mode_n in range(1, n_modes + 1):
        target = mode_n * lambda_0 / 2.0
        idx = np.searchsorted(optical_path, target)
        if idx < N_grid:
            radii.append(r_grid[idx])
            mode_numbers.append(mode_n)

    return np.array(radii), np.array(mode_numbers)


def qpo_frequencies(M, a_star=0.0, n_modes=5):
    """
    Quasi-Periodic Oscillation (QPO) frequencies from impedance
    band resonance.

    QPO frequency for mode n is the Keplerian orbital frequency at
    the n-th impedance resonance radius.  The frequency RATIOS
    between adjacent modes are the key observable.

    Parameters
    ----------
    M : float       Central mass [kg]
    a_star : float  Kerr spin parameter
    n_modes : int   Number of QPO modes

    Returns
    -------
    frequencies : ndarray   QPO frequencies [Hz], shape (n_modes,)
    radii : ndarray         Resonance radii [m], shape (n_modes,)
    ratios : ndarray        Frequency ratios ν_n / ν_1, shape (n_modes,)
    """
    radii, modes = impedance_orbital_radii(M, a_star, n_modes)
    freqs = keplerian_frequency(radii, M)
    ratios = freqs / freqs[0] if len(freqs) > 0 else np.array([])
    return freqs, radii, ratios


# ---------------------------------------------------------------------------
# 5.  Scale-Invariance Table
# ---------------------------------------------------------------------------

def scale_invariance_table():
    """
    Generate the electron ↔ black hole isomorphism mapping table.

    Returns
    -------
    table : list of dicts   Each dict contains:
        'property', 'electron', 'black_hole', 'relation'
    """
    rs_sun = schwarzschild_radius(M_SUN)

    table = [
        {
            'property': 'Confinement Boundary',
            'electron': f'ℓ_node = {L_NODE:.4e} m',
            'black_hole': f'r_s = 2GM/c² ({rs_sun:.2e} m for 1 M☉)',
            'relation': 'Both are Γ = ±1 total reflection boundaries',
        },
        {
            'property': 'Confinement Mechanism',
            'electron': 'Self-trapped photon (helix → unknot)',
            'black_hole': 'Dielectric rupture (n → ∞)',
            'relation': 'Both are impedance catastrophes',
        },
        {
            'property': '"Ground-State" Orbital',
            'electron': 'Bohr radius a₀ = ℓ_node / α',
            'black_hole': 'Photon sphere r_ph = 3GM/c²',
            'relation': 'Innermost stable circular standing wave',
        },
        {
            'property': 'Orbital Quantisation',
            'electron': 'de Broglie λ = 2πr/n  (integer standing waves)',
            'black_hole': '∫n(r)dr = nλ₀/2  (impedance band resonance)',
            'relation': 'Same standing-wave topology at different scales',
        },
        {
            'property': 'Shell Gaps',
            'electron': 'Spectral lines (forbidden transitions)',
            'black_hole': 'Accretion disk QPOs',
            'relation': '1/d impedance mismatch frequencies',
        },
        {
            'property': 'Interior Physics',
            'electron': 'Constructive: topology preserved',
            'black_hole': 'Destructive: topology melts (phase transition)',
            'relation': 'Exterior identical; interior inverted',
        },
        {
            'property': 'Impedance',
            'electron': f'Z₀ = {Z_0:.2f} Ω (invariant)',
            'black_hole': 'Z(r) → ∞ at horizon (open circuit)',
            'relation': 'Achromatic vs. catastrophic impedance',
        },
    ]
    return table


# ---------------------------------------------------------------------------
# 6.  Console Report
# ---------------------------------------------------------------------------

def print_report(M_solar=10.0, a_star=0.0):
    """
    Print a diagnostic report for a given black hole.

    Parameters
    ----------
    M_solar : float   Mass in solar masses
    a_star : float    Kerr spin parameter
    """
    M = M_solar * M_SUN
    rs = schwarzschild_radius(M)
    rph = photon_sphere_radius(M)
    risco = isco_radius(M, a_star)

    print("=" * 70)
    print("  BLACK HOLE AS MACROSCOPIC ELECTRON ORBITAL")
    print("  AVE Impedance Resonance Solver")
    print("=" * 70)
    print(f"\n  Mass:  {M_solar:.1f} M☉  =  {M:.3e} kg")
    print(f"  Spin:  a* = {a_star}")
    print(f"\n  Schwarzschild radius:   r_s   = {rs:.3e} m")
    print(f"  Photon sphere ('1s'):   r_ph  = {rph:.3e} m  = {rph/rs:.2f} r_s")
    print(f"  ISCO ('ground state'):  r_ISCO = {risco:.3e} m  = {risco/rs:.2f} r_s")

    print(f"\n{'─' * 70}")
    print("  QUANTISED IMPEDANCE BAND RADII (Standing-Wave Resonance)")
    print(f"{'─' * 70}")

    freqs, radii, ratios = qpo_frequencies(M, a_star, n_modes=6)

    print(f"  {'Mode':>4s}  {'Radius [m]':>14s}  {'r/r_s':>8s}  {'ν_QPO [Hz]':>12s}  {'ν/ν₁':>6s}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*8}  {'─'*12}  {'─'*6}")
    for i in range(len(freqs)):
        print(f"  {i+1:4d}  {radii[i]:14.4e}  {radii[i]/rs:8.3f}  {freqs[i]:12.4e}  {ratios[i]:6.3f}")

    # Highlight the 3:2 ratio if present
    if len(ratios) >= 2:
        r21 = ratios[1] / ratios[0] if ratios[0] > 0 else 0
        print(f"\n  ν₂/ν₁ ratio = {r21:.4f}")
        if abs(r21 - 1.5) < 0.1:
            print("  ★ Close to 3:2 ratio observed in X-ray binary QPOs!")

    print(f"\n{'─' * 70}")
    print("  SCALE-INVARIANCE TABLE: Electron ↔ Black Hole")
    print(f"{'─' * 70}")
    for row in scale_invariance_table():
        print(f"\n  {row['property']}")
        print(f"    Electron:    {row['electron']}")
        print(f"    Black Hole:  {row['black_hole']}")
        print(f"    Relation:    {row['relation']}")

    print(f"\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # GRS 1915+105: ~14 M☉ stellar-mass black hole with known 67/113 Hz QPOs
    print("\n" + "▓" * 70)
    print("  TARGET: GRS 1915+105 (Stellar-Mass X-ray Binary)")
    print("▓" * 70)
    print_report(M_solar=14.0, a_star=0.7)

    # Sgr A*: ~4 × 10⁶ M☉ supermassive black hole
    print("\n" + "▓" * 70)
    print("  TARGET: Sgr A* (Galactic Centre Supermassive)")
    print("▓" * 70)
    print_report(M_solar=4.0e6, a_star=0.5)
