"""
Planetary Magnetospheres: Impedance Profiles of Gas Giant Magnetic Fields
=========================================================================

Models planetary magnetospheres as impedance cavities in the solar wind.
Each planet's magnetic field creates a magnetopause: the boundary where
magnetic pressure balances solar wind dynamic pressure.

In AVE terms:
  - Magnetic field B(r) creates impedance Z_B = B/√(μ₀ρ)
  - Solar wind has impedance Z_sw = ρ_sw · v_sw
  - Magnetopause = impedance boundary where Z_B ≈ Z_sw
  - Γ at magnetopause determines what fraction of solar wind
    energy penetrates vs reflects

Key anomaly — Uranus:
  - Magnetic axis tilted 59° from rotation axis
  - Dipole offset by 0.3 R_U from center
  - Creates an asymmetric, time-varying impedance cavity
  - Unique in solar system: all other planets have <12° tilt

This module predicts:
  1. Magnetopause standoff distances from Z balance
  2. Uranus asymmetric Γ profile (day vs night side)
  3. Comparative impedance spectra: Earth, Jupiter, Saturn, Uranus, Neptune
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ave.core.constants import G, C_0, Z_0, MU_0
from ave.axioms.scale_invariant import reflection_coefficient


# ═══════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════

AU = 1.496e11           # Astronomical unit [m]
M_P = 1.6726e-27        # Proton mass [kg]


# ═══════════════════════════════════════════════════════════════
# Planet data
# ═══════════════════════════════════════════════════════════════

@dataclass
class PlanetMagnetosphere:
    """Magnetic and orbital properties of a planet."""
    name: str
    mass_kg: float              # Planet mass
    radius_m: float             # Equatorial radius
    a_orbital_au: float         # Orbital semi-major axis
    B_equatorial_T: float       # Surface equatorial magnetic field [T]
    dipole_tilt_deg: float      # Angle between magnetic and rotation axes
    dipole_offset_frac: float   # Dipole center offset / planet radius
    rotation_period_hr: float   # Sidereal rotation period

    @property
    def dipole_tilt_rad(self) -> float:
        return np.radians(self.dipole_tilt_deg)

    @property
    def dipole_moment(self) -> float:
        """Magnetic dipole moment [T·m³]."""
        return self.B_equatorial_T * self.radius_m**3


# Measured values from Voyager, Galileo, Cassini, ground-based
EARTH = PlanetMagnetosphere(
    name="Earth",
    mass_kg=5.972e24, radius_m=6.371e6,
    a_orbital_au=1.0,
    B_equatorial_T=3.12e-5,    # ~31.2 μT
    dipole_tilt_deg=11.5,
    dipole_offset_frac=0.07,
    rotation_period_hr=23.93,
)

JUPITER = PlanetMagnetosphere(
    name="Jupiter",
    mass_kg=1.898e27, radius_m=7.149e7,
    a_orbital_au=5.20,
    B_equatorial_T=4.28e-4,    # ~428 μT (strongest in solar system)
    dipole_tilt_deg=9.6,
    dipole_offset_frac=0.13,
    rotation_period_hr=9.93,
)

SATURN = PlanetMagnetosphere(
    name="Saturn",
    mass_kg=5.683e26, radius_m=6.027e7,
    a_orbital_au=9.58,
    B_equatorial_T=2.1e-5,     # ~21 μT
    dipole_tilt_deg=0.0,       # Nearly zero! (< 0.06°)
    dipole_offset_frac=0.04,
    rotation_period_hr=10.66,
)

URANUS = PlanetMagnetosphere(
    name="Uranus",
    mass_kg=8.681e25, radius_m=2.556e7,
    a_orbital_au=19.22,
    B_equatorial_T=2.3e-5,     # ~23 μT
    dipole_tilt_deg=59.0,      # THE ANOMALY: 59° tilt!
    dipole_offset_frac=0.31,   # Offset by 0.31 R_U from center
    rotation_period_hr=17.24,
)

NEPTUNE = PlanetMagnetosphere(
    name="Neptune",
    mass_kg=1.024e26, radius_m=2.476e7,
    a_orbital_au=30.07,
    B_equatorial_T=1.4e-5,     # ~14 μT
    dipole_tilt_deg=47.0,      # Also highly tilted (47°)
    dipole_offset_frac=0.55,   # Most offset in solar system
    rotation_period_hr=16.11,
)

ALL_PLANETS = [EARTH, JUPITER, SATURN, URANUS, NEPTUNE]


# ═══════════════════════════════════════════════════════════════
# Magnetospheric impedance
# ═══════════════════════════════════════════════════════════════

def dipole_field(planet: PlanetMagnetosphere, r_m: float,
                 theta_deg: float = 0.0) -> float:
    """
    Magnetic field magnitude from an offset tilted dipole.

    For a centred dipole: B(r,θ) = (μ₀ M / 4π r³) √(1 + 3cos²θ)
    With offset: effective r is shifted from planet center.

    Args:
        planet: Planet properties.
        r_m: Distance from planet center [m].
        theta_deg: Magnetic colatitude [degrees] (0 = pole, 90 = equator).

    Returns:
        Magnetic field strength [T].
    """
    # Offset correction
    offset_m = planet.dipole_offset_frac * planet.radius_m
    # Effective distance from dipole center (simplified 1D projection)
    r_eff = max(r_m - offset_m, planet.radius_m * 0.5)

    theta = np.radians(theta_deg)
    # Dipole field formula
    B = planet.B_equatorial_T * (planet.radius_m / r_eff)**3 * \
        np.sqrt(1 + 3 * np.cos(theta)**2) / 2.0
    return B


def magnetic_pressure(B_T: float) -> float:
    """
    Magnetic pressure from field strength.

    P_B = B² / (2μ₀)

    Args:
        B_T: Magnetic field [T].

    Returns:
        Magnetic pressure [Pa].
    """
    return B_T**2 / (2 * MU_0)


def solar_wind_dynamic_pressure(r_au: float) -> float:
    """
    Solar wind dynamic pressure at distance r from Sun.

    P_sw = ½ ρ v² where ρ = n_p · m_p, n_p ∝ 1/r², v ≈ 400 km/s

    At 1 AU: P_sw ≈ 2.2 nPa

    Args:
        r_au: Distance from Sun [AU].

    Returns:
        Dynamic pressure [Pa].
    """
    n_p_1au = 5e6          # protons/m³ at 1 AU
    v_sw = 400e3           # m/s
    n_p = n_p_1au / r_au**2
    return 0.5 * n_p * M_P * v_sw**2


def magnetopause_standoff(planet: PlanetMagnetosphere) -> float:
    """
    Magnetopause standoff distance from pressure balance.

    At the magnetopause: P_B(r_mp) = P_sw
    B²(r_mp)/(2μ₀) = ½ ρ_sw v_sw²

    For a dipole: B ∝ 1/r³, so r_mp ∝ (M/P_sw)^(1/6)

    Args:
        planet: Planet properties.

    Returns:
        Magnetopause standoff distance [m].
    """
    P_sw = solar_wind_dynamic_pressure(planet.a_orbital_au)
    # Solve B(r_mp)²/(2μ₀) = P_sw for r_mp
    # B(r) = B_eq × (R_p/r)³ × (1/2) at equator
    B_eq_factor = planet.B_equatorial_T / 2.0  # Equatorial factor
    # B(r)² = (B_eq/2)² × (R/r)⁶
    # P_B = B²/(2μ₀) = (B_eq/2)²(R/r)⁶ / (2μ₀) = P_sw
    # (R/r)⁶ = P_sw × 2μ₀ / (B_eq/2)²
    ratio6 = P_sw * 2 * MU_0 / B_eq_factor**2
    if ratio6 <= 0:
        return planet.radius_m * 100  # Fallback
    r_mp = planet.radius_m / ratio6**(1.0/6.0)
    return r_mp


def magnetopause_standoff_Rp(planet: PlanetMagnetosphere) -> float:
    """Standoff distance in units of planet radii."""
    return magnetopause_standoff(planet) / planet.radius_m


def magnetic_impedance(B_T: float, rho_kg_m3: float) -> float:
    """
    Alfvén wave impedance of a magnetized plasma.

    Z_A = B / √(μ₀ ρ)  [Ω equivalent for energy flux]

    This is the impedance "seen" by incoming solar wind.

    Args:
        B_T: Magnetic field [T].
        rho_kg_m3: Mass density [kg/m³].

    Returns:
        Alfvén impedance [m/s] (V_A = B/√(μ₀ρ)).
    """
    return B_T / np.sqrt(MU_0 * rho_kg_m3)


def magnetopause_reflection(planet: PlanetMagnetosphere) -> float:
    """
    Reflection coefficient at the magnetopause.

    Using the universal reflection_coefficient function:
    Z₁ = solar wind impedance, Z₂ = magnetospheric Alfvén impedance

    Args:
        planet: Planet properties.

    Returns:
        Reflection coefficient Γ at magnetopause.
    """
    r_mp = magnetopause_standoff(planet)

    # Solar wind impedance at magnetopause
    n_sw = 5e6 / planet.a_orbital_au**2  # protons/m³
    rho_sw = n_sw * M_P
    v_sw = 400e3
    Z_sw = rho_sw * v_sw  # Dynamic impedance [kg/(m²·s)]

    # Magnetospheric Alfvén impedance at magnetopause
    B_mp = dipole_field(planet, r_mp, theta_deg=90)  # Equatorial
    Z_mag = magnetic_impedance(B_mp, rho_sw * 0.1)  # Reduced density inside

    return float(reflection_coefficient(Z_sw, Z_mag))


def uranus_asymmetric_profile(n_points: int = 360) -> dict:
    """
    Uranus magnetopause profile as function of magnetic longitude.

    Due to the 59° dipole tilt AND 0.31 R_U offset, Uranus's
    magnetopause is highly asymmetric — the standoff distance
    varies dramatically with rotational phase.

    Returns:
        Dict with 'longitude_deg', 'r_mp_Rp' (standoff in R_U),
        'B_surface_T', 'Gamma'.
    """
    longitudes = np.linspace(0, 360, n_points, endpoint=False)
    r_mp = np.zeros(n_points)
    B_surf = np.zeros(n_points)
    Gamma = np.zeros(n_points)

    P_sw = solar_wind_dynamic_pressure(URANUS.a_orbital_au)

    for i, lon in enumerate(longitudes):
        # Magnetic colatitude varies with longitude due to tilt
        # At longitude φ, the sub-solar magnetic colatitude is:
        # θ_mag = arccos(cos(tilt) × cos(φ))  (simplified)
        theta_mag = np.degrees(np.arccos(
            np.cos(URANUS.dipole_tilt_rad) * np.cos(np.radians(lon))
        ))

        # Offset projection along sub-solar line
        offset_proj = URANUS.dipole_offset_frac * URANUS.radius_m * \
                       np.cos(np.radians(lon - 30))  # Offset azimuth ~30°

        # Effective surface B at this longitude
        B_surf[i] = dipole_field(URANUS, URANUS.radius_m + abs(offset_proj),
                                 theta_deg=theta_mag)

        # Find standoff where B²/(2μ₀) = P_sw
        B_factor = B_surf[i] / 2.0
        if B_factor > 0:
            ratio6 = P_sw * 2 * MU_0 / B_factor**2
            r_mp_m = URANUS.radius_m / ratio6**(1.0/6.0) if ratio6 > 0 else 50 * URANUS.radius_m
        else:
            r_mp_m = 50 * URANUS.radius_m
        r_mp[i] = r_mp_m / URANUS.radius_m

        # Reflection at this point
        n_sw = 5e6 / URANUS.a_orbital_au**2
        rho_sw = n_sw * M_P
        Z_sw = rho_sw * 400e3
        B_mp = dipole_field(URANUS, r_mp_m, theta_deg=theta_mag)
        Z_mag = magnetic_impedance(B_mp, rho_sw * 0.1) if B_mp > 0 else 1.0
        Gamma[i] = float(reflection_coefficient(Z_sw, Z_mag))

    return {
        'longitude_deg': longitudes,
        'r_mp_Rp': r_mp,
        'B_surface_T': B_surf,
        'Gamma': Gamma,
        'asymmetry_ratio': np.max(r_mp) / np.min(r_mp),
    }


def comparative_magnetosphere_table() -> list:
    """
    Compare all gas giant magnetospheres.

    Returns:
        List of dicts with planet name, standoff, Γ, dipole properties.
    """
    results = []

    # Observed standoff distances [R_p] from spacecraft
    observed_standoff = {
        'Earth': 10.0,
        'Jupiter': 63.0,
        'Saturn': 22.0,
        'Uranus': 25.0,
        'Neptune': 26.0,
    }

    for planet in ALL_PLANETS:
        r_mp_Rp = magnetopause_standoff_Rp(planet)
        Gamma = magnetopause_reflection(planet)
        obs = observed_standoff.get(planet.name, None)

        results.append({
            'name': planet.name,
            'B_eq_uT': planet.B_equatorial_T * 1e6,
            'dipole_tilt_deg': planet.dipole_tilt_deg,
            'dipole_offset_frac': planet.dipole_offset_frac,
            'r_standoff_Rp': r_mp_Rp,
            'r_observed_Rp': obs,
            'error_pct': abs(r_mp_Rp - obs) / obs * 100 if obs else None,
            'Gamma_magnetopause': Gamma,
            'dipole_moment_Tm3': planet.dipole_moment,
            'symmetry': 'symmetric' if planet.dipole_tilt_deg < 15 else 'asymmetric',
        })

    return results
