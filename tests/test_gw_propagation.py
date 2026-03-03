"""
Test suite for Gravitational Wave Propagation (Gap 2).

Verifies:
  1. Schwarzschild radius matches GR
  2. Event horizon: Z → 0, Γ → −1 (same as Pauli, plasma, Meissner)
  3. GW strain is far below saturation (linear propagation)
  4. Refractive index > 1 near mass (gravitational lensing)
  5. Black hole echo delay is physically reasonable
"""

import numpy as np
import pytest

from ave.gravity.gw_propagation import (
    schwarzschild_radius,
    epsilon_eff_schwarzschild,
    mu_eff_schwarzschild,
    gravitational_impedance,
    horizon_reflection,
    gw_strain_to_voltage,
    is_linear_propagation,
    gw_local_speed,
    refractive_index,
    echo_delay,
    gw_propagation_summary,
)
from ave.core.constants import C_0, EPSILON_0, MU_0, Z_0, G, V_SNAP


M_SUN = 1.989e30  # Solar mass [kg]


class TestSchwarzschildRadius:
    """r_s = 2GM/c² must match GR."""

    def test_sun(self):
        """For the Sun: r_s ≈ 2.95 km."""
        r_s = schwarzschild_radius(M_SUN)
        assert r_s == pytest.approx(2953, rel=0.01)

    def test_30_solar(self):
        """For a 30 M☉ black hole: r_s ≈ 88.6 km."""
        r_s = schwarzschild_radius(30 * M_SUN)
        assert r_s == pytest.approx(88600, rel=0.01)

    def test_proportional(self):
        """r_s scales linearly with M."""
        assert schwarzschild_radius(2 * M_SUN) == pytest.approx(
            2 * schwarzschild_radius(M_SUN), rel=1e-10)


class TestSchwarzschildImpedance:
    """Z(r) → 0 at the horizon, Z(r) → Z₀ at infinity."""

    def test_far_field_vacuum(self):
        """Far from mass: ε → ε₀, μ → μ₀, Z → Z₀."""
        r_s = schwarzschild_radius(30 * M_SUN)
        r = 1e6 * r_s  # Very far
        assert float(epsilon_eff_schwarzschild(r, r_s)) == pytest.approx(
            EPSILON_0, rel=1e-4)
        assert float(mu_eff_schwarzschild(r, r_s)) == pytest.approx(
            MU_0, rel=1e-4)
        assert float(gravitational_impedance(r, r_s)) == pytest.approx(
            Z_0, rel=1e-4)

    def test_near_horizon_z_drops(self):
        """Near horizon: Z << Z₀."""
        r_s = schwarzschild_radius(30 * M_SUN)
        r = 1.01 * r_s
        Z = float(gravitational_impedance(r, r_s))
        assert Z < 0.1 * Z_0

    def test_horizon_gamma_minus_one(self):
        """At horizon: Γ → −1 (total reflection)."""
        r_s = schwarzschild_radius(30 * M_SUN)
        r = 1.001 * r_s
        gamma = float(horizon_reflection(r, r_s))
        assert gamma < -0.95

    def test_far_field_gamma_zero(self):
        """Far away: Γ → 0 (matched — flat space)."""
        r_s = schwarzschild_radius(30 * M_SUN)
        r = 1e6 * r_s
        gamma = float(horizon_reflection(r, r_s))
        assert abs(gamma) < 0.001


class TestGWLinearPropagation:
    """LIGO GW must be in the linear regime (no saturation)."""

    def test_ligo_strain_is_linear(self):
        """h = 10⁻²¹ at 100 Hz must be linear."""
        assert is_linear_propagation(1e-21, 100.0)

    def test_strain_voltage_is_tiny(self):
        """V_GW / V_SNAP ~ 10⁻¹⁹ for LIGO GW."""
        V_gw = gw_strain_to_voltage(1e-21, 100.0)
        ratio = V_gw / V_SNAP
        assert ratio < 1e-10  # Many orders of magnitude below saturation

    def test_gw_always_below_saturation(self):
        """Even h = 1 produces V_gw << V_SNAP — GW can NEVER saturate."""
        # This is a key AVE prediction: gravitational waves operate
        # in a fundamentally different regime than EM waves. Even
        # impossibly large strain h=1 is still linear.
        V_gw = gw_strain_to_voltage(1.0, 100.0)
        ratio = V_gw / V_SNAP
        assert ratio < 1e-3, f"V_gw/V_SNAP = {ratio:.2e}, expected << 1"


class TestRefractiveIndex:
    """Gravity well must have n > 1 (lensing)."""

    def test_far_field_n_equals_one(self):
        """Far from mass: n → 1 (flat space)."""
        r_s = schwarzschild_radius(30 * M_SUN)
        n = float(refractive_index(1e12 * r_s, r_s))
        assert n == pytest.approx(1.0, abs=1e-6)

    def test_near_mass_n_greater_than_one(self):
        """Near mass: n > 1 (light bends)."""
        r_s = schwarzschild_radius(30 * M_SUN)
        n = float(refractive_index(10 * r_s, r_s))
        assert n > 1.0

    def test_monotonically_increasing_inward(self):
        """n increases as r decreases (stronger lensing)."""
        r_s = schwarzschild_radius(30 * M_SUN)
        r = np.array([100, 50, 20, 10, 5]) * r_s
        n = refractive_index(r, r_s)
        assert np.all(np.diff(n) > 0)  # n increases as r decreases


class TestBlackHoleEchoes:
    """Echo delay must be positive and physically reasonable."""

    def test_echo_delay_positive(self):
        """Echo delay must be > 0."""
        dt = echo_delay(30 * M_SUN)
        assert dt > 0

    def test_echo_delay_30_solar(self):
        """For 30 M☉: Δt ~ 0.001 - 0.1 s."""
        dt = echo_delay(30 * M_SUN)
        assert 1e-4 < dt < 1.0

    def test_echo_scales_with_mass(self):
        """Larger mass → longer echo delay."""
        dt_30 = echo_delay(30 * M_SUN)
        dt_60 = echo_delay(60 * M_SUN)
        assert dt_60 > dt_30


class TestSummary:
    """Summary function should produce complete output."""

    def test_summary_runs(self):
        """Summary should run without errors."""
        result = gw_propagation_summary(30.0, 1e-21)
        assert result['linear_propagation'] is True
        assert len(result['profiles']) > 0
        assert result['r_s_m'] > 0
