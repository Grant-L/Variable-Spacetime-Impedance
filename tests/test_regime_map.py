"""
Tests for the Universal Regime Map
===================================
Verifies that every domain classifier maps known physical objects
to the correct regime.
"""
import pytest
import numpy as np
from ave.core.regime_map import (
    classify_regime,
    em_voltage_regime, em_field_regime,
    gravity_regime, bcs_regime, magnetic_regime,
    nuclear_regime, gw_regime, protein_regime,
    galactic_regime,
    REGIME_LINEAR, REGIME_NONLINEAR, REGIME_YIELD, REGIME_RUPTURED,
    regime_equations,
)


class TestClassifyRegime:
    """Core classification function."""

    def test_linear(self):
        info = classify_regime(0.05, 1.0)
        assert info.regime == REGIME_LINEAR
        assert info.S > 0.99

    def test_nonlinear(self):
        info = classify_regime(0.5, 1.0)
        assert info.regime == REGIME_NONLINEAR
        assert 0.4 < info.S < 1.0

    def test_yield(self):
        info = classify_regime(0.95, 1.0)
        assert info.regime == REGIME_YIELD
        assert info.S < 0.4

    def test_ruptured(self):
        info = classify_regime(1.5, 1.0)
        assert info.regime == REGIME_RUPTURED
        assert info.S == 0.0

    def test_exact_boundary(self):
        info = classify_regime(1.0, 1.0)
        assert info.regime == REGIME_RUPTURED

    def test_zero_amplitude(self):
        info = classify_regime(0.0, 1.0)
        assert info.regime == REGIME_LINEAR
        assert info.S == 1.0


class TestEMDomain:
    """Electromagnetic regime classification."""

    def test_lab_voltage(self):
        info = em_voltage_regime(1000)  # 1kV
        assert info.regime == REGIME_LINEAR

    def test_ponder05_30kv(self):
        info = em_voltage_regime(30e3)
        assert info.regime == REGIME_NONLINEAR
        assert abs(info.r - 0.687) < 0.01

    def test_ponder05_43kv(self):
        info = em_voltage_regime(43e3)
        assert info.regime == REGIME_YIELD

    def test_lab_efield(self):
        info = em_field_regime(1e6)  # 1 MV/m
        assert info.regime == REGIME_LINEAR


class TestGravityDomain:
    """Gravitational regime classification."""

    def test_solar_surface(self):
        M_sun = 1.989e30
        r_sun = 6.96e8
        info = gravity_regime(M_sun, r_sun)
        assert info.regime == REGIME_LINEAR
        assert info.r < 1e-4

    def test_neutron_star(self):
        """A 1.4 M_sun NS at 10km: ε₁₁ = 7GM/(c²r) ≈ 1.46 → Regime IV.
        In AVE, the NS surface is INSIDE the saturation boundary, which
        is the AVE analog of the Buchdahl limit."""
        M_ns = 2.8e30  # ~1.4 M_sun
        r_ns = 1e4      # 10 km
        info = gravity_regime(M_ns, r_ns)
        assert info.regime == REGIME_RUPTURED
        assert info.r > 1.0

    def test_black_hole_horizon(self):
        """At r = r_s = 2GM/c², ε₁₁ = 7GM/(c²r) = 7/2 = 3.5 → Regime IV."""
        M = 10 * 1.989e30  # 10 solar masses
        G = 6.67430e-11
        r_s = 2 * G * M / (3e8)**2
        info = gravity_regime(M, r_s)
        assert info.regime == REGIME_RUPTURED


class TestMagneticDomain:

    def test_lab_magnet(self):
        info = magnetic_regime(10)  # 10 T
        assert info.regime == REGIME_LINEAR

    def test_magnetar(self):
        info = magnetic_regime(1e10)
        assert info.regime == REGIME_RUPTURED


class TestGWDomain:

    def test_ligo(self):
        info = gw_regime(1e-21)
        assert info.regime == REGIME_LINEAR

    def test_ns_merger(self):
        info = gw_regime(0.01)
        assert info.regime == REGIME_NONLINEAR


class TestBCSDomain:

    def test_below_tc(self):
        info = bcs_regime(4.0, 9.2)  # Nb at 4K (Tc=9.2K)
        assert info.regime == REGIME_NONLINEAR

    def test_at_tc(self):
        info = bcs_regime(9.1, 9.2)  # Just below Tc
        assert info.regime == REGIME_YIELD


class TestRegimeEquations:

    def test_linear_equations(self):
        eqs = regime_equations(REGIME_LINEAR)
        assert "ε_eff" in eqs
        assert "ε₀" in eqs["ε_eff"][0]

    def test_nonlinear_equations(self):
        eqs = regime_equations(REGIME_NONLINEAR)
        assert "√(1 - r²)" in eqs["ε_eff"][0]

    def test_all_regimes_valid(self):
        for r in [REGIME_LINEAR, REGIME_NONLINEAR, REGIME_YIELD, REGIME_RUPTURED]:
            eqs = regime_equations(r)
            assert len(eqs) == 5


class TestSummaryOutput:

    def test_summary_string(self):
        info = em_voltage_regime(30e3)
        s = info.summary()
        assert "NONLINEAR" in s
        assert "30000" in s or "3.0000e+04" in s
