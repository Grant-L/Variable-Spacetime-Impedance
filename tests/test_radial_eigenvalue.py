"""
Tests for the radial eigenvalue solver (ABCD cascade, E2d-ii through E2i).

Validates:
  - Hydrogen (Z=1): exact 13.606 eV
  - He+ (Z=2): exact 54.423 eV
  - Li 2s (Z=3): 5.58 eV ABCD, 5.32 eV with Op2 crossing
  - Helper functions: enclosed charge, numerical sigma, wavefunction extraction
  - All constants from constants.py, zero free parameters
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ave.core.constants import RY_EV, A_0, P_C
from ave.solvers.radial_eigenvalue import (
    radial_eigenvalue_abcd,
    _enclosed_charge_fraction,
    _z_net_smooth,
    _numerical_enclosed_charge,
)


class TestExactHydrogenic:
    """Hydrogen-like atoms with no inner shells — must match Z²·Ry/n² exactly."""

    def test_hydrogen_1s(self):
        """H 1s: E = Ry = 13.606 eV."""
        E = radial_eigenvalue_abcd(1, 1, 0, [])
        assert abs(E - RY_EV) / RY_EV < 0.001, f"H 1s: {E:.4f} vs {RY_EV:.4f}"

    def test_he_plus_1s(self):
        """He+ 1s: E = 4·Ry = 54.423 eV."""
        E = radial_eigenvalue_abcd(2, 1, 0, [])
        expected = 4.0 * RY_EV
        assert abs(E - expected) / expected < 0.001, f"He+ 1s: {E:.4f} vs {expected:.4f}"

    def test_li_2plus_1s(self):
        """Li²⁺ 1s: E = 9·Ry = 122.45 eV."""
        E = radial_eigenvalue_abcd(3, 1, 0, [])
        expected = 9.0 * RY_EV
        assert abs(E - expected) / expected < 0.001, f"Li2+ 1s: {E:.4f} vs {expected:.4f}"


class TestLithium2s:
    """Li 2s with 1s² inner shell — ABCD cascade + Op2 crossing."""

    def test_abcd_result(self):
        """ABCD cascade (no Op2): ~5.58 eV, within 5% of experiment."""
        E = radial_eigenvalue_abcd(3, 2, 0, [(1, 2)])
        assert 5.0 < E < 6.0, f"Li 2s ABCD: {E:.4f} not in [5.0, 6.0]"
        # Should be ~3.5% above experiment (5.39 eV)
        assert abs(E - 5.39) / 5.39 < 0.05, f"Li 2s error: {abs(E-5.39)/5.39*100:.1f}%"

    def test_op2_crossing_correction(self):
        """Op2 crossing: δE = E × P_C/4 → final ~5.32 eV (1.2% error)."""
        E_abcd = radial_eigenvalue_abcd(3, 2, 0, [(1, 2)])
        delta_op2 = E_abcd * P_C / 4.0
        E_corrected = E_abcd - delta_op2
        assert abs(E_corrected - 5.39) / 5.39 < 0.02, (
            f"Li 2s with Op2: {E_corrected:.4f} eV, error "
            f"{abs(E_corrected-5.39)/5.39*100:.1f}%"
        )


class TestEnclosedCharge:
    """Enclosed charge fraction σ(r) from hydrogenic 1s density."""

    def test_sigma_at_zero(self):
        """σ(0) = 0: no charge enclosed at origin."""
        sigma = _enclosed_charge_fraction(0.0, 3.0)
        assert sigma == 0.0

    def test_sigma_at_infinity(self):
        """σ(∞) → 1: all charge enclosed."""
        sigma = _enclosed_charge_fraction(100.0 * A_0, 3.0)
        assert abs(sigma - 1.0) < 1e-10

    def test_sigma_monotonic(self):
        """σ(r) must be monotonically increasing."""
        r_grid = np.linspace(0.01 * A_0, 10.0 * A_0, 100)
        sigma = np.array([_enclosed_charge_fraction(r, 3.0) for r in r_grid])
        assert np.all(np.diff(sigma) >= 0), "σ(r) not monotonic"

    def test_z_net_smooth(self):
        """Z_net should be Z at r=0 and Z-N at r→∞."""
        z_near = _z_net_smooth(0.001 * A_0, 3, [(1, 2)])
        z_far = _z_net_smooth(100.0 * A_0, 3, [(1, 2)])
        assert abs(z_near - 3.0) < 0.01, f"Z_net near nucleus: {z_near}"
        assert abs(z_far - 1.0) < 0.01, f"Z_net far out: {z_far}"


class TestNumericalSigma:
    """Numerical enclosed charge from arbitrary ψ(r)."""

    def test_normalisation(self):
        """σ(r_max) = 1 by normalisation."""
        r = np.linspace(0.01 * A_0, 5.0 * A_0, 500)
        # Fake hydrogenic psi: r * exp(-r/a0)
        psi = r * np.exp(-r / A_0)
        sigma = _numerical_enclosed_charge(r, psi)
        assert abs(sigma[-1] - 1.0) < 0.01, f"σ(r_max) = {sigma[-1]}"

    def test_monotonic(self):
        """Numerical σ must be monotonic."""
        r = np.linspace(0.01 * A_0, 5.0 * A_0, 500)
        psi = r * np.exp(-r / A_0)
        sigma = _numerical_enclosed_charge(r, psi)
        assert np.all(np.diff(sigma) >= -1e-15), "Numerical σ not monotonic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
