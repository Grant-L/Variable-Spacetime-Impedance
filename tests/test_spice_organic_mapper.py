"""
Test Suite: SPICE Organic Mapper — Zero-Parameter AVE Derivation
================================================================
Locks in the axiom-derived L = m/ξ² and C = ξ²/k mapping,
ensuring no regressions or magic numbers creep back in.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'mechanics'))

from ave.core.constants import (
    e_charge, HBAR, M_E, C_0, Z_0, MU_0, EPSILON_0, L_NODE, XI_TOPO
)
from spice_organic_mapper import (
    ATOMIC_INDUCTANCE, COVALENT_CAPACITANCE, BOND_FORCE_CONSTANTS,
    XI_TOPO_SQ, get_inductance, get_capacitance, get_force_constant,
    CARBOXYL_LOAD_R,
)


# ─── Constants ────────────────────────────────────────────────────────
DA = 1.66053906660e-27  # kg per Dalton (CODATA 2018)
XI2 = XI_TOPO**2        # transduction constant squared


class TestTransductionConstant:
    """ξ_topo must derive from e and ℓ_node with no hardcoded value."""

    def test_xi_topo_definition(self):
        """ξ = e / ℓ_node"""
        assert XI_TOPO == pytest.approx(e_charge / L_NODE, rel=1e-10)

    def test_xi_topo_expanded(self):
        """ξ = e·m_e·c / ℏ  (fully expanded)"""
        xi_from_fundamentals = e_charge * M_E * C_0 / HBAR
        assert XI_TOPO == pytest.approx(xi_from_fundamentals, rel=1e-10)

    def test_xi_squared_in_mapper(self):
        """Mapper's XI_TOPO_SQ must equal XI_TOPO² from constants."""
        assert XI_TOPO_SQ == pytest.approx(XI2, rel=1e-12)


class TestAtomicInductance:
    """L_atom = m_atom / ξ² — no free parameters."""

    EXPECTED_MASSES = {
        'H':  1.00794,
        'C': 12.0107,
        'N': 14.0067,
        'O': 15.9994,
        'S': 32.065,
    }

    @pytest.mark.parametrize("elem,mass_da", EXPECTED_MASSES.items())
    def test_inductance_formula(self, elem, mass_da):
        """Each L must satisfy L = m/ξ² exactly."""
        expected_L = (mass_da * DA) / XI2
        actual_L = get_inductance(elem)
        assert actual_L == pytest.approx(expected_L, rel=1e-8)

    def test_inductance_proportional_to_mass(self):
        """L must be strictly proportional to atomic mass."""
        L_H = get_inductance('H')
        L_C = get_inductance('C')
        L_O = get_inductance('O')
        # L_C / L_H should equal m_C / m_H
        assert (L_C / L_H) == pytest.approx(12.0107 / 1.00794, rel=1e-6)
        assert (L_O / L_H) == pytest.approx(15.9994 / 1.00794, rel=1e-6)

    def test_inductance_monotonic(self):
        """H < C < N < O < S in inductance (mass ordering)."""
        vals = [get_inductance(e) for e in ['H', 'C', 'N', 'O', 'S']]
        assert all(a < b for a, b in zip(vals, vals[1:]))

    def test_no_picohenry_scale(self):
        """Values must be in femtohenries, NOT picohenries (old heuristic)."""
        L_H = get_inductance('H')
        # Old heuristic: L_H ≈ 10 pH = 1e-11 H.  New: L_H ≈ 9.7 fH = 9.7e-15 H.
        assert L_H < 1e-13, "L_H should be in fH range, not pH"
        assert L_H > 1e-16, "L_H should be in fH range"

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="Unknown element"):
            get_inductance('Xe')


class TestBondCapacitance:
    """C_bond = ξ² / k — derived from force constants."""

    @pytest.mark.parametrize("bond,k", BOND_FORCE_CONSTANTS.items())
    def test_capacitance_formula(self, bond, k):
        """Each C must satisfy C = ξ²/k exactly."""
        expected_C = XI2 / k
        actual_C = get_capacitance(bond)
        assert actual_C == pytest.approx(expected_C, rel=1e-8)

    def test_stronger_bond_lower_capacitance(self):
        """Double bond must have lower C than single bond (stiffer = less compliant)."""
        assert get_capacitance('C=C') < get_capacitance('C-C')
        assert get_capacitance('C=O') < get_capacitance('C-O')

    def test_reverse_lookup(self):
        """'H-C' should return the same as 'C-H'."""
        assert get_capacitance('H-C') == get_capacitance('C-H')
        assert get_capacitance('O-H') == get_capacitance('H-O')

    def test_no_femtofarad_scale(self):
        """Values must be in attofarads, NOT femtofarads (old heuristic)."""
        C_CC = get_capacitance('C-C')
        # Old: ~144 fF = 1.44e-13 F.  New: ~486 aF = 4.86e-16 F
        assert C_CC < 1e-14, "C_CC should be in aF range, not fF"
        assert C_CC > 1e-18, "C_CC should be in aF range"

    def test_unknown_bond_raises(self):
        with pytest.raises(ValueError, match="Unknown bond"):
            get_capacitance('Xe-Kr')


class TestSelfConsistency:
    """The mapping must recover known physics exactly."""

    def test_ch_stretch_frequency(self):
        """f_res for C-H must match ~3000 cm⁻¹ (the input IR data)."""
        L_C = get_inductance('C')
        L_H = get_inductance('H')
        C_CH = get_capacitance('C-H')
        # Reduced-mass inductance
        L_red = (L_C * L_H) / (L_C + L_H)
        f_res = 1.0 / (2 * np.pi * np.sqrt(L_red * C_CH))
        nu_cm = f_res / (C_0 * 100)
        assert nu_cm == pytest.approx(3000, rel=0.01)  # within 1%

    def test_cc_stretch_frequency(self):
        """f_res for C-C must match ~1000 cm⁻¹."""
        L_C = get_inductance('C')
        C_CC = get_capacitance('C-C')
        L_red = L_C / 2  # symmetric
        f_res = 1.0 / (2 * np.pi * np.sqrt(L_red * C_CC))
        nu_cm = f_res / (C_0 * 100)
        assert nu_cm == pytest.approx(1000, rel=0.01)

    def test_co_carbonyl_stretch(self):
        """f_res for C=O must match ~1700 cm⁻¹."""
        L_C = get_inductance('C')
        L_O = get_inductance('O')
        C_CO = get_capacitance('C=O')
        L_red = (L_C * L_O) / (L_C + L_O)
        f_res = 1.0 / (2 * np.pi * np.sqrt(L_red * C_CO))
        nu_cm = f_res / (C_0 * 100)
        assert nu_cm == pytest.approx(1700, rel=0.01)

    def test_impedance_physical(self):
        """Z = √(L/C) must equal √(mk)/ξ² (mechanical impedance)."""
        m_C = 12.0107 * DA
        k_CH = get_force_constant('C-H')
        L = get_inductance('C')
        C = get_capacitance('C-H')
        Z_electrical = np.sqrt(L / C)
        Z_mechanical = np.sqrt(m_C * k_CH) / XI2
        assert Z_electrical == pytest.approx(Z_mechanical, rel=1e-6)

    def test_wave_speed(self):
        """v = 1/√(LC) must equal √(k/m) (bond sound speed)."""
        m_C = 12.0107 * DA
        k_CC = get_force_constant('C-C')
        L = get_inductance('C')
        C = get_capacitance('C-C')
        v_electrical = 1.0 / np.sqrt(L * C)
        v_mechanical = np.sqrt(k_CC / m_C)
        assert v_electrical == pytest.approx(v_mechanical, rel=1e-6)


class TestCarboxylLoad:
    """Termination resistance must be Z_0, not a hardcoded number."""

    def test_load_derives_from_z0(self):
        """Load impedance string must contain Z_0 value."""
        z0_str = f"{Z_0:.4f}"
        assert z0_str in CARBOXYL_LOAD_R

    def test_load_not_hardcoded_377(self):
        """Must not use rounded 377 Ω."""
        assert "377.00" not in CARBOXYL_LOAD_R
        assert "377Ohm" not in CARBOXYL_LOAD_R


class TestNoMagicNumbers:
    """Regression tests: ensure old heuristics never return."""

    def test_no_scale_factor_10(self):
        """The old MASS_TO_INDUCTANCE_SCALE = 10.0 must not exist."""
        import spice_organic_mapper as mod
        assert not hasattr(mod, 'MASS_TO_INDUCTANCE_SCALE')

    def test_no_base_bond_capacitance(self):
        """The old BASE_BOND_CAPACITANCE = 50000.0 must not exist."""
        import spice_organic_mapper as mod
        assert not hasattr(mod, 'BASE_BOND_CAPACITANCE')

    def test_hydrogen_not_10pH(self):
        """L_H must NOT be ~10 pH (the old heuristic)."""
        L_H = get_inductance('H')
        assert abs(L_H - 10.08e-12) > 1e-13, "L_H still at old heuristic value"
