"""
Test suite for the protein bond constants bridge (Gap 6).

Verifies:
  1. Backbone constants are physically reasonable
  2. Bohr radius derivation matches CODATA
  3. Z_TOPO covers all 20 amino acids
  4. Derivation chain is consistent
"""

import numpy as np
import pytest

from ave.solvers.protein_bond_constants import (
    CA_CA_BOND_LENGTH_ANGSTROM,
    CA_CA_BOND_LENGTH_M,
    BOHR_RADIUS_ANGSTROM,
    D_PROTON_FM,
    D0_OVER_BOHR,
    Q_BACKBONE,
    BACKBONE_BONDS,
    BACKBONE_ANGLES,
    Z_TOPO,
)
from ave.core.constants import D_PROTON, L_NODE, ALPHA


class TestBackboneConstants:
    """Backbone bond lengths must be physically reasonable."""

    def test_ca_ca_is_3_8(self):
        """Cα-Cα distance must be 3.80 Å."""
        assert CA_CA_BOND_LENGTH_ANGSTROM == pytest.approx(3.80, abs=0.01)

    def test_ca_ca_in_meters(self):
        """Cα-Cα in meters must be 3.80e-10 m."""
        assert CA_CA_BOND_LENGTH_M == pytest.approx(3.80e-10, rel=0.01)

    def test_bond_lengths_physically_reasonable(self):
        """All backbone bonds should be 0.8–1.7 Å."""
        for name, bond in BACKBONE_BONDS.items():
            d = bond['length_A']
            assert 0.80 < d < 1.7, f"{name} = {d:.2f} Å out of range"

    def test_bond_angles_physically_reasonable(self):
        """All angles should be 100°–130°."""
        for name, angle in BACKBONE_ANGLES.items():
            assert 100 < angle < 130, f"{name} = {angle:.1f}° out of range"


class TestDerivationChain:
    """The chain from axioms to protein backbone must be consistent."""

    def test_bohr_radius(self):
        """a₀ = ℓ_node / α ≈ 0.529 Å."""
        assert BOHR_RADIUS_ANGSTROM == pytest.approx(0.529, rel=0.01)

    def test_bohr_from_constants(self):
        """a₀ = L_NODE / ALPHA from constants.py."""
        expected = L_NODE / ALPHA * 1e10  # to Å
        assert BOHR_RADIUS_ANGSTROM == pytest.approx(expected, rel=1e-10)

    def test_proton_radius(self):
        """d_p ≈ 0.841 fm."""
        assert D_PROTON_FM == pytest.approx(0.841, abs=0.002)

    def test_d0_over_bohr_ratio(self):
        """d₀/a₀ ratio must be consistent."""
        assert D0_OVER_BOHR == pytest.approx(
            CA_CA_BOND_LENGTH_ANGSTROM / BOHR_RADIUS_ANGSTROM,
            rel=1e-10,
        )


class TestImpedanceTable:
    """Z_TOPO must cover all 20 standard amino acids."""

    STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

    def test_all_20_present(self):
        """All 20 standard amino acids must have Z_TOPO entries."""
        assert set(Z_TOPO.keys()) == self.STANDARD_AA

    def test_all_nonzero(self):
        """All Z_TOPO magnitudes must be > 0."""
        for aa, z in Z_TOPO.items():
            assert abs(z) > 0, f"{aa}: |Z| = 0"

    def test_hydrophobic_real_only(self):
        """Hydrophobic residues should have X ≈ 0."""
        hydrophobic = "AVILMFWPG"
        for aa in hydrophobic:
            assert Z_TOPO[aa].imag == pytest.approx(0.0, abs=1e-10), \
                f"{aa}: X = {Z_TOPO[aa].imag:.4f} should be 0"

    def test_charged_have_reactance(self):
        """Charged residues must have X ≠ 0."""
        charged = "DEKR"
        for aa in charged:
            assert abs(Z_TOPO[aa].imag) > 0.01, \
                f"{aa}: X = {Z_TOPO[aa].imag:.4f} should be nonzero"

    def test_q_backbone(self):
        """Q factor should be ~7."""
        assert Q_BACKBONE == pytest.approx(7.0, rel=0.01)
