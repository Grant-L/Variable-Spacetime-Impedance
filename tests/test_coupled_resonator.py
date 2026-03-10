"""
Test Coupled Resonator Solver
==============================

Tests for the universal coupled resonator framework: nuclear binding,
atomic ionization energies, and molecular bond energies — all from
a single impedance coupling formalism.
"""

import numpy as np
import pytest
from ave.solvers.coupled_resonator import (
    complete_graph_eigenvalues,
    coupled_resonator_binding,
    hierarchical_binding,
    nuclear_mass,
    ionization_energy,
    atom_port_impedance,
    molecular_bond_distance,
    molecular_bond_energy,
    K_COUPLING,
)


# ═══════════════════════════════════════════════════════════════════════
# Graph Theory
# ═══════════════════════════════════════════════════════════════════════

class TestCompleteGraphEigenvalues:
    def test_K2_eigenvalues(self):
        lam = complete_graph_eigenvalues(2)
        np.testing.assert_allclose(sorted(lam), [-1, 1])

    def test_K4_eigenvalues(self):
        lam = complete_graph_eigenvalues(4)
        np.testing.assert_allclose(sorted(lam), [-1, -1, -1, 3])

    def test_sum_is_zero(self):
        """Trace of adjacency matrix = 0."""
        for n in [2, 3, 5, 10]:
            assert abs(sum(complete_graph_eigenvalues(n))) < 1e-10


# ═══════════════════════════════════════════════════════════════════════
# Nuclear Binding
# ═══════════════════════════════════════════════════════════════════════

class TestNuclearBinding:
    def test_deuteron_binding_positive(self):
        B, _ = coupled_resonator_binding(2, K_COUPLING)
        assert B > 0, "Deuteron must be bound"

    def test_alpha_binding_greater_than_deuteron(self):
        B_d, _ = coupled_resonator_binding(2, K_COUPLING)
        B_a, _ = coupled_resonator_binding(4, K_COUPLING)
        assert B_a > B_d

    def test_nuclear_mass_helium_4(self):
        mass, binding = nuclear_mass(2, 4)
        assert binding > 0
        assert mass > 0

    def test_hierarchical_binding_positive(self):
        B, B_alpha, B_inter = hierarchical_binding(3)
        assert B > 0
        assert B_alpha > 0
        assert B_inter >= 0


# ═══════════════════════════════════════════════════════════════════════
# Atomic Ionization
# ═══════════════════════════════════════════════════════════════════════

class TestIonizationEnergy:
    def test_hydrogen_ie(self):
        IE = ionization_energy(1)
        assert 13.0 < IE < 14.0, f"H IE = {IE}, expected ~13.6 eV"

    def test_helium_ie_higher_than_hydrogen(self):
        IE_H = ionization_energy(1)
        IE_He = ionization_energy(2)
        assert IE_He > IE_H

    def test_lithium_ie_lower_than_helium(self):
        IE_He = ionization_energy(2)
        IE_Li = ionization_energy(3)
        assert IE_Li < IE_He, "Li IE < He IE (new shell)"


# ═══════════════════════════════════════════════════════════════════════
# Molecular Bonds
# ═══════════════════════════════════════════════════════════════════════

class TestMolecularBonds:
    def test_bond_distance_positive(self):
        r_A = 1e-10
        r_B = 1e-10
        d = molecular_bond_distance(r_A, r_B)
        assert d > 0

    def test_bond_energy_positive(self):
        B, k = molecular_bond_energy(13.6, 13.6)
        assert B > 0
        assert 0 < k <= 1

    def test_double_bond_stronger(self):
        B1, _ = molecular_bond_energy(13.6, 13.6, n_bonds=1)
        B2, _ = molecular_bond_energy(13.6, 13.6, n_bonds=2)
        assert B2 > B1

    def test_atom_port_impedance_positive(self):
        r = atom_port_impedance(6, 11.26)  # Carbon
        assert r > 0
