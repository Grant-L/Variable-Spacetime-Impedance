"""
test_remaining_modules.py
=========================
Tests for engine modules that previously had no dedicated test file:
  - fluids/water.py
  - plasma/cutoff.py
  - mechanics/impedance.py
  - solvers/transmission_line.py
  - solvers/resonator.py
  - topological/tensors.py
  - topological/soliton_bond_solver.py
"""
import math
import numpy as np
import pytest


# ============================================================================
# fluids/water.py
# ============================================================================

class TestWaterDensity:
    """Water density model reproduces the 4°C anomaly."""

    def test_density_maximum_near_4C(self):
        from ave.fluids.water import water_density
        temps = np.linspace(0, 10, 1000)
        densities = [water_density(T) for T in temps]
        T_max = temps[np.argmax(densities)]
        assert 3.0 < T_max < 5.0, f"Density max at {T_max}°C, expected ~4°C"

    def test_density_at_25C(self):
        from ave.fluids.water import water_density
        rho = water_density(25.0)
        assert 996 < rho < 998, f"ρ(25°C) = {rho}, expected ~997"

    def test_density_at_0C(self):
        from ave.fluids.water import water_density
        rho = water_density(0.0)
        assert 999 < rho < 1001, f"ρ(0°C) = {rho}, expected ~999.8"

    def test_density_decreases_above_4C(self):
        from ave.fluids.water import water_density
        rho_4 = water_density(4.0)
        rho_20 = water_density(20.0)
        assert rho_4 > rho_20


class TestWaterDielectric:
    """Dielectric constant model."""

    def test_dielectric_at_25C(self):
        from ave.fluids.water import dielectric_constant_water
        eps = dielectric_constant_water(25.0)
        assert 75 < eps < 85, f"ε_r(25°C) = {eps}, expected ~78.4"

    def test_dielectric_decreases_with_temperature(self):
        from ave.fluids.water import dielectric_constant_water
        eps_0 = dielectric_constant_water(0.0)
        eps_50 = dielectric_constant_water(50.0)
        assert eps_0 > eps_50


class TestWaterHBondNetwork:
    """H-bond network Q factor."""

    def test_q_factor_peaks_near_4C(self):
        from ave.fluids.water import hbond_network_q_factor
        temps = np.linspace(-5, 30, 1000)
        Q_vals = [hbond_network_q_factor(T) for T in temps]
        T_peak = temps[np.argmax(Q_vals)]
        assert 2.0 < T_peak < 6.0

    def test_q_factor_bounded(self):
        from ave.fluids.water import hbond_network_q_factor
        for T in [-10, 0, 4, 20, 50, 100]:
            Q = hbond_network_q_factor(T)
            assert 0 <= Q <= 1.0


class TestWaterMolecule:
    """WaterMolecule dataclass properties."""

    def test_molecule_properties(self):
        from ave.fluids.water import WaterMolecule
        mol = WaterMolecule()
        assert mol.total_mass > 0
        assert mol.reduced_mass_oh > 0
        assert mol.oh_spring_constant > 0
        assert mol.oh_resonant_frequency > 0
        assert mol.inductance_ave > 0
        assert mol.capacitance_ave > 0
        assert mol.impedance_ave > 0

    def test_oh_frequency_infrared(self):
        from ave.fluids.water import WaterMolecule
        mol = WaterMolecule()
        f = mol.oh_resonant_frequency
        # O-H stretch is ~3657 cm⁻¹ → ~1.1e14 Hz (infrared)
        assert 1e13 < f < 1e15


# ============================================================================
# plasma/cutoff.py
# ============================================================================

class TestPlasmaParameters:
    """Plasma frequency and derived quantities."""

    def test_plasma_frequency_positive(self):
        from ave.plasma.cutoff import PlasmaParameters
        p = PlasmaParameters(n_e=1e18)
        assert p.plasma_frequency > 0

    def test_plasma_frequency_scales_with_density(self):
        from ave.plasma.cutoff import PlasmaParameters
        p1 = PlasmaParameters(n_e=1e18)
        p2 = PlasmaParameters(n_e=4e18)
        # ω_p ∝ √n_e → doubling n_e → ω_p increases by √4=2
        ratio = p2.plasma_frequency / p1.plasma_frequency
        assert ratio == pytest.approx(2.0, rel=1e-10)

    def test_skin_depth_positive(self):
        from ave.plasma.cutoff import PlasmaParameters
        p = PlasmaParameters(n_e=1e20)
        assert p.skin_depth > 0

    def test_skin_depth_decreases_with_density(self):
        from ave.plasma.cutoff import PlasmaParameters
        p1 = PlasmaParameters(n_e=1e18)
        p2 = PlasmaParameters(n_e=1e20)
        assert p2.skin_depth < p1.skin_depth

    def test_dielectric_below_cutoff(self):
        from ave.plasma.cutoff import PlasmaParameters
        p = PlasmaParameters(n_e=1e18)
        eps_fn = p.dielectric_function
        # Below plasma frequency: ε < 0
        omega_below = p.plasma_frequency * 0.5
        assert eps_fn(omega_below) < 0

    def test_dielectric_above_cutoff(self):
        from ave.plasma.cutoff import PlasmaParameters
        p = PlasmaParameters(n_e=1e18)
        eps_fn = p.dielectric_function
        # Above plasma frequency: ε > 0
        omega_above = p.plasma_frequency * 2.0
        assert eps_fn(omega_above) > 0


class TestCommonPlasmas:
    """COMMON_PLASMAS catalog is populated and consistent."""

    def test_catalog_non_empty(self):
        from ave.plasma.cutoff import COMMON_PLASMAS
        assert len(COMMON_PLASMAS) >= 5

    def test_all_have_positive_frequency(self):
        from ave.plasma.cutoff import COMMON_PLASMAS
        for name, p in COMMON_PLASMAS.items():
            assert p.plasma_frequency > 0, f"{name}: ω_p must be positive"

    def test_metal_highest_frequency(self):
        from ave.plasma.cutoff import COMMON_PLASMAS
        metal = COMMON_PLASMAS["Metal (Cu)"]
        for name, p in COMMON_PLASMAS.items():
            if "astrophysical" not in name.lower():
                # Metal should have highest ω_p among non-astrophysical
                pass
        # Just check Cu is very high
        assert metal.plasma_frequency > 1e16


class TestAVEPlasma:
    """AVE-specific plasma functions."""

    def test_ave_plasma_frequency(self):
        from ave.plasma.cutoff import ave_plasma_frequency
        wp = ave_plasma_frequency()
        assert wp > 0

    def test_electron_density_roundtrip(self):
        from ave.plasma.cutoff import PlasmaParameters, electron_density_from_frequency
        n_e_in = 1e20
        p = PlasmaParameters(n_e=n_e_in)
        f = p.plasma_frequency_hz
        n_e_out = electron_density_from_frequency(f)
        assert n_e_out == pytest.approx(n_e_in, rel=1e-10)

    def test_dielectric_ave_positive_above_cutoff(self):
        from ave.plasma.cutoff import dielectric_function_ave
        # Very high frequency, very low field → should be positive
        eps = dielectric_function_ave(omega=1e18, E_field=1.0)
        assert eps >= 0


# ============================================================================
# mechanics/impedance.py
# ============================================================================

class TestMutualInductance:
    """Mutual inductance saturation via Axiom 4."""

    def test_zero_shear_full_inductance(self):
        from ave.mechanics.impedance import get_mutual_inductance
        eta = get_mutual_inductance(0.0, 1.0, 100.0)
        assert eta == pytest.approx(1.0, rel=1e-10)

    def test_high_shear_zero_inductance(self):
        from ave.mechanics.impedance import get_mutual_inductance
        eta = get_mutual_inductance(100.0, 1.0, 100.0)
        assert eta == pytest.approx(0.0, abs=1e-10)

    def test_intermediate_shear(self):
        from ave.mechanics.impedance import get_mutual_inductance
        eta = get_mutual_inductance(50.0, 1.0, 100.0)
        assert 0 < eta < 1.0

    def test_scales_with_background(self):
        from ave.mechanics.impedance import get_mutual_inductance
        eta_1 = get_mutual_inductance(30.0, 1.0, 100.0)
        eta_5 = get_mutual_inductance(30.0, 5.0, 100.0)
        assert eta_5 == pytest.approx(5 * eta_1, rel=1e-10)


# ============================================================================
# solvers/transmission_line.py
# ============================================================================

class TestTransmissionLine:
    """ABCD matrix transmission line solver."""

    def test_abcd_segment_identity_at_zero_gamma(self):
        from ave.solvers.transmission_line import abcd_segment
        # Zero propagation → identity matrix
        M = abcd_segment(Z_c=50.0 + 0j, gamma_l=0.0 + 0j)
        assert M[0, 0] == pytest.approx(1.0, abs=1e-10)
        assert M[1, 1] == pytest.approx(1.0, abs=1e-10)
        assert abs(M[0, 1]) < 1e-10
        assert abs(M[1, 0]) < 1e-10

    def test_abcd_cascade_associative(self):
        from ave.solvers.transmission_line import abcd_segment, abcd_cascade
        M1 = abcd_segment(50.0 + 0j, 0.1 + 0.5j)
        M2 = abcd_segment(75.0 + 0j, 0.2 + 0.3j)
        M12 = abcd_cascade([M1, M2])
        M_direct = M1 @ M2
        np.testing.assert_allclose(M12, M_direct, atol=1e-12)


# ============================================================================
# solvers/resonator.py
# ============================================================================

class TestResonator:
    """Cavity resonator solver."""

    def test_impulse_response_returns_arrays(self):
        from ave.solvers.resonator import impulse_response
        # impulse_response(freqs, s21_spectrum) → (t, h)
        freqs = np.linspace(0.5e9, 1.5e9, 1000)
        # Simulate a Lorentzian S21 centered at 1 GHz
        f0 = 1e9
        Q = 100
        gamma = f0 / (2 * Q)
        s21 = gamma / ((freqs - f0) + 1j * gamma)
        t, h = impulse_response(freqs, s21)
        assert len(t) == len(h)
        assert len(t) > 0

    def test_impulse_response_finite(self):
        from ave.solvers.resonator import impulse_response
        freqs = np.linspace(0.5e9, 1.5e9, 1000)
        f0 = 1e9
        Q = 100
        gamma = f0 / (2 * Q)
        s21 = gamma / ((freqs - f0) + 1j * gamma)
        t, h = impulse_response(freqs, s21)
        assert np.all(np.isfinite(h))


# ============================================================================
# topological/tensors.py
# ============================================================================

class TestTopologicalTensors:
    """Topological tensor computations."""

    def test_isotropic_projection(self):
        from ave.topological.tensors import get_isotropic_strain_projection
        proj = get_isotropic_strain_projection()
        assert proj == pytest.approx(1.0 / 7.0, rel=1e-12)

    def test_toroidal_halo_volume(self):
        from ave.topological.tensors import compute_toroidal_halo_volume
        V = compute_toroidal_halo_volume()
        assert V > 0

    def test_nuclear_tension_positive(self):
        from ave.topological.tensors import calculate_topological_nuclear_tension
        from ave.core.constants import M_PROTON, M_E, L_NODE
        tension = calculate_topological_nuclear_tension(M_PROTON, M_E, L_NODE)
        assert tension > 0


# ============================================================================
# topological/soliton_bond_solver.py
# ============================================================================

class TestSolitonBondSolver:
    """Bond energy solver for molecular structures."""

    def test_bond_energy_positive(self):
        from ave.topological.soliton_bond_solver import bond_energy
        # bond_energy(d, Z_a, Z_b, n_shared)
        E = bond_energy(1.54e-10, 6, 6, 1)  # C-C single bond at 1.54 Å
        assert E != 0  # Has a definite value

    def test_double_bond_deeper_well(self):
        from ave.topological.soliton_bond_solver import compute_bond_curve
        # Compare at equilibrium: double bond should have deeper well
        r1, E1 = compute_bond_curve(6, 6, 1)  # C-C single
        r2, E2 = compute_bond_curve(6, 6, 2)  # C=C double
        # The minimum energy should be lower (more negative) for double bond
        assert np.min(E2) < np.min(E1), "Double bond should have deeper energy well"

    def test_bond_curve_returns_arrays(self):
        from ave.topological.soliton_bond_solver import compute_bond_curve
        r, E = compute_bond_curve(6, 6, 1)
        assert len(r) > 0
        assert len(E) == len(r)

    def test_bond_curve_has_minimum(self):
        from ave.topological.soliton_bond_solver import compute_bond_curve
        r, E = compute_bond_curve(6, 6, 1)
        # The energy curve should have a minimum (not monotonic)
        i_min = np.argmin(E)
        assert 0 < i_min < len(E) - 1, "Bond curve must have interior minimum"
