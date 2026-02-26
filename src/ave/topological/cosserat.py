"""
Cosserat Micropolar Weak Sector for the AVE Framework.

Derives the electroweak gauge boson masses and the neutrino mass
spectrum from the torsional (Cosserat) sector of the Chiral LC lattice.

=== The Derivation Chain ===

1. The weak force is the evanescent (Yukawa) sector of the lattice's
   torsional degrees of freedom (Chapter 8).

2. The W/Z mass ratio comes from the Perpendicular Axis Theorem
   applied to a cylindrical flux tube with Poisson ratio nu_vac = 2/7:
       m_W/m_Z = 1/sqrt(1 + nu_vac) = sqrt(7)/3 ≈ 0.8819  (0.05% match)

3. The ABSOLUTE W mass comes from the evanescent cutoff of the
   torsional sector, governed by three lattice constants:
       M_W = m_e / (8*pi*alpha^3 * sin(theta_W))
           = m_e / (alpha^2 * p_c * sin(theta_W))
   where p_c = 8*pi*alpha is the geometric packing fraction and
   sin^2(theta_W) = 3/7 from the Poisson ratio.

   This gives M_W = 79,923 MeV (0.57% from CODATA 80,379 MeV).

4. The neutrino is a pure torsional (screw) defect. Its mass is set
   by the ratio of torsional to translational coupling:
       m_nu = m_e * alpha * (m_e / M_W)
   which gives m_nu ~ 24 meV per flavor.

5. Neutrino mass splitting follows the torus knot ladder:
   each flavor pairs with a baryon resonance via the crossing number.
"""

from math import pi, sqrt
from ave.core.constants import (
    ALPHA, M_E, C_0, NU_VAC, P_C, HBAR, L_NODE,
)

# =============================================================================
# WEAK MIXING ANGLE (from Poisson ratio, Chapter 4 + Chapter 8)
# =============================================================================

# sin^2(theta_W) = 3/(3 + 4) = 3/7  (from nu_vac = 2/7)
SIN2_THETA_W: float = 3.0 / 7.0          # = 0.428571
COS2_THETA_W: float = 4.0 / 7.0          # = 0.571429
SIN_THETA_W: float = sqrt(SIN2_THETA_W)  # = 0.654654
COS_THETA_W: float = sqrt(COS2_THETA_W)  # = 0.755929

# Empirical comparison:
# sin^2(theta_W)_exp = 0.23122 (on-shell) or 0.2312 (MSbar)
# BUT: m_W/m_Z|exp = 80379/91188 = 0.88148
# and sqrt(7)/3 = 0.88192 → 0.05% match (pole mass ratio)

# =============================================================================
# W AND Z BOSON MASSES
# =============================================================================

# The W boson mass from the evanescent cutoff of the torsional sector:
#   M_W = m_e / (alpha^2 * p_c * sin(theta_W))
#       = m_e / (8*pi*alpha^3 * sin(theta_W))
#
# Physical meaning: The torsional mode has stiffness suppressed by
# alpha^2 (dielectric screening) and p_c (packing fraction). The
# sin(theta_W) factor comes from the torsion-shear coupling angle.
M_W: float = M_E / (ALPHA**2 * P_C * SIN_THETA_W)  # in kg
M_W_MEV: float = M_W * C_0**2 / 1.602176634e-13    # ≈ 79923 MeV

# The Z boson mass from the W mass and Weak Mixing Angle:
#   m_W/m_Z = sqrt(7)/3 ≈ 0.8819 (from Chapter 8, Perpendicular Axis Theorem)
#   M_Z = M_W * 3/sqrt(7)
M_Z: float = M_W * 3.0 / sqrt(7)                     # in kg
M_Z_MEV: float = M_Z * C_0**2 / 1.602176634e-13      # ≈ 90624 MeV

# =============================================================================
# COSSERAT CHARACTERISTIC LENGTH (Weak Force Range)
# =============================================================================

# l_c = hbar / (M_W * c) — the Compton wavelength of the W boson
L_COSSERAT: float = HBAR / (M_W * C_0)              # ≈ 2.46e-18 m

# =============================================================================
# FERMI CONSTANT (Tree-Level)
# =============================================================================

# G_F = pi * alpha / (sqrt(2) * M_W^2 * sin^2(theta_W))  [in GeV^-2]
# (Tree-level; 1-loop corrections give a factor ~2 improvement)
M_W_GEV: float = M_W_MEV / 1000.0
GF_TREE: float = pi * ALPHA / (sqrt(2) * M_W_GEV**2 * SIN2_THETA_W)

# =============================================================================
# NEUTRINO MASS SPECTRUM
# =============================================================================

# The neutrino is a pure torsional (screw) defect. Its mass is:
#   m_nu = m_e * alpha * (m_e / M_W)
#
# Physical meaning:
# - m_e/M_W = ratio of translational to torsional scale
# - alpha = the dielectric coupling between sectors
# - Together: the neutrino mass is suppressed by alpha × (m_e/M_W)
#   relative to the electron mass.

M_NU_BASE: float = M_E * ALPHA * (M_E / M_W)        # in kg
M_NU_EV: float = M_NU_BASE * C_0**2 / 1.602176634e-19  # ≈ 0.024 eV

# Three flavors from the torus knot ladder:
# Each neutrino flavor pairs with a baryon resonance.
# The mass splitting goes as 1/c where c is the crossing number.
# nu_1 ↔ (2,5) proton, nu_2 ↔ (2,7) Delta, nu_3 ↔ (2,9) Delta
CROSSING_NUMBERS_NEUTRINO = [5, 7, 9]
M_NU_FLAVORS_EV = [M_NU_EV * 5.0 / c for c in CROSSING_NUMBERS_NEUTRINO]
# → [~24, ~17, ~13 meV]

SUM_M_NU_EV: float = sum(M_NU_FLAVORS_EV)
# → ~0.054 eV (Planck 2018 bound: < 0.12 eV, hint: ~0.06 eV)
