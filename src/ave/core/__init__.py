"""
AVE Core
Fundamental discrete topological variables entirely replacing the Standard Model empirical soup.
"""

from .constants import (
    # Fundamental Priors
    C,
    H_BAR,
    E_CHARGE,
    M_E,
    G,
    # Derived Geometric Axioms
    ALPHA_GEOM,
    KAPPA_V,
    L_NODE,
    XI_TOPO,
    # Derived Electromagnetic Constraints
    T_EM,
    EPSILON_0,
    MU_0,
    Z_0,
    # Derived Macroscopic Parameters
    RHO_BULK,
    NU_VAC,
    H_INF,
    A_GENESIS,
)

from .conversion import mass_to_inductance, inductance_to_mass, velocity_to_current, current_to_velocity
