"""
AVE Geophysics Module
======================
Seismic impedance matching and FDTD wave propagation.
"""

from ave.geophysics.seismic import (
    SeismicLayer,
    reflection_coefficient,
    transmission_coefficient,
    travel_time,
    all_reflections,
    build_1d_impedance_profile,
)
from ave.geophysics.seismic_fdtd import (
    build_seismic_engine,
    verify_impedance_consistency,
    compute_boundary_reflections,
)

__all__ = [
    # Seismic layers
    "SeismicLayer",
    "reflection_coefficient",
    "transmission_coefficient",
    "travel_time",
    "all_reflections",
    "build_1d_impedance_profile",
    # FDTD
    "build_seismic_engine",
    "verify_impedance_consistency",
    "compute_boundary_reflections",
]
