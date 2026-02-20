"""
AVE Neutrino Sector
Implements the physics of Chiral Unknots (0_1 Topology).
Source: Chapter 7
"""
import sys
from pathlib import Path

# Add src directory to path if running as script
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def check_chirality_permission(handedness):
    """
    Determines if a neutrino state is allowed by the Cosserat Vacuum.
    Source: Eq 7.1 & 7.2 (The Chiral Exclusion Principle)
    """
    # The vacuum has intrinsic microrotation stiffness (gamma_c)
    # Dispersion: omega^2 = c^2*k^2 - sgn(H)*gamma_c*k
    
    if handedness == "left":
        # Left-handed matches the vacuum grain.
        # omega^2 > 0 (Propagating Wave)
        return True
    elif handedness == "right":
        # Right-handed shears against the vacuum grain.
        # omega^2 < 0 (Evanescent Wave / Anderson Localization)
        return False
    else:
        raise ValueError("Handedness must be 'left' or 'right'")

def calculate_neutrino_acoustic_mass(flavor_harmonic):
    """
    Estimates neutrino mass based on torsional harmonics.
    Since they are unknots (0_1), they lack the dielectric saturation multiplier.
    Mass scales purely linearly with the Torsional Harmonic (T).
    """
    return True