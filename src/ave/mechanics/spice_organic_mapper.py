"""
SPICE Organic Mapper (Applied Vacuum Engineering)
=================================================
Maps organic chemical topologies (Atomic Nuclei and Covalent Bonds)
into absolute Macroscopic Inductance (L) and Capacitance (C) values 
for SPICE (.cir) circuit simulation.

In AVE:
- Mass = Geometric Inertia = Inductance (Henry, H)
- Covalent Bond = Dielectric Stress = Capacitance (Farad, F)
- Chirality = Geometric Winding (Phase Angle)
"""

import numpy as np

# Core AVE Vacuum Constants 
from ave.core.constants import Z_0, C_0  

# ---------------------------------------------------------
# 1. ATOMIC MASS TO INDUCTANCE (L) MAPPING
# ---------------------------------------------------------
# Let's define a scaling factor to bring atomic mass units (Da) 
# into the picoHenry (pH) range for high-frequency RF simulation.
# 1 Dalton (Da) = 1.66053906660e-27 kg
# In AVE, L is proportional to the knot's rotational inertia.
# We'll use a heuristic scaling for macroscopic SPICE translation.

MASS_TO_INDUCTANCE_SCALE = 10.0 # picoHenries (pH) per Dalton

ATOMIC_INDUCTANCE = {
    # Element: Inductance in picoHenries (pH)  [Mass * Scale]
    'H':  1.008 * MASS_TO_INDUCTANCE_SCALE,   # Hydrogen
    'C': 12.011 * MASS_TO_INDUCTANCE_SCALE,   # Carbon
    'N': 14.007 * MASS_TO_INDUCTANCE_SCALE,   # Nitrogen
    'O': 15.999 * MASS_TO_INDUCTANCE_SCALE,   # Oxygen
    'S': 32.065 * MASS_TO_INDUCTANCE_SCALE,   # Sulfur
}


# ---------------------------------------------------------
# 2. COVALENT BOND TO CAPACITANCE (C) MAPPING
# ---------------------------------------------------------
# A covalent bond is not a rigid stick; it is a region of dielectric 
# compliance (epsilon_eff) between two massive nodes.
# A "stronger" tighter bond has LESS compliance (lower Capacitance)
# A "weaker" longer bond has MORE compliance (higher Capacitance)
# We map standard bond dissociation energies (kJ/mol) to femtoFarads (fF).

# Heuristic: The tighter the bond (higher Energy), the LESS C it acts like.
# C = (Base_Constant) / Bond_Energy
BASE_BOND_CAPACITANCE = 50000.0 # femtoFarads (fF) baseline scalar

COVALENT_CAPACITANCE = {
    # Bond: Capacitance in femtoFarads (fF) [Base / Bond Energy(kJ/mol)]
    'C-C': BASE_BOND_CAPACITANCE / 347.0,  # Single Carbon-Carbon
    'C=C': BASE_BOND_CAPACITANCE / 614.0,  # Double Carbon-Carbon
    'C-H': BASE_BOND_CAPACITANCE / 413.0,  # Carbon-Hydrogen
    'C-N': BASE_BOND_CAPACITANCE / 293.0,  # Carbon-Nitrogen
    'C=O': BASE_BOND_CAPACITANCE / 799.0,  # Carbon-Oxygen (Carbonyl)
    'C-O': BASE_BOND_CAPACITANCE / 358.0,  # Carbon-Oxygen (Single)
    'N-H': BASE_BOND_CAPACITANCE / 391.0,  # Nitrogen-Hydrogen
    'O-H': BASE_BOND_CAPACITANCE / 463.0,  # Oxygen-Hydrogen
    'S-S': BASE_BOND_CAPACITANCE / 226.0,  # Disulfide Bridge
    'C-S': BASE_BOND_CAPACITANCE / 259.0,  # Carbon-Sulfur
}

# ---------------------------------------------------------
# 3. FUNCTIONAL GROUP DEFINITIONS
# ---------------------------------------------------------
# Amino acids have standardized input driving ports and output sink ports.

# Amino Group (NH3+) -> High Frequency Source
AMINO_SOURCE_FREQ = "100GHz" # Base bio-resonant drive frequency
AMINO_SOURCE_VOLT = "1V"     # Normalized driving amplitude

# Carboxyl Group (COO-) -> Geometric Sink
CARBOXYL_LOAD_R = f"{np.sqrt(1.256637e-6/8.85418e-12):.2f}Ohm" # Terminate the molecule into vacuum impedance (Derived)

def get_inductance(element: str) -> float:
    """Returns the inductance (pH) of an atomic node."""
    if element not in ATOMIC_INDUCTANCE:
        raise ValueError(f"Unknown organic element for SPICE modeling: {element}")
    return ATOMIC_INDUCTANCE[element]

def get_capacitance(bond: str) -> float:
    """Returns the capacitance (fF) of a covalent stress bond."""
    # Allow reverse lookups (e.g., 'H-C' instead of 'C-H')
    if bond in COVALENT_CAPACITANCE:
        return COVALENT_CAPACITANCE[bond]
    
    rev_bond = f"{bond[-1]}{bond[1:-1]}{bond[0]}"
    if rev_bond in COVALENT_CAPACITANCE:
        return COVALENT_CAPACITANCE[rev_bond]

    raise ValueError(f"Unknown covalent bond for SPICE modeling: {bond}")

if __name__ == "__main__":
    import math
    print(f"--- AVE Organic SPICE Mapper ---")
    print(f"Carbon (C) Inductance: {get_inductance('C'):.2f} pH")
    print(f"Oxygen (O) Inductance: {get_inductance('O'):.2f} pH")
    print(f"C-C Single Bond Capacitance: {get_capacitance('C-C'):.2f} fF")
    print(f"C=O Double Bond Capacitance: {get_capacitance('C=O'):.2f} fF")
    print(f"Molecule driven by terminal {CARBOXYL_LOAD_R} Vacuum Impedance load.")
