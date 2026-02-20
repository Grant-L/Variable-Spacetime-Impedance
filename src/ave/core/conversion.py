"""
AVE Topo-Kinematic Conversions
Implements Axiom 2: The Isomorphism between Charge and Spatial Dislocation.
Guarantees dimensional homogeneity between Electrical and Continuum Mechanics.
"""
import sys
from pathlib import Path

# Add src directory to path if running as script (before imports)
# This allows the file to be run directly: python src/ave/core/conversion.py
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ave.core import constants as k

def charge_to_length(coulombs):
    """ [C] -> [m] """
    return coulombs / k.XI_TOPO

def length_to_charge(meters):
    """ [m] -> [C] """
    return meters * k.XI_TOPO

def voltage_to_force(volts):
    """ [V] -> [N] """
    return volts * k.XI_TOPO

def force_to_voltage(newtons):
    """ [N] -> [V] """
    return newtons / k.XI_TOPO

def current_to_velocity(amperes):
    """ [A] -> [m/s] """
    return amperes / k.XI_TOPO

def velocity_to_current(v):
    """ [m/s] -> [A] """
    return v * k.XI_TOPO

def resistance_to_viscosity(ohms):
    """ [Ohms] -> [kg/s] (Mechanical Impedance / Fluidic Drag) """
    return ohms * (k.XI_TOPO**2)

def viscosity_to_resistance(kg_per_s):
    """ [kg/s] -> [Ohms] """
    return kg_per_s / (k.XI_TOPO**2)

def inductance_to_mass(henries):
    """ [H] -> [kg] (Inertia as Back-EMF) """
    return henries * (k.XI_TOPO**2)

def mass_to_inductance(kg):
    """ [kg] -> [H] """
    return kg / (k.XI_TOPO**2)

def capacitance_to_compliance(farads):
    """ [F] -> [m/N] (Mechanical Compliance) """
    return farads / (k.XI_TOPO**2)

def compliance_to_capacitance(m_per_n):
    """ [m/N] -> [F] """
    return m_per_n * (k.XI_TOPO**2)