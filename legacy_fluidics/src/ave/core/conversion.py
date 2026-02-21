"""
AVE Topo-Kinematic Conversions
Implements Axiom 2: The Isomorphism between Charge and Spatial Dislocation.
Guarantees dimensional homogeneity between Electrical and Continuum Mechanics.
"""
from ave.core import constants as k

def charge_to_length(coulombs): return coulombs / k.XI_TOPO
def length_to_charge(meters): return meters * k.XI_TOPO
def voltage_to_force(volts): return volts * k.XI_TOPO
def force_to_voltage(newtons): return newtons / k.XI_TOPO
def current_to_velocity(amperes): return amperes / k.XI_TOPO
def velocity_to_current(v): return v * k.XI_TOPO

def resistance_to_viscosity(ohms): return ohms * (k.XI_TOPO**2)
def viscosity_to_resistance(kg_per_s): return kg_per_s / (k.XI_TOPO**2)
def inductance_to_mass(henries): return henries * (k.XI_TOPO**2)
def mass_to_inductance(kg): return kg / (k.XI_TOPO**2)
def capacitance_to_compliance(farads): return farads / (k.XI_TOPO**2)
def compliance_to_capacitance(m_per_n): return m_per_n * (k.XI_TOPO**2)