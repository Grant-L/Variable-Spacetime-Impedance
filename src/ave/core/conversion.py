"""
AVE Topo-Kinematic Conversions
Implements Axiom 2: The Isomorphism between Charge and Spatial Dislocation.
Source: Chapter 1 (Eq 1.1) & Chapter 2 (Eq 2.1)
"""
from ave.core import constants as k

def charge_to_length(coulombs):
    """
    Converts Electrical Charge (C) to Spatial Dislocation (m).
    Rule: [Q] == [L] scaled by xi_topo.
    """
    return coulombs / k.xi_topo

def length_to_charge(meters):
    """
    Converts Spatial Dislocation (m) to Electrical Charge (C).
    """
    return meters * k.xi_topo

def voltage_to_force(volts):
    """
    Converts Electrical Potential (V) to Mechanical Force (N).
    Rule: F = V * xi_topo
    Source: Chapter 12
    """
    return volts * k.xi_topo

def force_to_voltage(newtons):
    """
    Converts Mechanical Force (N) to Electrical Potential (V).
    """
    return newtons / k.xi_topo

def current_to_velocity(amperes):
    """
    Converts Electrical Current (A) to Kinematic Velocity (m/s).
    Rule: I = v * xi_topo
    """
    return amperes / k.xi_topo

def resistance_to_viscosity(ohms):
    """
    Converts Electrical Resistance (Ohms) to Mechanical Impedance/Drag (kg/s).
    Rule: 1 Ohm = xi_topo^2 kg/s
    Source: Eq 12.6
    """
    return ohms * (k.xi_topo**2)