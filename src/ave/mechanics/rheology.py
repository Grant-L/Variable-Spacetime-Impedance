"""
AVE Rheology
Implements Bingham-Plastic non-Newtonian dynamics, Sagnac-RLVE, and yield limits.
Source: Chapter 4 (Macroscopic Yield) & Chapter 11 (Continuum Fluidics)
"""
import math
from ave.core import constants as k
from ave.mechanics import moduli

def calculate_microscopic_point_yield_kev():
    """Kinetic energy required to shatter the vacuum node via point-collision."""
    tau_yield = moduli.calculate_bingham_yield_stress()
    F_yield = tau_yield * (k.l_node**2)
    
    coulomb_k = 1.0 / (4.0 * math.pi * k.epsilon_0)
    E_k_joules = math.sqrt(F_yield * (k.e_charge**2 * coulomb_k))
    return E_k_joules / (k.e_charge * 1000.0)

def check_heavy_fermion_stability(mass_ev):
    """Checks if static inductive stress per node exceeds Bingham point-yield."""
    energy_per_node_ev = mass_ev * k.alpha_geom 
    yield_limit_ev = calculate_microscopic_point_yield_kev() * 1000.0
    return energy_per_node_ev <= yield_limit_ev, energy_per_node_ev

def evaluate_superfluid_transition(local_shear_stress):
    """Evaluates if local vacuum has transitioned to frictionless superfluid."""
    tau_yield = moduli.calculate_bingham_yield_stress()
    if local_shear_stress > tau_yield:
        return True, 0.0 # Superfluid
    return False, moduli.calculate_kinematic_viscosity()

def calculate_sagnac_rlve_entrainment(omega_rotor, rotor_density):
    """Estimates the topological phase shift signature index for the Sagnac-RLVE."""
    bulk_vac_density = moduli.calculate_bulk_density()
    coupling_ratio = rotor_density / bulk_vac_density
    return omega_rotor * coupling_ratio * k.alpha_geom