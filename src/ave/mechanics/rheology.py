"""
AVE Rheology
Implements the Bingham-Plastic non-Newtonian dynamics of the vacuum condensate,
the Sagnac-RLVE falsification test, and the microscopic point-yield threshold.
Source: Chapter 4 (Macroscopic Yield) & Chapter 11 (Continuum Fluidics)
"""
import math
from ave.core import constants as k
from ave.mechanics import moduli

def calculate_microscopic_point_yield_kev():
    """
    Calculates the kinetic energy required for two ions to shatter the 
    vacuum node via direct point-collision (The Particle Decay Paradox threshold).
    Source: Eq 4.10
    """
    tau_yield = moduli.calculate_bingham_yield_stress()
    F_yield = tau_yield * (k.l_node**2)
    
    # Coulomb constant: 1 / (4 * pi * epsilon_0)
    coulomb_k = 1.0 / (4.0 * math.pi * k.epsilon_0)
    
    # E_k = sqrt( F_yield * (e^2 / 4*pi*eps_0) )
    E_k_joules = math.sqrt(F_yield * (k.e_charge**2 * coulomb_k))
    
    # Convert Joules to keV
    E_k_kev = E_k_joules / (k.e_charge * 1000.0)
    return E_k_kev

def check_heavy_fermion_stability(mass_ev):
    """
    Checks if the static inductive stress per node exceeds the Bingham point-yield limit.
    Source: Section 4.3.1 (Resolving the Heavy Fermion Paradox)
    """
    # Distribute the rest mass energy over the topological core (Q-factor = alpha^-1)
    energy_per_node_ev = mass_ev * k.alpha_geom 
    yield_limit_ev = calculate_microscopic_point_yield_kev() * 1000.0
    
    if energy_per_node_ev > yield_limit_ev:
        return False, energy_per_node_ev  # Unstable (Decays)
    else:
        return True, energy_per_node_ev   # Stable

def evaluate_superfluid_transition(local_shear_stress):
    """
    Evaluates if the local vacuum has transitioned to a frictionless superfluid.
    Condition: tau_local > tau_yield
    Source: Chapter 11.2 (The Bingham Plastic Transition)
    """
    tau_yield = moduli.calculate_bingham_yield_stress()
    is_superfluid = local_shear_stress > tau_yield
    
    if is_superfluid:
        return True, 0.0  # Frictionless slipstream
    else:
        return False, moduli.calculate_kinematic_viscosity()

def calculate_sagnac_rlve_entrainment(omega_rotor, rotor_radius, rotor_density):
    """
    Estimates the topological phase shift signature index for the Sagnac-RLVE.
    Source: Section 11.2.1
    """
    bulk_vac_density = moduli.calculate_bulk_density()
    coupling_ratio = rotor_density / bulk_vac_density
    
    # Expected frame dragging angular velocity
    omega_frame = omega_rotor * coupling_ratio * k.alpha_geom
    return omega_frame