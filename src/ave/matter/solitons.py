"""
AVE Topological Soliton Base Library
Provides the foundational physics class for instantiating dynamic, 
continuous topological flux tubes (particles) within the M_A condensate.
"""
import sys
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

# Add src directory to path if running as script
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ave.core import constants as k
from ave.core import conversion

class TopologicalSoliton:
    """
    Base class for any continuous 1D topological knot embedded in 3D space.
    Handles dynamic coordinate tracking, relativistic kinematics, rotations, 
    and spatial field generation.
    """
    def __init__(self, rest_mass_kg, charge_coulombs, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0)):
        self.m0 = rest_mass_kg
        self.charge = charge_coulombs
        self.pos = np.array(position, dtype=np.float64)
        self.vel = np.array(velocity, dtype=np.float64)
        self.rot = np.array(orientation, dtype=np.float64) # Euler angles
        
        # Axiom 1: FWHM of any flux tube is exactly 1.0 l_node
        self.sigma = k.L_NODE / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        
        # Base Inductance (L0 = m0 / xi^2)
        # Identifies Inertia identically as Back-EMF
        self.L0 = conversion.mass_to_inductance(self.m0)
        
        # Internal LC Resonance (Compton Frequency)
        self.omega_c = (self.m0 * k.C**2) / k.H_BAR
        self.internal_phase = 0.0
        
        # Initialize Relativistic state
        self.gamma = 1.0
        self.dynamic_mass = self.m0
        self.momentum = np.zeros(3)
        self._update_kinematics()

    def _update_kinematics(self):
        """
        Derives Special Relativity identically as Non-Linear Inductor Saturation.
        As Kinematic Current (I) -> I_max (c), Inductance (Mass) -> Infinity.
        Source: Chapter 12.2.2
        """
        v_mag = np.linalg.norm(self.vel)
        if v_mag >= k.C:
            raise ValueError("Kinematic velocity exceeds topological signal boundary (c).")
            
        # Map physical velocity to Topo-Kinematic Current
        I = conversion.velocity_to_current(v_mag)
        I_max = conversion.velocity_to_current(k.C)
        ratio = min(I / I_max, 0.9999999999) 
        
        # Lorentz Factor natively emerges from standard electrical engineering non-linear inductance
        self.gamma = 1.0 / math.sqrt(1.0 - ratio**2)
        self.dynamic_inductance = self.L0 * self.gamma
        
        self.dynamic_mass = conversion.inductance_to_mass(self.dynamic_inductance)
        self.momentum = self.dynamic_mass * self.vel

    def step_kinematics(self, dt_lab, external_force=np.array([0.0, 0.0, 0.0])):
        """Advances the internal continuous Hamiltonian state and macroscopic kinematics."""
        # The internal standing wave cycles strictly by Proper Time (tau)
        dt_tau = dt_lab / self.gamma
        self.internal_phase = (self.internal_phase + self.omega_c * dt_tau) % (2.0 * math.pi)
        
        # Apply external forces (dp/dt = F)
        self.momentum += np.array(external_force) * dt_lab
        
        p_mag = np.linalg.norm(self.momentum)
        if p_mag == 0:
            self.vel = np.zeros(3)
        else:
            self.vel = self.momentum / math.sqrt(self.m0**2 + (p_mag/k.C)**2)
            
        self.pos += self.vel * dt_lab
        self._update_kinematics()

    def get_lc_resonance_state(self):
        """
        Evaluates the instantaneous LC tank resonance of the soliton.
        Energy strictly oscillates between Electric (Dielectric Strain)
        and Magnetic (Inductive Flux) at the Compton frequency.
        """
        total_energy = self.dynamic_mass * (k.C**2)
        e_mag = total_energy * (math.cos(self.internal_phase)**2)
        e_elec = total_energy * (math.sin(self.internal_phase)**2)
        
        return {
            "E_total_joules": total_energy,
            "E_magnetic_inductive": e_mag,
            "E_electric_capacitive": e_elec,
            "phase_angle": self.internal_phase
        }

    def _apply_rotations(self, points):
        """Applies Euler rotations to the local knot curve."""
        rx, ry, rz = self.rot
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        
        R_total = Rz @ Ry @ Rx
        return (points @ R_total.T) + self.pos

    def get_parametric_core(self, resolution=500):
        """Must be implemented by child classes to return the 1D knot geometry."""
        raise NotImplementedError

    def generate_spatial_strain_field(self, X, Y, Z, resolution=500):
        """
        Embeds the particle onto a continuous 3D grid.
        Calculates the topological strain (V) applied to the metric.
        """
        curve_local = self.get_parametric_core(resolution)
        curve_global = self._apply_rotations(curve_local)
        
        grid_shape = X.shape
        grid_points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
        
        # KDTree finds the absolute shortest orthogonal distance from the grid to the flux tube
        tree = cKDTree(curve_global)
        distances, _ = tree.query(grid_points, k=1)
        
        strain_flat = np.exp(-(distances**2) / (2.0 * self.sigma**2))
        
        # Enforce Axiom 4 absolute boundary
        return np.clip(strain_flat.reshape(grid_shape), 0.0, 0.99999)