"""
AVE Lepton Sector
Implements the physics of the 3_1 Trefoil Knot (The Electron).
Source: Chapter 5 (The Golden Torus) & Chapter 4.3.1
"""
import sys
from pathlib import Path
import math

# Add src directory to path if running as script
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ave.core import constants as k
from ave.core import geometry

def calculate_theoretical_alpha():
    """
    Derives the Fine Structure Constant purely from Golden Torus geometry.
    This is the "Single Parameter" check - does geometry match experiment?
    Source: Eq 5.3
    """
    torus = geometry.GoldenTorus()
    
    # The topological impedance is the sum of Volumetric, Surface, and Linear terms.
    impedance = torus.topological_impedance()
    
    # Alpha is the inverse of this impedance
    alpha_theoretical = 1.0 / impedance
    return alpha_theoretical

def check_heavy_lepton_instability(mass_ev):
    """
    Checks if a heavy lepton (Muon/Tau) exceeds the 1D Yield Limit.
    Theory: Heavy leptons decay because their localized inductive tension 
    shatters their own topological mirror, making them a 'Leaky Cavity'.
    Source: Chapter 4.3.1 (Particle Decay Paradox)
    """
    # 1. Convert mass (eV) to Joules
    E_joules = mass_ev * k.E_CHARGE
    
    # 2. Calculate Internal Static Tension
    # A 3_1 knot has an ideal ropelength (L/d) of 16.37.
    # Tension = Energy / Length
    ropelength_nodes = 16.37
    knot_length_m = ropelength_nodes * k.L_NODE
    
    internal_tension_N = E_joules / knot_length_m
    
    # 3. Compare to the absolute 1D string yield limit (T_EM)
    # The absolute dynamic yield limit before snapping is T_EM (~0.212 N)
    is_unstable = internal_tension_N > k.T_EM
    
    return is_unstable, internal_tension_N, k.T_EM

if __name__ == "__main__":
    print("==================================================")
    print("AVE LEPTON SECTOR DIAGNOSTICS")
    print("==================================================\n")
    
    alpha = calculate_theoretical_alpha()
    print(f"[1] Derived Geometric Alpha: 1 / {1.0/alpha:.5f}")
    
    print("\n[2] The Leaky Cavity Paradox (Heavy Fermion Decay)")
    
    # The Electron (~0.511 MeV)
    m_e_ev = 510998.95
    unstable_e, t_e, y_e = check_heavy_lepton_instability(m_e_ev)
    print(f"    -> Electron Tension: {t_e:.4f} N (Yield: {y_e:.4f} N)")
    print(f"       Unstable? {unstable_e} (Stable Ground State)")
    
    # The Muon (~105.66 MeV)
    m_mu_ev = 105658375.5
    unstable_mu, t_mu, y_mu = check_heavy_lepton_instability(m_mu_ev)
    print(f"    -> Muon Tension:     {t_mu:.4f} N (Yield: {y_mu:.4f} N)")
    print(f"       Unstable? {unstable_mu} (Violently shatters the lattice limit!)")
    
    print("\n==================================================")