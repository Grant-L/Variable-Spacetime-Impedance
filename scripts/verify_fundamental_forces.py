"""
Protocol: Verify AVE Matter Sector Derivations
Checks: Strong Force Tension, W/Z Mass Ratio, Fractional Charges.
"""
import sys
import os
import math

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from ave.matter import particles
from ave.core import geometry

def run():
    print("--- VERIFYING FUNDAMENTAL FORCES ---")
    
    # [cite_start]1. Strong Force String Tension [cite: 458]
    # Prediction: ~160,000 N
    tension = particles.calculate_strong_force_tension()
    target_tension = 160000.0 # approx 1 GeV/fm
    err_tension = abs(tension - target_tension) / target_tension
    
    print(f"Strong Force Tension:")
    print(f"  AVE Theory: {tension:.2e} N")
    print(f"  Lattice QCD: {target_tension:.2e} N")
    if err_tension < 0.05: # 5% margin for the conversion factors
        print("  [PASS] Matches Borromean Strain Model")
    else:
        print(f"  [FAIL] Error {err_tension:.2%}")

    # [cite_start]2. Weak Interaction Mass Ratio [cite: 608]
    # Prediction: Sqrt(7)/3 approx 0.8819
    wz_ratio = particles.calculate_weak_mixing_angle_mass_ratio()
    target_ratio = 80.379 / 91.1876 # Experimental W/Z
    err_ratio = abs(wz_ratio - target_ratio) / target_ratio
    
    print(f"\nWeak Boson Mass Ratio (W/Z):")
    print(f"  AVE Theory: {wz_ratio:.5f} (Acoustic Mode Limit)")
    print(f"  Empirical:  {target_ratio:.5f}")
    if err_ratio < 0.001:
        print("  [PASS] Precision Match (<0.1%)")
    else:
        print(f"  [FAIL] Error {err_ratio:.2%}")

    # [cite_start]3. Fractional Charges [cite: 510]
    # Prediction: 1/3, 2/3
    proton_geo = geometry.BorromeanLinkage()
    charges = proton_geo.charge_fractionalization()
    print(f"\nQuark Charge Fractionalization:")
    print(f"  AVE Theory (Z3 Witten Effect): {['{:.2f}'.format(c) for c in charges]}")
    if 1.0/3.0 in [round(c, 5) for c in charges]:
        print("  [PASS] Recovers 1/3e and 2/3e")
    else:
        print("  [FAIL]")

if __name__ == "__main__":
    run()