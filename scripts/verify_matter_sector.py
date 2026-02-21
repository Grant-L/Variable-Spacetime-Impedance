"""
Protocol: Verify Matter Sector (Split Architecture)
Checks: Lepton Alpha, Baryon Tension, Neutrino Chirality.
"""

import os
import sys

import scipy.constants as const

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ave.matter import baryons, leptons, neutrinos


def run():
    print("--- VERIFYING MATTER SECTOR (MODULAR) ---")

    # 1. LEPTONS: Geometric Alpha
    alpha_calc = leptons.calculate_theoretical_alpha()
    target_alpha = const.fine_structure
    print("\n[Leptons] Fine Structure Constant:")
    print(f"  AVE Geometry: {alpha_calc:.9f}")
    print(f"  CODATA:       {target_alpha:.9f}")

    # 2. BARYONS: Strong Force
    force = baryons.calculate_strong_force_tension()
    print("\n[Baryons] Strong Force Tension:")
    print(f"  AVE Theory: {force:.2e} N")

    # 3. NEUTRINOS: Chiral Exclusion
    print("\n[Neutrinos] Chiral Permissibility:")
    left = neutrinos.check_chirality_permission("left")
    right = neutrinos.check_chirality_permission("right")
    print(f"  Left-Handed:  {'ALLOWED' if left else 'FORBIDDEN'}")
    print(f"  Right-Handed: {'ALLOWED' if right else 'FORBIDDEN'}")

    if left and not right:
        print("  [PASS] Matches Standard Model Parity Violation")
    else:
        print("  [FAIL] Parity Violation logic incorrect")


if __name__ == "__main__":
    run()
