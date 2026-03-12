import numpy as np
import scipy.integrate as integrate

Ry = 13.6057

def R_1s(r, z):
    return 2.0 * z**1.5 * np.exp(-z*r)

def R_4s(r, z):
    # Standard Hydrogen 4s Radial Wavefunction
    # R_40(r) = 1/4 * (1/4!)^0.5 * (Z/a0)^1.5 * e^(-Zr/4a0) * L_3^1(Zr/2a0)
    # L_3^1(x) = 4! / (3! 1!) * M(-3, 2, x)
    # = 4 * (1 - 3x/2 + 3x^2/8 - x^3/48)
    # Wait, the physics definition:
    x = z * r / 2.0
    poly = 1.0 - 0.75 * z * r + 0.125 * (z*r)**2 - (1.0/192.0) * (z*r)**3
    return 2.0 * (z/4.0)**1.5 * poly * np.exp(-z*r/4.0)

z = 1.0
norm4s, _ = integrate.quad(lambda r: R_4s(r,z)**2 * r**2, 0, np.inf)
if abs(norm4s - 1.0) > 1e-4:
    print(f"Norm 4s error: {norm4s}")

def integrand_inner(r2, r1):
    r_greater = max(r1, r2)
    return R_4s(r2, z)**2 * r2**2 / r_greater

def integrand_outer(r1):
    inner, _ = integrate.quad(lambda r2: integrand_inner(r2, r1), 0, np.inf, limit=200)
    return R_1s(r1, z)**2 * r1**2 * inner

J_num, _ = integrate.quad(integrand_outer, 0, np.inf, limit=200)
print(f"Numerical J(1s, 4s) for Z=1: {J_num * 2 * Ry:.4f} eV,  {J_num:.6f} Ha")

import sys
sys.path.append('src')
import ave.solvers.coupled_resonator as cr
J_ana = cr._coulomb_J_sub(1, 0, 1.0, 4, 0, 1.0)
print(f"Analytical _coulomb_J_sub(1s, 4s): {J_ana:.4f} eV,  {J_ana/(2*Ry):.6f} Ha")
