import numpy as np
import scipy.integrate as integrate

Ry = 13.6057

# Normalized wavefunctions
def R_1s(r, z):
    return 2.0 * z**1.5 * np.exp(-z*r)

def R_4s(r, z):
    u = z*r/2.0
    L = 1.0 - 3.0/4.0 * u + 1.0/8.0 * u**2 - 1.0/192.0 * u**3
    # Wait, the physics L_3^1(x) = 4 - 4x + x^2 - x^3/18 ?
    # Standard hydrogen: R_40 = 2(Z/4)^1.5 * (1 - 3/4 Zr/a0 + 1/8 (Zr/a0)^2 - 1/192 (Zr/a0)^3) * exp(-Zr/4a0)
    # Let's use the explicit one:
    # u = z*r
    poly = 1.0 - 0.75 * z * r + 0.125 * (z*r)**2 - (1.0/192.0) * (z*r)**3
    return 2.0 * (z/4.0)**1.5 * poly * np.exp(-z*r/4.0)

z = 1.0
print("Norm 1s:", integrate.quad(lambda r: R_1s(r,z)**2 * r**2, 0, np.inf)[0])
print("Norm 4s:", integrate.quad(lambda r: R_4s(r,z)**2 * r**2, 0, np.inf)[0])

# Coulomb integral J(1s, 4s)
def integrand_inner(r2, r1):
    r_greater = max(r1, r2)
    return R_4s(r2, z)**2 * r2**2 / r_greater

def integrand_outer(r1):
    inner = integrate.quad(lambda r2: integrand_inner(r2, r1), 0, np.inf, limit=200)[0]
    return R_1s(r1, z)**2 * r1**2 * inner

J_num = integrate.quad(integrand_outer, 0, np.inf, limit=200)[0]
print(f"Numerical J(1s, 4s) for Z=1: {J_num * 2 * Ry:.4f} eV,  {J_num:.6f} Ha")

import ave.solvers.coupled_resonator as cr
J_ana = cr._coulomb_J_sub(1, 0, 1.0, 4, 0, 1.0)
print(f"Analytical _coulomb_J_sub(1s, 4s): {J_ana:.4f} eV,  {J_ana/(2*Ry):.6f} Ha")
