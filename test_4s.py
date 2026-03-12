import ave.solvers.coupled_resonator as cr

subs = cr._fill_subshells(19)
# Use Ze = Z (unscreened test)
ze = [19.0] * len(subs)

E_4s_bare = -(19.0**2) * cr._RY_EV / 4**2
print(f"4s Bare Binding Energy: {E_4s_bare:.2f} eV")

repulsion = 0.0
# The 4s electron is at idx 5
n_4s, l_4s, c_4s = subs[5]
ze_4s = ze[5]

for idx in range(5): # core subshells
    n_c, l_c, c_core = subs[idx]
    ze_c = ze[idx]
    # Repulsion between one 4s electron and the full core subshell
    J = cr._coulomb_J_sub(n_4s, l_4s, ze_4s, n_c, l_c, ze_c)
    repulsion += J * c_core

print(f"4s Total Core Repulsion J (Unscreened): +{repulsion:.2f} eV")
print(f"Net 4s Energy (Unscreened J): {E_4s_bare + repulsion:.2f} eV")

# What about screened?
ze_screened = cr._scf_z_eff_v2(19, subs)
print(f"\nSCF Ze for 4s: {ze_screened[5]:.2f}")
# SCF non-interacting energy for 4s:
E_4s_scf = (ze_screened[5]**2 - 2.0*19.0*ze_screened[5]) * cr._RY_EV / 4**2
print(f"4s SCF Binding Energy component: {E_4s_scf:.2f} eV")

repulsion_scf = 0.0
for idx in range(5):
    n_c, l_c, c_core = subs[idx]
    # Repulsion between one 4s electron and the full core subshell
    J = cr._coulomb_J_sub(n_4s, l_4s, ze_screened[5], n_c, l_c, ze_screened[idx])
    repulsion_scf += J * c_core

print(f"4s Total Core Repulsion J (Screened): +{repulsion_scf:.2f} eV")
print(f"Net 4s Energy (Screened J): {E_4s_scf + repulsion_scf:.2f} eV")
