import numpy as np
import ave.solvers.coupled_resonator as cr

experimental_ie = {
    1: 13.598, 2: 24.587, 3: 5.391, 4: 9.322, 5: 8.298, 6: 11.260, 
    7: 14.534, 8: 13.618, 9: 17.422, 10: 21.564, 11: 5.139, 12: 7.646,
    13: 5.985, 14: 8.151, 15: 10.486, 16: 10.360, 17: 12.967, 18: 15.759,
    19: 4.340, 20: 6.113, 21: 6.561, 22: 6.828, 23: 6.746, 24: 6.766,
    25: 7.434, 26: 7.902, 27: 7.881, 28: 7.639, 29: 7.726, 30: 9.394
}

print("Z | Element | Exp (eV) | v2 (DC) | Err% | v3 (DC+AC) | Err%")
print("-" * 65)

for Z in range(1, 31):
    exp = experimental_ie[Z]
    ie_v2 = cr.ionization_energy_v2(Z)
    err_v2 = (ie_v2 - exp) / exp * 100
    
    ie_v3 = cr.ionization_energy_v3(Z)
    err_v3 = (ie_v3 - exp) / exp * 100
    
    symbol = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", 
              "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", 
              "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"][Z-1]
              
    print(f"{Z:2d} | {symbol:2s}      | {exp:8.3f} | {ie_v2:7.3f} | {err_v2:+5.1f}% | {ie_v3:10.3f} | {err_v3:+5.1f}%")

