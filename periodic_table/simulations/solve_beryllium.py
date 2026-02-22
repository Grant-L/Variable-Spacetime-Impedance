import numpy as np
from scipy.optimize import minimize

M_P_RAW = 938.272
M_N_RAW = 939.565
K_MUTUAL = 11.33719
d = 0.85

# Exact Be-9 empirical mass is 9.0121831 * 931.49410242 = 8394.79 MeV
m_empirical = 8394.794

def get_be(outer_offset):
    # Core 1 (Alpha 1) shifted left
    # Core 2 (Alpha 2) shifted right
    # Bridged by 1 neutron at center
    
    alpha_1 = [
        (-outer_offset+d, d, d),
        (-outer_offset-d, -d, d),
        (-outer_offset-d, d, -d),
        (-outer_offset+d, -d, -d)
    ]
    
    alpha_2 = [
        (outer_offset+d, d, d),
        (outer_offset-d, -d, d),
        (outer_offset-d, d, -d),
        (outer_offset+d, -d, -d)
    ]
    
    bridge = [(0,0,0)]
    
    nodes = alpha_1 + alpha_2 + bridge
    
    be = 0
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            dist = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))
            be += K_MUTUAL / dist
    return be

def loss(x):
    # raw mass = 4*p + 5*n
    raw_mass = (4 * M_P_RAW) + (5 * M_N_RAW)
    theo_mass = raw_mass - get_be(x[0])
    return (theo_mass - m_empirical)**2

res = minimize(loss, [2.0])
print("Optimal outer offset:", res.x[0])
print("Theoretical Mass:", (4 * M_P_RAW) + (5 * M_N_RAW) - get_be(res.x[0]))
print("Empirical Mass:", m_empirical)
print("Bridge scaling factor (offset / d):", res.x[0] / d)
