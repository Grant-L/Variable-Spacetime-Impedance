import numpy as np
from scipy.optimize import minimize

M_P_RAW = 938.272
M_N_RAW = 939.565
K_MUTUAL = 11.33719
d_ideal = 0.85

m_empirical = 8394.794

def get_be(gamma, outer_offset):
    d = d_ideal * gamma
    alpha_1 = [
        (-outer_offset+d, d, d), (-outer_offset-d, -d, d),
        (-outer_offset-d, d, -d), (-outer_offset+d, -d, -d)
    ]
    alpha_2 = [
        (outer_offset+d, d, d), (outer_offset-d, -d, d),
        (offset_2_fixed, d, -d), (outer_offset+d, -d, -d)
    ]
    # fix typo in alpha_2:
    alpha_2 = [
        (outer_offset+d, d, d), (outer_offset-d, -d, d),
        (outer_offset-d, d, -d), (outer_offset+d, -d, -d)
    ]
    bridge = [(0,0,0)]
    nodes = alpha_1 + alpha_2 + bridge
    
    be = 0
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            pt1 = np.array(nodes[i])
            pt2 = np.array(nodes[j])
            dist = np.linalg.norm(pt1 - pt2)
            be += K_MUTUAL / dist
    return be

def loss(x):
    gamma, bridge = x
    raw_mass = (4 * M_P_RAW) + (5 * M_N_RAW)
    theo_mass = raw_mass - get_be(gamma, bridge)
    return (theo_mass - m_empirical)**2

# We want bridge to be relatively close, say 2.5 * d_ideal
# We expect gamma > 1.0 because the alpha cores must stretch to lose binding energy.
res = minimize(loss, [1.02, 2.5], bounds=[(1.0, 1.2), (1.5, 5.0)])
print("Optimal Alpha stretch (gamma):", res.x[0])
print("Optimal Bridge offset (d):", res.x[1])
print("Theo mass:", (4 * M_P_RAW) + (5 * M_N_RAW) - get_be(res.x[0], res.x[1]))

