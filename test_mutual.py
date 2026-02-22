import numpy as np

m_p_raw = 938.272
m_n_raw = 939.565
d = 0.85

def get_be(outer_mult):
    core = [
        (d, d, d), (-d, -d, d), (-d, d, -d), (d, -d, -d)
    ]
    outer = outer_mult * d
    shell = [
        (outer, -outer, outer), (-outer, -outer, -outer), (outer, outer, -outer)
    ]
    nodes = core + shell
    
    k_mutual = 11.33719
    be = 0
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            pt1 = np.array(nodes[i])
            pt2 = np.array(nodes[j])
            dist = np.linalg.norm(pt1 - pt2)
            be += k_mutual / dist
    return be

from scipy.optimize import minimize
target_be = 39.244

def loss(x):
    return (get_be(x[0]) - target_be)**2

res = minimize(loss, [2.2])
print("Optimal outer multiplier:", res.x[0])
print("BE at optimal:", get_be(res.x[0]))

