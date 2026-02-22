import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import pathlib
project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

from periodic_table.simulations.simulate_element import get_nucleon_coordinates

def calculate_vacuum_density(nodes, X, Y, z_slice=0.0):
    density_field = np.zeros_like(X)
    amplitude, epsilon = 100.0, 0.5
    for cx, cy, cz in nodes:
        dist_sq = (X - cx)**2 + (Y - cy)**2 + (z_slice - cz)**2
        density_field += amplitude / (dist_sq + epsilon)
    return density_field

Z, A = 12, 24
bound, grid_res = 130.0, 100
name = "magnesium_24"
title = "Magnesium-24 ($^{24}Mg$): $6\\alpha$ Octahedron (Z=0 Equatorial Cross-Section)"

nodes = get_nucleon_coordinates(Z, A)
if not nodes:
    print(f"Error: No coordinates")
    sys.exit(1)

x = np.linspace(-bound, bound, grid_res)
y = np.linspace(-bound, bound, grid_res)
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#0f0f0f')
ax.set_facecolor('#0f0f0f')

density = calculate_vacuum_density(nodes, X, Y, 0.0)
cmap = plt.cm.inferno
cmap.set_bad(color='#0f0f0f')
ax.imshow(density, extent=[-bound, bound, -bound, bound], origin='lower', cmap=cmap, alpha=0.9, vmin=0.0)

grad_y, grad_x = np.gradient(density)
ax.streamplot(x, y, grad_x, grad_y, color='#aaaaaa', linewidth=1.2, density=1.5, arrowstyle='->', arrowsize=1.5)

# Render nodes, scaling by visual depth
for cx, cy, cz in nodes:
    depth_scale = np.exp(-np.abs(cz / (bound/3.0)))
    ax.scatter(cx, cy, color='#00ffcc', s=300 * depth_scale, marker='+', linewidth=2, alpha=0.8)
    ax.scatter(cx, cy, color='#00ffcc', s=100 * depth_scale, edgecolor='#00ffcc', facecolor='none', linewidth=1.5, alpha=0.9)
            
ax.set_title(title, color='white', fontsize=16, pad=20)
ax.tick_params(colors='white')
        
outdir = "periodic_table/figures"
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, f"{name}_dynamic_flux.png"), facecolor=fig.get_facecolor(), dpi=300)
print(f"[*] Generated {name}_dynamic_flux.png")
