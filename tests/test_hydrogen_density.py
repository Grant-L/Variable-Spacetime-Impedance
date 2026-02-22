import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def generate_hydrogen_flux(out_file, title, bound=3.0, grid_res=200):
    x = np.linspace(-bound, bound, grid_res)
    y = np.linspace(-bound, bound, grid_res)
    X, Y = np.meshgrid(x, y)
    
    # Hydrogen-1 is a single Proton at the origin
    nucleons = [{'pos': (0,0,0), 'color': '#ff3366'}]
    
    z_slice = 0.0
    density_field = np.zeros_like(X)
    amplitude = 100.0
    epsilon = 0.5
    
    for n in nucleons:
        cx, cy, cz = n['pos']
        dist_sq = (X - cx)**2 + (Y - cy)**2 + (z_slice - cz)**2
        local_density = amplitude / (dist_sq + epsilon)
        density_field += local_density

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#0f0f0f')
    
    cmap = plt.cm.inferno
    cmap.set_bad(color='#0f0f0f')
    im = ax.imshow(density_field, extent=[-bound, bound, -bound, bound], origin='lower', cmap=cmap, alpha=0.9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.set_label('Topological Vacuum Strain (1/r)', color='white')
    
    grad_y, grad_x = np.gradient(density_field)
    
    ax.streamplot(x, y, grad_x, grad_y, color='#aaaaaa', linewidth=1.2, density=1.5, arrowstyle='->', arrowsize=1.5)
    
    for n in nucleons:
        cx, cy, cz = n['pos']
        ax.scatter(cx, cy, color=n['color'], s=500, marker='+', linewidth=3, alpha=0.8)
        ax.scatter(cx, cy, color=n['color'], s=150, edgecolor=n['color'], facecolor='none', linewidth=2, alpha=0.9)
        
    ax.set_title(title, color='white', fontsize=16, pad=20)
    ax.set_xlabel("X (Topological Node Lengths)", color='white', fontsize=12)
    ax.set_ylabel("Y (Topological Node Lengths)", color='white', fontsize=12)
    ax.tick_params(colors='white')
    
    os.makedirs('tests/outputs', exist_ok=True)
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[*] Saved Density Slice: {out_file}")

if __name__ == "__main__":
    generate_hydrogen_flux('tests/outputs/hydrogen_1_density.png', "Hydrogen-1 (Protium): Static Isotropic Flux")

