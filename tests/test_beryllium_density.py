import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def generate_density_flux(nucleons, z_slice, out_file, title, bound=5.0, grid_res=250):
    x = np.linspace(-bound, bound, grid_res)
    y = np.linspace(-bound, bound, grid_res)
    X, Y = np.meshgrid(x, y)
    
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
        if abs(cz - z_slice) < 1.0:
            ax.scatter(cx, cy, color=n['color'], s=500, marker='+', linewidth=3, alpha=0.8)
            ax.scatter(cx, cy, color=n['color'], s=150, edgecolor=n['color'], facecolor='none', linewidth=2, alpha=0.9)
        
    ax.set_title(title, color='white', fontsize=16, pad=20)
    ax.set_xlabel("X (Topological Node Lengths)", color='white', fontsize=12)
    ax.set_ylabel("Y (Topological Node Lengths)", color='white', fontsize=12)
    ax.tick_params(colors='white')
    
    # Clean up axis completely for aesthetics if desired, but we'll leave labels on heatmaps.
    os.makedirs('tests/outputs', exist_ok=True)
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[*] Saved Density Slice: {out_file}")

if __name__ == "__main__":
    d = 0.85
    gamma = 3.8259
    d_stretch = d * gamma
    outer = 2.5 * d
    
    nucleus = []
    
    # Alpha 1
    nucleus.append({'pos': (-outer+d_stretch, d_stretch, d_stretch), 'color': '#ff3366'})
    nucleus.append({'pos': (-outer-d_stretch, -d_stretch, d_stretch), 'color': '#ff3366'})
    nucleus.append({'pos': (-outer-d_stretch, d_stretch, -d_stretch), 'color': '#00ffcc'})
    nucleus.append({'pos': (-outer+d_stretch, -d_stretch, -d_stretch), 'color': '#00ffcc'})

    # Alpha 2
    nucleus.append({'pos': (outer+d_stretch, d_stretch, d_stretch), 'color': '#ff3366'})
    nucleus.append({'pos': (outer-d_stretch, -d_stretch, d_stretch), 'color': '#ff3366'})
    nucleus.append({'pos': (outer-d_stretch, d_stretch, -d_stretch), 'color': '#00ffcc'})
    nucleus.append({'pos': (outer+d_stretch, -d_stretch, -d_stretch), 'color': '#00ffcc'})

    # Bridge Neutron
    nucleus.append({'pos': (0, 0, 0), 'color': '#99ffee'})
    
    # 1. Z=0 slice (Equatorial showing the bridge neutron)
    generate_density_flux(nucleus, z_slice=0.0, out_file='tests/outputs/beryllium_9_density_equator.png', 
                          title="Beryllium-9: Equatorial Density Matrix ($Z=0.0$)", bound=8.0)
                          
    # 2. Z=d_stretch slice (Showing the upper stretched alpha structures)
    generate_density_flux(nucleus, z_slice=d_stretch, out_file='tests/outputs/beryllium_9_density_z_pos.png', 
                          title=f"Beryllium-9: Deep Alpha Stretches ($Z={d_stretch:.2f}$)", bound=8.0)
