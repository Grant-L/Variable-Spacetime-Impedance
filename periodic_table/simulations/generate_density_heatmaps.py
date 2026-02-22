import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure the core framework is in PATH
import pathlib
project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))
from periodic_table.simulations.simulate_element import get_nucleon_coordinates

def calculate_vacuum_density(x, y, z, nodes, grid_size=400, bounds=15.0):
    """
    Computes a 2D slice of the 3D continuous 'vacuum strain' density field.
    Density rho(r) ~ sum(1 / r_i) where r_i is distance to each proton/neutron.
    We limit rho to avoid singularities at the exact node centers.
    """
    X, Y = np.meshgrid(np.linspace(-bounds, bounds, grid_size),
                       np.linspace(-bounds, bounds, grid_size))
    
    Z = np.full_like(X, z)
    density = np.zeros_like(X)
    
    for _, (nx, ny, nz) in enumerate(nodes):
        r = np.sqrt((X - nx)**2 + (Y - ny)**2 + (Z - nz)**2)
        r = np.clip(r, 0.4, None)  # Cap peak intensity near the core
        density += 1.0 / r
        
    return X, Y, density

def plot_density_slice(X, Y, density, nodes, z_slice, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
    ax.set_facecolor('black')
    
    # Use inferno colormap for high-energy density visualizations
    vmax_val = 14 if len(nodes) > 10 else 12  # Scale heat for larger nuclei
    im = ax.imshow(density, extent=[X.min(), X.max(), Y.min(), Y.max()],
                   origin='lower', cmap='inferno', alpha=0.9, vmin=0, vmax=vmax_val)
                   
    # Calculate gradient for streamline topological flux
    DY, DX = np.gradient(density)
    
    # Plot continuous energy flow streamlines over the heatmap
    ax.streamplot(X, Y, DX, DY, color='white', linewidth=0.5, density=1.5, arrowsize=0.8)
    
    # Plot distinct node centers that fall roughly within this Z-slice
    for nx, ny, nz in nodes:
        if abs(nz - z_slice) < 5.0: 
            ax.plot(nx, ny, 'wo', markersize=3, alpha=0.8) # White core
            ax.plot(nx, ny, 'co', markersize=6, alpha=0.4) # Cyan glow
            
    ax.set_title(title, color='white', pad=20)
    ax.set_xlabel('Spatial Radius ($d$)', color='white')
    ax.set_ylabel('Spatial Radius ($d$)', color='white')
    ax.tick_params(colors='white')
    
    cbar = plt.colorbar(im, ax=ax, label='Vacuum Strain Density ($\\rho$)')
    cbar.ax.yaxis.label.set_color('white')
    plt.gcf().axes[-1].tick_params(colors='white')
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"[*] Saved: {filename}")

if __name__ == "__main__":
    elements = [
        {"name": "Hydrogen-1",  "Z": 1, "A": 1,  "bounds": 10.0, "cuts": [0]},
        {"name": "Helium-4",    "Z": 2, "A": 4,  "bounds": 10.0, "cuts": [0, 0.81]},
        {"name": "Lithium-7",   "Z": 3, "A": 7,  "bounds": 15.0, "cuts": [0, 9.72]}, # Core & Shell cut
        {"name": "Beryllium-9", "Z": 4, "A": 9,  "bounds": 15.0, "cuts": [0, 5.0]},
        {"name": "Boron-11",    "Z": 5, "A": 11, "bounds": 15.0, "cuts": [0, 11.84]}, # Core & Shell cut
        {"name": "Carbon-12",   "Z": 6, "A": 12, "bounds": 65.0, "cuts": [0]},        # Massive Ring cut
        {"name": "Nitrogen-14", "Z": 7, "A": 14, "bounds": 30.0, "cuts": [0, 5.0]}
    ]
    
    outdir = "periodic_table/figures"
    
    print("--- AVE Unified Density Generator ---")
    for el in elements:
        print(f"Generating slices for {el['name']} (Z={el['Z']})...")
        nodes = get_nucleon_coordinates(el['Z'], el['A'])
        if not nodes:
            print(f"[!] No coordinate definition found for {el['name']}. Skipping.")
            continue
            
        for cut_z in el['cuts']:
            cut_label = "equator" if cut_z == 0 else f"z_pos_{str(cut_z).replace('.','_')}"
            
            # Special case for core vs shell semantic titling on Lith/Boron
            if "Shell cut" in str(el['cuts']) and cut_z != 0:
                 cut_label = "outer_shell"
            elif "Ring cut" in str(el['cuts']):
                 cut_label = "ring"
            
            X, Y, density = calculate_vacuum_density(0, 0, cut_z, nodes, bounds=el['bounds'])
            filename = f"{outdir}/{el['name'].lower().replace('-','_')}_density_{cut_label}.png"
            title = f"{el['name']} Topological Network\nVacuum Strain Density Slice ($Z={cut_z}d$)"
            
            plot_density_slice(X, Y, density, nodes, cut_z, title, filename)
            
    print("--- Generation Complete ---")
