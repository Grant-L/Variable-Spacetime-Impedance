"""
AVE COMPUTATIONAL SOLVER v2.0: 3D Borromean Tensor Trace
------------------------------------------------------------------
A parameter-free finite-element integration of the localized 
orthogonal crossing energy required by the 6^3_2 proton topology.
Strictly bounded by the physical QED Packing Fraction and Mass Stiffening.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def compute_physical_tensor_deficit():
    print("==========================================================")
    print(" AVE 3D TENSOR SOLVER: PURE GEOMETRIC INTEGRATION")
    print("==========================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Engine: {device}")
    
    start_time = time.time()

    # --- 1. SPATIAL GRID ---
    GRID_SIZE = 8.0     
    RESOLUTION = 250    # 15.6 million nodes
    dV = (GRID_SIZE / RESOLUTION)**3
    
    limit = GRID_SIZE / 2.0
    x = torch.linspace(-limit, limit, RESOLUTION, device=device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')

    # --- 2. TOPOLOGICAL GEOMETRY (Axiom 1 Constraints) ---
    # To satisfy Axiom 1, the physical diameter (FWHM) of the tube is exactly 1.0 l_node
    # For a Gaussian exp(-r^2 / 2sigma^2), FWHM = 2 * sqrt(2 * ln(2)) * sigma
    sigma = 1.0 / np.sqrt(8 * np.log(2)) 
    
    # Axiom 1: Hard-sphere minimum distance = 1.0 l_node (Z offset = +/- 0.5)
    z_offset = 0.5
    
    # Base profiles
    V1_b = torch.exp(-(Y**2 + (Z - z_offset)**2) / (2 * sigma**2))
    V2_b = torch.exp(-(X**2 + (Z + z_offset)**2) / (2 * sigma**2))

    # --- 3. AXIOM 4 DIELECTRIC SATURATION ---
    S_base = torch.sqrt(V1_b**2 + V2_b**2)
    max_natural_strain = torch.max(S_base)
    
    # Scale the local topological potential so the geometric maximum 
    # perfectly touches the varactor breakdown limit (0.99995)
    YIELD_LIMIT = 0.99995
    V1 = V1_b * (YIELD_LIMIT / max_natural_strain)
    V2 = V2_b * (YIELD_LIMIT / max_natural_strain)
    
    S_total = torch.sqrt(V1**2 + V2**2)
    S_total = torch.clamp(S_total, 0.0, 0.999999)

    # --- 4. ANALYTICAL GRADIENTS ---
    print("[*] Evaluating pure non-linear spatial gradients...")
    
    dV1_dy = -Y / (sigma**2) * V1
    dV1_dz = -(Z - z_offset) / (sigma**2) * V1
    
    dV2_dx = -X / (sigma**2) * V2
    dV2_dz = -(Z + z_offset) / (sigma**2) * V2
    
    # Cross Product Vector
    cross_x = dV1_dy * dV2_dz
    cross_y = -dV1_dz * dV2_dx
    cross_z = -dV1_dy * dV2_dx
    
    Tensor_Cross_Term = cross_x**2 + cross_y**2 + cross_z**2
    Varactor_Denominator = torch.sqrt(1.0 - S_total**2)
    
    # Pure dimensionless geometric density
    Energy_Density = 0.25 * (Tensor_Cross_Term / Varactor_Denominator)
    
    # Integrate Volume
    Crossing_Volume = (torch.sum(Energy_Density) * dV).item()
    Total_Geometric_Volume = Crossing_Volume * 6
    
    # --- 5. THE PHYSICAL SCALING PROJECTION ---
    ALPHA = 1 / 137.036
    KAPPA_V = 8 * np.pi * ALPHA  # QED Volumetric Packing Fraction (~0.1834)
    MASS_STIFFENING = 1836.15    # Proton-Electron Mass Ratio
    
    # Convert pure geometry to physical mass using AVE framework axioms
    Tensor_Deficit = Total_Geometric_Volume * MASS_STIFFENING * KAPPA_V
    
    TARGET_DEFICIT = 1836.15 - 1162.0  # 674.15 m_e
    
    calc_time = time.time() - start_time
    
    print("\n==========================================================")
    print("                      RESULTS                             ")
    print("==========================================================")
    print(f"Geometric Volume of 1 crossing:        {Crossing_Volume:.4f}")
    print(f"Total Borromean Geometric Volume (x6): {Total_Geometric_Volume:.4f}")
    print("----------------------------------------------------------")
    print(f"Applying Physical Multipliers:")
    print(f"  * Mass Stiffening:         x {MASS_STIFFENING}")
    print(f"  * QED Packing Fraction:    x {KAPPA_V:.4f}")
    print("----------------------------------------------------------")
    print(f"Calculated Tensor Deficit:             {Tensor_Deficit:.2f} m_e")
    print(f"Empirical Target (1836 - 1162):        {TARGET_DEFICIT:.2f} m_e")
    
    error = abs(Tensor_Deficit - TARGET_DEFICIT) / TARGET_DEFICIT * 100
    print(f"Deviation from Empirical Mass:         {error:.3f}%")
    print(f"Integration computed in {calc_time:.3f} seconds.")
    print("==========================================================")

    # --- 6. VISUALIZATION ---
    center_idx = RESOLUTION // 2
    slice_2d = Energy_Density[:, :, center_idx].cpu().numpy()
    
    plt.figure(figsize=(10, 8), facecolor='#0B0F19')
    ax = plt.gca()
    ax.set_facecolor('#0B0F19')
    
    # Power law strictly for visual clarity of the halo
    im = ax.imshow(slice_2d**0.6, cmap='magma', origin='lower', 
                   extent=[-limit, limit, -limit, limit])
    
    plt.title("AVE: The Topological Tensor Halo\n(Mass Generation at a single Borromean Intersection)", 
              color='white', fontsize=14, pad=20, weight='bold')
              
    plt.xlabel(r"Spatial Distance X ($l_{node}$)", color='white', fontsize=12)
    plt.ylabel(r"Spatial Distance Y ($l_{node}$)", color='white', fontsize=12)
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Topological Tensor Strain Density', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig("manuscript/chapters/00_derivations/simulations/outputs/topological_tensor_halo.png", dpi=300, facecolor='#0B0F19')
    print("\n[*] Saved visualization to 'topological_tensor_halo.png'")
    #plt.show()

if __name__ == "__main__":
    compute_physical_tensor_deficit()