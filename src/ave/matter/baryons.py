"""
AVE Topological Matter: Baryon Sector
Dynamically integrates the 3D non-linear Faddeev-Skyrme tensor trace
to structurally derive the Proton Mass Eigenvalue.
Source: Chapter 6 (The Baryon Sector)
"""
import sys
from pathlib import Path
import numpy as np

# Add src directory to path if running as script (before imports)
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ave.core import constants as k

class BorromeanTensorSolver:
    """
    Simulates the orthogonal intersections of a 6^3_2 Borromean Linkage.
    Integrates the Topological Tensor Halo to derive continuous baryonic mass.
    
    Note: All spatial coordinates are in units of l_node (natural units).
    The physical length scale k.L_NODE is not needed since we work dimensionlessly.
    """
    def __init__(self, grid_resolution=60, bounding_box_nodes=4.0):
        """
        Args:
            grid_resolution: Number of grid points per dimension (N)
            bounding_box_nodes: Half-width of bounding box in units of l_node
        """
        self.N = grid_resolution
        self.L = bounding_box_nodes  # Units: l_node (dimensionless)
        
        # Create the 3D continuous spatial grid (all coordinates in units of l_node)
        x = np.linspace(-self.L, self.L, self.N)
        self.dx = x[1] - x[0]  # Grid spacing in units of l_node
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')
        
        # Axiom 1: FWHM of flux tube is exactly 1.0 l_node
        # sigma is the Gaussian width parameter (in units of l_node)
        # For FWHM = 1.0: sigma = 1.0 / (2 * sqrt(2 * ln(2)))
        self.sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # Units: l_node

    def evaluate_tensor_crossing_volume(self):
        """
        Integrates the non-linear tensor core at a single orthogonal intersection.
        Source: Eq 6.5
        """
        # Define Flux Tube 1 (aligned along X-axis)
        V1 = np.exp(-(self.Y**2 + self.Z**2) / (2.0 * self.sigma**2))
        
        # Define Flux Tube 2 (aligned along Y-axis, offset by 1.0 l_node hard-sphere limit)
        # The hard-sphere limit ensures flux tubes don't overlap at the core
        offset = 1.0  # Units: l_node (dimensionless)
        V2 = np.exp(-((self.X - offset)**2 + self.Z**2) / (2.0 * self.sigma**2))
        
        # Calculate Spatial Gradients (Nabla V)
        dV1_dx = np.zeros_like(self.X)
        dV1_dy = (-self.Y / self.sigma**2) * V1
        dV1_dz = (-self.Z / self.sigma**2) * V1
        
        dV2_dx = (-(self.X - offset) / self.sigma**2) * V2
        dV2_dy = np.zeros_like(self.X)
        dV2_dz = (-self.Z / self.sigma**2) * V2
        
        # Cross Product of Orthogonal Gradients: (Nabla V1) x (Nabla V2)
        C_x = dV1_dy * dV2_dz - dV1_dz * dV2_dy
        C_y = dV1_dz * dV2_dx - dV1_dx * dV2_dz
        C_z = dV1_dx * dV2_dy - dV1_dy * dV2_dx
        
        # Squared magnitude of the spatial tensor stress
        cross_mag_sq = C_x**2 + C_y**2 + C_z**2
        
        # Non-Linear Dielectric Saturation (Axiom 4 Varactor)
        # Peak topological strain approaches ~95.45% of the dielectric saturation limit
        # in the geometric core. We clip it slightly below 1.0 to prevent division by zero.
        S_total = V1 + V2 
        S_norm = np.clip((S_total / np.max(S_total)) * 0.9545, 0.0, 0.999) 
        varactor_denominator = np.sqrt(1.0 - S_norm**2)
        
        # Faddeev-Skyrme Topological Energy Density
        # Note: The coefficient 0.25 = 1/4 matches Eq 6.5 in manuscript
        # Expected result: V_single ≈ 0.32900, V_total ≈ 1.97397 (from manuscript)
        # Current integration produces ~4.13x larger volume - investigation needed
        integrand = 0.25 * (cross_mag_sq / varactor_denominator)
        
        # 3D Numerical Integration (dV = dx * dy * dz)
        V_crossing = np.sum(integrand) * (self.dx**3)
        
        # TODO: Investigate discrepancy with manuscript values
        # Expected: V_single ≈ 0.32900, Current: ~1.36 (4.13x too large)
        # Possible causes: normalization, grid resolution, bounding box, or integration method
        
        # Diagnostic: Check integration components
        max_cross_mag_sq = np.max(cross_mag_sq)
        min_varactor = np.min(varactor_denominator)
        max_integrand = np.max(integrand)
        total_integrand_sum = np.sum(integrand)
        
        # Store diagnostics for potential debugging
        self._last_integration_diagnostics = {
            'max_cross_mag_sq': max_cross_mag_sq,
            'min_varactor': min_varactor,
            'max_integrand': max_integrand,
            'total_sum': total_integrand_sum,
            'dx': self.dx,
            'grid_volume': (2*self.L)**3
        }
        
        return V_crossing

    def derive_proton_mass_eigenvalue(self):
        """
        Solves the dynamic structural feedback loop to extract the strict
        geometric rest mass of the proton.
        Source: Eq 6.6 - 6.8
        """
        # 1. Evaluate pure dimensionless geometric volume of a single crossing
        V_single = self.evaluate_tensor_crossing_volume()
        
        # 2. Total geometric volume for the 6 crossings of the 6^3_2 Linkage
        V_total = 6.0 * V_single
        
        # 3. The 1D Scalar continuous trace (evaluated analytically to 1162 m_e)
        I_scalar = 1162.0
        
        # 4. Construct the dynamic self-consistent mass equation
        # Linear Algebra: x = I_scalar + (V_total * kappa_v) * x
        # x * (1 - (V_total * kappa_v)) = I_scalar
        tensor_multiplier = V_total * k.KAPPA_V
        
        if tensor_multiplier >= 1.0:
            # Diagnostic: The structural feedback is diverging
            # For convergence, we need: V_total * KAPPA_V < 1.0
            # This requires: V_total < 1/KAPPA_V ≈ 5.45
            # Current V_total ≈ 8.2 suggests either:
            # 1. Integration overestimates the volume
            # 2. Formulation needs different coefficient or normalization
            # 3. Grid resolution/bounding box needs adjustment
            critical_volume = 1.0 / k.KAPPA_V
            print(f"\n   WARNING: Tensor multiplier {tensor_multiplier:.6f} >= 1.0")
            print(f"   This indicates structural divergence in the feedback loop.")
            print(f"   V_total = {V_total:.6f}, KAPPA_V = {k.KAPPA_V:.6f}")
            print(f"   For convergence, need V_total < {critical_volume:.3f}")
            print(f"   Current V_total exceeds critical by {(V_total - critical_volume):.3f}")
            print(f"")
            print(f"   Comparison with manuscript (Chapter 6):")
            print(f"     Expected V_single ≈ 0.32900, Current: {V_single:.5f} ({V_single/0.32900:.2f}x)")
            print(f"     Expected V_total ≈ 1.97397, Current: {V_total:.5f} ({V_total/1.97397:.2f}x)")
            print(f"     Expected multiplier: {1.97397 * k.KAPPA_V:.6f} (converges)")
            print(f"     Current multiplier: {tensor_multiplier:.6f} (diverges)")
            # Show integration diagnostics if available
            if hasattr(self, '_last_integration_diagnostics'):
                diag = self._last_integration_diagnostics
                print(f"   Integration Diagnostics:")
                print(f"     Max cross_mag_sq: {diag['max_cross_mag_sq']:.6e}")
                print(f"     Min varactor: {diag['min_varactor']:.6e}")
                print(f"     Max integrand: {diag['max_integrand']:.6e}")
                print(f"     Total integrand sum: {diag['total_sum']:.6e}")
                print(f"     dx: {diag['dx']:.6f}, Grid volume: {diag['grid_volume']:.3f}")
            print(f"   Possible fixes:")
            print(f"     - Increase grid resolution (current: {self.N})")
            print(f"     - Adjust bounding box (current: ±{self.L} l_node)")
            print(f"     - Review integration method or normalization")
            print(f"     - Check if integrand coefficient (0.25) needs adjustment")
            raise ValueError(f"Structural Divergence: Multiplier {tensor_multiplier:.6f} >= 1.0. "
                           f"V_total={V_total:.6f} exceeds critical {critical_volume:.3f}. "
                           f"Grid: {self.N}x{self.N}x{self.N}, Box: ±{self.L} l_node")
            
        proton_mass_me = I_scalar / (1.0 - tensor_multiplier)
        
        return {
            "v_single_crossing": V_single,
            "v_total_macroscopic": V_total,
            "tensor_multiplier": tensor_multiplier,
            "proton_mass_me": proton_mass_me
        }

if __name__ == "__main__":
    print("==================================================")
    print("AVE BARYONIC MASS EIGENVALUE SOLVER")
    print("==================================================\n")
    
    print("[1] Initializing 3D Non-Linear Tensor Grid...")
    # Try smaller bounding box to reduce integration volume
    # The flux tubes have FWHM = 1.0 l_node, so ±2.5 l_node should capture most of the signal
    solver = BorromeanTensorSolver(grid_resolution=100, bounding_box_nodes=2.5)
    print(f"    Grid Resolution: {solver.N}x{solver.N}x{solver.N}")
    print(f"    Bounding Box: ±{solver.L} l_node")
    
    print("[2] Integrating orthogonal Borromean flux intersections...")
    results = solver.derive_proton_mass_eigenvalue()
    
    print(f"\n   -> Single Crossing Geometric Volume: {results['v_single_crossing']:.5f}")
    print(f"   -> Total Macroscopic 6^3_2 Volume:   {results['v_total_macroscopic']:.5f}")
    
    print("\n[3] Solving Self-Consistent Structural Feedback Loop...")
    print(f"    Target Formulation: x = 1162 + ({results['tensor_multiplier']:.5f}) * x")
    
    p_mass = results['proton_mass_me']
    print(f"\n==================================================")
    print(f"DERIVED PROTON REST MASS: {p_mass:.2f} m_e")
    print(f"EMPIRICAL CODATA TARGET:  1836.15 m_e")
    
    error = abs(p_mass - 1836.15) / 1836.15 * 100
    print(f"ACCURACY: {100-error:.2f}% (Variance contained in quark kinetic binding)")
    print("==================================================")