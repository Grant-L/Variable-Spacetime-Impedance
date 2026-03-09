"""
Universal Topological Operators
===============================

This module defines the three fundamental, scale-invariant operators of the
Applied Vacuum Engineering (AVE) framework:

1. Impedance (Z)
2. Saturation (S)
3. Reflection (Gamma)

These operators are domain-agnostic and should be imported by all downstream 
solvers (Nuclear, Fluid, EE, and Protein Folding) to ensure strict adherence 
to the core axioms without local redefinitions.
"""

def universal_impedance(mu, eps):
    """
    Operator 1: The Universal Impedance Operator (Z)
    Defines the resistance of an arbitrary medium to a propagating wave.
    
    Args:
        mu: Inertial/magnetic density (e.g., mu_0, fluid shear resistance, inductance)
        eps: Elastic compliance (e.g., eps_0, fluid density, capacitance)
        
    Returns:
        Z: The characteristic impedance of the medium.
    """
    # Duck-typing allows this to work with both numpy and jax.numpy
    return (mu / eps) ** 0.5


def universal_saturation(A, A_yield):
    """
    Operator 2: The Universal Saturation Operator (S)
    Imposes the geometric percolation limit of the 3D lattice. Strain cannot
    increase infinitely; as the limit is approached, the metric non-linearly stiffens.
    
    Args:
        A: The current strain amplitude (e.g., Voltage, Velocity, Fluid Stress)
        A_yield: The absolute topological yield limit of the domain
        
    Returns:
        S: The saturation factor (real-valued in [0, 1]).
    """
    # Clip the ratio to a maximum of 1.0 to prevent imaginary roots (metric breakdown)
    # Using simple operators to support both raw floats, numpy arrays, and jax arrays.
    # Note: duck-typing assumes the caller handles the appropriate jnp/np where
    # branching/clipping is complex. A simple algebraic form is best for cross-compatibility.
    import numpy as np
    
    # Try to use JAX if the input is a JAX array, otherwise use NumPy
    is_jax = hasattr(A, 'device_buffer') or 'jax' in str(type(A))
    
    if is_jax:
        import jax.numpy as jnp
        ratio = jnp.clip(A / A_yield, -1.0, 1.0)
        return jnp.sqrt(1.0 - ratio**2)
    else:
        ratio = np.clip(A / A_yield, -1.0, 1.0)
        return np.sqrt(1.0 - ratio**2)


def universal_reflection(Z1, Z2, eps=1e-12):
    """
    Operator 3: The Universal Reflection Operator (Gamma)
    Governs how much energy is transferred versus reflected when a wave
    encounters a boundary between two topological impedances (Z1 -> Z2).
    
    Args:
        Z1: The characteristic impedance of the source/incident medium
        Z2: The characteristic impedance of the target medium
        eps: Small numerical constant to prevent div-by-zero, especially
             useful for auto-differentiation in JAX loss functions.
        
    Returns:
        Gamma: The reflection coefficient (-1.0 to 1.0).
    """
    return (Z2 - Z1) / (Z2 + Z1 + eps)


def universal_pairwise_energy(r, K, d_sat):
    """
    Operator 4: Full 3-Regime Pairwise Potential (Impedance-Based)

    Computes the interaction energy between two nodes at separation r
    using the FULL impedance matching dynamics of the saturated lattice.

    The three regimes are encoded through the local impedance:

    1. LINEAR (r >> d_sat): Z ≈ Z₀, Γ ≈ 0 → U ≈ -K/r
    2. NON-LINEAR (r ~ d_sat): Z rises, partial reflection → reduced coupling
    3. SATURATED (r ≤ d_sat): Z → ∞, Γ → 1 → repulsive wall (Pauli exclusion)

    The potential is:
        U(r) = -(K/r) × (T² - R²)
    where:
        A(r)  = d_sat/r           (strain amplitude, normalized to yield at d_sat)
        Z(r)  = Z₀ / (1-A²)^¼    (impedance at strain, from scale_invariant)
        Γ(r)  = (Z-Z₀)/(Z+Z₀)    (reflection coefficient)
        R² = Γ², T² = 1 - Γ²     (power reflection/transmission)

    This naturally produces:
        - No equilibrium from the potential alone (monotonic beyond wall)
        - Equilibrium comes from the eigenvalue (5-step regime boundary method)
        - Repulsive wall at exactly r = d_sat with no ad-hoc parameters

    Args:
        r: Separation distance (scalar or array, same units as d_sat)
        K: Coupling constant (K_MUTUAL, αℏc, Gm², etc.)
        d_sat: Saturation radius (D_PROTON, Slater radius, r_s, etc.)

    Returns:
        U: Pairwise energy. Negative = attractive, positive = repulsive wall.
    """
    import numpy as np

    is_jax = hasattr(r, 'device_buffer') or 'jax' in str(type(r))
    if is_jax:
        import jax.numpy as jnp
        ratio_sq = jnp.clip((d_sat / r) ** 2, 0.0, 1.0 - 1e-15)
        # Impedance ratio: Z/Z₀ = 1/(1-A²)^(1/4) where A² = ratio_sq
        S_quarter = (1.0 - ratio_sq) ** 0.25
        Z_ratio = 1.0 / S_quarter  # Z_local / Z_0
        Gamma = (Z_ratio - 1.0) / (Z_ratio + 1.0)
        Gamma_sq = Gamma ** 2
        T_sq = 1.0 - Gamma_sq
        return -(K / r) * (T_sq - Gamma_sq)
    else:
        if np.isscalar(r):
            ratio_sq = (d_sat / r) ** 2
            if ratio_sq >= 1.0:
                # Deep inside saturation: Γ → 1, T² → 0, R² → 1
                # U → +(K/r) (full reflection = Pauli wall)
                return K / r
            S_quarter = (1.0 - ratio_sq) ** 0.25
            Z_ratio = 1.0 / S_quarter
            Gamma = (Z_ratio - 1.0) / (Z_ratio + 1.0)
            Gamma_sq = Gamma ** 2
            return -(K / r) * (1.0 - 2.0 * Gamma_sq)
        else:
            r = np.asarray(r, dtype=float)
            result = np.zeros_like(r)
            ratio_sq = (d_sat / r) ** 2
            wall = ratio_sq >= 1.0
            ok = ~wall
            # Saturated wall: Pauli repulsion
            result[wall] = K / r[wall]
            # Dynamic regime
            S_quarter = (1.0 - ratio_sq[ok]) ** 0.25
            Z_ratio = 1.0 / S_quarter
            Gamma = (Z_ratio - 1.0) / (Z_ratio + 1.0)
            Gamma_sq = Gamma ** 2
            result[ok] = -(K / r[ok]) * (1.0 - 2.0 * Gamma_sq)
            return result


def universal_pairwise_gradient(r, K, d_sat):
    """
    Analytical gradient (dU/dr) of the full 3-regime impedance potential.

    For U(r) = -(K/r)(1 - 2Γ²), computed via numerical derivative
    for correctness (the analytical form is complex due to the
    impedance chain).

    Sign convention: positive = repulsive force, negative = attractive.

    Args:
        r: Separation distance
        K: Coupling constant
        d_sat: Saturation radius

    Returns:
        dU_dr: Gradient of the pairwise potential.
    """
    import numpy as np
    dr = 1e-8 * (r if np.isscalar(r) else np.maximum(r, 1e-15))
    U_plus = universal_pairwise_energy(r + dr, K, d_sat)
    U_minus = universal_pairwise_energy(r - dr, K, d_sat)
    return (U_plus - U_minus) / (2.0 * dr)

