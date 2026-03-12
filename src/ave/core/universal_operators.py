"""
Universal Topological Operators
===============================

This module defines the nine fundamental, scale-invariant operators of the
Applied Vacuum Engineering (AVE) framework:

1. Impedance (Z)               — Axiom 1
2. Saturation (S)              — Axiom 4
3. Reflection (Γ)              — Axiom 3
4. Pairwise Energy (U)         — Axioms 1-4
5. Y-Matrix → S-Matrix (Y→S)   — Axiom 3 (multiport)
6. Eigenvalue Target (λ_min)   — Axiom 3 (eigenstate)
7. Spectral Analysis (FFT)     — DSP complement to SPICE
8. Packing Reflection (Γ_pack) — Axioms 3+4 (macroscopic)
9. Steric Reflection (Γ_steric) — Axiom 3 (pairwise exclusion)

These operators are domain-agnostic and should be imported by all downstream
solvers (Nuclear, Fluid, EE, and Protein Folding) to ensure strict adherence
to the core axioms without local redefinitions.
"""

from ave.core.constants import EPS_NUMERICAL, EPS_CLIP, EPS_DIVZERO

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


def universal_reflection(Z1, Z2, eps=EPS_NUMERICAL):
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
        ratio_sq = jnp.clip((d_sat / r) ** 2, 0.0, 1.0 - EPS_CLIP)
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
    dr = 1e-8 * (r if np.isscalar(r) else np.maximum(r, EPS_CLIP))
    U_plus = universal_pairwise_energy(r + dr, K, d_sat)
    U_minus = universal_pairwise_energy(r - dr, K, d_sat)
    return (U_plus - U_minus) / (2.0 * dr)


def universal_ymatrix_to_s(Y, Y0=1.0):
    """
    Operator 5: The Universal Y-Matrix → S-Matrix Conversion

    Converts an N-port nodal admittance matrix [Y] to its scattering
    matrix [S].  This is the multiport generalisation of the reflection
    operator Γ = (Z₂ - Z₁)/(Z₂ + Z₁).

    The conversion follows:
        [S] = (I + [Y]/Y₀)⁻¹ · (I − [Y]/Y₀)

    which is equivalent to:
        [S] = (I − Z₀[Y]) · (I + Z₀[Y])⁻¹     (impedance normalised)

    This operator is domain-agnostic and is used at:
      - Nuclear scale:  K_MUTUAL eigenvalues from nuclear Y-matrix
      - Protein scale:  λ_min(S†S) for fold eigenstate
      - Antenna scale:  S-parameters for HOPF-01 matching

    Args:
        Y:  NxN complex admittance matrix (numpy or jax array)
        Y0: Reference admittance (scalar). Default: 1.0 (normalised)

    Returns:
        S:  NxN complex scattering matrix
    """
    import numpy as np

    is_jax = hasattr(Y, 'device_buffer') or 'jax' in str(type(Y))

    if is_jax:
        import jax.numpy as jnp
        N = Y.shape[0]
        I = jnp.eye(N, dtype=Y.dtype)
        Y_norm = Y / Y0
        A = I + Y_norm
        B = I - Y_norm
        return jnp.linalg.solve(A, B)
    else:
        N = Y.shape[0]
        I = np.eye(N, dtype=Y.dtype)
        Y_norm = Y / Y0
        A = I + Y_norm
        B = I - Y_norm
        return np.linalg.solve(A, B)


def universal_eigenvalue_target(S):
    """
    Operator 6: The Universal Eigenvalue Ground-State Target

    Computes the smallest eigenvalue of S†S for an N-port scattering
    matrix [S].  When λ_min → 0, the network has a zero singular value:
    one mode is perfectly absorbed — the system is in its geometric
    ground state.

    This operator is domain-agnostic:
      - Nuclear scale:  λ_min(S†S) = 0 → nuclear binding eigenstate
      - Protein scale:  λ_min(S†S) = 0 → native fold
      - Antenna scale:  λ_min(S†S) = 0 → impedance-matched resonance

    Args:
        S:  NxN complex scattering matrix (numpy or jax array)

    Returns:
        lambda_min:  Smallest eigenvalue of S†S (real, ≥ 0)
    """
    import numpy as np

    is_jax = hasattr(S, 'device_buffer') or 'jax' in str(type(S))

    if is_jax:
        import jax.numpy as jnp
        SdS = jnp.conj(S.T) @ S
        eigenvalues = jnp.linalg.eigvalsh(SdS)
        return eigenvalues[0]  # smallest eigenvalue
    else:
        SdS = np.conj(S.T) @ S
        eigenvalues = np.linalg.eigvalsh(SdS)
        return eigenvalues[0]


def universal_spectral_analysis(Z_sequence):
    """
    Operator 7: The Universal Impedance Spectral Analyser

    Computes the spatial Fourier transform of a 1D impedance profile.
    Returns the mode amplitudes and dominant spatial frequencies.

    For a protein backbone:
      - Peak at k ≈ N/3.7 → α-helix periodicity (Q/2 ≈ 3.7 residues/turn)
      - Peak at k ≈ N/2.0 → β-sheet periodicity (2 residues/strand)
      - DC component (k=0) → mean impedance level

    For any 1D impedance sequence:
      - Peaks identify resonant mode spacings
      - Power spectrum P(k) = |FFT|² gives the spatial PSD
      - Autocorrelation R(n) = IFFT(P) gives correlation length

    This is the DSP complement to the time-domain SPICE integrator.

    Args:
        Z_sequence:  1D array of impedance values (real or complex)

    Returns:
        dict with keys:
          'spectrum':     Complex FFT coefficients
          'power':        Power spectral density |FFT|²
          'frequencies':  Spatial frequency indices (0 to N-1)
          'autocorr':     Spatial autocorrelation function
          'dominant_k':   Top 5 dominant spatial frequencies (by power)
          'dominant_periods': Corresponding spatial periods (residues)
    """
    import numpy as np

    Z = np.asarray(Z_sequence, dtype=complex)
    N = len(Z)

    # FFT of the impedance sequence
    spectrum = np.fft.fft(Z)

    # Power spectral density
    power = np.abs(spectrum) ** 2

    # Autocorrelation via Wiener-Khinchin
    autocorr = np.real(np.fft.ifft(power))
    autocorr /= autocorr[0] if autocorr[0] != 0 else 1.0  # normalise

    # Dominant modes (skip DC at k=0)
    k_indices = np.arange(N)
    power_no_dc = power.copy()
    power_no_dc[0] = 0  # mask DC
    top_k = np.argsort(power_no_dc)[::-1][:5]
    top_periods = np.where(top_k > 0, N / top_k, np.inf)

    return {
        'spectrum': spectrum,
        'power': power,
        'frequencies': k_indices,
        'autocorr': autocorr,
        'dominant_k': top_k,
        'dominant_periods': top_periods,
    }


def universal_packing_reflection(Rg_sq, N, r_node, eta_eq):
    """
    Operator 8: The Universal Packing Reflection Coefficient

    Computes the macroscopic reflection coefficient Γ_pack that measures
    how far a confined system is from its Axiom 4 equilibrium packing.

    FULL DERIVATION CHAIN (zero free parameters):
    ─────────────────────────────────────────────
    α  = 7.2973e-3           (Axiom 2: fine-structure constant, INPUT)
    P_C = 8πα ≈ 0.1834       (Axiom 4: volumetric percolation threshold)
    ν   = 2/7                (Axiom 3: Poisson ratio → 7 compliance modes)

    Of the 7 lattice compliance modes, only 5 are TRANSVERSE (spatial).
    The 2 longitudinal modes carry energy along the chain but do not
    contribute to 3D inter-element packing contacts.

    η_eq = P_C × (1 - ν) = P_C × 5/7 ≈ 0.1310

    For a system of N nodes, finite-size correction:
        η(N) = η_eq × (1 - 1/N)

    Node radius r_node is also axiom-derived (for protein):
        d₀ = 3.80 Å      (measured BC: backbone pitch)
        J  = (1/√3)(1+P_C)  (sp³ projection + packing correction)
        r_node = d₀ × J / 2  (half the steric exclusion diameter)
        = 1.298 Å

    Steps:
        1. η(N) = η_eq × (1 - 1/N)            (Axiom 4 + finite-size)
        2. V_node = (4/3)π r_node³             (geometry)
        3. R_target = (3NV / (4πη))^(1/3)      (uniform sphere model)
        4. Rg_target = √(3/5) × R_target       (sphere Rg)
        5. Γ_pack = (Rg - Rg_target)/(Rg + Rg_target)  (Axiom 3)

    Cross-scale application:
      - Protein:   r_node = R_NODE = 1.298 Å, η_eq = ETA_EQ = 0.131
      - Nuclear:   r_node = d_proton, η_eq = 1 (close-packed)
      - Fluid:     r_node = molecular radius, η_eq from lattice type

    Args:
        Rg_sq:    Radius of gyration SQUARED (same units as r_node²)
        N:        Number of nodes in the system
        r_node:   Axiom-derived node radius (R_NODE for protein)
        eta_eq:   Equilibrium packing fraction (ETA_EQ = P_C × 5/7 for protein)

    Returns:
        Gamma_pack_sq: Γ_pack² — the macroscopic packing mismatch power.
                       Add directly to any eigenvalue loss function.
    """
    import numpy as np

    is_jax = hasattr(Rg_sq, 'device_buffer') or 'jax' in str(type(Rg_sq))

    if is_jax:
        import jax.numpy as jnp
        _max = jnp.maximum
        _sqrt = jnp.sqrt
        _pi = jnp.pi
    else:
        _max = np.maximum
        _sqrt = np.sqrt
        _pi = np.pi

    # Axiom 4: equilibrium packing fraction with finite-size correction
    eta_target = eta_eq * (1.0 - 1.0 / _max(N, 2.0))

    # Geometry: volume per node → target sphere radius → target Rg
    V_res = (4.0 / 3.0) * _pi * r_node**3
    R_target = (3.0 * N * V_res / (4.0 * _pi * eta_target + EPS_NUMERICAL)) ** (1.0 / 3.0)
    Rg_target = _sqrt(3.0 / 5.0) * R_target

    # Axiom 3: packing reflection coefficient
    Rg_actual = _sqrt(Rg_sq + EPS_NUMERICAL)
    Gamma_pack = (Rg_actual - Rg_target) / (Rg_actual + Rg_target + EPS_NUMERICAL)

    return Gamma_pack ** 2


def universal_steric_reflection(dists, R_excl, mask):
    """
    Operator 9: The Universal Steric Reflection Coefficient

    Computes the pairwise steric exclusion reflection coefficient using
    Axiom 3 applied at the atomic/node level.

    DERIVATION:
    ───────────
    Axiom 3 states that the reflection coefficient at any impedance
    boundary is:
        Γ = (Z₁ - Z₂) / (Z₁ + Z₂)

    For steric exclusion, the "impedance boundary" is the Pauli exclusion
    sphere. When two nodes approach closer than their exclusion distance R,
    the overlap creates an impedance mismatch proportional to the fractional
    violation:

        Γ_ij = max(0, (R_excl - d_ij) / (R_excl + d_ij))

    Properties:
      - d ≥ R:  Γ = 0     (no violation, no reflection)
      - d = 0:  Γ = 1     (total overlap, total reflection)
      - d = R/2: Γ = 1/3  (partial overlap)
      - Γ² ∈ [0, 1]       — same units as Ops 6 and 8

    The total steric reflection is the average Γ² over all
    non-bonded pairs:
        ⟨Γ²_steric⟩ = (1/N_pairs) Σ_{i<j} Γ_ij²

    This operator is domain-agnostic:
      - Protein:  R = R_STERIC_CC, d = Cα distances, mask = |i-j| ≥ 3
      - Nuclear:  R = d_proton, d = nucleon distances
      - Fluid:    R = molecular radius, d = particle distances

    Args:
        dists:   (N, N) distance matrix between all node pairs
        R_excl:  Exclusion radius (scalar or (N,N) matrix for heterogeneous)
        mask:    (N, N) boolean mask of non-bonded pairs to consider

    Returns:
        Gamma_steric_sq: ⟨Γ²⟩ — average pairwise steric mismatch power.
                         Add directly to any eigenvalue loss function.
    """
    import numpy as np

    is_jax = hasattr(dists, 'device_buffer') or 'jax' in str(type(dists))

    if is_jax:
        import jax.numpy as jnp
        _max = jnp.maximum
        _sum = jnp.sum
        _where = jnp.where
        _triu = jnp.triu
    else:
        _max = np.maximum
        _sum = np.sum
        _where = np.where
        _triu = np.triu

    # Axiom 3: pairwise reflection coefficient
    gamma = _max(0.0, (R_excl - dists) / (R_excl + dists + EPS_NUMERICAL))
    gamma = _where(mask, gamma, 0.0)

    # Upper triangle to avoid double-counting
    gamma_upper = _triu(gamma, k=1)

    # Average over non-bonded pairs
    n_pairs = _max(1.0, _sum(_triu(mask.astype(float), k=1)))

    return _sum(gamma_upper ** 2) / n_pairs
