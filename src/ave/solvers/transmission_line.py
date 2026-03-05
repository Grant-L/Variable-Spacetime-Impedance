r"""
Transmission Line Network Solver (Scale-Invariant)
====================================================

Domain-agnostic transmission line calculations used at EVERY AVE scale:
  - Protein backbone: ABCD cascade with Y-shunt at Cα junctions
  - PONDER-01 antenna: stub-loaded TL matching
  - Seismic PREM:     layered impedance profile
  - Stellar interiors: radial impedance cascade

The SAME three operators appear at all scales:
  1. ABCD matrix for a TL segment  → propagation
  2. ABCD matrix for a shunt Y     → junction coupling
  3. S₁₁ from total ABCD           → reflection (mismatch)

This module provides two backends:
  - NumPy  (analysis, plotting, debugging)
  - JAX    (gradient-based optimisation, JIT compilation)

Both backends compute the SAME physics.  The JAX version uses
lax.fori_loop for efficient gradient computation through long
cascades (e.g. 3N-1 segments for N-residue proteins).

Examples
--------
>>> from ave.solvers.transmission_line import (
...     abcd_segment, abcd_shunt, abcd_cascade,
...     s11_from_abcd, s11_frequency_sweep,
... )
>>> Z = [3.61, 3.46, 2.94]  # N-Cα, Cα-C, C-N impedances
>>> gamma = [0.1+1j, 0.1+1j, 0.1+1.3j]  # propagation constants
>>> Y = [0.05, 0.0]  # junction shunts
>>> s11 = s11_frequency_sweep(Z, gamma, Y, Z_source=1.0, Z_load=1.0)
"""

import numpy as np
from typing import Optional, Union, Sequence

# ====================================================================
# NumPy backend (analysis, plotting)
# ====================================================================

def abcd_segment(Z_c: complex, gamma_l: complex) -> np.ndarray:
    r"""
    ABCD matrix for a single transmission line segment.

    .. math::
        \begin{bmatrix} A & B \\ C & D \end{bmatrix} =
        \begin{bmatrix}
            \cosh(\gamma\ell)     & Z_c \sinh(\gamma\ell) \\
            \sinh(\gamma\ell)/Z_c & \cosh(\gamma\ell)
        \end{bmatrix}

    This is the SAME matrix at every scale.

    Args:
        Z_c: Characteristic impedance of the segment.
        gamma_l: Complex propagation constant × length (α + jβ)·ℓ.

    Returns:
        2×2 ABCD matrix (complex).
    """
    ch = np.cosh(gamma_l)
    sh = np.sinh(gamma_l)
    return np.array([
        [ch,        Z_c * sh],
        [sh / Z_c,  ch      ],
    ], dtype=complex)


def abcd_shunt(Y: complex) -> np.ndarray:
    r"""
    ABCD matrix for a shunt admittance at a junction.

    .. math::
        \begin{bmatrix} A & B \\ C & D \end{bmatrix} =
        \begin{bmatrix} 1 & 0 \\ Y & 1 \end{bmatrix}

    Args:
        Y: Shunt admittance (scalar, complex).

    Returns:
        2×2 ABCD matrix.
    """
    return np.array([
        [1.0 + 0j,  0.0 + 0j],
        [Y,         1.0 + 0j],
    ], dtype=complex)


def abcd_stub(Z_stub: complex, gamma_l_stub: complex,
              termination: str = 'open') -> np.ndarray:
    r"""
    ABCD matrix for a TL stub attached at a junction.

    A stub is a short TL segment that terminates in either an open
    or short circuit.  It creates frequency-dependent impedance
    at the junction — the mechanism for bandpass/bandstop filtering
    in RF engineering.

    Open stub:   Y_stub = j·tan(βℓ) / Z_stub  (resonant at λ/4)
    Short stub:  Y_stub = -j·cot(βℓ) / Z_stub (resonant at λ/2)

    In protein folding, H-bonds act as stubs: short TL segments
    connecting non-adjacent backbone positions through space.

    Args:
        Z_stub: Characteristic impedance of the stub line.
        gamma_l_stub: Propagation constant × stub length.
        termination: 'open' or 'short'.

    Returns:
        2×2 ABCD matrix (equivalent shunt admittance).
    """
    if termination == 'open':
        # Y_input = tanh(γℓ) / Z_stub
        Y = np.tanh(gamma_l_stub) / Z_stub
    elif termination == 'short':
        # Y_input = 1 / (Z_stub · tanh(γℓ))
        Y = 1.0 / (Z_stub * np.tanh(gamma_l_stub) + 1e-20)
    else:
        raise ValueError(f"Unknown termination: {termination}")
    return abcd_shunt(Y)


def abcd_cascade(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """
    Cascade multiply a sequence of ABCD matrices.

    Args:
        matrices: List of 2×2 ABCD matrices.

    Returns:
        Total 2×2 ABCD matrix.
    """
    result = np.eye(2, dtype=complex)
    for M in matrices:
        result = result @ M
    return result


def s11_from_abcd(M: np.ndarray,
                  Z_source: float = 1.0,
                  Z_load: float = 1.0) -> complex:
    r"""
    Compute S₁₁ (reflection coefficient) from a total ABCD matrix.

    .. math::
        \Gamma = \frac{A + B/Z_L - C Z_S - D}
                      {A + B/Z_L + C Z_S + D}

    This is the SAME formula at every scale:
      - Particle: Γ at Pauli boundary
      - Protein:  S₁₁ at backbone cascade
      - Antenna:  VSWR at feed point
      - Galaxy:   impedance mismatch at core

    Args:
        M: 2×2 ABCD matrix.
        Z_source: Source impedance.
        Z_load: Load impedance.

    Returns:
        Complex reflection coefficient Γ.
    """
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    numer = A + B / Z_load - C * Z_source - D
    denom = A + B / Z_load + C * Z_source + D + 1e-20
    return numer / denom


def s11_power(M: np.ndarray,
              Z_source: float = 1.0,
              Z_load: float = 1.0) -> float:
    """S₁₁ power reflection coefficient |Γ|²."""
    gamma = s11_from_abcd(M, Z_source, Z_load)
    return float(np.real(gamma * np.conj(gamma)))


def s21_from_abcd(M: np.ndarray,
                  Z_source: float = 1.0,
                  Z_load: float = 1.0) -> complex:
    r"""
    Compute S₂₁ (transmission coefficient) from ABCD matrix.

    .. math::
        S_{21} = \frac{2}{A + B/Z_L + C Z_S + D}

    Args:
        M: 2×2 ABCD matrix.
        Z_source: Source impedance.
        Z_load: Load impedance.

    Returns:
        Complex transmission coefficient.
    """
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    denom = A + B / Z_load + C * Z_source + D + 1e-20
    return 2.0 / denom


def s11_frequency_sweep(
    seg_Z: Sequence[float],
    seg_gamma_l: Sequence[complex],
    junction_Y: Optional[Sequence[complex]] = None,
    stubs: Optional[dict] = None,
    freqs: Sequence[float] = (0.5, 0.8, 1.0, 1.3, 2.0),
    Z_source: float = 1.0,
    Z_load: float = 1.0,
) -> dict:
    r"""
    Frequency-swept S₁₁ for a loaded TL cascade.

    Builds the full ABCD cascade at each frequency:
      segment₀ → junction₀ → segment₁ → junction₁ → ...

    Args:
        seg_Z: Characteristic impedance per segment.
        seg_gamma_l: Propagation constant × length per segment (at ω=1).
            Scales with frequency: γℓ(ω) = γℓ(1) × ω.
        junction_Y: Shunt admittance at each junction (N_seg - 1).
        stubs: Dict mapping junction index to stub parameters:
            {idx: {'Z': float, 'gamma_l': complex, 'termination': str}}
        freqs: Frequency points to sweep.
        Z_source: Source impedance.
        Z_load: Load impedance.

    Returns:
        Dict with 's11_power', 's21_phase', 'freqs' arrays.
    """
    N = len(seg_Z)
    if junction_Y is None:
        junction_Y = [0.0] * (N - 1)
    if stubs is None:
        stubs = {}

    s11_powers = []
    s21_phases = []

    for freq in freqs:
        matrices = []
        for i in range(N):
            # Segment ABCD (frequency-scaled propagation)
            gl = seg_gamma_l[i] * freq
            matrices.append(abcd_segment(seg_Z[i], gl))

            # Junction shunt + stubs (between segments)
            if i < N - 1:
                Y_total = junction_Y[i]

                # Add stub admittance if present at this junction
                if i in stubs:
                    s = stubs[i]
                    stub_gl = s['gamma_l'] * freq
                    if s.get('termination', 'open') == 'open':
                        Y_stub = np.tanh(stub_gl) / s['Z']
                    else:
                        Y_stub = 1.0 / (s['Z'] * np.tanh(stub_gl) + 1e-20)
                    Y_total = Y_total + Y_stub

                if Y_total != 0:
                    matrices.append(abcd_shunt(Y_total))

        M_total = abcd_cascade(matrices)
        gamma = s11_from_abcd(M_total, Z_source, Z_load)
        s21 = s21_from_abcd(M_total, Z_source, Z_load)

        s11_powers.append(float(np.real(gamma * np.conj(gamma))))
        s21_phases.append(float(np.angle(s21)))

    return {
        's11_power': np.array(s11_powers),
        's21_phase': np.array(s21_phases),
        'freqs': np.array(list(freqs)),
    }


# ====================================================================
# JAX backend (gradient-based optimisation)
# ====================================================================

def _try_import_jax():
    """Import JAX if available, return (jax, jnp, lax) or None."""
    try:
        import jax
        import jax.numpy as jnp
        from jax import lax
        return jax, jnp, lax
    except ImportError:
        return None


def abcd_cascade_jax(seg_Zc, cosh_arr, sinh_arr, seg_Y, n_segs, n_junctions):
    """
    JAX-compatible ABCD cascade via lax.fori_loop.

    This is the CORE scale-invariant engine used by the protein fold
    solver.  It handles lossy TL segments with shunt admittances at
    junctions.

    Args:
        seg_Zc: (n_segs,) characteristic impedances per segment
        cosh_arr: (n_segs,) cosh(γℓ) per segment (precomputed)
        sinh_arr: (n_segs,) sinh(γℓ) per segment (precomputed)
        seg_Y: (n_junctions,) shunt admittance per junction
        n_segs: Number of TL segments
        n_junctions: Number of junctions

    Returns:
        (4,) array [A, B, C, D] — the total ABCD matrix elements
    """
    _jax_mod = _try_import_jax()
    if _jax_mod is None:
        raise ImportError("JAX not available for cascade_jax")
    jax, jnp, lax = _jax_mod

    init_state = jnp.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j])

    def cascade_step(i, state):
        A, B, C, D = state[0], state[1], state[2], state[3]
        ch = cosh_arr[i]
        sh = sinh_arr[i]
        Zc = seg_Zc[i] + 1e-12

        # TL segment: standard ABCD multiplication
        A_n = A * ch + B * (sh / Zc)
        B_n = A * (Zc * sh) + B * ch
        C_n = C * ch + D * (sh / Zc)
        D_n = C * (Zc * sh) + D * ch

        # Junction shunt admittance
        Y = jnp.where(i < n_junctions,
                      seg_Y[jnp.clip(i, 0, n_junctions - 1)], 0.0)
        C_n = C_n + Y * A_n
        D_n = D_n + Y * B_n

        return jnp.array([A_n, B_n, C_n, D_n])

    return lax.fori_loop(0, n_segs, cascade_step, init_state)


def s11_from_abcd_jax(abcd_state, Z_source, Z_load):
    """
    JAX-compatible S₁₁ extraction from ABCD state vector.

    Args:
        abcd_state: (4,) array [A, B, C, D]
        Z_source: Source impedance (scalar)
        Z_load: Load impedance (scalar)

    Returns:
        (s11_power, s21_phase) tuple
    """
    _jax_mod = _try_import_jax()
    if _jax_mod is None:
        raise ImportError("JAX not available")
    jax, jnp, lax = _jax_mod

    A, B, C, D = abcd_state[0], abcd_state[1], abcd_state[2], abcd_state[3]
    Z0 = Z_source

    numer = A + B / Z_load - C * Z0 - D
    denom = A + B / Z_load + C * Z0 + D + 1e-20
    gamma = numer / denom
    s11_power = jnp.real(gamma * jnp.conj(gamma))

    s21 = 2.0 / denom
    s21_phase = jnp.angle(s21)

    return s11_power, s21_phase
