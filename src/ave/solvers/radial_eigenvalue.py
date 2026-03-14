"""
Radial Eigenvalue Solver
========================

Finds the energy eigenvalue of an electron (n, l) in a multi-electron
atom by solving the radial standing wave condition on the vacuum lattice.

This is the atomic-scale instance of the universal Op3 → Op6 chain:
    1. Build V_eff(r) from Axiom 2 (Coulomb) + Axiom 1 (angular momentum)
    2. Compute local wavenumber k(r) from Axiom 4 (soliton dispersion)
    3. Accumulate phase ∫k dr in each region (Sommerfeld integral)
    4. Compute reflection coefficient Γ at each shell boundary (Op3)
    5. Find E where total phase = n_r·π (Op6, root-finding)

Same algorithm used at nuclear, antenna, and galactic scales.
~65 lines of new physics; all operators and constants reused.
"""

import numpy as np
from scipy import integrate

from ave.core.constants import ALPHA, HBAR, C_0, M_E, A_0, RY_EV, e_charge
from ave.core.universal_operators import universal_reflection


# ---------------------------------------------------------------------------
# Step 1: Piece-wise radial potential
# ---------------------------------------------------------------------------

def _z_net(r, Z, shells):
    """Effective nuclear charge at radius r — STEP FUNCTION (legacy).

    Gauss's law (Axiom 2): each inner shell at R_a with N_a electrons
    reduces the enclosed charge by N_a when r > R_a.
    """
    z = float(Z)
    for R_a, N_a in shells:
        if r > R_a:
            z -= N_a
    return max(z, 0.0)


def _enclosed_charge_fraction(r, Z_eff):
    """Smooth enclosed-charge fraction σ(r) for a 1s (n=1, l=0) shell.

    From Eq. (smooth_screening) in the LaTeX:
        σ(r) = 1 - exp(-2·Z_eff·r/a₀)·(1 + 2·Z_eff·r/a₀ + 2·Z_eff²·r²/a₀²)

    This is the exact integral of the normalised hydrogenic 1s density
    from 0 to r.  Derived from Axiom 2 (Coulomb → 1s solution) +
    Axiom 2 (Gauss's law on the resulting density).

    Args:
        r:      Radial position [m].
        Z_eff:  Effective charge seen by the inner-shell soliton.

    Returns:
        σ:  Enclosed charge fraction (0 → 1) [dimensionless].
    """
    x = 2.0 * Z_eff * r / A_0  # dimensionless
    return 1.0 - np.exp(-x) * (1.0 + x + 0.5 * x**2)


def _z_net_smooth(r, Z, shells):
    """Effective nuclear charge — graded impedance taper (Axiom 2, E2f).

    Each inner shell contributes a smooth enclosed-charge fraction
    σ(r) from its radial density (Eq. smooth_screening).
    Uses the UNPERTURBED density (Z_eff = Z) — no imported numbers.

    Dim check: σ is dimensionless. Z_net = Z - Σ N_a·σ(r) dimensionless. ✓

    Args:
        r:      Radial position [m].
        Z:      Nuclear charge (integer).
        shells: List of (n_shell, N_a) — winding number of inner shell
                and its electron count.

    Returns:
        Z_net:  Effective charge at radius r (dimensionless).
    """
    z = float(Z)
    for n_shell, N_a in shells:
        # Z_eff = Z for the taper profile (unperturbed, pure Axiom 2)
        sigma = _enclosed_charge_fraction(r, float(Z))
        z -= N_a * sigma
    return max(z, 0.0)


def _V_eff(r, Z, l, shells):
    """Effective radial potential V_eff(r).

    V_eff = -Z_net(r)·αℏc/r + l(l+1)ℏ²/(2m_e r²)    [J]

    Axiom 2 (Coulomb) + Axiom 1 (angular standing wave).
    Note: l(l+1), not l², from the angular eigenvalue on the
    discrete lattice (Step 1(h), Eq. radial_wave).
    """
    z = _z_net(r, Z, shells)
    V_coulomb = -z * ALPHA * HBAR * C_0 / r              # [J]
    V_centrifugal = l * (l + 1) * HBAR**2 / (2.0 * M_E * r**2)  # [J]
    return V_coulomb + V_centrifugal


# ---------------------------------------------------------------------------
# Step 2: Local wavenumber
# ---------------------------------------------------------------------------

def _k_local(r, E, Z, l, shells):
    """Local soliton wavenumber k(r) = √(2m_e(E - V_eff)) / ℏ.

    From Axiom 4 (soliton mass) + energy conservation.
    Returns real k in allowed regions, 0 in forbidden regions.
    Units: [m⁻¹].
    """
    V = _V_eff(r, Z, l, shells)
    KE = E - V  # kinetic energy [J]
    if KE <= 0.0:
        return 0.0
    return np.sqrt(2.0 * M_E * KE) / HBAR


# ---------------------------------------------------------------------------
# Step 3: Radial ODE solver (Eq. radial_wave from LaTeX)
# ---------------------------------------------------------------------------

def _radial_ode(r, y, E_J, Z_net, l):
    """Right-hand side of the radial wave equation.

    Eq. (radial_wave):
        ψ'' + [2m_e/ℏ² (E + Z_net·αℏc/r) - l(l+1)/r²] ψ = 0

    Written as a first-order system:
        y[0] = ψ,   y[1] = ψ'
        y[0]' = y[1]
        y[1]' = [l(l+1)/r² - 2m_e/ℏ²(E + Z_net·αℏc/r)] × y[0]

    All physics from Axiom 2 (Coulomb) + Axiom 4 (dispersion)
    + Axiom 1 (angular winding).

    Args:
        r:     Radial position [m].
        y:     [ψ, ψ'] state vector.
        E_J:   Energy [J] (negative for bound states).
        Z_net: Effective charge (dimensionless).
        l:     Angular winding number.

    Returns:
        [ψ', ψ''] — derivatives.
    """
    psi, dpsi = y
    # Effective potential coefficient (1/r² and 1/r terms)
    centrifugal = l * (l + 1) / r**2
    coulomb = 2.0 * M_E * Z_net * ALPHA * C_0 / (HBAR * r)
    energy_term = 2.0 * M_E * E_J / HBAR**2

    # ψ'' = [centrifugal - coulomb - energy_term] × ψ
    d2psi = (centrifugal - coulomb - energy_term) * psi
    return [dpsi, d2psi]


def _solve_radial_ode(r_start, r_end, psi0, dpsi0, E_J, Z_net, l,
                      n_points=500):
    """Integrate the radial wave equation from r_start to r_end.

    Uses scipy.integrate.solve_ivp with RK45 (adaptive step).

    Args:
        r_start, r_end: Integration limits [m].
        psi0, dpsi0:    Initial conditions [ψ(r_start), ψ'(r_start)].
        E_J:            Energy [J].
        Z_net:          Effective charge in this region.
        l:              Angular winding number.
        n_points:       Number of output points.

    Returns:
        r_arr:    Radial positions [m].
        psi_arr:  ψ values.
        dpsi_arr: ψ' values.
    """
    from scipy.integrate import solve_ivp

    r_eval = np.linspace(r_start, r_end, n_points)

    sol = solve_ivp(
        _radial_ode, [r_start, r_end], [psi0, dpsi0],
        args=(E_J, Z_net, l),
        t_eval=r_eval, method='RK45',
        rtol=1e-12, atol=1e-14
    )

    return sol.t, sol.y[0], sol.y[1]


# ---------------------------------------------------------------------------
# Step 4: ABCD transfer matrix per section (Eq. abcd_section)
# ---------------------------------------------------------------------------

def _abcd_section(r1, r2, E_J, Z_net, l):
    """Build the 2×2 ABCD transfer matrix for one radial TL section.

    Maps (ψ, ψ') at r1 to (ψ, ψ') at r2 via two IVP integrations:
        IVP 1: IC = (1, 0) → column 1 of ABCD: [A, C]
        IVP 2: IC = (0, 1) → column 2 of ABCD: [B, D]

    Eq. (abcd_section) from LaTeX.

    Dimensional analysis:
        A, D: dimensionless (ψ→ψ, ψ'→ψ')
        B: [m]  (ψ'→ψ, i.e. derivative drives field)
        C: [1/m]  (ψ→ψ', i.e. field drives derivative)
        det(ABCD) = AD - BC = 1 (Wronskian conservation)

    Args:
        r1, r2: Section boundaries [m].
        E_J:    Energy [J] (negative for bound states).
        Z_net:  Effective charge in this region (dimensionless).
        l:      Angular winding number.

    Returns:
        ABCD: 2×2 numpy array (transfer matrix).
    """
    # IVP 1: basis vector (ψ=1, ψ'=0)
    _, psi1, dpsi1 = _solve_radial_ode(r1, r2, 1.0, 0.0, E_J, Z_net, l,
                                        n_points=2)
    # IVP 2: basis vector (ψ=0, ψ'=1)
    _, psi2, dpsi2 = _solve_radial_ode(r1, r2, 0.0, 1.0, E_J, Z_net, l,
                                        n_points=2)

    # ABCD = [[ψ₁(r₂), ψ₂(r₂)],
    #          [ψ'₁(r₂), ψ'₂(r₂)]]
    ABCD = np.array([
        [psi1[-1],  psi2[-1]],
        [dpsi1[-1], dpsi2[-1]]
    ])

    return ABCD


# ---------------------------------------------------------------------------
# Step 5: Eigenvalue condition (Eq. abcd_eigenvalue from LaTeX)
# ---------------------------------------------------------------------------

def _eigenvalue_condition(E_eV, Z, n, l, shells):
    """Eigenvalue target for the ABCD cascade — graded taper model (E2f).

    The inner shell is modelled as a graded impedance taper:
    N_sec thin sections with Z_net sampled from the smooth
    enclosed-charge function σ(r) (Eq. smooth_screening).

    Procedure:
        1. Inner BC (r → 0): regular Coulomb solution (Axiom 2+4).
        2. Build ABCD cascade of N_sec thin sections, each with
           Z_net(r_i) = Z − N_inner·σ(r_i).
        3. Outer BC: ψ' + κ·ψ = 0 (decaying solution).

    Args:
        E_eV:   Trial binding energy [eV, positive].
        Z:      Nuclear charge.
        n:      Total winding number.
        l:      Angular winding number.
        shells: List of (n_shell, N_a) — winding number of inner shell
                and electron count. Z_eff = Z (unperturbed, Axiom 2).

    Returns:
        f:  Eigenvalue residual (zero at correct E).
    """
    E_J = -abs(E_eV) * e_charge  # Binding energy → negative [J]

    # Decay constant κ = √(2m_e|E|)/ℏ  [1/m]
    kappa = np.sqrt(2.0 * M_E * abs(E_J)) / HBAR

    # Inner starting point (avoid r=0 singularity)
    r_min = 0.005 * A_0

    # Inner BC: regular Coulomb solution near origin.
    # At r_min, Z_net ≈ Z (no screening near nucleus).
    if l == 0:
        x = float(Z) * r_min / A_0
        psi_init = r_min * (1.0 - x)
        dpsi_init = 1.0 - 2.0 * x
    else:
        psi_init = r_min**(l + 1)
        dpsi_init = (l + 1) * r_min**l

    # Outer boundary
    N_inner = sum(N_a for _, N_a in shells)
    z_outer = max(Z - N_inner, 1.0)
    r_max = 3.0 * n**2 * A_0 / z_outer
    r_max = max(r_max, 7.0 * A_0)

    # Build graded taper: N_sec thin sections from r_min to r_max
    N_sec = 20
    edges = np.linspace(r_min, r_max, N_sec + 1)

    state = np.array([psi_init, dpsi_init])

    for i in range(N_sec):
        r1 = edges[i]
        r2 = edges[i + 1]
        r_mid = 0.5 * (r1 + r2)

        # Z_net at midpoint from smooth screening (E2f)
        z_seg = _z_net_smooth(r_mid, Z, shells)

        # ABCD for this thin section (Op5)
        M = _abcd_section(r1, r2, E_J, z_seg, l)

        # Cascade: state at r2 = M × state at r1
        state = M @ state

    psi_out, dpsi_out = state

    # Eigenvalue condition: ψ' + κ·ψ = 0 (Op6)
    f = dpsi_out + kappa * psi_out
    scale = max(abs(dpsi_out), abs(kappa * psi_out), 1e-30)
    return f / scale


def radial_eigenvalue_abcd(Z, n, l, shells):
    """Find the energy eigenvalue using the ABCD cascade.

    Bracket (Axiom 2):
        E_hi = Z²·Ry/n²       (bare Coulomb, no screening)
        E_lo = Z_net²·Ry/n²   (full Gauss screening)

    Root-finds B_total(E) = 0 within this bracket.

    Args:
        Z:      Nuclear charge.
        n:      Total winding number (principal).
        l:      Angular winding number.
        shells: List of (n_shell, N_a) — inner-shell winding number and
                electron count. Z_eff = Z (Axiom 2, unperturbed).

    Returns:
        E_eV:   Eigenvalue [eV, positive = binding energy].
    """
    from scipy.optimize import brentq

    # Axiom 2 bracket
    N_inner = sum(N_a for _, N_a in shells)
    z_screened = max(Z - N_inner, 1.0)

    E_hi = float(Z)**2 * RY_EV / n**2   # bare Coulomb
    E_lo = z_screened**2 * RY_EV / n**2  # full screening

    # No shells or trivial: return exact
    if not shells or abs(E_hi - E_lo) < 1e-10:
        return E_hi

    # Widen slightly
    E_hi *= 1.05
    E_lo *= 0.95

    # Verify sign change
    f_hi = _eigenvalue_condition(E_hi, Z, n, l, shells)
    f_lo = _eigenvalue_condition(E_lo, Z, n, l, shells)

    if f_hi * f_lo > 0:
        # Scan for bracket
        E_scan = np.linspace(E_lo, E_hi, 200)
        f_scan = [_eigenvalue_condition(E, Z, n, l, shells) for E in E_scan]
        for i in range(len(f_scan) - 1):
            if f_scan[i] * f_scan[i+1] < 0:
                E_lo, E_hi = E_scan[i], E_scan[i+1]
                break
        else:
            return z_screened**2 * RY_EV / n**2  # fallback

    E_root = brentq(lambda E: _eigenvalue_condition(E, Z, n, l, shells),
                    E_lo, E_hi, xtol=1e-6, rtol=1e-10)
    return E_root


# ---------------------------------------------------------------------------
# Step 5b: Self-consistent iteration (E2h, lattice fluid mechanics)
# ---------------------------------------------------------------------------

def _eigenvalue_condition_general(E_eV, Z, n, l, z_net_func, N_inner):
    """Eigenvalue condition with a GENERAL z_net function.

    Same as _eigenvalue_condition but accepts any z_net callable,
    enabling self-consistent iteration with numerical screening.

    Args:
        E_eV:       Trial energy [eV, positive].
        Z:          Nuclear charge.
        n:          Principal winding number.
        l:          Angular winding number.
        z_net_func: Callable(r) → Z_net at radius r.
        N_inner:    Total inner-shell electron count (for r_max).

    Returns:
        f:  Eigenvalue residual.
    """
    E_J = -abs(E_eV) * e_charge
    kappa = np.sqrt(2.0 * M_E * abs(E_J)) / HBAR
    r_min = 0.005 * A_0

    # Inner BC: Z_net ≈ Z at r_min (unscreened near nucleus)
    if l == 0:
        x = float(Z) * r_min / A_0
        psi_init = r_min * (1.0 - x)
        dpsi_init = 1.0 - 2.0 * x
    else:
        psi_init = r_min**(l + 1)
        dpsi_init = (l + 1) * r_min**l

    z_outer = max(Z - N_inner, 1.0)
    r_max = 3.0 * n**2 * A_0 / z_outer
    r_max = max(r_max, 7.0 * A_0)

    N_sec = 20
    edges = np.linspace(r_min, r_max, N_sec + 1)
    state = np.array([psi_init, dpsi_init])

    for i in range(N_sec):
        r1, r2 = edges[i], edges[i + 1]
        r_mid = 0.5 * (r1 + r2)
        z_seg = z_net_func(r_mid)
        M = _abcd_section(r1, r2, E_J, z_seg, l)
        state = M @ state

    psi_out, dpsi_out = state
    f = dpsi_out + kappa * psi_out
    scale = max(abs(dpsi_out), abs(kappa * psi_out), 1e-30)
    return f / scale


def _extract_wavefunction(E_eV, Z, n, l, z_net_func, N_inner, n_grid=200):
    """Extract ψ(r) on a grid at a given energy.

    Runs the ABCD cascade section-by-section, solving the ODE in each
    section to produce a fine-grained ψ(r) array.

    Args:
        E_eV:       Energy [eV, positive].
        Z, n, l:    Quantum numbers.
        z_net_func: Callable(r) → Z_net.
        N_inner:    Total inner electrons.
        n_grid:     Points per section.

    Returns:
        r_arr:    Radial grid [m], shape (N_sec × n_grid,).
        psi_arr:  ψ values on that grid.
    """
    E_J = -abs(E_eV) * e_charge
    r_min = 0.005 * A_0
    z_outer = max(Z - N_inner, 1.0)
    r_max = 3.0 * n**2 * A_0 / z_outer
    r_max = max(r_max, 7.0 * A_0)

    # Inner BC
    if l == 0:
        x = float(Z) * r_min / A_0
        psi_init = r_min * (1.0 - x)
        dpsi_init = 1.0 - 2.0 * x
    else:
        psi_init = r_min**(l + 1)
        dpsi_init = (l + 1) * r_min**l

    N_sec = 20
    edges = np.linspace(r_min, r_max, N_sec + 1)

    all_r = []
    all_psi = []
    psi0, dpsi0 = psi_init, dpsi_init

    for i in range(N_sec):
        r1, r2 = edges[i], edges[i + 1]
        r_mid = 0.5 * (r1 + r2)
        z_seg = z_net_func(r_mid)

        r_arr, psi_arr, dpsi_arr = _solve_radial_ode(
            r1, r2, psi0, dpsi0, E_J, z_seg, l, n_points=n_grid
        )
        all_r.append(r_arr)
        all_psi.append(psi_arr)
        psi0, dpsi0 = psi_arr[-1], dpsi_arr[-1]

    r_full = np.concatenate(all_r)
    psi_full = np.concatenate(all_psi)
    return r_full, psi_full


def _numerical_enclosed_charge(r_grid, psi_grid):
    """Compute σ(r) = enclosed charge fraction from ψ(r).

    σ(r) = ∫₀ʳ |ψ|² 4πr'² dr' / ∫₀^∞ |ψ|² 4πr'² dr'

    This is the numerical Gauss integral (Axiom 2) on an
    arbitrary (non-hydrogenic) density.  Generalises
    Eq. (smooth_screening) for the SCF iteration.

    Args:
        r_grid:   Radial positions [m].
        psi_grid: ψ values.

    Returns:
        sigma:    Enclosed fraction at each r_grid point [0→1].
    """
    integrand = np.abs(psi_grid)**2 * 4.0 * np.pi * r_grid**2
    cumulative = np.cumsum(0.5 * (integrand[:-1] + integrand[1:])
                          * np.diff(r_grid))
    cumulative = np.insert(cumulative, 0, 0.0)  # σ(0) = 0
    total = cumulative[-1]
    if total > 0:
        return cumulative / total
    return np.zeros_like(r_grid)


def radial_eigenvalue_scf(Z, n, l, inner_shells, max_iter=10, tol=0.001):
    """Self-consistent ABCD eigenvalue solver (E2h).

    Iterates the lattice Euler equation:
      1. Compute 1s density → σ₁ₛ(r)
      2. ABCD cascade for 2s → E₂ₛ, ψ₂ₛ(r)
      3. Compute σ₂ₛ(r) from ψ₂ₛ
      4. Update 1s: each 1s sees Z − σ_other_1s(r) − σ₂ₛ(r)
      5. Recompute σ₁ₛ from updated 1s density
      6. Repeat until |ΔE| < tol

    Zero new physics: same ODE, same ABCD, same Axiom 2.

    Args:
        Z:            Nuclear charge.
        n:            Principal winding number of outer electron.
        l:            Angular winding number (must be 0 for penetration).
        inner_shells: List of (n_shell, N_a) for inner shells.
        max_iter:     Maximum iterations.
        tol:          Convergence tolerance [eV].

    Returns:
        E_eV:   Converged eigenvalue [eV].
        info:   Dict with iteration history.
    """
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    N_inner = sum(N_a for _, N_a in inner_shells)

    # --- Iteration 0: analytic screening (current result) ---
    def z_net_analytic(r):
        return _z_net_smooth(r, Z, inner_shells)

    E_prev = radial_eigenvalue_abcd(Z, n, l, inner_shells)
    history = [E_prev]

    # Build the grid for wavefunction extraction
    z_outer = max(Z - N_inner, 1.0)
    r_max = 3.0 * n**2 * A_0 / z_outer
    r_max = max(r_max, 7.0 * A_0)

    for iteration in range(1, max_iter + 1):
        # --- Extract 2s wavefunction at current eigenvalue ---
        z_net_current = z_net_analytic if iteration == 1 else z_net_updated
        r_2s, psi_2s = _extract_wavefunction(
            E_prev, Z, n, l, z_net_current, N_inner
        )
        sigma_2s = _numerical_enclosed_charge(r_2s, psi_2s)
        sigma_2s_interp = interp1d(r_2s, sigma_2s, bounds_error=False,
                                    fill_value=(0.0, 1.0))

        # --- Solve each 1s electron with updated screening ---
        # Each 1s sees: Z − σ_other_1s(r) − σ_2s(r)
        # For 2 electrons in 1s: each sees 1 other 1s + 1 2s
        def z_net_1s(r):
            # Other 1s screening (analytic, Z_eff=Z for unperturbed)
            sig_other_1s = _enclosed_charge_fraction(r, float(Z))
            # 2s screening (numerical)
            sig_2s = float(sigma_2s_interp(r))
            return max(float(Z) - sig_other_1s - sig_2s, 0.0)

        # Solve 1s ODE to get updated 1s density
        n_1s = inner_shells[0][0]  # n=1
        r_min_1s = 0.005 * A_0
        r_max_1s = 5.0 * A_0  # 1s decays fast
        E_1s_J = -float(Z)**2 * RY_EV * e_charge  # approximate 1s energy

        r_1s_grid = np.linspace(r_min_1s, r_max_1s, 500)
        # Integrate 1s ODE with position-dependent Z_net
        psi0 = r_min_1s
        dpsi0 = 1.0
        from scipy.integrate import solve_ivp
        def ode_1s(r, y):
            z = z_net_1s(r)
            V = -z * ALPHA * HBAR * C_0 / r
            k_sq = 2.0 * M_E * (E_1s_J + abs(V)) / HBAR**2
            return [y[1], -k_sq * y[0]]

        sol = solve_ivp(ode_1s, [r_min_1s, r_max_1s], [psi0, dpsi0],
                        t_eval=r_1s_grid, method='RK45',
                        rtol=1e-12, atol=1e-14)
        r_1s = sol.t
        psi_1s = sol.y[0]

        # Compute updated σ₁ₛ from numerical 1s density
        sigma_1s_num = _numerical_enclosed_charge(r_1s, psi_1s)
        sigma_1s_interp = interp1d(r_1s, sigma_1s_num, bounds_error=False,
                                    fill_value=(0.0, 1.0))

        # --- Build updated z_net for 2s cascade ---
        N_1s = inner_shells[0][1]  # number of 1s electrons

        def z_net_updated(r):
            sig = float(sigma_1s_interp(r))
            return max(float(Z) - N_1s * sig, 0.0)

        # --- Solve 2s with updated screening ---
        E_hi = float(Z)**2 * RY_EV / n**2 * 1.05
        E_lo = z_outer**2 * RY_EV / n**2 * 0.95

        f_hi = _eigenvalue_condition_general(E_hi, Z, n, l, z_net_updated,
                                             N_inner)
        f_lo = _eigenvalue_condition_general(E_lo, Z, n, l, z_net_updated,
                                             N_inner)

        if f_hi * f_lo > 0:
            E_scan = np.linspace(E_lo, E_hi, 200)
            for i in range(len(E_scan) - 1):
                fa = _eigenvalue_condition_general(E_scan[i], Z, n, l,
                                                    z_net_updated, N_inner)
                fb = _eigenvalue_condition_general(E_scan[i+1], Z, n, l,
                                                    z_net_updated, N_inner)
                if fa * fb < 0:
                    E_lo, E_hi = E_scan[i], E_scan[i+1]
                    break
            else:
                break  # can't find bracket

        E_new = brentq(
            lambda E: _eigenvalue_condition_general(E, Z, n, l,
                                                     z_net_updated, N_inner),
            E_lo, E_hi, xtol=1e-6, rtol=1e-10
        )

        history.append(E_new)

        if abs(E_new - E_prev) < tol:
            return E_new, {'iterations': iteration, 'history': history,
                           'converged': True}
        E_prev = E_new

    return E_prev, {'iterations': max_iter, 'history': history,
                    'converged': False}

# ---------------------------------------------------------------------------
# Step 6: Reflection at shell boundaries (Op3) — diagnostic
# ---------------------------------------------------------------------------

def _reflection_phase(E, Z, l, R_a, N_a, shells):
    """Phase shift from reflection at shell boundary R_a.

    Uses universal_reflection() (Op3) — same operator as nuclear/antenna.

    Returns φ_Γ = arg(Γ) [rad].  For real Γ: 0 or π.
    """
    # Build shells just inside and just outside R_a
    dr = R_a * 1e-8  # infinitesimal offset

    k_in = _k_local(R_a - dr, E, Z, l, shells)
    k_out = _k_local(R_a + dr, E, Z, l, shells)

    if k_in < 1e-20 or k_out < 1e-20:
        return 0.0  # evanescent region, no reflection phase

    # Op3: Γ = (Z2 - Z1)/(Z2 + Z1) — here Z ∝ 1/k (impedance)
    # For wave matching: Γ = (k_out - k_in)/(k_out + k_in)
    gamma = universal_reflection(k_in, k_out)

    # Phase of real Γ: 0 if Γ > 0, π if Γ < 0
    if gamma < 0:
        return np.pi
    return 0.0


# ---------------------------------------------------------------------------
# Step 5: Eigenvalue condition (Op6) — root-finding target
# ---------------------------------------------------------------------------

def _total_phase(E_eV, Z, n, l, shells):
    """Total radial phase for trial energy E.

    f(E) = Σ φ_i + Σ φ_Γ - n_r·π

    At the eigenvalue: f(E) = 0.
    """
    E_J = -abs(E_eV) * e_charge  # convert to Joules (binding = negative)
    n_r = n - l - 1               # radial node count

    if n_r < 0:
        return 1e10  # invalid quantum numbers

    # Classical turning points
    # For V_eff = -Z_net·αℏc/r + l²ℏ²/(2m_er²), find where V_eff = E
    # Inner turning point: r_min (where centrifugal barrier = E - Coulomb)
    # Outer turning point: r_max (where Coulomb = E)

    # For l = 0: r_min → 0 (no barrier).  Use small cutoff.
    # For l > 0: solve V_eff(r) = E numerically.
    r_min = A_0 * 1e-4  # ~5e-15 m, well inside any shell
    if l > 0:
        # Centrifugal barrier: find inner turning point
        # V_eff(r_min) = E → solve numerically
        for r_try in np.logspace(-15, -9, 200):
            V = _V_eff(r_try, Z, l, shells)
            if E_J > V:
                r_min = r_try
                break

    # Outer turning point: where V_eff = E (Coulomb dominates)
    # For Z_net_outer, r_max ≈ -Z_net·αℏc / E
    z_outer = _z_net(1.0, Z, shells)  # Z_net at large r
    if z_outer <= 0:
        z_outer = 1.0
    r_max = z_outer * ALPHA * HBAR * C_0 / abs(E_J)
    # Refine: find where k(r) → 0
    for r_try in np.linspace(r_max * 0.5, r_max * 2.0, 200):
        if _k_local(r_try, E_J, Z, l, shells) < 1e-5:
            r_max = r_try
            break

    if r_max <= r_min:
        return 1e10

    # Collect all boundaries within [r_min, r_max]
    boundaries = sorted([R_a for R_a, _ in shells
                         if r_min < R_a < r_max])

    # Build list of integration segments
    segment_edges = [r_min] + boundaries + [r_max]

    # Accumulate phase
    total_phi = 0.0

    # Phase integrals in each segment
    for i in range(len(segment_edges) - 1):
        r_lo = segment_edges[i]
        r_hi = segment_edges[i + 1]
        if r_hi > r_lo:
            phi_i = _phase_integral(r_lo, r_hi, E_J, Z, l, shells)
            total_phi += phi_i

    # Reflection phases at each shell boundary
    for R_a, N_a in shells:
        if r_min < R_a < r_max:
            phi_gamma = _reflection_phase(E_J, Z, l, R_a, N_a, shells)
            total_phi += phi_gamma

    # Op6: eigenvalue condition
    target = n_r * np.pi
    return total_phi - target


def radial_eigenvalue(Z, n, l, shells, E_guess_eV=None):
    """Find the energy eigenvalue for electron (n, l) in a multi-electron atom.

    Uses the radial waveguide model (E2d):
    Op3 (reflection at shell boundaries) + Op6 (phase matching).

    Bracket (Axiom 2):
        E_hi = Z²·Ry/n²         (bare Coulomb, no screening)
        E_lo = Z_net²·Ry/n²     (full Gauss screening)
    The true eigenvalue with partial penetration lies between them.

    Args:
        Z:          Nuclear charge (integer).
        n:          Total winding number (principal).
        l:          Angular winding number.
        shells:     List of (R_a_m, N_a) — inner shell radii [m]
                    and electron counts.
        E_guess_eV: Not used (bracket is Axiom-derived).

    Returns:
        E_eV:       Eigenvalue energy [eV, positive = binding energy].
    """
    from scipy.optimize import brentq

    # Axiom 2 brackets: bare vs fully screened
    N_inner = sum(N_a for _, N_a in shells)
    z_screened = max(Z - N_inner, 1.0)

    E_hi = float(Z)**2 * RY_EV / n**2       # bare Coulomb (deep binding)
    E_lo = z_screened**2 * RY_EV / n**2      # full screening (shallow)

    # For no shells: E_hi = E_lo, return exact answer
    if not shells or abs(E_hi - E_lo) < 1e-10:
        return E_hi

    # Widen bracket slightly to ensure sign change
    E_hi *= 1.01
    E_lo *= 0.99

    # Verify sign change
    f_hi = _total_phase(E_hi, Z, n, l, shells)
    f_lo = _total_phase(E_lo, Z, n, l, shells)

    if f_hi * f_lo > 0:
        # No sign change — scan for it
        n_scan = 100
        E_scan = np.linspace(E_lo, E_hi, n_scan)
        f_scan = np.array([_total_phase(E, Z, n, l, shells)
                           for E in E_scan])
        for i in range(len(f_scan) - 1):
            if f_scan[i] * f_scan[i+1] < 0:
                E_lo = E_scan[i]
                E_hi = E_scan[i+1]
                break
        else:
            # Still no bracket — return first-order estimate
            return z_screened**2 * RY_EV / n**2

    # Brent's method for robust root-finding (Op6)
    E_root = brentq(lambda E: _total_phase(E, Z, n, l, shells),
                    E_lo, E_hi, xtol=1e-6, rtol=1e-10)

    return E_root
