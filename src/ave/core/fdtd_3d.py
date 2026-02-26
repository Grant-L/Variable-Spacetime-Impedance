"""
3D Finite-Difference Time-Domain (FDTD) Maxwell Solver Engine
=============================================================

Non-Linear AVE Solver implementing Axiom 4 dielectric saturation.

This module provides a rigorous, time-evolved 3D Maxwell equation solver
utilizing the standard Yee-cell grid architecture. The electric field update
uses a spatially and temporally varying permittivity:

    ε_eff(V) = ε₀ · √(1 − (V/V_yield)²)

This non-linearity causes the solver to:
  - Recover linear Maxwell exactly when E << E_crit (most of space)
  - Slow local phase velocity near strong fields (dielectric drag → mass)
  - Diverge the update coefficient near saturation (energy trapping)
  - Support both linear_only=True mode (for benchmarking) and full non-linear

Includes 1st-order Mur Absorbing Boundary Conditions (ABCs).
"""

import numpy as np
from ave.core.constants import C_0, MU_0, EPSILON_0, V_SNAP


class FDTD3DEngine:
    """
    3D FDTD Maxwell solver with optional Axiom 4 non-linear vacuum.

    Args:
        nx, ny, nz: Grid dimensions.
        dx: Cell size [m].
        linear_only: If True, uses constant ε₀ (standard Maxwell). Default False.
        v_yield: Dielectric yield voltage per cell [V]. Default is V_SNAP = m_e c²/e.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        dx: float = 0.01,
        linear_only: bool = False,
        v_yield: float = V_SNAP,
    ):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.linear_only = linear_only
        self.v_yield = v_yield

        # Physical Constants
        self.c = float(C_0)
        self.mu_0 = float(MU_0)
        self.epsilon_0 = float(EPSILON_0)

        # CFL Condition for 3D stability: dt <= dx / (c * √3)
        self.dt = self.dx / (self.c * np.sqrt(3.0))

        # Core Field Matrices (E and H vectors)
        self.Ex = np.zeros((nx, ny, nz))
        self.Ey = np.zeros((nx, ny, nz))
        self.Ez = np.zeros((nx, ny, nz))

        self.Hx = np.zeros((nx, ny, nz))
        self.Hy = np.zeros((nx, ny, nz))
        self.Hz = np.zeros((nx, ny, nz))

        # Magnetic update coefficient (constant — μ₀ is not modified by Axiom 4)
        self.ch = self.dt / (self.mu_0 * self.dx)

        # Linear electric update coefficient (used when linear_only=True)
        self.ce_linear = self.dt / (self.epsilon_0 * self.dx)

        # Mur 1st-Order ABC coefficient
        abc_coef = (self.c * self.dt - self.dx) / (self.c * self.dt + self.dx)
        self.abc_coef = abc_coef

        # ABC boundary memory vectors
        self.ex_y0 = np.zeros((nx, nz)); self.ex_yn = np.zeros((nx, nz))
        self.ex_z0 = np.zeros((nx, ny)); self.ex_zn = np.zeros((nx, ny))

        self.ey_x0 = np.zeros((ny, nz)); self.ey_xn = np.zeros((ny, nz))
        self.ey_z0 = np.zeros((nx, ny)); self.ey_zn = np.zeros((nx, ny))

        self.ez_x0 = np.zeros((ny, nz)); self.ez_xn = np.zeros((ny, nz))
        self.ez_y0 = np.zeros((nx, nz)); self.ez_yn = np.zeros((nx, nz))

        # Diagnostics
        self.timestep = 0
        self.max_strain_ratio = 0.0  # Track peak |E·dx / V_yield| for stability

    def _compute_local_epsilon(self, E_component: np.ndarray) -> np.ndarray:
        """
        Compute the local non-linear permittivity per cell per component.

        ε_eff = ε₀ · √(1 − (E·dx / V_yield)²)

        The local voltage across a cell is V_local = E · dx.
        The saturation ratio is V_local / V_yield.

        Returns:
            Per-cell permittivity array (same shape as E_component).
        """
        V_local = np.abs(E_component) * self.dx
        ratio_sq = (V_local / self.v_yield) ** 2

        # Track maximum strain for diagnostics
        max_r = np.max(ratio_sq) if ratio_sq.size > 0 else 0.0
        if max_r > self.max_strain_ratio:
            self.max_strain_ratio = max_r

        # Clip to prevent numerical instability near exact saturation
        # (ratio_sq >= 1.0 would mean dielectric rupture)
        ratio_sq = np.clip(ratio_sq, 0.0, 1.0 - 1e-12)

        return self.epsilon_0 * np.sqrt(1.0 - ratio_sq)

    def _compute_ce(self, E_component: np.ndarray) -> np.ndarray | float:
        """
        Compute the electric field update coefficient.

        In linear mode: ce = dt / (ε₀ · dx) — uniform scalar.
        In non-linear mode: ce = dt / (ε_eff(E) · dx) — per-cell array.
        """
        if self.linear_only:
            return self.ce_linear

        eps_eff = self._compute_local_epsilon(E_component)
        return self.dt / (eps_eff * self.dx)

    def update_magnetic_field(self):
        """Update H fields from the curl of E (Faraday's Law). Linear — μ₀ is constant."""
        # Hx
        self.Hx[:, :-1, :-1] -= self.ch * (
            (self.Ez[:, 1:, :-1] - self.Ez[:, :-1, :-1]) -
            (self.Ey[:, :-1, 1:] - self.Ey[:, :-1, :-1])
        )
        # Hy
        self.Hy[:-1, :, :-1] -= self.ch * (
            (self.Ex[:-1, :, 1:] - self.Ex[:-1, :, :-1]) -
            (self.Ez[1:, :, :-1] - self.Ez[:-1, :, :-1])
        )
        # Hz
        self.Hz[:-1, :-1, :] -= self.ch * (
            (self.Ey[1:, :-1, :] - self.Ey[:-1, :-1, :]) -
            (self.Ex[:-1, 1:, :] - self.Ex[:-1, :-1, :])
        )

    def update_electric_field(self):
        """
        Update E fields from the curl of H (Ampere's Law).

        In non-linear mode, the update coefficient ce is computed PER CELL
        from the local field amplitude, implementing Axiom 4:

            E^{n+1} = E^n + (dt / ε_eff(E^n)) · (∇×H) / dx
        """
        # --- Ex update ---
        curl_h_x = (
            (self.Hz[:, 1:, 1:] - self.Hz[:, :-1, 1:]) -
            (self.Hy[:, 1:, 1:] - self.Hy[:, 1:, :-1])
        )
        ce_x = self._compute_ce(self.Ex[:, 1:, 1:])
        self.Ex[:, 1:, 1:] += ce_x * curl_h_x

        # --- Ey update ---
        curl_h_y = (
            (self.Hx[1:, :, 1:] - self.Hx[1:, :, :-1]) -
            (self.Hz[1:, :, 1:] - self.Hz[:-1, :, 1:])
        )
        ce_y = self._compute_ce(self.Ey[1:, :, 1:])
        self.Ey[1:, :, 1:] += ce_y * curl_h_y

        # --- Ez update ---
        curl_h_z = (
            (self.Hy[1:, 1:, :] - self.Hy[:-1, 1:, :]) -
            (self.Hx[1:, 1:, :] - self.Hx[1:, :-1, :])
        )
        ce_z = self._compute_ce(self.Ez[1:, 1:, :])
        self.Ez[1:, 1:, :] += ce_z * curl_h_z

    def apply_mur_abc(self):
        """Apply 1st-Order Mur ABCs to all six faces."""
        c1 = self.abc_coef

        # X-Boundaries
        self.Ey[0, :, :] = self.ey_x0 + c1 * (self.Ey[1, :, :] - self.Ey[0, :, :])
        self.ey_x0[:, :] = self.Ey[1, :, :]
        self.Ez[0, :, :] = self.ez_x0 + c1 * (self.Ez[1, :, :] - self.Ez[0, :, :])
        self.ez_x0[:, :] = self.Ez[1, :, :]
        self.Ey[-1, :, :] = self.ey_xn + c1 * (self.Ey[-2, :, :] - self.Ey[-1, :, :])
        self.ey_xn[:, :] = self.Ey[-2, :, :]
        self.Ez[-1, :, :] = self.ez_xn + c1 * (self.Ez[-2, :, :] - self.Ez[-1, :, :])
        self.ez_xn[:, :] = self.Ez[-2, :, :]

        # Y-Boundaries
        self.Ex[:, 0, :] = self.ex_y0 + c1 * (self.Ex[:, 1, :] - self.Ex[:, 0, :])
        self.ex_y0[:, :] = self.Ex[:, 1, :]
        self.Ez[:, 0, :] = self.ez_y0 + c1 * (self.Ez[:, 1, :] - self.Ez[:, 0, :])
        self.ez_y0[:, :] = self.Ez[:, 1, :]
        self.Ex[:, -1, :] = self.ex_yn + c1 * (self.Ex[:, -2, :] - self.Ex[:, -1, :])
        self.ex_yn[:, :] = self.Ex[:, -2, :]
        self.Ez[:, -1, :] = self.ez_yn + c1 * (self.Ez[:, -2, :] - self.Ez[:, -1, :])
        self.ez_yn[:, :] = self.Ez[:, -2, :]

        # Z-Boundaries
        self.Ex[:, :, 0] = self.ex_z0 + c1 * (self.Ex[:, :, 1] - self.Ex[:, :, 0])
        self.ex_z0[:, :] = self.Ex[:, :, 1]
        self.Ey[:, :, 0] = self.ey_z0 + c1 * (self.Ey[:, :, 1] - self.Ey[:, :, 0])
        self.ey_z0[:, :] = self.Ey[:, :, 1]
        self.Ex[:, :, -1] = self.ex_zn + c1 * (self.Ex[:, :, -2] - self.Ex[:, :, -1])
        self.ex_zn[:, :] = self.Ex[:, :, -2]
        self.Ey[:, :, -1] = self.ey_zn + c1 * (self.Ey[:, :, -2] - self.Ey[:, :, -1])
        self.ey_zn[:, :] = self.Ey[:, :, -2]

    def inject_soft_source(self, field: str, x: int, y: int, z: int, amplitude: float):
        """Inject a soft source (additive) into a field component at (x, y, z)."""
        if field == 'Ex':
            self.Ex[x, y, z] += amplitude
        elif field == 'Ey':
            self.Ey[x, y, z] += amplitude
        elif field == 'Ez':
            self.Ez[x, y, z] += amplitude

    def total_field_energy(self) -> float:
        """
        Compute total electromagnetic energy in the grid.

        U = Σ (½ε_eff |E|² + ½μ₀ |H|²) · dx³
        """
        E_sq = self.Ex**2 + self.Ey**2 + self.Ez**2
        H_sq = self.Hx**2 + self.Hy**2 + self.Hz**2

        if self.linear_only:
            u_e = 0.5 * self.epsilon_0 * E_sq
        else:
            # Non-linear permittivity per cell
            E_mag = np.sqrt(E_sq)
            V_local = E_mag * self.dx
            ratio_sq = np.clip((V_local / self.v_yield)**2, 0.0, 1.0 - 1e-12)
            eps_local = self.epsilon_0 * np.sqrt(1.0 - ratio_sq)
            u_e = 0.5 * eps_local * E_sq

        u_m = 0.5 * self.mu_0 * H_sq
        return float(np.sum((u_e + u_m) * self.dx**3))

    def step(self):
        """Execute one complete dt timestep of the Maxwell Yee-cell algorithm."""
        self.update_magnetic_field()
        self.update_electric_field()
        self.apply_mur_abc()
        self.timestep += 1
