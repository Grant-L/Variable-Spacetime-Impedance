"""
AVE Eulerian-Lagrangian Coupling
Translates discrete topological Lagrangian knots (Particles/Rotors)
into macroscopic continuous Eulerian strain and velocity fields on the grid.
"""

import numpy as np
from scipy.spatial import cKDTree

from ave.matter.solitons import TopologicalSoliton
from ave.solvers.grid_3d import EulerianGrid3D


def apply_topological_kinematics_to_grid(grid: EulerianGrid3D, soliton: TopologicalSoliton):
    """
    Translates the momentum and sheer geometry of a topological Golden Torus
    (or any macroscopic mass) directly into localized velocity gradients on the grid.
    """
    X, Y, Z = grid.get_mesh()
    grid_points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

    # 1. Identify the topological parametric backbone curve
    knot_core_resolution = 500
    try:
        curve_local = soliton.get_parametric_core(knot_core_resolution)
    except NotImplementedError:
        # Fallback to a single point mass for base solitons
        curve_local = np.array([[0.0, 0.0, 0.0]])

    curve_global = soliton._apply_rotations(curve_local)

    # 2. Find closest distance from each grid node to the continuous flux tube
    tree = cKDTree(curve_global)
    distances, _ = tree.query(grid_points, k=1)

    # 3. Create a spatial weight factor (Gaussian FWHM matching l_node)
    # The knot drags the local metric fluid strongly at the core, decaying outward.
    spatial_coupling = np.exp(-(distances**2) / (2.0 * soliton.sigma**2))
    spatial_coupling = spatial_coupling.reshape((grid.Nx, grid.Ny, grid.Nz))

    # 4. Inject Lagrangian kinematics into the Eulerian grid
    # A topological knot in motion IS a fluid vortex advecting the metric
    grid.vx += spatial_coupling * soliton.vel[0]
    grid.vy += spatial_coupling * soliton.vel[1]
    grid.vz += spatial_coupling * soliton.vel[2]

    return grid


def apply_macroscopic_rotor_to_grid(grid: EulerianGrid3D, center, radius, omega_vec):
    """
    Simulates a macroscopic mechanical rotor (e.g. Sagnac-RLVE falsification rig).
    The rotor forces angular velocity onto the grid. If shear is high enough,
    a localized superfluid envelope will form.
    """
    X, Y, Z = grid.get_mesh()

    # Vector math for v = omega x r
    rx = X - center[0]
    ry = Y - center[1]
    rz = Z - center[2]

    r_mag = np.sqrt(rx**2 + ry**2 + rz**2)

    # Identify which grid points are strictly inside or on the rotor boundary
    rotor_mask = r_mag <= radius

    # Cross product: v = w x r
    v_rot_x = omega_vec[1] * rz - omega_vec[2] * ry
    v_rot_y = omega_vec[2] * rx - omega_vec[0] * rz
    v_rot_z = omega_vec[0] * ry - omega_vec[1] * rx

    # Force the grid fluid velocity strictly inside the rotor to match the rotor surface
    # This acts as the rigorous boundary condition triggering Bingham shear at the edge
    grid.vx[rotor_mask] = v_rot_x[rotor_mask]
    grid.vy[rotor_mask] = v_rot_y[rotor_mask]
    grid.vz[rotor_mask] = v_rot_z[rotor_mask]

    return grid
