"""
SPICE Transient Integrator
==========================

Universal explicit Euler integrator for lumped L-C-R networks.

This module extracts the time-stepping kinematics from the protein folding
engine and makes them available at any scale.  The equations are:

    v(t + Δt) = v(t) + [ -∇f(θ) - R·v(t) ] / L  ·  Δt     (Newton's 2nd)
    θ(t + Δt) = θ(t) + v(t + Δt) · Δt                       (Euler forward)

where:
    L = inertial mass (inductance analogue)
    R = dissipative friction (resistance analogue)
    ∇f = gradient of the potential (eigenvalue target, S₁₁, etc.)

Used at:
    - Protein scale: torsion-angle ring-down into native fold
    - Nuclear scale: bond geometry relaxation
    - Antenna scale: current distribution equilibration

All constants are set by the caller — this module contains ZERO
domain-specific physics.
"""

import numpy as np


def explicit_euler_step(theta, velocity, grad_f, L, R, dt, mask=None):
    """
    Single explicit Euler step for SPICE transient integration.

    Args:
        theta:    Current state vector (angles, positions, etc.)
        velocity: Current velocity vector
        grad_f:   Gradient of the target function at theta
        L:        Inertial mass per DOF (scalar or array)
        R:        Damping resistance per DOF (scalar or array)
        dt:       Timestep (physical, not a learning rate)
        mask:     Optional binary mask (1 = active, 0 = frozen DOF)

    Returns:
        new_theta, new_velocity: Updated state and velocity vectors
    """
    g = grad_f
    if mask is not None:
        g = g * mask

    # Physical SPICE Euler: a = (-∇V - R·v) / L
    acceleration = (-g - R * velocity) / L
    new_velocity = velocity + acceleration * dt
    new_theta = theta + new_velocity * dt

    return new_theta, new_velocity


def integrate_transient(theta_init, grad_fn, L, R, dt, n_steps,
                        mask=None, grad_clip=None):
    """
    Full transient integration loop (NumPy, non-JIT).

    Integrates the state vector theta through n_steps of explicit Euler,
    using grad_fn(theta) to compute the potential gradient at each step.

    Args:
        theta_init: Initial state vector
        grad_fn:    Callable theta → gradient array (same shape as theta)
        L:          Inertial mass per DOF
        R:          Damping resistance per DOF
        dt:         Physical timestep
        n_steps:    Number of integration steps
        mask:       Optional binary mask for active DOFs
        grad_clip:  Optional maximum gradient norm (e.g. 2π for angular)

    Returns:
        theta_final:    Final state vector
        velocity_final: Final velocity vector
        trajectory:     List of (theta, f_val) at each step
    """
    theta = np.array(theta_init, dtype=float)
    velocity = np.zeros_like(theta)
    trajectory = []

    for step in range(n_steps):
        g = grad_fn(theta)

        # Replace NaN gradients with zero
        g = np.where(np.isnan(g), 0.0, g)

        # Optional gradient clipping
        if grad_clip is not None:
            g_norm = np.sqrt(np.sum(g**2) + 1e-12)
            if g_norm > grad_clip:
                g = g * grad_clip / g_norm

        theta, velocity = explicit_euler_step(
            theta, velocity, g, L, R, dt, mask=mask)

        trajectory.append((theta.copy(), float(np.sum(g**2))))

    return theta, velocity, trajectory


# ── JAX backend (optional, for JIT-compiled loops) ──────────────

try:
    import jax
    import jax.numpy as jnp
    from jax import lax

    def explicit_euler_step_jax(theta, velocity, grad_f, L, R, dt, mask=None):
        """JAX-traceable single Euler step."""
        g = grad_f
        if mask is not None:
            g = g * mask
        acceleration = (-g - R * velocity) / L
        new_velocity = velocity + acceleration * dt
        new_theta = theta + new_velocity * dt
        return new_theta, new_velocity

    def integrate_transient_jax(theta_init, velocity_init, grad_fn,
                                 L, R, dt, n_steps, mask=None,
                                 grad_clip=2.0 * jnp.pi):
        """
        JIT-compiled transient integration via lax.fori_loop.

        Args:
            theta_init:    Initial state (JAX array)
            velocity_init: Initial velocity (JAX array, usually zeros)
            grad_fn:       JAX-traceable callable theta → gradient
            L, R, dt:      Physical parameters
            n_steps:       Number of Euler steps
            mask:          Optional DOF mask
            grad_clip:     Maximum gradient norm (default: 2π for angular)

        Returns:
            theta_final, velocity_final
        """
        def step_fn(i, carry):
            theta, vel = carry
            g = grad_fn(theta)
            g = jnp.where(jnp.isnan(g), 0.0, g)

            # Gradient clipping
            g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-12)
            g = jnp.where(g_norm > grad_clip, g * grad_clip / g_norm, g)

            if mask is not None:
                g = g * mask

            acceleration = (-g - R * vel) / L
            new_vel = vel + acceleration * dt
            new_theta = theta + new_vel * dt
            return (new_theta, new_vel)

        return lax.fori_loop(0, n_steps, step_fn, (theta_init, velocity_init))

    HAS_JAX = True

except ImportError:
    HAS_JAX = False
