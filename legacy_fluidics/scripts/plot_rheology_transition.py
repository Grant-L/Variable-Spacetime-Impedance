import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add src to path securely
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ave.mechanics import moduli

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/sim_outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_rheological_phase_transition():
    print("--- GENERATING BINGHAM-PLASTIC SUPERFLUID TRANSITION VISUALIZATION ---")

    # Get the exact mathematically derived topological yield stress
    tau_yield = moduli.calculate_bingham_yield_stress()

    # Generate a range of metric shear stresses from 0 to 3x the yield limit
    shear_stresses = np.linspace(0, tau_yield * 3.0, 500)

    # The effective viscosity is derived from the Bingham model.
    # Below tau_yield, it's effectively infinite (rigid solid).
    # Above tau_yield, it drops rapidly to 0 (superfluid).
    # We will plot the normalized Newtonian Viscosity vs the Bingham Viscosity.

    baseline_viscosity = moduli.calculate_kinematic_viscosity()

    effective_viscosities = np.zeros_like(shear_stresses)

    for i, tau in enumerate(shear_stresses):
        if tau < tau_yield:
            # Rigid Matrix (Dark Matter Proxy, high structural resistance)
            effective_viscosities[i] = baseline_viscosity * 100.0  # Cap for plotting
        else:
            # Superfluid Phase (Viscosity completely annihilated)
            effective_viscosities[i] = 0.0

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("#050508")
    ax.set_facecolor("#050508")

    # Plot standard Newtonian fluid baseline (doesn't change with shear)
    ax.plot(
        shear_stresses,
        np.full_like(shear_stresses, baseline_viscosity),
        ls="--",
        lw=2,
        color="#555555",
        label=r"Continuum Approximation (Newtonian)",
    )

    # Plot the AVE Topological Rheology
    ax.plot(shear_stresses, effective_viscosities, color="#ff3366", lw=3, label=r"Topological Viscosity ($\eta_{eff}$)")

    # Highlight regions
    ax.axvspan(0, tau_yield, color="#1a1a3a", alpha=0.5, label="Rigid Matrix (Dark Matter)")
    ax.axvspan(
        tau_yield, tau_yield * 3.0, color="#0a2a1a", alpha=0.5, label="Frictionless Superfluid (Planetary Orbit)"
    )

    ax.axvline(tau_yield, color="#00ffcc", ls=":", lw=2, label=r"Avalanche Threshold ($\tau_{yield}$)")

    # Formatting
    ax.set_title(
        r"Bingham-Plastic Dielectric Rupture ($\mathcal{M}_A$ Fluidics)", color="white", weight="bold", fontsize=14
    )
    ax.set_xlabel(r"Applied Metric Shear Stress ($\tau$)", color="white", weight="bold")
    ax.set_ylabel(r"Effective Kinematic Viscosity ($\eta_{eff}$)", color="white", weight="bold")

    ax.spines["bottom"].set_color("#333333")
    ax.spines["top"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["right"].set_color("#333333")
    ax.tick_params(colors="white")

    # Clean x-axis to be relative to tau_yield
    ax.set_xticks([0, tau_yield, tau_yield * 2, tau_yield * 3])
    ax.set_xticklabels(["0", r"$\tau_{yield}$", r"$2\tau_{yield}$", r"$3\tau_{yield}$"])

    # Clean up y-axis to be relative
    ax.set_yticks([0, baseline_viscosity, baseline_viscosity * 100])
    ax.set_yticklabels(["0 (Superfluid)", r"$\nu_{vac}$", r"$\to \infty$ (Rigid Solid)"])

    ax.legend(facecolor="#111111", edgecolor="white", labelcolor="white", loc="upper right")

    out_path = os.path.join(OUTPUT_DIR, "bingham_superfluid_transition.png")
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved to {out_path}")


if __name__ == "__main__":
    plot_rheological_phase_transition()
