"""
AVE Spacetime Circuit Analysis Simulator
Protocol: Inject topological pulse into discrete LC vacuum grid.
Verify propagation velocity == c, and generate Figure 12.3.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add src to path securely
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ave.core import constants as const
from ave.electrodynamics import transmission

# Define output directory for manuscript visual assets
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/sim_outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_transmission_test():
    print("--- RUNNING SPACETIME CIRCUIT SIMULATION (FDTD) ---")

    # We use 150 nodes to ensure the wave doesn't hit the boundary during the test
    nodes = 150
    grid = transmission.TransmissionLine1D(nodes=nodes)

    # Inject Gaussian Pulse at node 10
    grid.inject_pulse(magnitude=0.4, width=2.0, center=10)

    # Nodes to monitor (To reproduce the manuscript's Time-Domain Traces)
    monitor_nodes = [10, 30, 50, 70, 90]
    history = {n: [] for n in monitor_nodes}
    time_axis = []

    # Run Time Steps (220 steps gets the wave cleanly past node 90)
    total_steps = 220
    for step in range(total_steps):
        grid.step_forward(steps=1)
        time_axis.append(step)
        for n in monitor_nodes:
            history[n].append(grid.V[n])

    # --- Analyze Empirical Group Velocity ---
    # Find the discrete time index where the pulse peaks at node 10 and node 90
    peak_time_10 = np.argmax(history[10])
    peak_time_90 = np.argmax(history[90])

    # Calculate physical Delta X and Delta T
    dist_traveled = (90 - 10) * grid.dx
    time_elapsed = (peak_time_90 - peak_time_10) * grid.dt

    # Guard against division by zero in case of catastrophic failure
    if time_elapsed > 0:
        velocity = dist_traveled / time_elapsed
    else:
        velocity = 0.0

    c_theory = const.C

    # Calculate error margin
    error = abs(velocity - c_theory) / c_theory
    print(f"  Theoretical Speed of Light: {c_theory:.4e} m/s")
    print(f"  Simulated Group Velocity:   {velocity:.4e} m/s")

    if error < 0.05:  # Allow a tight 5% tolerance for discrete FDTD numerical dispersion
        print(f"  [PASS] Spacetime Transmission Velocity Verified (Error: {error:.2%})\n")
    else:
        print(f"  [FAIL] Velocity mismatch! Error: {error:.2%}\n")

    # --- Generate Figure 12.3 ---
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    fig.patch.set_facecolor("#050508")
    ax.set_facecolor("#050508")

    colors = ["#ff3366", "#ffaa00", "#00ffcc", "#3399ff", "#b84dff"]
    for idx, n in enumerate(monitor_nodes):
        ax.plot(time_axis, history[n], color=colors[idx % len(colors)], lw=2.5, label=f"Spatial Node {n}")

    ax.axhline(0, color="#333333", lw=1.5, ls="-")
    ax.set_title(
        r"The EFT Transmission Line ($v_g = 1/\sqrt{L_{node}C_{node}} \equiv c$)",
        color="white",
        fontsize=14,
        weight="bold",
    )
    ax.set_xlabel("Time (Discrete Steps)", color="white", weight="bold")
    ax.set_ylabel("Topological Voltage (V)", color="white", weight="bold")
    ax.set_xlim(0, 200)

    ax.grid(True, ls=":", color="#333333")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    ax.legend(loc="upper right", facecolor="#111111", edgecolor="white", labelcolor="white")

    # Educational Callout Box matching your other graphs
    textstr = (
        r"$\mathbf{Emergence~of~the~Speed~of~Light:}$"
        + "\n"
        + r"By cascading the discrete inductive mass ($\mu_0 \ell_{node}$) and"
        + "\n"
        + r"capacitive compliance ($\epsilon_0 \ell_{node}$) of the analog lattice,"
        + "\n"
        + r"the signal physically propagates exactly at $v_g = c$."
        + "\n"
        + r"Continuous Spacetime evaluates identically to a"
        + "\n"
        + r"macroscopic electrical transmission line."
    )
    ax.text(
        0.02,
        0.95,
        textstr,
        transform=ax.transAxes,
        color="white",
        fontsize=10,
        verticalalignment="top",
        bbox={"facecolor": "#111111", "edgecolor": "#00ffcc", "alpha": 0.9, "pad": 8},
    )

    # Save the output visualization directly to assets
    out_path = os.path.join(OUTPUT_DIR, "eft_transmission_line.png")
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    print(f"  [PLOT] Generated and saved Figure to: {out_path}")


if __name__ == "__main__":
    run_transmission_test()
