import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

root_dir = Path(__file__).parent.parent.parent.parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ave.matter.atoms import CarbonAtom
from ave.core import constants as k


def render_carbon():
    atom = CarbonAtom()
    a0 = atom.a_0
    bounds = a0 * 3.5
    N = 800
    X, Y = np.meshgrid(np.linspace(-bounds, bounds, N), np.linspace(-bounds, bounds, N))

    R_eff = atom.Z * 3.0 * k.L_NODE
    V_tot = R_eff / np.clip(np.sqrt(X**2 + Y**2), R_eff, None)

    electrons = [atom.e1, atom.e2] + atom.outer_electrons
    for e in electrons:
        V_tot += e.r_core / np.clip(np.sqrt((X - e.pos[0]) ** 2 + (Y - e.pos[1]) ** 2), e.r_core, None)

    V_tot = np.clip(V_tot, 1e-4, 0.99999)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    c = ax.pcolormesh(X / a0, Y / a0, V_tot, cmap="magma", norm=LogNorm(vmin=1e-3, vmax=1.0), shading="auto")

    ax.add_patch(plt.Circle((0, 0), atom.a_1s / a0, color="white", fill=False, linestyle="-", alpha=0.6))
    ax.add_patch(plt.Circle((0, 0), atom.a_2s / a0, color="cyan", fill=False, linestyle="--", alpha=0.5))

    ax.scatter([0], [0], color="white", s=100, edgecolor="yellow", zorder=5)
    ax.scatter(
        [atom.e1.pos[0] / a0, atom.e2.pos[0] / a0],
        [atom.e1.pos[1] / a0, atom.e2.pos[1] / a0],
        color="cyan",
        s=40,
        zorder=5,
    )

    outer_x = [e.pos[0] / a0 for e in atom.outer_electrons]
    outer_y = [e.pos[1] / a0 for e in atom.outer_electrons]
    ax.scatter(outer_x, outer_y, color="magenta", s=60, zorder=5)

    ax.set_aspect("equal")
    ax.set_title(r"Topological Carbon ($^{12}$C) - Tetrahedral/Quadratic Saturation", fontsize=15, pad=15)
    fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04).set_label(r"Dielectric Strain", rotation=270, labelpad=20)

    out_dir = root_dir / "periodic_table_of_knots" / "chapters" / "006_carbon" / "simulations" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "carbon_strain.png", bbox_inches="tight")


if __name__ == "__main__":
    render_carbon()
