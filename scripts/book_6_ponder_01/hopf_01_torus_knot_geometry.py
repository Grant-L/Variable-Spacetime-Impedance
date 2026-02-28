#!/usr/bin/env python3
r"""
HOPF-01: Torus Knot PCB Trace Geometry Generator
===================================================

Generates the actual 2D trace coordinates for all 4 torus knot antennas
on a single FR-4 panel. Includes:
  1. Parametric (p,q) torus knot → planar stereographic projection
  2. Trace length calibration to match catalog (60, 90, 120, 150 mm)
  3. SMA feed-point pad placement
  4. High-res PNG figure of all 4 knot traces
  5. DXF coordinate export (importable into KiCad)

Usage:
    PYTHONPATH=src python scripts/book_6_ponder_01/hopf_01_torus_knot_geometry.py
"""

import sys
import os
import pathlib
import numpy as np

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

from ave.core.constants import ALPHA

# ══════════════════════════════════════════════════════════════
# Torus knot catalog
# ══════════════════════════════════════════════════════════════
KNOTS = [
    (2, 3,  0.060, r'$(2,3)$ Trefoil'),
    (2, 5,  0.090, r'$(2,5)$ Cinquefoil'),
    (3, 7,  0.120, r'$(3,7)$'),
    (3, 11, 0.150, r'$(3,11)$'),
]

# SMA pad dimensions (mm)
SMA_PAD_W = 1.6  # Standard SMA edge-launch pad width
SMA_PAD_H = 3.0  # Pad length


def torus_knot_3d(p, q, N=2000, R=1.0, r=0.4):
    """Generate 3D (p,q) torus knot on a torus of radii R, r."""
    t = np.linspace(0, 2 * np.pi, N)
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)
    return x, y, z


def stereographic_project(x, y, z):
    """Stereographic projection from 3D torus knot to 2D plane."""
    # Project from the north pole of a surrounding sphere
    denom = 1.0 - z / (np.max(np.abs(z)) + 1e-6) * 0.5
    x2d = x / denom
    y2d = y / denom
    return x2d, y2d


def compute_trace_length_2d(x2d, y2d):
    """Compute total arc length of 2D trace in meters."""
    dx = np.diff(x2d)
    dy = np.diff(y2d)
    return np.sum(np.sqrt(dx**2 + dy**2))


def scale_to_target_length(x2d, y2d, target_m):
    """Scale 2D coordinates so total arc length matches target (meters)."""
    current = compute_trace_length_2d(x2d, y2d)
    scale = target_m / current
    return x2d * scale, y2d * scale


def generate_knot_trace(p, q, target_length_m, N=4000):
    """Generate a 2D PCB trace for a (p,q) torus knot."""
    x3d, y3d, z3d = torus_knot_3d(p, q, N=N)
    x2d, y2d = stereographic_project(x3d, y3d, z3d)
    x2d, y2d = scale_to_target_length(x2d, y2d, target_length_m)
    # Convert to mm for PCB design
    x_mm = x2d * 1000
    y_mm = y2d * 1000
    return x_mm, y_mm


def export_dxf(knot_traces, filepath):
    """Export all knot traces as a simple DXF file (polylines)."""
    with open(filepath, 'w') as f:
        f.write("0\nSECTION\n2\nENTITIES\n")
        for name, x_mm, y_mm in knot_traces:
            f.write("0\nPOLYLINE\n8\n0\n66\n1\n")
            for xi, yi in zip(x_mm, y_mm):
                f.write(f"0\nVERTEX\n8\n0\n10\n{xi:.4f}\n20\n{yi:.4f}\n30\n0.0\n")
            f.write("0\nSEQEND\n")
        f.write("0\nENDSEC\n0\nEOF\n")
    print(f"  📐 DXF exported: {filepath}")


def main():
    print("=" * 80)
    print("  HOPF-01: Torus Knot PCB Trace Geometry Generator")
    print("=" * 80)

    knot_traces = []
    knot_data = []

    for p, q, L_target, label in KNOTS:
        x_mm, y_mm = generate_knot_trace(p, q, L_target)
        actual_length = compute_trace_length_2d(x_mm / 1000, y_mm / 1000) * 1000
        extent = max(np.ptp(x_mm), np.ptp(y_mm))

        knot_traces.append((f"knot_{p}_{q}", x_mm, y_mm))
        knot_data.append({
            'p': p, 'q': q, 'label': label,
            'L_target': L_target * 1000,
            'L_actual': actual_length,
            'extent': extent,
            'x_mm': x_mm, 'y_mm': y_mm,
        })

        print(f"  {label:<20} L_target={L_target*1000:.0f}mm  "
              f"L_actual={actual_length:.1f}mm  extent={extent:.1f}mm")

    # ── Export DXF ──
    dxf_path = project_root / "assets" / "sim_outputs" / "hopf_01_knot_traces.dxf"
    export_dxf(knot_traces, dxf_path)

    # ── Generate figure ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(20, 20))
        fig.patch.set_facecolor('#0a0a0a')

        colors = ['#00ffcc', '#ff6b6b', '#ffd93d', '#6bcaff']

        # Panel layout: 2×2 grid of knot traces + 1 combined bottom panel
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25,
                     height_ratios=[1, 1, 0.6])

        for i, kd in enumerate(knot_data):
            row, col = divmod(i, 2)
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor('#111111')

            # Plot the trace
            ax.plot(kd['x_mm'], kd['y_mm'], color=colors[i], lw=1.8, solid_capstyle='round')

            # Mark feed point (start/end)
            ax.plot(kd['x_mm'][0], kd['y_mm'][0], 's', color='white',
                   markersize=8, markeredgecolor=colors[i], markeredgewidth=2,
                   zorder=10, label='SMA Feed')

            # SMA pad rectangle
            pad_x = kd['x_mm'][0] - SMA_PAD_W / 2
            pad_y = kd['y_mm'][0] - SMA_PAD_H / 2
            rect = plt.Rectangle((pad_x, pad_y), SMA_PAD_W, SMA_PAD_H,
                                linewidth=1.5, edgecolor='#ffcc00',
                                facecolor='#ffcc00', alpha=0.3)
            ax.add_patch(rect)

            ax.set_title(f'{kd["label"]}  —  L = {kd["L_target"]:.0f} mm',
                        color=colors[i], fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('X (mm)', color='#cccccc', fontsize=10)
            ax.set_ylabel('Y (mm)', color='#cccccc', fontsize=10)
            ax.set_aspect('equal')
            ax.tick_params(colors='#888899')
            ax.grid(True, alpha=0.1, color='white')
            ax.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333',
                     labelcolor='white', loc='upper right')
            for spine in ax.spines.values():
                spine.set_color('#333')

        # ── Bottom panel: All knots overlaid to show relative scale ──
        ax_all = fig.add_subplot(gs[2, :])
        ax_all.set_facecolor('#111111')

        spacing = 0
        for i, kd in enumerate(knot_data):
            offset = spacing
            ax_all.plot(kd['x_mm'] + offset, kd['y_mm'],
                       color=colors[i], lw=1.5, label=kd['label'])
            ax_all.plot(kd['x_mm'][0] + offset, kd['y_mm'][0], 's',
                       color='white', markersize=6, markeredgecolor=colors[i],
                       markeredgewidth=1.5, zorder=10)
            spacing += kd['extent'] + 5

        ax_all.set_title('All 4 Knots — Single Panel Layout (Relative Scale)',
                        color='white', fontsize=14, fontweight='bold', pad=10)
        ax_all.set_xlabel('X (mm)', color='#cccccc', fontsize=10)
        ax_all.set_ylabel('Y (mm)', color='#cccccc', fontsize=10)
        ax_all.set_aspect('equal')
        ax_all.tick_params(colors='#888899')
        ax_all.grid(True, alpha=0.1, color='white')
        ax_all.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#333',
                     labelcolor='white', loc='upper right')
        for spine in ax_all.spines.values():
            spine.set_color('#333')

        # Save
        out_dir = project_root / "assets" / "sim_outputs"
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / "hopf_01_knot_traces.png"
        fig.savefig(out_path, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  📊 Plot saved: {out_path}")
    except ImportError:
        print("\n  ⚠️  matplotlib not available — skipping plots")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
