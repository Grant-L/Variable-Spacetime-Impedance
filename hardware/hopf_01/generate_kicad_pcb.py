#!/usr/bin/env python3
r"""
HOPF-01: KiCad PCB Generator — Wire-Stitched Knot Fixture
===========================================================

Generates a .kicad_pcb where the PCB is a mechanical fixture for
enameled inductor wire, NOT copper-traced antennas.

For each torus knot:
  - Unplated drill holes every ~3mm along the knot path
  - Full knot curve on F.SilkS as a winding guide
  - Over/under crossing markers on silkscreen
  - A copper SMA pad at the feed point for soldering the wire start
  - An anchor pad at the wire endpoint

The user threads enameled magnet wire through the holes to create
a true 3D torus knot with real over/under crossings.

Usage:
    PYTHONPATH=src python hardware/hopf_01/generate_kicad_pcb.py
"""

import sys
import pathlib
import numpy as np
import uuid
from datetime import datetime

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

# ══════════════════════════════════════════════════════════════
# Board Parameters
# ══════════════════════════════════════════════════════════════
BOARD_W = 120.0      # mm (sized for longer wire paths)
BOARD_H = 120.0      # mm
HOLE_SPACING = 3.0   # mm between stitching holes
HOLE_DRILL = 1.0     # mm drill diameter (for ~0.5mm enameled wire)
CORNER_R = 2.0       # mm corner radius
MOUNT_INSET = 4.0    # mm from board edge

# Knot placement grid (2×2, centered)
GRID_CX = BOARD_W / 2
GRID_CY = BOARD_H / 2
QUAD_SPACING = 28.0  # mm between knot centers (room for longer traces)

KNOTS = [
    (2, 3,  0.120, '(2,3) Trefoil',    GRID_CX - QUAD_SPACING, GRID_CY - QUAD_SPACING),
    (2, 5,  0.160, '(2,5) Cinquefoil', GRID_CX + QUAD_SPACING, GRID_CY - QUAD_SPACING),
    (3, 7,  0.200, '(3,7)',             GRID_CX - QUAD_SPACING, GRID_CY + QUAD_SPACING),
    (3, 11, 0.250, '(3,11)',            GRID_CX + QUAD_SPACING, GRID_CY + QUAD_SPACING),
]


def new_uuid():
    return str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════
# 3D Torus Knot Parameterization
# ══════════════════════════════════════════════════════════════

def torus_knot_3d(p, q, N=4000, R=1.0, r=0.4):
    """Generate 3D coordinates of a (p,q) torus knot.

    Returns x, y, z arrays in normalized coordinates.
    """
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)
    return x, y, z, t


def scale_knot_to_length(x, y, L_target_m, N=4000):
    """Scale 2D knot coordinates so total arc length = L_target_m.

    Returns x_mm, y_mm in millimeters.
    """
    dx = np.diff(x)
    dy = np.diff(y)
    seg_lengths = np.sqrt(dx**2 + dy**2)
    total = np.sum(seg_lengths)
    scale = L_target_m / total
    x_mm = x * scale * 1000
    y_mm = y * scale * 1000
    return x_mm, y_mm


def find_crossings(x, y, z, N):
    """Find self-intersection points in the 2D projection.

    Returns list of (crossing_x, crossing_y, over_strand_idx, under_strand_idx).
    """
    crossings = []
    # Downsample for O(n²) search — check every 20th segment pair
    step = max(1, N // 200)
    checked = set()

    for i in range(0, N - 2, step):
        for j in range(i + step * 3, N - 1, step):
            # Check if segment i→i+step crosses segment j→j+step
            x1, y1 = x[i], y[i]
            x2, y2 = x[min(i + step, N - 1)], y[min(i + step, N - 1)]
            x3, y3 = x[j], y[j]
            x4, y4 = x[min(j + step, N - 1)], y[min(j + step, N - 1)]

            # Line segment intersection test
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                continue

            t_param = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u_param = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

            if 0 < t_param < 1 and 0 < u_param < 1:
                cx = x1 + t_param * (x2 - x1)
                cy = y1 + t_param * (y2 - y1)

                # Avoid duplicate crossings (within 0.5mm)
                key = (round(cx, 0), round(cy, 0))
                if key in checked:
                    continue
                checked.add(key)

                # Determine over/under from z-coordinates
                zi = z[i] + t_param * (z[min(i + step, N - 1)] - z[i])
                zj = z[j] + u_param * (z[min(j + step, N - 1)] - z[j])

                crossings.append((cx, cy, zi > zj))  # True = strand i is OVER

    return crossings


# ══════════════════════════════════════════════════════════════
# KiCad S-Expression Generators
# ══════════════════════════════════════════════════════════════

def kicad_header():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""(kicad_pcb
  (version 20240108)
  (generator "hopf_01_generator")
  (generator_version "2.0")
  (general
    (thickness 1.6)
    (legacy_teardrops no)
  )
  (paper "A4")
  (title_block
    (title "HOPF-01 Wire-Stitched Knot Antenna Fixture")
    (date "{now}")
    (rev "2.0")
    (company "Applied Vacuum Engineering")
    (comment 1 "Enameled wire stitched through PCB holes")
    (comment 2 "Silkscreen shows winding guide + crossings")
    (comment 3 "Measurement: NanoVNA-H4 S11 sweep")
  )
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive")
    (33 "F.Adhes" user "F.Adhesive")
    (34 "B.Paste" user)
    (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user "B.Mask")
    (39 "F.Mask" user "F.Mask")
    (40 "Dwgs.User" user "User.Drawings")
    (41 "Cmts.User" user "User.Comments")
    (42 "Eco1.User" user "User.Eco1")
    (43 "Eco2.User" user "User.Eco2")
    (44 "Edge.Cuts" user)
    (45 "Margin" user)
    (46 "B.CrtYd" user "B.Courtyard")
    (47 "F.CrtYd" user "F.Courtyard")
    (48 "B.Fab" user "B.Fabrication")
    (49 "F.Fab" user "F.Fabrication")
  )
  (setup
    (pad_to_mask_clearance 0.05)
    (allow_soldermask_bridges_in_footprints no)
    (pcbplotparams
      (layerselection 0x00010fc_ffffffff)
      (plot_on_all_layers_selection 0x0000000_00000000)
    )
  )
  (net 0 "")
  (net 1 "GND")
  (net 2 "ANT1_SIG")
  (net 3 "ANT2_SIG")
  (net 4 "ANT3_SIG")
  (net 5 "ANT4_SIG")
"""


def board_outline():
    lines = []
    x0, y0 = 0, 0
    x1, y1 = BOARD_W, BOARD_H
    r = CORNER_R
    lines.append(f'  (gr_line (start {x0+r} {y0}) (end {x1-r} {y0}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    lines.append(f'  (gr_line (start {x1} {y0+r}) (end {x1} {y1-r}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    lines.append(f'  (gr_line (start {x1-r} {y1}) (end {x0+r} {y1}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    lines.append(f'  (gr_line (start {x0} {y1-r}) (end {x0} {y0+r}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    lines.append(f'  (gr_arc (start {x0+r} {y0+r}) (mid {x0+r*0.293} {y0+r*0.293}) (end {x0} {y0+r}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    lines.append(f'  (gr_arc (start {x1-r} {y0+r}) (mid {x1-r*0.293} {y0+r*0.293}) (end {x1-r} {y0}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    lines.append(f'  (gr_arc (start {x1-r} {y1-r}) (mid {x1-r*0.293} {y1-r*0.293}) (end {x1} {y1-r}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    lines.append(f'  (gr_arc (start {x0+r} {y1-r}) (mid {x0+r*0.293} {y1-r*0.293}) (end {x0+r} {y1}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    return "\n".join(lines)


def sma_pad(x, y, net_id, ref):
    """SMA edge-launch connector footprint with ground tabs.

    Center signal pad (thru-hole, 1.5mm pad, 1.2mm drill) + 4 ground
    tab pads (SMD, 1.0×2.0mm) for shield grounding. Matches typical
    SMA edge-mount connectors (e.g. Amphenol 132255).
    """
    uid = new_uuid()
    return f"""  (footprint "HOPF01:SMA_EdgeLaunch" (layer "F.Cu")
    (uuid "{uid}")
    (at {x:.4f} {y:.4f})
    (property "Reference" "{ref}" (at 0 -4) (layer "F.SilkS") (uuid "{new_uuid()}")
      (effects (font (size 1 1) (thickness 0.15)))
    )
    (property "Value" "SMA_50R" (at 0 4) (layer "F.Fab") (uuid "{new_uuid()}")
      (effects (font (size 0.8 0.8) (thickness 0.12)))
    )
    (pad "1" thru_hole circle (at 0 0) (size 1.5 1.5) (drill 1.2) (layers "*.Cu" "*.Mask")
      (net {net_id} "ANT{net_id-1}_SIG") (uuid "{new_uuid()}")
    )
    (pad "2" smd rect (at -1.6 -1.8) (size 1.0 2.0) (layers "F.Cu" "F.Paste" "F.Mask")
      (net 1 "GND") (uuid "{new_uuid()}")
    )
    (pad "3" smd rect (at 1.6 -1.8) (size 1.0 2.0) (layers "F.Cu" "F.Paste" "F.Mask")
      (net 1 "GND") (uuid "{new_uuid()}")
    )
    (pad "4" smd rect (at -1.6 1.8) (size 1.0 2.0) (layers "F.Cu" "F.Paste" "F.Mask")
      (net 1 "GND") (uuid "{new_uuid()}")
    )
    (pad "5" smd rect (at 1.6 1.8) (size 1.0 2.0) (layers "F.Cu" "F.Paste" "F.Mask")
      (net 1 "GND") (uuid "{new_uuid()}")
    )
  )
"""


def anchor_pad(x, y, ref):
    """Wire endpoint anchor — just a plated hole for soldering the wire end."""
    uid = new_uuid()
    return f"""  (footprint "HOPF01:AnchorPad" (layer "F.Cu")
    (uuid "{uid}")
    (at {x:.4f} {y:.4f})
    (property "Reference" "{ref}" (at 0 -2) (layer "F.SilkS") (uuid "{new_uuid()}")
      (effects (font (size 0.8 0.8) (thickness 0.12)))
    )
    (pad "1" thru_hole circle (at 0 0) (size 2.0 2.0) (drill 1.0) (layers "*.Cu" "*.Mask")
      (net 0 "") (uuid "{new_uuid()}")
    )
  )
"""


def mounting_hole(x, y, ref):
    uid = new_uuid()
    return f"""  (footprint "MountingHole:MountingHole_3.2mm_M3_Pad" (layer "F.Cu")
    (uuid "{uid}")
    (at {x:.4f} {y:.4f})
    (property "Reference" "{ref}" (at 0 -3) (layer "F.SilkS") (uuid "{new_uuid()}")
      (effects (font (size 1 1) (thickness 0.15)))
    )
    (pad "1" thru_hole circle (at 0 0) (size 6.0 6.0) (drill 3.2) (layers "*.Cu" "*.Mask")
      (net 1 "GND") (uuid "{new_uuid()}")
    )
  )
"""


def generate_knot_features(p, q, L_target, label, cx, cy, net_id):
    """Generate all PCB features for one wire-stitched torus knot.

    Returns: list of KiCad S-expression strings, feed_x, feed_y
    """
    N = 4000
    x3d, y3d, z3d, t = torus_knot_3d(p, q, N)

    # Scale 2D projection to target trace length
    x_mm, y_mm = scale_knot_to_length(x3d, y3d, L_target, N)

    # Center on placement point
    x_mm = x_mm - np.mean(x_mm) + cx
    y_mm = y_mm - np.mean(y_mm) + cy

    lines = []

    # ── 1. Silkscreen winding guide (full knot path) ──
    # Draw the knot as thin silkscreen lines
    step = max(1, N // 500)  # ~500 silkscreen line segments
    for i in range(0, N - step, step):
        j = min(i + step, N - 1)
        lines.append(
            f'  (gr_line (start {x_mm[i]:.4f} {y_mm[i]:.4f}) '
            f'(end {x_mm[j]:.4f} {y_mm[j]:.4f}) '
            f'(layer "F.SilkS") (width 0.25) (uuid "{new_uuid()}"))'
        )

    # ── 2. Stitching holes (unplated, every HOLE_SPACING mm) ──
    # Compute cumulative arc length
    dx = np.diff(x_mm)
    dy = np.diff(y_mm)
    seg_len = np.sqrt(dx**2 + dy**2)
    cum_len = np.concatenate([[0], np.cumsum(seg_len)])
    total_len = cum_len[-1]

    # Place holes at regular arc-length intervals
    hole_count = 0
    hole_positions = []
    target_dist = 0
    while target_dist <= total_len:
        # Find the index where cumulative length >= target_dist
        idx = np.searchsorted(cum_len, target_dist)
        if idx >= N:
            idx = N - 1
        hx, hy = x_mm[idx], y_mm[idx]
        hole_positions.append((hx, hy))

        # Unplated through-hole as a footprint
        uid = new_uuid()
        lines.append(f"""  (footprint "HOPF01:StitchHole" (layer "F.Cu")
    (uuid "{uid}")
    (at {hx:.4f} {hy:.4f})
    (pad "" np_thru_hole circle (at 0 0) (size {HOLE_DRILL} {HOLE_DRILL}) (drill {HOLE_DRILL}) (layers "*.Cu" "*.Mask")
      (uuid "{new_uuid()}")
    )
  )""")
        hole_count += 1
        target_dist += HOLE_SPACING

    # ── 3. Crossing markers ──
    # Scale z for crossing detection (use same scale as x,y)
    z_mm = z3d * (np.max(x_mm) - np.min(x_mm)) / (np.max(x3d) - np.min(x3d))

    crossings = find_crossings(x_mm, y_mm, z_mm, N)
    for cx_pt, cy_pt, is_over in crossings:
        marker = "OVER" if is_over else "UNDER"
        lines.append(
            f'  (gr_text "{marker}" (at {cx_pt:.2f} {cy_pt:.2f}) '
            f'(layer "F.SilkS") (uuid "{new_uuid()}")'
            f'\n    (effects (font (size 0.6 0.6) (thickness 0.1)))'
            f'\n  )'
        )
        # Small circle at crossing point
        lines.append(
            f'  (gr_circle (center {cx_pt:.2f} {cy_pt:.2f}) '
            f'(end {cx_pt + 0.5:.2f} {cy_pt:.2f}) '
            f'(layer "F.SilkS") (width 0.15) (uuid "{new_uuid()}"))'
        )

    # ── 4. Knot label (offset to avoid overlapping stitching holes) ──
    # Place label outside the knot footprint, toward the board center
    if cy < BOARD_H / 2:
        label_y = cy + 20  # below knot (toward center)
    else:
        label_y = cy - 20  # above knot (toward center)
    lines.append(
        f'  (gr_text "{label}" (at {cx} {label_y}) '
        f'(layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 1.5 1.5) (thickness 0.25)))'
        f'\n  )'
    )
    lines.append(
        f'  (gr_text "L={L_target*1000:.0f}mm  {hole_count} holes" (at {cx} {label_y + 2.5}) '
        f'(layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 0.8 0.8) (thickness 0.12)))'
        f'\n  )'
    )

    # ── 5. Start / End markers (offset away from pads) ──
    start_label_y = y_mm[0] - 3.5 if y_mm[0] < cy else y_mm[0] + 3.5
    lines.append(
        f'  (gr_text "START" (at {x_mm[0]:.2f} {start_label_y:.2f}) '
        f'(layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 0.7 0.7) (thickness 0.12)))'
        f'\n  )'
    )
    end_label_y = y_mm[-1] + 3.5 if y_mm[-1] < cy else y_mm[-1] - 3.5
    lines.append(
        f'  (gr_text "END" (at {x_mm[-1]:.2f} {end_label_y:.2f}) '
        f'(layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 0.7 0.7) (thickness 0.12)))'
        f'\n  )'
    )

    print(f"  {label:<20} feed=({x_mm[0]:.1f},{y_mm[0]:.1f}), "
          f"{hole_count} holes, {len(crossings)} crossings, L={total_len:.1f}mm")

    return "\n".join(lines), x_mm[0], y_mm[0], x_mm[-1], y_mm[-1]


def sma_ground_patch(x, y):
    """Small ground copper zone on B.Cu under an SMA connector.

    Only covers the SMA footprint area (~10×10mm), NOT the full board.
    Wire can route freely on both sides outside these patches.
    """
    uid = new_uuid()
    hw = 5.0  # half-width of the patch
    return f"""  (zone (net 1) (net_name "GND") (layer "B.Cu") (uuid "{uid}")
    (hatch edge 0.5)
    (connect_pads (clearance 0.3))
    (min_thickness 0.25)
    (filled_areas_thickness no)
    (fill yes (thermal_gap 0.5) (thermal_bridge_width 0.5))
    (polygon
      (pts
        (xy {x - hw} {y - hw})
        (xy {x + hw} {y - hw})
        (xy {x + hw} {y + hw})
        (xy {x - hw} {y + hw})
      )
    )
  )
"""


def main():
    print("=" * 60)
    print("  HOPF-01: Wire-Stitched Knot Fixture Generator")
    print("=" * 60)

    parts = [kicad_header()]
    parts.append(board_outline())

    # Mounting holes
    for x, y, ref in [
        (MOUNT_INSET, MOUNT_INSET, "MH1"),
        (BOARD_W - MOUNT_INSET, MOUNT_INSET, "MH2"),
        (BOARD_W - MOUNT_INSET, BOARD_H - MOUNT_INSET, "MH3"),
        (MOUNT_INSET, BOARD_H - MOUNT_INSET, "MH4"),
    ]:
        parts.append(mounting_hole(x, y, ref))

    # Torus knots
    for i, (p, q, L, label, cx, cy) in enumerate(KNOTS):
        features, fx, fy, ex, ey = generate_knot_features(p, q, L, label, cx, cy, i + 2)
        parts.append(features)

        # SMA connector at nearest board edge (edge-launch placement)
        # Offset toward the nearest top/bottom edge
        sma_edge_inset = 8.0  # mm from board edge
        if cy < GRID_CY:
            sma_y = sma_edge_inset       # near top edge
        else:
            sma_y = BOARD_H - sma_edge_inset  # near bottom edge
        sma_x = fx  # same x as feed point

        parts.append(sma_pad(sma_x, sma_y, i + 2, f"J{i+1}"))

        # Silkscreen guide line: SMA → first stitching hole
        parts.append(
            f'  (gr_line (start {sma_x:.4f} {sma_y:.4f}) '
            f'(end {fx:.4f} {fy:.4f}) '
            f'(layer "F.SilkS") (width 0.3) (uuid "{new_uuid()}"))'
        )

        # Anchor pad at wire end (offset similarly toward nearest edge)
        if cy < GRID_CY:
            anc_y = sma_edge_inset + 3.0
        else:
            anc_y = BOARD_H - sma_edge_inset - 3.0
        parts.append(anchor_pad(ex, anc_y, f"END{i+1}"))

        # Ground patch on B.Cu under each SMA connector
        parts.append(sma_ground_patch(sma_x, sma_y))

        # Ground stitching vias near SMA (F.Cu ground tabs → B.Cu patch)
        for vx_off, vy_off in [(-3.0, 0), (3.0, 0)]:
            uid = new_uuid()
            parts.append(
                f'  (via (at {sma_x + vx_off:.4f} {sma_y + vy_off:.4f}) '
                f'(size 0.8) (drill 0.4) (layers "F.Cu" "B.Cu") '
                f'(net 1) (uuid "{uid}"))'
            )

    # Title block (top of board, above SMA area)
    parts.append(
        f'  (gr_text "HOPF-01" (at {GRID_CX} 3) (layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 2.5 2.5) (thickness 0.4)))'
        f'\n  )'
    )
    parts.append(
        f'  (gr_text "Wire-Stitched Torus Knot Fixture" (at {GRID_CX} {BOARD_H / 2}) (layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 0.9 0.9) (thickness 0.13)))'
        f'\n  )'
    )
    parts.append(
        f'  (gr_text "Mount on 10mm standoffs - wire routes both sides" (at {GRID_CX} {BOARD_H - 3}) (layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 0.7 0.7) (thickness 0.1)))'
        f'\n  )'
    )

    parts.append(")")

    out_path = pathlib.Path(__file__).parent / "hopf_01.kicad_pcb"
    with open(out_path, "w") as f:
        f.write("\n".join(parts))

    print(f"\n  Saved: {out_path}")
    print(f"  Board: {BOARD_W}x{BOARD_H}mm, 2-layer FR-4, 1.6mm")
    print(f"  Wire: use 0.5mm (24 AWG) enameled magnet wire")
    print("=" * 60)


if __name__ == "__main__":
    main()
