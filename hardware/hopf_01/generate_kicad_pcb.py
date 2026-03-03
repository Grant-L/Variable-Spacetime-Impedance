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
BOARD_W = 160.0      # mm (expanded for 6 antennas: 3×2 grid)
BOARD_H = 120.0      # mm
HOLE_SPACING = 3.0   # mm between stitching holes
HOLE_DRILL = 1.0     # mm drill diameter (for ~0.5mm enameled wire)
MIN_HOLE_SPACING = 1.5  # mm center-to-center min (JLCPCB: 0.5mm edge-to-edge + 1.0mm drill)
CORNER_R = 2.0       # mm corner radius
MOUNT_INSET = 4.0    # mm from board edge

# Knot placement grid (3×2, centered)
GRID_CX = BOARD_W / 2
GRID_CY = BOARD_H / 2
COL_SPACING = 44.0   # mm between column centers
ROW_SPACING = 28.0   # mm between row centers

# Columns: left, center, right
col_L = GRID_CX - COL_SPACING
col_C = GRID_CX
col_R = GRID_CX + COL_SPACING
# Rows: top, bottom
row_T = GRID_CY - ROW_SPACING
row_B = GRID_CY + ROW_SPACING

KNOTS = [
    # Row 1 (top): torus knots ordered by pq/(p+q)
    (2, 3,  0.120, '(2,3) Trefoil',    col_L, row_T),
    (2, 5,  0.160, '(2,5) Cinquefoil', col_C, row_T),
    (3, 5,  0.170, '(3,5)',             col_R, row_T),
    # Row 2 (bottom): higher-order knots + control
    (3, 7,  0.200, '(3,7)',             col_L, row_B),
    (3, 11, 0.250, '(3,11)',            col_R, row_B),
]
# Control antenna: zero-topology meander with same length as trefoil
# This is NOT a torus knot — it's added separately
CONTROL_ANTENNA = {
    'L_wire': 0.120,  # same length as trefoil
    'label': 'CONTROL (meander)',
    'cx': col_C, 'cy': row_B,
}


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
  (net 6 "ANT5_SIG")
  (net 7 "ANT6_SIG")
"""


def board_outline():
    """Board outline with rounded corners on Edge.Cuts layer."""
    lines = []
    x0, y0 = 0, 0
    x1, y1 = BOARD_W, BOARD_H
    r = CORNER_R
    c45 = r * 0.7071  # r × cos(45°) for midpoint on arc

    # Straight edges (between corner tangent points)
    lines.append(f'  (gr_line (start {x0+r} {y0}) (end {x1-r} {y0}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    lines.append(f'  (gr_line (start {x1} {y0+r}) (end {x1} {y1-r}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    lines.append(f'  (gr_line (start {x1-r} {y1}) (end {x0+r} {y1}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    lines.append(f'  (gr_line (start {x0} {y1-r}) (end {x0} {y0+r}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')

    # Corner arcs — (start, mid, end) are all points ON the arc
    # Top-left: center (r, r), from (0, r) to (r, 0)
    lines.append(f'  (gr_arc (start {x0} {y0+r}) (mid {r-c45:.4f} {r-c45:.4f}) (end {x0+r} {y0}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    # Top-right: center (W-r, r), from (W-r, 0) to (W, r)
    lines.append(f'  (gr_arc (start {x1-r} {y0}) (mid {x1-r+c45:.4f} {r-c45:.4f}) (end {x1} {y0+r}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    # Bottom-right: center (W-r, H-r), from (W, H-r) to (W-r, H)
    lines.append(f'  (gr_arc (start {x1} {y1-r}) (mid {x1-r+c45:.4f} {y1-r+c45:.4f}) (end {x1-r} {y1}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')
    # Bottom-left: center (r, H-r), from (r, H) to (0, H-r)
    lines.append(f'  (gr_arc (start {x0+r} {y1}) (mid {r-c45:.4f} {y1-r+c45:.4f}) (end {x0} {y1-r}) (layer "Edge.Cuts") (width 0.1) (uuid "{new_uuid()}"))')

    return "\n".join(lines)


def sma_pad(x, y, net_id, ref):
    """SMA vertical-mount connector footprint with ground tabs.

    Center signal pad (thru-hole, 1.5mm pad, 1.2mm drill) + 4 ground
    tab pads (thru-hole, on *.Cu layers to connect F.Cu → B.Cu ground).
    Matches typical SMA panel-mount connectors (e.g. TE CONSMA003.062).
    """
    uid = new_uuid()
    return f"""  (footprint "HOPF01:SMA_PanelMount" (layer "F.Cu")
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
    (pad "2" thru_hole rect (at -1.6 -1.8) (size 1.2 2.2) (drill 0.8) (layers "*.Cu" "*.Mask")
      (net 1 "GND") (uuid "{new_uuid()}")
    )
    (pad "3" thru_hole rect (at 1.6 -1.8) (size 1.2 2.2) (drill 0.8) (layers "*.Cu" "*.Mask")
      (net 1 "GND") (uuid "{new_uuid()}")
    )
    (pad "4" thru_hole rect (at -1.6 1.8) (size 1.2 2.2) (drill 0.8) (layers "*.Cu" "*.Mask")
      (net 1 "GND") (uuid "{new_uuid()}")
    )
    (pad "5" thru_hole rect (at 1.6 1.8) (size 1.2 2.2) (drill 0.8) (layers "*.Cu" "*.Mask")
      (net 1 "GND") (uuid "{new_uuid()}")
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

        # Enforce JLCPCB minimum hole-to-hole spacing (1.5mm c-c for 1.0mm drill)
        # Check against ALL previously placed holes (not just the last one)
        # to catch overlaps at knot crossings where different path segments meet
        if hole_positions:
            too_close = False
            for phx, phy in hole_positions:
                if (hx - phx)**2 + (hy - phy)**2 < MIN_HOLE_SPACING**2:
                    too_close = True
                    break
            if too_close:
                target_dist += HOLE_SPACING
                continue

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

    # Compute knot bounding box for placing labels well outside it
    knot_x_min, knot_x_max = np.min(x_mm), np.max(x_mm)
    knot_y_min, knot_y_max = np.min(y_mm), np.max(y_mm)
    knot_margin = 4.0  # mm clearance beyond bounding box

    for cx_pt, cy_pt, is_over in crossings:
        marker = "OVER" if is_over else "UNDER"
        text_half_w = 2.0  # approx half-width of "UNDER" at 0.6mm font

        # Direction from knot center to crossing point
        dx = cx_pt - cx
        dy = cy_pt - cy
        dist = max(np.sqrt(dx**2 + dy**2), 0.1)
        ux, uy = dx / dist, dy / dist  # unit vector outward

        # Push label OUTSIDE the knot bounding box + margin
        # Find the distance from crossing to the bounding box edge in this direction
        if abs(ux) > 0.01:
            t_x = ((knot_x_max if ux > 0 else knot_x_min) - cx_pt) / ux
        else:
            t_x = 1e6
        if abs(uy) > 0.01:
            t_y = ((knot_y_max if uy > 0 else knot_y_min) - cy_pt) / uy
        else:
            t_y = 1e6
        # Distance to push past the bbox edge
        push = max(min(t_x, t_y), 0) + knot_margin + text_half_w

        lx = cx_pt + push * ux
        ly = cy_pt + push * uy

        # Clamp to board area (leave 2mm margin from edge)
        lx = np.clip(lx, 3.0, BOARD_W - 3.0)
        ly = np.clip(ly, 3.0, BOARD_H - 3.0)

        # Nudge if label center is too close to any hole (within 1.5mm)
        for hx, hy in hole_positions:
            if np.sqrt((lx - hx)**2 + (ly - hy)**2) < 1.5:
                lx += 2.0 * ux
                ly += 2.0 * uy
                lx = np.clip(lx, 3.0, BOARD_W - 3.0)
                ly = np.clip(ly, 3.0, BOARD_H - 3.0)
                break

        # Place label text
        lines.append(
            f'  (gr_text "{marker}" (at {lx:.2f} {ly:.2f}) '
            f'(layer "F.SilkS") (uuid "{new_uuid()}")'
            f'\n    (effects (font (size 0.6 0.6) (thickness 0.1)))'
            f'\n  )'
        )

        # Arrow from text EDGE (closest to crossing) to crossing circle
        # Start arrow 2.5mm from label center toward crossing
        arrow_start_x = lx - 2.5 * ux
        arrow_start_y = ly - 2.5 * uy
        lines.append(
            f'  (gr_line (start {arrow_start_x:.2f} {arrow_start_y:.2f}) '
            f'(end {cx_pt:.2f} {cy_pt:.2f}) '
            f'(layer "F.SilkS") (width 0.1) (uuid "{new_uuid()}"))'
        )
        # Small circle at crossing point
        lines.append(
            f'  (gr_circle (center {cx_pt:.2f} {cy_pt:.2f}) '
            f'(end {cx_pt + 0.5:.2f} {cy_pt:.2f}) '
            f'(layer "F.SilkS") (width 0.15) (uuid "{new_uuid()}"))'
        )

    # ── 4. Knot label with arrow toward knot center ──
    if cy < BOARD_H / 2:
        label_y = cy + 20
    else:
        label_y = cy - 20
    lines.append(
        f'  (gr_text "{label}" (at {cx} {label_y}) '
        f'(layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 1.5 1.5) (thickness 0.25)))'
        f'\n  )'
    )
    sub_y = label_y + (2.5 if cy < BOARD_H / 2 else -2.5)
    lines.append(
        f'  (gr_text "L={L_target*1000:.0f}mm  {hole_count} holes" (at {cx} {sub_y}) '
        f'(layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 0.8 0.8) (thickness 0.12)))'
        f'\n  )'
    )
    # Arrow from BELOW subtitle toward knot (clear of both text lines)
    if cy < BOARD_H / 2:
        arrow_start_y = sub_y + 1.5   # below subtitle
        arrow_end_y = sub_y + 4.5     # points toward knot (higher y = toward center)
    else:
        arrow_start_y = sub_y - 1.5   # above subtitle
        arrow_end_y = sub_y - 4.5     # points toward knot
    lines.append(
        f'  (gr_line (start {cx:.2f} {arrow_start_y:.2f}) '
        f'(end {cx:.2f} {arrow_end_y:.2f}) '
        f'(layer "F.SilkS") (width 0.2) (uuid "{new_uuid()}"))'
    )

    print(f"  {label:<20} feed=({x_mm[0]:.1f},{y_mm[0]:.1f}), "
          f"{hole_count} holes, {len(crossings)} crossings, L={total_len:.1f}mm")

    return "\n".join(lines), x_mm[0], y_mm[0], x_mm[-1], y_mm[-1]


def sma_ground_patch(x, y):
    """Ground copper zone on F.Cu under an SMA connector.

    Sized at 12×12mm to cover the full SMA footprint plus margin.
    Connects directly to F.Cu perimeter ground trace. SMA thru-hole
    pads bridge to B.Cu inherently.
    """
    uid = new_uuid()
    hw = 6.0  # half-width of the patch (12mm total)
    return f"""  (zone (net 1) (net_name "GND") (layer "F.Cu") (uuid "{uid}")
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


        # Ground patch on F.Cu under each SMA connector
        parts.append(sma_ground_patch(sma_x, sma_y))

    # ── Control antenna: zero-topology meander ──
    ctrl = CONTROL_ANTENNA
    ctrl_cx, ctrl_cy = ctrl['cx'], ctrl['cy']
    ctrl_L_m = ctrl['L_wire']  # meters
    ctrl_L_mm = ctrl_L_m * 1000  # mm
    ctrl_net_id = len(KNOTS) + 2
    ctrl_idx = len(KNOTS)

    # Meander geometry: vertical zigzag legs connected by horizontal runs.
    # Total wire = n_legs × leg_height + (n_legs - 1) × leg_spacing
    # Solve for leg_spacing given n_legs, leg_height, and target L:
    n_legs = 8
    leg_height = 12.0  # mm per vertical leg
    # ctrl_L_mm = n_legs * leg_height + (n_legs - 1) * leg_spacing
    remaining = ctrl_L_mm - n_legs * leg_height
    leg_spacing = remaining / max(n_legs - 1, 1)
    total_width = leg_spacing * (n_legs - 1)

    meander_x = []
    meander_y = []
    x_start = ctrl_cx - total_width / 2

    for leg in range(n_legs):
        x_pos = x_start + leg * leg_spacing
        if leg % 2 == 0:
            meander_x.extend([x_pos, x_pos])
            meander_y.extend([ctrl_cy - leg_height / 2, ctrl_cy + leg_height / 2])
        else:
            meander_x.extend([x_pos, x_pos])
            meander_y.extend([ctrl_cy + leg_height / 2, ctrl_cy - leg_height / 2])

    meander_x = np.array(meander_x)
    meander_y = np.array(meander_y)

    # Verify wire length
    actual_len = sum(np.sqrt(np.diff(meander_x)**2 + np.diff(meander_y)**2))

    # Place stitching holes along meander path (with dedup + min spacing)
    ctrl_lines = []
    hole_count = 0
    placed_holes = set()  # track (x, y) rounded to 0.01mm
    ctrl_hole_positions = []  # for spacing checks
    for seg in range(len(meander_x) - 1):
        x1, y1 = meander_x[seg], meander_y[seg]
        x2, y2 = meander_x[seg + 1], meander_y[seg + 1]
        seg_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        n_holes = max(1, int(seg_len / HOLE_SPACING))
        for h in range(n_holes + 1):
            frac = h / max(n_holes, 1)
            hx = x1 + frac * (x2 - x1)
            hy = y1 + frac * (y2 - y1)
            key = (round(hx, 2), round(hy, 2))
            if key in placed_holes:
                continue
            # JLCPCB min spacing check
            if ctrl_hole_positions:
                too_close = False
                for phx, phy in ctrl_hole_positions:
                    if np.sqrt((hx - phx)**2 + (hy - phy)**2) < MIN_HOLE_SPACING:
                        too_close = True
                        break
                if too_close:
                    continue
            placed_holes.add(key)
            ctrl_hole_positions.append((hx, hy))
            uid = new_uuid()
            ctrl_lines.append(f"""  (footprint "HOPF01:StitchHole" (layer "F.Cu")
    (uuid "{uid}")
    (at {hx:.4f} {hy:.4f})
    (pad "" np_thru_hole circle (at 0 0) (size {HOLE_DRILL} {HOLE_DRILL}) (drill {HOLE_DRILL}) (layers "*.Cu" "*.Mask")
      (uuid "{new_uuid()}")
    )
  )""")
            hole_count += 1

        # Silkscreen guide line
        ctrl_lines.append(
            f'  (gr_line (start {x1:.4f} {y1:.4f}) '
            f'(end {x2:.4f} {y2:.4f}) '
            f'(layer "F.SilkS") (width 0.25) (uuid "{new_uuid()}"))'
        )

    # Meander START label (near SMA feed) with arrow
    start_x, start_y = meander_x[0], meander_y[0]
    start_lbl_y = start_y + 4.0  # above the first hole
    ctrl_lines.append(
        f'  (gr_text "FEED" (at {start_x:.2f} {start_lbl_y:.2f}) '
        f'(layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 0.7 0.7) (thickness 0.12)))'
        f'\n  )'
    )
    ctrl_lines.append(
        f'  (gr_line (start {start_x:.2f} {start_lbl_y - 1.0:.2f}) '
        f'(end {start_x:.2f} {start_y + 1.0:.2f}) '
        f'(layer "F.SilkS") (width 0.15) (uuid "{new_uuid()}"))'
    )

    # Meander END label (open circuit end) with arrow
    end_x, end_y = meander_x[-1], meander_y[-1]
    end_lbl_y = end_y + 4.0
    ctrl_lines.append(
        f'  (gr_text "OPEN" (at {end_x:.2f} {end_lbl_y:.2f}) '
        f'(layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 0.7 0.7) (thickness 0.12)))'
        f'\n  )'
    )
    ctrl_lines.append(
        f'  (gr_line (start {end_x:.2f} {end_lbl_y - 1.0:.2f}) '
        f'(end {end_x:.2f} {end_y + 1.0:.2f}) '
        f'(layer "F.SilkS") (width 0.15) (uuid "{new_uuid()}"))'
    )

    parts.append("\n".join(ctrl_lines))

    # Control antenna labels
    label_y = ctrl_cy - 12 if ctrl_cy > BOARD_H / 2 else ctrl_cy + 12
    parts.append(
        f'  (gr_text "{ctrl["label"]}" (at {ctrl_cx} {label_y}) '
        f'(layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 1.2 1.2) (thickness 0.2)))'
        f'\n  )'
    )
    sub_y = label_y + (2.5 if ctrl_cy < BOARD_H / 2 else -2.5)
    parts.append(
        f'  (gr_text "L={actual_len:.0f}mm  pq/(p+q)=0  {hole_count} holes" (at {ctrl_cx} {sub_y}) '
        f'(layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 0.8 0.8) (thickness 0.12)))'
        f'\n  )'
    )

    # SMA for control — centered on meander start
    sma_y_ctrl = BOARD_H - 8.0
    sma_x_ctrl = meander_x[0]
    parts.append(sma_pad(sma_x_ctrl, sma_y_ctrl, ctrl_net_id, f"J{ctrl_idx+1}"))
    parts.append(sma_ground_patch(sma_x_ctrl, sma_y_ctrl))


    print(f"  {ctrl['label']:<20} feed=({meander_x[0]:.1f},{meander_y[0]:.1f}), "
          f"{hole_count} holes, 0 crossings, L={actual_len:.1f}mm")

    # ── Perimeter ground ring: F.Cu trace along board edges ──
    # Connects directly to SMA ground patches on F.Cu (both overlap at
    # the 5mm inset). No vias needed — SMA thru-hole pads already
    # bridge F.Cu ↔ B.Cu at each connector location.
    gnd_inset = 5.0  # mm from board edge
    gx0 = gnd_inset
    gy0 = gnd_inset
    gx1 = BOARD_W - gnd_inset
    gy1 = BOARD_H - gnd_inset
    gnd_trace_w = 2.0  # mm (wide enough for low-inductance ground return)

    # Perimeter rectangle on F.Cu
    for (sx, sy, ex, ey) in [
        (gx0, gy0, gx1, gy0),  # top
        (gx1, gy0, gx1, gy1),  # right
        (gx1, gy1, gx0, gy1),  # bottom
        (gx0, gy1, gx0, gy0),  # left
    ]:
        parts.append(
            f'  (segment (start {sx:.4f} {sy:.4f}) (end {ex:.4f} {ey:.4f}) '
            f'(width {gnd_trace_w}) (layer "F.Cu") (net 1) (uuid "{new_uuid()}"))'
        )

    print(f"  Perimeter ground: F.Cu trace at {gnd_inset}mm inset (no vias needed)")

    # Title block (top of board, above SMA area)
    parts.append(
        f'  (gr_text "HOPF-01 v3" (at {GRID_CX} 3) (layer "F.SilkS") (uuid "{new_uuid()}")'
        f'\n    (effects (font (size 2.5 2.5) (thickness 0.4)))'
        f'\n  )'
    )
    parts.append(
        f'  (gr_text "Wire-Stitched Torus Knot Fixture + Control" (at {GRID_CX} {BOARD_H / 2}) (layer "F.SilkS") (uuid "{new_uuid()}")'
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
