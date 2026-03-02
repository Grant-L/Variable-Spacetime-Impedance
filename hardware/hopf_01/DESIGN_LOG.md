# HOPF-01 Design Log

## Revision History

### v3.4 — JLCPCB DRC Compliance (2026-03-02)

**Problem**: 759 DRC violations from KiCad DRC check.

**Root causes & fixes**:

| # | Issue | Root Cause | Fix | Impact |
|---|---|---|---|---|
| 1 | 12 pad-to-pad clearance errors | END anchor pads placed 3mm from SMA ground pads | Removed all END pads entirely — wire terminates at last stitching hole | -12 errors |
| 2 | ~50 hole-to-hole warnings at crossings | Spacing filter only checked LAST hole, not all holes — missed cross-segment overlaps | Global O(n²) check against ALL placed holes | Holes reduced: (3,5) 57→47, (3,11) 84→62 |
| 3 | 14 co-located meander holes | Segment endpoints generated for both adjacent segments | Dedup via coordinate set + MIN_HOLE_SPACING check | Meander holes 54→40 |
| 4 | ~680 lib_footprint_issues | "HOPF01" library not in KiCad config | Benign — footprints are inline in .kicad_pcb | No action needed |
| 5 | Silkscreen overlapping holes | OVER/UNDER labels placed at crossing points | Labels pushed outside knot bounding box + 4mm margin, arrows from text edge | Clean silkscreen |
| 6 | No wire start/end indication on meander | No labels on meander | Added FEED/OPEN silkscreen labels with arrows | Clear assembly guide |

**Design constants changed**:
- `MIN_HOLE_SPACING`: new constant, 1.5mm center-to-center (JLCPCB: ≥0.5mm edge-to-edge for 1.0mm drill)
- Anchor pad offset: removed (pads deleted)

**Final hole counts** (with global spacing enforcement):

| Knot | Wire Length | Holes (before) | Holes (after) | Δ | Reason |
|---|---|---|---|---|---|
| (2,3) Trefoil | 120mm | 40 | 37 | -3 | 3 crossings |
| (2,5) Cinquefoil | 160mm | 54 | 51 | -3 | 4 crossings |
| (3,5) | 170mm | 57 | 47 | -10 | 10 crossings |
| (3,7) | 200mm | 67 | 55 | -12 | 14 crossings |
| (3,11) | 250mm | 84 | 62 | -22 | 22 crossings |
| Control (meander) | 120mm | 40 | 40 | 0 | No crossings |

**Observation**: Holes removed ≈ number of crossings. Each crossing where two path segments pass within 1.5mm of each other causes one hole to be skipped.

---

### v3.3 — Ground Architecture Simplification (2026-03-02)

- Switched perimeter ground trace from B.Cu to F.Cu
- Moved SMA ground patches from B.Cu to F.Cu
- Removed all per-SMA ground via rings (24 vias)
- Removed perimeter via stitching
- Rationale: SMA thru-hole pads bridge both layers; F.Cu trace directly connects to SMA ground patches

### v3.2 — (3,5) Knot Addition (2026-03-01)

- Added 5th torus knot: (3,5), L=170mm
- Updated all simulation scripts with 5th color in palettes
- Updated board size from 120×120mm to 160×120mm
- 6 SMA connectors (was 4)

### v3.1 — Wire-Stitched Design (2026-02-27)

- Replaced PCB trace coils with wire-stitched NPTH holes
- 24 AWG enameled magnet wire threaded through unplated holes
- Wire acts as free-space resonator — substrate-independent
- Board serves only as mechanical scaffold

### v3.0 — Initial Wire-Stitched Concept

- 2-layer FR-4, ENIG finish
- 4 torus knots + 1 control meander
- SMA edge-launch connectors

## Lessons Learned

1. **Always check ALL holes for spacing**, not just the previous one — knot crossings create path-to-path proximity that single-predecessor checks miss
2. **Anchor pads are unnecessary** for wire-stitched designs — the last stitching hole IS the wire end
3. **Silkscreen must be placed outside the knot bounding box**, not offset from individual features
4. **Arrow lines must start from text edge**, not text center, or they obscure the label
5. **JLCPCB edge-to-edge hole clearance** of 0.5mm maps to 1.5mm center-to-center for 1.0mm drills
