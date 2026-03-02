# HOPF-01 JLCPCB Ordering Guide

## Quote Settings

Upload the **Gerber ZIP** exported from KiCad (Plot → Gerber, then drill files).

| Parameter | Value | Notes |
|---|---|---|
| **Base Material** | FR-4 | Standard glass-epoxy |
| **Layers** | 2 | F.Cu + B.Cu |
| **Dimensions** | 160 × 120 mm | Auto-detected from Gerber |
| **PCB Qty** | 5 | Minimum practical order |
| **PCB Thickness** | 1.6 mm | Standard |
| **PCB Color** | Green (or Black) | Black hides the B.Cu bare copper aesthetic |
| **Silkscreen** | White | Default |
| **Surface Finish** | **ENIG** | Required for clean SMA soldering; HASL acceptable but ENIG preferred |
| **Copper Weight** | 1 oz (35 µm) | Standard |
| **Via Covering** | Tented | Default (we have no vias, but leave default) |
| **Board Outline Tolerance** | ±0.2 mm | Standard |
| **Remove Order Number** | Yes | Specify "Remove" to keep the board clean |
| **Confirm Production File** | Yes | Review before fabrication |
| **Flying Probe Test** | Fully Test | Confirms no shorts on GND traces |
| **Castellated Holes** | No | |
| **Edge Plating** | No | |
| **Impedance Control** | No | Wire antennas, not controlled-impedance traces |

## Critical Design Rule Compliance

| JLCPCB Rule | Required | Our Design | Status |
|---|---|---|---|
| Min NPTH drill | ≥ 0.5 mm | 1.0 mm | ✅ |
| Hole-to-hole (edge clearance) | ≥ 0.5 mm | ≥ 0.5 mm (1.5 mm c-c) | ✅ |
| Min track width | ≥ 0.127 mm | 0.5 mm (GND trace) | ✅ |
| Copper to edge | ≥ 0.3 mm | 5.0 mm (GND inset) | ✅ |
| Min annular ring (PTH) | ≥ 0.13 mm | 0.15 mm (SMA pads) | ✅ |
| Silkscreen line width | ≥ 0.15 mm | 0.15 mm | ✅ |
| Silkscreen text height | ≥ 0.8 mm | 0.8 mm (min) | ✅ |
| Silkscreen to pad | ≥ 0.15 mm | Auto-clipped by JLCPCB | ✅ |
| Pad-to-pad clearance | ≥ 0.2 mm | ≥ 8.0 mm (SMA to nearest) | ✅ |
| Board thickness | 0.4–2.4 mm | 1.6 mm | ✅ |

## Bill of Materials (order separately)

| Item | Qty | Part | Source |
|---|---|---|---|
| SMA Edge-Launch 50Ω | 6 | TE CONSMA003.062 | LCSC / Mouser |
| 24 AWG Enameled Wire | 5 m | Magnet wire, 0.51 mm | Amazon / eBay |
| M3 Standoffs (10 mm) | 4 | Nylon or brass | Amazon |
| M3 Screws | 8 | Pan head | Amazon |

## Assembly Notes

1. **No SMT assembly needed** — all components are hand-soldered
2. **Wire threading**: Follow the silkscreen guide lines and OVER/UNDER crossing markers
3. **SMA soldering**: Solder center pin to wire start, ground tabs to F.Cu ground patch
4. **Wire end**: Leave open at last stitching hole (open-circuit resonator)
5. **Meander control**: FEED end connects to SMA, OPEN end is left floating
