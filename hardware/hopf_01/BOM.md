# HOPF-01 v3 — Bill of Materials

Wire-stitched torus knot antenna fixture for AVE chiral verification.

| # | Item                                        | MPN / Spec           | Qty | Est. Price |
|---|---------------------------------------------|----------------------|-----|------------|
| 1 | PCB, 160×120mm, 2-layer FR-4, ENIG, 1.6mm   | JLCPCB custom        | 5   | $12        |
| 2 | SMA panel-mount connector, 50Ω               | TE CONSMA003.062     | 6   | $35        |
| 3 | 24 AWG enameled magnet wire, 1 lb spool      | Remington 24HPN      | 1   | $14        |
| 4 | SMA M–M cable, RG316, 30cm                   | Generic              | 1   | $8         |
| 5 | Mineral oil 500mL (transformer grade)         | Generic              | 1   | $10        |
| 6 | Glass dish ≥170×130mm, ≥25mm deep             | Pyrex 9×13"          | 1   | $10        |
| 7 | M3×10mm nylon standoffs + M3 hardware         | Generic              | 1   | $8         |
| 8 | SMA 50Ω terminators (for unused ports)        | Generic              | 5   | $5         |
| **TOTAL** |                                       |                      |     | **$102**   |

## Test Equipment (not included in total)

| Item | Est. Price |
|------|------------|
| LiteVNA-64 (1 MHz – 6.3 GHz, calibration kit) | $100 |
| NanoVNA-H4 (budget option, 1.5 GHz max)        | $60  |

## Board Specifications

- **Dimensions**: 160 × 120 mm
- **Layers**: 2 (F.Cu + B.Cu)
- **Stackup**: FR-4, 1.6mm, ENIG finish
- **Antennas**: 5 torus knots + 1 meander control = 6 total
- **SMA connectors**: 6 × panel-mount (thru-hole, GND on *.Cu layers)
- **Ground**: B.Cu local patches under SMAs, perimeter ground ring (via-stitched)
- **Stitching holes**: NPTH, 1.0mm drill (specify unplated in order)

## Wire Cutting Guide

| Antenna | Wire Length | + Solder Tails | Cut Length |
|---------|------------|----------------|------------|
| (2,3) Trefoil | 120 mm | 20 mm | 140 mm |
| (2,5) Cinquefoil | 160 mm | 20 mm | 180 mm |
| (3,5) | 170 mm | 20 mm | 190 mm |
| (3,7) | 200 mm | 20 mm | 220 mm |
| (3,11) | 250 mm | 20 mm | 270 mm |
| CONTROL (meander) | 120 mm | 20 mm | 140 mm |
