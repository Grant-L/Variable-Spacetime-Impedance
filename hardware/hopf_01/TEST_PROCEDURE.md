# HOPF-01 Wiring & Test Procedure

Wire-stitched torus knot antenna fixture for AVE chiral verification.
Board ordered from JLCPCB. See [BOM.md](BOM.md) for parts, [ORDERING.md](ORDERING.md) for fab settings.

---

## Phase 1: Preparation

### 1.1 Cut Wire Lengths

| # | Antenna | Topology | Wire Length | + Tails | Cut Length |
|---|---------|----------|------------|---------|------------|
| 1 | Trefoil | (2,3) | 120 mm | 20 mm | **140 mm** |
| 2 | Cinquefoil | (2,5) | 160 mm | 20 mm | **180 mm** |
| 3 | — | (3,5) | 170 mm | 20 mm | **190 mm** |
| 4 | — | (3,7) | 200 mm | 20 mm | **220 mm** |
| 5 | — | (3,11) | 250 mm | 20 mm | **270 mm** |
| 6 | CONTROL | meander | 120 mm | 20 mm | **140 mm** |

**Measure with a ruler**, not by eyeballing. ±0.5 mm tolerance is fine.

### 1.2 Strip Enamel

Strip 5 mm of enamel from **one end only** of each wire (the FEED end).
Methods: 600-grit sandpaper, lighter + alcohol wipe, or enamel stripper.

### 1.3 Mount Standoffs

Attach 4× M3 nylon standoffs (10 mm) to the corner mounting holes.
Board must be elevated to allow wire to route freely underneath.

### 1.4 Solder SMA Connectors

Solder 6× SMA edge-launch connectors to their marked positions.
- Center pin faces inward toward the board
- Ground tabs solder to F.Cu ground patches
- Verify continuity: center pin isolated from ground

---

## Phase 2: Wire Threading

### General Rules
1. **Follow the silkscreen trace** — white guide lines show the path
2. At **OVER** markers: wire passes OVER the board (in air on top)
3. At **UNDER** markers: wire passes UNDER the board (through the hole and under)
4. Thread from the **FEED** silkscreen label toward the **OPEN** label
5. Keep wire taut between holes, but don't kink it
6. Aim for ~1 mm clearance above/below the board at crossings

### Per-Antenna Procedure

For each antenna (start with the trefoil, work up in complexity):

1. Insert the stripped wire end through the **first** stitching hole nearest the SMA
2. Pull ~10 mm through to the underside for soldering later
3. Route the wire through successive holes following the silkscreen path
4. At each **crossing**:
   - The silkscreen shows **OVER** or **UNDER** for each strand
   - Thread accordingly — one strand goes above the board, the other below
5. At the last stitching hole (**OPEN** label): pull wire through, cut flush + 2 mm
6. **Do NOT solder the OPEN end** — it must be an open-circuit terminator

### Control Meander

The control antenna is a simple zigzag (zero topology). No crossings.
Just thread from FEED to OPEN following the silkscreen meander path.

### Solder FEED Connections

After all 6 antennas are threaded:
1. Flip the board over
2. Solder each wire's stripped end to the SMA center pin (small solder blob)
3. Verify: each SMA center pin has continuity to its wire, not to ground

---

## Phase 3: VNA Measurement (Air)

### 3.1 Calibrate

1. Connect VNA (LiteVNA-64 or NanoVNA-H4)
2. Perform **SOL** calibration at the SMA cable end:
   - **S**hort: SMA short standard
   - **O**pen: SMA open standard
   - **L**oad: 50 Ω SMA terminator
3. Save calibration

### 3.2 Measure Each Antenna

For each of the 6 antennas:

1. Connect VNA cable to the SMA connector
2. **Terminate all other 5 SMAs** with 50 Ω terminators (prevents cross-talk)
3. Set sweep: **300 MHz – 1.3 GHz**, 201 points
4. Record the **deepest S₁₁ dip** (minimum return loss)
5. Log: `f_res` (MHz), `S₁₁_min` (dB), `BW_3dB` (MHz)
6. **Repeat 10 times**, rotating the cable between measurements (averages connector noise)

### Expected Results (Air, ε_eff ≈ 1.295)

| Antenna | f_SM (GHz) | f_AVE (GHz) | Δf (MHz) | S₁₁ (dB) |
|---------|-----------|-------------|---------|----------|
| (2,3) Trefoil | 1.047 | 1.038 | 9.1 | ~ -24 |
| (2,5) Cinquefoil | 0.787 | 0.778 | 8.1 | ~ -23 |
| (3,5) | 0.740 | 0.730 | 10.0 | ~ -23 |
| (3,7) | 0.630 | 0.621 | 9.5 | ~ -23 |
| (3,11) | 0.505 | 0.496 | 8.5 | ~ -23 |
| CONTROL | 1.047 | 1.047 | 0.0 | ~ -24 |

---

## Phase 4: VNA Measurement (Mineral Oil)

1. Place the board (on standoffs) into a glass dish
2. Fill with mineral oil until the wire antennas are fully submerged (~20 mm deep)
3. Wait 2 minutes for air bubbles to clear
4. Re-calibrate SOL (same cable, cable stays dry above oil surface)
5. Repeat all 6 antenna measurements as in Phase 3

### Expected Results (Oil, ε_eff ≈ 2.265)

| Antenna | f_SM (GHz) | f_AVE (GHz) | Δf (MHz) |
|---------|-----------|-------------|---------|
| (2,3) | 0.794 | 0.787 | 7.2 |
| (2,5) | 0.597 | 0.591 | 6.4 |
| (3,5) | 0.562 | 0.554 | 7.9 |
| (3,7) | 0.478 | 0.471 | 7.5 |
| (3,11) | 0.383 | 0.377 | 6.7 |

---

## Phase 5: VNA Measurement (Vacuum, ~5 Torr)

> **Requires:** SMA vacuum feedthrough (KF-25 or panel-mount), roughing pump

1. Remove the board from the oil bath, clean thoroughly with isopropyl alcohol
2. Let dry completely (30 min or heat gun on low)
3. Place the board (on standoffs) inside the vacuum chamber
4. Connect the VNA cable via the **SMA vacuum feedthrough**
5. Pump down to **4–5 Torr** (roughing pump is sufficient)
6. Wait 2 minutes for pressure to stabilize
7. Re-calibrate SOL through the feedthrough cable
8. Repeat all 6 antenna measurements as in Phase 3

### Expected Results (Vacuum, ε_eff ≈ 1.294)

Frequencies will be essentially **identical to air** (shift < 200 kHz).
The critical test: **Δf/f must match Phases 3 and 4 exactly**.

| Antenna | f_SM (GHz) | f_AVE (GHz) | Δf (MHz) |
|---------|-----------|-------------|---------|
| (2,3) | 1.047 | 1.038 | 9.1 |
| (2,5) | 0.787 | 0.778 | 8.1 |
| (3,5) | 0.740 | 0.730 | 10.0 |
| (3,7) | 0.630 | 0.621 | 9.5 |
| (3,11) | 0.505 | 0.496 | 8.5 |

---

## Phase 6: Analysis

### 6.1 Extract the Anomaly

For each knot *i* and each medium:
```
Δf_i = f_measured_i − f_control × (L_control / L_i)
```
The control antenna provides the experimental `f_SM` reference.

### 6.2 Test the Scaling Law

Plot **Δf/f** vs **pq/(p+q)** for all 5 knots + the (0,0) control origin.

| Result | Meaning |
|--------|---------|
| Linear through origin, slope = α ≈ 1/137 | **AVE confirmed** |
| Same slope in air, oil, AND vacuum | **Substrate independence confirmed** |
| Zero, random, or non-linear | **AVE falsified** at this scale |
| Control shows nonzero shift | **Systematic error** — redo |

### 6.3 Model Selection

Fit two models to the data:
- **AVE**: Δf/f = α × pq/(p+q) — 0 free parameters (slope fixed at α)
- **Classical**: Δf/f = β × N_cross/L_self — 1 free parameter (β)

Use **AIC/BIC** to determine which model fits better.

---

## Decision Gate

| Phase | Medium | Pass Criterion | Confidence |
|:--|:--|:--|:--|
| 1 | Air | Δf/f monotonic in pq/(p+q), control null | Proceed |
| 2 | Oil | Δf/f identical to Phase 1 (±σ) | Strong |
| 3 | Vacuum | Δf/f identical to Phases 1 & 2 | **Decisive** |

- **All 3 pass**: Scaling law confirmed across 3 media → proceed to PONDER thrust
- **Any fail**: Scaling law falsified → re-examine chiral coupling term
