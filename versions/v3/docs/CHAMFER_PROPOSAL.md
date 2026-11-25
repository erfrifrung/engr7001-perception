# Proposal: print the blocks with a small top-edge chamfer (for WP1 / team decision)

**Ask:** print the Jenga blocks with a **1.0–1.5 mm × 45° chamfer on the four top edges** (and optionally the bottom edges), keeping the nominal 75 × 25 × 15 mm envelope. Commercial Jenga blocks have rounded or chamfered edges, so this is a realistic object choice, not a perception trick. WP1 owns the task objects (schedule 1.3.2.1), so this needs team agreement.

## Why

- Two or three same-colour blocks pushed flush have **no visible seam from above**. This is the standard Jenga layer. V3 resolves 0 of 36 synthetic flush triples, and no image model can find a seam that isn't in the image.
- A chamfer puts a physical **V-groove** at every seam: 2–3 mm wide and 1–1.5 mm deep for two touching blocks. It gives three independent cues the pipeline already measures:
  - a **depth trough** (`depth_trough_min_m = 1.8 mm` in `PolicyThresholds`);
  - an **RGB shadow line** (the existing RGB seam cue);
  - a break in the **side-view silhouette** (existing side verifier).

## Predictions (to test, not claims)

| Quantity | Prediction | Basis |
|---|---|---|
| Groove depth for 1.5 mm chamfers on both blocks | ~1.5 mm, width ~3 mm | geometry |
| Depth-trough detection at 0.4–0.5 m | Marginal per frame (D435 depth noise ≈ 1 mm there), reliable with 3–5 averaged stationary frames | D435 noise grows with distance squared; trough threshold 1.8 mm |
| RGB shadow-line detection | Reliable under the overhead lab lights if the groove is ≥ 2 mm wide | 1 px ≈ 0.5–0.6 mm at 0.5 m for 1280 × 720 |
| Coverage effect | flush pairs and triples become resolvable like seamed blocks (synthetic `seam=True` cases are already solved) | V3 synthetic results |

## Cost and risk

- Print time unchanged, no extra material. Make one test pair first.
- Risk: grooves catch light unevenly, and FDM top layers add their own lines. Mitigated by the periodic-stripe alias check (fixed in V3.1) and by keeping the geometric slab guard.

## Decisive test

Print 3 chamfered and 3 plain blocks. Record a flush pair and a flush triple of each, stationary, 60 frames. **Prediction:** chamfered seams are detected (depth or RGB) on ≥ 90% of frames and plain ones on ≤ 10%, with zero splits of a single chamfered block.
