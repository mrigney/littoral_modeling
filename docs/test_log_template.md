# Test log — <date>, <machine>

> Copy this file, fill it in, bring it back. **Do not hand-type run numbers.**
> Everything about the sea state and the mesh is already recorded; harvest it
> with one command at the end of the day and paste it into §0:
>
> ```bash
> python scripts/collect_run_info.py -o test_log_data.md
> ```
>
> What that *cannot* know is your camera and your eyes. That is what the rest of
> this file is for. A blank field is better than a guessed one — say "didn't
> get to it" rather than leaving something that reads as a result.

Commit tested: `git rev-parse --short HEAD` → ______
(should be `89e1cd9` or later — the band-limiting fix is in `463ad36`)

---

## 0. Run data

<!-- paste the output of collect_run_info.py here -->

---

## 1. Render log

One row per image. **The camera columns are the ones I cannot recover** — without
width and FOV I cannot predict where the patterning line should fall, and with
them I can check it to a few percent.

| # | Image file | Which run / mesh_dx | Width px | FOV deg | Camera height m | Look direction | Notes |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |
| 6 | | | | | | | |

---

## Tier 1 — is the patterning line a sampling artefact?

### T1.1 Orbit test

Fix the mesh, move the camera around it.

- **Prediction:** the line stays at a constant *range from the camera* and sweeps
  across the water as you move. It does **not** stay on the same patch of sea.
- **Observed:**
- **Verdict:** pass / fail / unclear
- If it stayed glued to the same water, note *where* — that would mean it is in
  the data and everything below is moot.

### T1.2 Resolution sweep — the decisive one

Same camera, same scene, render at 1280 / 2560 / 5120 px wide.

| Width px | Predicted line range | Observed line position | Notes |
|---|---|---|---|
| 1280 | ~6.9 km (at 45°) | | |
| 2560 | ~13.8 km | | |
| 5120 | ~27.6 km (beyond the domain — expect no line) | | |

- **Prediction:** the line moves back by 2× per doubling, from
  `d ≈ (λp/2)·W/FOV`. Nothing else in the system scales that way.
- **Observed:**
- **Verdict:**

### T1.3 FOV sweep

Fixed width, 45° vs 22.5°.

- **Prediction:** halving the FOV doubles the range to the line — same formula,
  different variable. Confirms it is angular resolution, not pixel count.
- **Observed:**
- **Verdict:**

### T1.4 Before/after the band-limiting fix

Re-run `straits` at 4 m on the new commit, same camera as your existing render.

- **Prediction:** the uniform striping is gone; the sea looks **calmer**
  (geometry `Hs` 0.27 m rather than 0.40 m — that is correct, the missing
  energy is in `mss`); **the line is still there**.
- **Observed:**
- **Verdict:**
- Attach both images side by side if you can.

### T1.5 Mesh-spacing sweep at fixed camera

4 / 2 / 1 / 0.5 m, one camera.

| mesh_dx | posts/λp | Line moved? | Notes |
|---|---|---|---|
| 4 m | 2.1 | | |
| 2 m | 4.2 | | |
| 1 m | 8.5 | | |
| 0.5 m | 17.0 | | |

- **Prediction:** the main line does **not** move — it is set by `λp`, `W`, `FOV`,
  not by `dx`. Surface quality below the line improves a lot. There may be a
  second, fainter onset closer in that *does* track `dx`.
- **Observed:**
- **Verdict:** if the main line tracks `dx`, my explanation is wrong — say so.

---

## Tier 2 — physics

### T2.1 `mss` channel instead of constant alpha

- **Prediction:** offshore looks identical (α is flat at ~0.22 out there);
  the **shorebreak** gains 20–30% roughness and stops looking glassy.
- **Observed:**

### T2.2 Anisotropic BSDF (`aniso`, `wdir_x/y`)

- **Prediction:** sun glint elongates across-wind instead of round.
- **Watch for:** a seam along the **middle of the channel**, in world space —
  that is the SDF medial axis where `shore_normal` flips 180°. First time `wdir`
  is actually exercised. If it appears, note whether it moves with the camera.
- **Observed:**

### T2.3 Sea-state sweep

| U10 m/s | Fetch m | λp from summary | mesh_dx chosen | posts/λp | Result |
|---|---|---|---|---|---|
| 4 | | | | | |
| 8 | 7000 | 8.48 | | | |
| 12 | | | | | |
| 16 | | | | | |

- **Prediction:** at high wind the tiles may fail the "no tile resolves k_p"
  assert — the tile set is still the test lake's. That is the model correctly
  refusing, not a crash to report. Note the exact message if it fires.
- **Observed:**

### T2.4 Foam

- **Known open issue:** `seed_rate` saturates — on `coastal_bay` coverage pins at
  1.0 across the whole surf band. Your config leaves `foam_coverage` at the 0.85
  default.
- **Check:** does `straits` saturate too? Look at the `foam` channel range in
  the mesh sidecar (max should be < 1.0) and at the surf-zone figure.
- **Observed:** foam channel min ____ max ____ mean ____

### T2.5 Temporal coherence

60 frames, `--mesh-t` 0 → 2 s.

- **Prediction:** smooth advection, no popping or swimming.
- **Observed:**

### T2.6 Reproducibility

Same config twice; different machines if convenient.

- **Prediction:** byte-identical PLYs (`sha256sum`).
- **Observed:**

---

## Tier 3 — scale and integration

### T3.1 Practical ceiling

`--mesh-full` on `straits` at 1 m, then 0.5 m.

| mesh_dx | Vertices | Build time | PLY size | Peak RAM |
|---|---|---|---|---|
| 1 m | | | | |
| 0.5 m | | | | |

- **Prediction:** ~50k verts/s single-threaded, so 0.5 m (~34 M verts) is
  ~11 min. Wall clock binds long before RAM does.

### T3.2 Houdini terrain PLY + clearance check

```bash
python scripts/run_scene.py configs/straits.yaml --mesh --mesh-full \
       --terrain-ply <your bed>.ply
python scripts/check_clearance.py runs/straits/mesh/water_0000.ply <your bed>.ply
```

- **Prediction:** zero vertices below the bed **if** the PLY is no finer than the
  `.npy` fields. Finer → intrusions near the waterline.
- **Observed:** vertices below bed ____ , worst ____ m
- PLY resolution used: ____ m; `.npy` field resolution: ____ m

### T3.3 Adversarial terrain

A bigger / differently shaped export: islands, a coast facing another way, a
non-square grid, a negative origin.

- **Prediction:** all handled; the loader refuses only all-water or all-land.
- **Observed:**

---

## Anything surprising

The most valuable section. Things that looked wrong, were slower than expected,
produced a confusing error, or just felt off — even without a clean repro.

-
-
-

## Errors hit

Paste tracebacks verbatim, with the command that produced them.

```
```
