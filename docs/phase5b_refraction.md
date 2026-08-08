# Phase 5b — refraction on a complex coastline

**Status: built, gated, and selectable as `nearshore.refraction: rays`. Not the
default.** `pywave/rays.py` passes every gate except 5b.7, including 5b.6 — the
one that motivated the phase, which was a strict xfail until the wiring landed
and is now a passing test measuring **exactly 0** where `snell` measures 0.448 m.

It is not the default because that is a judgement about rendered output, not
about the gates: selecting it changes the wave height in almost every sheltered
cell, and everything downstream — breaking, foam, wetness, swash — follows. On a
smooth synthetic beach `snell` is exact and free, so `rays` there is equal and
slower. Section 5b below has the per-gate detail.

A design for replacing the per-cell refraction approximation with something that
survives a real shoreline, plus the gate that would say it works.

---

## 1. Why the current implementation cannot be patched

Today, for each cell:

```
alpha_0 = angle between the deep-water wave direction and the local shore normal
alpha   = Snell's law applied to that angle at the local depth
Kr      = sqrt(|cos alpha_0| / |cos alpha|)
H       = H_deep * Ks * Kr
```

That `Kr` is a **ray-tube result derived for straight, parallel depth contours**.
Between two contours the ray spacing changes by `cos alpha / cos alpha_0`, energy
flux is conserved, and the height follows. The derivation needs the contours to
be straight and parallel *along the whole ray path*.

On the shipped synthetic beaches they are, so the formula is right there and the
Gate 5 checks pass. On a real coastline the premise is false, and three things
break — none of which is a coding error:

**a. The formula is evaluated per cell, but it is a statement about a path.**
`Kr` compares ray spacing *here* with ray spacing *in deep water*. There is no
ray. The cell has no memory of where the wave came from, so waves that should
have been bent long before arriving are treated as arriving straight.

**b. `shore_normal` is discontinuous, and legitimately so.** The bed is carved as
`depth = A * s^(2/3)` with `s` the distance to the shoreline. Any distance field
has a **medial axis** — points with two equally-near sources — where the
direction to the nearest source flips. Measured on the Strait of Hormuz export:
turn per cell of 0.445° off the axis against **up to 180°** on it. Depth stays
smooth; direction does not. Feed a 180° flip to `Kr` and it returns a
discontinuity.

Note this is *not* a defect in the export. The direction to the nearest shore
genuinely is discontinuous there. The defect is treating it as the contour
normal, which it only equals for a straight coast.

**c. `Kr -> 0` as `alpha_0 -> 90°`.** A wave running parallel to the shore normal
has a ray tube that never reaches the beach, and the formula annihilates it.
**2.10%** of that water body is within 3° of that condition, in wedges radiating
from every concavity. Gain there: 0.428 against 0.942 without refraction.

Smoothing the input does not rescue this — measured, smoothing the contour normal
at sigma from `lambda_p/8` to `lambda_p` reduced the p99 roughness by only ~25%.
A formula whose premise is void is not fixed by tidying its arguments.

---

## 2. What the replacement has to do

| Requirement | Why |
|---|---|
| Continuous `H` wherever the bed is continuous | The seams are the whole problem |
| Focusing on headlands, sheltering in bays | The physics worth keeping |
| No dependence on `shore_normal` | Discontinuous by construction |
| Directional spread, not one ray direction | A wind sea is not monochromatic |
| Deterministic, seed-reproducible | The existing contract |
| Precomputable per scene, not per frame | Animation must stay affordable |

That last row is the key architectural constraint: the wave field is
**stationary** given a bed and a sea state. Whatever we solve, we solve **once
per scene** and store as fields on the bathymetry grid, exactly as `foam` and
`wetness` already are. Per-frame cost must stay at today's level.

---

## 3. Three candidate solvers

### 3a. Ray tracing / wave-action balance — *recommended*

Integrate rays from deep water toward shore, obeying

```
d(x)/dt = c_g * (cos theta, sin theta)
d(theta)/dt = -(1/k) * dk/dn          (rays bend toward shallow)
```

and conserve wave action `E/omega` in each ray tube. `Kr` becomes the *measured*
tube width ratio rather than an assumed one. Cast rays for a fan of deep-water
directions weighted by the spreading function, accumulate energy onto the grid,
and `Hs_local` is the accumulated energy — continuous by construction because
energy is deposited by many overlapping tubes.

- **Handles:** focusing, sheltering, diffraction-free shadowing, caustics-as-
  bright-regions.
- **Fails at:** true diffraction (energy leaking into geometric shadow behind an
  island), and caustics where tube width goes to zero.
- **Cost:** `O(n_rays * path_length / step)`. For 7.5 × 8.6 km at 5 m steps and
  ~40 directions × ~2000 launch points, roughly **10^8 steps** — minutes,
  single-threaded, once per scene. Embarrassingly parallel over rays; 96 cores
  brings it to seconds.
- **Effort:** the largest piece, but self-contained. New module, no changes to
  the surface synthesis.

### 3b. Mild-slope equation

Solve the elliptic mild-slope equation for the complex wave amplitude over the
whole domain:

```
div(c*c_g*grad(phi)) + k^2*c*c_g*phi = 0
```

- **Handles:** refraction *and* diffraction, correctly, in one solve. This is the
  reference answer.
- **Cost:** a sparse linear solve on the full grid **per frequency and per
  direction**. On a 2804² grid that is ~7.9 M unknowns, complex-valued, and it is
  a Helmholtz problem — notoriously ill-conditioned for iterative solvers at high
  wavenumber. Needs ~10 frequencies × ~10 directions. **Hours**, and a hard
  dependency on a serious sparse solver.
- **Verdict:** correct but disproportionate. Worth building as an *offline
  reference* on a small domain to validate 3a against, not as the production path.

### 3c. Boussinesq / non-linear shallow water

Time-march the depth-averaged non-linear equations with dispersive correction
terms. This is what COULWAVE / FUNWAVE do.

- **Handles:** everything above, plus non-linear shoaling, wave-wave interaction,
  harmonic generation, run-up, and breaking as an emergent process rather than a
  `gamma_b` cap.
- **Cost:** an explicit time step on the full grid, CFL-limited. Resolving
  `lambda_p = 13.6 m` needs ~1 m cells; CFL at 18 m depth gives `dt ~ 0.05 s`;
  spin-up needs minutes of wave time. That is **10^4–10^5 steps on 10^7 cells** —
  GPU work, hours per scene, and it produces a *time series* rather than a
  stationary field.
- **Verdict:** wrong tool for this job. It would replace the entire FFT surface
  synthesis, not just refraction, and it discards the reproducibility and
  spectral-truth properties the whole model is built on. Worth knowing as the
  physical ceiling; not worth building.

**Recommendation: 3a for production, 3b offline on a small domain as the
validation reference, 3c never.**

---

## 4. Architecture

```
pywave/rays.py                       NEW
    RayField.solve(bathy, cfg)       -> stationary energy + direction fields
    RayField.sample(x, y)            -> (gain, theta) at arbitrary points
    RayField.save/load(dir)          cached per (bathy hash, sea state)

pywave/nearshore.py                  CHANGED
    refraction mode gains "rays"; snell/blend/none stay
    transform() reads gain and theta from a RayField when mode == "rays"

pywave/bathymetry.py                 UNCHANGED
scripts/run_scene.py                 solves once, caches, reports timing
```

The `RayField` is a scene-level artifact like the foam spin-up: computed once,
cached to `runs/<scene>/rays/`, keyed by a hash of the bathymetry and sea state
so it invalidates correctly. **Per-frame cost is a bilinear sample — unchanged
from today.**

### Runtime impact

| Stage | Today | With rays |
|---|---|---|
| Scene setup | ~1 min (foam spin-up dominates) | **+ minutes** (once, parallel) |
| Per mesh frame | unchanged | unchanged |
| Animation of N frames | N × frame | N × frame + one solve |
| Memory | — | 2 extra float32 fields on the bathy grid |

The one-off cost lands in the same place the foam spin-up already does, which is
the honest place for it.

---

## 5. Gate 5b

Each check has a number, not a vibe. Gates 1–6 style.

| # | Check | Criterion |
|---|---|---|
| 5b.1 | **Straight beach reduces to Snell.** On a planar coast, ray `Kr` vs the analytic formula | within 2% |
| 5b.2 | **Continuity.** p99.9 one-cell jump in gain on the straits export | < 0.02 (matching `blend`, against 0.375 today) |
| 5b.3 | **No annihilation.** min gain over all wet cells | > 0.05 |
| 5b.4 | **Focusing survives.** mean gain on headlands vs adjacent bays | ratio > 1.2 |
| 5b.5 | **Energy conservation.** total flux in vs out across a deep-water control line | within 5% |
| 5b.6 | **Independence from `shore_normal`.** perturb `shore_normal` by 30°, resolve | gain unchanged to 1e-12 |
| 5b.7 | **Mild-slope agreement.** vs a 3b solve on a 500 m test domain | correlation > 0.9 |
| 5b.8 | **Reproducibility.** same scene twice | bitwise identical |
| 5b.9 | **Cost.** solve time on the 7.5 × 8.6 km straits export | recorded, not capped |

5b.6 is the one that would have caught today's bug, and it is worth writing first.

### Deliberately out of scope

Diffraction behind islands (needs 3b), wave–current interaction, reflection off
cliffs, non-linear transfer. Each gets a note in the approximations table.

---

## 5b. Implementation status

`pywave/rays.py` exists and is **not wired into anything** — the production path
is untouched until the gates pass.

| Gate | Status | Measured |
|---|---|---|
| 5b.1 straight beach reduces to Snell | **PASS** | worst 0.02 / 0.50 / 1.78% at 0 / 20 / 40° incidence, against 2% |
| 5b.2 continuity | **PASS** | 0.0123 on the full export, 0.0141 on the crop, against 0.02 |
| 5b.3 no annihilation | **PASS** | 0.0896 on the full export, 0.1164 on the crop, against 0.05 |
| 5b.4 focusing survives | **PASS** | headland/bay 1.34 at 2–5 m depth, against 1.2 |
| 5b.5 energy conservation | **PASS** | 0.024% flux drift across a focusing shoal, against 5% |
| 5b.6 independent of `shore_normal` | **PASS** | **exactly 0**, against 0.448 m for `snell` on the same bed |
| 5b.7 mild-slope agreement | not started | — |
| 5b.8 reproducibility | **PASS** | bitwise, on a repeat solve |
| 5b.9 cost | recorded | 14 s for the crop at `decimate=32`, 77 s for the full export at `decimate=4` |

Direction, added after the gate list was written, is validated the same way the
gain is: on a planar beach against Snell solved on the full dispersion relation,
worst error **0.017°** at 20° incidence and **0.038°** at 40°, over bands where
the rays turn 13.6° and 27.5°. 5b.1 alone could not have caught a solver that
turned the waves any way it liked, because a wave can turn without changing
height — and on a mild slope at normal incidence it mostly does.

Where each number comes from, because they are measured on two different scenes:

- `tests/test_rays.py` runs 5b.1 on the planar beach and 5b.2, 5b.3 and 5b.8 on
  the **701 m crop**, skipping the last three when no export is present. They
  share one solve at `decimate=32, n_dirs=15, rays_per_dir=1500, smooth_m=80`,
  stated once so that no gate can be passed by retuning the solve for it.
- `scripts/gate5b.py` runs the same checks on the **full 7.5 × 8.6 km export**,
  which is where the gates are actually stated and where a solve is minutes
  rather than seconds. It takes `--no-wind-sea` for the A/B below.

The crop is not the hard case and should not be read as one: its shadows are too
short to reach zero on their own. It is in the suite because it is affordable
and it is real bathymetry; the export is what the gate means.

### Measuring a "one-cell jump"

The number is meaningless without the spacing it is measured at, and two rules
matter more than the percentile:

- **Fix the separation, not the cell.** The reference figures below are at the
  full export's 4 m posts. The crop is on 0.25 m posts, so the same gate is a
  lag of 16 posts there. Comparing a 0.25 m jump against a 4 m criterion would
  pass anything.
- **Require water the whole way between, not just at both ends.** On a ragged
  shoreline two cells 4 m apart are routinely both wet with land in between.
  That pair is a shoreline, not a discontinuity in the wave field. This is how
  the metric bug announced itself: counting those pairs made p99 improve with
  smoothing while p99.9 got worse, because a wider kernel pulls more shore into
  each sample.

### 5b.2, on the Strait of Hormuz export at 4 m

Jumps counted **between two wet neighbours only**. A wet cell beside a dry one is
a shoreline, not a discontinuity in the wave field. Measuring across that
boundary makes every method look bad *and* makes smoothing look actively harmful
— which is how the metric bug announced itself, since p99 improved with smoothing
while p99.9 got worse.

| Field | p99 jump | p99.9 jump |
|---|---|---|
| `snell` | 0.0923 | **0.8819** |
| `blend` | 0.0093 | **0.3001** |
| rays, no smoothing | 0.0432 | 0.1530 |
| rays, σ = 40 m | 0.0109 | 0.0236 |
| **rays, σ = 80 m** | 0.0091 | **0.0121** |
| rays, σ = 160 m | 0.0058 | 0.0081 |

Deep-water median gain came out **0.997–0.999** against a required 1.0. Nothing
in the solve forces that, so it is an independent check that the launch
normalisation is right.

### What the smoothing is, honestly

Energy deposition is Monte Carlo, so the raw field is shot noise. At 116 ray
visits per cell the gain noise is ~4.6%, and 5b.2 needs under ~1.4% — 11× more
sampling. Two levers avoid paying that:

- **`decimate`** — solve on a coarser grid. The gain field varies over the
  *bathymetry's* length scale, not the mesh's, so 16 m instead of 4 m loses
  nothing real and multiplies visits per cell by 16. Solve time 348 s.
- **`smooth_m`** — a deposition kernel of finite width. A ray is a wave *packet*,
  not a line, and energy cannot localise finer than about a wavelength.

The kernel is currently an **empirical noise control, not a derived length**. The
principled version is more rays until it is unnecessary; σ = 80 m is what the
table above says suffices — 5.9 λ_p at the crop's λ_p = 13.6 m, which is well
past what "a ray is a wave packet" justifies. That belongs in the approximations
table when the phase lands, stated as such.

### The decimation seam

`decimate` cost a bug on the way in, and it is worth recording because it looked
exactly like physics. Upsampling the coarse gain back to the fine grid with
`zoom` blends neighbouring cells, so the zeros sitting in the dry cells were
dragged into the water within one *coarse* cell of every shoreline — a seam
created **after** the smoothing, which is why no amount of smoothing touched it.

It was invisible in aggregate and unmistakable once localised: at
`decimate=16` on the crop, **100%** of the worst 0.1% of 4 m gain jumps lay
inside that 4 m band, against 2.2% of the sampled pairs. Excluding the band
dropped p99.9 from 0.396 to 0.026.

The fix is to extend the gain into the dry cells by nearest-wet fill before
upsampling. Nearest-wet rather than the normalised zoom the smoothing step uses,
because water thinner than a coarse cell has no wet coarse neighbour at all, so
the normalised version divides 0/0 exactly where the fill matters. After the fix,
p99.9 is 0.026 with the band included and the worst jumps are no longer near
land at all.

### Remaining: 5b.3, geometric shadows

Ray theory has no diffraction, so a cell behind a headland that no ray reaches
gets exactly zero. The 25-direction fan over ±90° softens this without removing
it: min gain 0.000 on the full export, with **1.84%** of wet cells under 0.05.

More rays was never the answer, because the missing energy is not energy the
rays could have carried. A sheltered bay is not calm — the wind is still blowing
over it, and it grows **its own short-fetch sea**. `rays.wind_sea_floor` adds
that, and it is a real term rather than a clamp.

### How the floor avoids counting the open sea twice

For each direction of the fan, march upwind and ask how far the water runs
before it hits land. Then:

- A **blocked** direction contributes the energy the wind puts back in over that
  distance, `(F_blocked / F_scene)^1.10`.
- A **clear** direction — one that leaves the domain, or reaches the scene fetch,
  without crossing land — contributes **exactly zero**, because the rays already
  carried it. Adding it again would be the open sea counted twice, and it would
  surface as a gain floored at 1.0 across every unobstructed cell.

That second rule is what makes the term safe to *add* to the transported energy
instead of clamping over it. On fully exposed water the floor is identically
zero, so the sum is a no-op and nothing the rays computed is overwritten.

**The exponent is 1.10, not 1.00.** `Hs ~ sqrt(fetch)` is JONSWAP's
dimensionless-energy fit, and it is not what integrating *this* spectrum gives —
that sits a further `X~^0.05` above it, the same inconsistency recorded under
Gate 1 in the validation report. So `Hs ~ F^0.55` and energy `~ F^1.10`, and the
constant is pinned against `hs_spectral` itself rather than against the algebra.

Both directions are easy to read backwards, so both are worth stating.

| F / F_scene | Hs at 0.55 | Hs at 0.50 | naive is | energy at 1.10 | energy at 1.00 | 1.10 gives |
|---|---|---|---|---|---|---|
| 0.001 | 0.0224 | 0.0316 | +41% | 0.000501 | 0.00100 | 0.50× |
| 0.01 | 0.0794 | 0.1000 | +26% | 0.006310 | 0.01000 | 0.63× |
| 0.1 | 0.2818 | 0.3162 | +12% | 0.079433 | 0.10000 | 0.79× |

The extra `X~^0.05` makes this spectrum grow *faster* with fetch, so a
short-fetch cell sits **lower** than the plain energy law predicts. Taking the
exponent from the fit would over-floor every sheltered cell against a sea the
surface synthesis never produces.

And the exponent is not a 10% inflation of anything: the fetch ratio is capped
at 1, so 1.10 always yields **less** energy than 1.00 would, by 37% at a
hundredth of the scene fetch. It is the conservative choice of the two.

### The floor needs the same kernel as the rays, for a different reason

`shelter_fetch` asks a yes/no question of a geometric line, so a cell whose
upwind ray grazes a headland tip is "blocked at 6 km" while its neighbour is
"not blocked at all". The fan bounds that step at one direction's weight, and
that is not small enough. Added unsmoothed, it took the p99.9 gain jump on the
straits export from 0.0123 to **0.0235** — straight through gate 5b.2, having
just fixed 5b.3.

The ray field is smoothed because it is Monte Carlo. The floor is smoothed
because a wind sea has no knife edge either. Same kernel, unrelated
justifications, and both belong in the approximations table.

### The floor knows the height, and now the frequency too

One number for the floor is right about how much energy a sheltered cell has and
silent about **what kind of waves** it is. That silence is not a detail. A short
fetch does not scale the spectrum down, it moves the peak **up** — at 200 m of
fetch the peak period is 0.91 s against the incident sea's 2.95 s — so a
sheltered bay is short steep chop with no swell in it, not a small copy of the
incident swell.

`band_fetch_response` tabulates `m0_band(F) / m0_band(F_scene)` per band. On the
straits crop, as a multiplier on each band's *own* deep-water energy:

| local fetch | band 1 | band 2 | band 3 | scalar |
|---|---|---|---|---|
| 2000 m | 0.052 | **1.222** | 1.125 | 0.252 |
| 500 m | 0.000 | 0.023 | **1.057** | 0.055 |
| 200 m | 0.000 | 0.000 | **0.403** | 0.020 |

The scalar is wrong in both directions — 5× too much energy in the long band,
20× too little in the short one. Note bands 2 and 3 exceeding 1.0: a sheltered
cell genuinely holds *more* short-wave energy than open water, because the whole
spectrum shifted into that band, and a ratio capped at 1 cannot say so.

**What this does not change is the total.** Summed over bands weighted by their
deep-water shares, the banded form agrees with `(F/F_scene)^1.10` to 4e-4 — the
residual is the variance above the top band edge. So gate 5b.3, which is a
statement about total height, is indifferent to all of this. What depends on it
is every per-band quantity downstream, which is to say the texture of a
sheltered bay rather than its wave height.

`wind_sea_floor(..., bands=...)` returns `(n_bands, ny, nx)` and is ready for
the wiring step. `RayField.gain` is still a single field, because making it
per-band requires one ray solve per tile frequency — that is the same job as
teaching `nearshore.transform` a `"rays"` mode, and it is not done yet.

### What the floor actually bought

`scripts/gate5b.py configs/straits.yaml`, with and without `--no-wind-sea`, at
`decimate=4, 15 dirs x 1500 rays, sigma 80 m`:

| | min gain | wet cells under 0.05 | 5b.2 p99.9 |
|---|---|---|---|
| rays only | **0.000** | 1.84% | 0.0123 |
| + wind sea, floor unsmoothed | 0.0384 | 0.0030% | **0.0235** |
| **+ wind sea, floor smoothed** | **0.0896** | **0.0000%** | **0.0123** |

The third row is the point: 5b.3 goes from a hard zero to comfortably inside the
gate, and 5b.2 is **unchanged to four decimals** — the floor is not buying one
gate with the other. 99.3% of wet cells have some direction blocked, so this is
not a term that only touches a few corners of the domain; it is doing work
almost everywhere and still leaving the ray solution alone where it matters.

---

## 6. Order, as planned and as it went

1. ~~Gate 5b.6 as a failing test against today's code — pins the bug.~~ **done**
2. ~~`rays.py` with a straight beach only; pass 5b.1.~~ **done**
3. ~~Real bathymetry; pass 5b.2, 5b.3.~~ **done**
4. ~~Caching~~ **done**; `run_scene` integration outstanding.
5. ~~Focusing and energy checks; 5b.4, 5b.5.~~ **done**
6. Mild-slope reference on a small domain; 5b.7. **not started**

Three things had to be built that this list did not anticipate, and each was
forced by a measurement rather than foreseen:

- **A direction field.** The list is all about height, and `RayField` originally
  carried only gain — but `transform` rotates each tile by the local wave
  direction, and 5b.5's flux needs a direction to be a flux *of*. Validated
  against Snell's angle to 0.017°.
- **The wind-sea floor.** 5b.3 was not reachable by more rays, because the
  missing energy was never energy the rays could carry.
- **Per-band everything.** `transform` works per band and the solver did not.
  This was the last thing standing between the gates passing and the mode being
  selectable, and it is why 5b.6 stayed an xfail for so long after the physics
  was right.

### What is left

| | |
|---|---|
| `run_scene` integration | solve once per scene, cache, report timing |
| Deciding the default | needs the measurements below in front of a human |
| 5b.7 | a mild-slope reference solver on a small domain |

The default is deliberately not being changed as part of the wiring. Selecting
`rays` moves the wave height in almost every sheltered cell, and breaking, foam,
wetness and swash all read that height. Measured on the straits crop, at the
same sample points:

| mode | min Hs | median | max | cells at exactly 0 | breaking |
|---|---|---|---|---|---|
| `snell` | **0.00000** | 0.6544 | 0.7007 | **0.066%** | 11.65% |
| `blend` | 0.0754 | 0.6699 | 0.7009 | 0 | 13.61% |
| `rays` | 0.0635 | 0.7086 | **0.7754** | 0 | 10.19% |

`Hs_deep` is 0.7118 m, so `rays` is the only mode whose maximum exceeds deep
water — that is the headland focusing, and `blend` cannot produce it at any
setting because it pins `Kr = 1`. Per-frame cost is 2.5 s against `blend`'s 2.3
and `snell`'s 4.1: sampling a precomputed field is cheaper than solving
dispersion per band, so the wiring makes frames *faster*, not slower.

What those numbers do not yet show is the interaction with foam. `coastal_bay`
already saturates at coverage 1.0 across its surf band, and more energy in
sheltered water can only push that further in. That should be measured before
the default moves, not after.
