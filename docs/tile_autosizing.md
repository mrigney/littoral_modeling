# Tile auto-sizing — deriving tile sizes from λ_p

**Status: planned. Nothing here is built.** Target v0.2.0.

Tile sizes are currently hand-written per scene. They are a function of the sea
state, they are copied between configs, and when they are wrong every gate still
passes. This replaces the copying with a derivation.

Prompted by a sea-state sweep (windspeed × direction × fetch, orbiting camera,
Mitsuba) in which the surface visibly repeated. It does not repeat, but
something real was wrong.

---

## 0. What this is not

**Not a fix for FFT periodicity.** Each tile is exactly periodic at its own
size, but the composite is not: incommensurate sizes and golden-angle rotations
([tiling.py:38](../pywave/tiling.py#L38)) mean the sum has no period. Nothing
here changes that, and nothing needs to.

**Not a change to the spectrum, the seed discipline, or the physics.** No new
model, no new tuning constant that is not derived from something already in the
config.

**Not a re-baseline.** See §5 — the evidence says the numbers do not move.

**Not a fix for run-to-run pattern correlation.** A sweep with a fixed seed
produces correlated wave patterns across sea states. That is a separate knob and
arguably a feature; §7 records it as an open question rather than solving it.

---

## 0a. Reading the numbers

Two normalizations appear throughout, and they run in opposite directions.

- **Lengths are in units of λ_p.** "largest tile 44.8 λ_p" means the tile's
  *physical side length* is 44.8 peak wavelengths — `coastal_bay`'s largest tile
  is 1583 m across a 35.3 m sea. It is not a statement about resolution.
- **Wavenumbers are in units of k_p.** Band edges and `k_max` — "first edge at
  2.0 k_p" — are spectral positions relative to the peak.

Since `λ_p = 2π/k_p` these are reciprocal, and for one tile at fixed `n`:

```
(L / λ_p) × (k_nyquist / k_p) = n / 2
```

`coastal_bay` tile 1: 44.8 × 5.7 = 256 = 512/2.

That identity is the whole tension in one line. **At fixed `n`, a tile cannot
get bigger without its Nyquist getting proportionally coarser.** Which is
desirable for tile 1 — the coarsening is what pulls `k_ref` down so the band
edges track k_p (§3) — and destructive for the top tile, where it drags `k_max`
down and moves resolved slope variance into the BSDF (§3a).

---

## 1. Two failures, and the visible one is the milder one

Measured with `TileSet.sizing()` on the shipped configs:

| config | λ_p | first band edge | band shares | largest tile | inert? |
|---|---|---|---|---|---|
| `test_lake` | 1.71 m | 2.1 k_p | 0.84 / 0.12 / 0.04 | 37.5 λ_p | no |
| `coastal_bay` | 35.3 m | 2.0 k_p | 0.82 / 0.13 / 0.05 | 44.8 λ_p | no |
| `straits_crop` | 13.6 m | 2.0 k_p | 0.83 / 0.13 / 0.05 | 44.7 λ_p | no |
| **`straits`** | **8.48 m** | **10.3 k_p** | **0.99 / 0.006 / 0.002** | **7.5 λ_p** | **yes** |

### 1a. Variety — what the eye sees

At 7.5 λ_p a tile holds a handful of wave groups, and those groups recur. Under
an orbiting camera with specular glint this is maximally visible, because glint
is the most sensitive detector of repetition there is — the module docstring
opens with exactly this warning
([tiling.py:3](../pywave/tiling.py#L3)).

This costs variety, not energy. Holding the Nyquist fixed, tile size moves the
realised `Hs` by under 0.1%.

### 1b. The bands going inert — what nothing sees

`band_edges` maps band *fractions* onto `k_ref = min_i k_nyquist_i`
([tiling.py:84](../pywave/tiling.py#L84)) — an absolute wavenumber fixed by the
tile geometry. Hold the geometry constant across a sweep and k_p slides around
underneath a pinned edge. On `straits` the first edge lands at 10.3 k_p and band
1 swallows 99.3% of the variance.

The cost is not cosmetic. Three FFTs a frame then compute **one representative
frequency between them**, so every frequency-dependent nearshore effect —
per-band shoaling, per-band refraction — collapses to a single band. `Hs` is
still right. The bands still sum. The LOD invariant still closes. The validation
report says PASS.

Across a sweep, an unknown fraction of runs are silently in this state.

---

## 2. `straits.yaml` is already broken, and the test that should catch it doesn't

[configs/straits.yaml](../configs/straits.yaml) carries `{64, 37, 23}` — the
`test_lake` tile block, on a sea five times larger.

[`test_every_shipped_scene_has_tiles_sized_for_its_own_sea`](../tests/test_surface.py#L501)
asserts `not s.notes` for every scene it checks, which is the right assertion.
But it iterates a hardcoded `("test_lake", "coastal_bay", "straits_crop")` and
omits `straits`. The name promises every shipped scene; the loop delivers three
of four, and the missing one is the one that fails.

**This is fixable today, independent of everything else here.** See step 1.

---

## 3. The recipe is already latent in the good configs

The three well-sized configs agree to within a few percent:

| | tile 1 | tile 2 | tile 3 |
|---|---|---|---|
| `coastal_bay` | 44.8 λ_p | 20.4 λ_p | 10.4 λ_p |
| `straits_crop` | 44.7 λ_p | 19.9 λ_p | **1.7 λ_p — never resized** |
| `test_lake` | 37.4 λ_p | 21.6 λ_p | 13.5 λ_p |

**Only tiles 1 and 2 scale.** `straits_crop`'s unresized third tile looks like
an oversight and is not — it is load-bearing. See §3a.

And the reason 45 λ_p is the right leading constant falls out algebraically.
With `L₁ = c₁·λ_p` and `n₁` fixed, tile 1 is the binding Nyquist:

```
k_ref = π·n₁/L₁ = n₁·k_p / (2·c₁)
first edge = f₁·k_ref = f₁·n₁·k_p / (2·c₁)
```

Setting the first edge to the documented target of 2 k_p and solving:

```
c₁ = f₁·n₁ / (2·target) = 0.35 × 512 / 4 = 44.8
```

which is `coastal_bay`'s 44.84 to three figures. The magic number is not magic;
it is what "put the first band edge at 2 k_p" evaluates to at `n₁ = 512`.

**So the derivation should take the target edge as the parameter and solve for
the size**, not hardcode 45. That way changing `n` keeps the bands correct
instead of silently moving them.

### The key consequence: fractions become sea-state-independent

Because `first_edge / k_p = f₁·n₁ / (2·c₁)` contains no sea-state term once
`L ∝ λ_p` with `n` fixed, the ratio is **constant across every sea state
automatically**.

This is worth stating plainly because it shrinks the change: the band
*fractions* do not need re-expressing in units of k_p. Deriving the sizes is
sufficient to fix §1b. (An earlier sketch of this work assumed both were
needed. Only the sizes are.)

### 3a. The third tile must NOT scale — it sets `k_max`

Tiles 1 and 2 scale with λ_p. The last tile does not, and scaling it is an
active mistake.

`k_ref` is set by tile 1, so the band edges do not depend on tile 3's size at
all. What tile 3 *does* set is `k_max` — the top of the resolved spectrum —
because it carries the highest band. At fixed `n`, growing `L₃` coarsens its
grid and drags `k_max` down with it.

Measured on `straits`, resizing all three tiles versus scaling only tiles 1–2:

| | first edge | band 1 share | `Hs` shift | resolved `mss` shift | `k_max` |
|---|---|---|---|---|---|
| today (broken) | 10.3 k_p | 99.3% | — | — | 34.97 |
| scale all three | 2.00 k_p | 82.9% | 3.3e-03 | **3.6e-01** | **9.14** |
| **scale 1–2 only** | 2.00 k_p | 82.5% | 9.4e-04 | 1.6e-03 | **34.97** |

Scaling all three fixes the bands and destroys 36% of the resolved slope
variance, dropping `k_max` by 3.8×. That energy is not lost — the LOD invariant
hands it to `submesh_mss` and the BSDF — but it moves a large amount of glint
from mesh geometry into sub-facet roughness, which visibly changes the render
for no reason anyone asked for.

**Rule: `k_max` is a rendering decision (driven by `mesh_dx` and the desired
geometry/BSDF split). λ_p has no business setting it.** The last tile's size and
`n` stay under user control; only the tiles below it are derived.

---

## 4. Why it is cheap: `n` never grows

The instinct is that bigger seas need bigger grids, so a sweep to high windspeed
explodes the FFT budget. That is not how these configs work.

`n` is held at 512/256/256 in *all* of them, and only `L` moves. Tile 1's
Nyquist — which sets `k_ref` and therefore the band edges — scales down with the
sea, and that is exactly what makes the edges track k_p (§3).

`k_max` is a different quantity and is deliberately **not** allowed to follow
it (§3a). Whatever sits above `k_max` is handed to `submesh_mss` and reaches the
BSDF as sub-facet roughness, which is the whole content of the LOD invariant:

```
mss_resolved(dx) + mss_above(π/dx) = mss_total
```

So the split is always physically accounted for. But "accounted for" is not
"free" visually — moving the boundary shifts glint between mesh geometry and
BSDF roughness, which is why §3a pins it rather than letting it drift with the
sea state.

Auto-sizing costs **zero additional FFT budget**: `n` is untouched, only `L`
moves.

---

## 5. Why this is not a re-baseline

Two pieces of existing evidence, both already in the suite:

**Aggregate quantities are invariant to resizing — provided §3a is respected.**
[`test_resizing_tiles_redistributes_energy_without_changing_it`](../tests/test_surface.py#L537)
already proves that moving the band edges leaves `Hs`, resolved `mss` and
`k_max` fixed to 1e-3. Its own summary: *"A resize is a redistribution, not a
different sea."*

Note that this test holds `k_max` fixed **by construction** — the `straits_crop`
tile set it resizes never moves the third tile. That is not a weakness in the
test; it is the invariant of §3a, encoded. Reproduced independently on `straits`
in the §3a table: pin tile 3 and `Hs` moves 9.4e-04 with `mss` at 1.6e-03; scale
it and `mss` moves 0.36. **A derivation that resizes the top tile would break
this test, and the test would be right.**

**The stored baselines cannot drift.**
[`make_baseline.py:59`](../tests/make_baseline.py#L59) constructs its config in
code, with the comment *"so `configs/` cannot drift into it"*, and pins explicit
`TileConfig` sizes. Auto-sizing is opt-in per config and never reaches it, so
`tests/baseline/*.npy` are untouched.

What *does* change: any config that switches from explicit tiles to derived
tiles gets a different **realisation** — a different arrangement of the same
sea. Renders change. Numbers do not. That distinction should be stated in the
release notes, because "the pictures changed but the validation report is
identical" will otherwise read as a bug.

---

## 6. The work, in order

Each step is independently shippable and leaves the suite green. Steps 1–2 are
worth doing even if the rest is abandoned.

### Step 1 — close the test gap *(small, do first, own branch)*

Make `test_every_shipped_scene_has_tiles_sized_for_its_own_sea` discover
`configs/*.yaml` by glob instead of a hardcoded triple. It will immediately fail
on `straits`, which is correct — that is the bug it was written to catch.

Then fix `straits.yaml` by hand to **`{380, 172, 23} @ 512/256/256`** — tiles 1
and 2 at 44.8/20.4 × 8.48 m, third tile left at 23 m per §3a. Verified: first
edge 2.00 k_p, shares [0.83, 0.13, 0.05], no notes, `k_max` unchanged at 34.97,
`Hs` shift 9.4e-04.

*Gate: the suite is green and the assertion now covers four configs, not three.*

### Step 2 — persist the sizing record

`scene_summary()` already calls `ts.sizing()`
([run_scene.py:101](../scripts/run_scene.py#L101)) and then discards most of it.
Write the full `TileSizing` — `lambda_p`, `first_edge_over_k_p`, `band_shares`,
`largest_tile_over_lambda_p`, `notes` — into `summary.json`, and surface
`first_edge_over_k_p` and `bands_are_inert` as columns in
`collect_run_info.py`.

Rationale: a sweep is only auditable after the fact if each run recorded what it
actually used. This is what makes the original symptom diagnosable from the run
directory instead of from memory.

*Gate: a run directory answers "were this run's bands live?" without rebuilding
the tileset.*

### Step 3 — the derivation

Add `derive_tile_sizes(lambda_p, n, fractions, target_edge_over_kp=2.0)` to
`tiling.py`, returning the sizes from §3. Pure function, no config coupling,
directly unit-testable across a wide λ_p range.

**It returns sizes for all tiles but the last.** Per §3a the top tile is a
rendering decision and stays as configured. The signature should make that
impossible to get wrong — take the leading tiles' `n` and return that many
sizes, rather than accepting all of them and quietly ignoring one.

Constraints it must satisfy, all already enforced downstream by `band_edges`:
every band fits inside its tile's `[k_min, k_nyquist]`. The derivation proposes;
`band_edges` validates. Do not duplicate the validation.

**Pick the ratios to avoid rational relationships.** `(45, 20, 10)` would be a
mistake — `20/10 = 2` exactly, and commensurate tile sizes reintroduce a
composite period, which is the one thing §0 says this must not break. Use
`(44.8, 20.4, 10.4)`: ratios 2.20 and 1.96.

*Gate: for λ_p from 0.5 m to 100 m, the derived set puts the first edge at
2.0 k_p ± 2%, band 1 holds < 0.90, `band_edges` raises for none of them, and
`k_max` is identical to the pinned-tile-only configuration at every λ_p.*

### Step 4 — wire it into config

Make `tiles:` optional in the scene YAML. When absent, derive from the config's
own `lambda_p` using a new optional `surface.tile_n: [512, 256, 256]` and
`surface.target_edge_over_kp: 2.0`. When present, use it verbatim and unchanged.

Derivation-with-override, not replacement. Every existing config keeps working
byte-identically, which is what keeps §5 true.

*Gate: all five shipped configs produce identical tilesets to today when they
pin tiles; a config with `tiles:` deleted produces a set whose `sizing()` has no
notes.*

### Step 5 — make inert bands loud

Promote `bands_are_inert` from a note to a warning at scene build, and to a hard
error under a `--strict` flag on `run_scene.py`.

*Gate: a deliberately mis-sized config warns on stdout and fails under
`--strict`.*

### Step 6 — sweep ergonomics

Only after 1–5. Whatever the sweep driver needs to vary sea state without
touching tiles at all — most likely just documenting that sweep configs should
omit `tiles:`.

---

## 7. Open questions

**Should `straits.yaml` still exist?** `straits_crop.yaml` is the maintained
one; `straits.yaml` looks like an unmaintained predecessor, which is consistent
with it being the only config left mis-sized and the only one omitted from the
test. If it is dead, deleting it is a better step 1 than resizing it.

**Seed policy across a sweep.** Fixed seed + fixed tile geometry means the wave
*pattern* is correlated run-to-run and only amplitudes change. For a controlled
sweep that isolates the variable, which may be exactly right. For a set of
independent-looking sea states it is wrong. This wants a decision, not a
default — and note that auto-sizing changes the geometry per sea state, so it
partially decorrelates the sweep as a side effect whether or not that was
wanted.

**Does `mesh_dx` belong in the derivation?** §3a says `k_max` is a rendering
decision, which raises the obvious follow-on: it is currently set indirectly, by
choosing `L₃` and `n₃` by hand. `band_limited(k_cut)` already exists to cut the
spectrum at a mesh's Nyquist. Whether the top tile should instead be derived
*from* `output.mesh_dx` is a real design question, but it is a different change
from this one and should not be bundled.

*(An earlier open question here asked whether the third tile's size mattered.
§3a answers it: it sets `k_max` and must be pinned. Resolved, not dropped.)*
