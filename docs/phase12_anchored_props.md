# Phase 12 — anchored objects in the water

**Status: architecture only. Nothing here is built.**

Placing a static, surface-piercing object — a cube on the bed, a piling, a
navigation mark, a moored barge — in a littoral scene, and deciding what, if
anything, the wave field should do about it.

Not urgent. Written down because the answer turned out to be smaller than
expected, and because most of it already works.

---

## 0. What this is not

**Not Phase 11.** A boat has a Kelvin wake because it is *moving*: the whole of
[phase11_wake_spray.md](phase11_wake_spray.md) hangs off a Froude number, and an
anchored object has no speed to form one from. There is no wake, no bow spray,
no turbulent trail. The two phases share nothing but the word "object" and
should not be bundled.

The one steady-flow effect an anchored object would have is a **current** wake,
and this package has no currents at all — wave–current interaction is in the
out-of-scope list of [phase5b_refraction.md](phase5b_refraction.md) and stays
there.

---

## 1. The question that decides the whole phase

Whether the water has to react at all. That is set by `ka`, with
`k = 2π/λ_p` and `a` the object's half-width:

- `ka ≪ 1` — the wave diffracts around the object and carries on. There is no
  shadow, no reflection worth drawing. **Doing nothing is correct.**
- `ka ≫ 1` — geometric shadowing is real and ray theory describes it.

Measured against the shipped scenes:

| scene | λ_p | 0.5 m | 2 m | 5 m | 20 m | 50 m |
|---|---|---|---|---|---|---|
| `test_lake` | 1.7 m | 0.92 | 3.68 | 9.21 | 36.8 | 92.1 |
| `coastal_bay` | 35.3 m | 0.04 | **0.18** | 0.45 | 1.78 | 4.45 |
| `straits_crop` | 13.6 m | 0.12 | **0.46** | 1.16 | 4.62 | 11.6 |

*(columns are full object width; `ka` on the half-width)*

**A 2 m cube in `coastal_bay`'s 35 m swell scores 0.18. The sea does not see
it.** That is not an approximation being tolerated — it is what happens. Swell
goes round pilings. Anything up to about 5 m in a realistic littoral sea state
can be placed with **no change to the wave model whatsoever**, and a phase that
starts by building wave–structure interaction has solved a problem the scenes do
not have.

The threshold to care is roughly `ka > 1`, so **~10 m in a 35 m sea, ~5 m in a
13.6 m sea.** Breakwaters, piers and moored barges qualify. Cubes and marks do
not.

---

## 2. Three layers, and you can stop after any of them

| | what it buys | cost |
|---|---|---|
| **L1 placement** | the object is in the render, seated on the bed | small |
| **L2 water aperture** | the water surface stops passing through it; foam collar | moderate |
| **L3 field reaction** | shadow and sheltering behind it | only meaningful at `ka > 1` |

### L1 — placement

`export.mitsuba_scene_dict` is **already** the pattern used for external assets
in land scenes: a flat dict of named shapes, each an external file plus a
material.

```python
"terrain": {"type": "ply", "filename": ..., "bsdf": {...}},
"water":   {"type": "ply", "filename": ..., "bsdf": {...}},
```

A third entry with a `to_world` is a few lines. Seating is arithmetic — the bed
elevation under a point is `z_w − depth`, and `Bathymetry.sample` returns the
depth. Proposed scene-config block:

```yaml
props:
  - mesh: models/cube.ply
    at: [1250.0, 830.0]     # world x, y
    seat: bed               # bed | waterline | fixed
    z: 0.0                  # offset from the seat
    yaw: 30.0
```

Nothing in the physics changes. This is the 90% case.

### L2 — the water aperture

Without it the water surface interpenetrates the object. In a ray tracer that
is often acceptable: the object is opaque, the water is a dielectric, and the
intersection reads as a waterline. What is missing is the wetness band and foam
collar, and up close the interpenetration shows.

The mechanism already exists. [mesh.py:48](../pywave/mesh.py#L48)
`water_extent_mask` builds the water mesh from `depth > trim_depth`; excluding a
footprint polygon is the same operation on the same mask.

### L3 — the field reaction

**To the wave field, a surface-piercing object is an island.** Burn the
footprint into the bathymetry as land (`depth ≤ 0`) and Phase 5b handles it
unchanged:

- `trace_rays` retires rays at `break_depth`, giving the shadow;
- `shelter_fetch` reads `depth ≤ 0` as land, so the lee fills with its own
  short-fetch wind sea instead of going to zero;
- refraction around the flanks falls out of the ray integration;
- `breaking_mask` fires on the shallow footprint, so a **foam collar appears for
  free**.

The caveat is the one ray theory always has: no diffraction, so the shadow is
too sharp. Trustworthy at `ka ≳ 3`; between 1 and 3 it will over-shelter.

---

## 3. Baking it into the Houdini export instead

The obvious shortcut — place the object in Houdini, merge it with the terrain,
export one mesh — is better than it first appears, but only in a specific
combination.

**What baking into the *fields* gives you.** The Phase 4 export is a set of
heightfield rasters, and a cube is representable as one (no overhangs). Its
footprint becomes `terrain_z` above `z_w`, hence `depth ≤ 0`, hence land. So
`water_extent_mask` cuts a hole in the water automatically — **L2 for free** —
`sdf` and `shore_normal` are rebuilt around it, breaking fires on its edge, and
the ray solver treats it as an island. That is L2 and L3 with no code at all.

**What baking costs you.** The object's geometry is degraded to the raster. A
2 m cube on a 4 m grid does not exist; on 0.25 m posts it is eight cells across
with staircase edges, and the terrain mesher joins adjacent posts, so the
vertical faces come out as one-cell ramps rather than walls. For background
clutter that is fine. For anything the camera looks at, it is not.

**The combination that works, and works today.** Both `run_scene.py` and
`animate_frames.py` already accept `--terrain-ply`, which substitutes an
externally authored bed mesh while the `.npy` fields continue to drive the
physics. So:

1. In Houdini, place the object and merge it with the terrain.
2. Export the heightfield rasters **including** the object — these drive the
   water aperture, the foam, and the wave field.
3. Export the merged mesh at full geometric fidelity as `terrain.ply`.
4. `run_scene.py <cfg> --terrain-ply terrain.ply`.
5. `scripts/check_clearance.py` to confirm the two agree.

That gives crisp geometry to the renderer and a consistent object to the
physics, with no new code. **It is the recommended route until a scene needs
something it cannot express.**

Its limits are worth stating: the object is welded to the bed, so it cannot move
between frames without re-exporting, cannot be instanced, and cannot carry its
own material distinct from the terrain BSDF. L1's `props` block exists to lift
exactly those three restrictions, which is why it is still worth building
eventually — but not first.

Note also step 2 is a **choice**: exporting the rasters *without* the object
keeps the fields pristine and leaves the water surface passing through it. That
is the right call when `ka ≪ 1` and the object is small, because burning a 2 m
cube into the bathymetry tells the ray solver to shadow something the sea would
have ignored.

---

## 4. Gate 12

| # | Check | Criterion |
|---|---|---|
| 12.1 | **Seating.** Object base against the bed elevation sampled at its position | within one bathymetry post |
| 12.2 | **Co-registration.** Prop, water and terrain in one frame; run `check_clearance.py` | no water vertex below the bed |
| 12.3 | **Small objects do not perturb the sea.** `Hs` field with and without a prop at `ka < 0.5` | identical to 1e-12 |
| 12.4 | **Large objects do.** Mean gain in the lee of a `ka > 3` obstacle vs its flanks | ratio < 0.7 |
| 12.5 | **The aperture is watertight.** No water vertex inside the footprint | zero |
| 12.6 | **Reproducibility.** Same scene twice | bitwise identical |

12.3 is the one worth writing first. It is the check that stops the phase
quietly acquiring wave–structure interaction that none of the target scenes
need, and it is the same shape as 5b.6 — an assertion that a quantity is
structurally absent rather than merely small.

### Deliberately out of scope

Wave reflection off the object, radiation from any motion, current-induced
wakes, scour around the base, and the object's own dynamics. An anchored object
that responds to waves is a moored-body problem and a different discipline.

---

## 5. Suggested order

1. Bake-and-substitute via `--terrain-ply`. **Already works; no code.**
2. Gate 12.3 as a test, against a small prop. Pins the scope.
3. L1 `props` block: config schema, seating, scene-dict entry. 12.1, 12.2.
4. L2 footprint aperture in `water_extent_mask`. 12.5.
5. L3 `obstacle: true`, burning the footprint into a working copy of the bed,
   gated on a computed `ka` with a warning below 1. 12.4.

Step 1 covers the cases in front of us today. Steps 3–5 are what turn it from a
Houdini workflow into a scene-file feature, and there is no urgency to them.
