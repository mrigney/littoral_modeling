# littoral_modeling — `pywave`

Spectral water surface synthesis for physics-based EO/IR littoral scene generation.

`pywave` builds a time-evolving water surface from a wind-wave spectrum and reports
the statistics of that surface — height variance, slope variance, directional
anisotropy — as physically derived numbers rather than art-directed knobs.

> **The governing rule.** The spectrum is the single source of truth. Wave height,
> slope statistics, BSDF roughness and LOD behaviour are all derived from one
> `S(k)`. Nothing is tuned independently; if two things disagree, the spectrum wins
> and the other one has a bug.

## Status

| Phase | Deliverable | Status |
|---|---|---|
| 1 | JONSWAP spectrum, spreading, moments | **Implemented** |
| 2 | Tessendorf FFT synthesis, multi-tile composition | **Implemented** |
| 3 | Validation suite, `docs/validation_report.md` | **Implemented** |
| 4 | Houdini terrain and lake basin | **Loader implemented** — `Bathymetry.from_export` reads a §4.5 export; the Houdini side is yours |
| 5 | Shoaling, refraction, breaking, foam | **Implemented** against that synthetic profile |
| 6 | Mesh generation, channel packing, PLY export | **Implemented** — constant post spacing; §6.2 LOD rings deliberately deferred |
| 5b | Refraction on a complex coastline | **Implemented**, selectable as `refraction: rays`, not the default — [docs/phase5b_refraction.md](docs/phase5b_refraction.md) |
| 7–10 | Mitsuba BSDF, emissivity, EMBER integration | Not started |
| 11 | Boat wake, turbulent wake, bow spray | **Architected**, not built — [docs/phase11_wake_spray.md](docs/phase11_wake_spray.md) |
| 12 | Anchored objects in the water | **Architected**, mostly not needed — [docs/phase12_anchored_props.md](docs/phase12_anchored_props.md) |

**New here?** [docs/gallery.md](docs/gallery.md) explains the whole model in
eight figures, with no code required.

## Use real terrain

Point a scene at a Phase 4 export and everything downstream follows — no code
changes, because the synthetic bathymetry was built to satisfy the same contract:

```yaml
scene:
  water_level: 100.0          # must equal grid_meta.json's z_w -- checked
  epsg: 32616                 # likewise
bathymetry:
  source: houdini_export      # the six §4.5 files; other keys then ignored
```

```bash
python scripts/run_scene.py configs/houdini_lake.yaml --mesh
```

The loader checks the export against the same assertion set the synthetic basin
is held to — `depth == z_w − terrain_z`, `sign(sdf) == −sign(depth)`, unit
inland-pointing normals — and refuses rather than proceeding on data that would
produce a plausible-looking wrong answer. `shore_normal` is accepted as either
`(2, ny, nx)` or `(ny, nx, 2)`.

Real terrain has no analytic profile, so `beach_slope()` **measures the bed**
instead of evaluating Dean's formula, and everything downstream (breaker type,
runup, swash) follows from the measurement. Exports are gitignored by default;
`git add -f` one deliberately if you want the Gate 4 checks to run in CI.

## Run a scene

Write a scene file, run one command, look at what comes out:

```bash
cp configs/test_lake.yaml configs/my_scene.yaml     # edit wind, fetch, beach…
python scripts/run_scene.py configs/my_scene.yaml
```

Everything lands in `runs/my_scene/`:

| Output | What it is |
|---|---|
| `summary.md` | Scene report — sea state, tiles, LOD budget, surf zone. **Start here.** |
| `overview.html` | One self-contained page. Open in a browser, or mail it to someone. |
| `gallery.md` | The eight figures with captions; renders on GitHub. |
| `figures/*.png` | The figures on their own, 150 dpi. |
| `channels/*.npy` | Per-cell nearshore fields + `manifest.json` — what Phase 6 will consume. |
| `summary.json` | Every number in `summary.md`, machine-readable. |
| `open.mp4`, `shore.mp4` | Animations, with `--animate`. |
| `mesh/water_0000.ply` | Displaced mesh + per-vertex channels, with `--mesh`. |

Takes about 40 s for a small lake, a couple of minutes for a large coastal scene.
Add `--quick` to skip the channel export and the HTML page.

Two example scenes ship, and they are deliberately unlike each other —
[`test_lake.yaml`](configs/test_lake.yaml) (5 m/s, 1 km fetch → 8.6 cm waves,
spilling breakers, 0.4 m swash) and [`coastal_bay.yaml`](configs/coastal_bay.yaml)
(12 m/s, 40 km fetch, embayed shore → 1.43 m waves, plunging breakers, 7 m
swash). Every figure, number and channel is derived from the config, so a
disagreement with the validation report means the code changed — not that the
picture went stale.

## Export a mesh

```bash
python scripts/run_scene.py configs/my_scene.yaml --mesh
python scripts/run_scene.py configs/my_scene.yaml --mesh \
       --mesh-dx 0.25 --mesh-region 400 380 500 405 --mesh-t 12.5

# the whole water surface, for renders where the camera is not known in advance.
# Only wet posts are meshed, so this is every square metre of water and no
# triangles over dry land. `output.mesh_full: true` makes it the default.
python scripts/run_scene.py configs/my_scene.yaml --mesh --mesh-full

# ... with the bed straight out of the tool that authored the terrain. This
# substitutes for terrain_0000.ply only -- `bathymetry.source` still has to
# point at the .npy export, which is what the model actually runs on.
python scripts/run_scene.py configs/my_scene.yaml --mesh --mesh-full \
       --terrain-ply /path/from/houdini/terrain.ply
python scripts/check_clearance.py runs/my_scene/mesh/water_0000.ply \
       /path/from/houdini/terrain.ply
```

The water surface's height limit is computed from the `.npy` fields, not from
whatever bed mesh you render. Keep the fields at least as fine as that mesh, or
the bed shows through at the waterline — on the shipped lake, fields 2× coarser
than the bed put 0.24% of water vertices as much as 70 cm under it. Guide §6 has
the numbers; `check_clearance.py` verifies any pair.

Writes **three things** into `runs/<scene>/mesh/`:

| File | What it is |
|---|---|
| `water_0000.ply` | Displaced water surface + per-vertex channels (`mss`, `wdir_x/y`, `aniso`, `depth`, `foam`, `wetness`) |
| `terrain_0000.ply` | The bed, co-registered with the water and slightly larger |
| `scene.py` | Starter Mitsuba scene **dict** for `mi.load_dict`, camera placed from the mesh bounds |
| `scene.xml` | The same scene as XML, for the `mitsuba` CLI |

Plus a JSON sidecar per mesh with the seed, wind, time and git sha that produced
it. A water mesh on its own is not renderable — nothing to sit on, nothing to
occlude it at the shoreline — so the terrain comes as standard.

`scene.py` is the primary form — it hands `mi.load_dict` a dictionary and
resolves the PLY paths against its own location, so it works from any directory:

```python
from scene import scene_dict
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
img = mi.render(mi.load_dict(scene_dict()), spp=256)
```

```bash
python scene.py --variant cuda_ad_rgb --spp 256 -o test.exr   # or run it directly
```

**Both are untested against Mitsuba**, which is a Phase 7 dependency and is not
installed here. They are generated from one set of parameters so they cannot
drift apart, and the suite checks the dict's structure and the XML's
well-formedness — but not that Mitsuba likes either. Expect a nudge.

The BSDFs are placeholders (diffuse sand; `roughdielectric` with Beckmann
`alpha = sqrt(mean(mss))` baked to a **constant**) because stock BSDFs cannot
read mesh attributes. The channels are in the PLY and unused — consuming them is
what the Phase 7 plugin is for.

**PLY, not OBJ.** OBJ has positions, normals and texture coordinates and no
mechanism for arbitrary per-vertex scalars, so an OBJ carries none of the
channels — which is everything that makes the mesh worth generating rather than
just a displaced plane. Mitsuba's PLY loader passes unknown properties through
as `mesh_attribute` textures, and that is the delivery path into Phase 7.
`--mesh-obj` writes one anyway for viewing in MeshLab or Blender; it says in its
own header what it dropped.

**Post spacing enters quadratically.** The shipped lake at its configured
0.125 m spacing is 8000 × 8000 = 64 M posts over the full 1 km domain, several
GB for one frame. So either coarsen `--mesh-dx` or bound `--mesh-region`;
bounding the region is what stands in for LOD rings until §6.2 is built. The
default is a shoreline window of about 900 posts a side.

| Flag | Default | What it does |
|---|---|---|
| `--mesh-dx D` | `output.mesh_dx` | Post spacing, metres |
| `--mesh-region X0 Y0 X1 Y1` | shoreline window | Bound the mesh, scene coordinates |
| `--mesh-t T` | `0.0` | Scenario time of the frame — free, the surface is a pure function of `t` |
| `--mesh-obj` | off | Also write an OBJ (no channels) |
| `--mesh-max-vertices N` | 12,000,000 | Raise the guard on a big machine |

Roughly 50,000 water vertices/s, single-threaded — 10 M vertices is about three
minutes and 780 MB. More cores will not help (no parallelism in `pywave`) and
nor will a GPU (no GPU code); a fast card matters for Mitsuba, not for this.
Full guidance, including a troubleshooting table, is in
[users_guide.md §16.8](docs/users_guide.md#6-meshes-and-export).

## Watch it move

```bash
python scripts/animate.py                          # both views, test lake
python scripts/animate.py --mode shore             # just the shoreline
python scripts/animate.py configs/coastal_bay.yaml --seconds 8
```

Two views: **open** looks straight down at open water, and **shore** is a plan
view across the waterline showing shoaling waves, the breaker line, foam in the
surf band and the swash edge breathing at the peak period over wet sand.

Writes MP4 when ffmpeg is available and animated GIF otherwise. GIF works out of
the box but wave texture compresses badly in 256 colours — expect several MB.
For files about a tenth the size, and playable anywhere:

```bash
pip install -e ".[video]"      # bundles an ffmpeg binary, no system install
```

Because the surface is a pure function of time, `--start 3600` renders an hour
into the scenario and costs exactly what `--start 0` costs.

Individual pieces, if you want just one:

```bash
python scripts/make_figures.py  --config configs/my_scene.yaml --out /tmp/figs
python scripts/make_overview.py --config configs/my_scene.yaml --out /tmp/page.html
```

## Install

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

Python 3.11+. Runtime deps: `numpy`, `scipy`, `pyyaml`.

## Validate

```bash
pytest                    # 119 checks, ~2 min
```

The suite is the traceability argument, not CI hygiene. Every test records the
number it measured and the reference it was judged against, and the run
regenerates [docs/validation_report.md](docs/validation_report.md) with a
`git_sha` header — that file is the V&V artifact. Two cookbook gate criteria are
provably unmeetable and carry substituted criteria; both are documented under
*Gate deviations* in the report.

## Quickstart

```python
import numpy as np
from pywave import load_config, spectrum, moments, tiling

cfg = load_config("configs/test_lake.yaml")

print(f"Tp     {1 / cfg.f_p:.2f} s")
print(f"Hs     {spectrum.hs_spectral(cfg.wind.speed, cfg.wind.fetch):.4f} m")
print(f"mss    {moments.mss_above(0.0, cfg.wind.speed, cfg.wind.fetch):.4f}")

tiles = tiling.TileSet.build(cfg)
field = tiles.sample(np.linspace(0, 50, 256), np.zeros(256), t=2.5)
print(f"h      {field.h.min():+.3f} .. {field.h.max():+.3f} m")
```

The surface is a pure function of `t` — no accumulated state, exactly reproducible
from the config seed on any machine, and any frame is computable without computing
the ones before it.

## Documentation

| Document | Answers |
|---|---|
| **[docs/users_guide.md](docs/users_guide.md)** | How do I run it? Config reference, CLI, API, recipes, troubleshooting. |
| **[docs/algorithms.md](docs/algorithms.md)** | How does it work, and why is it right? Physics, derivations, what is exact and what is approximated. |
| **[docs/validation_report.md](docs/validation_report.md)** | What did it measure on this commit? Generated by `pytest`. |
| **[docs/gallery.md](docs/gallery.md)** | The whole model in eight figures. |
Background: `water-surface-modeling-primer.md` (physics rationale) and
`littoral-water-implementation-cookbook.md` (phase plan and gate checklists).

Before writing code against this package, read the conventions table in
[`pywave/constants.py`](pywave/constants.py). Coordinate handedness, angle
direction, FFT normalisation and the two distinct quantities both called `m2` are
all fixed there, and violating any of them silently is the most common source of
multi-day debugging.

## Layout

```
pywave/
  constants.py    conventions, physical constants, water optical constants
  config.py       scene dataclasses + YAML load (the only place degrees exist)
  spectrum.py     PHASE 1: JONSWAP, spreading, dispersion, S(f) -> S(k)
  moments.py      PHASE 1: m0, mss_above(k), Tz, Cox-Munk cross-check
  surface.py      PHASE 2: Tessendorf FFT synthesis
  tiling.py       PHASE 2: multi-tile composition
  bathymetry.py   PHASE 5: synthetic Dean beach (stands in for the Phase 4 export)
  nearshore.py    PHASE 5: shoaling, Snell refraction, breaking, wetness
  foam.py         PHASE 5: foam with bounded, reproducible spin-up
  mesh.py         PHASE 6: water extent, triangulation, displacement
  channels.py     PHASE 6: per-vertex channels (the section 0.4 contract)
  export.py       PHASE 6: binary PLY, OBJ, provenance sidecar
tests/
  test_spectrum.py        PHASE 3: Gate 1 checks
  test_surface.py         PHASE 3: Gate 2 checks
  test_reproducibility.py PHASE 3: Gate 3, determinism + regression baseline
  test_terrain_export.py  PHASE 4: Gate 4, loading a terrain export
  test_nearshore.py       PHASE 5: Gate 5 checks
  test_mesh.py            PHASE 6: Gate 6 checks
  make_baseline.py        regenerates tests/baseline/ (a deliberate act)
scripts/
  run_scene.py       config -> every artifact, in one directory
  animate.py         open-water and shoreline clips
  make_figures.py    the eight figures + a markdown gallery
  make_overview.py   one self-contained shareable HTML page
  check_clearance.py does the water stay above a given bed mesh?
  collect_run_info.py harvests runs/ into one markdown table
configs/
  test_lake.yaml     reference scene: 5 m/s wind, 1 km fetch, straight beach
  coastal_bay.yaml   contrasting scene: 12 m/s, 40 km fetch, embayed shore
  houdini_lake.yaml  runs on a real exported terrain (`bathymetry.source`)
docs/
  users_guide.md        how to run it
  algorithms.md         how it works, and why it is right
  validation_report.md  generated by pytest -- the V&V artifact
  gallery.md            the model in eight figures
```

## Reference scene

`configs/test_lake.yaml` — U10 = 5 m/s, fetch = 1000 m, deep water:

```
Tp     1.045 s        lambda_p  1.705 m
Hs     0.0857 m       mss       0.0461  (RMS slope 12.1 deg)
```

The three-tile composite realises Hs = 0.0839 m (−1.6%) and closes the LOD
invariant `mss_resolved + mss_above(k_max) = mss_total` to 0.01%. Full numbers in
the [user's guide](docs/users_guide.md#10-reference-numbers).
