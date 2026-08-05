# pywave — User's Guide

Spectral water surface synthesis for physics-based EO/IR littoral scene generation.

This guide is **how to use it**. For *why it works the way it does* — the physics,
the derivations, what is exact and what is approximated — see
[algorithms.md](algorithms.md). For *what it measured on this commit*, see
[validation_report.md](validation_report.md).

> **In a hurry?** [gallery.md](gallery.md) explains the whole model in eight
> figures, no code required.

---

## Contents

1. [What this package is](#1-what-this-package-is)
2. [Installation](#2-installation)
3. [Run a scene](#3-run-a-scene)
4. [The config file](#4-the-config-file)
5. [Using real terrain](#5-using-real-terrain)
6. [Meshes and export](#6-meshes-and-export)
7. [Animations](#7-animations)
8. [Using the API](#8-using-the-api)
9. [Module reference](#9-module-reference)
10. [Reference numbers](#10-reference-numbers)
11. [Validating a build](#11-validating-a-build)
12. [Conventions](#12-conventions)
13. [Troubleshooting](#13-troubleshooting)
14. [Roadmap](#14-roadmap)

---

## 1. What this package is

`pywave` builds a **time-evolving water surface** from a wind-wave spectrum and
reports the **statistics of that surface** — height variance, slope variance,
directional anisotropy — as physically derived numbers rather than art-directed
knobs.

> **The governing rule.** The spectrum is the single source of truth. If two
> quantities disagree, the spectrum wins and the other one has a bug.

There are no artistic parameters. The one number that looks like a knob,
`surface.choppiness`, has a physical value of 1.0.

### What you get

| Capability | Entry point |
|---|---|
| JONSWAP spectrum, directional spreading, dispersion | `spectrum` |
| Height/slope moments, sub-grid slope variance, `Tz` | `moments` |
| A surface `h(x, y, t)` with exact analytic slopes | `surface.WaveTile` |
| Multi-tile composite that hides FFT periodicity | `tiling.TileSet` |
| Terrain: loaded export, or a synthetic Dean beach | `bathymetry.Bathymetry` |
| Shoaling, Snell refraction, depth-limited breaking | `nearshore.transform` |
| Swash wetness as a duty cycle, for the thermal channel | `nearshore.wetness_fraction` |
| Foam with bounded, reproducible spin-up | `foam.FoamModel` |
| Displaced mesh with per-vertex channels | `mesh.build_water_mesh` |
| Binary PLY that survives into Mitsuba | `export.write_ply` |

### What it does not do yet

No LOD rings (§6.2 of the cookbook), no BSDF, no emissivity, no EMBER
integration. See [Roadmap](#14-roadmap).

---

## 2. Installation

Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

Runtime deps: `numpy`, `scipy`, `pyyaml`. The `dev` extra adds `pytest`,
`matplotlib` and `pillow`.

```bash
pip install -e ".[video]"       # optional: bundles ffmpeg for MP4 animations
```

Verify:

```bash
python -c "import pywave; print(pywave.__version__)"
pytest                          # ~2 min
```

---

## 3. Run a scene

One command turns a scene file into every artifact the model can produce.

```bash
python scripts/run_scene.py                          # the shipped test lake
python scripts/run_scene.py configs/my_scene.yaml
python scripts/run_scene.py configs/my_scene.yaml --mesh --animate
```

Output lands in `runs/<scene>/`:

| File | What it is |
|---|---|
| `summary.md` | Scene report — sea state, tiles, LOD budget, surf zone. **Start here.** |
| `summary.json` | The same numbers, machine-readable |
| `gallery.md` | The eight figures with captions; renders on GitHub |
| `overview.html` | One self-contained page; mail it to someone |
| `figures/*.png` | The figures on their own, 150 dpi |
| `channels/*.npy` | Per-cell nearshore fields + `manifest.json` |
| `mesh/` | Water + terrain PLYs, Mitsuba scene (`--mesh`) |
| `open.mp4`, `shore.mp4` | Animations (`--animate`) |

| Flag | Effect |
|---|---|
| `--quick` | Skip the channel export and the HTML page |
| `--mesh` | Build and export the meshes |
| `--animate` | Render the two clips (adds a few minutes) |
| `--out DIR` | Write somewhere other than `runs/<scene>/` |

Roughly 40 s for a small lake, a couple of minutes for a large coastal scene.

### The shipped scenes

Three, deliberately unlike each other:

| | `test_lake` | `coastal_bay` | `houdini_lake` |
|---|---|---|---|
| Terrain | synthetic, straight | synthetic, embayed | **loaded export** |
| Wind / fetch | 5 m/s, 1 km | 12 m/s, 40 km | 5 m/s, 1 km |
| Hs | 0.086 m | 1.43 m | 0.085 m |
| Tp | 1.05 s | 4.75 s | 1.05 s |
| Foreshore | 6.7% | 12.3% | 31.5% |
| Breakers | spilling | plunging | plunging |
| Swash | 0.38 m | 7.1 m | 0.38 m |

Running two and diffing their `summary.md` is the fastest way to see which
quantities are scene-dependent and which are structural.

---

## 4. The config file

A scene is one YAML file. Copy `configs/test_lake.yaml` — every key is commented.

```yaml
scene:
  domain: [1000.0, 1000.0]      # m, extent in X and Y
  water_level: 100.0            # m, z_w
  epsg: 32616                   # CRS of the scene (metadata only)

wind:
  speed: 5.0                    # m/s at 10 m reference (U10)
  direction: 45.0               # deg CCW from +X, blowing TOWARD
  fetch: 1000.0                 # m

spectrum:
  model: jonswap                # only 'jonswap' is implemented
  gamma: 3.3                    # peak enhancement; 1.0 = Pierson-Moskowitz
  spreading: cos2s              # 'cos2s' | 'hasselmann'  ('donelan' raises)
  seed: 20260801                # the one seed for the whole scene

surface:
  tiles:                        # incommensurate sizes, disjoint bands
    - {size: 64.0, n: 512, band: [0.0, 0.35]}
    - {size: 37.0, n: 256, band: [0.35, 0.7]}
    - {size: 23.0, n: 256, band: [0.7, 1.0]}
  choppiness: 1.0               # physical value; do not tune

bathymetry:
  source: null                  # path to a terrain export; see section 5
  profile: planar               # synthetic only: planar | embayment
  shoreline: 400.0              # m, y of the waterline
  dean_a: 0.100                 # m^(1/3); omit to derive from grain_size
  grain_size: 0.25              # mm D50, ignored when dean_a is given
  max_depth: 5.0
  bank_slope: 0.08
  dx: 1.0                       # m, field grid spacing
  surf_dx: 0.25                 # m, refined spacing through the surf zone
  amplitude: 40.0               # m, embayment only
  wavelength: 400.0             # m, embayment only

nearshore:
  breaker_index: 0.78           # gamma_b
  foam_halflife: 3.0            # s
  foam_coverage: 0.85           # coverage a continuously breaking cell reaches
  refraction: true
  shoaling: true

output:
  fps: 30.0
  mesh_dx: 0.125                # m, water mesh post spacing
  lod_rings: [...]              # parsed; LOD rings not yet implemented
```

### Keys that need care

**`surface.tiles`** — the two ways to get this wrong:

- **Tiles must resolve the peak.** A tile's Nyquist is `π·n/size`, and at least
  one tile needs a Nyquist comfortably above `k_p = 2π/λ_p`. Doubling the wind
  roughly quadruples `λ_p`, so a tile set tuned for chop will not do for swell.
  Asserted at run time and in the test suite.
- **Keep sizes incommensurate.** 64/37/23, not 64/32/16. Sizes in simple ratios
  re-align their lattices and the periodicity the construction exists to hide
  comes straight back.

`band` values are *fractions*, not wavenumbers. They must be contiguous,
disjoint, and span exactly `[0, 1]` — a gap silently loses variance, so it is
rejected at load.

**`bathymetry.dx` / `surf_dx`** — `surf_dx` must be the finer of the two. It
exists because the surf zone can be a metre wide, which a 1 m grid cannot resolve
at all.

**`nearshore.foam_coverage`** — the coverage a continuously breaking cell settles
at. The seeding rate is *derived* from this and the half life, so the equilibrium
stays put whatever the half life is. Keep it below 1.0 so there is headroom
between the outer surf and the inner swash.

**`spectrum.seed`** — one integer for the whole scene. Per-tile streams are
spawned from it, so the surface is exactly reproducible on any machine.

Everything is validated on load: a mistake fails immediately with a message
naming the offending key.

---

## 5. Using real terrain

Point a scene at a Phase 4 terrain export and everything downstream follows —
no code changes, because the synthetic bathymetry satisfies the same contract.

```yaml
scene:
  water_level: 100.0          # MUST equal grid_meta.json's z_w
  epsg: 32616                 # likewise
bathymetry:
  source: houdini_export      # every other bathymetry key is then ignored
```

### What the export must contain

Six files, all `float32`, shape `(ny, nx)`, C-order, `[y, x]` indexing:

| File | Contents |
|---|---|
| `grid_meta.json` | `x0, y0, dx, dy, nx, ny, z_w, epsg` |
| `terrain_z.npy` | bed elevation [m] |
| `depth.npy` | `z_w − terrain_z`, positive in water |
| `sdf.npy` | signed distance to the shoreline, **positive inland** |
| `shore_normal.npy` | unit vectors pointing inland |
| `bottom_type.npy` | `uint8` class index (optional) |

`shore_normal` is accepted as either `(2, ny, nx)` or `(ny, nx, 2)` — the data
contract writes one, this package stores the other, and guessing would rotate
every refraction angle by ninety degrees.

### What is checked on load

The loader refuses rather than proceeding on data that would produce a
plausible-looking wrong answer:

- `depth == z_w − terrain_z` exactly
- `sign(sdf) == −sign(depth)` away from the contour
- `│shore_normal│ = 1`, pointing inland
- shapes match `grid_meta.json`
- **square cells** — `dx != dy` is refused, not silently distorted
- `terrain_z` agrees with `z_w − depth`
- **`scene.water_level` equals the export's `z_w`**, and likewise `epsg`

That last one matters more than it looks. A mismatch builds every mesh at the
config's value while measuring every depth from the export's, so water and
terrain agree *with each other* and both sit at the wrong absolute height —
perfect in isolation, metres out against anything else in the world.

### Path resolution

`source` is tried against the working directory, then the config file's own
directory, then the directory above it. A miss lists all three.

### What changes with real terrain

- `beach_slope()` **measures the bed** instead of evaluating Dean's formula.
- The shoreline is *found* rather than assumed at a fixed Y, so a coast may face
  any direction and the water may run off the domain edge.
- Every window, transect and camera orients from the shore normal.

Verified working: non-square grids, islands, water at the domain edge, any coast
orientation, arbitrary `dx` and origin including negative coordinates.

**Refused, with an explanation:** an all-water or all-land export. Every window
and camera is positioned relative to a waterline, so there is nothing sensible to
return.

### Scaling

| Grid | On disk | Load | RAM | Validate | Foam spin-up |
|---|---|---|---|---|---|
| 1024² | 21 MB | 0.16 s | 0.03 GB | 0.19 s | 18 s |
| 2048² | 84 MB | 0.48 s | 0.12 GB | 0.83 s | 71 s |
| 4096² | 336 MB | 1.34 s | 0.50 GB | 4.25 s | 5 min |

Everything is O(N²). The binding constraint is the foam spin-up, not memory. The
loader converts float32 → float64, so RAM is 2× the disk figure.
---

## 6. Meshes and export

```bash
python scripts/run_scene.py configs/my_scene.yaml --mesh
python scripts/run_scene.py configs/my_scene.yaml --mesh \
       --mesh-dx 0.25 --mesh-region 420 370 540 404 --mesh-t 12.5
```

Writes into `runs/<scene>/mesh/`:

| File | What it is |
|---|---|
| `water_0000.ply` | Displaced water surface + per-vertex channels |
| `terrain_0000.ply` | The bed, co-registered and slightly larger |
| `scene.py` | Mitsuba scene **dict** for `mi.load_dict` |
| `scene.xml` | The same scene as XML, for the `mitsuba` CLI |
| `*.json` | Provenance: `t`, wind, seed, `git_sha`, channel ranges |

| Flag | Default | Effect |
|---|---|---|
| `--mesh-dx D` | `output.mesh_dx` | Post spacing, metres |
| `--mesh-region X0 Y0 X1 Y1` | shoreline window | Bound the mesh, scene coordinates |
| `--mesh-t T` | `0.0` | Scenario time of the frame |
| `--mesh-obj` | off | Also write an OBJ (geometry only) |
| `--mesh-max-vertices N` | 12,000,000 | Raise the guard on a big machine |

### Choosing a spacing and a region

Post spacing enters **quadratically**, so it is the strongest lever:

| Vertices | Build time | PLY size |
|---|---|---|
| 300 k | ~6 s | 23 MB |
| 10 M | ~3 min | 780 MB |
| 100 M | ~33 min | 7.8 GB |

About **50,000 water vertices/s**, single-threaded. More cores will not help
(no parallelism in `pywave`) and nor will a GPU (no GPU code) — a fast card
matters for Mitsuba, not for this.

Meshing a whole domain at a fine spacing is usually millions of posts: the
reference lake at 0.125 m is 64 M over 1 km². So either coarsen `--mesh-dx` or
bound `--mesh-region`. **Bounding the region is what stands in for LOD rings**
until §6.2 exists. The default is a shoreline window about 900 posts a side,
clipped to the water body.

`--mesh-t` is free: the surface is a pure function of `t`, so frame 12.5 s costs
exactly what frame 0 costs.

### The PLY

Binary little-endian. Thirteen `float32` per vertex:

| # | Property | Unit | Meaning |
|---|---|---|---|
| 0–2 | `x, y, z` | m | **Displaced** position, Z up |
| 3–5 | `nx, ny, nz` | — | Analytic normal from the spectral slopes |
| 6 | `aniso` | — | Crosswind / upwind slope variance ratio |
| 7 | `depth` | m | Still-water depth at the rest position |
| 8 | `foam` | — | Foam coverage fraction, 0–1 |
| 9 | `mss` | — | Sub-mesh mean square slope; `α = √mss` |
| 10–11 | `wdir_x, wdir_y` | — | Local **refracted** wave direction |
| 12 | `wetness` | — | Submergence duty cycle, 0–1 |

Faces are `uchar` count + three `int32` — 13 bytes per triangle, CCW from +Z.
Size is `header + V×52 + F×13`, about 78 B/vertex in practice.

**Key channels by name, not index.** Position and normal are always first, but
the channels after them are written in *alphabetical* order, so adding one shifts
every index after it.

```python
from pywave import export
data, faces = export.read_ply("water_0000.ply")
data["mss"]     # (V,) float32
```

`read_ply` is deliberately narrow — it reads what `write_ply` produces, for the
round-trip test. For anything else use `plyfile`, `open3d` or `trimesh`, but
check first: some libraries silently drop unknown vertex properties, which is the
failure the format choice exists to avoid.

**PLY, not OBJ.** OBJ has positions, normals and texture coordinates and no
mechanism for arbitrary per-vertex scalars, so it carries **none** of the
channels — everything that makes the mesh worth generating. `--mesh-obj` writes
one for MeshLab and says in its own header what it dropped. For reference, one
window is 21.6 MB as PLY with all seven channels and 42.9 MB as OBJ with none.

### The Mitsuba scene

`scene.py` is the primary form. It hands `mi.load_dict` a dictionary and resolves
the PLY paths against its own location, so it works from any directory.

```python
from scene import scene_dict
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
img = mi.render(mi.load_dict(scene_dict()), spp=256)
```

```bash
python scene.py --variant cuda_ad_rgb --spp 256 -o test.exr
```

**Both forms are untested against Mitsuba** — it is a Phase 7 dependency and is
not installed here. They are generated from one set of parameters so they cannot
drift apart, and the suite checks the dict's structure and the XML's
well-formedness, but not that Mitsuba likes either.

The BSDFs are placeholders — diffuse sand, and `roughdielectric` with
`alpha = √(mean mss)` baked to a **constant**, because a stock BSDF cannot read
mesh attributes. The channels are in the PLY and unused; consuming them is
Phase 7's job (see [algorithms.md §16](algorithms.md#16-per-vertex-channels-and-the-lod-invariant)).

**First thing to run** when Mitsuba is available:

```python
import mitsuba as mi
mi.set_variant("scalar_rgb")
print(mi.traverse(mi.load_dict({"type": "ply", "filename": "water_0000.ply"})))
```

Look for `vertex_mss`, `vertex_foam` and the rest. If they are not there, the
delivery format needs rethinking before any C++ gets written.

---

## 7. Animations

```bash
python scripts/animate.py                          # both views
python scripts/animate.py --mode shore --seconds 8
python scripts/animate.py configs/coastal_bay.yaml --start 3600
python scripts/run_scene.py my_scene.yaml --animate
```

**`open`** looks straight down at open water. **`shore`** is a plan view across
the waterline: waves shoaling in, the breaker line, foam in the surf band, and
the swash edge breathing at the peak period over wet sand.

`--start` is free — the surface is a pure function of `t`, so an hour in costs
what the first second costs.

MP4 when ffmpeg is available, GIF otherwise. GIF needs no install but wave
texture is close to worst case for it, so expect several MB; `pip install -e
".[video]"` gets MP4 at roughly a tenth the size, encoded `libx264/yuv420p` so it
plays in QuickTime, PowerPoint and browsers.

**Foam does not pulse.** Breaking is a *statistical* condition here, not an
instantaneous one, so the seeding region is steady. Individual breaking events
would need a wave-by-wave model.

---

## 8. Using the API

```python
import numpy as np
from pywave import load_config, spectrum, moments, tiling, nearshore, mesh, export
from pywave.bathymetry import Bathymetry

cfg = load_config("configs/test_lake.yaml")

# --- what the spectrum says the sea state is ---
print(f"Tp   {1 / cfg.f_p:.2f} s")
print(f"Hs   {spectrum.hs_spectral(cfg.wind.speed, cfg.wind.fetch):.4f} m")
print(f"mss  {moments.mss_above(0.0, cfg.wind.speed, cfg.wind.fetch):.4f}")

# --- a surface ---
tiles = tiling.TileSet.build(cfg)
field = tiles.sample(np.linspace(0, 50, 256), np.zeros(256), t=2.5)
field.normals()            # (..., 3) unit normals, Z up

# --- the nearshore ---
bathy = Bathymetry.from_config(cfg, fine=True)
nf = nearshore.transform(tiles, bathy, cfg, x, y, t=2.5)
nf.hs_local, nf.breaking, nf.wetness, nf.depth

# --- a mesh ---
m = mesh.build_water_mesh(tiles, bathy, cfg, t=2.5, dx=0.125,
                          region=(444.0, 360.0, 556.0, 410.0))
m.validate()
export.export_frame(m, "out/", "water_0000")
```

### Recipes

**BSDF roughness for a given mesh resolution**

```python
k_cut = np.pi / 0.125
alpha = np.sqrt(moments.mss_above(k_cut, u10=5.0, fetch=1000.0))

up, cross = moments.mss_anisotropic(5.0, 1000.0, theta_wind, k_cut=k_cut)
alpha_u, alpha_v = np.sqrt(2 * up), np.sqrt(2 * cross)
```

**Many points at one time** — evaluate the grids once and pass them in:

```python
fields = tiles.evaluate_grids(t)
field = tiles.sample(X, Y, t, fields=fields)
```

**Finite depth** — every dispersion-aware function takes `depth`:

```python
tiles = tiling.TileSet.build(cfg, depth=3.0)
mss = moments.mss_above(0.0, u10, fetch, depth=3.0)
```

This applies a *uniform* depth to the whole tile; spatially varying depth is what
`nearshore.transform` is for.

---

## 9. Module reference

| Module | Contents |
|---|---|
| `constants` | Conventions table, `G`, `K_CAPILLARY`, water optical constants |
| `config` | Scene dataclasses + YAML load — the only place degrees exist |
| `spectrum` | JONSWAP, spreading, dispersion, `S(f) → S(k)`, initial amplitudes |
| `moments` | `m0`, `mss_above(k)`, `mss_anisotropic`, `Tz`, Cox–Munk |
| `surface` | FFT synthesis, `WaveTile`, `SurfaceField` |
| `tiling` | `TileSet`, band edges, periodic sampling, composition |
| `bathymetry` | `Bathymetry` — loaded export or synthetic Dean beach |
| `nearshore` | Shoaling, refraction, breaking, wetness, `transform` |
| `foam` | `FoamModel`, bounded spin-up |
| `mesh` | Water extent, triangulation, displacement, `TriMesh` |
| `channels` | Per-vertex channel packing |
| `export` | PLY, OBJ, terrain export, Mitsuba scene, provenance |

Key signatures:

```python
spectrum.jonswap_sf(f, u10, fetch, gamma)            -> S(f)   [m²/Hz]
spectrum.jonswap_sk(kx, ky, u10, fetch, theta, depth)-> S(k)   [m⁴]
spectrum.hs_spectral(u10, fetch, gamma)              -> Hs     [m]     ← quote this
spectrum.dispersion_k(omega, depth)                  -> k      [rad/m]

moments.mss_above(k_cut, u10, fetch, depth)          -> slope variance above k_cut
moments.mss_anisotropic(u10, fetch, theta, k_cut)    -> (up, cross)
moments.zero_crossing_period(u10, fetch, gamma)      -> Tz     [s]

surface.WaveTile.build(size, n, u10, fetch, theta, seed, band, rotation, depth)
tiling.TileSet.build(cfg, depth) / .sample(x, y, t)

Bathymetry.from_export(directory) / .from_config(cfg, fine) / .sample(x, y)
nearshore.transform(tileset, bathy, cfg, x, y, t, refraction, depth_limit)
foam.FoamModel(bathy, equilibrium, half_life).evaluate(breaking_at, cg, t)
mesh.build_water_mesh(...) / mesh.build_terrain_mesh(...)
export.write_ply(mesh, path) / export.write_terrain_export(bathy, directory)
```

`depth=None` selects the deep-water limit everywhere it appears.

---

## 10. Reference numbers

`configs/test_lake.yaml`: U10 = 5 m/s, fetch = 1000 m, γ = 3.3, deep water.
Every number is produced by the current code.

| Quantity | Value |
|---|---|
| Dimensionless fetch `X̃` | 392.4 |
| Peak frequency / period | 0.9568 Hz / 1.045 s |
| Peak wavelength `λp` | 1.705 m |
| `hs_spectral` | 0.0857 m |
| `fetch_limited_hs` | 0.0808 m |
| `Tz` | 0.816 s |
| `mss_above(0)` to k = 400 | 0.04611 |
| RMS slope | 12.1° |
| `mss_up` / `mss_cross` | 0.02560 / 0.02048 (ratio 1.25) |
| Cox–Munk at U12 = 5.13 m/s | 0.02928 |

| Composite | Value |
|---|---|
| Hs from `m0` | 0.0853 m |
| Realised `4·std(h)` | 0.0839 m (−1.6%) |
| Resolved mss (k < 34.97) | 0.02129 |
| LOD invariant closure | 0.01% |
| `min(J)` at choppiness 1.0 | 0.671 |

| Nearshore | Value |
|---|---|
| Foreshore slope | 6.7% |
| Iribarren ξ | 0.30 — spilling |
| Breaking depth | 0.11 m |
| Runup / swash | 2.5 cm / 0.38 m |
| Surf zone | 1.0 m |

---

## 11. Validating a build

```bash
pytest                      # ~2 min
pytest -m gate1             # spectrum and moments only
pytest -m "not slow"        # skips the long time-series checks
```

Running the suite regenerates **[validation_report.md](validation_report.md)** —
a table of every measured quantity against the reference it was judged on, with a
`git_sha` header. Tests record numbers, not just pass/fail, because a reviewer
needs "realised Hs = 0.0842 m vs 0.0853 m, −1.3%" rather than a green checkmark.

| File | Covers |
|---|---|
| `test_spectrum.py` | Gate 1 — Hs relation, spreading, the Jacobian, moments |
| `test_surface.py` | Gate 2 — realness, variance, LOD invariant, crest speed, `Tz` |
| `test_reproducibility.py` | Gate 3 — determinism, regression baseline, config contracts |
| `test_terrain_export.py` | Gate 4 — loading a terrain export |
| `test_nearshore.py` | Gate 5 — shoaling, refraction, breaking, foam |
| `test_mesh.py` | Gate 6 — mesh, channels, PLY round trip |

### The regression baseline

`tests/baseline/` holds a pinned tile and composite sample at fixed seed and `t`.
They are the only thing that catches an accidental convention change during
refactoring — a flipped FFT sign, a lost `N²`, a `flip_k` off-by-one. Those leave
the variance, the spectrum *and* the slope statistics all correct.

Regenerate with `python tests/make_baseline.py`, but treat it as deliberate: **if
a baseline test fails, assume the code changed, not that the baseline is stale.**

### Deviations

Four places the implementation departs from the cookbook, each with the
measurement that motivated it. Summarised in
[algorithms.md §17](algorithms.md#17-summary-of-approximations) and recorded in
full in the validation report.

---

## 12. Conventions

Fixed in `pywave/constants.py`, which every module imports. Read it before
writing code against this package.

| Item | Convention |
|---|---|
| Coordinates | Right-handed, **Z up**. X east, Y north. |
| Units | Metres, seconds, radians. Degrees only in config files. |
| Angular frequency | `ω` (rad/s) everywhere; `f` (Hz) only inside `spectrum.py`. |
| Wavenumber | `k` in rad/m, not cycles/m. |
| Direction | `θ` is the direction waves travel **toward**, CCW from +X. |
| Depth sign | `d > 0` in water; `depth = z_w − z_terrain`. |
| SDF sign | `s < 0` in water, `s > 0` on land — **opposite** to depth. |
| FFT | numpy default; multiply by `N²` going from spectrum to space. |
| Time origin | `t = 0` is scenario start; the surface is a pure function of `t`. |
| Random seed | One integer, used in `initial_amplitudes()` and nowhere else. |

**Two different quantities are both called `m₂`** — the wavenumber second moment
(mean square slope, dimensionless, sets BSDF roughness) and the frequency second
moment (rad²/s², sets `Tz`). In deep water slope variance is the *fourth*
frequency moment. See [algorithms.md §2](algorithms.md#2-conventions).

---

## 13. Troubleshooting

| Symptom | Cause |
|---|---|
| `exceeds max_vertices` | Coarsen `--mesh-dx`, shrink `--mesh-region`, or raise the guard |
| `no wet cells in region` | The region misses the water; check it against `summary.md`'s extents |
| Mesh is a thin strip | The default region is a shoreline window — pass `--mesh-region` |
| `scene.water_level is X but ... z_w = Y` | Make them match; see §5 |
| `does not name a terrain export` | The message lists every path tried |
| `no shoreline` | The export is all water or all land |
| `must be finer than dx` | `bathymetry.surf_dx` has to be smaller than `dx` |
| `tile bands must be contiguous` | Bands must span `[0,1]` with no gap or overlap |
| `no tile resolves k_p` | Tiles too coarse for this sea state; see §4 |
| Waves look wrong at the shore | Check `sign(sdf)`; it must be **negative in water** |
| GIF is enormous | `pip install -e ".[video]"` for MP4 |

---

## 14. Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | `spectrum.py`, `moments.py` | **Implemented** |
| 2 | `surface.py`, `tiling.py` | **Implemented** |
| 3 | Validation suite, generated report | **Implemented** |
| 4 | Terrain export | **Loader implemented**; the Houdini side is yours |
| 5 | `nearshore.py`, `foam.py` | **Implemented** |
| 6 | `mesh.py`, `channels.py`, `export.py` | **Implemented** — constant spacing; §6.2 LOD rings deferred |
| 7 | Mitsuba `roughwater` BSDF plugin | Not started |
| 8 | Spectral emissivity table | Not started |
| 9 | EMBER integration | Not started |
| 10 | Physical validation and traceability | Not started |

Known open items:

- **LOD rings (§6.2)** — bounded mesh regions stand in for now.
- **`bottom_type`** is currently binary (wet/dry) and carries nothing beyond
  `depth > 0`. Phase 7/8 will want real sediment classes.
- **Fetch is a single value** — a closed basin has a direction-dependent fetch.
- **`tiling.band_edges`** has an unreachable upper guard, left as a tripwire
  against a future change to how `k_ref` is defined.
