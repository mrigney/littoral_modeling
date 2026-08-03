# pywave — User's Guide

Spectral water surface synthesis for physics-based EO/IR littoral scene generation.

This guide covers everything currently implemented: **Phase 1** (spectrum and
moments), **Phase 2** (FFT surface synthesis and tiling), **Phase 3** (the
validation suite), and **Phase 5** (nearshore transformation, built against a
synthetic beach). Remaining phases are listed under [Roadmap](#12-roadmap).

---

## Contents

1. [What this package is](#1-what-this-package-is)
2. [Installation](#2-installation)
3. [Quickstart](#3-quickstart)
4. [Core concepts](#4-core-concepts)
5. [Conventions you must know](#5-conventions-you-must-know)
6. [The config file](#6-the-config-file)
7. [Module reference](#7-module-reference)
8. [Recipes](#8-recipes)
9. [Reference numbers — the test lake](#9-reference-numbers--the-test-lake)
10. [Known gotchas](#10-known-gotchas)
11. [Where the physics comes from](#11-where-the-physics-comes-from)
12. [Roadmap](#12-roadmap)
13. [Validating a build](#13-validating-a-build)
14. [The nearshore](#14-the-nearshore)
15. [Running a scene end to end](#15-running-a-scene-end-to-end)

> **In a hurry?** [gallery.md](gallery.md) explains the whole model in eight
> figures.

---

## 1. What this package is

`pywave` builds a **time-evolving water surface** from a wind-wave spectrum, and
reports the **statistics of that surface** — height variance, slope variance,
directional anisotropy — as physically derived numbers rather than art-directed
knobs.

It exists because a rendered water surface for EO/IR simulation has to be
defensible. Every appearance parameter downstream (microfacet roughness, glint
anisotropy, foam coverage) is a moment of one spectrum `S(k)`. Nothing is tuned
independently.

> **The governing rule.** The spectrum is the single source of truth. If two
> quantities disagree, the spectrum wins and the other one has a bug.

### What it gives you today

| Capability | Entry point |
|---|---|
| JONSWAP spectrum in frequency, with fetch-limited parameters | `spectrum.jonswap_sf`, `spectrum.jonswap_params` |
| Frequency-dependent directional spreading | `spectrum.spreading` |
| 2-D wavenumber spectrum `S(kx, ky)`, deep or finite depth | `spectrum.jonswap_sk` |
| Height/slope moments, sub-grid slope variance | `moments.mss_above`, `moments.m0` |
| Zero-crossing period | `moments.zero_crossing_period` |
| Cox–Munk empirical cross-check | `moments.cox_munk_from_u10` |
| A real surface `h(x, y, t)` with exact analytic slopes | `surface.WaveTile` |
| Multi-tile composite that hides FFT periodicity | `tiling.TileSet` |
| Synthetic beach with the Phase 4 field contract | `bathymetry.Bathymetry` |
| Shoaling, Snell refraction, depth-limited breaking | `nearshore.transform` |
| Swash wetness as a duty cycle, for the thermal channel | `nearshore.wetness_fraction` |
| Foam with bounded, reproducible spin-up | `foam.FoamModel` |

### What it does not do yet

No terrain import (Phase 4 — `bathymetry.py` supplies a synthetic stand-in), no
mesh export, no BSDF, no emissivity, no EMBER integration. See
[Roadmap](#12-roadmap).

---

## 2. Installation

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

Runtime dependencies are `numpy`, `scipy`, and `pyyaml`. The `dev` extra adds
`pytest` and `matplotlib`.

Verify:

```bash
python -c "import pywave; print(pywave.__version__)"
```

---

## 3. Quickstart

```python
import numpy as np
from pywave import load_config
from pywave import spectrum, moments, tiling

cfg = load_config("configs/test_lake.yaml")

# --- what the spectrum says the sea state is -----------------------------
print(f"peak period   {1 / cfg.f_p:.2f} s")          # 1.05 s
print(f"peak wavelen  {cfg.lambda_p:.2f} m")         # 1.71 m
print(f"Hs            {spectrum.hs_spectral(cfg.wind.speed, cfg.wind.fetch):.4f} m")

# --- total mean square slope, to the capillary cutoff --------------------
mss = moments.mss_above(0.0, cfg.wind.speed, cfg.wind.fetch)
print(f"mss total     {mss:.4f}   (RMS slope {np.degrees(np.arctan(np.sqrt(mss))):.1f} deg)")

# --- build and sample an actual surface ----------------------------------
tiles = tiling.TileSet.build(cfg)
print(f"composite Hs  {tiles.hs():.4f} m")

x = np.linspace(0.0, 50.0, 256)
y = np.zeros_like(x)
field = tiles.sample(x, y, t=2.5)

print(f"h range       {field.h.min():+.3f} .. {field.h.max():+.3f} m")
print(f"normals shape {field.normals().shape}")
```

`t` is seconds since scenario start. Ask for `t = 8000.0` directly — the surface
is a pure function of time, with no accumulated state and no drift.

### Or skip the API entirely

```bash
cp configs/test_lake.yaml configs/my_scene.yaml
python scripts/run_scene.py configs/my_scene.yaml
```

That writes a scene report, the eight figures, a self-contained HTML page and the
per-cell nearshore channels into `runs/my_scene/`. Start with `summary.md`.
Everything is derived from the config, so changing the wind, the fetch or the
beach changes every artifact. See [section 15](#15-running-a-scene-end-to-end).

---

## 4. Core concepts

### 4.1 One spectrum, many derived quantities

```
                    JONSWAP S(f)  +  spreading D(f, θ)
                                 │
                       Jacobian  │  S(k,θ) = S(f) D(f,θ) Cg / (2π k)
                                 ▼
                              S(kx, ky)
                    ┌────────────┴────────────┐
                    ▼                         ▼
             moments (analytic)        FFT synthesis
             m0 → Hs                   h(x,y,t)
             m2 → mss → BSDF α         slopes, displacement
             mss_above(k) → sub-grid    normals
```

The left branch is exact and unbounded in wavenumber. The right branch is a
discrete grid truncated at its Nyquist. The two are reconciled by the LOD
invariant below.

### 4.2 The LOD invariant

An FFT grid of spacing `dx` cannot represent anything shorter than `2 dx`. The
slope variance it is missing is not lost — it is *accounted for* and handed to
the BSDF as sub-facet roughness:

```
mss_resolved(dx)  +  mss_above(π/dx)  =  mss_total
```

This holds to 0.01% for the shipped test config. It is the mechanism that keeps
the surface looking identical across LOD transitions: geometry lost when the mesh
coarsens reappears as roughness, and the total radiometric response is unchanged.

```python
res   = tiles.mss()                                     # 0.02129, from the grids
above = moments.mss_above(tiles.k_max, u10, fetch)      # 0.02482, analytic tail
total = moments.mss_above(0.0, u10, fetch)              # 0.04611
assert abs(res + above - total) / total < 1e-3
```

### 4.3 Two different quantities are both called "m2"

This trips people up badly enough that the package refuses to abbreviate:

| Name | Definition | Sets |
|---|---|---|
| `moments.m2(...)` | `∫ k² S(k) d²k` — second moment in **wavenumber** | Mean square slope, BSDF roughness |
| `moments.moment_omega(2, ...)` | `∫ ω² S(ω) dω` — second moment in **angular frequency** | Zero-crossing period `Tz = 2π√(m0/m2_ω)` |

In deep water `ω² = g k`, so mean square slope is the **fourth** frequency moment,
not the second. Conflating them produces a `Tz` wrong by orders of magnitude.

### 4.4 Why three tiles instead of one

A single FFT tile is exactly periodic at its side length. The repetition is much
more visible than it sounds, because specular glint makes it obvious — the eye and
a change detector both lock onto repeated sparkle long before they notice repeated
displacement.

The fix is to sum several tiles with **incommensurate sizes**, each carrying a
**disjoint band** of the spectrum, each **rotated** by a multiple of the golden
angle. Disjointness is what makes the sum valid: variances add only for
uncorrelated components, so overlapping bands would double-count energy and the
composite would no longer integrate to `m0`. `SurfaceConfig` validates that the
bands are contiguous and non-overlapping at load time.

### 4.5 Hs has two values and neither is wrong

JONSWAP supplies two independently fitted power laws that are not mutually
consistent:

| Quantity | Fit | Implied `m0` scaling |
|---|---|---|
| scale parameter | `α = 0.076 X̃^-0.22` | `m0 ~ α f_p^-4 ~ X̃^1.10` |
| peak frequency | `f_p ~ X̃^-0.33` | |
| dimensionless energy | `ε = 1.6e-7 X̃` | `m0 ~ X̃^1.00` |

`spectrum.hs_spectral` integrates the spectrum (first pair);
`spectrum.fetch_limited_hs` uses the energy growth law (second). Their ratio is
`0.787 · X̃^0.05` — verified numerically to `p = 0.05001` over four decades of
fetch, against a structural prediction of exactly 0.05.

**Quote `hs_spectral`.** It is what a realised FFT surface actually reproduces.
For the test lake that is 0.0857 m against the fit's 0.0808 m, a 6% gap that no
correct implementation can close. Rescaling the spectrum to hit the fit would
change `α`, hence the Phillips tail, hence every downstream roughness — which is
exactly the independent tuning the design rule forbids.

---

## 5. Conventions you must know

Violating any of these silently is the fastest route to a multi-day debugging
session. They live in `pywave/constants.py`, which every other module imports, so
they are always one jump away from any code you write.

| Item | Convention |
|---|---|
| **Coordinates** | Right-handed, **Z up**. X east, Y north. Houdini is Y-up; convert once, at export from Houdini, nowhere else. |
| **Units** | Metres, seconds, radians internally. Degrees appear **only** in config files, and only for angles a human types. |
| **Angular frequency** | `omega` (rad/s) everywhere. `f` (Hz) only inside `spectrum.py`, where JONSWAP is defined. Never mix in one expression. |
| **Wavenumber** | `k` in rad/m (`k = 2π/λ`). Not cycles/m. |
| **Direction** | `theta` is the direction waves travel **toward**, CCW from +X. Oceanographers often use "coming from" — this package does not, anywhere. |
| **Water plane** | `z = z_w`, a scene constant. Wave displacement is relative to it. |
| **Depth sign** | `d > 0` in water, `d < 0` on land. `depth = z_w − z_terrain`. |
| **SDF sign** | `s < 0` in water, `s > 0` on land — deliberately **opposite** of depth, so `s` reads as "distance inland". |
| **FFT** | numpy default. `ifft2` includes the `1/N²`, so always multiply by `N²` going from spectral amplitudes to a spatial field. |
| **Time origin** | `t = 0` is scenario start. The surface is a pure function of `t`. No accumulated state, ever. |
| **Random seed** | One integer in config, used only in `spectrum.initial_amplitudes()`. Nowhere else. |

Degrees are converted to radians exactly once, in `config.load_config`. Nothing
downstream of that function ever sees a degree.

---

## 6. The config file

A scene is one YAML file. `configs/test_lake.yaml` is the reference:

```yaml
scene:
  domain: [1000.0, 1000.0]      # m, extent in X and Y
  water_level: 100.0            # m, z_w
  epsg: 32616                   # projected CRS of the scene

wind:
  speed: 5.0                    # m/s at 10 m reference height (U10)
  direction: 45.0               # deg CCW from +X, direction blowing TOWARD
  fetch: 1000.0                 # m

spectrum:
  model: jonswap                # only 'jonswap' is implemented
  gamma: 3.3                    # peak enhancement; 1.0 recovers Pierson-Moskowitz
  spreading: cos2s              # 'cos2s' | 'hasselmann'  ('donelan' raises)
  seed: 20260801                # the one seed for the whole scene

surface:
  tiles:                        # incommensurate sizes, disjoint bands
    - {size: 64.0, n: 512, band: [0.0, 0.35]}
    - {size: 37.0, n: 256, band: [0.35, 0.7]}
    - {size: 23.0, n: 256, band: [0.7, 1.0]}
  choppiness: 1.0               # physical value; do not tune

bathymetry:                     # the synthetic basin (PHASE 4 stand-in)
  profile: planar               # planar | embayment
  shoreline: 400.0              # m, y of the waterline; water below, land above
  dean_a: 0.100                 # m^(1/3); omit to derive from grain_size
  grain_size: 0.25              # mm, D50 -- ignored when dean_a is given
  max_depth: 5.0                # m
  bank_slope: 0.08
  dx: 1.0                       # m, field grid spacing
  surf_dx: 0.25                 # m, refined spacing through the surf zone
  amplitude: 40.0               # m, embayment only
  wavelength: 400.0             # m, embayment only

nearshore:                      # PHASE 5
  breaker_index: 0.78
  foam_halflife: 3.0            # s
  refraction: true
  shoaling: true

output:                         # PHASE 6 — parsed and validated, not yet used
  fps: 30.0
  mesh_dx: 0.125                # m, water mesh post spacing
  lod_rings: [{r: 100.0, dx: 0.125}, {r: 300.0, dx: 0.5}, {r: 1e9, dx: 2.0}]
```

### Key reference

**`scene`** — all three keys required. `domain` must be positive.

**`wind`** — all three required. `speed` and `fetch` must be positive.
`direction` is in **degrees** and is the direction the wind blows *toward*.

**`spectrum`** — all optional.
- `model` must be `jonswap`; anything else raises.
- `gamma` must be ≥ 1.
- `spreading` selects the exponent coefficients: `cos2s` (cookbook values) or
  `hasselmann` (as published). `donelan` raises `NotImplementedError` rather than
  silently falling back — a config naming a model that isn't running is worse than
  a crash.
- `seed` — one integer. `TileSet.build` spawns per-tile streams from it via
  `SeedSequence`, so each tile gets an independent reproducible stream while "one
  integer in config" stays true.

**`surface.tiles`** — required, at least one entry.
- `size` — tile side length in metres. Choose **incommensurate** values (64/37/23,
  not 64/32/16) so the tile lattices never re-align.
- `n` — samples per side, must be a **power of two**.
- `band` — `[lo, hi]` as *fractions*, not wavenumbers. Bands must be contiguous,
  disjoint, and together span exactly `[0, 1]`. A gap silently loses variance, so
  it is rejected at load.

**`bathymetry`** — all optional; describes the synthetic basin Phase 5 runs
against, and is replaced by the Houdini export when Phase 4 lands.
- `profile` — `planar` (straight shoreline) or `embayment` (cosine bays and
  headlands). Anything else raises.
- `shoreline` — Y coordinate of the waterline. Water lies *below* it in Y and
  land above, so a wind with a +Y component drives waves onto the beach.
- `dean_a` / `grain_size` — the Dean scale parameter directly, or the median
  grain diameter to derive it from. `dean_a` wins if both are given.
- `dx` / `surf_dx` — field grid spacing, and the refined spacing used for
  surf-zone products. `surf_dx` must be the finer of the two; it exists because
  the surf zone can be a metre wide, which a 1 m grid cannot resolve at all.

**`surface.choppiness`** — the horizontal displacement scale. `1.0` is the
physical value. Above ~1.3 the displacement map can fold through itself, inverting
normals; check with `WaveTile.jacobian(t).min() > 0`.

### How band fractions become wavenumbers

The fractions are mapped to real wavenumbers by `tiling.band_edges`:

- `k_ref = min(tile Nyquists)` — the most restrictive tile sets the scale, which
  guarantees every interior band fits on the grid carrying it.
- Interior edges are `fraction × k_ref`.
- The topmost band extends to **its own** tile's Nyquist, not to `k_ref`, so the
  finest tile contributes all the resolution it has instead of discarding its top
  octave.

For the shipped config (Nyquists 25.1 / 21.7 / 35.0 rad/m) this gives
`[0, 7.61) [7.61, 15.22) [15.22, 34.97)`. If a band cannot fit on its tile,
`band_edges` raises rather than clipping — clipping would lose variance in a way
that is very hard to trace back.

---

## 7. Module reference

### `pywave.constants`

Physical constants and the conventions table. Read this file before writing code
against the package.

```python
G                       # 9.81 m/s^2, fixed to match the cookbook's worked examples
JONSWAP_GAMMA           # 3.3
K_CAPILLARY             # 400 rad/m — upper limit of the slope integral
u10_to_u12(u10)         # neutral 1/7-power wind profile, for Cox-Munk comparison
WATER_IOR               # complex n+ik at 0.55 / 4.0 / 10.0 um (Hale & Querry 1973)
```

`K_CAPILLARY` is a **modelling choice, not a physical constant**. The Phillips
tail integrates logarithmically, so total mean square slope scales with
`log(k_max/k_p)`. It is stated once, here, and reported alongside any mss figure.

### `pywave.config`

```python
cfg = load_config("configs/test_lake.yaml")

cfg.scene.domain, cfg.scene.water_level, cfg.scene.epsg
cfg.wind.speed, cfg.wind.direction_rad, cfg.wind.fetch
cfg.surface.tiles[0].size, .n, .band, .dx, .k_min, .k_nyquist

cfg.jonswap        # (alpha, f_p, x_tilde)
cfg.f_p            # peak frequency [Hz]
cfg.omega_p        # [rad/s]
cfg.k_p            # deep-water peak wavenumber [rad/m]
cfg.lambda_p       # peak wavelength [m]
cfg.spectrum_kwargs()   # {'u10':…, 'fetch':…, 'gamma':…} for the moment functions
```

All config dataclasses are frozen and validate on construction, so an invalid
scene fails at load rather than three modules downstream.

### `pywave.spectrum` — Phase 1

```python
dimensionless_fetch(u10, fetch)          -> X̃ = g·X/U10²
jonswap_params(u10, fetch)               -> (alpha, f_p, x_tilde)
jonswap_sf(f, u10, fetch, gamma)         -> S(f) [m²/Hz]
fetch_limited_hs(u10, fetch)             -> Hs from the energy growth fit [m]
hs_spectral(u10, fetch, gamma)           -> Hs from integrating S(f) [m]   ← quote this
hs_ratio_spectral_to_fit(u10, fetch)     -> the 0.787·X̃^0.05 relation

spreading_exponent(f, f_p, model)        -> s(f)
spreading_norm(s, mode)                  -> N(s); 'closed' (production) or 'quad' (oracle)
spreading(theta, f, f_p, theta_wind)     -> D(f,θ), ∫D dθ = 1

dispersion_omega(k, depth)               -> ω² = g·k·tanh(k·d)
dispersion_k(omega, depth)               -> inverse, Eckart-seeded Newton
group_velocity(k, depth)                 -> Cg = n·ω/k
shoaling_n(k, depth)                     -> n = ½(1 + 2kd/sinh 2kd)

jonswap_sk(kx, ky, u10, fetch, theta_wind, depth) -> S(kx,ky) [m⁴]
initial_amplitudes(s_k, dk, seed)        -> complex h0(k) at t=0
```

`depth=None` selects the deep-water limit everywhere it appears. `jonswap_sf` is
evaluated in log space so the `f^-5` singularity is never formed.

The Jacobian in `jonswap_sk` is the one thing worth re-deriving if a result looks
wrong. Variance conservation `∫S(f)df = ∫∫S(k,θ) k dk dθ` gives

```
S(k,θ) = S(f) · D(f,θ) · Cg / (2π k)
```

The division by `k` comes from the **polar area element**, not from the frequency
Jacobian. Confusing the two is the most common way to be wrong here, and it shows
up as an Hs off by a factor you will spend days hunting.

**Note the sign in `initial_amplitudes`:** `h0 = 0.5·(ξ_r + i·ξ_i)·√(S)·dk` gives
`E|h0|² = 0.5·S·dk²`; the two-term time evolution then sums two independent
contributions for `E|h̃|² = S·dk²` per mode, hence `Σ E|h̃|² = m0`. Tessendorf's
written formula uses `1/√2` because he folds a factor of 2 into his `P_h`. If Hs
comes out `√2` too large, this is why.

### `pywave.moments` — Phase 1

```python
# gridded (from an FFT grid — truncated at its Nyquist)
m0(s_k, kx, ky)                          -> height variance [m²]
m2(s_k, kx, ky)                          -> resolved slope variance [-]

# analytic radial integrals (log-spaced, out to the capillary cutoff)
mss_between(k_lo, k_hi, u10, fetch, order=2)   -> band-limited moment
mss_above(k_cut, u10, fetch)                   -> ∫_{|k|>k_cut} k² S d²k   ← the workhorse
mss_anisotropic(u10, fetch, theta_wind)        -> (mss_up, mss_cross)

# frequency moments
moment_omega(order, u10, fetch)          -> ∫ ω^order S(ω) dω
zero_crossing_period(u10, fetch)         -> Tz = 2π√(m0/m2_ω) [s]

# oracles and cross-checks
integrate_sk_polar(u10, fetch, theta_wind)     -> brute-force 2-D polar integral
cox_munk_mss(u12) / cox_munk_from_u10(u10)     -> empirical slope variance
cox_munk_anisotropic(u12)                      -> (up, cross)
```

The radial integrals are **log-spaced**, and that is a correctness requirement
rather than an optimisation: the slope integrand falls as `1/k`, so equal
contributions come from equal *ratios* of `k`, and a linear grid puts essentially
all its samples in the last octave and under-reports the total.

`integrate_sk_polar` deliberately calls `jonswap_sk` on Cartesian `(kx, ky)`
instead of reusing the collapsed radial form, so it is an *independent* check of
the Jacobian and the spreading normalisation. An oracle that shared the analytic
shortcut would agree with it no matter how wrong both were.

`mss_anisotropic` returns the upwind/crosswind split, feeding the anisotropic
Beckmann parameters `α_u = √(2·mss_up)`, `α_v = √(2·mss_cross)` in Phase 7. By
construction `mss_up + mss_cross = mss_above(k_cut)`.

### `pywave.surface` — Phase 2

```python
tile = WaveTile.build(size=64.0, n=512, u10=5.0, fetch=1000.0,
                      theta_wind=np.radians(45.0), seed=20260801,
                      band=(0.0, 7.6), rotation=0.0, depth=None)

tile.m0(), tile.mss()                    # band-limited variances from the grid
tile.k_min, tile.k_nyquist, tile.dx
field = tile.evaluate(t=2.5)             # -> SurfaceField
jac   = tile.jacobian(t=2.5, choppiness=1.0)
```

`SurfaceField` carries five arrays, all indexed `[y, x]`:

```python
field.h            # vertical displacement about z_w [m]
field.dx_disp      # horizontal displacement x [m], before the choppiness scale
field.dy_disp      # horizontal displacement y [m]
field.slope_x      # dh/dx [-], analytic (spectral), not finite-differenced
field.slope_y      # dh/dy [-]

field.normals()        # (..., 3) unit normals, Z up
field.mss_resolved()   # realised <sx² + sy²>
field_a + field_b      # valid only for disjoint spectral bands
```

**Use `field.normals()`, not mesh normals.** Mesh normals are a finite-difference
approximation of a field already known exactly, and they lose precisely the
high-frequency content that matters most radiometrically.

Three properties of this construction matter more than anything else for
closed-loop simulation:

1. **Evaluable at arbitrary `t`.** No integration, no frame-to-frame state. Frame
   8000 costs what frame 1 costs and does not drift.
2. **Exactly deterministic from the seed.** Same seed, same `t`, same surface, on
   any machine, forever. That is a V&V property you can write down.
3. **Trivially parallel.** Any node can compute any frame with no coordination.

Tessendorf's `ω` quantisation for seamless looping is deliberately **not**
implemented — it introduces a detectable period, which is poison both for
training-set diversity and for closed-loop temporal analysis.

Construction is the expensive part (spectrum evaluation plus the RNG draw) and
happens once. `evaluate` is five inverse FFTs — a few milliseconds at 512².

### `pywave.tiling` — Phase 2

```python
ts = TileSet.build(cfg, depth=None)

ts.m0(), ts.mss(), ts.hs()               # totals; valid as plain sums because disjoint
ts.k_max                                 # highest wavenumber the composite resolves

fields = ts.evaluate_grids(t)            # cache this if sampling repeatedly
field  = ts.sample(x, y, t, fields=fields, order=3)
```

Also exported: `GOLDEN_ANGLE`, `tile_rotations(n)`, `band_edges(tiles)`,
`sample_periodic`, `sample_bilinear_periodic`, `composite_surface`.

Rotations come from the tile **index**, not the RNG — drawing them from the seed
would make the surface depend on it through two independent paths, violating "the
seed is used in `initial_amplitudes` and nowhere else".

**Sampling defaults to cubic (`order=3`) and should stay that way.** Bilinear is a
low-pass filter, and the tiles carry meaningful energy right to their Nyquist. For
a sinusoid at wavenumber `k` on a grid of spacing `d`, sampled at a uniformly
distributed sub-cell offset, retained power is `∫₀¹|1 − u + u·e^{ikd}|² du`, which
at Nyquist (`kd = π`) is `∫₀¹(1−2u)² du = 1/3` — two thirds of the top-octave power
destroyed. Measured on the test config: bilinear loses ~6% of Hs, cubic ~1%.

---

## 8. Recipes

### Get the BSDF roughness for a given mesh resolution

```python
from pywave import moments
import numpy as np

mesh_dx = 0.125                                  # m
k_cut   = np.pi / mesh_dx                        # 25.1 rad/m
mss_sub = moments.mss_above(k_cut, u10=5.0, fetch=1000.0)
alpha   = np.sqrt(mss_sub)                       # isotropic Beckmann alpha
```

For the anisotropic form:

```python
up, cross = moments.mss_anisotropic(5.0, 1000.0, theta_wind, k_cut=k_cut)
alpha_u, alpha_v = np.sqrt(2 * up), np.sqrt(2 * cross)
```

### Render a time series at one point

```python
ts = tiling.TileSet.build(cfg)
times = np.arange(0.0, 60.0, 1.0 / 30.0)
h = np.array([ts.sample(np.array([10.0]), np.array([10.0]), t).h[0] for t in times])
```

Slow, because it rebuilds every tile grid per frame. For many points at one time,
evaluate the grids once and pass them in:

```python
fields = ts.evaluate_grids(t)
field  = ts.sample(X, Y, t, fields=fields)
```

### Check the surface is sane

```python
tile  = ts.tiles[0]
field = tile.evaluate(0.0)

# variance matches the spectrum
assert abs(4 * np.std(field.h) - 4 * np.sqrt(tile.m0())) / (4 * np.sqrt(tile.m0())) < 0.05

# no folded displacement at physical choppiness
assert tile.jacobian(0.0, choppiness=1.0).min() > 0.0

# LOD invariant closes
res, above = ts.mss(), moments.mss_above(ts.k_max, cfg.wind.speed, cfg.wind.fetch)
total = moments.mss_above(0.0, cfg.wind.speed, cfg.wind.fetch)
assert abs(res + above - total) / total < 1e-3
```

### Work in finite depth

Every dispersion-aware function takes `depth`:

```python
s_k = spectrum.jonswap_sk(kx, ky, u10, fetch, theta_wind, depth=3.0)
ts  = tiling.TileSet.build(cfg, depth=3.0)
mss = moments.mss_above(0.0, u10, fetch, depth=3.0)
```

This applies a **uniform** depth to the whole tile. Spatially varying depth —
which is what a real littoral scene needs — is Phase 5.

---

## 9. Reference numbers — the test lake

`configs/test_lake.yaml`: U10 = 5 m/s, fetch = 1000 m, γ = 3.3, seed = 20260801,
deep water. Every number below is produced by the current code.

**Spectrum (Phase 1)**

| Quantity | Value |
|---|---|
| dimensionless fetch `X̃` | 392.4 |
| `alpha` | 0.02043 |
| peak frequency `f_p` | 0.9568 Hz |
| peak period `T_p` | 1.045 s |
| peak wavenumber `k_p` | 3.684 rad/m |
| peak wavelength `λ_p` | 1.705 m |
| `hs_spectral` | 0.0857 m |
| `fetch_limited_hs` | 0.0808 m |
| `Tz` (full band) | 0.816 s |
| `Tz` (band-limited to `k_max`) | 0.854 s |
| `mss_above(0)` to `k = 400` | 0.04611 |
| RMS slope | 12.1° |
| Cox–Munk at U12 = 5.13 m/s | 0.02928 |
| `mss_up` / `mss_cross` | 0.02560 / 0.02048 (ratio 1.25) |

Total mss sits 1.6× above the Cox–Munk value. That is expected and is not a
normalisation bug: Cox–Munk is an open-ocean, long-fetch fit, and a small
freshwater body at 1 km fetch has a different high-frequency balance. A factor of
10 would mean a bug; a factor under 2 means the ocean and the lake are different,
which they are.

**Composite surface (Phase 2)**

| Tile | `k_min` | `k_nyq` | band [rad/m] | `m0` [m²] | `mss` |
|---|---|---|---|---|---|
| L=64, n=512 | 0.098 | 25.13 | [0, 7.61) | 3.826e-4 | 0.00704 |
| L=37, n=256 | 0.170 | 21.74 | [7.61, 15.22) | 5.518e-5 | 0.00605 |
| L=23, n=256 | 0.273 | 34.97 | [15.22, 34.97) | 1.714e-5 | 0.00820 |
| **composite** | | | | **4.553e-4** | **0.02129** |

| Check | Value |
|---|---|
| composite Hs from `m0` | 0.0853 m |
| realised `4·std(h)`, 2e5 random samples | 0.0839 m (−1.6%) |
| realised mss, same samples | 0.01970 (−7.5%, the cubic sampling loss) |
| skewness of `h` | 0.010 |
| `min(J)` at choppiness 1.0 | 0.671 (no folding) |
| LOD invariant closure | 0.01% |

---

## 10. Known gotchas

**The spreading exponent is discontinuous at `f_p`.** The `cos2s` branches give
`s = 9.77` below the peak and `s = 11.5` above — an ~18% step exactly at the peak.
The published Hasselmann coefficients (6.97 / 9.77) are also discontinuous, by a
different amount, so this is a property of the fitted model, not a transcription
error. It does **not** affect `Hs` or any Gate 1 quantity, because `D` integrates
to 1 at every frequency and divides out of `m0`. It **does** affect directional
quantities — the `mss_up`/`mss_cross` anisotropy, hence BSDF tangent-frame
anisotropy in Phase 7. Use `model="hasselmann"` if that matters.

**Two sign conventions in the cookbook are wrong for numpy's FFT.** Both are
corrected here, with the derivation in the docstrings:

- *Time evolution.* The cookbook and Tessendorf write `h0(k)·e^{+iωt}`. Combined
  with numpy's `Σ h̃ e^{+ikx}` synthesis, the dominant term then carries phase
  `kx + ωt` and the whole sea marches **backwards**, into the wind. This package
  uses `e^{-iωt}`. Hermitian symmetry is unaffected by the flip, so the surface
  stays exactly real either way — which is why the error survives a realness
  check, a variance check, a spectrum check *and* a slope check, and is caught only
  by tracking a crest at `c = ω/k`.
- *Horizontal displacement.* The cookbook writes `-1j`; with `+ikx` synthesis that
  broadens crests and sharpens troughs, i.e. inverts the profile. This package uses
  `D̃ = +i·(k/|k|)·h̃`, which is exactly the Gerstner trochoid and sharpens crests
  as real waves do. Verify by measuring compression at crests:
  `corr(J − 1, h) < 0`.

**`spreading_norm` has two modes and they must agree.** `closed` uses the Wallis
integral via `gammaln` and is the production path; `quad` is direct quadrature and
is the test oracle. The cookbook prescribes quadrature, arguing this is not a hot
path — but `jonswap_sk` runs on 512² grids, where per-element quadrature is a
262144 × 8192 temporary, about 17 GB. The closed form keeps the cookbook's actual
intent (verify the normalisation numerically, don't trust it) at 1/1000 the cost.

**`moment_omega(4)` diverges** logarithmically — it is mean square slope in
disguise. A bare call without a band limit is meaningless. The divergence is
physical, not a bug.

**`flip_k` needs the roll.** On an `fftfreq` grid, `-k_j` lives at index `(-j) mod
N`. Reversing maps `j → N−1−j`, so a further roll by +1 is required. Getting this
wrong yields a complex-valued surface — but the Nyquist row and column are their
own negatives on an even grid, which makes some wrong implementations *nearly*
work. A spot check on a few bins is not enough.

**Vector quantities must be rotated back out of the tile frame.** `composite_surface`
rotates world sample points into each tile's frame by `−φ`, then rotates the
resulting displacement and slope vectors back by `+φ`. Forgetting step three
produces normals that are subtly wrong per tile in a way that averages out in
`mss` and survives every scalar check — it shows up only as anisotropy pointing
the wrong way, i.e. glint elongated along the wrong axis. (The separate
counter-rotation of the *wind* direction happens at tile construction.)

---

## 11. Where the physics comes from

- Hasselmann, K. et al. (1973). "Measurements of wind-wave growth and swell decay
  during the Joint North Sea Wave Project (JONSWAP)." *Deutsche Hydrographische
  Zeitschrift*, Suppl. A8.
- Hasselmann, D., Dunckel, M. & Ewing, J. (1980). "Directional wave spectra
  observed during JONSWAP 1973." *J. Phys. Oceanogr.* **10**, 1264–1280.
- Cox, C. & Munk, W. (1954). "Measurement of the roughness of the sea surface from
  photographs of the sun's glitter." *J. Opt. Soc. Am.* **44**(11), 838–850.
- Hale, G. & Querry, M. (1973). "Optical constants of water in the 200-nm to 200-µm
  wavelength region." *Appl. Opt.* **12**(3), 555–563.
- Tessendorf, J. "Simulating Ocean Water." SIGGRAPH course notes.
- Eckart, C. (1952). Approximate dispersion solution, used to seed the Newton
  iteration in `dispersion_k`.

Project-internal: `water-surface-modeling-primer.md` (the physics rationale) and
`littoral-water-implementation-cookbook.md` (the phase plan and gate checklists).

---

## 12. Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | `spectrum.py`, `moments.py` | **Implemented** |
| 2 | `surface.py`, `tiling.py` | **Implemented** |
| 3 | `tests/`, `docs/validation_report.md` | **Implemented** — 72 checks, see [Validating a build](#13-validating-a-build) |
| 4 | Terrain and lake basin in Houdini | Not started — `bathymetry.py` stands in |
| 5 | `nearshore.py`, `foam.py` — shoaling, refraction, breaking | **Implemented** — see [The nearshore](#14-the-nearshore) |
| 6 | `mesh.py`, `channels.py`, `export.py` — displaced mesh, LOD | Not started |
| 7 | Mitsuba `roughwater` BSDF plugin | Not started |
| 8 | Emissivity table | Not started |
| 9 | EMBER integration | Not started |
| 10 | Validation and documentation | Not started |

Phase 4 (terrain) and Phase 5 (nearshore) are the next steps. The config keys for
phases 5 and 6 (`nearshore`, `output`) already parse and validate, so those
sections of a scene file are stable ahead of the code that consumes them.

---

## 13. Validating a build

```bash
pytest                      # 46 checks, ~34 s
pytest -m gate1             # spectrum and moments only
pytest -m "not slow"        # skips the 14 s zero-crossing time series
```

Running the suite regenerates **[validation_report.md](validation_report.md)**,
which is the V&V artifact — a table of every measured quantity against the
reference it was judged on, with a `git_sha` header. Tests record numbers, not
just pass/fail, because a reviewer needs "realised Hs = 0.0842 m vs 0.0853 m,
−1.3%" rather than a green checkmark.

| File | Covers |
|---|---|
| `tests/test_spectrum.py` | Gate 1 — Hs relation, spreading normalisation, the Jacobian, dispersion, moments, Cox–Munk |
| `tests/test_surface.py` | Gate 2 — realness, variance, LOD invariant, Jacobian folding, crest speed, Tz |
| `tests/test_reproducibility.py` | Gate 3 — determinism, statelessness, regression baseline, config contracts |

### The regression baseline

`tests/baseline/` holds a pinned tile and a pinned composite sample, generated at
a fixed seed and `t`. They are the only thing that catches an accidental
convention change during later refactoring — a flipped FFT sign, a lost factor of
`N²`, a `flip_k` off-by-one. Those leave the variance, the spectrum *and* the
slope statistics all correct, so nothing else would notice.

Regenerate with `python tests/make_baseline.py`, but treat that as a deliberate
act: **if a baseline test fails, assume the code changed, not that the baseline
is stale.** Only regenerate after confirming the new behaviour is intended, and
say so in the commit message.

The baseline scene is defined in code inside `make_baseline.py` rather than
loaded from `configs/`, so editing a shipped config cannot silently invalidate
the regression record.

### Two gate criteria are deliberately substituted

Both are documented in full under *Gate deviations* in the generated report:

1. **Gate 1's "Hs within 2%"** is unmeetable — JONSWAP's `alpha` and `f_p` fits
   imply `m0 ~ X̃^1.10` while its energy growth law implies `m0 ~ X̃^1.00`, so the
   two agree at exactly one fetch. The suite instead pins the *relation*
   `hs_spectral / fetch_limited_hs = 0.78696 · X̃^0.05` across four decades of
   fetch, which is strictly stronger: the measured exponent is 0.050007 against a
   structural prediction of exactly 0.05.
2. **Gate 2's "skewness in [0, 0.3]"** is unmeetable because this model is linear
   in elevation. The Gerstner displacement relocates sample points horizontally
   but never changes the set of elevation values, so it cannot create the
   crest/trough asymmetry that skews a real sea. Measured area-weighted skewness
   is −0.015 at choppiness 1.0 versus −0.011 with displacement off. The suite
   asserts Gaussianity instead (`|skew| < 0.05`, `|excess kurtosis| < 0.1`) and
   reports the displaced value without bounding it. Positive elevation skewness
   requires second-order Stokes bound harmonics, which are not implemented.

---

## 14. The nearshore

Phase 5 transforms the deep-water surface as it approaches shore. It is built
and validated against a **synthetic** Dean beach (`pywave.bathymetry`), because a
synthetic profile has closed-form answers — Green's law, Snell's law, the breaker
index — that an exported heightfield cannot supply.

### 14.1 What the numbers say to build

Run the arithmetic before writing code (cookbook 5.1). For the test lake:

```
Hs = 0.086 m,  Tp = 1.05 s,  λp = 1.7 m,  max depth 5 m
deep-water cutoff = λ/2 = 0.85 m
```

**The lake is deep water everywhere except the last few metres.** Breaking
happens in ~10 cm of water, the surf zone is ~1 m wide, and the swash excursion
is 0.38 m. At 1 m GSD that entire zone is sub-pixel. So:

- **Shoaling and refraction are geometry.** They act over tens of metres and are
  resolved at sensor scale. Modelled as fields applied to the surface.
- **Breaking and swash are channels.** Sub-pixel. Modelled as per-cell fractional
  coverage. Building animated swash geometry would be weeks of work invisible at
  this resolution.

### 14.2 Bathymetry — the Phase 4 stand-in

```python
from pywave.bathymetry import Bathymetry

beach = Bathymetry.dean_beach()          # straight shoreline, contours parallel
bay   = Bathymetry.dean_embayment()      # cosine shoreline: bays and headlands

beach.depth          # z_w - z_terrain, positive in water   [ny, nx]
beach.sdf            # signed distance to the waterline, positive INLAND
beach.shore_normal   # (2, ny, nx) unit vectors pointing inland
beach.validate()     # the cookbook 4.5 assertion set; raises on failure
depth, sdf, normal = beach.sample(x, y)  # bilinear, at world coordinates
```

The profile is Dean's `h = A·y^(2/3)` — concave up, which real beaches are and a
linear ramp is not. `A` comes from sediment grain size (`DEAN_A`,
`dean_A_for_grain_size`).

**The field contract is identical to what Houdini will export**, so Phase 4 is a
loader swap rather than a physics change. `validate()` is written to be pointed
at real data unmodified.

One ordering subtlety: Dean's profile is a function of distance offshore, which
is `|sdf|` — so the signed distance must exist *before* the depth does. The
shoreline is defined geometrically, the sdf comes from an exact Euclidean
distance transform of the water mask, and the depth follows from the sdf.

### 14.3 Shoaling

```python
from pywave import nearshore

ks = nearshore.shoaling_coefficient(omega, depth)   # Ks = sqrt(Cg_deep / Cg)
```

Takes `omega`, never `k` — frequency is what is conserved as a wave shoals.

`Ks` is **not monotonic**: it dips to 0.913 near `kd = 1.2` before rising, and
recovers Green's law `Ks ~ d^(-1/4)` in the shallow limit. An implementation that
clamps `Ks ≥ 1` on the assumption that shoaling makes waves bigger would pass a
shallow-water check and still be wrong across the whole intermediate band —
which, for these 1-second waves, is where the beach actually is.

Applied **per spectral band**, not as one scalar, because it is
frequency-dependent (`nearshore.tile_frequencies` reduces each tile to its energy
centroid). This is a concrete reason the tiles carry disjoint bands.

### 14.4 Refraction

```python
theta, alpha = nearshore.refraction_angle(theta_deep, shore_normal, depth, omega)
kr = nearshore.refraction_coefficient(alpha_deep, alpha)
```

Snell's law against the full dispersion relation: `sin(α)/c = const`. The
invariant is conserved to 2.6e-16 on a planar beach.

`Kr = sqrt(cos α_deep / cos α) ≤ 1` always, on straight parallel contours:
oblique waves spread their energy over a longer stretch of shoreline. At 45°
incidence on this beach, `Kr` very nearly cancels the shoaling gain — so a test
that did not control for incidence angle would conclude shoaling does nothing.

Total local height is `H = H_deep · Ks · Kr`.

`nearshore.refraction_angle_blend` implements the cookbook's cheaper
depth-weighted blend. It is **not** the production path — it has no frequency
dependence and over-aligns at the waterline. The disagreement is measured in the
validation report (38.5° peak).

### 14.5 Breaking and wetness

```python
nearshore.breaking_mask(hs_local, depth, gamma_b=0.78)
nearshore.iribarren_number(slope, hs, l0)   # xi = 0.30 here -> spilling
nearshore.hunt_runup(xi, hs)                # 2.5 cm vertical
nearshore.swash_width(runup, slope)         # 0.38 m horizontal
nearshore.wetness_fraction(sdf, band)       # duty cycle -> the thermal channel
nearshore.wetness(sdf, band, t, period)     # instantaneous -> the shader
```

Inside the surf zone the height is depth-limited: `H = γ_b·d` exactly, so the
zone saturates instead of letting `Ks` diverge at the waterline.

**Two wetness functions, deliberately.** `wetness_fraction` is the fraction of a
wave period a point spends submerged — a closed-form duty cycle,
`(1/π)·arccos(2s/W − 1)`. That is what drives thermal inertia, and hence the cold
capillary-fringe line that is one of the most diagnostic features in a daytime
littoral LWIR image. A per-frame binary mask would be the wrong input to Hotts.
`wetness` is the instantaneous smoothstepped field for rendering. They agree in
shape but are not the same function.

### 14.6 Foam — the one stateful field

```python
from pywave import foam

model = foam.FoamModel(bathy=beach, half_life=3.0)
f = model.evaluate(lambda t: breaking_at(t), cg, t=120.0)   # cold-started
```

Foam has frame-to-frame memory, which would otherwise destroy the "any node, any
frame" property. The fix is **bounded spin-up**: to evaluate time `t`, start from
zero far enough back that the discarded history has decayed below tolerance.

The window is set by the half life, **not** by a frame count:

```
T_spin = t_half · log2(1/tol)      # 23 s for 0.5% at a 3 s half life
```

`foam.spinup_steps()` computes it and `evaluate` calls it by default. The
cookbook's "30 frames is plenty" is one second against a 3 s half life, leaving
79% of the initial condition intact — measured cold-vs-sequential error 2.2% per
cell, against the 1% the gate asks for. At the default it is 0.61%.

Foam does not sub-step at the frame rate: advection is semi-Lagrangian and
unconditionally stable, so 0.25 s costs 6× less for no visible difference.

### 14.7 Putting it together

```python
nf = nearshore.transform(tileset, beach, cfg, x, y, t)

nf.surface     # SurfaceField, shoaled/refracted/depth-limited
nf.hs_local    # local significant height after transformation
nf.breaking    # bool mask
nf.wetness     # duty cycle in the swash band
nf.depth, nf.sdf, nf.shoaling, nf.limiter
```

**What is approximate, stated plainly.** The FFT surface is
translation-invariant; refraction is not. `transform` rotates the local wave
*direction* — displacement and slope vectors, hence the surface normal, hence
everything the BSDF sees — but does not re-solve the wave field, so crest
*positions* do not move. Shoaling is applied as a per-band amplitude scale, exact
for the amplitude and ignoring the accompanying wavelength shortening. A full
mild-slope or Boussinesq solver is a much larger project and unnecessary at 8 cm
wave heights.

What is *not* approximated is the coefficient physics: `shoaling_coefficient` and
`refraction_angle` solve the full dispersion relation and Snell's law, and the
tests check them against closed-form answers at every cell.

---

## 15. Running a scene end to end

```bash
python scripts/run_scene.py                          # the shipped test lake
python scripts/run_scene.py configs/my_scene.yaml    # your own
python scripts/run_scene.py configs/my_scene.yaml --out /tmp/run --quick
```

One command takes a scene file and produces every artifact the model currently
knows how to make. Nothing is wired to a particular scene: change the wind, the
fetch or the beach and the figures, the numbers and the exported channels all
follow.

### What you get

```
runs/<scene>/
  summary.md          scene report — start here
  summary.json        the same numbers, machine-readable
  gallery.md          the eight figures with captions
  overview.html       one self-contained page; mail it to someone
  figures/*.png       the figures on their own, 150 dpi
  channels/
    manifest.json     grid georeferencing, units, ranges
    depth.npy         still-water depth, positive in water
    sdf.npy           signed distance to the waterline, positive inland
    shore_normal_x.npy, shore_normal_y.npy
    hs_local.npy      local significant height after the transform
    shoaling_gain.npy Hs_local / Hs_deep
    breaking.npy      1 where the depth-limited height is exceeded
    wetness.npy       fraction of a wave period submerged
    foam.npy          surf-zone foam coverage
    elevation.npy     wave elevation about z_w at t = 0
    slope_x.npy, slope_y.npy
```

The channels are on the **refined** grid (`bathymetry.surf_dx`), because the surf
zone can be a metre wide and the coarse grid cannot resolve it. They are what
Phase 6 will pack into per-vertex mesh attributes.

Two notes on how they are computed, both recorded in `manifest.json` so nobody
has to guess:

- Channels are evaluated with the wind blowing **straight onshore**, which makes
  the refraction coefficient exactly 1 and isolates shoaling. The deep-water
  surface in the figures uses the config's own wind direction.
- Everything is at `t = 0`, except foam, which is spun up for its full
  reproducibility window first.

### Writing a scene file

Copy [`configs/test_lake.yaml`](../configs/test_lake.yaml) — every key is
commented — and see [section 6](#6-the-config-file) for the full reference. The
config is validated on load, so a mistake fails immediately with a message
naming the offending key rather than surfacing three modules later.

Two things worth getting right, because they are the ones that bite:

**Tiles must resolve the peak.** The composite is only as good as its grids. A
tile's Nyquist is `π·n/size`, and at least one tile needs a Nyquist comfortably
above `k_p = 2π/λ_p`. Doubling the wind roughly quadruples `λ_p`, so a tile set
tuned for chop will not do for swell. `run_scene.py` asserts this and the test
suite checks every shipped config.

**Keep tile sizes incommensurate.** 64/37/23 rather than 64/32/16. Sizes in
simple ratios re-align their lattices and the periodicity the multi-tile
construction exists to hide comes straight back.

### The two shipped scenes

They are deliberately different regimes, not a rescale of one another:

| | `test_lake` | `coastal_bay` |
|---|---|---|
| Wind / fetch | 5 m/s, 1 km | 12 m/s, 40 km |
| Hs | 0.086 m | 1.43 m |
| Tp | 1.05 s | 4.75 s |
| λp | 1.71 m | 35.3 m |
| Shoreline | straight | embayed, ±220 m |
| Iribarren ξ | 0.33 — **spilling** | 0.61 — **plunging** |
| Surf zone | 1.0 m | 48.6 m |
| Swash excursion | 0.38 m | 7.1 m |

Running both and diffing the two `summary.md` files is the fastest way to see
which quantities are scene-dependent and which are structural.

### Runtime

About 40 s for the test lake, roughly two minutes for the coastal bay — the cost
is dominated by tile construction, which scales with the FFT grids, and by the
foam spin-up. `--quick` skips the channel export and the HTML page and roughly
halves it.

### Animations

```bash
python scripts/animate.py                          # both views, test lake
python scripts/animate.py --mode shore --seconds 8
python scripts/animate.py configs/coastal_bay.yaml --start 3600
python scripts/run_scene.py configs/my_scene.yaml --animate   # as part of a run
```

**`open`** looks straight down at open water — the composite surface evolving,
waves travelling along the wind.

**`shore`** is a plan view across the waterline: waves shoaling in, the breaker
line, foam in the surf band, and the swash edge advancing and retreating at the
peak period over wet sand. This is the view that shows what the nearshore model
is actually for.

`--start` is free. The surface is a pure function of `t`, so rendering an hour
into the scenario costs exactly what rendering the first second costs — no
spin-up, no drift.

#### Format

MP4 when ffmpeg is available, animated GIF otherwise. GIF works with no extra
install, but wave texture is close to the worst case for it — high-entropy
noise squeezed into 256 colours — so expect several megabytes for a few seconds.

```bash
pip install -e ".[video]"     # bundles an ffmpeg binary; no system install
```

That gets you MP4 at roughly a tenth the size, encoded `libx264 / yuv420p` so it
plays in QuickTime, PowerPoint and browsers. Without the `yuv420p` constraint
ffmpeg picks a 4:4:4 profile that several of those silently refuse.

#### What animates, and what does not

The wave surface, the swash edge and the wet-sand band all evolve with `t`.

**Foam does not pulse.** Breaking in this model is a *statistical* condition
(`Hs_local > γ_b·d`), not an instantaneous one, so the seeding region is steady
and the foam field sits near its equilibrium. Individual breaking events would
need a wave-by-wave model, which is well outside Phase 5. Foam is stepped
sequentially across the animation frames rather than cold-started per frame —
sequential generation is precisely the case where that is legitimate, since the
bounded spin-up machinery exists to support *random* access.

#### A note on what the test lake looks like

Undramatic, and correctly so. At 5 m/s over 1 km of fetch the waves are 8.6 cm,
the surf zone is a metre wide and foam coverage peaks around 0.27. The shoreline
clip shows a thin wet band and a modest white line, because that is what such a
lake does. `coastal_bay.yaml` — 1.43 m waves, a 48.6 m surf zone, plunging
breakers — is the scene to look at if you want to see the nearshore model
working hard.
