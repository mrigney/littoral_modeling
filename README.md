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
| 4 | Houdini terrain and lake basin | Not started — `bathymetry.py` supplies a synthetic Dean profile satisfying the same field contract |
| 5 | Shoaling, refraction, breaking, foam | **Implemented** against that synthetic profile |
| 6–10 | Mesh export, Mitsuba BSDF, emissivity, EMBER integration | Not started |

**New here?** [docs/gallery.md](docs/gallery.md) explains the whole model in
eight figures, with no code required.

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

Takes about 40 s for a small lake, a couple of minutes for a large coastal scene.
Add `--quick` to skip the channel export and the HTML page.

Two example scenes ship, and they are deliberately unlike each other —
[`test_lake.yaml`](configs/test_lake.yaml) (5 m/s, 1 km fetch → 8.6 cm waves,
spilling breakers, 0.4 m swash) and [`coastal_bay.yaml`](configs/coastal_bay.yaml)
(12 m/s, 40 km fetch, embayed shore → 1.43 m waves, plunging breakers, 7 m
swash). Every figure, number and channel is derived from the config, so a
disagreement with the validation report means the code changed — not that the
picture went stale.

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
pytest                    # 72 checks, ~50 s
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

- **[docs/users_guide.md](docs/users_guide.md)** — full user's guide: concepts,
  conventions, config reference, module reference, recipes, known gotchas.
- `water-surface-modeling-primer.md` — the physics rationale.
- `littoral-water-implementation-cookbook.md` — phase plan and gate checklists.

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
tests/
  test_spectrum.py        PHASE 3: Gate 1 checks
  test_surface.py         PHASE 3: Gate 2 checks
  test_reproducibility.py PHASE 3: Gate 3, determinism + regression baseline
  test_nearshore.py       PHASE 5: Gate 5 checks
  make_baseline.py        regenerates tests/baseline/ (a deliberate act)
scripts/
  run_scene.py     config -> every artifact, in one directory
  animate.py       open-water and shoreline clips
  make_figures.py  the eight figures + a markdown gallery
  make_overview.py one self-contained shareable HTML page
configs/
  test_lake.yaml    reference scene: 5 m/s wind, 1 km fetch, straight beach
  coastal_bay.yaml  contrasting scene: 12 m/s, 40 km fetch, embayed shore
docs/
  gallery.md            the model in eight figures
  users_guide.md
  validation_report.md  generated by pytest -- the V&V artifact
```

## Reference scene

`configs/test_lake.yaml` — U10 = 5 m/s, fetch = 1000 m, deep water:

```
Tp     1.045 s        lambda_p  1.705 m
Hs     0.0857 m       mss       0.0461  (RMS slope 12.1 deg)
```

The three-tile composite realises Hs = 0.0839 m (−1.6%) and closes the LOD
invariant `mss_resolved + mss_above(k_max) = mss_total` to 0.01%. Full numbers in
the [user's guide](docs/users_guide.md#9-reference-numbers--the-test-lake).
