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
| 3 | Validation suite, `docs/validation_report.md` | Not started |
| 4–10 | Terrain, nearshore, mesh export, Mitsuba BSDF, emissivity, integration | Not started |

## Install

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

Python 3.11+. Runtime deps: `numpy`, `scipy`, `pyyaml`.

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
configs/
  test_lake.yaml  the reference scene: 5 m/s wind, 1 km fetch
docs/
  users_guide.md
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
