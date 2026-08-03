"""Regenerate the committed regression baseline.

    python tests/make_baseline.py

The baseline is the only thing that catches an accidental convention change --
a flipped FFT sign, a lost factor of N^2, a `flip_k` off-by-one -- during later
refactoring. Those changes are otherwise nearly undetectable: they leave the
variance, the spectrum and the slope statistics all correct.

**Regenerating it is a deliberate act.** If a test fails against the baseline,
the default assumption is that the code changed, not that the baseline is stale.
Only rerun this after confirming the new behaviour is the intended one, and say
so in the commit message.

The scene is defined here in code rather than loaded from `configs/`, so that
editing a shipped config cannot silently invalidate the regression record.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pywave import surface, tiling  # noqa: E402
from pywave.config import (  # noqa: E402
    Config,
    SceneConfig,
    SpectrumConfig,
    SurfaceConfig,
    TileConfig,
    WindConfig,
)

BASELINE_DIR = Path(__file__).resolve().parent / "baseline"

# -- the pinned scene --------------------------------------------------------
SEED = 20260801
TIME = 2.5
U10 = 5.0
FETCH = 1000.0
THETA_WIND = np.radians(45.0)
GAMMA = 3.3
TILE_SIZE = 64.0
TILE_N = 128
N_PROBES = 1024


def baseline_config() -> Config:
    """The composite scene, constructed in code so `configs/` cannot drift into it."""
    return Config(
        scene=SceneConfig(domain=(1000.0, 1000.0), water_level=100.0, epsg=32616),
        wind=WindConfig(speed=U10, direction_rad=THETA_WIND, fetch=FETCH),
        spectrum=SpectrumConfig(model="jonswap", gamma=GAMMA, spreading="cos2s", seed=SEED),
        surface=SurfaceConfig(
            tiles=(
                TileConfig(size=64.0, n=128, band=(0.0, 0.35)),
                TileConfig(size=37.0, n=64, band=(0.35, 0.7)),
                TileConfig(size=23.0, n=64, band=(0.7, 1.0)),
            ),
            choppiness=1.0,
        ),
    )


def probe_points() -> tuple[np.ndarray, np.ndarray]:
    """Fixed world coordinates at which the composite is sampled."""
    rng = np.random.default_rng(99)
    return rng.uniform(0.0, 250.0, N_PROBES), rng.uniform(0.0, 250.0, N_PROBES)


def build_tile_baseline() -> np.ndarray:
    """`(3, N, N)` stack of `h`, `slope_x`, `slope_y` for one pinned tile."""
    tile = surface.WaveTile.build(TILE_SIZE, TILE_N, U10, FETCH, THETA_WIND,
                                  seed=SEED, gamma=GAMMA)
    f = tile.evaluate(TIME)
    return np.stack([f.h, f.slope_x, f.slope_y])


def build_composite_baseline() -> np.ndarray:
    """`(5, N_PROBES)` stack of all five fields from the composite."""
    ts = tiling.TileSet.build(baseline_config())
    x, y = probe_points()
    f = ts.sample(x, y, TIME)
    return np.stack([f.h, f.dx_disp, f.dy_disp, f.slope_x, f.slope_y])


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    tile_arr = build_tile_baseline()
    comp_arr = build_composite_baseline()

    np.save(BASELINE_DIR / "tile.npy", tile_arr)
    np.save(BASELINE_DIR / "composite.npy", comp_arr)

    import pywave

    meta = {
        "generated": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC",
        "git_sha": _git_sha(),
        "pywave_version": pywave.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "scene": {
            "seed": SEED, "t": TIME, "u10": U10, "fetch": FETCH,
            "theta_wind_rad": float(THETA_WIND), "gamma": GAMMA,
        },
        "tile": {
            "size": TILE_SIZE, "n": TILE_N,
            "fields": ["h", "slope_x", "slope_y"],
            "shape": list(tile_arr.shape),
            "sha256": hashlib.sha256(tile_arr.tobytes()).hexdigest(),
            "hs": float(4.0 * np.std(tile_arr[0])),
        },
        "composite": {
            "n_probes": N_PROBES,
            "fields": ["h", "dx_disp", "dy_disp", "slope_x", "slope_y"],
            "shape": list(comp_arr.shape),
            "sha256": hashlib.sha256(comp_arr.tobytes()).hexdigest(),
            "hs": float(4.0 * np.std(comp_arr[0])),
        },
    }
    (BASELINE_DIR / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                encoding="utf-8")

    print(f"wrote {BASELINE_DIR / 'tile.npy'}       {tile_arr.shape} "
          f"({tile_arr.nbytes / 1024:.0f} KiB)")
    print(f"wrote {BASELINE_DIR / 'composite.npy'}  {comp_arr.shape} "
          f"({comp_arr.nbytes / 1024:.0f} KiB)")
    print(f"wrote {BASELINE_DIR / 'metadata.json'}")
    print(f"  tile Hs      {meta['tile']['hs']:.6f} m")
    print(f"  composite Hs {meta['composite']['hs']:.6f} m")


if __name__ == "__main__":
    main()
