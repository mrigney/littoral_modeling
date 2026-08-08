"""Measure Gate 5b on a scene, at whatever size that scene really is.

    python scripts/gate5b.py configs/straits.yaml
    python scripts/gate5b.py configs/straits_crop.yaml --decimate 32
    python scripts/gate5b.py configs/straits.yaml --no-wind-sea   # the A/B

Why this is a script and not a test
-----------------------------------
Gate 5b.2 and 5b.3 are stated on the full 7.5 x 8.6 km export, and a solve at
that size is minutes, not seconds. `tests/test_rays.py` therefore runs the same
checks on the 701 m crop, where they are affordable but where the shadows are
too short to be the hard case.

That leaves the numbers quoted in `docs/phase5b_refraction.md` for the full
export as numbers somebody typed. This regenerates them, so they can be argued
with.

Reading the output
------------------
`--no-wind-sea` turns off the locally generated sea and leaves pure ray theory,
which is the comparison that shows what the floor is for. On the straits export
the two runs differ by everything on the 5b.3 line and by nothing at all on the
5b.2 line, which is the claim worth being able to check.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pywave import load_config, rays                       # noqa: E402
from pywave.bathymetry import Bathymetry                   # noqa: E402

GATE_LAG_M = 4.0
"""The separation the continuity gate is stated at [m]. See the note below."""


def lag_jumps(gain, wet, lag):
    """`|dgain|` over `lag` posts, wherever every post in between is also wet.

    Two rules, and the second is the one that is easy to get wrong:

    * **Fix the separation, not the cell.** A "one-cell jump" is not a number
      until the cell size is fixed. The reference figures are at the full
      export's 4 m posts, so a 0.25 m export must be sampled 16 posts apart or
      it will pass anything.
    * **Require water the whole way between.** On a ragged shoreline two cells
      4 m apart are routinely both wet with land in between. That pair is a
      shoreline, not a discontinuity in the wave field.
    """
    out = []
    for axis in (0, 1):
        n = wet.shape[axis]
        run = None
        for k in range(lag + 1):
            s = [slice(None)] * 2
            s[axis] = slice(k, n - lag + k)
            run = wet[tuple(s)] if run is None else (run & wet[tuple(s)])
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[axis] = slice(None, n - lag)
        hi[axis] = slice(lag, None)
        out.append(np.abs(gain[tuple(lo)] - gain[tuple(hi)])[run])
    return np.concatenate(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", nargs="?", default="configs/straits.yaml")
    p.add_argument("--decimate", type=int, default=4,
                   help="solve on every Nth post (default 4)")
    p.add_argument("--dirs", type=int, default=15, help="fan directions")
    p.add_argument("--rays", type=int, default=1500, help="rays per direction")
    p.add_argument("--smooth", type=float, default=80.0,
                   help="deposition kernel sigma [m]")
    p.add_argument("--no-wind-sea", action="store_true",
                   help="pure ray theory, no locally generated sea")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    if not cfg.bathymetry.is_export:
        p.error(f"{a.config} describes a synthetic basin. Gate 5b is stated on "
                f"a terrain export -- a synthetic coast is where the closed "
                f"form still works, so it cannot fail these checks and passing "
                f"them there would mean nothing.")
    bathy = Bathymetry.from_config(cfg)

    wet = bathy.depth > 0.0
    print(f"scene       {a.config}")
    print(f"grid        {bathy.depth.shape} at {bathy.meta.dx:g} m, "
          f"{wet.mean():.1%} water, max depth {bathy.depth.max():.1f} m")
    print(f"sea state   Tp {1 / cfg.f_p:.2f} s, toward "
          f"{np.degrees(cfg.wind.direction_rad):.0f} deg, "
          f"U10 {cfg.wind.speed:g} m/s, fetch {cfg.wind.fetch:g} m")
    print(f"solve       decimate {a.decimate}, {a.dirs} dirs x {a.rays} rays, "
          f"sigma {a.smooth:g} m, wind sea {not a.no_wind_sea}", flush=True)

    t = time.perf_counter()
    rf = rays.RayField.solve(bathy, cfg, cfg.omega_p, decimate=a.decimate,
                             n_dirs=a.dirs, rays_per_dir=a.rays,
                             smooth_m=a.smooth, wind_sea=not a.no_wind_sea)
    elapsed = time.perf_counter() - t

    lag = max(int(round(GATE_LAG_M / bathy.meta.dx)), 1)
    j = lag_jumps(rf.gain, wet, lag)
    g = rf.gain[wet]

    def row(gate, name, measured, criterion, ok):
        print(f"  {gate:<5} {name:<34} {measured:>12}  {criterion:<16} "
              f"{'PASS' if ok else 'FAIL'}")

    print(f"\n  solve grid  {rf.meta['solve_dx']:g} m, "
          f"{rf.meta['mean_visits']:.0f} ray visits/cell, "
          f"{rf.meta['sheltered_cells']} of {rf.meta['wet_cells']} cells "
          f"sheltered\n")
    print(f"  {'gate':<5} {'check':<34} {'measured':>12}  {'criterion':<16}")
    row("5b.2", f"p99.9 gain jump over {GATE_LAG_M:g} m",
        f"{np.percentile(j, 99.9):.4f}", "< 0.02", np.percentile(j, 99.9) < 0.02)
    row("", f"  p99 / worst",
        f"{np.percentile(j, 99):.4f}/{j.max():.4f}", "--", True)
    row("5b.3", "min gain over wet cells", f"{g.min():.4f}", "> 0.05",
        g.min() > 0.05)
    row("", "  fraction under 0.05", f"{(g < 0.05).mean():.4%}", "--", True)
    row("5b.9", "solve wall time", f"{elapsed:.1f} s", "recorded", True)
    print(f"\n  median gain over wet water {np.median(g):.4f}; "
          f"deep-water median is 1 by construction, this one is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
