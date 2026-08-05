"""Does the water surface stay above the bed? Ask the two meshes, not the model.

Every clearance guarantee inside ``pywave`` is about the water surface and the
*depth field*. A renderer sees neither. It sees two triangle meshes, and those
agree with the field only as closely as their own resolution allows. The moment
the bed mesh comes from somewhere else -- exported straight out of Houdini,
possibly at a resolution unrelated to the ``.npy`` fields -- the guarantee stops
being structural and becomes something to measure.

    python scripts/check_clearance.py runs/houdini_lake/mesh/water_0000.ply \
                                      runs/houdini_lake/mesh/terrain_0000.ply
    python scripts/check_clearance.py water.ply /path/from/houdini/terrain.ply

Measured on the shipped 1 m lake export, water and bed both meshed at 0.5 m:

    bed mesh built from       water vertices below it      worst
    ----------------------    -------------------------    --------
    the same 1 m fields       0                            --
    fields decimated to 2 m   0.24%                        0.70 m
    fields decimated to 4 m   0.69%                        1.97 m

The pattern is not about the meshes' spacings. It is about *information*: the
water's height limit is computed from the fields, so any bed detail the fields
never saw is detail the water does not know to stay above. A bed mesh finer than
the fields will show through. Keep the ``.npy`` export at least as fine as the
mesh you intend to render, and run this to confirm it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pywave import export as pw_export  # noqa: E402


def as_regular_grid(terrain):
    """Recognise a bed that is a lattice, and return ``(Z, x0, y0, dx)``.

    Worth the check: both this package's terrain mesh and a Houdini heightfield
    export are lattices, and bilinear sampling of one is seconds where general
    point location in a few million triangles is many minutes. ``None`` when the
    vertices are not a lattice, which sends the caller to the slow path.
    """
    x, y, z = terrain["x"], terrain["y"], terrain["z"]
    ux, uy = np.unique(x), np.unique(y)
    if len(ux) < 2 or len(uy) < 2 or len(ux) * len(uy) != len(x):
        return None
    dx, dy = np.diff(ux), np.diff(uy)
    if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])
            and np.isclose(dx[0], dy[0])):
        return None
    Z = np.full((len(uy), len(ux)), np.nan)
    Z[np.searchsorted(uy, y), np.searchsorted(ux, x)] = z
    if np.isnan(Z).any():                       # a lattice with holes in it
        return None
    return Z, ux[0], uy[0], float(dx[0])


def sample_bed(terrain, faces, x, y):
    """Bed elevation under each (x, y), by point location in the bed mesh."""
    grid = as_regular_grid(terrain)
    if grid is not None:
        from scipy.ndimage import map_coordinates

        Z, x0, y0, dx = grid
        print(f"  bed is a regular {Z.shape[0]}x{Z.shape[1]} lattice at "
              f"{dx:g} m -- sampling it directly")
        rows, cols = (y - y0) / dx, (x - x0) / dx
        z = map_coordinates(Z, np.stack([rows, cols]), order=1, mode="nearest")
        inside = ((rows >= -0.5) & (rows <= Z.shape[0] - 0.5)
                  & (cols >= -0.5) & (cols <= Z.shape[1] - 0.5))
        return np.where(inside, z, np.nan)

    # General mesh. matplotlib's trapezoid-map finder takes the triangulation as
    # given rather than computing its own, which matters: a Delaunay
    # retriangulation would not reproduce the connectivity the renderer sees.
    from matplotlib.tri import LinearTriInterpolator, Triangulation

    print(f"  bed is an unstructured mesh -- building a point-location "
          f"structure over {len(faces):,} triangles, this is the slow path")
    tri = Triangulation(terrain["x"], terrain["y"], faces)
    z = LinearTriInterpolator(tri, terrain["z"])(x, y)
    return np.ma.filled(z, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("water", type=Path, help="water surface PLY")
    ap.add_argument("terrain", type=Path,
                    help="bed PLY -- from this package or from a DCC tool")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="check a random N vertices instead of all of them")
    ap.add_argument("--tolerance", type=float, default=0.0, metavar="M",
                    help="clearance below this counts as an intrusion [m]")
    args = ap.parse_args()

    water, _ = pw_export.read_ply_any(args.water)
    terrain, tfaces = pw_export.read_ply_any(args.terrain)
    print(f"water   {len(water['x']):,} vertices  {args.water}")
    print(f"terrain {len(terrain['x']):,} vertices, {len(tfaces):,} triangles"
          f"  {args.terrain}")

    for name, m in (("water", water), ("terrain", terrain)):
        print(f"  {name:7s} x {m['x'].min():9.2f} {m['x'].max():9.2f}   "
              f"y {m['y'].min():9.2f} {m['y'].max():9.2f}   "
              f"z {m['z'].min():8.2f} {m['z'].max():8.2f}")

    x, y, z = water["x"], water["y"], water["z"]
    # 'depth' is this package's own channel; a foreign water mesh may lack it,
    # in which case every vertex is treated as wet.
    wet = water["depth"] > 0.0 if "depth" in water else np.ones(len(x), bool)
    idx = np.flatnonzero(wet)
    if args.sample and args.sample < len(idx):
        idx = np.random.default_rng(0).choice(idx, args.sample, replace=False)
        print(f"\nsampling {len(idx):,} of {int(wet.sum()):,} wet vertices")

    print("\nlocating water vertices in the bed mesh ...", flush=True)
    bed = sample_bed(terrain, tfaces, x[idx], y[idx])

    outside = ~np.isfinite(bed)
    if outside.any():
        print(f"  WARNING: {outside.sum():,} water vertices "
              f"({100 * outside.mean():.2f}%) lie outside the bed mesh. The bed "
              f"must cover the water, or the surface has nothing to sit on.")
    ok = ~outside
    if not ok.any():
        print("\nthe two meshes do not overlap at all -- check their extents "
              "and coordinate systems above.")
        return 2

    clear = z[idx][ok] - bed[ok]
    below = clear < -args.tolerance
    q = np.percentile(clear, [0.1, 1, 50])
    print(f"\nclearance over {ok.sum():,} vertices")
    print(f"  min {clear.min():+.5f} m   0.1st pct {q[0]:+.4f}   "
          f"1st pct {q[1]:+.4f}   median {q[2]:+.3f}")
    print(f"  below the bed: {below.sum():,}  ({100 * below.mean():.4f}%)")

    if below.any():
        d = -clear[below]
        print(f"  intrusion depth: median {np.median(d):.4f} m, "
              f"worst {d.max():.4f} m")
        if "depth" in water:
            dep = water["depth"][idx][ok][below]
            print(f"  at still-water depths {dep.min():.3f}-{dep.max():.3f} m")
            print("  (intrusions clustered near the waterline point at a "
                  "resolution mismatch; ones in deep water point at a "
                  "coordinate or water_level mismatch instead)")
        print("\nFAIL -- the bed shows through the water surface.")
        return 1

    print("\nOK -- the water surface is above the bed everywhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
