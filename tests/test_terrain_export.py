"""PHASE 4 -- Gate 4 (partial): loading a terrain export.

Phase 4 itself is the Houdini side, which is not testable from here.  What *is*
testable, and what the whole synthetic-first strategy was for, is that the
loader accepts an export satisfying the section 4.5 contract and hands the rest
of the package something indistinguishable from a synthetic basin.

Most of this runs against a **round trip** -- write a synthetic basin out as an
export, read it back -- so the loader is exercised on a machine that has never
run Houdini.  The checks against a real export skip when one is not present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pywave import export, nearshore, tiling
from pywave.bathymetry import Bathymetry

pytestmark = pytest.mark.gate4

REPO = Path(__file__).resolve().parent.parent
REAL_EXPORT = REPO / "houdini_export"
has_real = pytest.mark.skipif(not (REAL_EXPORT / "grid_meta.json").exists(),
                              reason="no houdini_export/ present")


@pytest.fixture
def synthetic_export(tmp_path):
    """A synthetic basin written out in the Phase 4 format."""
    bathy = Bathymetry.dean_beach(nx=128, ny=192, dx=1.0, shoreline_y=150.0)
    export.write_terrain_export(bathy, tmp_path / "export")
    return bathy, tmp_path / "export"


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------


def test_export_round_trip_preserves_the_fields(record, synthetic_export):
    """Write a basin, read it back, get the same fields.

    float32 on disk, so depth carries a rounding error and the rest is exact.
    """
    original, directory = synthetic_export
    loaded = Bathymetry.from_export(directory)

    assert loaded.meta.shape == original.meta.shape
    assert loaded.meta.dx == original.meta.dx
    assert loaded.meta.origin == original.meta.origin
    assert loaded.meta.water_level == original.meta.water_level

    worst = max(float(np.abs(loaded.depth - original.depth).max()),
                float(np.abs(loaded.sdf - original.sdf).max()),
                float(np.abs(loaded.shore_normal - original.shore_normal).max()))
    record("4", "terrain export round trip, worst field error", worst, 0.0, 1e-5,
           unit="m", note="float32 on disk; the contract specifies it, and a "
                          "millimetre of rounding on a depth field is immaterial.",
           passed=worst < 1e-5)
    assert worst < 1e-5


def test_loaded_bathymetry_has_no_analytic_profile(synthetic_export):
    """`dean_a` is None after loading, which is what switches `beach_slope`.

    A loaded basin that kept a Dean parameter would evaluate a formula against
    terrain that does not follow it, and would be believed.
    """
    _, directory = synthetic_export
    loaded = Bathymetry.from_export(directory)
    assert loaded.dean_a is None
    assert loaded.bottom_type is not None
    assert loaded.summary()["source"] == "export"


def test_measured_slope_matches_the_analytic_one(record, tmp_path):
    """Measuring the bed must reproduce the profile it was built from.

    Regression test for a real bug: the slope band was an absolute 0.5 m around
    a 0.1 m target depth, which on a concave-up profile admits everything out to
    0.6 m depth -- tens of metres of seabed. The median then reported the slope
    at the median *distance offshore* rather than at the target depth, giving
    0.51x the right answer at every resolution, so it never looked like a
    discretisation artefact. The band is now relative to the target depth.
    """
    worst, rows = 0.0, []
    for dx in (1.0, 0.5, 0.25):
        ny = int(round(256.0 / dx))
        bathy = Bathymetry.dean_beach(nx=64, ny=ny, dx=dx,
                                      shoreline_y=0.9 * ny * dx)
        analytic = bathy.beach_slope()
        export.write_terrain_export(bathy, tmp_path / f"e{dx}")
        measured = Bathymetry.from_export(tmp_path / f"e{dx}",
                                          validate=False).beach_slope()
        rel = abs(measured - analytic) / analytic
        worst = max(worst, rel)
        rows.append(f"dx={dx}: {measured:.4f} vs {analytic:.4f}")

    record("4", "measured vs analytic foreshore slope (worst)", worst, 0.0, 0.20,
           note="Dean beach, so the analytic answer is exact. " + "; ".join(rows)
                + ". Scatter is discretisation; a systematic factor would mean "
                  "the measurement is measuring the wrong thing.",
           passed=worst < 0.20)
    assert worst < 0.20


def test_shore_normal_accepted_in_either_layout(tmp_path, synthetic_export):
    """`(2, ny, nx)` and `(ny, nx, 2)` both load, and agree.

    The section 0.4 table writes one and this package stores the other. Guessing
    would transpose the normals on half the exports in existence, rotating every
    refraction angle by ninety degrees -- which no scalar check would notice.
    """
    original, directory = synthetic_export
    a = Bathymetry.from_export(directory)

    other = tmp_path / "transposed"
    other.mkdir()
    for name in ("terrain_z", "depth", "sdf", "bottom_type"):
        np.save(other / f"{name}.npy", np.load(directory / f"{name}.npy"))
    (other / "grid_meta.json").write_text(
        (directory / "grid_meta.json").read_text(encoding="utf-8"), encoding="utf-8")
    np.save(other / "shore_normal.npy",
            np.moveaxis(np.load(directory / "shore_normal.npy"), 0, -1))

    b = Bathymetry.from_export(other)
    assert np.array_equal(a.shore_normal, b.shore_normal)


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------


def test_missing_files_are_named(tmp_path):
    """A partial export fails immediately, saying what a full one needs."""
    with pytest.raises(FileNotFoundError, match="grid_meta.json"):
        Bathymetry.from_export(tmp_path)

    (tmp_path / "grid_meta.json").write_text(json.dumps(
        {"x0": 0, "y0": 0, "dx": 1, "dy": 1, "nx": 4, "ny": 4, "z_w": 0}))
    with pytest.raises(FileNotFoundError, match="terrain_z.npy"):
        Bathymetry.from_export(tmp_path)


def test_non_square_cells_are_refused(tmp_path, synthetic_export):
    """`dx != dy` is rejected rather than silently distorting every gradient."""
    _, directory = synthetic_export
    bad = tmp_path / "rect"
    bad.mkdir()
    for f in directory.glob("*.npy"):
        np.save(bad / f.name, np.load(f))
    meta = json.loads((directory / "grid_meta.json").read_text(encoding="utf-8"))
    meta["dy"] = meta["dx"] * 2.0
    (bad / "grid_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="square cells"):
        Bathymetry.from_export(bad)


def test_inconsistent_terrain_and_depth_are_refused(tmp_path, synthetic_export):
    """`terrain_z` and `z_w - depth` must agree, or one of them is stale."""
    _, directory = synthetic_export
    bad = tmp_path / "stale"
    bad.mkdir()
    for f in directory.glob("*.npy"):
        np.save(bad / f.name, np.load(f))
    (bad / "grid_meta.json").write_text(
        (directory / "grid_meta.json").read_text(encoding="utf-8"), encoding="utf-8")
    np.save(bad / "terrain_z.npy", np.load(bad / "terrain_z.npy") + 0.5)

    with pytest.raises(ValueError, match="disagrees"):
        Bathymetry.from_export(bad)


def test_shape_mismatch_is_refused(tmp_path, synthetic_export):
    """Arrays must match `grid_meta`, and the message says to check [y, x]."""
    _, directory = synthetic_export
    bad = tmp_path / "shape"
    bad.mkdir()
    for f in directory.glob("*.npy"):
        np.save(bad / f.name, np.load(f))
    (bad / "grid_meta.json").write_text(
        (directory / "grid_meta.json").read_text(encoding="utf-8"), encoding="utf-8")
    np.save(bad / "depth.npy", np.load(bad / "depth.npy").T)

    with pytest.raises(ValueError, match=r"depth\.npy has shape"):
        Bathymetry.from_export(bad)


# ---------------------------------------------------------------------------
# The real export, when it is there
# ---------------------------------------------------------------------------


@has_real
def test_real_export_satisfies_the_contract(record):
    """The shipped Houdini export passes the same checks as synthetic data."""
    bathy = Bathymetry.from_export(REAL_EXPORT)     # validate=True by default
    s = bathy.summary()

    record("4", "real export grid", f"{s['shape'][0]}x{s['shape'][1]} @ {s['dx']} m",
           note=f"extent {[round(v) for v in s['extent']]} m, z_w = "
                f"{s['water_level']} m, EPSG {s['epsg']}. Water covers "
                f"{100 * s['water_fraction']:.1f}% of the domain, max depth "
                f"{s['max_depth']:.1f} m.")
    record("4", "real export foreshore slope", s["foreshore_slope"],
           note="Measured from the bed, not assumed. The synthetic test lake is "
                "0.067 and the cookbook assumes ~0.05, so this shore is far "
                "steeper and lands in a different breaker regime.")

    assert s["source"] == "export"
    assert bathy.dean_a is None
    assert 0.0 < s["water_fraction"] < 1.0
    assert s["foreshore_slope"] > 0.0


@has_real
def test_real_export_drives_the_nearshore_physics(record, cfg):
    """Shoaling, refraction and breaking all run on real terrain.

    The point of Phase 5 having been built against a synthetic profile: nothing
    here is a new code path, only new data.
    """
    from dataclasses import replace

    bathy = Bathymetry.from_export(REAL_EXPORT)
    scene_cfg = replace(cfg, scene=replace(cfg.scene,
                                           water_level=bathy.meta.water_level))
    ts = tiling.TileSet.build(scene_cfg)

    x0, x1, y0, y1 = bathy.meta.extent
    xa, ya = bathy.meta.axes()
    X, Y = np.meshgrid(xa[::16], ya[::16], indexing="xy")

    nf = nearshore.transform(ts, bathy, scene_cfg, X.ravel(), Y.ravel(), 0.0)

    assert np.all(np.isfinite(nf.surface.h))
    assert np.all(np.isfinite(nf.hs_local))
    assert np.all(nf.hs_local >= 0.0)

    wet = nf.depth > 0.0
    assert wet.any(), "no wet samples on the real terrain"
    saturated = nf.breaking & wet
    if saturated.any():
        ratio = nf.hs_local[saturated] / (scene_cfg.nearshore.breaker_index
                                          * nf.depth[saturated])
        assert np.allclose(ratio, 1.0, atol=1e-6), "depth limiter not applied"

    slope = bathy.beach_slope()
    l0 = float(nearshore.deep_water_wavelength(1.0 / scene_cfg.f_p))
    xi = float(nearshore.iribarren_number(slope, ts.hs(), l0))
    record("4", "Iribarren number on the real terrain", xi,
           note=f"Breaker type: {nearshore.breaker_type(xi)}. The synthetic lake "
                f"gives 0.33 (spilling) on a 6.7% foreshore; this shore is "
                f"{100 * slope:.0f}% and lands in a different regime.")
    assert xi > 0.0


@has_real
def test_surf_zone_is_sub_cell_on_the_real_export(record, cfg):
    """State the resolution limit rather than letting it surprise someone.

    A steep foreshore makes the surf zone narrow, and a 1 m export cannot
    resolve it. That is not a defect -- at 1 m GSD the surf zone is sub-pixel
    anyway, which is why breaking is carried as coverage rather than geometry --
    but it does mean a transect through the breaker line has nothing to show.
    """
    bathy = Bathymetry.from_export(REAL_EXPORT)
    ts = tiling.TileSet.build(cfg)

    slope = bathy.beach_slope()
    d_break = float(nearshore.breaker_depth(ts.hs(), cfg.nearshore.breaker_index))
    surf_width = d_break / slope
    cells = surf_width / bathy.meta.dx

    record("4", "surf zone width on the real terrain", surf_width, unit="m",
           note=f"= breaking depth {d_break:.3f} m / foreshore slope "
                f"{slope:.3f}. At the export's {bathy.meta.dx:g} m posts that is "
                f"{cells:.2f} cells, so one cell spans "
                f"{bathy.meta.dx * slope / d_break:.1f}x the breaking depth. "
                f"Resolving it would need finer bathymetry near the shore.")
    assert surf_width > 0.0


# ---------------------------------------------------------------------------
# Scene-level consistency
# ---------------------------------------------------------------------------


def test_water_level_mismatch_is_refused(record, tmp_path, cfg):
    """`scene.water_level` and the export's `z_w` must agree.

    If they do not, every mesh is built at the config's value while every depth
    is measured from the export's. Water and terrain then stay consistent *with
    each other* and both sit at the wrong absolute height -- which looks perfect
    in isolation and is metres out against anything else in the world. Measured
    at a deliberate 50 m mismatch: both meshes shifted by exactly 50 m.
    """
    from dataclasses import replace

    bathy = Bathymetry.dean_beach(nx=64, ny=64, dx=4.0, shoreline_y=200.0,
                                  water_level=100.0)
    export.write_terrain_export(bathy, tmp_path / "e")

    from pywave.config import BathymetryConfig

    good = replace(cfg, scene=replace(cfg.scene, water_level=100.0, epsg=32616),
                   bathymetry=BathymetryConfig(source=str(tmp_path / "e")))
    Bathymetry.from_config(good)          # agrees: fine

    bad = replace(good, scene=replace(good.scene, water_level=50.0))
    with pytest.raises(ValueError, match="scene.water_level"):
        Bathymetry.from_config(bad)

    bad_crs = replace(good, scene=replace(good.scene, epsg=4326))
    with pytest.raises(ValueError, match="epsg"):
        Bathymetry.from_config(bad_crs)

    record("4", "water level / CRS agreement between scene and export",
           "enforced", "enforced",
           note="A silent mismatch is self-consistent between the two meshes "
                "and wrong against the rest of the world, so it is refused "
                "rather than warned about.")


def test_terrain_with_no_shoreline_is_refused(tmp_path, cfg):
    """An all-water or all-land export fails with an explanation.

    Every window, transect and camera is positioned relative to a waterline. An
    export with none is almost always a mistake in the terrain, and returning
    some arbitrary corner -- which is what an unguarded search does -- produces
    a mesh that is not wrong so much as meaningless.
    """
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    import make_figures as mf

    from dataclasses import replace

    from pywave.config import BathymetryConfig

    n, dx, zw = 48, 4.0, 100.0
    depth = np.full((n, n), 5.0)                      # wet everywhere
    sdf = np.full((n, n), -20.0)
    normal = np.zeros((2, n, n))
    normal[1] = 1.0
    directory = tmp_path / "allwater"
    directory.mkdir()
    for name, arr in (("terrain_z", zw - depth), ("depth", depth), ("sdf", sdf)):
        np.save(directory / f"{name}.npy", arr.astype(np.float32))
    np.save(directory / "shore_normal.npy", normal.astype(np.float32))
    (directory / "grid_meta.json").write_text(json.dumps(
        {"x0": 0.0, "y0": 0.0, "dx": dx, "dy": dx, "nx": n, "ny": n,
         "z_w": zw, "epsg": 32616}), encoding="utf-8")

    # Two layers catch it, and the loader gets there first: with no waterline
    # there is no inland direction, so the shore-normal check cannot pass.
    with pytest.raises(AssertionError, match="shore_normal"):
        Bathymetry.from_export(directory)

    # And if validation is skipped, the shoreline search says so plainly rather
    # than returning some arbitrary corner.
    bathy = Bathymetry.from_export(directory, validate=False)
    assert (bathy.depth > 0).all()

    scene_cfg = replace(cfg, scene=replace(cfg.scene, water_level=zw, epsg=32616),
                        bathymetry=BathymetryConfig(source=str(directory)))
    scene = mf.Scene(scene_cfg)
    scene._cache["bathy"] = bathy          # bypass the loader's own refusal
    with pytest.raises(ValueError, match="no shoreline"):
        _ = scene.shore_ref
