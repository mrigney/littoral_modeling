"""PHASE 6 -- Gate 6: mesh generation, channel packing and export.

Two of the cookbook's Gate 6 items concern LOD rings, which are deliberately not
built (see `pywave.mesh`). Everything else applies, and the PLY round-trip in
particular is the five-minute check section 6.6 asks for before Phase 7 is bet
on the format carrying custom attributes.
"""

from __future__ import annotations

import numpy as np
import pytest

from pywave import channels as ch
from pywave import export, mesh, moments
from pywave.bathymetry import Bathymetry

pytestmark = pytest.mark.gate6

REGION = (400.0, 380.0, 460.0, 402.0)
DX = 0.25


@pytest.fixture(scope="module")
def meshed(cfg):
    """One mesh, reused: building it is the expensive part of this file."""
    from pywave import tiling

    ts = tiling.TileSet.build(cfg)
    bathy = Bathymetry.from_config(cfg, fine=True)
    m = mesh.build_water_mesh(ts, bathy, cfg, t=1.25, dx=DX, region=REGION)
    return m, ts, bathy


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_mesh_is_structurally_sound(record, meshed):
    """Finite everywhere, unit normals, no degenerate or out-of-range faces."""
    m, _, _ = meshed
    stats = m.validate()

    record("6", "mesh vertices", stats["n_vertices"],
           note=f"{stats['n_faces']:,} triangles at {DX} m post spacing over a "
                f"{REGION[2] - REGION[0]:.0f} x {REGION[3] - REGION[1]:.0f} m region.")
    record("6", "max |‖normal‖ - 1|", stats["normal_unit_error"], 0.0, 1e-5,
           passed=stats["normal_unit_error"] < 1e-5)
    record("6", "min normal z component", stats["min_normal_z"],
           note="Must stay positive; a downward normal means the surface folded "
                "through itself.",
           passed=stats["min_normal_z"] > 0.0)
    assert stats["min_normal_z"] > 0.0


def test_mesh_covers_the_water_and_stops(record, meshed, cfg):
    """Every wet post is meshed, and the mesh stops within the swash margin.

    Both halves matter. Gaps leave holes at the waterline; overshoot puts water
    geometry up the beach where it will z-fight or float.
    """
    m, ts, bathy = meshed
    x0, y0, x1, y1 = REGION
    xs = np.arange(x0, x1 + 1e-9, DX)
    ys = np.arange(y0, y1 + 1e-9, DX)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    depth, sdf, _ = bathy.sample(X, Y)

    margin = mesh._swash_margin(ts, bathy, cfg)
    wet = depth > 0.02
    meshed_mask = mesh.water_extent_mask(depth, DX, 0.02, margin)

    missing = int(np.sum(wet & ~meshed_mask))
    overshoot = float(sdf[meshed_mask].max())

    record("6", "wet posts not meshed", missing, 0.0, 0.0,
           note="A gap here shows up as a hole at the waterline.",
           passed=missing == 0)
    record("6", "furthest inland meshed post (sdf)", overshoot, unit="m",
           note=f"Swash margin is {margin:.2f} m; the mesh is dilated landward "
                f"by that much so the swash band has geometry to live on.",
           passed=overshoot <= margin + 2 * DX)

    assert missing == 0
    assert overshoot <= margin + 2 * DX


def test_no_z_fighting_with_terrain(record, meshed, cfg):
    """Where the mesh runs onto land it sits *below* the bed, not through it.

    The margin puts water geometry slightly inland by design. That is only safe
    if it ends up buried: coincident surfaces are what z-fighting is.
    """
    m, _, bathy = meshed
    rest_depth = m.channels["depth"]
    onshore = rest_depth < 0.0
    assert onshore.any(), "no onshore vertices; the margin test is not exercising anything"

    terrain_z = cfg.scene.water_level - rest_depth[onshore]
    gap = terrain_z - m.vertices[onshore, 2]
    worst = float(gap.min())

    record("6", "smallest terrain-minus-mesh clearance onshore", worst, unit="m",
           note=f"{int(onshore.sum())} vertices sit landward of the waterline. "
                f"Positive means the bed is above the water mesh, so the water "
                f"is hidden rather than fighting for the same pixels.",
           passed=worst >= 0.0)
    assert worst >= 0.0


def test_analytic_normals_differ_from_face_normals_but_not_wildly(record, meshed):
    """A sanity band, not equality -- they *should* differ.

    Face normals are a finite difference of a field known exactly, and they lose
    the sub-cell content the analytic slopes keep. Agreement to machine
    precision would mean the analytic normals had been silently replaced by
    geometric ones; agreement to nothing at all would mean a convention error.
    """
    m, _, _ = meshed
    face_n = m.face_normals()
    vert_n = m.normals[m.faces].mean(axis=1)
    vert_n /= np.linalg.norm(vert_n, axis=1, keepdims=True)

    ang = np.degrees(np.arccos(np.clip(np.sum(face_n * vert_n, axis=1), -1.0, 1.0)))
    record("6", "analytic vs face normal angle, mean", float(ang.mean()), unit="deg",
           note=f"p95 {np.percentile(ang, 95):.1f} deg, max {ang.max():.1f} deg. "
                f"At {DX} m posts the mesh barely resolves the finest spectral "
                f"band, so a few degrees of disagreement is the expected cost of "
                f"differencing rather than a defect.",
           passed=float(ang.mean()) < 15.0)
    assert 0.05 < float(ang.mean()) < 15.0


# ---------------------------------------------------------------------------
# Channels and the LOD invariant
# ---------------------------------------------------------------------------


def test_lod_invariant_holds_at_the_mesh_spacing(record, meshed, cfg, scene):
    """`mss_resolved(dx) + mss_above(pi/dx) = mss_total` at this mesh.

    With one spacing everywhere this is a single check rather than the
    per-vertex bookkeeping section 6.4 describes, but it is the same identity
    and the same failure if it is skipped: distant water rendered mirror-smooth
    because its roughness was baked at the fine spacing.
    """
    u10, fetch, gamma, _ = scene
    m, _, _ = meshed

    total = moments.mss_above(0.0, u10, fetch, gamma=gamma)
    sub = ch.submesh_mss(cfg, DX)
    resolved = moments.mss_between(moments.K_MIN_DEFAULT, np.pi / DX, u10, fetch,
                                   gamma=gamma, order=2)
    rel = abs(resolved + sub - total) / total

    record("6", "LOD invariant at the mesh spacing", resolved + sub, total, 0.01,
           note=f"Mesh carries {100 * resolved / total:.0f}% of the slope "
                f"variance as geometry; the BSDF gets the remaining "
                f"{100 * sub / total:.0f}% as roughness "
                f"(Beckmann alpha = {np.sqrt(sub):.4f}).",
           passed=rel < 0.01)
    assert rel < 0.01
    # Offshore vertices sit at the deep-water value; inshore ones exceed it.
    assert m.channels["mss"].max() >= sub * (1.0 - 1e-6)
    assert abs(float(np.median(m.channels["mss"])) - sub) / sub < 0.05


def test_submesh_mss_tracks_depth_in_the_surf_band(record, cfg):
    """The sub-mesh share is a scene constant offshore and is not, inshore.

    The tempting simplification -- "sub-mesh waves are short, so they are always
    deep-water waves" -- is true past `kd ~ 2.5` and badly wrong below it, which
    is precisely the surf and swash band. This pins both halves of that.
    """
    dx = cfg.output.mesh_dx
    k_cut = np.pi / dx
    deep = ch.submesh_mss(cfg, dx)

    depths = np.array([0.03, 0.05, 0.10, 0.20, 1.0, 5.0])
    got = ch.submesh_mss(cfg, dx, depths)
    direct = np.array([moments.mss_above(k_cut, cfg.wind.speed, cfg.wind.fetch,
                                         depth=float(d), gamma=cfg.spectrum.gamma)
                       for d in depths])

    interp_err = float(np.max(np.abs(got - direct) / direct))
    shallow_lift = float(got[0] / deep - 1.0)
    offshore_err = float(np.max(np.abs(got[-2:] - deep) / deep))

    record("6", "lookup vs direct quadrature for sub-mesh mss", interp_err, 0.0,
           0.02,
           note="`mss_above` is a radial quadrature, far too slow per vertex, so "
                "it is interpolated over a log-spaced depth table. This bounds "
                "the interpolation error against calling it directly.",
           passed=interp_err < 0.02)
    record("6", "sub-mesh mss at 3 cm depth vs deep water", shallow_lift,
           note=f"kd = {k_cut * 0.03:.2f} there. Treating the sub-mesh share as "
                f"a scene constant would understate the roughness handed to the "
                f"BSDF by this much, in the one band anyone looks at.",
           passed=shallow_lift > 0.1)
    record("6", "sub-mesh mss at 1-5 m depth vs deep water", offshore_err, 0.0,
           1e-6,
           note="Converged past kd ~ 2.5, so the common case costs nothing.",
           passed=offshore_err < 1e-6)

    assert interp_err < 0.02
    assert shallow_lift > 0.1
    assert offshore_err < 1e-6
    # Monotone non-increasing with depth, and strictly decreasing while the
    # bottom is still felt. Past convergence successive values are *exactly*
    # equal, so a strict test would fail there for the right reason.
    assert np.all(np.diff(got) <= 0), "shallower water must mean more sub-mesh slope"
    assert np.all(np.diff(got[depths < 10.0 / k_cut]) < 0)


def test_channels_satisfy_the_contract(record, meshed):
    """Every section 0.4 per-vertex property is present and in range."""
    m, _, _ = meshed
    required = {"mss", "wdir_x", "wdir_y", "aniso", "depth", "foam"}
    assert required <= set(m.channels), f"missing {required - set(m.channels)}"

    wd = np.hypot(m.channels["wdir_x"], m.channels["wdir_y"])
    worst = float(np.max(np.abs(wd - 1.0)))
    record("6", "max ‖wave direction‖ - 1", worst, 0.0, 1e-5,
           note="Unit by construction; a drift here would tilt the anisotropy "
                "frame in Phase 7.",
           passed=worst < 1e-5)

    assert worst < 1e-5
    assert np.all(m.channels["foam"] >= 0.0) and np.all(m.channels["foam"] <= 1.0)
    assert np.all(m.channels["wetness"] >= 0.0) and np.all(m.channels["wetness"] <= 1.0)
    assert np.all(m.channels["mss"] > 0.0)


def test_wave_direction_is_refracted_not_the_global_wind(record, meshed, cfg):
    """`wdir` must follow the local waves, not the wind (cookbook 6.5).

    If it were the global wind it would be constant, and glint in the surf zone
    would elongate along the wrong axis.
    """
    m, _, _ = meshed
    ang = np.degrees(np.arctan2(m.channels["wdir_y"], m.channels["wdir_x"]))
    spread = float(ang.max() - ang.min())
    wind = cfg.wind.direction_deg

    record("6", "spread of per-vertex wave direction", spread, unit="deg",
           note=f"Global wind is {wind:.0f} deg; the mesh spans "
                f"{ang.min():.0f} to {ang.max():.0f} deg as waves refract toward "
                f"the contours. A constant field would mean refraction never "
                f"reached the channel.",
           passed=spread > 1.0)
    assert spread > 1.0


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_ply_round_trip_preserves_every_channel(record, meshed, tmp_path):
    """Custom per-vertex properties survive a write/read cycle exactly.

    This is what Phase 7 depends on: Mitsuba reaches these through
    `mesh_attribute` textures, and if the format dropped them the BSDF would
    have nothing to shade with.
    """
    m, _, _ = meshed
    path = export.write_ply(m, tmp_path / "water.ply")
    data, faces = export.read_ply(path)

    worst = 0.0
    for i, name in enumerate(("x", "y", "z")):
        worst = max(worst, float(np.abs(data[name] - m.vertices[:, i]).max()))
    for i, name in enumerate(("nx", "ny", "nz")):
        worst = max(worst, float(np.abs(data[name] - m.normals[:, i]).max()))
    for name, arr in m.channels.items():
        assert name in data, f"channel {name!r} lost in the PLY"
        worst = max(worst, float(np.abs(data[name] - arr).max()))

    record("6", "PLY round-trip, worst property error", worst, 0.0, 0.0,
           note=f"{len(data)} float properties per vertex: position, normal and "
                f"{len(m.channels)} channels. Binary little-endian float32, so "
                f"exact rather than merely close.",
           passed=worst == 0.0)
    assert worst == 0.0
    assert np.array_equal(faces, m.faces)


def test_obj_is_written_and_is_honest_about_being_lossy(meshed, tmp_path):
    """OBJ carries geometry only, and says so in its own header."""
    m, _, _ = meshed
    path = export.write_obj(m, tmp_path / "water.obj", quiet=True)
    head = path.read_text(encoding="ascii")[:600]

    assert "NOT representable in OBJ" in head
    for name in ("mss", "foam", "depth"):
        assert name in head, "the header should name what it dropped"

    counts = {"v ": 0, "vn ": 0, "f ": 0}
    with path.open(encoding="ascii") as fh:
        for line in fh:
            for k in counts:
                if line.startswith(k):
                    counts[k] += 1
    assert counts["v "] == m.n_vertices
    assert counts["vn "] == m.n_vertices
    assert counts["f "] == m.n_faces


def test_frame_metadata_records_provenance(meshed, tmp_path):
    """The sidecar must answer "what produced this frame" on its own."""
    import json

    m, _, _ = meshed
    path = export.write_frame_metadata(m, tmp_path / "water.json")
    doc = json.loads(path.read_text(encoding="utf-8"))

    for key in ("t", "seed", "wind_speed", "wind_direction_deg", "fetch",
                "mesh_dx", "git_sha", "n_vertices", "n_faces", "channels"):
        assert key in doc, f"metadata missing {key!r}"
    assert doc["n_vertices"] == m.n_vertices
    assert set(doc["channels"]) == set(m.channels)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_mesh_is_bit_identical_when_rebuilt(record, cfg):
    """Same config, same t, same mesh -- the Gate 6 cross-machine property."""
    from pywave import tiling

    bathy = Bathymetry.from_config(cfg, fine=True)
    built = [mesh.build_water_mesh(tiling.TileSet.build(cfg), bathy, cfg, t=1.25,
                                   dx=DX, region=REGION) for _ in range(2)]

    assert np.array_equal(built[0].faces, built[1].faces)
    worst = max(float(np.max(np.abs(built[0].vertices - built[1].vertices))),
                float(np.max(np.abs(built[0].normals - built[1].normals))))
    for name, arr in built[0].channels.items():
        worst = max(worst, float(np.max(np.abs(arr - built[1].channels[name]))))

    record("6", "max difference between two builds of the same frame", worst,
           0.0, 0.0, passed=worst == 0.0)
    assert worst == 0.0


def test_vertex_budget_refuses_rather_than_thrashing(cfg):
    """A plausible typo asks for 64 M posts; it should fail with the arithmetic.

    The shipped 0.125 m spacing over a 1 km domain is exactly that case, which
    is why the guard exists and why the message says what to change.
    """
    from pywave import tiling

    ts = tiling.TileSet.build(cfg)
    bathy = Bathymetry.from_config(cfg)

    with pytest.raises(ValueError, match="exceeds max_vertices"):
        mesh.build_water_mesh(ts, bathy, cfg, dx=0.125, max_vertices=1_000_000)

    with pytest.raises(ValueError, match="must be positive"):
        mesh.build_water_mesh(ts, bathy, cfg, dx=0.0)
    with pytest.raises(ValueError, match="region must be"):
        mesh.build_water_mesh(ts, bathy, cfg, dx=1.0, region=(10.0, 10.0, 5.0, 20.0))


def test_triangulation_requires_all_four_corners():
    """A cell is emitted only when its whole quad is inside the mask."""
    m = np.zeros((4, 4), dtype=bool)
    m[1:3, 1:3] = True
    idx, faces = mesh.triangulate_mask(m)
    assert idx[m].min() == 0 and idx[m].max() == 3
    assert np.all(idx[~m] == -1)
    assert faces.shape == (2, 3)          # one quad -> two triangles

    m[2, 2] = False                       # break a corner
    _, faces = mesh.triangulate_mask(m)
    assert faces.shape == (0, 3)
