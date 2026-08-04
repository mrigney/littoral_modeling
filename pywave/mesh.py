"""PHASE 6 -- water mesh generation.

Turns the surface into geometry: a regular grid of posts over the wet area,
displaced by the wave field, carrying the per-vertex channels a renderer needs.

Constant post spacing, deliberately
-----------------------------------
The cookbook's section 6.2 specifies concentric LOD rings, each with its own
spacing, stitched at the boundaries.  That is not implemented here.  This module
meshes at one spacing everywhere, which makes the geometry trivial (a regular
grid triangulates into two triangles per cell with no seams to stitch and no
transition rows to get wrong) and collapses the LOD invariant from a per-vertex
bookkeeping problem into a single global check.

The cost of that choice is vertex count, and it is not small.  Post spacing
enters quadratically: the shipped test lake at its configured 0.125 m spacing is
8000 x 8000 = 64 million posts over the full 1 km domain, which is several
gigabytes of mesh for one frame.  So either the spacing is coarsened or the mesh
is bounded to a region of interest, and :func:`build_water_mesh` takes a
``region`` for exactly that reason.  Bounding the region is what stands in for
LOD rings until Phase 6.2 is built.

``max_vertices`` refuses rather than swapping the machine to death; it is a
guard against a plausible typo, not a policy.

Reference: cookbook sections 6.1, 6.3, 6.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bathymetry import Bathymetry

__all__ = ["TriMesh", "WaterMesh", "build_water_mesh", "build_terrain_mesh",
           "water_extent_mask", "triangulate_mask"]

DEFAULT_MAX_VERTICES = 12_000_000


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def water_extent_mask(depth: np.ndarray, dx: float, trim_depth: float = 0.02,
                      margin: float = 0.5) -> np.ndarray:
    """Cells to mesh: wet, plus a margin landward for the swash to live in.

    Trimming at a small positive depth rather than at exactly zero is not
    fussiness.  The band around ``depth = 0`` is where a mesh built on a
    depth threshold goes slivery, and where it z-fights against the terrain it
    is supposed to sit on.  Cutting at 2 cm and then dilating landward gives a
    clean edge that still covers the swash.

    ``margin`` is in metres and is dilated in every direction; only the landward
    part matters, but a symmetric dilation is cheaper and harmless offshore
    where the mask is already true.
    """
    from scipy.ndimage import binary_dilation

    wet = depth > trim_depth
    if margin > 0 and wet.any():
        r = max(int(round(margin / dx)), 1)
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        wet = binary_dilation(wet, structure=(x**2 + y**2 <= r * r))
    return wet


def triangulate_mask(mask: np.ndarray):
    """Two triangles per cell whose four corners are all inside ``mask``.

    Returns ``(vertex_index, faces)`` where ``vertex_index`` is an int array
    shaped like ``mask`` holding each kept post's vertex id (-1 where dropped),
    and ``faces`` is ``(F, 3)`` int32.

    Requiring all four corners keeps the boundary a clean staircase along cell
    edges. Emitting a triangle whenever any three corners are present would
    chase the waterline more closely, at the cost of exactly the slivers
    section 6.1 says to avoid.
    """
    ny, nx = mask.shape
    idx = np.full(mask.shape, -1, dtype=np.int64)
    idx[mask] = np.arange(int(mask.sum()), dtype=np.int64)

    quad = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    if not quad.any():
        return idx, np.zeros((0, 3), dtype=np.int32)

    v00 = idx[:-1, :-1][quad]
    v01 = idx[:-1, 1:][quad]
    v10 = idx[1:, :-1][quad]
    v11 = idx[1:, 1:][quad]

    # Counter-clockwise seen from +Z, so the winding agrees with the analytic
    # normal (which always has a positive Z component).
    faces = np.concatenate([
        np.stack([v00, v01, v11], axis=1),
        np.stack([v00, v11, v10], axis=1),
    ]).astype(np.int32)
    return idx, faces


# ---------------------------------------------------------------------------
# The mesh
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriMesh:
    """A triangle mesh with per-vertex channels.

    Used for both the water surface and the bed. They differ in where the
    normals come from -- analytic for water, geometric for terrain -- but the
    container and the export path are the same.
    """

    vertices: np.ndarray
    """``(V, 3)`` float32 positions [m], scene coordinates, Z up."""
    faces: np.ndarray
    """``(F, 3)`` int32 triangle indices."""
    normals: np.ndarray
    """``(V, 3)`` float32 unit normals."""
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    """Per-vertex float32 channels; see the section 0.4 contract."""
    meta: dict = field(default_factory=dict)
    """Provenance: time, spacing, region, seed, derived quantities."""

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def bounds(self):
        """``(min_xyz, max_xyz)`` of the displaced geometry."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def face_normals(self) -> np.ndarray:
        """Geometric normals from the triangles -- a *check* on the analytic ones.

        Never use these for rendering. They are a finite-difference
        approximation of a field already known exactly, and they lose precisely
        the high-frequency content that matters radiometrically (cookbook 6.3).
        They exist so the two can be compared, which is a Gate 6 check.
        """
        v = self.vertices
        a, b, c = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        n = np.cross(b - a, c - a)
        mag = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(mag > 0, mag, 1.0)

    def validate(self) -> dict:
        """Structural checks. Raises on anything a renderer would choke on."""
        if self.faces.size and (self.faces.min() < 0
                                or self.faces.max() >= self.n_vertices):
            raise AssertionError("face indices out of range")
        for name, arr in (("vertices", self.vertices), ("normals", self.normals)):
            if not np.all(np.isfinite(arr)):
                raise AssertionError(f"{name} contains non-finite values")
        for name, arr in self.channels.items():
            if arr.shape != (self.n_vertices,):
                raise AssertionError(f"channel {name!r} has shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise AssertionError(f"channel {name!r} contains non-finite values")

        norm = np.linalg.norm(self.normals, axis=1)
        worst = float(np.max(np.abs(norm - 1.0))) if norm.size else 0.0
        if worst > 1e-5:
            raise AssertionError(f"normals not unit length (max error {worst:.2e})")
        if self.normals.size and float(self.normals[:, 2].min()) <= 0.0:
            raise AssertionError("a normal points downward; the surface has folded")

        degenerate = 0
        if self.faces.size:
            f = self.faces
            degenerate = int(np.sum((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2])
                                    | (f[:, 0] == f[:, 2])))
            if degenerate:
                raise AssertionError(f"{degenerate} degenerate triangles")

        return {"n_vertices": self.n_vertices, "n_faces": self.n_faces,
                "normal_unit_error": worst,
                "min_normal_z": float(self.normals[:, 2].min()) if self.normals.size else 1.0}


WaterMesh = TriMesh
"""Backwards-compatible alias; the container was never water-specific."""


def build_water_mesh(
    tileset,
    bathy: Bathymetry,
    cfg,
    t: float = 0.0,
    *,
    dx: float | None = None,
    region: tuple[float, float, float, float] | None = None,
    trim_depth: float = 0.02,
    margin: float | None = None,
    refraction: str = "snell",
    foam: np.ndarray | None = None,
    foam_bathy: Bathymetry | None = None,
    max_vertices: int = DEFAULT_MAX_VERTICES,
) -> WaterMesh:
    """Build the displaced water mesh at time ``t``.

    Parameters
    ----------
    dx : post spacing [m]. Defaults to ``cfg.output.mesh_dx``.
    region : ``(x0, y0, x1, y1)`` in scene coordinates. Defaults to the whole
        bathymetry extent, which at fine spacings is usually far too much
        geometry -- see the module docstring.
    trim_depth : cut the mesh at this depth rather than at zero [m].
    margin : landward dilation past the waterline [m]. Defaults to the swash
        excursion, so the swash band has geometry to live on.
    foam : optional foam coverage on ``foam_bathy``'s grid, sampled onto the
        vertices. Omitted rather than faked when absent.

    Notes
    -----
    Displacement follows cookbook 6.3: the rest position is offset horizontally
    by the choppiness-scaled displacement and vertically by the transformed wave
    height. Normals come from the *analytic* spectral slopes, never from the
    triangles.
    """
    from . import nearshore

    dx = float(dx if dx is not None else cfg.output.mesh_dx)
    if dx <= 0:
        raise ValueError(f"mesh spacing must be positive, got {dx}")

    x0b, x1b, y0b, y1b = bathy.meta.extent
    if region is None:
        x0, y0, x1, y1 = x0b, y0b, x1b, y1b
    else:
        x0, y0, x1, y1 = (float(v) for v in region)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"region must be (x0, y0, x1, y1) with x1 > x0, "
                             f"y1 > y0; got {region}")

    nx = max(int(np.floor((x1 - x0) / dx)) + 1, 2)
    ny = max(int(np.floor((y1 - y0) / dx)) + 1, 2)
    if nx * ny > max_vertices:
        raise ValueError(
            f"{nx} x {ny} = {nx * ny:,} posts exceeds max_vertices="
            f"{max_vertices:,}. Coarsen `dx` (it enters quadratically -- "
            f"doubling it quarters the count) or pass a smaller `region`. "
            f"At dx = {dx} m the whole area needs "
            f"{(x1 - x0) * (y1 - y0) / dx**2:,.0f} posts.")

    xs = x0 + np.arange(nx) * dx
    ys = y0 + np.arange(ny) * dx
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    depth_g, sdf_g, _ = bathy.sample(X, Y)
    if margin is None:
        margin = _swash_margin(tileset, bathy, cfg)
    mask = water_extent_mask(depth_g, dx, trim_depth, margin)
    if not mask.any():
        raise ValueError(
            f"no wet cells in region {(x0, y0, x1, y1)} at trim_depth="
            f"{trim_depth} m; depth ranges {depth_g.min():.3f} to "
            f"{depth_g.max():.3f} m there")

    idx, faces = triangulate_mask(mask)
    vx, vy = X[mask], Y[mask]

    nf = nearshore.transform(tileset, bathy, cfg, vx, vy, t, refraction=refraction)
    s = nf.surface
    chop = cfg.surface.choppiness

    vertices = np.stack([
        vx + chop * s.dx_disp,
        vy + chop * s.dy_disp,
        cfg.scene.water_level + s.h,
    ], axis=1).astype(np.float32)

    n = np.stack([-s.slope_x, -s.slope_y, np.ones_like(s.h)], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    from .channels import vertex_channels

    channels = vertex_channels(tileset, bathy, cfg, vx, vy, nf, dx,
                               foam=foam, foam_bathy=foam_bathy,
                               refraction=refraction)

    meta = {
        "t": float(t),
        "mesh_dx": dx,
        "region": [x0, y0, x1, y1],
        "grid": [int(ny), int(nx)],
        "trim_depth": trim_depth,
        "margin": float(margin),
        "refraction": refraction,
        "choppiness": float(chop),
        "water_level": float(cfg.scene.water_level),
        "seed": int(cfg.spectrum.seed),
        "wind_speed": float(cfg.wind.speed),
        "wind_direction_deg": float(cfg.wind.direction_deg),
        "fetch": float(cfg.wind.fetch),
        "scene": cfg.name,
    }
    return WaterMesh(vertices=vertices, faces=faces,
                     normals=n.astype(np.float32), channels=channels, meta=meta)


def _swash_margin(tileset, bathy, cfg) -> float:
    """Landward margin: the swash excursion, so the wet band has geometry."""
    from . import nearshore

    slope = bathy.beach_slope()
    l0 = float(nearshore.deep_water_wavelength(1.0 / cfg.f_p))
    hs = tileset.hs()
    xi = nearshore.iribarren_number(slope, hs, l0)
    return float(nearshore.swash_width(nearshore.hunt_runup(xi, hs), slope))


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------


def build_terrain_mesh(
    bathy: Bathymetry,
    cfg,
    *,
    dx: float | None = None,
    region: tuple[float, float, float, float] | None = None,
    pad: float | None = None,
    tileset=None,
    max_vertices: int = DEFAULT_MAX_VERTICES,
) -> TriMesh:
    """Mesh the bed, co-registered with the water mesh.

    A water mesh on its own is not renderable: it has nothing to sit on, nothing
    to occlude it at the shoreline, and no bottom to see through in EO.  This
    builds the matching terrain over the same region, in the same coordinates,
    so the two PLYs drop into a scene together.

    Parameters
    ----------
    region : the water mesh's region. Pass the same tuple.
    pad : extra terrain beyond ``region`` on every side [m]. Defaults to four
        posts, so the bed extends slightly past the water and no sliver of
        background shows through at the edge.
    tileset : optional; only used to size the swash band for the ``wetness``
        channel. Without it, wetness is omitted rather than guessed.

    Normals here are geometric, and that is correct
    ----------------------------------------------
    The water mesh takes its normals from the analytic spectral slopes, because
    for water the field *is* known exactly and differencing it would throw away
    the high-frequency content that matters most (cookbook 6.3).  The bed is the
    opposite case: it only exists as a sampled grid -- the shoreline comes from
    a Euclidean distance transform, not a closed form -- so there is no analytic
    slope to prefer, and a central difference of the height field is the honest
    answer rather than a compromise.

    Co-registration, which is the whole point
    ----------------------------------------
    Both meshes are built from the same bathymetry on the same world grid, so a
    post at ``(x, y)`` means the same place in both.  Where the water mesh runs
    landward of the waterline it sits at the still-water level and the bed rises
    above it, so the water is hidden rather than z-fighting.  Offshore, the
    depth-limited breaking criterion keeps the wave trough above the bed: with
    ``Hs <= gamma_b d`` the elevation stays inside roughly ``+/- 0.6 d``, so the
    surface cannot punch through the bottom in shallow water.  That is asserted
    in the tests rather than assumed.
    """
    dx = float(dx if dx is not None else cfg.output.mesh_dx)
    if dx <= 0:
        raise ValueError(f"mesh spacing must be positive, got {dx}")
    if pad is None:
        pad = 4.0 * dx

    x0b, x1b, y0b, y1b = bathy.meta.extent
    if region is None:
        x0, y0, x1, y1 = x0b, y0b, x1b, y1b
    else:
        x0, y0, x1, y1 = (float(v) for v in region)
        x0, y0 = max(x0 - pad, x0b), max(y0 - pad, y0b)
        x1, y1 = min(x1 + pad, x1b), min(y1 + pad, y1b)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty terrain region {(x0, y0, x1, y1)}")

    nx = max(int(np.floor((x1 - x0) / dx)) + 1, 2)
    ny = max(int(np.floor((y1 - y0) / dx)) + 1, 2)
    if nx * ny > max_vertices:
        raise ValueError(
            f"terrain {nx} x {ny} = {nx * ny:,} posts exceeds max_vertices="
            f"{max_vertices:,}; coarsen dx or shrink the region")

    xs = x0 + np.arange(nx) * dx
    ys = y0 + np.arange(ny) * dx
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    depth, sdf, _ = bathy.sample(X, Y)
    z = cfg.scene.water_level - depth

    # Every post is meshed: unlike water, the bed has no extent question.
    mask = np.ones(depth.shape, dtype=bool)
    _, faces = triangulate_mask(mask)

    # Central differences of the height field. np.gradient returns d/dy first
    # for [y, x] indexing.
    dzdy, dzdx = np.gradient(z, dx)
    n = np.stack([-dzdx, -dzdy, np.ones_like(z)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    channels = {
        "depth": depth.ravel().astype(np.float32),
        "sdf": sdf.ravel().astype(np.float32),
        "slope": np.hypot(dzdx, dzdy).ravel().astype(np.float32),
    }
    if tileset is not None:
        from . import nearshore

        band = _swash_margin(tileset, bathy, cfg)
        channels["wetness"] = nearshore.wetness_fraction(
            sdf, band).ravel().astype(np.float32)

    vertices = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1).astype(np.float32)
    meta = {
        "kind": "terrain",
        "mesh_dx": dx,
        "region": [x0, y0, x1, y1],
        "grid": [int(ny), int(nx)],
        "pad": float(pad),
        "water_level": float(cfg.scene.water_level),
        "dean_A": float(cfg.bathymetry.a),
        "profile": cfg.bathymetry.profile,
        "scene": cfg.name,
        "epsg": int(cfg.scene.epsg),
    }
    return TriMesh(vertices=vertices, faces=faces,
                   normals=n.reshape(-1, 3).astype(np.float32),
                   channels=channels, meta=meta)
