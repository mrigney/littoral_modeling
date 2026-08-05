"""PHASE 4/5 -- bathymetry, whether synthetic or loaded from a terrain export.

Phase 4 builds the lake basin in Houdini and exports six fields.  Phase 5 needs
three of them -- ``depth``, ``sdf`` and ``shore_normal`` -- and nothing else.

Two constructors produce the same object.  :meth:`Bathymetry.from_export` reads
a real Phase 4 export; :meth:`Bathymetry.dean_beach` and friends manufacture the
fields analytically.  The synthetic path came first deliberately, so the
nearshore physics could be written and validated before Houdini existed, and it
remains useful afterwards as the oracle -- see below.

Why this is worth doing rather than waiting for Phase 4
------------------------------------------------------
A synthetic profile has closed-form answers.  Shoaling on a Dean beach can be
checked against Green's law, refraction against Snell's law, and the breaker
line against ``d = H/gamma_b`` -- all exactly, at every cell.  An exported
heightfield gives none of that: it can only be checked against itself.  So the
synthetic profile is not a placeholder to be thrown away, it is the *oracle*
Phase 4's real bathymetry will be checked against.

The contract is deliberately identical to what ``export_fields.py`` will write
(cookbook section 4.5), so swapping in real data is a loader change and not a
physics change:

============  ==============================================================
Field         Definition
============  ==============================================================
depth         ``z_w - z_terrain``.  Positive in water, negative on land.
sdf           Signed distance to the ``z = z_w`` contour.  **Negative in
              water, positive on land** -- the opposite sign to depth, so
              ``s`` reads as "distance inland" (cookbook section 0.3).
shore_normal  ``grad(s) / |grad(s)|``, unit, pointing **inland**.
============  ==============================================================

Reference
---------
Dean, R.G. (1977). "Equilibrium beach profiles: U.S. Atlantic and Gulf coasts."
    Ocean Engineering Report No. 12, University of Delaware.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "DEAN_A",
    "GridMeta",
    "Bathymetry",
    "dean_depth",
    "dean_slope",
    "dean_A_for_grain_size",
]


# Dean's scale parameter A [m^(1/3)] by sediment grain size.  Dean (1977),
# tabulated in most coastal engineering texts.  Coarser sediment stands at a
# steeper equilibrium profile.
DEAN_A = {
    "fine_sand": 0.079,      # D50 ~ 0.15 mm
    "medium_sand": 0.100,    # D50 ~ 0.25 mm
    "coarse_sand": 0.125,    # D50 ~ 0.50 mm
    "gravel": 0.200,         # D50 ~ 2 mm
}
"""``A`` in ``h = A * y^(2/3)``, keyed by sediment class [m^(1/3)]."""


def dean_A_for_grain_size(d50_mm: float) -> float:
    """Dean's ``A`` from median grain diameter [mm].

    Fits ``A = 0.21 * D50^0.48`` over 0.1-1 mm, which reproduces the tabulated
    values in :data:`DEAN_A` to a few percent. Outside that range it is an
    extrapolation and should be treated as such.
    """
    if d50_mm <= 0:
        raise ValueError(f"d50_mm must be positive, got {d50_mm}")
    return 0.21 * d50_mm**0.48


def dean_depth(offshore_distance, a: float) -> np.ndarray:
    """Equilibrium beach profile ``h = A * y^(2/3)`` [m].

    Concave-up, which is what real beaches are and what a linear ramp is not.
    ``offshore_distance`` is measured from the waterline; negative values (i.e.
    landward) return 0.
    """
    y = np.maximum(np.asarray(offshore_distance, dtype=np.float64), 0.0)
    return a * y ** (2.0 / 3.0)


def dean_slope(offshore_distance, a: float) -> np.ndarray:
    """Local bed slope ``dh/dy = (2/3) A y^(-1/3)`` [-].

    Diverges at the waterline -- the Dean profile is not differentiable there.
    Callers that need a slope at ``y = 0`` (runup, Iribarren) should evaluate it
    a realistic swash-width offshore instead; :meth:`Bathymetry.beach_slope`
    does exactly that.
    """
    y = np.maximum(np.asarray(offshore_distance, dtype=np.float64), 1e-9)
    return (2.0 / 3.0) * a * y ** (-1.0 / 3.0)


@dataclass(frozen=True)
class GridMeta:
    """Georeferencing for a field grid -- mirrors Phase 4's ``grid_meta.json``."""

    origin: tuple[float, float]
    """World coordinates of cell ``[0, 0]`` [m]."""
    dx: float
    """Cell size [m], square cells."""
    shape: tuple[int, int]
    """``(ny, nx)``."""
    water_level: float
    """``z_w`` [m]."""
    epsg: int = 32616

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` [m]."""
        ny, nx = self.shape
        x0, y0 = self.origin
        return x0, x0 + nx * self.dx, y0, y0 + ny * self.dx

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinate axes ``(x, y)`` [m]."""
        ny, nx = self.shape
        x0, y0 = self.origin
        return (x0 + (np.arange(nx) + 0.5) * self.dx,
                y0 + (np.arange(ny) + 0.5) * self.dx)


@dataclass(frozen=True)
class Bathymetry:
    """Depth, signed distance and shore normal on a regular grid.

    All arrays are ``[y, x]``, matching the section 0.4 data contract.
    """

    meta: GridMeta
    depth: np.ndarray
    """``z_w - z_terrain`` [m]. Positive in water."""
    sdf: np.ndarray
    """Signed distance to the waterline [m]. Negative in water."""
    shore_normal: np.ndarray
    """``(2, ny, nx)`` unit vectors pointing inland."""
    dean_a: float | None = None
    """Dean scale parameter [m^(1/3)] for a synthetic profile.

    ``None`` for bathymetry loaded from an export: real terrain has no analytic
    profile, and :meth:`beach_slope` measures the bed instead of evaluating a
    formula.  Which of the two is in force is therefore never ambiguous.
    """
    bottom_type: np.ndarray | None = None
    """``uint8`` class index per cell, from the Phase 4 export.  ``None`` for
    synthetic bathymetry, which has no sediment classification."""

    # -- construction --------------------------------------------------------

    @classmethod
    def from_export(cls, directory, *, validate: bool = True,
                    sdf_tol: float = 1.0) -> "Bathymetry":
        """Load a Phase 4 terrain export.

        Reads the six files cookbook section 4.5 specifies, checks them against
        the same assertion set the synthetic bathymetry is held to, and returns
        a :class:`Bathymetry` that every downstream module already knows how to
        use.  That is the whole point of having built Phase 5 against a
        synthetic profile satisfying the identical contract: this is a loader,
        not a physics change.

        ============  ===========================================
        File          Contents
        ============  ===========================================
        grid_meta.json  ``x0, y0, dx, dy, nx, ny, z_w, epsg``
        terrain_z.npy   bed elevation [m]
        depth.npy       ``z_w - terrain_z``, positive in water
        sdf.npy         signed distance, positive **inland**
        shore_normal.npy  unit vectors pointing inland
        bottom_type.npy   ``uint8`` class index (optional)
        ============  ===========================================

        ``shore_normal`` is accepted as either ``(2, ny, nx)`` or ``(ny, nx, 2)``
        -- the section 0.4 table writes the latter while this package stores the
        former, and silently transposing the wrong one would rotate every
        refraction angle by ninety degrees in a way no scalar check would catch.

        Raises
        ------
        FileNotFoundError
            If a required file is missing, naming all of them.
        ValueError
            If the grid is non-square, the shapes disagree with
            ``grid_meta.json``, or the fields fail the section 4.5 checks.
        """
        import json

        directory = Path(directory)
        meta_path = directory / "grid_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found. A Phase 4 export needs grid_meta.json, "
                f"terrain_z.npy, depth.npy, sdf.npy and shore_normal.npy "
                f"(bottom_type.npy optional).")
        raw = json.loads(meta_path.read_text(encoding="utf-8"))

        dx, dy = float(raw["dx"]), float(raw.get("dy", raw["dx"]))
        if abs(dx - dy) > 1e-9:
            raise ValueError(
                f"grid_meta gives dx={dx} and dy={dy}. This package assumes "
                f"square cells throughout -- resample the export, or the "
                f"wavenumber grids and every gradient below will be wrong by "
                f"the aspect ratio.")

        ny, nx = int(raw["ny"]), int(raw["nx"])
        meta = GridMeta(origin=(float(raw["x0"]), float(raw["y0"])), dx=dx,
                        shape=(ny, nx), water_level=float(raw["z_w"]),
                        epsg=int(raw.get("epsg", 32616)))

        def _load(name, required=True):
            path = directory / f"{name}.npy"
            if not path.exists():
                if required:
                    raise FileNotFoundError(f"{path} not found")
                return None
            return np.load(path)

        terrain_z = _load("terrain_z")
        depth = _load("depth")
        sdf = _load("sdf")
        normal = _load("shore_normal")
        bottom_type = _load("bottom_type", required=False)

        for name, arr in (("terrain_z", terrain_z), ("depth", depth),
                          ("sdf", sdf)):
            if arr.shape != (ny, nx):
                raise ValueError(
                    f"{name}.npy has shape {arr.shape}, but grid_meta says "
                    f"({ny}, {nx}). Check the export is [y, x] and C-ordered.")

        normal = np.asarray(normal)
        if normal.shape == (ny, nx, 2):
            normal = np.moveaxis(normal, -1, 0)
        elif normal.shape != (2, ny, nx):
            raise ValueError(
                f"shore_normal.npy has shape {normal.shape}; expected "
                f"(2, {ny}, {nx}) or ({ny}, {nx}, 2).")

        if bottom_type is not None and bottom_type.shape != (ny, nx):
            raise ValueError(
                f"bottom_type.npy has shape {bottom_type.shape}, expected "
                f"({ny}, {nx})")

        bathy = cls(meta=meta,
                    depth=np.ascontiguousarray(depth, dtype=np.float64),
                    sdf=np.ascontiguousarray(sdf, dtype=np.float64),
                    shore_normal=np.ascontiguousarray(normal, dtype=np.float64),
                    dean_a=None,
                    bottom_type=(None if bottom_type is None
                                 else np.ascontiguousarray(bottom_type)))

        # The export carries terrain_z explicitly; the property derives it from
        # depth. They must agree, or the two halves of the export disagree about
        # where the bed is.
        residual = float(np.max(np.abs(
            np.asarray(terrain_z, dtype=np.float64) - bathy.terrain_z)))
        if residual > 1e-4:
            raise ValueError(
                f"terrain_z.npy disagrees with z_w - depth by up to "
                f"{residual:.3e} m. One of the two is stale.")

        if validate:
            bathy.validate(sdf_tol=sdf_tol)
        return bathy

    def summary(self) -> dict:
        """Descriptive numbers, for a report or a sanity check after loading."""
        wet = self.depth > 0.0
        gy, gx = np.gradient(self.terrain_z, self.meta.dx)
        slope = np.hypot(gx, gy)
        near = wet & (self.depth < 1.0)
        return {
            "shape": list(self.meta.shape),
            "dx": self.meta.dx,
            "extent": [float(v) for v in self.meta.extent],
            "water_level": self.meta.water_level,
            "epsg": self.meta.epsg,
            "source": "synthetic" if self.dean_a is not None else "export",
            "dean_a": self.dean_a,
            "water_fraction": float(wet.mean()),
            "max_depth": float(self.depth.max()),
            "max_land_elevation_above_zw": float(-self.depth.min()),
            "foreshore_slope": self.beach_slope(),
            "median_slope_under_1m_depth": (float(np.median(slope[near]))
                                            if near.any() else None),
            "shoreline_cells": int(np.sum(np.abs(self.sdf) < 0.75 * self.meta.dx)),
            "has_bottom_type": self.bottom_type is not None,
        }

    @classmethod
    def from_config(cls, cfg, fine: bool = False) -> "Bathymetry":
        """Build the basin a scene file describes.

        ``fine=True`` uses ``bathymetry.surf_dx`` instead of ``dx``, and crops to
        a band around the waterline rather than the whole domain -- a refined
        grid over the full scene would be enormous and is pointless offshore,
        where nothing varies on that scale.

        Cropping keeps the world coordinates of the coarse grid, so a point
        sampled from either grid lands in the same place.
        """
        b = cfg.bathymetry

        if b.is_export:
            # Real terrain has one resolution -- whatever Houdini exported.
            # `fine` is meaningless here and returning the same grid is the
            # honest answer: the mesh samples bathymetry bilinearly and may be
            # finer than it, but no interpolation creates bathymetry that was
            # never measured.
            source = Path(b.source)
            if not source.is_absolute() and cfg.source_path is not None:
                candidate = cfg.source_path.parent.parent / source
                if candidate.exists():
                    source = candidate
            loaded = cls.from_export(source)

            # The scene and the export both carry a water level, and they must
            # agree. If they do not, every mesh is built at the config's value
            # while every depth is measured from the export's, so the water and
            # terrain stay consistent *with each other* and both sit at the
            # wrong absolute height -- which looks perfect in isolation and is
            # metres out against anything else in the world.
            delta = abs(cfg.scene.water_level - loaded.meta.water_level)
            if delta > 1e-6:
                raise ValueError(
                    f"scene.water_level is {cfg.scene.water_level} but "
                    f"{source}/grid_meta.json says z_w = "
                    f"{loaded.meta.water_level} ({delta:g} m apart). Both meshes "
                    f"would be built at the config's value and every depth "
                    f"measured from the export's, putting the whole scene "
                    f"{delta:g} m off in world Z while looking self-consistent. "
                    f"Set scene.water_level to match the export.")
            if cfg.scene.epsg != loaded.meta.epsg:
                raise ValueError(
                    f"scene.epsg is {cfg.scene.epsg} but the export says "
                    f"{loaded.meta.epsg}; they must describe the same CRS.")
            return loaded

        width, height = cfg.scene.domain

        if fine:
            dx = b.surf_dx
            # Generous enough to hold the surf zone, the swash and the offshore
            # run-up to a few peak wavelengths -- plus, on a curved shoreline,
            # the full bay-to-headland excursion. Sizing the window off the
            # nominal shoreline alone crops the headlands clean off.
            margin = max(20.0 * cfg.lambda_p, 40.0)
            wander = b.amplitude if b.profile == "embayment" else 0.0
            y0 = max(0.0, b.shoreline - margin - wander)
            y1 = min(height, b.shoreline + 0.25 * margin + wander)
        else:
            dx = b.dx
            y0, y1 = 0.0, height

        nx = max(int(round(width / dx)), 8)
        ny = max(int(round((y1 - y0) / dx)), 8)

        common = dict(nx=nx, ny=ny, dx=dx, shoreline_y=b.shoreline, dean_a=b.a,
                      max_depth=b.max_depth, bank_slope=b.bank_slope,
                      water_level=cfg.scene.water_level, origin=(0.0, y0),
                      epsg=cfg.scene.epsg)
        if b.profile == "embayment":
            return cls.dean_embayment(amplitude=b.amplitude, wavelength=b.wavelength,
                                      **common)
        return cls.dean_beach(**common)

    @classmethod
    def dean_beach(
        cls,
        nx: int = 512,
        ny: int = 512,
        dx: float = 1.0,
        shoreline_y: float = 400.0,
        dean_a: float = DEAN_A["medium_sand"],
        max_depth: float = 5.0,
        bank_slope: float = 0.08,
        water_level: float = 100.0,
        origin: tuple[float, float] = (0.0, 0.0),
        epsg: int = 32616,
    ) -> "Bathymetry":
        """A straight shoreline running along +X, water on the **-Y** side.

        Land occupies ``y > shoreline_y``, so the shore normal points along +Y
        and a wind blowing toward the north-east drives waves *onto* the beach.
        Getting that the wrong way round makes every refraction angle obtuse and
        every shoaling coefficient meaningless, so the orientation is fixed here
        rather than left to the caller.

        The simplest useful case: contours are straight and parallel, which is
        exactly the geometry Snell's law is stated for, so refraction has a
        closed-form answer everywhere.
        """
        return cls._from_shoreline(
            lambda x: np.full_like(x, shoreline_y), nx, ny, dx, dean_a,
            max_depth, bank_slope, water_level, origin, epsg,
        )

    @classmethod
    def dean_embayment(
        cls,
        nx: int = 512,
        ny: int = 512,
        dx: float = 1.0,
        shoreline_y: float = 380.0,
        amplitude: float = 40.0,
        wavelength: float = 400.0,
        dean_a: float = DEAN_A["medium_sand"],
        max_depth: float = 5.0,
        bank_slope: float = 0.08,
        water_level: float = 100.0,
        origin: tuple[float, float] = (0.0, 0.0),
        epsg: int = 32616,
    ) -> "Bathymetry":
        """A cosine-perturbed shoreline: alternating bays and headlands.

        Curved contours are what make refraction interesting -- wave energy
        focuses on headlands and spreads in bays. A straight beach cannot show
        that, so this case exists to check the shore normal is doing real work
        rather than being a constant.
        """
        def curve(x):
            return shoreline_y + amplitude * np.cos(2.0 * np.pi * x / wavelength)

        return cls._from_shoreline(curve, nx, ny, dx, dean_a, max_depth,
                                   bank_slope, water_level, origin, epsg)

    @classmethod
    def _from_shoreline(cls, curve, nx, ny, dx, dean_a, max_depth, bank_slope,
                        water_level, origin, epsg) -> "Bathymetry":
        """Build the three fields from a shoreline curve ``y = curve(x)``.

        The order matters and is not the obvious one. Dean's profile is a
        function of *distance offshore*, which is `|sdf|` -- so the signed
        distance has to exist before the depth does. Deriving the sdf from a
        depth field that was itself built from a distance would be circular.
        Here the shoreline is defined geometrically, the sdf comes from an exact
        Euclidean distance transform of the resulting water mask (cookbook 4.5
        specifically calls for EDT rather than an approximate SDF, because Phase
        5 differentiates it), and the depth follows from the sdf.
        """
        from scipy.ndimage import distance_transform_edt, gaussian_filter

        meta = GridMeta(origin=origin, dx=float(dx), shape=(int(ny), int(nx)),
                        water_level=float(water_level), epsg=int(epsg))
        x_axis, y_axis = meta.axes()
        xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")

        water = yy < curve(xx)
        if not water.any() or water.all():
            raise ValueError("shoreline does not cross the grid; check the curve "
                             "and the domain extent")

        # Exact Euclidean distance to the waterline, in metres, both sides.
        inside = distance_transform_edt(water, sampling=dx)
        outside = distance_transform_edt(~water, sampling=dx)
        # s > 0 on land, s < 0 in water. The half-cell offset centres the zero
        # contour on the boundary rather than on the first land cell.
        sdf = (outside - inside) + np.where(water, 0.5 * dx, -0.5 * dx)

        offshore = np.maximum(-sdf, 0.0)
        depth = np.minimum(dean_depth(offshore, dean_a), max_depth)
        depth = np.where(water, depth, -bank_slope * sdf)

        # Shore normal. Cookbook 4.5: smooth s slightly before differentiating,
        # because raw EDT is noisy at the pixel level and that noise propagates
        # straight into the refraction direction.
        s_smooth = gaussian_filter(sdf, sigma=1.5, mode="nearest")
        gy, gx = np.gradient(s_smooth, dx)
        mag = np.hypot(gx, gy)
        mag = np.where(mag > 1e-12, mag, 1.0)
        shore_normal = np.stack([gx / mag, gy / mag])

        return cls(meta=meta, depth=depth, sdf=sdf,
                   shore_normal=shore_normal, dean_a=float(dean_a))

    def crop(self, x_range: tuple[float, float],
             y_range: tuple[float, float]) -> "Bathymetry":
        """A view-sized sub-grid, keeping world coordinates.

        The origin moves with the crop, so a point sampled from the crop and the
        same point sampled from the parent land in the same place.

        Exists because anything that iterates over the grid -- foam advection
        above all -- costs the whole domain otherwise. A 200 m animation window
        on a 4 km scene touches under 2% of the cells, and stepping the other
        98% every frame is pure waste.
        """
        x0, y0 = self.meta.origin
        dx = self.meta.dx
        ny, nx = self.meta.shape

        i0 = int(np.clip(np.floor((y_range[0] - y0) / dx), 0, ny - 2))
        i1 = int(np.clip(np.ceil((y_range[1] - y0) / dx) + 1, i0 + 2, ny))
        j0 = int(np.clip(np.floor((x_range[0] - x0) / dx), 0, nx - 2))
        j1 = int(np.clip(np.ceil((x_range[1] - x0) / dx) + 1, j0 + 2, nx))

        meta = GridMeta(origin=(x0 + j0 * dx, y0 + i0 * dx), dx=dx,
                        shape=(i1 - i0, j1 - j0),
                        water_level=self.meta.water_level, epsg=self.meta.epsg)
        return Bathymetry(
            meta=meta,
            depth=self.depth[i0:i1, j0:j1].copy(),
            sdf=self.sdf[i0:i1, j0:j1].copy(),
            shore_normal=self.shore_normal[:, i0:i1, j0:j1].copy(),
            dean_a=self.dean_a,
            bottom_type=(None if self.bottom_type is None
                         else self.bottom_type[i0:i1, j0:j1].copy()),
        )

    # -- derived -------------------------------------------------------------

    @property
    def water_mask(self) -> np.ndarray:
        return self.depth > 0.0

    @property
    def terrain_z(self) -> np.ndarray:
        """Bed elevation [m]. ``depth = z_w - terrain_z`` by construction."""
        return self.meta.water_level - self.depth

    def beach_slope(self, at_depth: float = 0.1, band: float | None = None) -> float:
        """Representative foreshore slope [-], evaluated at a finite depth.

        "The beach slope" is only meaningful once you say where: the Dean
        profile's slope diverges at the waterline, and real terrain is simply
        not constant.  Runup and the Iribarren number want the slope over the
        swash zone, so the default samples at 10 cm depth.

        Two paths, and which one runs is decided by whether an analytic profile
        exists rather than by a flag:

        ``dean_a`` set (synthetic)
            Evaluate the profile exactly.  Cheap, and it is the number the Gate
            5 checks are pinned against.

        ``dean_a`` is ``None`` (loaded from an export)
            Measure the bed.  Take the **median** ``|grad z|`` over wet cells
            whose depth is within ``band`` of ``at_depth``, falling back to a
            widening search if that shell is empty.  Median rather than mean
            because a real shoreline includes cliffs, gullies and the odd
            vertical face, and a mean is dragged around by them; the median
            describes the slope most of the shoreline actually has.

            ``band`` defaults to **half of** ``at_depth``, and that being
            relative rather than absolute matters more than it looks.  A fixed
            band of, say, 0.5 m around a 0.1 m target admits everything from the
            waterline out to 0.6 m depth -- which on a concave-up profile is
            tens of metres of seabed spanning an enormous range of slopes.  The
            median then reports the slope at the median *distance offshore*,
            not at the target depth.  Measured against a Dean beach that reads
            0.51x the analytic answer, at every resolution, so it does not
            announce itself as a discretisation artefact.

        A single number for a whole coastline is a real simplification -- a
        scene with both cliffs and mudflats has no one foreshore slope, and
        everything downstream of this (breaker type, runup, swash width) is
        therefore a scene average rather than a local value.  Making those
        per-cell is a sensible later refinement; it is not what the cookbook
        asks for and it is not what Phase 5 was validated against.
        """
        if self.dean_a is not None:
            y = (at_depth / self.dean_a) ** 1.5
            return float(dean_slope(y, self.dean_a))

        gy, gx = np.gradient(self.terrain_z, self.meta.dx)
        slope = np.hypot(gx, gy)

        if band is None:
            band = 0.5 * at_depth

        wet = self.depth > 0.0
        for width in (band, 2.0 * band, 5.0 * band, 20.0 * band):
            shell = wet & (np.abs(self.depth - at_depth) <= width)
            if shell.sum() >= 16:
                return float(np.median(slope[shell]))

        if not wet.any():
            raise ValueError("no wet cells; cannot measure a foreshore slope")
        return float(np.median(slope[wet]))

    def sample(self, x, y):
        """Bilinear sample of ``(depth, sdf, shore_normal)`` at world points.

        Returns ``(depth, sdf, normal)`` where ``normal`` has shape ``(2, ...)``.
        Coordinates outside the grid clamp to the edge -- these fields are not
        periodic, unlike the wave tiles.
        """
        from scipy.ndimage import map_coordinates

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x0, y0 = self.meta.origin
        coords = np.stack([(y - y0) / self.meta.dx - 0.5,
                           (x - x0) / self.meta.dx - 0.5])

        def grab(field):
            return map_coordinates(field, coords, order=1, mode="nearest")

        nx_ = grab(self.shore_normal[0])
        ny_ = grab(self.shore_normal[1])
        mag = np.hypot(nx_, ny_)
        mag = np.where(mag > 1e-12, mag, 1.0)
        return grab(self.depth), grab(self.sdf), np.stack([nx_ / mag, ny_ / mag])

    # -- the Phase 4 assertion set -------------------------------------------

    def validate(self, sdf_tol: float = 1.0) -> dict[str, float]:
        """The checks cookbook section 4.5 requires of any loaded field set.

        Written against this synthetic data so that the same function can be
        pointed at Houdini's export in Phase 4 without modification.

        ``sdf_tol`` is in metres and bounds the band around the waterline in
        which the depth/sdf sign relation is allowed to disagree: both fields
        are discretised on the same grid, so within about one cell of the
        contour the two signs can legitimately differ.

        Returns the measured quantities, and raises if any check fails.
        """
        ny, nx = self.meta.shape
        if self.depth.shape != (ny, nx):
            raise AssertionError(f"depth shape {self.depth.shape} != {(ny, nx)}")
        if self.sdf.shape != (ny, nx):
            raise AssertionError(f"sdf shape {self.sdf.shape} != {(ny, nx)}")
        if self.shore_normal.shape != (2, ny, nx):
            raise AssertionError(f"shore_normal shape {self.shore_normal.shape}")

        # depth = z_w - terrain_z, exactly
        depth_residual = float(np.max(np.abs(
            self.depth - (self.meta.water_level - self.terrain_z))))
        if depth_residual > 1e-12:
            raise AssertionError(f"depth != z_w - terrain_z (max {depth_residual:.3e})")

        # sign(sdf) == -sign(depth), away from the contour
        far = np.abs(self.sdf) > sdf_tol
        agree = np.sign(self.sdf[far]) == -np.sign(self.depth[far])
        disagree_frac = float(1.0 - agree.mean())
        if disagree_frac > 0.0:
            raise AssertionError(
                f"sign(sdf) != -sign(depth) at {100 * disagree_frac:.3f}% of cells "
                f"further than {sdf_tol} m from the waterline")

        # |shore_normal| = 1 in the nearshore band
        band = np.abs(self.sdf) < 60.0
        norm = np.hypot(self.shore_normal[0], self.shore_normal[1])[band]
        norm_err = float(np.max(np.abs(norm - 1.0)))
        if norm_err > 1e-6:
            raise AssertionError(f"|shore_normal| != 1 (max error {norm_err:.3e})")

        # the normal must point inland, i.e. up the sdf gradient
        gy, gx = np.gradient(self.sdf, self.meta.dx)
        dot = (self.shore_normal[0] * gx + self.shore_normal[1] * gy)[band]
        inland_frac = float((dot > 0).mean())
        if inland_frac < 0.99:
            raise AssertionError(
                f"shore_normal points inland at only {100 * inland_frac:.1f}% of "
                f"nearshore cells")

        return {
            "depth_residual": depth_residual,
            "sdf_sign_disagreement": disagree_frac,
            "shore_normal_magnitude_error": norm_err,
            "shore_normal_inland_fraction": inland_frac,
            "max_depth": float(self.depth.max()),
            "beach_slope_at_10cm": self.beach_slope(),
        }
