"""PHASE 5b -- refraction by ray integration and energy accumulation.

Design: ``docs/phase5b_refraction.md``.

Why this exists
---------------
:func:`pywave.nearshore.refraction_coefficient` computes

    Kr = sqrt(|cos alpha_deep| / |cos alpha_local|)

which is a **ray-tube result derived for straight, parallel depth contours**,
evaluated per cell from ``shore_normal`` -- the direction to the nearest shore.
Two things follow, and both are fatal on a real coastline:

* ``Kr`` compares ray spacing *here* against ray spacing in deep water. That is a
  statement about a **path**, and a cell has no memory of where the wave came
  from.
* ``shore_normal`` is discontinuous on the medial axis of any distance-derived
  bed -- measured at up to 180 degrees per cell -- so ``Kr`` inherits a
  discontinuity that is not in the water.

This module replaces the formula with a measurement. Rays are integrated through
the celerity field and their energy is accumulated onto a grid; ``Kr`` stops
being assumed and becomes whatever the ray density turns out to be.

**The property that fixes the bug:** wave celerity ``c`` is a function of *depth
alone*. Depth is continuous wherever the bed is, so ``c`` is continuous, so the
ray paths are continuous. ``shore_normal`` is never read. Gate 5b.6 asserts
exactly that.

Physics
-------
For steady bathymetry the absolute frequency ``omega`` is conserved along a ray,
and the local wavenumber solves the dispersion relation ``omega^2 = g k tanh(kd)``.
With phase speed ``c = omega/k`` the ray equations are (Dean & Dalrymple 1991):

    dx/ds = cos(theta)
    dy/ds = sin(theta)
    dtheta/ds = (1/c) [ sin(theta) dc/dx - cos(theta) dc/dy ]

The last line bends rays toward slower water, which is to say toward shallower
water -- the whole of refraction in one term.

Energy, not tube width
----------------------
The textbook route to ``Kr`` is to track the separation ``b`` of neighbouring
rays and use ``Kr = sqrt(b_0/b)``. That divides by a quantity which goes to zero
at caustics, which is where refraction is most interesting.

Instead each ray carries power and deposits it along its path. Energy density in
a cell is what the rays leave behind:

    E_cell = sum_rays  P * ds / (c_g * A_cell)

Convergence raises the density because more rays arrive, not because a
denominator shrinks. That is bounded, continuous when enough rays are cast, and
needs no special case at a caustic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import G

__all__ = ["celerity_fields", "launch_line", "trace_rays", "RayField"]


# ---------------------------------------------------------------------------
# The medium
# ---------------------------------------------------------------------------


def celerity_fields(depth: np.ndarray, omega: float, dx: float,
                    min_depth: float = 0.05):
    """``(c, cg, dc_dx, dc_dy)`` on the bathymetry grid, for one frequency.

    ``min_depth`` floors the depth used for the wave solution. Celerity goes to
    zero at the waterline and its gradient diverges, which would make the ray
    integrator take infinitely small steps for a wave that has already broken.
    The floor is a numerical bound, not physics; rays are absorbed at the
    breaking depth long before it matters.
    """
    from .spectrum import dispersion_k, group_velocity

    d = np.maximum(np.asarray(depth, dtype=np.float64), min_depth)
    k = dispersion_k(omega, d)
    c = omega / np.maximum(k, 1e-12)
    cg = group_velocity(k, d)

    # Central differences. c inherits depth's smoothness and nothing else --
    # in particular it does not inherit shore_normal's discontinuity, which is
    # the entire point of this module.
    dc_dy, dc_dx = np.gradient(c, dx)
    return c, cg, dc_dx, dc_dy


def _bilinear(field: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Sample ``field`` at fractional grid coordinates, clamped at the edges."""
    ny, nx = field.shape
    x0 = np.clip(np.floor(gx).astype(np.int64), 0, nx - 2)
    y0 = np.clip(np.floor(gy).astype(np.int64), 0, ny - 2)
    tx = np.clip(gx - x0, 0.0, 1.0)
    ty = np.clip(gy - y0, 0.0, 1.0)
    return ((1 - ty) * ((1 - tx) * field[y0, x0] + tx * field[y0, x0 + 1])
            + ty * ((1 - tx) * field[y0 + 1, x0] + tx * field[y0 + 1, x0 + 1]))


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def launch_line(x_from, x_to, y, theta, n_rays, omega, depth_at_launch):
    """A uniform beam across a line, with the reference energy density it carries.

    Getting this wrong is easy and the error is silent. Rays are spaced along the
    launch *line*, but the quantity that sets energy density is their
    **perpendicular** separation ``b = spacing * cos(alpha)``, where ``alpha`` is
    the angle between the ray and the line's normal. Omitting the cosine inflates
    the gain by ``1/sqrt(cos alpha)`` -- 3.2% at 20 degrees, 14.3% at 40 -- which
    looks like a plausible physical result rather than a bug.

    Returns ``(x0, y0, theta0, e_ref)``; divide the accumulated energy by
    ``e_ref`` and take the square root to get gain.
    """
    from .spectrum import dispersion_k, group_velocity

    x0 = np.linspace(x_from, x_to, int(n_rays))
    y0 = np.full(int(n_rays), float(y))
    th0 = np.full(int(n_rays), float(theta))

    spacing = (x_to - x_from) / max(int(n_rays) - 1, 1)
    # Angle between the ray and the launch line's normal (the y axis here).
    alpha = np.arctan2(np.cos(theta), -np.sin(theta))
    b = spacing * abs(np.cos(alpha))

    cg = group_velocity(dispersion_k(omega, depth_at_launch), depth_at_launch)
    e_ref = 1.0 / (b * cg)
    return x0, y0, th0, float(e_ref)


def trace_rays(depth, omega, dx, x0, y0, theta0, *, origin=(0.0, 0.0),
               power=None, ds=None, max_steps=None, break_depth=0.0,
               min_depth=0.05, wrap_x=False):
    """March rays through the celerity field, accumulating energy density.

    Every ray advances together as an array; there is no Python loop over rays.

    Parameters
    ----------
    depth : ``(ny, nx)`` still-water depth [m].
    omega : angular frequency [rad/s], conserved along every ray.
    dx : grid spacing [m].
    x0, y0, theta0 : launch position [m] and direction [rad], one per ray.
    origin : ``(x, y)`` of grid cell (0, 0) in scene coordinates [m].
    power : per-ray power, arbitrary units. Defaults to 1 for every ray; pass a
        weight array to impose a directional spread.
    ds : step length [m]. Defaults to ``dx/2`` -- half a cell, so no cell is
        stepped over.
    break_depth : rays are retired once shallower than this. Beyond breaking the
        ray approximation has no meaning and the energy is not conserved anyway.
    wrap_x : wrap rays through the x boundaries instead of retiring them. Correct
        only for a domain that is genuinely periodic alongshore -- a synthetic
        planar beach -- and wrong for real terrain. It exists so the straight-beach
        validation is not corrupted by rays leaving the side: at 40 degrees
        incidence the alongshore drift over a 400 m cross-shore run is 336 m, so
        without it the middle of the domain is fed by rays that were never
        launched, and the gain reads zero.

    Returns
    -------
    ``(energy, hits)`` -- accumulated energy density and a ray-visit count, both
    on the bathymetry grid. ``hits`` is diagnostic: cells with few visits have a
    noisy answer and the caller should say so rather than pretend otherwise.
    """
    depth = np.asarray(depth, dtype=np.float64)
    ny, nx = depth.shape
    c, cg, dc_dx, dc_dy = celerity_fields(depth, omega, dx, min_depth)

    ds = float(ds if ds is not None else 0.5 * dx)
    if max_steps is None:
        max_steps = int(4 * (nx + ny) * dx / ds)

    x = np.asarray(x0, dtype=np.float64).copy()
    y = np.asarray(y0, dtype=np.float64).copy()
    th = np.asarray(theta0, dtype=np.float64).copy()
    p = (np.ones_like(x) if power is None
         else np.asarray(power, dtype=np.float64).copy())
    alive = np.ones(x.shape, dtype=bool)

    energy = np.zeros_like(depth)
    hits = np.zeros_like(depth)
    ox, oy = origin
    area = dx * dx

    def grid_coords(xx, yy):
        return (xx - ox) / dx, (yy - oy) / dx

    for _ in range(max_steps):
        if not alive.any():
            break
        xa, ya, tha = x[alive], y[alive], th[alive]
        gx, gy = grid_coords(xa, ya)

        c_r = _bilinear(c, gx, gy)
        cg_r = _bilinear(cg, gx, gy)
        dcx = _bilinear(dc_dx, gx, gy)
        dcy = _bilinear(dc_dy, gx, gy)

        # Deposit before stepping, so a ray contributes to the cell it is in.
        # Bilinear scatter rather than nearest-cell: with nearest-cell the
        # accumulator inherits the ray-launch pattern as visible striping.
        w = p[alive] * ds / (np.maximum(cg_r, 1e-9) * area)
        ix = np.clip(np.floor(gx).astype(np.int64), 0, nx - 2)
        iy = np.clip(np.floor(gy).astype(np.int64), 0, ny - 2)
        fx = np.clip(gx - ix, 0.0, 1.0)
        fy = np.clip(gy - iy, 0.0, 1.0)
        for jy, jx, wt in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)),
                           (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
            np.add.at(energy, (iy + jy, ix + jx), w * wt)
            np.add.at(hits, (iy + jy, ix + jx), wt)

        # Midpoint (RK2). Refraction turns rays through large angles over a few
        # cells near a headland, and forward Euler visibly cuts those corners.
        dth = (np.sin(tha) * dcx - np.cos(tha) * dcy) / np.maximum(c_r, 1e-9)
        th_m = tha + 0.5 * ds * dth
        x_m = xa + 0.5 * ds * np.cos(tha)
        y_m = ya + 0.5 * ds * np.sin(tha)
        gxm, gym = grid_coords(x_m, y_m)
        c_m = _bilinear(c, gxm, gym)
        dcx_m = _bilinear(dc_dx, gxm, gym)
        dcy_m = _bilinear(dc_dy, gxm, gym)
        dth_m = (np.sin(th_m) * dcx_m - np.cos(th_m) * dcy_m) / np.maximum(c_m, 1e-9)

        x[alive] = xa + ds * np.cos(th_m)
        y[alive] = ya + ds * np.sin(th_m)
        th[alive] = tha + ds * dth_m

        if wrap_x:
            span = nx * dx
            x[alive] = ox + np.mod(x[alive] - ox, span)

        gx2, gy2 = grid_coords(x[alive], y[alive])
        inside = (gy2 >= 0) & (gy2 <= ny - 1)
        if not wrap_x:
            inside &= (gx2 >= 0) & (gx2 <= nx - 1)
        deep_enough = _bilinear(depth, np.clip(gx2, 0, nx - 1),
                                np.clip(gy2, 0, ny - 1)) > break_depth
        still = inside & deep_enough
        idx = np.flatnonzero(alive)
        alive[idx[~still]] = False

    return energy, hits
