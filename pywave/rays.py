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

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import G

__all__ = ["celerity_fields", "launch_line", "trace_rays", "shelter_fetch",
           "band_fetch_response", "wind_sea_floor", "RayField",
           "BandedRayField", "CACHE_VERSION"]


CACHE_VERSION = 1
"""Bump on any change to this module that would change the answer.

A cache that quietly returns a stale field is worse than having no cache at all:
the solve is the slowest thing in the scene, so a wrong one is exactly the sort
of result nobody re-runs to check. Every other input is hashed automatically --
this constant covers the one thing hashing cannot see, which is the code.
"""


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


@dataclass
class RayAccumulator:
    """What a ray sweep leaves behind on the grid.

    A record rather than a tuple because the direction moments are meaningless
    on their own -- ``e_cos`` and ``e_sin`` are only ever read through
    :attr:`theta`, and a caller unpacking four bare arrays would have to know
    that.
    """

    energy: np.ndarray
    """Accumulated energy density, arbitrary units."""
    hits: np.ndarray
    """Ray-visit count. Low means a noisy answer; diagnostic only."""
    e_cos: np.ndarray
    """Energy-weighted sum of ``cos(theta)``."""
    e_sin: np.ndarray
    """Energy-weighted sum of ``sin(theta)``."""

    @property
    def theta(self) -> np.ndarray:
        """Energy-weighted mean direction of travel [rad], CCW from +X.

        ``atan2`` of the summed components, which is the circular mean. Not the
        mean of the angles: rays at +179 and -179 degrees travel almost the same
        way and their arithmetic mean points backwards.
        """
        return np.arctan2(self.e_sin, self.e_cos)

    def flux(self, cg: np.ndarray):
        """``(Fx, Fy)`` energy flux density, given the local group velocity.

        Read straight off the direction moments rather than reconstructed from
        :attr:`theta`, and the difference is not cosmetic. ``E * sin(theta)``
        uses the sine of the *mean* direction; the flux needs the *mean of the
        sines*, which is exactly what ``e_sin`` already holds. The two agree
        only where every ray through a cell travels the same way.

        Measured on a shoal that focuses without breaking: flux across lines
        from deep water to the far side is conserved to 0.024% through these
        moments, against a 5.8% drift through ``sin(theta)`` -- all of it
        downstream of the caustic, where rays cross and
        :attr:`directionality` falls to 0.87.
        """
        return cg * self.e_cos, cg * self.e_sin

    @property
    def directionality(self) -> np.ndarray:
        """``|sum of unit vectors| / sum of weights``, in ``[0, 1]``.

        1 where every ray through a cell travels the same way, 0 where they
        cancel. It is the honest companion to :attr:`theta`: in a cell fed from
        two directions at once the mean direction is a real number and a poor
        description, and only this says so.
        """
        mag = np.hypot(self.e_cos, self.e_sin)
        return np.where(self.energy > 0, mag / np.maximum(self.energy, 1e-30), 0.0)


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
    A :class:`RayAccumulator` carrying energy density, a ray-visit count, and
    the direction moments. ``hits`` is diagnostic: cells with few visits have a
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
    # Rays are usually launched *outside* the grid so they arrive with the
    # right direction and spacing rather than being born on the boundary. They
    # must therefore be allowed to fly in: a ray is retired when it has been
    # inside and then left, not merely for being outside. Without this
    # distinction every ray dies on its first step and the field comes back
    # empty -- measured once at 99.88% of wet cells never visited.
    entered = np.zeros(x.shape, dtype=bool)

    energy = np.zeros(ny * nx, dtype=np.float64)
    hits = np.zeros(ny * nx, dtype=np.float64)
    # Energy-weighted direction, accumulated as its two components rather than
    # as an angle. Averaging angles directly is wrong at the branch cut -- rays
    # at +179 and -179 degrees are travelling almost the same way and average to
    # 0, which points them backwards. Summing the unit vectors and taking atan2
    # at the end has no branch cut and weights by energy for free.
    e_cos = np.zeros(ny * nx, dtype=np.float64)
    e_sin = np.zeros(ny * nx, dtype=np.float64)
    ox, oy = origin
    area = dx * dx

    buf_idx: list = []
    buf_e: list = []
    buf_h: list = []
    buf_c: list = []
    buf_s: list = []
    buffered = 0
    flush_every = 8_000_000

    def _flush():
        nonlocal buffered
        if not buf_idx:
            return
        idx = np.concatenate(buf_idx)
        for acc, buf in ((energy, buf_e), (hits, buf_h),
                         (e_cos, buf_c), (e_sin, buf_s)):
            acc[:] += np.bincount(idx, weights=np.concatenate(buf),
                                  minlength=ny * nx)
        buf_idx.clear(); buf_e.clear(); buf_h.clear()
        buf_c.clear(); buf_s.clear()
        buffered = 0

    def grid_coords(xx, yy):
        return (xx - ox) / dx, (yy - oy) / dx

    for _ in range(max_steps):
        if not alive.any():
            break
        idx_alive = np.flatnonzero(alive)
        xa, ya, tha = x[alive], y[alive], th[alive]
        gx, gy = grid_coords(xa, ya)
        in_now = (gx >= 0) & (gx <= nx - 1) & (gy >= 0) & (gy <= ny - 1)
        entered[idx_alive] |= in_now

        c_r = _bilinear(c, gx, gy)
        cg_r = _bilinear(cg, gx, gy)
        dcx = _bilinear(dc_dx, gx, gy)
        dcy = _bilinear(dc_dy, gx, gy)

        # Deposit before stepping, so a ray contributes to the cell it is in.
        # Bilinear scatter rather than nearest-cell: with nearest-cell the
        # accumulator inherits the ray-launch pattern as visible striping.
        #
        # Buffered, because the scatter is the whole cost of the solve. Measured
        # on a 2150x1874 grid: np.add.at runs at 495 ns/entry on the ~40k entries
        # one step produces, while np.bincount over a batched 8M entries runs at
        # 27 ns. Same arithmetic, 18x apart -- the per-call overhead of touching
        # a 4M-cell accumulator simply has to be amortised.
        w = p[alive] * ds / (np.maximum(cg_r, 1e-9) * area) * in_now
        ix = np.clip(np.floor(gx).astype(np.int64), 0, nx - 2)
        iy = np.clip(np.floor(gy).astype(np.int64), 0, ny - 2)
        fx = np.clip(gx - ix, 0.0, 1.0)
        fy = np.clip(gy - iy, 0.0, 1.0)
        base = iy * nx + ix
        w_cos = w * np.cos(tha)
        w_sin = w * np.sin(tha)
        for off, wt in ((0, (1 - fx) * (1 - fy)), (1, fx * (1 - fy)),
                        (nx, (1 - fx) * fy), (nx + 1, fx * fy)):
            buf_idx.append(base + off)
            buf_e.append(w * wt)
            buf_h.append(wt * in_now)
            buf_c.append(w_cos * wt)
            buf_s.append(w_sin * wt)
        buffered += 4 * gx.size
        if buffered >= flush_every:
            _flush()

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
        d_here = _bilinear(depth, np.clip(gx2, 0, nx - 1),
                           np.clip(gy2, 0, ny - 1))
        was_in = entered[idx_alive]
        # Retire on running aground, or on leaving after having arrived. A ray
        # still on its way in is neither.
        still = np.where(was_in, inside & (d_here > break_depth), True)
        alive[idx_alive[~still]] = False

    _flush()
    return RayAccumulator(energy=energy.reshape(ny, nx),
                          hits=hits.reshape(ny, nx),
                          e_cos=e_cos.reshape(ny, nx),
                          e_sin=e_sin.reshape(ny, nx))


# ---------------------------------------------------------------------------
# What the rays cannot carry: the locally generated sea
# ---------------------------------------------------------------------------

# Hs grows as fetch^0.55 in this model, so energy grows as fetch^1.10.
#
# Not a fitted number and not a round one. JONSWAP's dimensionless-energy law
# gives ``Hs ~ sqrt(X~)``, and this package's spectrum sits a further
# ``X~^0.05`` above that fit -- the documented inconsistency between JONSWAP's
# two independently fitted power laws, see spectrum.hs_ratio_spectral_to_fit.
# Since ``X~`` is linear in fetch, the exponent that matters here is the sum.
#
# The extra 0.05 makes this spectrum grow *faster* with fetch, so a short-fetch
# cell sits *lower* than the plain energy law says -- 12% lower in Hs per decade
# of fetch ratio, 26% at a hundredth of the scene fetch. Using 0.5 would
# therefore over-floor every sheltered cell, against a sea the surface synthesis
# does not produce. test_5b3b pins this against `hs_spectral` itself.
#
# Note the direction on the energy side, which is easy to read backwards: the
# fetch ratio is capped at 1, so the 1.10 exponent always yields *less* energy
# than 1.00 would, by 21% at a tenth of the scene fetch and 37% at a hundredth.
_HS_FETCH_EXPONENT = 0.50 + 0.05
_ENERGY_FETCH_EXPONENT = 2.0 * _HS_FETCH_EXPONENT


def shelter_fetch(depth, dx, thetas, *, origin=(0.0, 0.0), max_fetch=None,
                  min_depth_for_land: float = 0.0):
    """Distance upwind to the nearest land, per cell, per direction.

    Returns ``(n_dirs, ny, nx)`` in metres, with ``inf`` where the path is
    **clear** -- meaning it left the domain, or reached ``max_fetch``, without
    crossing land. The distinction between "blocked at 200 m" and "clear" is the
    whole content of this function; a clear direction is one the rays already
    account for, and only a blocked one has a sheltered sea behind it.

    ``thetas`` are the directions the waves travel **toward**, matching the ray
    fan, so the march is backwards along each of them.

    The march is one grid cell per step with an active set that shrinks as cells
    resolve, which is what keeps this affordable: on a real coastline most cells
    hit land within a few hundred metres and drop out immediately, and the ones
    that do not leave the domain.
    """
    depth = np.asarray(depth, dtype=np.float64)
    ny, nx = depth.shape
    ox, oy = origin
    land = depth <= min_depth_for_land
    if max_fetch is None:
        max_fetch = float(dx * (nx + ny))

    yy, xx = np.mgrid[0:ny, 0:nx]
    x_home = ox + xx.ravel() * dx
    y_home = oy + yy.ravel() * dx
    wet0 = ~land.ravel()

    out = np.full((len(thetas), ny * nx), np.inf)
    n_steps = int(np.ceil(max_fetch / dx)) + 1

    for i, th in enumerate(thetas):
        # Backwards: upwind is the direction the waves came from.
        vx, vy = -np.cos(th) * dx, -np.sin(th) * dx
        active = np.flatnonzero(wet0)
        x = x_home[active].copy()
        y = y_home[active].copy()
        for step in range(1, n_steps + 1):
            x += vx
            y += vy
            gx = (x - ox) / dx
            gy = (y - oy) / dx
            inside = (gx >= 0) & (gx <= nx - 1) & (gy >= 0) & (gy <= ny - 1)
            # Nearest cell, not bilinear: "is this land" is a question about a
            # mask, and interpolating a mask invents half-land that is neither.
            ix = np.clip(np.rint(gx).astype(np.int64), 0, nx - 1)
            iy = np.clip(np.rint(gy).astype(np.int64), 0, ny - 1)
            hit = inside & land[iy, ix]

            out[i, active[hit]] = step * dx
            keep = inside & ~hit
            if not keep.any():
                break
            active = active[keep]
            x = x[keep]
            y = y[keep]

    return out.reshape(len(thetas), ny, nx)


def band_fetch_response(bands, u10, scene_fetch, gamma, *, n_samples: int = 64,
                        g: float = G):
    """``m0_band(F) / m0_band(F_scene)`` per band, tabulated on log fetch.

    The scalar floor uses ``(F/F_scene)^1.10``, which is the *total* energy
    ratio and says nothing about where that energy sits in frequency. That is
    not a detail. A short fetch does not scale the spectrum down, it **moves the
    peak up**, so a sheltered bay is short steep chop with no swell in it --
    not a small copy of the incident swell.

    Measured on the straits crop, as a multiplier on each band's own deep-water
    energy:

    ==============  ========  ========  ========  ==================
    local fetch     band 1    band 2    band 3    scalar
    ==============  ========  ========  ========  ==================
    2000 m          0.052     1.222     1.125     0.252
    500 m           0.000     0.023     1.057     0.055
    200 m           0.000     0.000     0.403     0.020
    ==============  ========  ========  ========  ==================

    The scalar is wrong in both directions -- 5x too much energy in the long
    band, 20x too little in the short one -- and note bands 2 and 3 exceeding
    1.0, which a ratio capped at 1 cannot express at all. A sheltered cell
    genuinely has *more* short-wave energy than open water, because the whole
    spectrum shifted into that band.

    What this does **not** change is the total: summed over bands weighted by
    their deep-water shares, this agrees with ``(F/F_scene)^1.10`` to 4e-4. So
    gate 5b.3, which is about total height, is indifferent to it. What depends
    on it is every per-band quantity downstream.

    Returns ``(log_fetch, table)`` with ``table.shape == (len(bands), n_samples)``,
    ready for :func:`numpy.interp` on ``log(F)``.
    """
    from .moments import moment_omega
    from .spectrum import dispersion_omega

    bands = [(float(lo), float(hi)) for lo, hi in bands]
    # The shortest fetch worth tabulating is one where the sea is negligible;
    # below it the table is flat at ~0 and the interpolation clamps.
    f_lo_fetch = max(float(scene_fetch) * 1e-5, 1e-3)
    fetches = np.geomspace(f_lo_fetch, float(scene_fetch), int(n_samples))

    def _f_of_k(k):
        return float(dispersion_omega(np.array(max(k, 1e-12)), None, g)) / (2 * np.pi)

    table = np.zeros((len(bands), len(fetches)))
    for b, (k_lo, k_hi) in enumerate(bands):
        lo = _f_of_k(k_lo) if k_lo > 0 else 1e-6
        hi = _f_of_k(k_hi)
        ref = moment_omega(0, u10, float(scene_fetch), gamma, f_lo=lo, f_hi=hi, g=g)
        if ref <= 0:
            continue
        for i, f in enumerate(fetches):
            table[b, i] = moment_omega(0, u10, float(f), gamma,
                                       f_lo=lo, f_hi=hi, g=g) / ref
    return np.log(fetches), table


def wind_sea_floor(depth, dx, thetas, weights, scene_fetch, *,
                   origin=(0.0, 0.0), max_fetch=None,
                   bands=None, u10=None, gamma=None):
    """Energy a sheltered cell has anyway, relative to the deep-water sea.

    Ray theory has no diffraction, so a cell no ray reaches gets exactly zero.
    That is a statement about the *incident* wave field and it is fine as far as
    it goes -- but a sheltered bay is not calm. The wind is still blowing over
    it, and it grows its own short-fetch sea.

    So for each direction of the fan, ask how far upwind the water runs before
    it hits land. A **blocked** direction contributes the energy the wind puts
    back in over that distance; a **clear** direction contributes nothing here,
    because the rays already carried it, and adding it again would count the
    open sea twice. In fully exposed water every direction is clear and this
    returns exactly zero, which is what makes it safe to add rather than clamp.

    ``weights`` are the fan's directional weights and must sum to 1, so that a
    hypothetical cell blocked at the full scene fetch in every direction scores
    exactly 1 rather than ``n_dirs``.

    Pass ``bands`` (with ``u10`` and ``gamma``) to get the floor resolved by
    spectral band instead of as one number -- see :func:`band_fetch_response`
    for why one number is not enough. Without it the closed-form total is used,
    which is right for height and silent about frequency.

    Returns ``(ny, nx)``, or ``(n_bands, ny, nx)`` when ``bands`` is given, in
    the same units as ``(ray gain)**2`` -- for bands, relative to *that band's*
    own deep-water energy, which is how ``nearshore.transform`` weights each
    tile.
    """
    fetch = shelter_fetch(depth, dx, thetas, origin=origin, max_fetch=max_fetch)
    blocked = np.isfinite(fetch)          # a clear direction contributes nothing
    w = np.asarray(weights, dtype=np.float64)

    if bands is None:
        ratio = np.where(blocked, fetch / float(scene_fetch), 0.0)
        return np.sum(w[:, None, None]
                      * np.minimum(ratio, 1.0) ** _ENERGY_FETCH_EXPONENT, axis=0)

    if u10 is None or gamma is None:
        raise ValueError("bands= needs u10= and gamma= to evaluate the "
                         "short-fetch spectrum; they cannot be inferred from "
                         "the bathymetry")
    log_f, table = band_fetch_response(bands, u10, scene_fetch, gamma)
    safe = np.clip(np.where(blocked, fetch, 1.0), np.exp(log_f[0]),
                   float(scene_fetch))
    log_safe = np.log(safe)
    out = np.empty((len(table),) + fetch.shape[1:])
    for b in range(len(table)):
        r = np.where(blocked, np.interp(log_safe, log_f, table[b]), 0.0)
        out[b] = np.tensordot(w, r, axes=(0, 0))
    return out


# ---------------------------------------------------------------------------
# The scene-level solve
# ---------------------------------------------------------------------------


@dataclass
class RayField:
    """Refraction and shoaling gain over a whole bathymetry, from ray density.

    A scene-level artifact, like the foam spin-up: the wave field is
    **stationary** given a bed and a sea state, so this is solved once and then
    sampled per frame. Per-frame cost is a bilinear interpolation.
    """

    gain: np.ndarray
    """``(ny, nx)`` amplitude gain, 1.0 in deep water."""
    hits: np.ndarray
    """``(ny, nx)`` ray-visit density. Low counts mean a noisy answer."""
    theta: np.ndarray
    """``(ny, nx)`` energy-weighted mean direction of travel [rad], CCW from +X.

    What ``nearshore.transform`` needs to rotate each tile by, and the half of
    refraction that ``gain`` does not carry: a wave can turn without changing
    height, and on a mild slope at normal incidence it mostly does.
    """
    directionality: np.ndarray
    """``(ny, nx)`` in ``[0, 1]``: how uni-directional the sea is at that cell.

    1 where every ray agrees, 0 where they cancel. Read it before trusting
    :attr:`theta` -- behind a headland, energy arrives from both sides of the
    obstacle at once, and a mean direction there is a real number describing a
    sea that has no single direction.
    """
    omega: float
    meta: dict = field(default_factory=dict)

    @classmethod
    def solve(cls, bathy, cfg, omega: float, *, n_dirs: int = 21,
              rays_per_dir: int = 3000, spread_deg: float = 90.0,
              ds_frac: float = 0.5, break_depth: float = 0.05,
              min_hits: float = 1.0, decimate: int = 1,
              smooth_m: float | None = None,
              wind_sea: bool = True,
              band: tuple[float, float] | None = None) -> "RayField":
        """Integrate a directional wave field across the bathymetry.

        A single beam would give hard geometric shadows behind every headland --
        ray theory has no diffraction, so a sheltered cell gets exactly zero.
        Launching a **fan** weighted by the spreading function is not a
        refinement bolted on afterwards; it is what makes shadow zones physical,
        because a real sea arrives from a range of directions and the edges of
        that range reach in behind obstacles.

        Rays enter from a line upwind of the whole domain, perpendicular to their
        own direction, so deep water is seeded uniformly whatever the geometry.

        ``decimate`` and ``smooth_m`` exist because energy deposition is Monte
        Carlo, and Monte Carlo on a fine grid is shot noise. Both are physical
        rather than cosmetic:

        * The gain field varies over the **bathymetry's** length scale --
          hundreds of metres -- not the mesh's. Solving on a coarser grid and
          interpolating loses nothing real, and gives ``decimate**2`` times more
          ray visits per cell.
        * A ray is a wave *packet*, not a line. Energy cannot be localised finer
          than about a wavelength, so a deposition kernel of that width is more
          correct than a point, not less.

        Measured on a 4 m Strait of Hormuz export at the defaults: 47 ray visits
        per cell, giving 7.3% gain noise and a p99.9 one-cell jump of 0.753 --
        which is sampling, not water. Reaching Gate 5b.2 by ray count alone would
        need 27x more sampling and about seven hours.

        ``wind_sea`` adds the locally generated sea of :func:`wind_sea_floor` to
        the transported energy. It is the one term here that is not ray theory,
        and it is on by default because leaving it off does not give a
        conservative answer -- it gives a sheltered bay a wave height of exactly
        zero, which is further from the truth than any approximation in it.

        ``band`` is the ``(k_lo, k_hi)`` this solve represents, in rad/m. It
        changes only the wind sea, which is then integrated over that band of
        the short-fetch spectrum rather than taken from the closed-form total --
        the difference between a sheltered bay made of chop and one made of
        scaled-down swell. The rays themselves already carry the band through
        ``omega``. Leave it ``None`` for a single-frequency solve over the whole
        spectrum, which is what the standalone gates measure.
        """
        from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom

        from .moments import spreading

        full_depth = np.asarray(bathy.depth, dtype=np.float64)
        full_shape = full_depth.shape
        dec = max(int(decimate), 1)
        depth = full_depth[::dec, ::dec] if dec > 1 else full_depth
        ny, nx = depth.shape
        dx = float(bathy.meta.dx) * dec
        x0g, y0g = bathy.meta.extent[0], bathy.meta.extent[2]
        ds = ds_frac * dx

        # Directions, weighted by how much energy the sea actually sends each way.
        half = np.radians(spread_deg)
        rel = np.linspace(-half, half, n_dirs)
        th_w = cfg.wind.direction_rad
        d_theta = rel[1] - rel[0] if n_dirs > 1 else 2 * half
        w = spreading(th_w + rel, np.full(n_dirs, cfg.f_p), cfg.f_p, th_w,
                      model=cfg.spectrum.spreading)
        w = np.asarray(w, dtype=np.float64) * d_theta
        w = w / w.sum()

        corners = np.array([[x0g, y0g], [x0g + nx * dx, y0g],
                            [x0g, y0g + ny * dx], [x0g + nx * dx, y0g + ny * dx]])
        d_ref = float(np.percentile(depth[depth > 0], 95))

        energy = np.zeros((ny, nx))
        hits = np.zeros((ny, nx))
        e_cos = np.zeros((ny, nx))
        e_sin = np.zeros((ny, nx))
        for theta, weight in zip(th_w + rel, w):
            if weight <= 0:
                continue
            fwd = np.array([np.cos(theta), np.sin(theta)])
            perp = np.array([-np.sin(theta), np.cos(theta)])
            # Span the domain's shadow on the perpendicular, and start behind it.
            a = corners @ perp
            b = corners @ fwd
            pad = 2.0 * dx
            s0 = np.linspace(a.min() - pad, a.max() + pad, rays_per_dir)
            start = b.min() - pad
            xs = start * fwd[0] + s0 * perp[0]
            ys = start * fwd[1] + s0 * perp[1]

            spacing = (s0[-1] - s0[0]) / max(rays_per_dir - 1, 1)
            acc = trace_rays(depth, omega, dx,
                             xs, ys, np.full(rays_per_dir, theta),
                             origin=(x0g, y0g),
                             power=np.full(rays_per_dir, weight * spacing),
                             ds=ds, break_depth=break_depth)
            energy += acc.energy
            hits += acc.hits
            # Summed across the fan before any angle is taken, so the mean
            # direction is over the whole directional sea rather than per beam.
            e_cos += acc.e_cos
            e_sin += acc.e_sin

        # Normalise on deep water, where by construction nothing has refracted
        # yet and the gain must be exactly 1. Doing it empirically rather than
        # analytically also absorbs the discretisation of the fan.
        wet = depth > 0.0

        def _smooth_wet(fld):
            """Gaussian over water only, normalised by the smoothed wet mask.

            Without the mask normalisation the kernel averages dry cells into
            the water and every shoreline loses height.
            """
            sig = float(smooth_m) / dx
            num = gaussian_filter(np.where(wet, fld, 0.0), sig, mode="nearest")
            den = gaussian_filter(wet.astype(np.float64), sig, mode="nearest")
            return np.where(den > 1e-6, num / np.maximum(den, 1e-6), fld)

        # Smooth the ENERGY, not the gain: energy is what the rays sampled and
        # is the quantity whose noise is Poisson.
        if smooth_m:
            energy = _smooth_wet(energy)

        deep = wet & (depth >= d_ref)
        ref = float(np.median(energy[deep])) if deep.sum() > 50 else \
            float(np.median(energy[wet]))
        e_gain = np.maximum(energy, 0.0) / max(ref, 1e-30)

        # The locally generated sea, added *after* the normalisation, because it
        # is stated relative to the scene's own deep-water energy rather than to
        # anything the rays measured.
        floor_e = 0.0
        if wind_sea:
            extra = ({} if band is None
                     else dict(bands=[band], u10=cfg.wind.speed,
                               gamma=cfg.spectrum.gamma))
            floor_e = wind_sea_floor(depth, dx, th_w + rel, w,
                                     cfg.wind.fetch, origin=(x0g, y0g),
                                     max_fetch=cfg.wind.fetch, **extra)
            if band is not None:
                floor_e = floor_e[0]
            # Smoothed with the same kernel as the ray energy, and for a reason
            # that is not the same one. The ray field is smoothed because it is
            # Monte Carlo; this is smoothed because `shelter_fetch` asks a
            # yes/no question of a geometric line, so a cell whose upwind ray
            # grazes a headland tip is "blocked at 6 km" while its neighbour is
            # "not blocked at all". The fan bounds that step at one direction's
            # weight, which is not small enough: unsmoothed it took the p99.9
            # gain jump on the straits export from 0.0123 to 0.0235, straight
            # through gate 5b.2. A wind sea has no knife edge either.
            if smooth_m:
                floor_e = _smooth_wet(floor_e)
            e_gain = e_gain + floor_e

        gain = np.sqrt(e_gain)
        gain = np.where(wet, gain, 0.0)

        # Smooth the direction as its two components, for the same reason they
        # are accumulated that way: a Gaussian over an angle field averages
        # across the branch cut and produces directions no ray ever travelled.
        if smooth_m:
            e_cos, e_sin = _smooth_wet(e_cos), _smooth_wet(e_sin)
        mag = np.hypot(e_cos, e_sin)
        directionality = np.where(wet & (energy > 0),
                                  mag / np.maximum(energy, 1e-30), 0.0)
        # Carried as a unit vector rather than an angle all the way through the
        # upsample below, for the same branch-cut reason.
        u = np.where(mag > 0, e_cos / np.maximum(mag, 1e-30), 1.0)
        v = np.where(mag > 0, e_sin / np.maximum(mag, 1e-30), 0.0)

        thin = int((wet & (hits < min_hits)).sum())
        visits = float(hits[wet].mean()) if wet.any() else 0.0
        if dec > 1:
            # Extend the gain into the dry cells before upsampling. Bilinear
            # interpolation mixes neighbours, so leaving the zeros there drags
            # them into the water within one *coarse* cell of every shoreline --
            # a seam that has nothing to do with the wave field and that no
            # amount of smoothing removes, because it is created after the
            # smoothing. Measured on the straits crop at decimate=16: every one
            # of the worst 0.1% of 4 m gain jumps sat inside that 4 m band, and
            # p99.9 fell from 0.396 to 0.026 when they stopped being generated.
            #
            # Nearest-wet fill rather than a normalised zoom: water thinner than
            # a coarse cell has no wet coarse neighbour at all, so the division
            # a normalised zoom would do is 0/0 exactly where the fill matters.
            src = distance_transform_edt(~wet, return_distances=False,
                                         return_indices=True)
            zf = (full_shape[0] / ny, full_shape[1] / nx)
            crop = (slice(0, full_shape[0]), slice(0, full_shape[1]))
            gain, u, v, directionality = (
                zoom(f[tuple(src)], zf, order=1)[crop]
                for f in (gain, u, v, directionality))
            hits = zoom(hits, zf, order=0)[crop]
            full_wet = full_depth > 0.0
            gain = np.where(full_wet, gain, 0.0)
            directionality = np.where(full_wet, directionality, 0.0)

        theta = np.arctan2(v, u)

        return cls(gain=gain, hits=hits, theta=theta,
                   directionality=directionality, omega=float(omega),
                   meta={"n_dirs": int(n_dirs), "rays_per_dir": int(rays_per_dir),
                         "spread_deg": float(spread_deg), "ds": float(ds),
                         "d_ref": d_ref, "break_depth": float(break_depth),
                         "decimate": dec, "solve_dx": dx,
                         "smooth_m": float(smooth_m or 0.0),
                         "mean_visits": visits, "thin_cells": thin,
                         "wet_cells": int(wet.sum()),
                         "wind_sea": bool(wind_sea),
                         "sheltered_cells": int(np.sum(wet & (np.asarray(floor_e)
                                                              > 0.0)))})

    # -- caching -------------------------------------------------------------
    #
    # The solve is minutes on a real export and the field is stationary, so it
    # belongs on disk beside the foam spin-up rather than in front of every
    # frame. What makes that safe is entirely the key.

    @staticmethod
    def cache_key(bathy, cfg, omega: float, **solve_kwargs) -> str:
        """A digest of everything that changes the answer, and nothing else.

        Every solve parameter is read from :meth:`solve`'s own signature with
        defaults applied, rather than listed here. That is deliberate: a list
        would have to be updated in lockstep with the signature, and the failure
        mode when someone forgets is not a crash but a stale field returned with
        complete confidence. Adding a parameter to ``solve`` now changes the key
        automatically, whether or not the caller passes it.

        The bathymetry goes in as raw bytes with its dtype and shape, not as a
        summary statistic -- two different beds can share a mean depth.
        """
        sig = inspect.signature(RayField.solve)
        bound = sig.bind(bathy, cfg, omega, **solve_kwargs)
        bound.apply_defaults()
        params = {k: v for k, v in bound.arguments.items()
                  if k not in ("bathy", "cfg", "cls")}

        h = hashlib.blake2b(digest_size=16)
        h.update(f"pywave.rays v{CACHE_VERSION}".encode())

        depth = np.ascontiguousarray(bathy.depth)
        h.update(f"{depth.dtype}|{depth.shape}".encode())
        h.update(depth.tobytes())
        m = bathy.meta
        h.update(json.dumps([m.dx, list(m.origin), list(m.shape),
                             m.water_level], sort_keys=True).encode())

        w = cfg.wind
        h.update(json.dumps({
            "u10": w.speed, "fetch": w.fetch, "dir": w.direction_rad,
            "gamma": cfg.spectrum.gamma, "spreading": cfg.spectrum.spreading,
            "solve": {k: (float(v) if isinstance(v, (int, float)) and
                          not isinstance(v, bool) else v)
                      for k, v in params.items()},
        }, sort_keys=True, default=str).encode())
        return h.hexdigest()

    def save(self, directory, key: str) -> Path:
        """Write to ``<directory>/rayfield_<key>.npz``.

        Fields are stored as float32. The gain is a ratio near 1 and the angle
        is in radians, so seven significant figures is far past what a Monte
        Carlo solve with a metres-wide smoothing kernel can justify, and the
        difference is 2x on disk.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rayfield_{key}.npz"
        np.savez_compressed(
            path,
            gain=self.gain.astype(np.float32),
            hits=self.hits.astype(np.float32),
            theta=self.theta.astype(np.float32),
            directionality=self.directionality.astype(np.float32),
            omega=np.float64(self.omega),
            meta=np.array(json.dumps(self.meta, default=str)),
        )
        return path

    @classmethod
    def load(cls, directory, key: str) -> "RayField | None":
        """Read a cached field back, or ``None`` if it is not there.

        ``None`` rather than an exception: a cache miss is the normal first run,
        not an error. A *corrupt* file is an error and is left to raise.
        """
        path = Path(directory) / f"rayfield_{key}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as z:
            return cls(gain=z["gain"].astype(np.float64),
                       hits=z["hits"].astype(np.float64),
                       theta=z["theta"].astype(np.float64),
                       directionality=z["directionality"].astype(np.float64),
                       omega=float(z["omega"]),
                       meta=json.loads(str(z["meta"])))

    @classmethod
    def cached(cls, bathy, cfg, omega: float, *, cache_dir=None,
               **solve_kwargs) -> "RayField":
        """:meth:`solve`, reading and writing a cache when given somewhere to.

        With ``cache_dir=None`` this is exactly ``solve`` -- no cache, no file,
        no surprise. Callers that do not care about persistence should not have
        to opt out of it.
        """
        if cache_dir is None:
            return cls.solve(bathy, cfg, omega, **solve_kwargs)
        key = cls.cache_key(bathy, cfg, omega, **solve_kwargs)
        hit = cls.load(cache_dir, key)
        if hit is not None:
            hit.meta = dict(hit.meta, cache="hit", cache_key=key)
            return hit
        rf = cls.solve(bathy, cfg, omega, **solve_kwargs)
        rf.meta = dict(rf.meta, cache="miss", cache_key=key)
        rf.save(cache_dir, key)
        return rf

    def sample(self, bathy, x, y):
        """``(gain, theta)`` at arbitrary scene coordinates.

        The direction is interpolated as a unit vector and recombined with
        ``atan2``, not interpolated as an angle -- halfway between +179 and
        -179 degrees is 180, not 0.
        """
        gx = (np.asarray(x, dtype=np.float64) - bathy.meta.extent[0]) / bathy.meta.dx
        gy = (np.asarray(y, dtype=np.float64) - bathy.meta.extent[2]) / bathy.meta.dx
        u = _bilinear(np.cos(self.theta), gx, gy)
        v = _bilinear(np.sin(self.theta), gx, gy)
        return _bilinear(self.gain, gx, gy), np.arctan2(v, u)


@dataclass
class BandedRayField:
    """One :class:`RayField` per spectral band, which is what production needs.

    ``nearshore.transform`` applies shoaling and refraction **per band** -- that
    is the entire reason the tiles carry disjoint bands rather than merely
    different sizes -- while :meth:`RayField.solve` takes one ``omega``. A
    single-frequency field is enough to gate the solver against Snell and
    against the straits export, and not enough to replace the production path.
    This closes that gap.

    Refraction is dispersive, and the spread is not subtle: at 3 m depth on the
    resized `straits_crop` bands, the long band has shoaled to ``Ks = 0.925``
    while the short band is still at 1.000 and has not felt the bottom at all.

    Each band is cached under its own key, so changing one tile's geometry
    re-solves that band alone.
    """

    bands: tuple[RayField, ...]
    band_edges: tuple[tuple[float, float], ...]
    """``(k_lo, k_hi)`` per band [rad/m], matching ``tiling.band_edges``."""
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.bands)

    def __getitem__(self, i) -> RayField:
        return self.bands[i]

    @classmethod
    def solve(cls, bathy, cfg, tileset, *, cache_dir=None, **solve_kwargs):
        """Solve every band of ``tileset``, caching each independently.

        The representative frequency of each band is
        :func:`nearshore.tile_frequencies` -- its energy centroid -- which is
        exactly what ``transform`` already uses for that band's ``Ks``. Taking
        it from anywhere else would give the two a different idea of what the
        band is.
        """
        from .nearshore import tile_frequencies
        from .tiling import band_edges as _band_edges

        edges = _band_edges(cfg.surface.tiles)
        omegas = tile_frequencies(tileset)
        if len(edges) != len(tileset.tiles):
            raise ValueError(
                f"{len(edges)} band edges for {len(tileset.tiles)} tiles; the "
                f"tile set was built from a different config than this one")

        fields, solved = [], 0
        for om, edge in zip(omegas, edges):
            rf = RayField.cached(bathy, cfg, float(om), cache_dir=cache_dir,
                                 band=(float(edge[0]), float(edge[1])),
                                 **solve_kwargs)
            solved += rf.meta.get("cache") != "hit"
            fields.append(rf)

        return cls(bands=tuple(fields),
                   band_edges=tuple((float(a), float(b)) for a, b in edges),
                   meta={"n_bands": len(fields),
                         "omegas": [float(o) for o in omegas],
                         "periods_s": [float(2 * np.pi / o) for o in omegas],
                         "solved": int(solved),
                         "from_cache": len(fields) - int(solved)})

    def matching(self, tileset) -> "BandedRayField":
        """The subset of these bands that lines up with ``tileset``'s tiles.

        Exists for one caller: :func:`mesh.build_water_mesh` band-limits the
        tiles at the mesh Nyquist before sampling, which can leave fewer tiles
        than the field was solved for. ``band_limited`` truncates from the
        **top** -- it removes high-``k`` content -- so the surviving tiles are a
        prefix and index alignment holds.

        The one inexactness is that the new top band is truncated, so its energy
        centroid is slightly lower than the frequency its ray field was solved
        at. That is immaterial by construction: the content being removed is the
        shortest waves in the set, which do not feel the bottom until the last
        few centimetres of depth and whose gain is therefore ~1 across almost
        the whole domain. Trimming a band whose answer is 1 does not change the
        frequency at which 1 is the answer.

        Done explicitly rather than by letting ``transform`` quietly use the
        first *n* bands, because every other mismatch it could hide -- a field
        solved against a different config, a stale cache -- is a real bug.
        """
        n = len(tileset.tiles)
        if n > len(self.bands):
            raise ValueError(
                f"tile set has {n} tiles but this field has only "
                f"{len(self.bands)} bands; it was solved against a different "
                f"config")
        if n == len(self.bands):
            return self
        return BandedRayField(bands=self.bands[:n], band_edges=self.band_edges[:n],
                              meta=dict(self.meta, truncated_from=len(self.bands)))

    def sample(self, bathy, x, y):
        """``[(gain, theta), ...]``, one pair per band, at scene coordinates."""
        return [rf.sample(bathy, x, y) for rf in self.bands]
