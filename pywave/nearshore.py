"""PHASE 5 -- nearshore transformation: shoaling, refraction, breaking, wetness.

Scope, decided by the numbers rather than by ambition (cookbook section 5.1).
For the test lake::

    Hs = 0.086 m,  Tp = 1.05 s,  lambda_p = 1.7 m,  max depth 5 m

The deep-water cutoff is ``lambda/2 = 0.85 m``, so **the lake is deep water
everywhere except the last few metres**.  Breaking happens in ~11 cm of water;
on this beach the surf zone is about 1.5 m wide and the swash excursion 0.4 m.
At 1 m GSD that entire zone is sub-pixel.

Hence the split this module implements:

* **Shoaling and refraction are geometry.**  They act over tens of metres, they
  are resolved at sensor scale, and they change what the surface looks like far
  outside the surf zone.  Modelled as fields applied to the wave surface.
* **Breaking and swash are channels.**  Sub-pixel.  Modelled as per-cell
  fractional coverage, not as animated geometry.  Building displaced swash would
  be weeks of work invisible at this resolution.

What is an approximation, stated plainly
----------------------------------------
The FFT surface is translation-invariant; refraction is not.  A rigorous
treatment needs a mild-slope or Boussinesq solver, which is a much larger
project and unnecessary at 8 cm wave heights.  So refraction here is a per-cell
post-process: it rotates the local wave *direction* -- the displacement and
slope vectors, hence the surface normal, hence everything the BSDF sees -- but
it does not re-solve the wave field, so crest *positions* are unchanged.
Shoaling is applied as a per-band amplitude scale, which is exact for the
amplitude and ignores the accompanying wavelength shortening.

Both are documented as approximations in the validation report rather than
buried.  What is *not* approximated is the coefficient physics itself:
``shoaling_coefficient`` and ``refraction_angle`` solve the full dispersion
relation and Snell's law, and the tests check them against Green's law and
against closed-form Snell at every cell.

References
----------
Dean, R.G. & Dalrymple, R.A. (1991).  *Water Wave Mechanics for Engineers and
    Scientists.*  World Scientific.
Hunt, I.A. (1959).  "Design of seawalls and breakwaters."  J. Waterways and
    Harbors Division, ASCE 85, 123-152.
Battjes, J.A. (1974).  "Surf similarity."  Proc. 14th Coastal Engineering Conf.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bathymetry import Bathymetry
from .constants import G
from .spectrum import dispersion_k, dispersion_omega, group_velocity, shoaling_n
from .surface import SurfaceField

__all__ = [
    "shoaling_coefficient",
    "refraction_angle",
    "refraction_coefficient",
    "refraction_angle_blend",
    "deep_water_wavelength",
    "iribarren_number",
    "breaker_type",
    "breaker_depth",
    "breaking_mask",
    "hunt_runup",
    "swash_width",
    "wetness",
    "wetness_fraction",
    "tile_frequencies",
    "NearshoreField",
    "transform",
]


# ---------------------------------------------------------------------------
# Shoaling
# ---------------------------------------------------------------------------


def shoaling_coefficient(omega, depth, g: float = G) -> np.ndarray:
    """``Ks = sqrt(Cg_deep / Cg_local)`` -- the amplitude gain from shoaling.

    Energy flux ``E * Cg`` is conserved as a wave train moves into shallow
    water, and ``E ~ H^2``, so ``H / H_deep = sqrt(Cg_deep / Cg)``.  The local
    ``Cg`` comes from solving ``omega^2 = g k tanh(kd)`` at the local depth.

    ``omega`` is the invariant here: frequency does not change as a wave shoals,
    wavenumber does.  Passing a wavenumber to this function instead would be
    wrong in a way that is hard to see, so the signature takes ``omega`` only.

    Notes
    -----
    ``Ks`` is not monotonic.  It dips to a minimum of about 0.913 near
    ``kd = 1.2`` before rising, because ``n = Cg/c`` grows faster than ``c``
    falls at first.  That dip is real and is asserted in the tests -- an
    implementation that clamps ``Ks >= 1`` has thrown away physics.

    In the shallow limit it recovers Green's law, ``Ks ~ d^(-1/4)``.
    """
    omega = np.asarray(omega, dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)

    # Guard the waterline: Ks diverges as d -> 0. Callers cap it via the
    # depth-limited breaking criterion; here we just avoid the singularity.
    d_safe = np.maximum(d, 1e-6)

    cg_deep = 0.5 * g / omega
    k = dispersion_k(omega, d_safe, g)
    cg = shoaling_n(k, d_safe) * omega / k

    ks = np.sqrt(cg_deep / cg)
    return np.where(d > 0.0, ks, 1.0)


def deep_water_wavelength(period, g: float = G) -> np.ndarray:
    """``L0 = g T^2 / (2 pi)`` [m]."""
    t = np.asarray(period, dtype=np.float64)
    return g * t**2 / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# Refraction
# ---------------------------------------------------------------------------


def _wrap(angle):
    """Wrap to ``[-pi, pi]``."""
    return np.mod(np.asarray(angle, dtype=np.float64) + np.pi, 2.0 * np.pi) - np.pi


def _incidence(theta_deep, shore_normal):
    """Angle of the wave direction from the inland shore normal, in ``[-pi, pi]``."""
    theta_n = np.arctan2(shore_normal[1], shore_normal[0])
    return _wrap(np.asarray(theta_deep, dtype=np.float64) - theta_n), theta_n


def refraction_angle(theta_deep, shore_normal, depth, omega, g: float = G):
    """Refracted wave direction from Snell's law.  Returns ``(theta, alpha)``.

    ``sin(alpha) / c = const`` along a ray, where ``alpha`` is measured from the
    shore normal.  With ``c_deep = g / omega`` and ``c = omega / k``::

        sin(alpha_local) = sin(alpha_deep) * c_local / c_deep

    Exact for straight parallel depth contours, which is what
    :meth:`~pywave.bathymetry.Bathymetry.dean_beach` provides -- so the tests
    have a closed-form answer to check against.  On a curved shoreline it is
    applied locally, using the local shore normal, which is the standard
    ray-theory approximation.

    Parameters
    ----------
    theta_deep : deep-water wave direction, CCW from +X, direction travelling
        **toward** [rad].
    shore_normal : ``(2, ...)`` unit vectors pointing **inland**.
    depth, omega : local still-water depth [m] and angular frequency [rad/s].

    Returns
    -------
    (theta, alpha)
        ``theta`` -- refracted direction in world frame [rad].
        ``alpha`` -- refracted angle from the shore normal [rad], signed.

    Waves already travelling offshore (``|alpha_deep| > pi/2``) are returned
    unchanged: Snell's law describes a ray crossing contours toward shore, and
    silently "refracting" an offshore-bound wave would invent energy transport
    in the wrong direction.
    """
    alpha_deep, theta_n = _incidence(theta_deep, shore_normal)
    d = np.asarray(depth, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    c_deep = g / omega
    k = dispersion_k(omega, np.maximum(d, 1e-6), g)
    c_local = omega / k

    sin_alpha = np.clip(np.sin(alpha_deep) * c_local / c_deep, -1.0, 1.0)
    alpha = np.arcsin(sin_alpha)

    shoreward = (np.cos(alpha_deep) > 0.0) & (d > 0.0)
    alpha = np.where(shoreward, alpha, alpha_deep)
    return _wrap(theta_n + alpha), alpha


def refraction_coefficient(alpha_deep, alpha_local) -> np.ndarray:
    """``Kr = sqrt(|cos(alpha_deep)| / |cos(alpha_local)|)`` -- ray convergence.

    Between two adjacent rays the spacing scales as ``cos(alpha)``, so as a wave
    turns toward the normal the rays *spread* along the shore and the height
    drops.  ``Kr <= 1`` for straight parallel contours, always: oblique waves
    deliver their energy over a longer stretch of shoreline than normal-incident
    ones.

    Total local height is ``H = H_deep * Ks * Kr``.

    Waves travelling offshore
    -------------------------
    Absolute values, and the reason is not cosmetic.  A wave heading *away* from
    the shore is returned unrefracted by :func:`refraction_angle`
    (``alpha_local == alpha_deep``), and the ray-spacing ratio is then exactly
    1 -- no convergence, no divergence.  Taking ``max(cos(alpha_deep), 0)``
    instead sends ``Kr`` to **zero** there, which does not attenuate the wave so
    much as delete it.

    That is invisible on an open coast, where the wind blows onshore everywhere
    and the case never arises.  On a closed basin it removes the sea from the
    entire downwind half of the shoreline: measured on a real lake export,
    47% of wet cells had exactly zero wave height, silently, because every
    quantity downstream of ``Kr`` is a product and zero propagates quietly.
    """
    ca = np.abs(np.cos(np.asarray(alpha_deep, dtype=np.float64)))
    cl = np.abs(np.cos(np.asarray(alpha_local, dtype=np.float64)))
    # Grazing incidence sends the ratio to infinity. It cannot arise for a
    # shoreward wave -- refraction turns toward the normal, so |cos| only grows
    # -- and for anything else the honest answer is "no refraction applied".
    safe = cl > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        kr = np.sqrt(ca / np.where(safe, cl, 1.0))
    return np.where(safe, kr, 1.0)


def refraction_angle_blend(theta_deep, shore_normal, depth, d_ref: float):
    """The cookbook's smoothstep blend toward the shore normal (section 5.3).

    ``w = clip(1 - d/d_ref, 0, 1)``, then interpolate the wave direction toward
    the shore normal by ``w`` along the shorter arc.

    Kept alongside :func:`refraction_angle` because the cookbook prescribes it
    and it is cheaper, but it is **not** the production path: it has no
    frequency dependence, so every spectral band would turn at the same rate,
    and it reaches full shore-normal alignment at ``d = 0`` regardless of the
    incident angle.  Snell does neither.  The tests measure the disagreement.
    """
    alpha_deep, theta_n = _incidence(theta_deep, shore_normal)
    d = np.asarray(depth, dtype=np.float64)
    w = np.clip(1.0 - d / d_ref, 0.0, 1.0)
    return _wrap(theta_n + (1.0 - w) * alpha_deep), (1.0 - w) * alpha_deep


# ---------------------------------------------------------------------------
# Breaking
# ---------------------------------------------------------------------------


def iribarren_number(slope: float, hs, l0) -> np.ndarray:
    """Surf similarity parameter ``xi = tan(beta) / sqrt(Hs / L0)``.

    Sets the breaker type and, through Hunt's formula, the runup.
    """
    hs = np.asarray(hs, dtype=np.float64)
    l0 = np.asarray(l0, dtype=np.float64)
    return slope / np.sqrt(np.maximum(hs, 1e-12) / l0)


def breaker_type(xi) -> str | np.ndarray:
    """Classify from the Iribarren number (Battjes 1974).

    ``xi < 0.5`` spilling, ``0.5 <= xi < 3.3`` plunging, ``xi >= 3.3`` surging.
    """
    xi = np.asarray(xi, dtype=np.float64)
    out = np.where(xi < 0.5, "spilling", np.where(xi < 3.3, "plunging", "surging"))
    return str(out) if out.ndim == 0 else out


def breaker_depth(h_local, gamma_b: float = 0.78) -> np.ndarray:
    """Depth at which a wave of height ``h_local`` breaks: ``d = H / gamma_b``."""
    return np.asarray(h_local, dtype=np.float64) / gamma_b


def breaking_mask(h_local, depth, gamma_b: float = 0.78) -> np.ndarray:
    """``True`` where the wave height exceeds the depth-limited maximum."""
    h = np.asarray(h_local, dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)
    return (d > 0.0) & (h > gamma_b * d)


# ---------------------------------------------------------------------------
# Swash and wetness
# ---------------------------------------------------------------------------


def hunt_runup(xi, hs) -> np.ndarray:
    """Vertical runup ``R = xi * Hs`` [m] (Hunt 1959)."""
    return np.asarray(xi, dtype=np.float64) * np.asarray(hs, dtype=np.float64)


def swash_width(runup, slope: float) -> np.ndarray:
    """Horizontal swash excursion ``R / tan(beta)`` [m]."""
    return np.asarray(runup, dtype=np.float64) / slope


def _smoothstep(t):
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def wetness(sdf, width, t: float = 0.0, period: float | None = None) -> np.ndarray:
    """Instantaneous wet fraction, phase-modulated at the wave period.

    1 in the water, falling smoothly to 0 at the landward limit of the swash.
    With ``period`` given, the waterline oscillates over the swash band at that
    period, which is what makes a rendered shoreline breathe.

    ``sdf`` is distance inland (positive on land), so the swash occupies
    ``0 <= s <= width``.
    """
    s = np.asarray(sdf, dtype=np.float64)
    w = np.asarray(width, dtype=np.float64)
    if period is not None and period > 0:
        edge = w * (0.5 + 0.5 * np.sin(2.0 * np.pi * t / period))
    else:
        edge = w
    edge = np.maximum(edge, 1e-9)
    return _smoothstep(1.0 - s / edge)


def wetness_fraction(sdf, width) -> np.ndarray:
    """Fraction of the wave period a point spends submerged -- the thermal channel.

    With the swash edge at ``e(t) = W (1 + sin(2 pi t / T)) / 2``, a point at
    distance ``s`` inland is wet whenever ``e(t) > s``.  That holds for a
    fraction of the period::

        f(s) = 1/2 - arcsin(2 s / W - 1) / pi
             = (1/pi) * arccos(2 s / W - 1)        0 <= s <= W

    Note this is the duty cycle of a **hard** waterline crossing, which is not
    the same as the time average of :func:`wetness` -- that function returns a
    smoothstepped ramp at each instant, deliberately, because a hard edge
    aliases badly when rendered.  The two agree in shape but the smoothstepped
    average is softer at both ends.  Use this one for anything thermal and that
    one for anything visual.

    Why this matters more than the instantaneous field: wet sand has much higher
    thermal inertia than dry, so the damp band reads as a cold line in daytime
    LWIR.  That line is one of the most diagnostic features in a littoral
    thermal image, and it is set by the *duty cycle* of wetting, not by where
    the water happens to be in any one frame.  It is also what Hotts needs as a
    boundary condition -- a per-frame binary mask would be the wrong input.
    """
    s = np.asarray(sdf, dtype=np.float64)
    w = np.maximum(np.asarray(width, dtype=np.float64), 1e-9)
    u = np.clip(2.0 * s / w - 1.0, -1.0, 1.0)
    return np.where(s <= 0.0, 1.0, np.where(s >= w, 0.0, np.arccos(u) / np.pi))


# ---------------------------------------------------------------------------
# Applying it all to a tile set
# ---------------------------------------------------------------------------


def tile_frequencies(tileset, g: float = G) -> np.ndarray:
    """Energy-weighted representative angular frequency of each tile [rad/s].

    Shoaling is frequency-dependent, so it must be applied per spectral band
    rather than as one scalar (cookbook section 5.2) -- which is a concrete
    reason the tiles carry *disjoint* bands rather than merely different sizes.
    Each tile is reduced to the one frequency that carries its energy centroid::

        k_rep = sum(k S) / sum(S),     omega_rep = sqrt(g k_rep)

    The deep-water dispersion relation is used because ``omega`` is the quantity
    conserved during shoaling, and these tiles were built in deep water.
    """
    out = []
    for tile in tileset.tiles:
        total = float(np.sum(tile.s_k))
        if total <= 0.0:
            out.append(0.0)
            continue
        k_rep = float(np.sum(tile.k * tile.s_k) / total)
        out.append(float(dispersion_omega(np.array(k_rep), None, g)))
    return np.array(out)


@dataclass(frozen=True)
class NearshoreField:
    """A transformed surface plus the per-cell channels that go with it."""

    surface: SurfaceField
    """Wave surface after shoaling, refraction and depth limiting."""
    depth: np.ndarray
    """Still-water depth at each sample [m]."""
    sdf: np.ndarray
    """Signed distance to the waterline [m]; positive inland."""
    hs_local: np.ndarray
    """Local significant wave height after transformation [m]."""
    shoaling: np.ndarray
    """Energy-weighted effective amplitude gain ``Ks * Kr`` [-]."""
    breaking: np.ndarray
    """``True`` where the unlimited height exceeds ``gamma_b * depth``."""
    wetness: np.ndarray
    """Time-averaged submergence fraction in the swash band [-]."""
    limiter: np.ndarray
    """Factor applied to enforce the depth-limited height [-]; 1 outside the surf."""

    def surf_zone_width(self, along_axis_coords) -> float:
        """Width of the breaking band along a monotone transect [m]."""
        coords = np.asarray(along_axis_coords, dtype=np.float64)
        if not np.any(self.breaking):
            return 0.0
        inside = coords[self.breaking]
        return float(inside.max() - inside.min())


def transform(
    tileset,
    bathy: Bathymetry,
    cfg,
    x,
    y,
    t: float,
    *,
    fields=None,
    order: int = 3,
    refraction: str | None = None,
    depth_limit: bool = True,
) -> NearshoreField:
    """Apply the nearshore transformation to a composite surface.

    Order of operations, which is not arbitrary:

    1. Sample the bathymetry at the requested world points.
    2. Per tile, solve dispersion at the local depth for that tile's
       representative frequency, giving ``Ks``; refract to get ``alpha`` and
       hence ``Kr``.
    3. Form the *unlimited* local ``Hs`` from the per-band variances, and mark
       breaking where it exceeds ``gamma_b * depth``.
    4. Derive one depth-limiting factor from that, applied to every band, so the
       surf zone saturates instead of letting ``Ks`` diverge at the waterline.
    5. Sample the tiles with the resulting per-band weights, rotating the vector
       quantities by the local refraction angle.

    Step 4 has to come before step 5 and has to be shared across bands: applying
    it per band independently would change the spectral shape inside the surf
    zone, which is not what depth limiting does.

    Parameters
    ----------
    refraction : ``"snell"`` (default), ``"blend"`` (the cookbook's
        depth-weighted slerp) or ``"none"``.
    depth_limit : cap the local height at ``gamma_b * depth``. Turning this off
        is only useful for testing the raw shoaling gain, which diverges at the
        waterline.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # None means "whatever the scene asked for". Hard-coding "snell" here is how
    # nearshore.refraction in the config came to be ignored by every mesh.
    if refraction is None:
        refraction = getattr(cfg.nearshore, "refraction", "snell")

    depth, sdf, normal = bathy.sample(x, y)
    gamma_b = cfg.nearshore.breaker_index
    theta_deep = cfg.wind.direction_rad

    omegas = tile_frequencies(tileset)
    d_ref = 3.0 * cfg.lambda_p

    weights, rotations = [], []
    variance = np.zeros_like(depth)

    for tile, omega in zip(tileset.tiles, omegas):
        if omega <= 0.0:
            weights.append(np.zeros_like(depth))
            rotations.append(np.zeros_like(depth))
            continue

        ks = shoaling_coefficient(omega, depth)

        if refraction == "snell":
            theta, alpha = refraction_angle(theta_deep, normal, depth, omega)
            alpha_deep, _ = _incidence(theta_deep, normal)
            kr = refraction_coefficient(alpha_deep, alpha)
        elif refraction == "blend":
            theta, _ = refraction_angle_blend(theta_deep, normal, depth, d_ref)
            kr = np.ones_like(depth)
        elif refraction == "none":
            theta = np.full_like(depth, theta_deep)
            kr = np.ones_like(depth)
        else:
            raise ValueError(f"unknown refraction mode {refraction!r}; "
                             f"use 'snell', 'blend' or 'none'")

        gain = np.where(depth > 0.0, ks * kr, 0.0)
        weights.append(gain)
        rotations.append(_wrap(theta - theta_deep))
        variance = variance + gain**2 * tile.m0()

    hs_unlimited = 4.0 * np.sqrt(np.maximum(variance, 0.0))
    breaking = breaking_mask(hs_unlimited, depth, gamma_b)

    if depth_limit:
        with np.errstate(divide="ignore", invalid="ignore"):
            limiter = np.where(
                hs_unlimited > 0.0,
                np.minimum(1.0, gamma_b * np.maximum(depth, 0.0) / np.maximum(hs_unlimited, 1e-12)),
                1.0,
            )
    else:
        limiter = np.ones_like(depth)

    hs_local = hs_unlimited * limiter
    hs_deep = 4.0 * np.sqrt(tileset.m0())

    from .tiling import composite_surface

    field = composite_surface(
        tileset.tiles, x, y, t, fields=fields, order=order,
        weights=[w * limiter for w in weights], rotate=rotations,
    )

    # Swash band, from the deep-water sea state and the foreshore slope.
    slope = bathy.beach_slope()
    l0 = deep_water_wavelength(1.0 / cfg.f_p)
    xi = float(iribarren_number(slope, hs_deep, l0))
    band = float(swash_width(hunt_runup(xi, hs_deep), slope))

    return NearshoreField(
        surface=field,
        depth=depth,
        sdf=sdf,
        hs_local=hs_local,
        shoaling=np.where(depth > 0.0, hs_local / max(hs_deep, 1e-12), 0.0),
        breaking=breaking,
        wetness=wetness_fraction(sdf, band),
        limiter=limiter,
    )
