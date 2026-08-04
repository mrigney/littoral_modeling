"""PHASE 6 -- per-vertex channel packing.

The section 0.4 data contract: what travels with each mesh vertex from Python
into the renderer.  Keeping these per-vertex rather than global is what makes
the LOD invariant enforceable (cookbook 6.4) and what lets a BSDF vary across a
scene without a texture lookup.

===============  =====  =========================================================
Channel          Type   Meaning
===============  =====  =========================================================
``mss``          f32    Sub-mesh mean square slope **at this vertex's spacing
                        and local depth**.  Becomes the microfacet roughness in
                        Phase 7.  Depth matters only in the surf band, but there
                        it matters a lot -- see :func:`submesh_mss`.
``wdir_x/y``     f32    Local wave direction, unit.  From the *refracted*
                        heading, not the global wind -- nearshore anisotropy has
                        to follow the local waves or glint elongates the wrong
                        way in the surf zone.
``aniso``        f32    Crosswind / upwind slope variance ratio.
``depth``        f32    Local still-water depth [m].
``foam``         f32    Foam coverage fraction, 0-1.
``wetness``      f32    Submergence duty cycle in the swash band, 0-1.  Not in
                        the original contract; carried because it is what the
                        thermal solver needs and it costs four bytes.
===============  =====  =========================================================
"""

from __future__ import annotations

import numpy as np

from .constants import G

__all__ = ["vertex_channels", "submesh_mss", "CHANNEL_UNITS"]

CHANNEL_UNITS = {
    "mss": "-", "wdir_x": "-", "wdir_y": "-", "aniso": "-",
    "depth": "m", "foam": "-", "wetness": "-",
}


def submesh_mss(cfg, dx: float, depth=None):
    """Slope variance below a mesh of spacing ``dx`` -- the BSDF's share.

    ``mss_above(pi / dx)``, which with the resolved part sums to the total no
    matter what ``dx`` is.  That identity is the LOD invariant.

    Returns a scalar when ``depth`` is ``None``, otherwise an array matching it.

    Why depth cannot be ignored here, although it nearly can
    -------------------------------------------------------
    The tempting argument is that sub-mesh waves are short by construction --
    at a 0.125 m mesh the cut is ``k = 25 rad/m``, wavelengths under 25 cm --
    so they are deep-water waves everywhere and their slope variance is a scene
    constant.  That is true over almost the whole domain and false exactly where
    it matters.  Measured on the shipped lake at ``dx = 0.125 m``:

    ========  =======  ==========================
    depth     ``kd``   sub-mesh mss vs deep water
    ========  =======  ==========================
    3 cm      0.75     **+44%**
    5 cm      1.26     +11%
    10 cm     2.51     +0.6%
    20 cm +   >5       +0.0%
    ========  =======  ==========================

    Converged past ``kd ~ 2.5``, and badly wrong below it -- and below it is the
    surf and swash band, which is the part anyone looks at.  A coarser mesh is
    worse: at ``dx = 0.25 m`` the 3 cm error is +142%.

    So the depth is used.  ``mss_above`` is a radial quadrature and far too slow
    to call per vertex, so it is evaluated on a log-spaced depth lookup and
    interpolated, which is exact to the width of a bin and costs a few
    milliseconds.  Depths past convergence take the deep-water value directly
    rather than the top bin, so the common case carries no interpolation error
    at all.

    Note the contrast with the *resolved* variance, which shoals strongly. Both
    halves of the LOD split move with depth; they just move differently.
    """
    from .moments import mss_above

    k_cut = np.pi / dx
    deep = float(mss_above(k_cut, cfg.wind.speed, cfg.wind.fetch,
                           gamma=cfg.spectrum.gamma))
    if depth is None:
        return deep

    d = np.asarray(depth, dtype=np.float64)
    # tanh(kd) = 1 to within 1e-8 by kd = 10, so nothing below the cut
    # wavenumber feels the bottom past this depth.
    d_conv = 10.0 / k_cut
    d_lo = 0.02 * d_conv

    grid = np.geomspace(d_lo, d_conv, 48)
    table = np.array([mss_above(k_cut, cfg.wind.speed, cfg.wind.fetch,
                                depth=float(dd), gamma=cfg.spectrum.gamma)
                      for dd in grid])

    # Clamp below d_lo rather than extrapolate: the integral diverges as d -> 0,
    # and any water that shallow has already broken, so the mesh there is
    # carrying swash rather than a wave field.
    out = np.interp(np.clip(d, d_lo, d_conv), grid, table)
    return np.where(d >= d_conv, deep, out)


def _local_wave_direction(cfg, bathy, x, y, depth, refraction: str):
    """Unit vector along the local wave heading, refracted where it matters."""
    from . import nearshore

    theta_deep = cfg.wind.direction_rad
    if refraction == "none":
        theta = np.full(np.shape(depth), theta_deep, dtype=np.float64)
    else:
        _, _, normal = bathy.sample(x, y)
        omega = 2.0 * np.pi * cfg.f_p
        if refraction == "blend":
            theta, _ = nearshore.refraction_angle_blend(
                theta_deep, normal, depth, 3.0 * cfg.lambda_p)
        else:
            theta, _ = nearshore.refraction_angle(theta_deep, normal, depth, omega)
    return np.cos(theta), np.sin(theta)


def vertex_channels(tileset, bathy, cfg, x, y, nf, dx: float, *,
                    foam=None, foam_bathy=None, refraction: str = "snell") -> dict:
    """Build the per-vertex channel set for a mesh at spacing ``dx``.

    ``nf`` is the :class:`~pywave.nearshore.NearshoreField` already evaluated at
    ``(x, y)``; passing it in avoids paying for the transform twice.
    """
    from . import nearshore
    from .moments import mss_anisotropic

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size

    k_cut = np.pi / dx
    mss = submesh_mss(cfg, dx, nf.depth)

    up, cross = mss_anisotropic(cfg.wind.speed, cfg.wind.fetch,
                                cfg.wind.direction_rad, k_cut=k_cut,
                                gamma=cfg.spectrum.gamma)
    aniso = float(cross / up) if up > 0 else 1.0

    wdir_x, wdir_y = _local_wave_direction(cfg, bathy, x, y, nf.depth, refraction)

    channels = {
        "mss": np.asarray(mss, dtype=np.float32),
        "wdir_x": np.asarray(wdir_x, dtype=np.float32),
        "wdir_y": np.asarray(wdir_y, dtype=np.float32),
        "aniso": np.full(n, aniso, dtype=np.float32),
        "depth": np.asarray(nf.depth, dtype=np.float32),
        "wetness": np.asarray(nf.wetness, dtype=np.float32),
    }

    if foam is not None:
        grid = foam_bathy if foam_bathy is not None else bathy
        channels["foam"] = _sample_grid(foam, grid, x, y).astype(np.float32)
    else:
        # Zero rather than absent: a renderer that expects the channel should
        # get a valid one, and "no foam anywhere" is the honest reading of "no
        # foam field was supplied".
        channels["foam"] = np.zeros(n, dtype=np.float32)

    return channels


def _sample_grid(field: np.ndarray, bathy, x, y) -> np.ndarray:
    """Bilinear sample of a bathymetry-grid field at world coordinates."""
    from scipy.ndimage import map_coordinates

    m = bathy.meta
    x0, y0 = m.origin
    coords = np.stack([(np.asarray(y, dtype=np.float64) - y0) / m.dx - 0.5,
                       (np.asarray(x, dtype=np.float64) - x0) / m.dx - 0.5])
    return map_coordinates(field, coords, order=1, mode="nearest")
