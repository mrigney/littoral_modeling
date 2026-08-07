"""PHASE 5b -- Gate 5b: refraction that survives a complex coastline.

The design is in docs/phase5b_refraction.md. This file is written ahead of the
implementation: 5b.6 pins the defect that motivates the whole phase, and the
rest fill in as `pywave/rays.py` grows.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pywave import nearshore
from pywave.bathymetry import Bathymetry

pytestmark = pytest.mark.gate5b


def _rotate(normal: np.ndarray, degrees: float) -> np.ndarray:
    """Turn every shore normal by a fixed angle, keeping it a unit vector."""
    a = np.radians(degrees)
    ca, sa = np.cos(a), np.sin(a)
    return np.stack([ca * normal[0] - sa * normal[1],
                     sa * normal[0] + ca * normal[1]])


@pytest.mark.xfail(
    strict=True,
    reason="Gate 5b.6: the defect Phase 5b exists to fix. Snell's Kr is computed "
           "from shore_normal -- the direction to the nearest shore -- which is "
           "discontinuous on the medial axis of any distance-derived bed. Until "
           "rays.py lands, perturbing that direction moves the answer. Strict, so "
           "this fails loudly the moment it starts passing.")
def test_5b6_wave_field_is_independent_of_shore_normal(record, cfg):
    """The answer must not depend on which shore happens to be nearest.

    `shore_normal` is a property of the *shoreline geometry*, not of the wave
    field. Waves refract in response to the depth they are travelling over. A
    solver that is doing refraction properly cannot notice if that vector is
    rotated, because it should never have read it.

    Rotating by 30 degrees is a fair test: it is far larger than any numerical
    tolerance and far smaller than the 180 degrees the field genuinely jumps by
    on the medial axis.
    """
    from pywave import tiling

    ts = tiling.TileSet.build(cfg)
    bathy = Bathymetry.from_config(cfg, fine=True)
    ax, ay = bathy.meta.axes()
    X, Y = np.meshgrid(ax, ay, indexing="xy")
    x, y = X.ravel(), Y.ravel()

    turned = dataclasses.replace(
        bathy, shore_normal=_rotate(bathy.shore_normal, 30.0))

    base = nearshore.transform(ts, bathy, cfg, x, y, 0.0, refraction="snell")
    other = nearshore.transform(ts, turned, cfg, x, y, 0.0, refraction="snell")

    wet = np.asarray(base.depth) > 0.0
    d = float(np.abs(np.asarray(base.hs_local)[wet]
                     - np.asarray(other.hs_local)[wet]).max())
    record("5b", "max |dHs| under a 30 deg shore_normal rotation", d, unit="m",
           tol=1e-12,
           note="Refraction must respond to depth, not to the direction of the "
                "nearest shore. Non-zero here is the medial-axis seam.",
           passed=d < 1e-12)
    assert d < 1e-12


# ---------------------------------------------------------------------------
# 5b.1 -- the integrator against the one geometry where the closed form is right
# ---------------------------------------------------------------------------


def _planar_beach(nx=240, ny=240, dx=1.0, slope=0.02):
    """Depth falling linearly to zero, uniform alongshore."""
    y = np.arange(ny) * dx
    return np.maximum(y * slope, 0.0)[:, None] * np.ones((1, nx)), dx


def _analytic_gain(depth_profile, omega, d_deep, alpha0):
    """`Ks * Kr` from the closed form, normalised to 1 in deep water."""
    from pywave.spectrum import dispersion_k

    d = np.maximum(depth_profile, 1e-6)
    ks = (nearshore.shoaling_coefficient(omega, d)
          / nearshore.shoaling_coefficient(omega, d_deep))
    k = dispersion_k(omega, d)
    k_deep = dispersion_k(omega, d_deep)
    sin_a = np.sin(alpha0) * (k_deep / k)
    alpha = np.arcsin(np.clip(sin_a, -1.0, 1.0))
    kr = np.sqrt(np.cos(alpha0) / np.maximum(np.cos(alpha), 1e-9))
    return ks * kr


@pytest.mark.parametrize("alpha_deg", [0.0, 20.0, 40.0])
def test_5b1_straight_beach_reproduces_snell(record, alpha_deg):
    """On straight parallel contours the rays must agree with the closed form.

    A planar beach is the one geometry where `Kr = sqrt(cos a0 / cos a)` is
    exactly right, so it is the only honest place to check the integrator.
    Agreement here plus disagreement on a real coast is the whole thesis of
    Phase 5b: it localises the error in the formula's premise rather than in
    either implementation.
    """
    from pywave import rays
    from pywave.spectrum import dispersion_k

    depth, dx = _planar_beach()
    ny, nx = depth.shape
    d_deep = depth.max()
    omega = 2 * np.pi / 6.0
    alpha0 = np.radians(alpha_deg)

    x0, y0, th0, e_ref = rays.launch_line(
        0.0, nx * dx, (ny - 1.5) * dx, -np.pi / 2 + alpha0, 6000, omega, d_deep)
    energy, _ = rays.trace_rays(depth, omega, dx, x0, y0, th0,
                                ds=0.25 * dx, break_depth=0.05, wrap_x=True)
    gain = np.sqrt(np.maximum(energy, 0.0) / e_ref)[:, nx // 2]

    prof = depth[:, nx // 2]
    want = _analytic_gain(prof, omega, d_deep, alpha0)
    band = (prof > 0.3) & (prof < 0.9 * d_deep)
    err = np.abs(gain[band] - want[band]) / want[band]
    worst = float(err.max())

    record("5b", f"ray vs analytic Ks*Kr, straight beach, {alpha_deg:g} deg",
           worst, tol=0.02, unit="",
           note=f"{int(band.sum())} cells over depths 0.3-{0.9 * d_deep:.1f} m; "
                f"median {100 * np.median(err):.2f}%. The energy-accumulation "
                f"solver is not given the closed form anywhere -- it integrates "
                f"rays through the celerity field and counts what arrives.",
           passed=worst < 0.02)
    assert worst < 0.02, f"worst relative error {100 * worst:.2f}%"


def test_5b1b_launch_normalisation_carries_the_obliquity_cosine(record):
    """Rays are spaced along a line; energy density needs their *normal* spacing.

    Omitting `cos(alpha)` inflates the gain by `1/sqrt(cos alpha)`: 3.2% at 20
    degrees, 14.3% at 40. Both are small enough to read as a plausible physical
    result rather than a bug, which is exactly why this is pinned.
    """
    from pywave import rays

    omega, d = 2 * np.pi / 6.0, 8.0
    ref0 = rays.launch_line(0.0, 100.0, 0.0, -np.pi / 2, 101, omega, d)[3]
    worst = 0.0
    for deg in (20.0, 40.0):
        ref = rays.launch_line(0.0, 100.0, 0.0, -np.pi / 2 + np.radians(deg),
                               101, omega, d)[3]
        got = ref / ref0
        want = 1.0 / np.cos(np.radians(deg))
        worst = max(worst, abs(got - want) / want)
    record("5b", "launch reference energy carries cos(alpha)", worst, tol=1e-12,
           note="e_ref must scale as 1/cos(alpha) with obliquity.",
           passed=worst < 1e-12)
    assert worst < 1e-12
