"""PHASE 5b -- Gate 5b: refraction that survives a complex coastline.

The design is in docs/phase5b_refraction.md. This file is written ahead of the
implementation: 5b.6 pins the defect that motivates the whole phase, and the
rest fill in as `pywave/rays.py` grows.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from pywave import nearshore
from pywave.bathymetry import Bathymetry

pytestmark = pytest.mark.gate5b

REPO = Path(__file__).resolve().parent.parent
CROP = REPO / "straits_crop"
has_crop = pytest.mark.skipif(
    not (CROP / "grid_meta.json").exists(),
    reason="needs the straits_crop terrain export; run Houdini to produce one")


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


# ---------------------------------------------------------------------------
# 5b.2 / 5b.3 -- the solver on bathymetry with no closed form anywhere
# ---------------------------------------------------------------------------

# One configuration for every check below. Stated once, deliberately: a gate
# that each test were free to tune its own solve for would measure the tuning
# rather than the solver.
SOLVE = dict(decimate=32, n_dirs=15, rays_per_dir=1500, smooth_m=80.0)

GATE_LAG_M = 4.0
"""Separation the continuity gate is stated at [m].

A "one-cell jump" is only a number once the cell size is fixed. The reference
figures in `docs/phase5b_refraction.md` -- 0.8819 for `snell`, 0.3001 for
`blend` -- were measured at the 4 m posts of the full straits export, so the
comparison is only fair if this is measured over 4 m too. The crop is on 0.25 m
posts, so that is a lag of 16 of them, not a neighbour.
"""


@pytest.fixture(scope="module")
def crop_rays():
    """One ray solve on the 701 m straits crop, shared by the checks below.

    Real bathymetry rather than a synthetic coast, because the whole premise of
    Phase 5b is that synthetic coasts are where the closed form still works. A
    cosine embayment has no medial axis worth the name, no shadowing geometry,
    and no shoreline raggedness -- it would pass every gate here while proving
    nothing about the case that motivated the phase.
    """
    from pywave import load_config, rays

    cfg = load_config(REPO / "configs" / "straits_crop.yaml")
    bathy = Bathymetry.from_export(CROP)
    return bathy, rays.RayField.solve(bathy, cfg, cfg.omega_p, **SOLVE), cfg


def _lag_jumps(gain, wet, lag):
    """`|dgain|` over `lag` posts, counted only where every post between is wet.

    The "both ends wet" rule is not enough and the difference is not subtle. On
    a ragged shoreline two cells 4 m apart are routinely both wet with land in
    between -- that pair is a shoreline, not a discontinuity in the wave field.
    Counting it makes every method look bad, and makes smoothing look actively
    *harmful*, since a wider kernel pulls more shore into each sample. That is
    how the earlier version of this metric announced itself: p99 improved with
    smoothing while p99.9 got worse.
    """
    out = []
    for axis in (0, 1):
        n = wet.shape[axis]
        run = None
        for k in range(lag + 1):
            s = [slice(None)] * 2
            s[axis] = slice(k, n - lag + k)
            run = wet[tuple(s)] if run is None else (run & wet[tuple(s)])
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[axis] = slice(None, n - lag)
        hi[axis] = slice(lag, None)
        out.append(np.abs(gain[tuple(lo)] - gain[tuple(hi)])[run])
    return np.concatenate(out)


@pytest.mark.slow
@has_crop
def test_5b2_gain_is_continuous_on_a_real_coastline(record, crop_rays):
    """Gate 5b.2: p99.9 gain jump over 4 m of open water, against 0.02.

    This is the gate the phase exists for. `snell` scores 0.8819 here and the
    depth-weighted `blend` 0.3001, both because they read `shore_normal`, which
    genuinely flips by up to 180 degrees on the medial axis of a
    distance-derived bed. The rays never read it: celerity is a function of
    depth alone, so the ray paths inherit the bed's smoothness and nothing else.
    """
    bathy, rf, _ = crop_rays
    wet = bathy.depth > 0.0
    lag = int(round(GATE_LAG_M / bathy.meta.dx))
    j = _lag_jumps(rf.gain, wet, lag)
    p999 = float(np.percentile(j, 99.9))

    record("5b", f"p99.9 gain jump over {GATE_LAG_M:g} m of open water",
           p999, tol=0.02, unit="",
           note=f"{j.size / 1e6:.1f}M sampled pairs on the 701 m straits crop; "
                f"p99 {np.percentile(j, 99):.4f}, worst {j.max():.4f}. Against "
                f"0.8819 for `snell` and 0.3001 for `blend` on the full export. "
                f"The deposition kernel is sigma = {SOLVE['smooth_m']:g} m, an "
                f"empirical noise control rather than a derived length -- see "
                f"the approximations note in docs/phase5b_refraction.md.",
           passed=p999 < 0.02)
    assert p999 < 0.02


@pytest.mark.slow
@has_crop
def test_5b2b_deep_water_gain_is_one_without_being_told_to(record, crop_rays):
    """Nothing in the solve forces this, so it checks the launch normalisation.

    The gain is normalised on the median of the deepest 5% of the domain, so
    *that* median is 1 by construction. The median over all wet water is not:
    it comes out at 1 only if the rays are launched with the right spacing and
    the right per-ray power, and a mistake in either shows up here as a bias.
    """
    bathy, rf, _ = crop_rays
    med = float(np.median(rf.gain[bathy.depth > 0.0]))
    record("5b", "median gain over all wet water", med, reference=1.0, tol=0.02,
           unit="", note="Not imposed: the normalisation is taken on the deepest "
                         "5% only. A launch-spacing or per-ray-power error biases "
                         "this away from 1 while leaving the field's shape intact.",
           passed=abs(med - 1.0) < 0.02)
    assert abs(med - 1.0) < 0.02


@pytest.mark.slow
@has_crop
def test_5b3_no_wet_cell_is_annihilated_on_the_crop(record, crop_rays):
    """Gate 5b.3: min gain over wet cells, against 0.05. Scoped to the crop.

    Read the name. This passes on 701 m of coastline and does **not** close
    gate 5b.3, which is stated on the full 7.5 x 8.6 km export and still gives
    0.006 there. Ray theory has no diffraction, so a cell behind a headland
    that no ray reaches gets exactly zero; a 15-direction fan over +/-90 degrees
    softens that without removing it, and the longer the sheltered fetch, the
    deeper the hole. The crop simply has no shadow long enough to reach zero.

    The fix is not more rays. It is a floor from locally generated waves -- a
    sheltered bay is not calm, it has its own short-fetch wind sea -- which is
    the next piece of work and is not built yet.

    What this does pin is that nothing *else* annihilates a cell: not the
    decimation seam, not the deep-water normalisation, not the ray-retirement
    rule.
    """
    bathy, rf, _ = crop_rays
    g = rf.gain[bathy.depth > 0.0]
    lo = float(g.min())
    dark = float((g < 0.05).mean())
    record("5b", "min gain over wet cells (701 m crop)", lo, tol=0.05, unit="",
           note=f"{dark:.3%} of wet cells below 0.05. Scoped to the crop: the "
                f"full straits export gives 0.006 and gate 5b.3 remains open "
                f"there, pending the short-fetch wind-sea floor.",
           passed=lo > 0.05)
    assert lo > 0.05


@pytest.mark.slow
@has_crop
def test_5b8_the_solve_is_reproducible(record, crop_rays):
    """Gate 5b.8: same scene, same answer, bitwise.

    The solve has no RNG in it, which is exactly why this is worth asserting --
    an accumulation order that varied with the deposition buffer's flush
    boundary would break it silently, and floating-point addition is not
    associative.
    """
    from pywave import rays

    bathy, rf, cfg = crop_rays
    again = rays.RayField.solve(bathy, cfg, cfg.omega_p, **SOLVE)
    same = bool(np.array_equal(rf.gain, again.gain))
    record("5b", "ray solve is bitwise reproducible", str(same), reference="True",
           note="No RNG, but the energy accumulator is summed in a buffered "
                "order; floating-point addition is not associative, so a "
                "flush-boundary-dependent order would show up here.",
           passed=same)
    assert same
