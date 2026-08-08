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
    acc = rays.trace_rays(depth, omega, dx, x0, y0, th0,
                          ds=0.25 * dx, break_depth=0.05, wrap_x=True)
    gain = np.sqrt(np.maximum(acc.energy, 0.0) / e_ref)[:, nx // 2]

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


@pytest.mark.parametrize("alpha_deg", [20.0, 40.0])
def test_5b1c_ray_direction_reproduces_snells_angle(record, alpha_deg):
    """Gain is only half of refraction; the other half is which way it points.

    A wave can turn without changing height, and on a mild slope at normal
    incidence it mostly does -- so a solver validated on `Ks*Kr` alone could be
    turning the waves any way at all and still pass 5b.1. This is the same
    planar beach, checking the direction the energy actually travels against
    Snell's law solved on the full dispersion relation.

    `nearshore.transform` rotates each tile by this angle, so an error here is
    a surface whose waves run across the beach instead of into it.
    """
    from pywave import rays
    from pywave.spectrum import dispersion_k

    depth, dx = _planar_beach()
    ny, nx = depth.shape
    d_deep = depth.max()
    omega = 2 * np.pi / 6.0
    alpha0 = np.radians(alpha_deg)

    x0, y0, th0, _ = rays.launch_line(
        0.0, nx * dx, (ny - 1.5) * dx, -np.pi / 2 + alpha0, 6000, omega, d_deep)
    acc = rays.trace_rays(depth, omega, dx, x0, y0, th0,
                          ds=0.25 * dx, break_depth=0.05, wrap_x=True)

    prof = depth[:, nx // 2]
    c = omega / dispersion_k(omega, np.maximum(prof, 1e-6))
    c_deep = omega / dispersion_k(omega, d_deep)
    want = -np.pi / 2 + np.arcsin(np.clip(np.sin(alpha0) * c / c_deep, -1.0, 1.0))
    got = acc.theta[:, nx // 2]

    band = (prof > 0.3) & (prof < 0.9 * d_deep)
    err = np.degrees(np.abs(np.angle(np.exp(1j * (got[band] - want[band])))))
    worst = float(err.max())
    turn = np.degrees(want[band].max() - want[band].min())
    record("5b", f"ray direction vs Snell, straight beach, {alpha_deg:g} deg",
           worst, tol=0.5, unit="deg",
           note=f"{int(band.sum())} cells; median {np.median(err):.3f} deg. The "
                f"rays turn {turn:.1f} deg across this band, so the tolerance is "
                f"{100 * 0.5 / max(turn, 1e-9):.1f}% of the effect being "
                f"measured, not of the angle itself.",
           passed=worst < 0.5)
    assert worst < 0.5, f"worst direction error {worst:.3f} deg"


def test_mean_direction_survives_the_branch_cut(record):
    """Rays at +179 and -179 degrees travel nearly the same way.

    Averaging their angles gives 0 -- exactly backwards. The accumulator sums
    unit vectors instead, so this is a property of the representation rather
    than a special case, and it is pinned because the failure is silent: a
    direction field that is wrong only near the cut looks fine everywhere else.
    """
    from pywave import rays

    a, b = np.radians(179.0), np.radians(-179.0)
    acc = rays.RayAccumulator(
        energy=np.array([[2.0]]), hits=np.array([[2.0]]),
        e_cos=np.array([[np.cos(a) + np.cos(b)]]),
        e_sin=np.array([[np.sin(a) + np.sin(b)]]))
    got = float(np.degrees(acc.theta[0, 0]))
    err = abs(abs(got) - 180.0)
    record("5b", "mean of +179 and -179 degrees", abs(got), reference=180.0,
           tol=0.1, unit="deg",
           note=f"The arithmetic mean of the two angles is 0 deg, pointing the "
                f"sea the other way. Directionality here is "
                f"{float(acc.directionality[0, 0]):.4f} -- near 1, correctly "
                f"reporting that these rays do agree.",
           passed=err < 0.1)
    assert err < 0.1
    assert float(acc.directionality[0, 0]) > 0.999


def test_directionality_reports_when_a_mean_direction_is_meaningless(record):
    """Two beams meeting head-on have a mean direction and no direction.

    `theta` is always a number. Behind a headland, energy arrives round both
    sides at once and that number describes nothing. `directionality` is what
    distinguishes the two cases, and without it a caller has no way to tell.
    """
    from pywave import rays

    opposed = rays.RayAccumulator(
        energy=np.array([[2.0]]), hits=np.array([[2.0]]),
        e_cos=np.array([[np.cos(0.0) + np.cos(np.pi)]]),
        e_sin=np.array([[np.sin(0.0) + np.sin(np.pi)]]))
    d = float(opposed.directionality[0, 0])
    record("5b", "directionality of two opposed beams", d, reference=0.0,
           tol=1e-9, unit="",
           note="Equal energy from opposite directions cancels to 0, so the "
                "mean direction is reported as untrustworthy rather than "
                "silently returned as a plausible angle.",
           passed=d < 1e-9)
    assert d < 1e-9


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
    """Gate 5b.3: min gain over wet cells, against 0.05. Measured on the crop.

    Read the name: 701 m of coastline is not the hard case. Ray theory has no
    diffraction, so a cell behind a headland that no ray reaches gets exactly
    zero, and the longer the sheltered run the deeper the hole -- the crop has
    no shadow long enough to reach zero on its own, and scored 0.098 here even
    before `wind_sea_floor` existed.

    The gate is stated on the full 7.5 x 8.6 km export, where pure ray theory
    gives **0.000** with 1.84% of wet cells under 0.05, and the wind-sea floor
    takes it to **0.0896** while leaving 5b.2 unchanged at 0.0123. That solve is
    minutes rather than seconds, so it lives in `scripts/gate5b.py` and not
    here; run it with and without `--no-wind-sea` to see the A/B.

    What this pins is that nothing *else* annihilates a cell -- not the
    decimation seam, not the deep-water normalisation, not the ray-retirement
    rule -- on bathymetry that has a real medial axis and a real ragged shore.
    """
    bathy, rf, _ = crop_rays
    g = rf.gain[bathy.depth > 0.0]
    lo = float(g.min())
    dark = float((g < 0.05).mean())
    record("5b", "min gain over wet cells (701 m crop)", lo, tol=0.05, unit="",
           note=f"{dark:.3%} of wet cells below 0.05. The gate is stated on the "
                f"full export, where scripts/gate5b.py measures 0.0896 with the "
                f"wind-sea floor against 0.000 without it; the crop is the "
                f"affordable stand-in, not the hard case.",
           passed=lo > 0.05)
    assert lo > 0.05


# ---------------------------------------------------------------------------
# 5b.4 / 5b.5 -- does it conserve energy, and does it still focus
# ---------------------------------------------------------------------------


def test_5b5_energy_flux_is_conserved_across_a_shoal(record):
    """Gate 5b.5: flux in against flux out, criterion 5%.

    A shoal deep enough that nothing breaks and periodic alongshore so nothing
    leaves the sides, which leaves conservation as the only thing being tested.
    The shoal focuses hard enough to form a caustic, so this also exercises the
    case the energy-accumulation scheme was chosen for: the textbook ray-tube
    `Kr = sqrt(b_0/b)` divides by a width that goes to zero right here.

    Read through the direction moments, not through `sin(theta)`. The flux needs
    the mean of the sines and `theta` is the sine of the mean; they part company
    exactly where rays cross, which is 5.8% by the far side.
    """
    from pywave import rays
    from pywave.spectrum import dispersion_k, group_velocity

    nx, ny, dx = 200, 200, 5.0
    X, Y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx)
    depth = 20.0 - 15.0 * np.exp(-(((X - 500) / 150) ** 2
                                   + ((Y - 500) / 150) ** 2))
    omega = 2 * np.pi / 8.0

    n_rays = 4000
    acc = rays.trace_rays(depth, omega, dx,
                          np.linspace(0.0, nx * dx, n_rays),
                          np.full(n_rays, (ny - 1.5) * dx),
                          np.full(n_rays, -np.pi / 2),
                          ds=0.25 * dx, break_depth=0.05, wrap_x=True)

    cg = group_velocity(dispersion_k(omega, np.maximum(depth, 1e-6)), depth)
    _, fy = acc.flux(cg)
    rows = (190, 170, 140, 100, 60, 30, 10)
    flux = np.array([fy[r].sum() * dx for r in rows])
    worst = float(np.abs(flux / flux[0] - 1).max())
    dirn = float(min(acc.directionality[r].min() for r in rows))

    record("5b", "energy flux drift from deep water to the far side", worst,
           reference=0.0, tol=0.05, unit="",
           note=f"Seven control lines across a focusing shoal. Directionality "
                f"falls to {dirn:.3f} downstream, so the rays genuinely cross; "
                f"reading the same flux through sin(theta) instead of the "
                f"moments drifts 5.8%.",
           passed=worst < 0.05)
    assert worst < 0.05


@pytest.mark.slow
def test_5b4_headlands_focus_and_bays_shelter(record):
    """Gate 5b.4: mean gain on headlands against adjacent bays, ratio > 1.2.

    The physics worth keeping. Continuity and no-annihilation are both
    satisfiable by a solver that has quietly flattened the field into a
    depth-only shoaling curve, and this is the check that says it has not.

    A cosine embayment, so headland and bay are known by construction rather
    than inferred from a shape analysis of real terrain. The measurement band is
    stated because the answer depends on it and honestly should: focusing
    accumulates shorewards, so the ratio is 1.10 at 5-10 m and 1.34 at 2-5 m.
    Quoting the number without the depth would be quoting the band.
    """
    import dataclasses

    from pywave import load_config, rays
    from pywave.bathymetry import Bathymetry

    cfg = load_config(REPO / "configs" / "coastal_bay.yaml")
    cfg = dataclasses.replace(
        cfg, wind=dataclasses.replace(cfg.wind, direction_rad=np.pi / 2))

    wavelength = 400.0
    bathy = Bathymetry.dean_embayment(nx=400, ny=400, dx=2.0, shoreline_y=600.0,
                                      amplitude=100.0, wavelength=wavelength,
                                      dean_a=0.25, max_depth=15.0)
    rf = rays.RayField.solve(bathy, cfg, cfg.omega_p, decimate=2, n_dirs=15,
                             rays_per_dir=1500, smooth_m=40.0)

    ax, ay = bathy.meta.axes()
    X, _ = np.meshgrid(ax, ay)
    # Land juts seaward where the cosine is -1, and the water intrudes where
    # it is +1 -- so the headlands sit half a wavelength off the bays.
    headland = np.abs(((X - wavelength / 2) % wavelength)) < 60.0
    bay = np.abs(X % wavelength) < 60.0
    band = (bathy.depth > 2.0) & (bathy.depth < 5.0)

    h = float(rf.gain[band & headland].mean())
    b = float(rf.gain[band & bay].mean())
    ratio = h / b
    record("5b", "headland / bay mean gain, 2-5 m depth", ratio, tol=1.2,
           unit="", note=f"headland {h:.4f} against bay {b:.4f} on a 400 m "
                         f"cosine embayment. Focusing accumulates shorewards: "
                         f"the same measurement is 1.10 at 5-10 m depth, so the "
                         f"band is part of the claim.",
           passed=ratio > 1.2)
    assert ratio > 1.2


# ---------------------------------------------------------------------------
# 5b.3 -- the one term here that is not ray theory
# ---------------------------------------------------------------------------


def test_5b3b_the_fetch_growth_exponent_matches_this_spectrum(record):
    """The floor must grow like the sea this package actually synthesises.

    `Hs ~ sqrt(fetch)` is the JONSWAP dimensionless-energy fit, and it is *not*
    what integrating this module's spectrum gives -- that sits a further
    `X~^0.05` above it, the documented inconsistency between JONSWAP's two
    independently fitted power laws.

    The extra 0.05 makes this spectrum grow *faster* with fetch, so a
    short-fetch cell sits *lower* than the plain energy law predicts. Taking the
    exponent from the fit would therefore over-floor every sheltered cell, by
    12% in Hs per decade of fetch ratio and 26% at a hundredth of the scene
    fetch, against a sea the surface synthesis never produces.

    So this checks the constant against `hs_spectral` itself -- a quadrature
    over the real spectrum -- rather than against the algebra it came from.
    """
    from pywave.rays import _HS_FETCH_EXPONENT
    from pywave.spectrum import hs_spectral

    u10, f_scene = 12.0, 40000.0
    worst = 0.0
    for f in (100.0, 1000.0, 5000.0, 20000.0):
        got = hs_spectral(u10, f) / hs_spectral(u10, f_scene)
        want = (f / f_scene) ** _HS_FETCH_EXPONENT
        worst = max(worst, abs(got - want) / got)
    record("5b", "fetch growth exponent vs the integrated spectrum", worst,
           tol=1e-3, unit="",
           note=f"Hs/Hs_scene = (F/F_scene)^{_HS_FETCH_EXPONENT} over four "
                f"decades of fetch ratio. The naive sqrt(F) sits 26% high at "
                f"F/F_scene = 0.01, so it would over-floor sheltered water.",
           passed=worst < 1e-3)
    assert worst < 1e-3


def test_shelter_fetch_measures_distance_to_land_and_knows_when_there_is_none(record):
    """A wall upwind at a known distance, and open water in the other direction.

    The `inf` case is the one that carries the weight. "Blocked at 900 m" and
    "not blocked at all" have to be different answers, because a clear direction
    is one the rays already carry and flooring it would count the open sea
    twice -- which would show up as gain clamped to 1 across the whole domain.
    """
    from pywave import rays

    nx = ny = 60
    dx = 10.0
    depth = np.full((ny, nx), 8.0)
    depth[:, :10] = -1.0                      # land wall on the -X side

    # theta = 0 travels toward +X, so upwind is -X and every cell sees the wall.
    # theta = pi travels toward -X: upwind is +X, open all the way out.
    fetch = rays.shelter_fetch(depth, dx, [0.0, np.pi])

    j = np.arange(nx)
    want = np.where(j >= 10, (j - 9) * dx, np.nan)[10:]
    got = fetch[0, ny // 2, 10:]
    err = float(np.abs(got - want).max())

    clear = bool(np.all(np.isinf(fetch[1, :, 10:])))
    record("5b", "shelter fetch to a known wall", err, reference=0.0, tol=1e-9,
           unit="m", note=f"50 cells at 10 m posts; the downwind direction "
                          f"returns inf everywhere (clear = {clear}), which is "
                          f"what stops the floor being applied to open water.",
           passed=err < 1e-9 and clear)
    assert err < 1e-9
    assert clear


def test_5b3c_exposed_water_receives_no_floor_at_all(record):
    """Exactly zero, not merely small -- that is what makes it safe to *add*.

    The floor is added to the transported energy rather than clamped over it.
    That is only defensible if it vanishes where the rays already have the
    answer; otherwise it is the open sea counted twice, and it would show up as
    a gain floored at 1.0 across every unobstructed cell.
    """
    from pywave import rays

    depth = np.full((40, 40), 10.0)           # no land anywhere
    thetas = np.radians([20.0, 45.0, 70.0])
    weights = np.full(3, 1 / 3)
    floor = rays.wind_sea_floor(depth, 25.0, thetas, weights, 7000.0)
    worst = float(np.abs(floor).max())
    record("5b", "wind-sea floor on fully exposed water", worst, reference=0.0,
           tol=0.0, unit="",
           note="Every direction leaves the domain without crossing land, so "
                "every direction is already carried by the rays.",
           passed=worst == 0.0)
    assert worst == 0.0


def _wall_scene(nx=40, ny=40, dx=50.0, land_cols=2):
    """Open water with a land wall on the -X side, so upwind fetch is exact."""
    depth = np.full((ny, nx), 8.0)
    depth[:, :land_cols] = -1.0
    return depth, dx


def test_per_band_floor_agrees_with_the_scalar_on_the_total(record):
    """Splitting the floor across bands must not create or destroy energy.

    The two are computed by different routes -- the scalar from the closed-form
    `(F/F_scene)^1.10`, the banded one by integrating the short-fetch spectrum
    over each band's wavenumber range -- so agreeing on the share-weighted total
    is a real cross-check on both, not an identity.

    It also settles what the per-band work does and does not affect: gate 5b.3
    is a statement about total height, so it is indifferent to this. What
    depends on it is every per-band quantity downstream of `transform`.
    """
    from pywave import load_config, rays, tiling
    from pywave.tiling import band_edges

    cfg = load_config(REPO / "configs" / "straits_crop.yaml")
    edges = band_edges(cfg.surface.tiles)
    ts = tiling.TileSet.build(cfg)
    shares = np.array([t.m0() / ts.m0() for t in ts.tiles])

    depth, dx = _wall_scene()
    th, w = np.array([0.0]), np.array([1.0])
    scalar = rays.wind_sea_floor(depth, dx, th, w, cfg.wind.fetch)
    banded = rays.wind_sea_floor(depth, dx, th, w, cfg.wind.fetch, bands=edges,
                                 u10=cfg.wind.speed, gamma=cfg.spectrum.gamma)

    wet = depth[0] > 0
    tot = np.tensordot(shares, banded[:, :, wet], axes=(0, 0))
    worst = float(np.abs(tot - scalar[:, wet]).max())
    record("5b", "per-band floor vs closed-form total", worst, reference=0.0,
           tol=2e-3, unit="",
           note="Sum over bands weighted by their deep-water shares, against "
                "(F/F_scene)^1.10. The small residual is the variance above the "
                "top band edge, which the banded form drops and the closed form "
                "keeps.",
           passed=worst < 2e-3)
    assert worst < 2e-3


def test_a_sheltered_bay_is_chop_not_scaled_down_swell(record):
    """A short fetch moves the peak up; it does not shrink the spectrum.

    This is the whole reason one number will not do. At 250 m of fetch the
    long band is empty and the short band is at half its deep-water energy --
    a scalar floor puts 82% of that energy in the long band instead, which is
    the right wave height made of entirely the wrong waves.

    The short band exceeding 1.0 at some fetches is not a bug and is worth
    pinning: a sheltered cell really does hold more short-wave energy than open
    water, because the whole spectrum has shifted into that band. A ratio capped
    at 1 cannot represent it.
    """
    from pywave import load_config, rays
    from pywave.tiling import band_edges

    cfg = load_config(REPO / "configs" / "straits_crop.yaml")
    edges = band_edges(cfg.surface.tiles)
    depth, dx = _wall_scene()
    banded = rays.wind_sea_floor(depth, dx, np.array([0.0]), np.array([1.0]),
                                 cfg.wind.fetch, bands=edges,
                                 u10=cfg.wind.speed, gamma=cfg.spectrum.gamma)

    row, col = 20, 6                      # 250 m of fetch at 50 m posts
    b1, b2, b3 = (float(banded[b, row, col]) for b in range(3))
    over_one = float(banded[2].max())
    record("5b", "short-band / long-band floor ratio at 250 m fetch",
           b3 / max(b1, 1e-12), tol=None, unit="",
           note=f"bands read {b1:.4f} / {b2:.4f} / {b3:.4f} of their own "
                f"deep-water energy. The short band peaks at {over_one:.3f} "
                f"across this scene -- above 1, which the scalar form cannot "
                f"express.",
           passed=b3 > b1)
    assert b3 > b2 > b1, "energy must move UP the bands as fetch shortens"
    assert over_one > 1.0, "a short-fetch sea outstrips deep water at high k"


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
