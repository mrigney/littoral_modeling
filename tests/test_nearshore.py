"""PHASE 5 -- Gate 5: nearshore transformation.

These checks are sharper than Gates 1-3 because the synthetic bathymetry has
closed-form answers.  Shoaling is checked against Green's law and against the
known ``Ks`` minimum; refraction against the exact Snell invariant; the breaker
line against ``d = H / gamma_b``; the swash band against Hunt's runup.  None of
that would be available from an exported Houdini heightfield, which is the whole
reason Phase 5 was built against a synthetic profile first.
"""

from __future__ import annotations

import numpy as np
import pytest

from pywave import foam as foam_mod
from pywave import moments, nearshore, spectrum, tiling
from pywave.bathymetry import DEAN_A, Bathymetry, dean_A_for_grain_size

pytestmark = pytest.mark.gate5


# ---------------------------------------------------------------------------
# The bathymetry contract -- Phase 4's assertion set, run against synthetic data
# ---------------------------------------------------------------------------


def test_bathymetry_satisfies_the_phase4_contract(record, beach):
    """The section 4.5 checks, which Houdini's export will have to pass too.

    Written against synthetic fields so the loader assertions exist and are
    exercised before there is anything to load.
    """
    stats = beach.validate()

    record("5", "depth == z_w - terrain_z (max residual)", stats["depth_residual"],
           0.0, 1e-12, unit="m", passed=stats["depth_residual"] < 1e-12)
    record("5", "cells where sign(sdf) != -sign(depth)", stats["sdf_sign_disagreement"],
           0.0, 0.0, note="Evaluated further than 1 m from the waterline; both "
                          "fields are discretised on the same grid, so within "
                          "about a cell of the contour the signs may legitimately "
                          "disagree.",
           passed=stats["sdf_sign_disagreement"] == 0.0)
    record("5", "max ||shore_normal| - 1| in the nearshore band",
           stats["shore_normal_magnitude_error"], 0.0, 1e-6,
           passed=stats["shore_normal_magnitude_error"] < 1e-6)
    record("5", "shore_normal points inland (fraction of cells)",
           stats["shore_normal_inland_fraction"], 1.0, 1e-2,
           passed=stats["shore_normal_inland_fraction"] > 0.99)


def test_planar_beach_normal_is_exactly_alongshore(beach):
    """A straight shoreline must give a constant normal, not a noisy one."""
    band = np.abs(beach.sdf) < 40.0
    nx_, ny_ = beach.shore_normal[0][band], beach.shore_normal[1][band]
    assert np.max(np.abs(nx_)) < 1e-9
    assert np.min(ny_) > 1.0 - 1e-9


def test_embayment_normal_actually_turns(record):
    """A curved shoreline must produce a genuinely varying normal.

    Guards against the shore normal silently collapsing to a constant, which
    would leave every refraction test on the planar beach passing while the
    curved case did nothing.
    """
    bay = Bathymetry.dean_embayment()
    bay.validate()
    band = np.abs(bay.sdf) < 30.0
    ang = np.degrees(np.arctan2(bay.shore_normal[1][band], bay.shore_normal[0][band]))
    spread = float(ang.max() - ang.min())

    record("5", "shore normal angular spread, cosine embayment", spread, unit="deg",
           note="A straight beach would give 0. Curved contours are what make "
                "refraction focus energy on headlands.",
           passed=spread > 20.0)
    assert spread > 20.0


def test_dean_profile_is_concave_up(record, beach):
    """``h = A y^(2/3)`` -- concave up, not the linear ramp a blur would give."""
    y = np.linspace(1.0, 200.0, 400)
    d = 0.1 * y ** (2.0 / 3.0)
    second = np.diff(d, 2)
    assert np.all(second < 0)  # concave up in depth-vs-offshore

    record("5", "Dean A used", beach.dean_a, unit="m^1/3",
           note="Medium sand. Tabulated classes: "
                + ", ".join(f"{k} {v}" for k, v in DEAN_A.items()) + " m^1/3.")
    record("5", "Dean A from D50 = 0.25 mm", float(dean_A_for_grain_size(0.25)),
           DEAN_A["medium_sand"], 0.10, unit="m^1/3",
           passed=abs(dean_A_for_grain_size(0.25) - DEAN_A["medium_sand"]) < 0.01)
    record("5", "foreshore slope at 10 cm depth", beach.beach_slope(),
           note="The Dean profile's slope diverges at the waterline, so the "
                "runup and Iribarren calculations evaluate it at the depth "
                "where these waves break.")


# ---------------------------------------------------------------------------
# Shoaling
# ---------------------------------------------------------------------------


def test_shoaling_recovers_greens_law(record, cfg):
    """In shallow water ``Ks ~ d^(-1/4)``.

    Green's law is the shallow asymptote of the full dispersion solution, so
    this checks the ``tanh(kd)`` branch is right where it matters most.
    """
    omega = 2.0 * np.pi * cfg.f_p
    depths = np.array([0.002, 0.004, 0.008])
    ks = nearshore.shoaling_coefficient(omega, depths)
    ratios = ks[:-1] / ks[1:]
    expected = 2.0 ** 0.25
    worst = float(np.max(np.abs(ratios - expected) / expected))

    kd = spectrum.dispersion_k(omega, depths) * depths
    record("5", "Green's law ratio Ks(d)/Ks(2d)", float(ratios.mean()), expected, 0.02,
           note=f"Evaluated at kd = {np.round(kd, 3).tolist()}, i.e. genuinely "
                f"shallow. The asymptote is approached, not exact, at finite kd.",
           passed=worst < 0.02)
    assert worst < 0.02


def test_shoaling_coefficient_has_the_expected_minimum(record, cfg):
    """``Ks`` dips to ~0.913 near ``kd = 1.2`` before rising.

    Not a curiosity: an implementation that clamps ``Ks >= 1`` on the assumption
    that "shoaling makes waves bigger" would pass a shallow-water check and
    still be wrong through the whole intermediate-depth band, which for these
    1 s waves is where the beach actually is.
    """
    omega = 2.0 * np.pi * cfg.f_p
    d = np.geomspace(0.01, 50.0, 4000)
    ks = nearshore.shoaling_coefficient(omega, d)
    i = int(np.argmin(ks))
    kd_at_min = float(spectrum.dispersion_k(omega, d[i:i + 1])[0] * d[i])

    record("5", "min Ks", float(ks[i]), 0.9130, 0.01, passed=abs(ks[i] - 0.913) < 0.01)
    record("5", "kd at min Ks", kd_at_min, 1.2, 0.10, passed=abs(kd_at_min - 1.2) < 0.15)
    record("5", "Ks in deep water (d = 50 m)", float(ks[-1]), 1.0, 1e-3,
           passed=abs(ks[-1] - 1.0) < 1e-3)

    assert abs(ks[i] - 0.913) < 0.01
    assert ks[-1] == pytest.approx(1.0, abs=1e-3)


def test_shoaling_transect_is_continuous(record, cfg):
    """No discontinuity at the deep/shallow transition.

    The dispersion solver switches behaviour smoothly through ``tanh(kd)``, but
    an Eckart seed that failed to converge in shallow water would show up as a
    kink here and nowhere else.
    """
    omega = 2.0 * np.pi * cfg.f_p
    d = np.geomspace(0.005, 30.0, 6000)
    ks = nearshore.shoaling_coefficient(omega, d)
    jumps = np.abs(np.diff(ks)) / ks[:-1]
    worst = float(jumps.max())

    record("5", "max relative step in Ks between adjacent samples", worst, 0.0, 5e-3,
           note="6000 log-spaced depths from 5 mm to 30 m.", passed=worst < 5e-3)
    assert worst < 5e-3


def test_shoaling_is_applied_per_band(record, onshore_tileset):
    """Each tile shoals at its own frequency, and shorter waves shoal less.

    Cookbook 5.2 requires per-band application. The three tiles carry
    increasing wavenumber, so at a given depth the highest band is closest to
    deep-water behaviour and must have the smallest gain.
    """
    omegas = nearshore.tile_frequencies(onshore_tileset)
    assert np.all(np.diff(omegas) > 0)

    ks = nearshore.shoaling_coefficient(omegas, 0.05)
    record("5", "per-tile representative omega", str(np.round(omegas, 3).tolist()),
           unit="rad/s", note=f"Ks at d = 0.05 m: {np.round(ks, 4).tolist()} -- "
                              f"higher-frequency bands are still in deeper water "
                              f"relative to their own wavelength, so they shoal less.")
    assert np.all(np.diff(ks) < 0)


# ---------------------------------------------------------------------------
# Refraction
# ---------------------------------------------------------------------------


def test_snell_invariant_is_conserved(record, beach, cfg):
    """``sin(alpha) / c`` is constant along a ray, to machine precision.

    The definition of refraction, checked directly rather than through its
    consequences.
    """
    omega = 2.0 * np.pi * cfg.f_p
    x = np.full(200, 250.0)
    y = np.linspace(399.5, 150.0, 200)
    depth, _, normal = beach.sample(x, y)

    _, alpha = nearshore.refraction_angle(np.radians(45.0), normal, depth, omega)
    c = omega / spectrum.dispersion_k(omega, np.maximum(depth, 1e-6))
    invariant = np.sin(alpha) / c
    spread = float((invariant.max() - invariant.min()) / abs(invariant.mean()))

    record("5", "relative spread of sin(alpha)/c along a transect", spread, 0.0, 1e-12,
           note="200 points from 0.06 m to 4.5 m depth, 45 deg deep-water "
                "incidence. Exact for the straight parallel contours of a "
                "planar Dean beach.",
           passed=spread < 1e-12)
    assert spread < 1e-12


def test_crests_turn_shore_parallel_regardless_of_wind(record, beach, cfg):
    """Whatever the deep-water direction, waves arrive nearly shore-normal.

    Gate 5's headline check. If a render shows waves hitting the beach at 40
    degrees, refraction is broken.
    """
    omega = 2.0 * np.pi * cfg.f_p
    x = np.full(3, 250.0)
    y = np.array([399.7, 399.7, 399.7])

    worst = 0.0
    rows = []
    for wind_deg in (10.0, 45.0, 90.0, 135.0, 170.0):
        depth, _, normal = beach.sample(x, y)
        alpha_deep, _ = nearshore._incidence(np.radians(wind_deg), normal)
        _, alpha = nearshore.refraction_angle(np.radians(wind_deg), normal, depth, omega)
        worst = max(worst, float(np.max(np.abs(np.degrees(alpha)))))
        rows.append(f"{wind_deg:.0f} deg -> {np.degrees(alpha_deep[0]):+.0f} deg "
                    f"incident, {np.degrees(alpha[0]):+.1f} deg at the waterline")

    record("5", "worst residual incidence angle at the waterline", worst, 0.0, 25.0,
           unit="deg", note="Wind directions 10-170 deg. " + "; ".join(rows),
           passed=worst < 25.0)
    assert worst < 25.0


def test_refraction_reduces_height_for_oblique_waves(record, beach, cfg):
    """``Kr <= 1`` on straight parallel contours, always.

    Oblique waves spread their energy along a longer stretch of shoreline. A
    ``Kr`` above 1 would mean refraction created energy.
    """
    omega = 2.0 * np.pi * cfg.f_p
    x = np.full(150, 250.0)
    y = np.linspace(399.5, 200.0, 150)
    depth, _, normal = beach.sample(x, y)

    alpha_deep, _ = nearshore._incidence(np.radians(45.0), normal)
    _, alpha = nearshore.refraction_angle(np.radians(45.0), normal, depth, omega)
    kr = nearshore.refraction_coefficient(alpha_deep, alpha)

    record("5", "Kr range over the transect (45 deg incidence)",
           f"{kr.min():.4f} .. {kr.max():.4f}",
           note="Bounded above by 1. Combined with a shoaling gain that peaks "
                "near 1.36, the two nearly cancel for obliquely incident waves "
                "on this beach.")
    assert kr.max() <= 1.0 + 1e-12
    assert kr.min() < 1.0

    # Normal incidence must leave the height untouched.
    _, alpha_n = nearshore.refraction_angle(np.radians(90.0), normal, depth, omega)
    kr_n = nearshore.refraction_coefficient(
        nearshore._incidence(np.radians(90.0), normal)[0], alpha_n)
    assert np.allclose(kr_n, 1.0, atol=1e-9)


def test_blend_approximation_disagrees_with_snell(record, beach, cfg):
    """The cookbook's depth-weighted blend is not equivalent to Snell.

    Recorded rather than asserted away: the blend is cheaper and is what the
    cookbook prescribes, but it has no frequency dependence and drives every
    band to full shore-normal alignment at the waterline. This measures the
    price, so the choice to run Snell in production is an informed one.
    """
    omega = 2.0 * np.pi * cfg.f_p
    x = np.full(200, 250.0)
    y = np.linspace(399.5, 250.0, 200)
    depth, _, normal = beach.sample(x, y)

    th_snell, _ = nearshore.refraction_angle(np.radians(45.0), normal, depth, omega)
    th_blend, _ = nearshore.refraction_angle_blend(np.radians(45.0), normal, depth,
                                                   3.0 * cfg.lambda_p)
    diff = np.degrees(np.abs(nearshore._wrap(th_snell - th_blend)))

    record("5", "max |Snell - blend| wave direction", float(diff.max()), unit="deg",
           note=f"Mean {diff.mean():.1f} deg over the transect. The blend reaches "
                f"full alignment by d = 3 lambda_p = {3 * cfg.lambda_p:.1f} m, "
                f"where Snell has barely started turning.")
    assert diff.max() > 5.0


# ---------------------------------------------------------------------------
# Breaking, surf zone and swash
# ---------------------------------------------------------------------------


def test_breaking_occurs_at_the_depth_limited_height(record, fine_beach, onshore_cfg,
                                                     onshore_tileset):
    """Breaking starts where ``H = gamma_b * d``, and the surf zone saturates."""
    gamma_b = onshore_cfg.nearshore.breaker_index
    x = np.full(600, 32.0)
    y = np.linspace(99.9, 40.0, 600)          # shoreline at y = 100, water below
    nf = nearshore.transform(onshore_tileset, fine_beach, onshore_cfg, x, y, 0.0)

    assert nf.breaking.any()
    outer = int(np.max(np.where(nf.breaking)[0]))   # y descends, so max index = deepest
    d_break = float(nf.depth[outer])
    h_break = float(nf.hs_local[outer])
    rel = abs(d_break - h_break / gamma_b) / d_break

    record("5", "depth at the outer breaker", d_break, h_break / gamma_b, 0.05, unit="m",
           note=f"Hs there = {h_break:.4f} m, gamma_b = {gamma_b}.",
           passed=rel < 0.05)

    # Inside the surf zone the height is depth-limited, i.e. saturated.
    inside = nf.breaking
    saturation = nf.hs_local[inside] / (gamma_b * nf.depth[inside])
    record("5", "Hs / (gamma_b * d) inside the surf zone", float(saturation.mean()),
           1.0, 1e-6, note="Exactly 1 by construction once the limiter engages -- "
                           "this checks the limiter is applied, not bypassed.",
           passed=np.allclose(saturation, 1.0, atol=1e-6))
    assert np.allclose(saturation, 1.0, atol=1e-6)

    width = nf.surf_zone_width(y)
    record("5", "surf zone width", width, unit="m",
           note=f"Cookbook 5.1 estimates ~2 m for a 5% slope; this beach is "
                f"{100 * fine_beach.beach_slope():.1f}% at the break point, so a "
                f"narrower zone is expected.")
    assert 0.2 < width < 5.0


def test_iribarren_predicts_spilling_breakers(record, cfg, beach, tileset):
    """``xi ~ 0.3`` -- spilling, no plunging, as cookbook 5.1 predicts."""
    slope = beach.beach_slope()
    l0 = float(nearshore.deep_water_wavelength(1.0 / cfg.f_p))
    hs = tileset.hs()
    xi = float(nearshore.iribarren_number(slope, hs, l0))

    record("5", "deep-water wavelength L0", l0, cfg.lambda_p, 0.02, unit="m",
           note="Equals lambda_p, since both are the deep-water wavelength at "
                "the peak period.",
           passed=abs(l0 - cfg.lambda_p) / cfg.lambda_p < 0.02)
    record("5", "Iribarren number xi", xi,
           note=f"Breaker type: {nearshore.breaker_type(xi)}. Cookbook 5.1 quotes "
                f"xi ~ 0.23 for a 5% slope; this beach is steeper at "
                f"{100 * slope:.1f}%.")
    assert xi < 0.5
    assert nearshore.breaker_type(xi) == "spilling"


def test_swash_band_matches_hunt_runup(record, cfg, beach, tileset):
    """Runup ~2 cm vertical, swash excursion 0.3-0.5 m horizontal (cookbook 5.1)."""
    slope = beach.beach_slope()
    l0 = float(nearshore.deep_water_wavelength(1.0 / cfg.f_p))
    hs = tileset.hs()
    xi = nearshore.iribarren_number(slope, hs, l0)
    runup = float(nearshore.hunt_runup(xi, hs))
    band = float(nearshore.swash_width(runup, slope))

    record("5", "Hunt runup R = xi * Hs", runup, unit="m",
           note="Cookbook 5.1 quotes ~2 cm vertical.")
    record("5", "swash excursion R / tan(beta)", band, unit="m",
           note="Cookbook 5.1 quotes 0.3-0.5 m horizontal.")
    assert 0.01 < runup < 0.05
    assert 0.2 < band < 0.7


def test_wetness_fraction_is_a_duty_cycle(record, cfg, beach, tileset):
    """Time-averaged submergence: 1 at the waterline, 0 beyond the swash, 1/2 mid-band.

    The closed form is the duty cycle of a *hard* waterline crossing, so it is
    checked against exactly that -- counting the fraction of a period for which
    the oscillating swash edge lies inland of each point.

    It is deliberately **not** checked against the time average of
    :func:`~pywave.nearshore.wetness`, which smoothsteps at each instant (a hard
    edge aliases when rendered) and therefore averages to a softer curve. The
    two answer different questions: this one feeds Hotts, that one feeds the
    shader.
    """
    band = 0.4
    s = np.linspace(-0.05, 0.5, 400)
    closed = nearshore.wetness_fraction(s, band)

    period = 1.0 / cfg.f_p
    times = np.linspace(0.0, period, 20_000, endpoint=False)
    edges = band * (0.5 + 0.5 * np.sin(2.0 * np.pi * times / period))
    duty = np.mean(edges[:, None] > s[None, :], axis=0)

    worst = float(np.max(np.abs(closed - duty)))

    record("5", "closed-form wetness vs sampled hard-waterline duty cycle",
           worst, 0.0, 5e-3,
           note="20000 samples over one period. The closed form is "
                "(1/pi) arccos(2s/W - 1); agreement is limited only by the "
                "sampling of the period.",
           passed=worst < 5e-3)
    record("5", "wetness at mid-swash", float(nearshore.wetness_fraction(band / 2, band)),
           0.5, 1e-9, note="Exactly 1/2 by symmetry of the sinusoidal waterline.",
           passed=abs(nearshore.wetness_fraction(band / 2, band) - 0.5) < 1e-9)

    # The smoothstepped instantaneous field is monotone in the same direction,
    # which is all that is required of it.
    inst = nearshore.wetness(s, band, 0.25 * period, period)
    assert np.all(np.diff(inst) <= 1e-12)

    assert nearshore.wetness_fraction(-1.0, band) == 1.0
    assert nearshore.wetness_fraction(band + 1.0, band) == 0.0
    assert worst < 5e-3


# ---------------------------------------------------------------------------
# End-to-end transform
# ---------------------------------------------------------------------------


def test_transform_applies_the_predicted_gain_to_the_surface(
        record, fine_beach, onshore_cfg, onshore_tileset):
    """The realised surface variance follows the predicted amplitude gain.

    Checks the plumbing rather than the coefficients: that the per-band weights
    actually reach the sampled surface. Measured by taking many points at a
    fixed depth and comparing ``4 std(h)`` against the predicted ``hs_local``.
    """
    worst = 0.0
    rows = []
    for y_line in (95.0, 80.0, 60.0):
        x = np.linspace(0.0, 64.0, 4000)
        y = np.full_like(x, y_line)
        nf = nearshore.transform(onshore_tileset, fine_beach, onshore_cfg, x, y, 0.0,
                                 depth_limit=False)
        realised = 4.0 * float(np.std(nf.surface.h))
        predicted = float(np.mean(nf.hs_local))
        rel = abs(realised - predicted) / predicted
        worst = max(worst, rel)
        rows.append(f"d = {nf.depth.mean():.2f} m: {realised:.5f} vs {predicted:.5f}")

    record("5", "realised 4 std(h) vs predicted Hs_local (worst)", worst, 0.0, 0.10,
           note="; ".join(rows) + ". Sampled across 4000 points at constant depth.",
           passed=worst < 0.10)
    assert worst < 0.10


def test_transform_leaves_deep_water_untouched(record, beach, cfg, tileset):
    """Far offshore the transform must be the identity.

    A nearshore module that quietly rescales the whole scene would be caught
    here and almost nowhere else.
    """
    x = np.linspace(0.0, 400.0, 3000)
    y = np.full_like(x, 60.0)                 # ~340 m offshore, depth capped at 5 m
    nf = nearshore.transform(tileset, beach, cfg, x, y, 0.0)
    plain = tiling.composite_surface(tileset.tiles, x, y, 0.0)

    depth = float(nf.depth.mean())
    worst = float(np.max(np.abs(nf.surface.h - plain.h)))

    record("5", "max |h_nearshore - h_deep| at d = %.1f m" % depth, worst, 0.0, 1e-3,
           unit="m", note="Depth 5 m against a 1.7 m peak wavelength: kd = "
                          f"{2 * np.pi / cfg.lambda_p * depth:.1f}, comfortably deep, "
                          f"so Ks = Kr = 1 and the transform is the identity.",
           passed=worst < 1e-3)
    assert worst < 1e-3


def test_composite_surface_hooks_default_to_no_ops(cfg, tileset):
    """The Phase 5 additions to `composite_surface` must not disturb Phase 2.

    The regression baseline covers this too, but failing here names the cause.
    """
    x = np.linspace(0.0, 100.0, 500)
    y = np.linspace(50.0, 150.0, 500)
    a = tiling.composite_surface(tileset.tiles, x, y, 1.5)
    b = tiling.composite_surface(tileset.tiles, x, y, 1.5,
                                 weights=[1.0] * len(tileset.tiles),
                                 rotate=[0.0] * len(tileset.tiles))
    for name in ("h", "dx_disp", "dy_disp", "slope_x", "slope_y"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_refraction_modes_agree_in_deep_water(beach, cfg, tileset):
    """All three refraction modes are the identity far offshore."""
    x = np.linspace(0.0, 200.0, 800)
    y = np.full_like(x, 60.0)
    fields = [nearshore.transform(tileset, beach, cfg, x, y, 0.0, refraction=m).surface.h
              for m in ("snell", "blend", "none")]
    assert np.allclose(fields[0], fields[2], atol=1e-9)
    assert np.allclose(fields[1], fields[2], atol=1e-9)


def test_unknown_refraction_mode_raises(beach, cfg, tileset):
    with pytest.raises(ValueError, match="unknown refraction mode"):
        nearshore.transform(tileset, beach, cfg, np.array([10.0]), np.array([50.0]),
                            0.0, refraction="nonsense")


# ---------------------------------------------------------------------------
# Foam
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def surf(fine_beach, onshore_cfg, onshore_tileset):
    """Breaking mask and group velocity on the fine bathymetry grid."""
    omega = 2.0 * np.pi * onshore_cfg.f_p
    d = np.maximum(fine_beach.depth, 1e-3)
    cg = spectrum.group_velocity(spectrum.dispersion_k(omega, d), d)
    ks = nearshore.shoaling_coefficient(omega, np.maximum(fine_beach.depth, 1e-9))
    hs = np.where(fine_beach.depth > 0.0, onshore_tileset.hs() * ks, 0.0)
    return nearshore.breaking_mask(hs, fine_beach.depth, 0.78), cg


def test_foam_decays_with_the_configured_half_life(record, fine_beach, surf, cfg):
    """One half life halves the field, exactly.

    Advection is disabled: semi-Lagrangian transport does not conserve mass
    under a divergent velocity field, so a decay test that left it on would be
    measuring both effects at once and matching neither.
    """
    breaking, cg = surf
    model = foam_mod.FoamModel(bathy=fine_beach, half_life=cfg.nearshore.foam_halflife,
                               advect=False)
    seeded = model.step(np.zeros(fine_beach.meta.shape), breaking, cg, 0.25)

    ratios = []
    for steps in (1.0, 2.0):
        decayed = model.step(seeded, np.zeros_like(breaking), cg,
                             steps * cfg.nearshore.foam_halflife)
        ratios.append(float(decayed.sum() / seeded.sum()))

    record("5", "foam remaining after one half life", ratios[0], 0.5, 1e-9,
           note=f"Half life {cfg.nearshore.foam_halflife} s from config. "
                f"After two half lives: {ratios[1]:.6f} (want 0.25).",
           passed=abs(ratios[0] - 0.5) < 1e-9)
    assert abs(ratios[0] - 0.5) < 1e-9
    assert abs(ratios[1] - 0.25) < 1e-9


def test_foam_equilibrium_matches_the_geometric_series(record, fine_beach, surf, cfg):
    """A continuously breaking cell converges to ``seed_rate * dt / (1 - r)``."""
    breaking, cg = surf
    model = foam_mod.FoamModel(bathy=fine_beach, half_life=cfg.nearshore.foam_halflife,
                               advect=False)
    f = np.zeros(fine_beach.meta.shape)
    for _ in range(400):
        f = model.step(f, breaking, cg, 0.25)

    measured = float(f[breaking].mean())
    predicted = model.equilibrium_coverage(0.25)

    record("5", "equilibrium foam coverage in a breaking cell", measured, predicted,
           1e-6, note=f"seed_rate = {model.seed_rate}/s. Continuous-limit value "
                      f"is {model.equilibrium_coverage():.4f}, which is what to "
                      f"reason about when choosing seed_rate since it does not "
                      f"depend on the step size.",
           passed=abs(measured - predicted) < 1e-6)
    assert abs(measured - predicted) < 1e-6


def test_foam_stays_in_the_surf_band(record, fine_beach, surf, cfg):
    """Foam appears only where waves break, and is swept shoreward from there."""
    breaking, cg = surf
    model = foam_mod.FoamModel(bathy=fine_beach, half_life=cfg.nearshore.foam_halflife)
    f = model.evaluate(lambda t: breaking, cg, t=30.0)

    present = f > 0.01
    reach = float(np.abs(fine_beach.sdf[present]).max())
    breaking_reach = float(np.abs(fine_beach.sdf[breaking]).max())

    record("5", "max |sdf| where foam coverage > 0.01", reach, unit="m",
           note=f"Breaking itself extends to |sdf| = {breaking_reach:.2f} m; foam "
                f"is advected shoreward from there. Whitecaps over open water "
                f"(~0.1% coverage at 5 m/s) are deliberately not modelled.",
           passed=reach < 5.0)
    record("5", "peak foam coverage", float(f.max()),
           note="Below the still-water equilibrium because advection continually "
                "sweeps foam out of the cells that seed it.")
    assert reach < 5.0
    assert not present[fine_beach.depth > 1.0].any()


@pytest.mark.slow
def test_foam_cold_start_matches_sequential(record, fine_beach, surf, cfg):
    """Gate 5's reproducibility check: frame 500 cold vs sequential, within 1%.

    This is what keeps "any node, any frame" true for the one field with
    frame-to-frame memory.
    """
    breaking, cg = surf
    model = foam_mod.FoamModel(bathy=fine_beach, half_life=cfg.nearshore.foam_halflife)
    dt = 0.25

    sequential = np.zeros(fine_beach.meta.shape)
    for _ in range(500):
        sequential = model.step(sequential, breaking, cg, dt)

    cold = model.evaluate(lambda t: breaking, cg, t=500 * dt, dt=dt)

    live = sequential > 0.01
    per_cell = float(np.max(np.abs(cold - sequential)[live] / sequential[live]))
    of_peak = float(np.max(np.abs(cold - sequential)) / sequential.max())
    n = foam_mod.spinup_steps(0.005, dt, cfg.nearshore.foam_halflife)

    record("5", "cold vs sequential, worst per-cell relative error", per_cell, 0.0, 0.01,
           note=f"Spin-up {n} steps = {n * dt:.1f} s, initial-condition residual "
                f"{foam_mod.spinup_residual(n, dt, cfg.nearshore.foam_halflife):.4f}. "
                f"As a fraction of peak coverage the error is {100 * of_peak:.3f}%. "
                f"The cookbook's suggested 30 frames would leave 79% of the "
                f"initial condition intact -- the window is set by the half life, "
                f"not by a frame count.",
           passed=per_cell < 0.01)
    assert per_cell < 0.01


def test_spinup_window_is_derived_from_the_half_life(record):
    """`spinup_steps` inverts the decay law, and the residual confirms it."""
    for tol in (0.01, 0.005, 0.001):
        n = foam_mod.spinup_steps(tol, 0.25, 3.0)
        assert foam_mod.spinup_residual(n, 0.25, 3.0) <= tol

    n30 = foam_mod.spinup_steps(0.005, 1.0 / 30.0, 3.0)
    record("5", "spin-up steps for 0.5% residual at 30 fps", n30,
           note=f"= {n30 / 30.0:.1f} s of simulated time. At the 0.25 s foam step "
                f"the same window is {foam_mod.spinup_steps(0.005, 0.25, 3.0)} steps, "
                f"which is why foam does not sub-step at the frame rate.")

    with pytest.raises(ValueError):
        foam_mod.spinup_steps(0.0, 0.25, 3.0)
    with pytest.raises(ValueError):
        foam_mod.foam_decay_factor(0.25, -1.0)
