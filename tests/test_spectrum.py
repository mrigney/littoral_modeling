"""PHASE 3 -- Gate 1: spectrum and moments.

The check that matters most here is the Jacobian
(:func:`test_jacobian_conserves_variance`). Getting `S(k,theta) = S(f) D Cg /
(2 pi k)` wrong produces an `Hs` that is off by a constant factor, which looks
like a plausible sea state and survives every other check in this file.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from pywave import constants, moments, spectrum

pytestmark = pytest.mark.gate1


# ---------------------------------------------------------------------------
# Significant wave height
# ---------------------------------------------------------------------------


def test_hs_relation_holds_across_fetch(record, scene):
    """The spectral/fit Hs ratio follows `0.78696 * X~^0.05` over four decades.

    This replaces the cookbook's "within 2%" criterion, which no correct
    implementation can satisfy -- see the Gate deviations section of the report.
    It is a strictly stronger constraint: it must hold at every fetch, not at
    one, and it pins the *exponent*, so a constant normalisation error (exponent
    0) or a Jacobian factor error would both break it.
    """
    _, _, gamma, _ = scene

    u10s = np.array([3.0, 5.0, 8.0, 12.0, 20.0])
    fetches = np.array([3e2, 1e3, 1e4, 1e5, 1e6, 5e6])

    worst = 0.0
    log_x, log_ratio = [], []
    for u10 in u10s:
        for fetch in fetches:
            ratio = spectrum.hs_spectral(u10, fetch, gamma) / spectrum.fetch_limited_hs(u10, fetch)
            pred = spectrum.hs_ratio_spectral_to_fit(u10, fetch)
            worst = max(worst, abs(ratio - pred) / pred)
            log_x.append(np.log(spectrum.dimensionless_fetch(u10, fetch)))
            log_ratio.append(np.log(ratio))

    exponent, log_coeff = np.polyfit(log_x, log_ratio, 1)

    record("1", "Hs ratio vs predicted relation (worst over 30 scenes)",
           worst, 0.0, 1e-3, note=(
               f"U10 3-20 m/s, fetch 300 m - 5000 km, i.e. X~ from "
               f"{spectrum.dimensionless_fetch(20.0, 3e2):.0f} to "
               f"{spectrum.dimensionless_fetch(3.0, 5e6):.2e}."),
           passed=worst < 1e-3)
    record("1", "fitted exponent p in hs_ratio = c * X~^p", float(exponent), 0.05, 1e-3,
           note="Structural prediction is exactly 0.05. A constant-factor "
                "normalisation bug would give p = 0.",
           passed=abs(exponent - 0.05) < 1e-3)
    record("1", "fitted coefficient c", float(np.exp(log_coeff)), 0.78696, 1e-2,
           passed=abs(np.exp(log_coeff) - 0.78696) / 0.78696 < 1e-2)

    assert worst < 1e-3
    assert abs(exponent - 0.05) < 1e-3


def test_test_lake_hs_values(record, scene):
    """Report both Hs values for the scene, and confirm the gap is the predicted one."""
    u10, fetch, gamma, _ = scene
    hs_s = spectrum.hs_spectral(u10, fetch, gamma)
    hs_f = spectrum.fetch_limited_hs(u10, fetch)

    record("1", "Hs (spectral, `hs_spectral`)", hs_s, unit="m",
           note="This is the scene's Hs -- the value a realised FFT surface reproduces.")
    record("1", "Hs (energy growth fit, `fetch_limited_hs`)", hs_f, unit="m")
    record("1", "spectral / fit ratio", hs_s / hs_f,
           spectrum.hs_ratio_spectral_to_fit(u10, fetch), 1e-3,
           note=f"Gap is {100 * (hs_s / hs_f - 1):+.1f}%; the cookbook's 2% "
                f"criterion is unmeetable, see Gate deviations.",
           passed=True)

    assert abs(hs_s / hs_f - spectrum.hs_ratio_spectral_to_fit(u10, fetch)) < 1e-3


def test_peak_frequency_from_integration(record, scene):
    """The numerical argmax of S(f) matches the closed-form f_p within 2%."""
    u10, fetch, gamma, _ = scene
    _, fp_closed, _ = spectrum.jonswap_params(u10, fetch)

    f = np.linspace(fp_closed / 8.0, fp_closed * 8.0, 20_001)
    fp_numeric = float(f[np.argmax(spectrum.jonswap_sf(f, u10, fetch, gamma))])
    rel = abs(fp_numeric - fp_closed) / fp_closed

    record("1", "f_p from argmax of S(f)", fp_numeric, fp_closed, 0.02, unit="Hz",
           passed=rel < 0.02)
    record("1", "T_p", 1.0 / fp_closed, unit="s")
    record("1", "lambda_p (deep water)", 2.0 * np.pi * constants.G / (2 * np.pi * fp_closed) ** 2,
           unit="m")
    assert rel < 0.02


# ---------------------------------------------------------------------------
# Directional spreading
# ---------------------------------------------------------------------------


def test_spreading_integrates_to_one(record, scene):
    """`int D dtheta = 1` to 1e-6 at every frequency.

    Adaptive quadrature, deliberately: uniform sampling of `cos^(2s)` converges
    only algebraically, so a rectangle rule at n = 8192 still shows 5e-5 error
    that has nothing to do with the normalisation being tested.
    """
    u10, fetch, _, theta_wind = scene
    _, f_p, _ = spectrum.jonswap_params(u10, fetch)

    worst, worst_at = 0.0, None
    for ratio in (0.125, 0.25, 0.5, 0.8, 1.0, 1.2, 2.0, 5.0, 20.0, 100.0):
        f = ratio * f_p
        val, _ = quad(
            lambda th: float(spectrum.spreading(np.array([th]), np.array([f]), f_p, theta_wind)[0]),
            -np.pi, np.pi, limit=400, epsabs=1e-13, epsrel=1e-13,
        )
        if abs(val - 1.0) > worst:
            worst, worst_at = abs(val - 1.0), ratio

    record("1", "max |int D dtheta - 1| over f/f_p in [0.125, 100]", worst, 0.0, 1e-6,
           note=f"Worst at f/f_p = {worst_at}. Adaptive quadrature; the closed-form "
                f"normalisation is exact to machine precision.",
           passed=worst < 1e-6)
    assert worst < 1e-6


def test_spreading_norm_closed_matches_quadrature(record):
    """The production closed form agrees with the quadrature oracle to 1e-10.

    `spreading_norm` exists in two implementations precisely so this can be
    asserted: the closed form is what runs on 512^2 grids, the quadrature is what
    the cookbook prescribes, and neither is trusted without the other.
    """
    s = np.array([0.5, 1.0, 5.0, 9.77, 11.5, 50.0, 200.0])
    closed = spectrum.spreading_norm(s, "closed")
    quadrature = spectrum.spreading_norm(s, "quad", n_quad=200_000)
    rel = float(np.max(np.abs(closed - quadrature) / closed))

    record("1", "max rel. difference, spreading_norm closed vs quad", rel, 0.0, 1e-10,
           note=f"Over s = {[float(v) for v in s]}.", passed=rel < 1e-10)
    assert rel < 1e-10


def test_spreading_is_finite_and_non_negative(scene):
    """No NaNs from the fractional power, over the full angular range."""
    u10, fetch, _, theta_wind = scene
    _, f_p, _ = spectrum.jonswap_params(u10, fetch)
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 2001)
    for ratio in (0.2, 1.0, 5.0):
        d = spectrum.spreading(theta, np.full_like(theta, ratio * f_p), f_p, theta_wind)
        assert np.all(np.isfinite(d))
        assert np.all(d >= 0.0)


def test_unimplemented_spreading_models_raise():
    """`donelan` raises rather than silently falling back to cos2s."""
    with pytest.raises(NotImplementedError, match="Donelan"):
        spectrum.spreading_exponent(np.array([1.0]), 1.0, model="donelan")
    with pytest.raises(ValueError, match="unknown spreading model"):
        spectrum.spreading_exponent(np.array([1.0]), 1.0, model="nonsense")


# ---------------------------------------------------------------------------
# The Jacobian -- the important one
# ---------------------------------------------------------------------------


def test_jacobian_conserves_variance(record, scene):
    """`int S(k) d2k` equals `int S(f) df` within 1%.

    The wavenumber side is integrated by brute force on a Cartesian polar grid
    via `jonswap_sk`, so it exercises the `Cg / (2 pi k)` Jacobian and the
    spreading normalisation independently of the collapsed radial form used
    everywhere else. An oracle sharing that shortcut would agree with it no
    matter how wrong both were.
    """
    u10, fetch, gamma, theta_wind = scene

    m0_f = moments.moment_omega(0, u10, fetch, gamma)
    m0_k = moments.integrate_sk_polar(u10, fetch, theta_wind, gamma=gamma, order=0)
    rel = abs(m0_k - m0_f) / m0_f

    record("1", "m0 from int S(k) d2k (2-D polar, via jonswap_sk)", m0_k, m0_f, 0.01,
           unit="m^2", note="The Jacobian check. Independent evaluation on both sides.",
           passed=rel < 0.01)
    record("1", "Hs from the wavenumber integral", 4.0 * np.sqrt(m0_k),
           4.0 * np.sqrt(m0_f), 0.01, unit="m", passed=rel < 0.01)
    assert rel < 0.01


def test_jacobian_holds_in_finite_depth(record, scene):
    """Variance conservation also holds with `tanh(kd)` active."""
    u10, fetch, gamma, theta_wind = scene
    depth = 3.0

    m0_k = moments.integrate_sk_polar(u10, fetch, theta_wind, depth=depth, gamma=gamma, order=0)
    m0_r = moments.mss_between(moments.K_MIN_DEFAULT, constants.K_CAPILLARY,
                               u10, fetch, depth=depth, gamma=gamma, order=0)
    rel = abs(m0_k - m0_r) / m0_r

    record("1", f"m0, finite depth d = {depth} m (polar vs radial)", m0_k, m0_r, 0.01,
           unit="m^2", passed=rel < 0.01)
    assert rel < 0.01


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------


def test_dispersion_round_trip(record):
    """`dispersion_k(dispersion_omega(k))` returns k, deep and finite depth."""
    k = np.geomspace(1e-2, 400.0, 500)
    worst_deep = float(np.max(np.abs(spectrum.dispersion_k(spectrum.dispersion_omega(k)) - k) / k))

    worst_shallow = 0.0
    for depth in (0.5, 3.0, 20.0):
        back = spectrum.dispersion_k(spectrum.dispersion_omega(k, depth), depth)
        worst_shallow = max(worst_shallow, float(np.max(np.abs(back - k) / k)))

    record("1", "dispersion round trip, deep water", worst_deep, 0.0, 1e-12,
           passed=worst_deep < 1e-12)
    record("1", "dispersion round trip, d = 0.5 / 3 / 20 m", worst_shallow, 0.0, 1e-10,
           note="Newton iteration seeded with Eckart's approximation.",
           passed=worst_shallow < 1e-10)
    assert worst_deep < 1e-12
    assert worst_shallow < 1e-10


def test_group_velocity_limits(record):
    """Cg -> c/2 in deep water and -> c in shallow."""
    k = 1.0
    cg_deep = float(spectrum.group_velocity(np.array([k]))[0])
    c_deep = float(spectrum.dispersion_omega(np.array([k]))[0]) / k
    ratio_deep = cg_deep / c_deep

    k_sh, d_sh = 1e-3, 0.5
    cg_sh = float(spectrum.group_velocity(np.array([k_sh]), d_sh)[0])
    c_sh = float(spectrum.dispersion_omega(np.array([k_sh]), d_sh)[0]) / k_sh
    ratio_sh = cg_sh / c_sh

    record("1", "Cg/c, deep water limit", ratio_deep, 0.5, 1e-9, passed=abs(ratio_deep - 0.5) < 1e-9)
    record("1", "Cg/c, shallow limit (kd = 5e-4)", ratio_sh, 1.0, 1e-6,
           passed=abs(ratio_sh - 1.0) < 1e-6)
    assert abs(ratio_deep - 0.5) < 1e-9
    assert abs(ratio_sh - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------


def test_mss_above_zero_is_the_total(record, scene):
    """`mss_above(0)` equals the full radial integral to the capillary cutoff."""
    u10, fetch, gamma, _ = scene
    total = moments.mss_above(0.0, u10, fetch, gamma=gamma)
    direct = moments.mss_between(moments.K_MIN_DEFAULT, constants.K_CAPILLARY,
                                 u10, fetch, gamma=gamma, order=2)
    rel = abs(total - direct) / direct

    record("1", "mss_above(0) (total mean square slope)", total, direct, 1e-9,
           note=f"Integrated to k = {constants.K_CAPILLARY} rad/m "
                f"(lambda = {2 * np.pi / constants.K_CAPILLARY * 100:.1f} cm). "
                f"The Phillips tail integrates logarithmically, so this number "
                f"depends on where that cut is placed -- it is a modelling choice.",
           passed=rel < 1e-9)
    record("1", "RMS slope", float(np.degrees(np.arctan(np.sqrt(total)))), unit="deg")
    assert rel < 1e-9


def test_mss_above_decreases_monotonically(record, scene):
    """`mss_above(k)` is strictly decreasing in k."""
    u10, fetch, gamma, _ = scene
    k_cuts = np.geomspace(1e-3, 0.999 * constants.K_CAPILLARY, 80)
    vals = np.array([moments.mss_above(k, u10, fetch, gamma=gamma) for k in k_cuts])
    diffs = np.diff(vals)
    strictly_decreasing = bool(np.all(diffs < 0.0))

    record("1", "mss_above(k) strictly decreasing", str(strictly_decreasing), "True",
           note=f"Over 80 log-spaced cuts on [1e-3, {constants.K_CAPILLARY}] rad/m. "
                f"Largest (i.e. least negative) step: {diffs.max():.3e}.",
           passed=strictly_decreasing)
    assert strictly_decreasing
    assert moments.mss_above(constants.K_CAPILLARY, u10, fetch, gamma=gamma) == 0.0


def test_mss_anisotropic_splits_the_total(record, scene):
    """`mss_up + mss_cross = mss_above(k_cut)`, since cos^2 + sin^2 = 1."""
    u10, fetch, gamma, theta_wind = scene
    up, cross = moments.mss_anisotropic(u10, fetch, theta_wind, gamma=gamma, n_theta=1024)
    total = moments.mss_above(0.0, u10, fetch, gamma=gamma)
    rel = abs(up + cross - total) / total

    record("1", "mss_up", up, passed=True)
    record("1", "mss_cross", cross, passed=True)
    record("1", "mss_up + mss_cross vs mss_above(0)", up + cross, total, 1e-3, passed=rel < 1e-3)
    record("1", "upwind / crosswind anisotropy ratio", up / cross,
           note="Cox & Munk measured ~1.3 at open-ocean fetch.")
    assert rel < 1e-3
    assert up > cross


def test_mss_within_factor_three_of_cox_munk(record, scene):
    """Sanity bound against the Cox & Munk (1954) empirical fit.

    A factor of 10 would mean a normalisation bug. A factor of 2 means the
    ocean and the lake are different bodies of water, which they are: Cox-Munk
    is an open-ocean, long-fetch fit and this is a 1 km freshwater fetch.
    """
    _, _, gamma, _ = scene

    worst = 0.0
    for u10 in (3.0, 5.0, 10.0, 15.0):
        for fetch, label in ((1e3, "lake"), (5e5, "ocean")):
            mss = moments.mss_above(0.0, u10, fetch, gamma=gamma)
            cm = moments.cox_munk_from_u10(u10)
            ratio = mss / cm
            worst = max(worst, max(ratio, 1.0 / ratio))
            if u10 == 5.0:
                record("1", f"mss / Cox-Munk, U10 = {u10} m/s, {label} fetch", ratio, 1.0, 3.0,
                       note=f"mss = {mss:.5f}, Cox-Munk = {cm:.5f} at U12 = "
                            f"{constants.u10_to_u12(u10):.2f} m/s.",
                       passed=1 / 3 < ratio < 3)

    record("1", "worst mss/Cox-Munk factor over U10 3-15 m/s, both fetches", worst, 1.0, 3.0,
           passed=worst < 3.0)
    assert worst < 3.0


def test_zero_crossing_period_uses_the_frequency_moment(record, scene):
    """Tz = 2 pi sqrt(m0 / m2_omega), and m2_omega is not the slope moment.

    Guards the one confusion the package is built to prevent: in deep water
    `omega^2 = g k`, so mean square slope is the *fourth* frequency moment. Using
    the wavenumber m2 here gives a Tz wrong by orders of magnitude.
    """
    u10, fetch, gamma, _ = scene
    tz = moments.zero_crossing_period(u10, fetch, gamma)
    m0 = moments.moment_omega(0, u10, fetch, gamma)
    m2_omega = moments.moment_omega(2, u10, fetch, gamma)
    expected = 2.0 * np.pi * np.sqrt(m0 / m2_omega)

    _, f_p, _ = spectrum.jonswap_params(u10, fetch)
    mss = moments.mss_above(0.0, u10, fetch, gamma=gamma)

    record("1", "Tz (full band)", tz, expected, 1e-9, unit="s", passed=abs(tz - expected) < 1e-9)
    record("1", "Tz / Tp", tz * f_p,
           note="Physically 0.7-0.8 for a JONSWAP sea; a value near 1 or near 0.1 "
                "means the wrong moment is in the denominator.")
    record("1", "m2 in angular frequency (sets Tz)", m2_omega, unit="rad^2/s^2",
           note=f"Distinct from the wavenumber second moment (mean square slope, "
                f"{mss:.5f}, dimensionless), which sets BSDF roughness. The two are "
                f"not interchangeable and carry different units; in deep water "
                f"slope variance is the *fourth* frequency moment.")

    assert abs(tz - expected) < 1e-9
    assert 0.6 < tz * f_p < 0.9


def test_moment_omega_band_limiting(scene):
    """Band-limiting reduces the moment, and a null band returns 0."""
    u10, fetch, gamma, _ = scene
    full = moments.moment_omega(2, u10, fetch, gamma)
    partial = moments.moment_omega(2, u10, fetch, gamma, f_hi=2.0)
    assert 0.0 < partial < full
    assert moments.moment_omega(0, u10, fetch, gamma, f_lo=5.0, f_hi=1.0) == 0.0
