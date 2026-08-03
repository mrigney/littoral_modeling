"""PHASE 1 -- spectral moments.

The bridge between the abstract spectrum and things you can measure.  Mean
square slope -- the quantity that sets microfacet roughness in the BSDF -- is a
moment of the wave spectrum.  It is not a free parameter and it is not an
artistic choice.

Two families of moment live here and they are NOT interchangeable:

Wavenumber moments (``m0``, ``m2``, ``mss_above``)
    ``int k^n S(k) d2k``.  ``m2`` is total mean square slope.  These drive BSDF
    roughness and the LOD invariant.

Frequency moments (``moment_omega``)
    ``int omega^n S(omega) domega``.  ``moment_omega(2)`` drives the
    zero-crossing period ``Tz = 2 pi sqrt(m0/m2_omega)``.

In deep water ``omega^2 = g k``, so mean square slope is the *fourth* frequency
moment.  The cookbook writes both as "m2"; conflating them gives a Tz that is
wrong by orders of magnitude, so nothing here abbreviates to a bare ``m2``
except the one function the cookbook names that way.

The analytic-integral / FFT-grid split
--------------------------------------
Two ways of getting a moment appear below, and the difference is deliberate:

* ``*_grid`` functions sum over a discrete FFT grid.  That grid gives you
  geometry, and it truncates at its own Nyquist.
* ``mss_above`` and friends integrate the analytic spectrum on a **log-spaced**
  radial grid out to the capillary cutoff.  Because the Phillips tail
  integrates logarithmically (``k^2 S(k) k dk ~ dk/k``), a linear grid
  systematically under-reports the slope variance -- every decade contributes
  equally, and the linear grid resolves the last decade and nothing else.

That difference is the whole point: the FFT grid supplies the geometry, the
analytic integral supplies the sub-mesh statistics the geometry is missing, and
``mss_resolved(dx) + mss_above(pi/dx) = mss_total`` is the invariant that keeps
them consistent across LOD (cookbook section 6.4).

References
----------
Cox, C. & Munk, W. (1954).  "Measurement of the roughness of the sea surface
    from photographs of the sun's glitter."  J. Opt. Soc. Am. 44(11), 838-850.
"""

from __future__ import annotations

import numpy as np

from .constants import G, JONSWAP_GAMMA, K_CAPILLARY, u10_to_u12
from .spectrum import (
    dispersion_omega,
    group_velocity,
    jonswap_params,
    jonswap_sf,
    jonswap_sk,
    spreading,
)

__all__ = [
    "m0",
    "m2",
    "m0_grid",
    "m2_grid",
    "mss_above",
    "mss_between",
    "mss_anisotropic",
    "moment_omega",
    "zero_crossing_period",
    "integrate_sk_polar",
    "cox_munk_mss",
    "cox_munk_anisotropic",
    "K_MIN_DEFAULT",
]

K_MIN_DEFAULT = 1e-3
"""Lower limit of the radial wavenumber integrals [rad/m].

lambda = 6 km, far longer than anything a 1 km fetch can generate, and the
JONSWAP exponential has underflowed to zero long before this.  It exists only
so the log grid has a finite lower bound.
"""

_N_K_DEFAULT = 4000
_N_THETA_DEFAULT = 256


# ---------------------------------------------------------------------------
# Grid moments -- for a spectrum already sampled on an FFT grid
# ---------------------------------------------------------------------------


def _axis_spacing(a: np.ndarray, axis: int) -> float:
    """Spacing of a (possibly ``fftfreq``-ordered) wavenumber axis."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        return float(abs(a[1] - a[0]))
    if a.ndim == 2:
        return float(abs(a[0, 1] - a[0, 0]) if axis == 1 else abs(a[1, 0] - a[0, 0]))
    raise ValueError("wavenumber axes must be 1-D or 2-D")


def m0_grid(s_k: np.ndarray, dkx: float, dky: float) -> float:
    """Height variance from a gridded spectrum: ``sum S * dkx * dky`` [m^2]."""
    return float(np.sum(s_k) * dkx * dky)


def m2_grid(s_k: np.ndarray, kx: np.ndarray, ky: np.ndarray,
            dkx: float, dky: float) -> float:
    """Mean square slope from a gridded spectrum: ``sum k^2 S dkx dky`` [-]."""
    k2 = np.asarray(kx) ** 2 + np.asarray(ky) ** 2
    return float(np.sum(k2 * s_k) * dkx * dky)


def m0(s_k: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> float:
    """Height variance [m^2] from a gridded spectrum (cookbook signature).

    ``kx``/``ky`` may be 1-D axes or 2-D meshgrids; spacing is inferred.
    ``Hs = 4*sqrt(m0)``.
    """
    return m0_grid(s_k, _axis_spacing(kx, 1), _axis_spacing(ky, 0))


def m2(s_k: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> float:
    """Total mean square slope [-] from a gridded spectrum (cookbook signature).

    This is the **wavenumber** second moment, i.e. slope variance -- not the
    frequency second moment that appears in ``Tz``.  See module docstring.

    Note that on an FFT grid this is truncated at the grid Nyquist, so it is the
    *resolved* slope variance, not the total.  For the total, use
    ``mss_above(0.0, ...)``, which integrates the analytic spectrum out to the
    capillary cutoff.
    """
    kx2 = np.asarray(kx, dtype=np.float64)
    ky2 = np.asarray(ky, dtype=np.float64)
    if kx2.ndim == 1:
        kx2, ky2 = np.meshgrid(kx2, ky2, indexing="xy")
    return m2_grid(s_k, kx2, ky2, _axis_spacing(kx, 1), _axis_spacing(ky, 0))


# ---------------------------------------------------------------------------
# Analytic radial integrals -- the LOD workhorses
# ---------------------------------------------------------------------------


def _log_radial_grid(k_lo: float, k_hi: float, n_k: int):
    """Geometric bin edges and midpoints on ``[k_lo, k_hi]``.

    Log spacing is not an optimisation, it is a correctness requirement: the
    integrand of the slope moment falls as ``1/k``, so equal contributions come
    from equal *ratios* of k, and a linear grid puts essentially all of its
    samples in the last octave.
    """
    if not (0 < k_lo < k_hi):
        raise ValueError(f"need 0 < k_lo < k_hi, got {k_lo}, {k_hi}")
    edges = np.geomspace(k_lo, k_hi, n_k + 1)
    mid = np.sqrt(edges[:-1] * edges[1:])
    dk = np.diff(edges)
    return mid, dk


def mss_between(
    k_lo: float,
    k_hi: float,
    u10: float,
    fetch: float,
    depth: float | None = None,
    gamma: float = JONSWAP_GAMMA,
    order: int = 2,
    n_k: int = _N_K_DEFAULT,
    g: float = G,
) -> float:
    """``int_{k_lo}^{k_hi} k^order S(k) d2k`` -- the omnidirectional radial integral.

    ``order=0`` gives the band-limited height variance, ``order=2`` the
    band-limited mean square slope.

    The directional integral is done analytically.  Because
    ``S(k,theta) = S(f) D(f,theta) Cg / (2 pi k)`` and ``int D dtheta = 1`` by
    construction, the azimuthal integral of ``d2k = k dk dtheta`` collapses::

        int int k^n S(k,theta) k dk dtheta  =  int k^n S(f(k)) Cg(k)/(2pi) dk

    This is exact, not an approximation -- and it is also why spreading cannot
    affect ``m0`` or total ``mss``, only their directional split.
    """
    k, dk = _log_radial_grid(max(k_lo, K_MIN_DEFAULT), k_hi, n_k)
    omega = dispersion_omega(k, depth, g)
    f = omega / (2.0 * np.pi)
    s_f = jonswap_sf(f, u10, fetch, gamma, g)
    cg = group_velocity(k, depth, g)
    return float(np.sum(k**order * s_f * cg / (2.0 * np.pi) * dk))


def mss_above(
    k_cut: float,
    u10: float,
    fetch: float,
    depth: float | None = None,
    gamma: float = JONSWAP_GAMMA,
    k_max: float = K_CAPILLARY,
    n_k: int = _N_K_DEFAULT,
    g: float = G,
) -> float:
    """Slope variance carried by wavelengths shorter than ``2 pi / k_cut``.

    ``int_{|k| > k_cut} k^2 S(k) d2k``, integrated to the capillary cutoff.

    THIS IS THE FUNCTION EVERYTHING ELSE CALLS.  Per-vertex in Phase 6 it
    supplies the sub-mesh slope variance that becomes the BSDF's Beckmann
    ``alpha = sqrt(mss)``, and it is what makes the LOD invariant enforceable::

        mss_resolved(lod_dx) + mss_above(pi/lod_dx) = mss_total

    ``mss_above(0)`` is the total, and the result decreases monotonically in
    ``k_cut`` -- both asserted at Gate 1.

    ``k_max`` is a *modelling choice*, not a physical constant: the Phillips
    tail integrates logarithmically, so the answer scales with
    ``log(k_max/k_p)``.  It is stated once in ``constants.K_CAPILLARY`` and
    reported in the validation report so the number is never silently rescaled.
    """
    if k_cut >= k_max:
        return 0.0
    return mss_between(max(k_cut, K_MIN_DEFAULT), k_max, u10, fetch,
                       depth, gamma, order=2, n_k=n_k, g=g)


def mss_anisotropic(
    u10: float,
    fetch: float,
    theta_wind: float,
    k_cut: float = 0.0,
    depth: float | None = None,
    gamma: float = JONSWAP_GAMMA,
    k_max: float = K_CAPILLARY,
    n_k: int = _N_K_DEFAULT,
    n_theta: int = _N_THETA_DEFAULT,
    spreading_model: str = "cos2s",
    g: float = G,
) -> tuple[float, float]:
    """Upwind / crosswind slope variance split.

    Returns ``(mss_up, mss_cross)`` where::

        mss_up    = int (k . u_hat)^2 S(k) d2k     u_hat along theta_wind
        mss_cross = int (k . v_hat)^2 S(k) d2k     v_hat perpendicular

    By construction ``mss_up + mss_cross = mss_above(k_cut)``, since
    ``cos^2 + sin^2 = 1``; that identity is asserted in the tests.  These feed
    the anisotropic Beckmann parameters in Phase 7,
    ``alpha_u = sqrt(2*mss_up)``, ``alpha_v = sqrt(2*mss_cross)``.

    Away from shore this is spatially constant; Phase 5 makes it vary with the
    refracted local wave direction.  Provided now so the Phase 6 per-vertex
    channel contract does not have to change later.
    """
    k, dk = _log_radial_grid(max(k_cut, K_MIN_DEFAULT), k_max, n_k)
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    dtheta = 2.0 * np.pi / n_theta

    _, f_p, _ = jonswap_params(u10, fetch, g)
    omega = dispersion_omega(k, depth, g)
    f = omega / (2.0 * np.pi)
    s_f = jonswap_sf(f, u10, fetch, gamma, g)
    cg = group_velocity(k, depth, g)

    # radial part of k^2 S(k,theta) * k dk, without the directional factor
    radial = (k**2 * s_f * cg / (2.0 * np.pi) * dk)[:, None]
    d_ft = spreading(theta[None, :], f[:, None], f_p, theta_wind, spreading_model)

    rel = theta[None, :] - theta_wind
    up = float(np.sum(radial * d_ft * np.cos(rel) ** 2) * dtheta)
    cross = float(np.sum(radial * d_ft * np.sin(rel) ** 2) * dtheta)
    return up, cross


# ---------------------------------------------------------------------------
# Frequency moments -- the zero-crossing period
# ---------------------------------------------------------------------------


def moment_omega(
    order: int,
    u10: float,
    fetch: float,
    gamma: float = JONSWAP_GAMMA,
    f_lo: float | None = None,
    f_hi: float | None = None,
    n_f: int = 8000,
    g: float = G,
) -> float:
    """``int omega^order S(omega) domega`` = ``int (2 pi f)^order S(f) df``.

    ``order=0`` gives ``m0`` (height variance) computed purely in frequency --
    the independent value the Gate 1 Jacobian check compares the wavenumber
    integral against.  ``order=2`` gives the ``m2`` that appears in
    ``Tz = 2 pi sqrt(m0/m2)``.

    ``f_lo``/``f_hi`` band-limit the integral.  They exist so a realised FFT
    surface can be compared against theory *over the band that surface actually
    resolves*, which is the only fair comparison -- see
    :func:`zero_crossing_period`.

    Note ``order=4`` diverges logarithmically (it is mean square slope in
    disguise), so a bare ``moment_omega(4)`` is meaningless without a band
    limit.  That divergence is physical, not a bug.
    """
    _, f_p, _ = jonswap_params(u10, fetch, g)
    lo = f_p / 8.0 if f_lo is None else max(f_lo, 1e-6)
    hi = dispersion_omega(K_CAPILLARY) / (2.0 * np.pi) if f_hi is None else f_hi
    if lo >= hi:
        return 0.0
    edges = np.geomspace(lo, hi, n_f + 1)
    f = np.sqrt(edges[:-1] * edges[1:])
    df = np.diff(edges)
    s_f = jonswap_sf(f, u10, fetch, gamma, g)
    return float(np.sum((2.0 * np.pi * f) ** order * s_f * df))


def zero_crossing_period(
    u10: float,
    fetch: float,
    gamma: float = JONSWAP_GAMMA,
    f_lo: float | None = None,
    f_hi: float | None = None,
    g: float = G,
) -> float:
    """Mean zero-crossing period ``Tz = 2 pi sqrt(m0 / m2_omega)`` [s].

    Band-limit with ``f_lo``/``f_hi`` to match a realised surface's resolved
    band.  This matters: ``omega^2 S(omega)`` still has a fat tail, so a surface
    truncated at its grid Nyquist genuinely has a longer Tz than the untruncated
    spectrum, and comparing against the untruncated value would look like a bug
    when it is just missing high-frequency content.
    """
    mm0 = moment_omega(0, u10, fetch, gamma, f_lo, f_hi, g=g)
    mm2 = moment_omega(2, u10, fetch, gamma, f_lo, f_hi, g=g)
    if mm2 <= 0.0:
        return float("nan")
    return float(2.0 * np.pi * np.sqrt(mm0 / mm2))


# ---------------------------------------------------------------------------
# Direct 2-D polar integration -- the Gate 1 Jacobian oracle
# ---------------------------------------------------------------------------


def integrate_sk_polar(
    u10: float,
    fetch: float,
    theta_wind: float,
    depth: float | None = None,
    gamma: float = JONSWAP_GAMMA,
    order: int = 0,
    k_lo: float = K_MIN_DEFAULT,
    k_hi: float = K_CAPILLARY,
    n_k: int = 3000,
    n_theta: int = 180,
    spreading_model: str = "cos2s",
    g: float = G,
) -> float:
    """``int int k^order S(kx,ky) k dk dtheta``, by brute force on a polar grid.

    Deliberately calls :func:`~pywave.spectrum.jonswap_sk` on Cartesian
    ``(kx, ky)`` rather than reusing the collapsed radial form in
    :func:`mss_between`.  That makes it an *independent* evaluation of the
    ``Cg/(2 pi k)`` Jacobian and the spreading normalisation, which is exactly
    what the Gate 1 check needs -- an oracle that shares the analytic shortcut
    would agree with the shortcut no matter how wrong both were.
    """
    k, dk = _log_radial_grid(k_lo, k_hi, n_k)
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    dtheta = 2.0 * np.pi / n_theta

    kx = k[:, None] * np.cos(theta)[None, :]
    ky = k[:, None] * np.sin(theta)[None, :]
    s_k = jonswap_sk(kx, ky, u10, fetch, theta_wind, depth, gamma,
                     spreading_model, g)

    weight = (k**order * k * dk)[:, None]
    return float(np.sum(s_k * weight) * dtheta)


# ---------------------------------------------------------------------------
# Cox-Munk cross-check
# ---------------------------------------------------------------------------


def cox_munk_mss(u12: float) -> float:
    """Empirical mean square slope, Cox & Munk (1954), clean surface.

    ``u12`` is wind speed at **12 m**, their reference height; use
    ``constants.u10_to_u12`` to convert.

    Caveat that matters for a lake: this is an open-ocean, long-fetch fit.  A
    small freshwater body at short fetch has a different high-frequency balance
    and this relation will over- or under-predict.  Treat it as a sanity bound,
    not ground truth -- Gate 1 only asks for agreement within a factor of ~3,
    because a factor of 10 means a normalisation bug while a factor of 2 means
    the ocean and the lake are different, which they are.
    """
    return 0.003 + 0.00512 * u12


def cox_munk_anisotropic(u12: float) -> tuple[float, float]:
    """Cox & Munk (1954) upwind / crosswind slope variance.  Returns ``(up, cross)``."""
    up = 0.00316 * u12
    cross = 0.003 + 0.00192 * u12
    return up, cross


def cox_munk_from_u10(u10: float) -> float:
    """Convenience: Cox-Munk mss for a config wind speed given at 10 m."""
    return cox_munk_mss(u10_to_u12(u10))
