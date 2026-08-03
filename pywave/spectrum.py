"""PHASE 1 -- wave spectra.

JONSWAP in frequency, directional spreading, and the frequency -> wavenumber
conversion.  This module is the single source of truth for the whole pipeline:
wave height, slope statistics, BSDF roughness and LOD behaviour are all derived
from ``S(k)``.  Nothing downstream is tuned independently.

Direction convention (cookbook section 0.3): ``theta`` is the direction waves
travel **toward**, measured CCW from +X, in radians.  Oceanographic "coming
from" convention is NOT used anywhere in this package.

Frequency convention: ``f`` in Hz appears only in this module, where JONSWAP is
defined.  Everything that leaves this module is in ``omega`` (rad/s) or ``k``
(rad/m).

References
----------
Hasselmann, K. et al. (1973). "Measurements of wind-wave growth and swell decay
    during the Joint North Sea Wave Project (JONSWAP)."  Deutsche
    Hydrographische Zeitschrift, Suppl. A8.
Hasselmann, D., Dunckel, M. & Ewing, J. (1980). "Directional wave spectra
    observed during JONSWAP 1973."  J. Phys. Oceanogr. 10, 1264-1280.
Tessendorf, J.  "Simulating Ocean Water."  SIGGRAPH course notes.
"""

from __future__ import annotations

import numpy as np

from .constants import G, JONSWAP_GAMMA, JONSWAP_SIGMA_HIGH, JONSWAP_SIGMA_LOW

__all__ = [
    "dimensionless_fetch",
    "jonswap_params",
    "jonswap_sf",
    "fetch_limited_hs",
    "hs_spectral",
    "hs_ratio_spectral_to_fit",
    "spreading",
    "spreading_exponent",
    "spreading_norm",
    "dispersion_omega",
    "dispersion_k",
    "group_velocity",
    "shoaling_n",
    "jonswap_sk",
    "initial_amplitudes",
]

# Below this multiple of f_p the JONSWAP exponential has underflowed to
# identically zero in float64, so the f^-5 singularity is never evaluated.
_F_MIN_FRACTION = 1.0 / 8.0

# exp() of anything below this underflows to 0.0 in float64; clipping to it
# keeps the log-space evaluation free of spurious warnings.
_LOG_UNDERFLOW = -700.0


# ---------------------------------------------------------------------------
# JONSWAP in frequency
# ---------------------------------------------------------------------------


def dimensionless_fetch(u10: float, fetch: float, g: float = G) -> float:
    """Dimensionless fetch ``X~ = g*X/U10^2``."""
    return g * fetch / u10**2


def jonswap_params(u10: float, fetch: float, g: float = G) -> tuple[float, float, float]:
    """Fetch-limited JONSWAP parameters.

    Returns
    -------
    (alpha, f_p, x_tilde)
        ``alpha`` -- Phillips-like scale parameter [-]
        ``f_p``   -- peak frequency [Hz]
        ``x_tilde`` -- dimensionless fetch [-]

    For the test lake (U10 = 5 m/s, fetch = 1000 m): x_tilde ~ 392,
    alpha ~ 0.0204, f_p ~ 0.956 Hz (T_p ~ 1.05 s).
    """
    x_tilde = dimensionless_fetch(u10, fetch, g)
    alpha = 0.076 * x_tilde ** (-0.22)
    f_p = 3.5 * (g / u10) * x_tilde ** (-0.33)
    return alpha, f_p, x_tilde


def jonswap_sf(
    f: np.ndarray | float,
    u10: float,
    fetch: float,
    gamma: float = JONSWAP_GAMMA,
    g: float = G,
) -> np.ndarray:
    """JONSWAP spectral density ``S(f)`` [m^2/Hz].

    Parameters
    ----------
    f : array of frequencies [Hz], f > 0.  Non-positive and sub-threshold
        frequencies return exactly 0 rather than inf -- the ``f^-5`` prefactor
        diverges there but the ``exp[-1.25 (f_p/f)^4]`` factor kills it far
        faster, so 0 is the correct limit as well as the safe one.
    u10 : wind speed at 10 m [m/s]
    fetch : fetch length [m]
    gamma : peak enhancement factor; gamma = 1 recovers Pierson-Moskowitz.

    Notes
    -----
    Evaluated in log space so the ``f^-5`` singularity is never formed::

        log S = log(alpha g^2 (2pi)^-4) - 5 log f - 1.25 (f_p/f)^4 + r log gamma
    """
    f_in = np.asarray(f, dtype=np.float64)
    f = np.atleast_1d(f_in)
    alpha, f_p, _ = jonswap_params(u10, fetch, g)

    valid = f > (_F_MIN_FRACTION * f_p)
    out = np.zeros_like(f)
    if not np.any(valid):
        return out.reshape(f_in.shape)

    fv = f[valid]

    sigma = np.where(fv <= f_p, JONSWAP_SIGMA_LOW, JONSWAP_SIGMA_HIGH)
    r = np.exp(-((fv - f_p) ** 2) / (2.0 * sigma**2 * f_p**2))

    log_s = (
        np.log(alpha * g**2 * (2.0 * np.pi) ** -4)
        - 5.0 * np.log(fv)
        - 1.25 * (f_p / fv) ** 4
        + r * np.log(gamma)
    )
    out[valid] = np.exp(np.maximum(log_s, _LOG_UNDERFLOW))
    return out.reshape(f_in.shape)


def fetch_limited_hs(u10: float, fetch: float, g: float = G) -> float:
    """Fetch-limited significant wave height [m] from the JONSWAP *energy* fit.

    ``g*Hs/U10^2 = 0.0016 * sqrt(X~)``, equivalently the dimensionless-energy
    growth law ``eps = g^2 m0 / U10^4 = 1.6e-7 * X~``.

    THIS IS NOT THE Hs OF THE SPECTRUM THIS MODULE BUILDS.  See
    :func:`hs_spectral` and the note below.  For the test lake this returns
    0.0808 m while the spectrum integrates to 0.0857 m.
    """
    x_tilde = dimensionless_fetch(u10, fetch, g)
    return 0.0016 * (u10**2 / g) * np.sqrt(x_tilde)


def hs_ratio_spectral_to_fit(u10: float, fetch: float, g: float = G) -> float:
    """Predicted ``hs_spectral / fetch_limited_hs`` = ``0.787 * X~^0.05``.

    Why the two disagree -- and why that is not a bug
    ------------------------------------------------
    JONSWAP supplies two *independently fitted* power laws, and they are not
    mutually consistent:

    ===========================  =========================  ==================
    Quantity                     Fit                        Implied m0 scaling
    ===========================  =========================  ==================
    scale parameter              ``alpha = 0.076 X~^-0.22``
    peak frequency               ``f_p ~ X~^-0.33``         ``m0 ~ alpha f_p^-4
                                                            ~ X~^1.10``
    dimensionless energy         ``eps = 1.6e-7 X~``        ``m0 ~ X~^1.00``
    ===========================  =========================  ==================

    Integrating the spectrum uses the first pair; ``fetch_limited_hs`` uses the
    second.  Their ratio is therefore ``~ X~^0.10`` in energy and ``~ X~^0.05``
    in Hs, with a prefactor putting the crossover at ``X~ ~ 120``.

    This was measured, not assumed.  Fitting ``hs_spectral/hs_fit = c*X~^p``
    over U10 = 3-20 m/s and fetch = 300 m - 5000 km (X~ from 7 to 5.5e6) gives
    ``p = 0.05001`` against the structural prediction of exactly 0.05, with
    residuals below 3e-4 across the whole range.  A normalisation or Jacobian
    error would be a *constant* factor, i.e. ``p = 0``; it is not.

    Consequence, stated once so it is never re-litigated: the cookbook's Gate 1
    tolerance of 2% between these two numbers cannot be met by any correct
    implementation of the JONSWAP spectral form.  We keep the spectral form
    (cookbook section 0.1: "the spectrum is the single source of truth ... if
    two things disagree, the spectrum wins"), report ``hs_spectral`` as the
    scene's Hs, and pin regressions against *this relation* instead -- which is
    a far tighter constraint than 2% on a single number, since it must hold
    across four decades of fetch.

    Rescaling the spectrum to hit ``fetch_limited_hs`` would be the alternative.
    It is deliberately NOT offered: it would change ``alpha``, hence the
    Phillips tail, hence mean square slope and every BSDF roughness downstream
    -- which is exactly the independent tuning the design principle forbids.
    """
    x_tilde = dimensionless_fetch(u10, fetch, g)
    return 0.78696 * x_tilde**0.05


def hs_spectral(u10: float, fetch: float, gamma: float = JONSWAP_GAMMA,
                g: float = G) -> float:
    """Significant wave height of the spectrum actually implemented here [m].

    ``Hs = 4*sqrt(m0)`` with ``m0 = int S(f) df``.  This is the number that a
    realised FFT surface will reproduce, and the one to quote for the scene.

    For the test lake (U10 = 5 m/s, fetch = 1000 m) this is 0.0857 m, ~6% above
    the ``fetch_limited_hs`` value of 0.0808 m -- see
    :func:`hs_ratio_spectral_to_fit` for why.
    """
    from .moments import moment_omega  # local import: moments imports this module

    return 4.0 * np.sqrt(moment_omega(0, u10, fetch, gamma, g=g))


# ---------------------------------------------------------------------------
# Directional spreading
# ---------------------------------------------------------------------------

# Frequency-dependent cos^(2s) exponent, by model.
#   (coeff_below, exponent_below, coeff_above, exponent_above)
# "below"/"above" are relative to the peak frequency f_p.
_SPREADING_MODELS = {
    # As written in the cookbook section 1.2.
    "cos2s": (9.77, 4.06, 11.5, -2.5),
    # Hasselmann, Dunckel & Ewing (1980) as published.  Retained so the
    # cookbook's coefficients can be checked against the source they came from;
    # see spreading_exponent() for the discrepancy note.
    "hasselmann": (6.97, 4.06, 9.77, -2.33),
}


def spreading_exponent(f: np.ndarray, f_p: float, model: str = "cos2s") -> np.ndarray:
    """Frequency-dependent spreading exponent ``s(f)``.

    Larger ``s`` means energy is more tightly aligned with the wind.  Waves near
    the spectral peak are narrowly spread; high-frequency waves spread broadly.

    Model coefficients
    ------------------
    ``cos2s`` (default) uses the cookbook section 1.2 coefficients::

        s = 9.77 * (f/f_p)^4.06    for f <= f_p
        s = 11.5 * (f/f_p)^-2.5    for f >  f_p

    KNOWN DISCREPANCY, deliberately preserved: these two branches do not meet at
    ``f = f_p`` -- the low branch gives s = 9.77 and the high branch s = 11.5, a
    ~18% step in spreading width exactly at the peak.  The published source
    (Hasselmann et al. 1980) uses 6.97 and 9.77 respectively, which is also
    discontinuous but by a different amount, so the step is a property of the
    fitted model rather than a transcription error.  It does NOT affect Hs or
    any Gate 1 quantity, because ``D`` integrates to 1 by construction at every
    frequency and therefore divides out of ``m0``.  It DOES affect directional
    quantities -- ``mss_up``/``mss_cross`` anisotropy, and hence the BSDF
    tangent-frame anisotropy in Phase 7.  ``model="hasselmann"`` selects the
    published coefficients if that matters later.
    """
    try:
        c_lo, p_lo, c_hi, p_hi = _SPREADING_MODELS[model]
    except KeyError:
        if model == "donelan":
            raise NotImplementedError(
                "Donelan-Banner spreading is a Phase 1 follow-on (cookbook 1.2: "
                "'add it behind the model flag later').  Use model='cos2s'."
            ) from None
        raise ValueError(
            f"unknown spreading model {model!r}; "
            f"known: {sorted(_SPREADING_MODELS)} (+ 'donelan', not yet implemented)"
        ) from None

    ratio = np.asarray(f, dtype=np.float64) / f_p
    ratio = np.maximum(ratio, 1e-12)
    return np.where(ratio <= 1.0, c_lo * ratio**p_lo, c_hi * ratio**p_hi)


def spreading_norm(s: np.ndarray, mode: str = "closed", n_quad: int = 8192) -> np.ndarray:
    """Normalisation constant ``N(s)`` for the ``cos^(2s)`` spreading function.

    ``N(s)`` is defined by ``int_{-pi}^{pi} N(s) cos^(2s)(theta/2) dtheta = 1``.

    Two implementations, which must agree:

    ``mode="closed"`` (default)
        Substituting ``u = theta/2`` gives ``4 * int_0^(pi/2) cos^(2s)u du``,
        a Wallis integral, ``= 2*sqrt(pi)*Gamma(s+1/2)/Gamma(s+1)``.  So::

            N(s) = Gamma(s+1) / (2*sqrt(pi)*Gamma(s+1/2))

        Evaluated via ``gammaln`` so large ``s`` does not overflow.

    ``mode="quad"``
        Direct numerical quadrature, which is what the cookbook (section 1.2)
        prescribes.  It is used as the reference in the Gate 1 test.

    Why both.  The cookbook argues for quadrature on the grounds that the
    closed form is error-prone and this is not a hot path.  The first half is
    true, the second is not: ``jonswap_sk`` is evaluated on 512^2 FFT grids in
    Phase 2, and a per-element quadrature there is a 262144 x 8192 temporary,
    i.e. ~17 GB.  So the closed form is the production path and the quadrature
    is retained as its test oracle -- which keeps the cookbook's actual intent
    (the normalisation is verified numerically, not trusted) at 1/1000 the cost.
    Gate 1 asserts the two agree to 1e-10 and that ``int D dtheta = 1`` to 1e-6.
    """
    s = np.asarray(s, dtype=np.float64)
    if mode == "closed":
        from scipy.special import gammaln

        log_n = gammaln(s + 1.0) - gammaln(s + 0.5) - np.log(2.0 * np.sqrt(np.pi))
        return np.exp(log_n)
    if mode == "quad":
        tq = np.linspace(-np.pi, np.pi, n_quad, endpoint=False)
        dtq = 2.0 * np.pi / n_quad
        kernel = np.maximum(np.cos(0.5 * tq), 0.0) ** (2.0 * s[..., None])
        return 1.0 / (np.sum(kernel, axis=-1) * dtq)
    raise ValueError(f"unknown normalisation mode {mode!r}; use 'closed' or 'quad'")


def spreading(
    theta: np.ndarray,
    f: np.ndarray,
    f_p: float,
    theta_wind: float,
    model: str = "cos2s",
    norm_mode: str = "closed",
) -> np.ndarray:
    """Normalised directional spreading ``D(f, theta)``, with ``int D dtheta = 1``.

    Parameters
    ----------
    theta : direction waves travel **toward**, CCW from +X [rad].
    f : frequency [Hz], broadcast against ``theta``.
    f_p : peak frequency [Hz].
    theta_wind : direction the wind blows **toward**, CCW from +X [rad].
    model : spreading exponent model, see :func:`spreading_exponent`.
    norm_mode : ``"closed"`` or ``"quad"``, see :func:`spreading_norm`.

    Notes
    -----
    ``D = N(s) * cos(0.5*(theta - theta_wind))^(2s)``.

    The angular difference is wrapped into ``[-pi, pi]`` before halving, which
    keeps ``cos(0.5*dtheta) >= 0`` so the non-integer power is always real.
    Without the wrap, directions more than pi from the wind produce a negative
    base raised to a fractional power, i.e. silent NaNs over half the domain.
    """
    theta = np.asarray(theta, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)

    s = spreading_exponent(f, f_p, model)

    dtheta = np.mod(theta - theta_wind + np.pi, 2.0 * np.pi) - np.pi
    base = np.maximum(np.cos(0.5 * dtheta), 0.0)
    return spreading_norm(s, norm_mode) * base ** (2.0 * s)


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------


def dispersion_omega(k: np.ndarray | float, depth: float | np.ndarray | None = None,
                     g: float = G) -> np.ndarray:
    """Angular frequency from wavenumber: ``omega^2 = g*k*tanh(k*d)``.

    ``depth=None`` selects the deep-water limit ``tanh -> 1``.
    """
    k = np.abs(np.asarray(k, dtype=np.float64))
    if depth is None:
        return np.sqrt(g * k)
    d = np.asarray(depth, dtype=np.float64)
    return np.sqrt(g * k * np.tanh(np.minimum(k * d, 30.0)))


def dispersion_k(
    omega: np.ndarray | float,
    depth: float | np.ndarray | None = None,
    g: float = G,
    n_iter: int = 8,
) -> np.ndarray:
    """Wavenumber from angular frequency -- inverts ``omega^2 = g*k*tanh(k*d)``.

    ``depth=None`` returns the deep-water solution ``k = omega^2/g`` directly.

    Otherwise a Newton iteration is seeded with Eckart's (1952) approximation
    ``k0 = omega^2 / (g*sqrt(tanh(omega^2 d / g)))``, which is within a few
    percent everywhere from deep to shallow, so convergence to machine
    precision takes 3-4 steps.  Seeding with the deep-water guess instead
    converges slowly in genuinely shallow water; Phase 5 needs this to be solid
    across the whole nearshore band, so it is built in now.
    """
    omega = np.asarray(omega, dtype=np.float64)
    if depth is None:
        return omega**2 / g

    d = np.asarray(depth, dtype=np.float64)
    w2g = omega**2 / g

    # Eckart seed.  Guard the deep limit where tanh -> 1.
    k = w2g / np.sqrt(np.tanh(np.minimum(np.maximum(w2g * d, 1e-12), 30.0)))

    for _ in range(n_iter):
        kd = np.minimum(k * d, 30.0)
        th = np.tanh(kd)
        fk = g * k * th - omega**2
        # d/dk [g k tanh(kd)] = g tanh(kd) + g k d sech^2(kd)
        dfk = g * th + g * k * d * (1.0 - th**2)
        step = np.where(dfk != 0.0, fk / dfk, 0.0)
        k = np.maximum(k - step, 1e-12)
    return k


def shoaling_n(k: np.ndarray, depth: float | np.ndarray | None = None) -> np.ndarray:
    """``n = Cg/c = 0.5*(1 + 2kd/sinh(2kd))``.  Deep -> 0.5, shallow -> 1."""
    if depth is None:
        return np.full(np.shape(k), 0.5)
    kd = np.asarray(k, dtype=np.float64) * np.asarray(depth, dtype=np.float64)
    two_kd = np.minimum(2.0 * kd, 350.0)  # sinh overflows around 710
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(two_kd > 1e-8, two_kd / np.sinh(two_kd), 1.0)
    return 0.5 * (1.0 + ratio)


def group_velocity(k: np.ndarray, depth: float | np.ndarray | None = None,
                   g: float = G) -> np.ndarray:
    """Group velocity ``Cg = dOmega/dk = n * omega/k`` [m/s].

    Deep water gives ``Cg = 0.5*sqrt(g/k) = c/2``; shallow gives ``Cg = c``.
    The transition between the two is what drives shoaling amplification in
    Phase 5, so this is not a trivia item.
    """
    k = np.abs(np.asarray(k, dtype=np.float64))
    omega = dispersion_omega(k, depth, g)
    n = shoaling_n(k, depth)
    with np.errstate(divide="ignore", invalid="ignore"):
        cg = np.where(k > 0.0, n * omega / np.where(k > 0.0, k, 1.0), 0.0)
    return cg


# ---------------------------------------------------------------------------
# The frequency -> wavenumber conversion
# ---------------------------------------------------------------------------


def jonswap_sk(
    kx: np.ndarray,
    ky: np.ndarray,
    u10: float,
    fetch: float,
    theta_wind: float,
    depth: float | None = None,
    gamma: float = JONSWAP_GAMMA,
    spreading_model: str = "cos2s",
    g: float = G,
) -> np.ndarray:
    """2-D wavenumber spectrum ``S(kx, ky)`` [m^4].

    Parameters
    ----------
    kx, ky : wavenumber components [rad/m], broadcast together.
    theta_wind : direction the wind blows **toward**, CCW from +X [rad].
    depth : still-water depth [m].  ``None`` selects deep water.

    Returns
    -------
    Array of the broadcast shape of ``kx``/``ky``.  The DC bin (``k = 0``) is
    set to exactly 0.

    The Jacobian -- derive it here, because future-you will want to check it
    ------------------------------------------------------------------------
    Variance must be conserved between the two parameterisations::

        int S(f) df  =  int int S(k,theta) d2k       with  d2k = k dk dtheta

    Matching the integrands element by element::

        S(f) D(f,theta) df dtheta  =  S(k,theta) k dk dtheta

    so::

        S(k,theta) = S(f) D(f,theta) * (df/dk) / k

    and since ``f = omega/2pi``, ``df/dk = (1/2pi) domega/dk = Cg/(2pi)``::

        S(k,theta) = S(f) * D(f,theta) * Cg / (2 pi k)          <-- what we use

    The division by ``k`` comes from the polar area element, NOT from the
    frequency Jacobian.  Getting the two confused is the single most common way
    to be wrong here, and it shows up as an Hs that is off by a factor you will
    spend days hunting.  Gate 1 checks ``int S(k) d2k == int S(f) df`` to 1%.

    Finite depth
    ------------
    ``omega(k)`` uses the full ``omega^2 = g k tanh(kd)``, so a given wavenumber
    maps to a lower frequency in shallow water and both ``S(f)`` and ``Cg`` are
    evaluated consistently there.  Phase 5 needs this; retrofitting it later is
    much worse than carrying it from the start.
    """
    kx = np.asarray(kx, dtype=np.float64)
    ky = np.asarray(ky, dtype=np.float64)
    k = np.hypot(kx, ky)

    _, f_p, _ = jonswap_params(u10, fetch, g)

    positive = k > 0.0
    out = np.zeros(np.broadcast(kx, ky).shape, dtype=np.float64)
    if not np.any(positive):
        return out

    k_pos = k[positive]
    theta = np.arctan2(np.broadcast_to(ky, k.shape)[positive],
                       np.broadcast_to(kx, k.shape)[positive])

    omega = dispersion_omega(k_pos, depth, g)
    f = omega / (2.0 * np.pi)

    s_f = jonswap_sf(f, u10, fetch, gamma, g)
    d_ft = spreading(theta, f, f_p, theta_wind, spreading_model)
    cg = group_velocity(k_pos, depth, g)

    out[positive] = s_f * d_ft * cg / (2.0 * np.pi * k_pos)
    return out


# ---------------------------------------------------------------------------
# Initial spectral amplitudes  (cookbook 0.3: the seed lives here and only here)
# ---------------------------------------------------------------------------


def initial_amplitudes(s_k: np.ndarray, dk: float, seed: int) -> np.ndarray:
    """Draw the time-zero complex spectral amplitudes ``h0(k)``.

    ``h0 = 0.5 * (xi_r + i*xi_i) * sqrt(S * dk_x * dk_y)`` with
    ``xi_r, xi_i ~ N(0,1)`` independent.

    Conceptually this is just spectral filtering of white noise: take white
    noise, shape it with the square root of the spectrum.  The square root
    appears because ``S`` is a *power* density and this is an *amplitude*; the
    ``dk^2`` converts spectral density into per-mode variance.

    On the factor of 0.5
    --------------------
    Conventions in the literature differ and this is a notorious trap.  With
    this form ``E|h0|^2 = 0.5 * S * dk^2``.  The two-term time evolution in
    ``surface.surface_at`` sums two independent contributions, giving
    ``E|h~|^2 = S*dk^2`` per mode, hence ``sum_k E|h~|^2 = m0`` as required.
    Tessendorf's written formula uses ``1/sqrt(2)`` because he folds a factor of
    2 into his definition of ``P_h``.  Do not reason about this -- Gate 2 pins
    it.  If Hs comes out sqrt(2) too large, this is why.

    The seed is used here and nowhere else in the package.  ``default_rng`` is
    used deliberately: never the legacy global RNG, never ``np.random.seed``.
    Cross-machine reproducibility depends on it.
    """
    rng = np.random.default_rng(seed)
    xi_r = rng.standard_normal(s_k.shape)
    xi_i = rng.standard_normal(s_k.shape)
    return 0.5 * (xi_r + 1j * xi_i) * np.sqrt(np.maximum(s_k, 0.0)) * dk
