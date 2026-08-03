"""Physical constants and the project-wide conventions.

CONVENTIONS -- cookbook section 0.3.  Violating any of these silently is the
most common source of multi-day debugging.  They are reproduced verbatim here
because this module is imported by everything else, so there is no way to write
code in this package without the conventions being one jump away.

===============  ==============================================================
Item             Convention
===============  ==============================================================
Coordinates      Right-handed, **Z up**.  X east, Y north.  Houdini is Y-up --
                 conversion happens **once**, at export from Houdini, nowhere
                 else.
Units            Metres, seconds, radians internally.  Degrees only in config
                 files and only for angles a human types.
Angular freq.    ``omega`` (rad/s) everywhere in code.  ``f`` (Hz) only inside
                 ``spectrum.py`` where JONSWAP is defined in Hz.  Never mix in
                 one expression.
Wavenumber       ``k`` in rad/m (``k = 2*pi/lambda``).  Not cycles/m.
Direction        ``theta`` = direction waves travel **toward**, measured CCW
                 from +X.  Oceanographers often use "coming from" -- we don't.
                 State it in every docstring.
Water plane      ``z = z_w``, a scene constant.  Wave displacement is relative
                 to it.
Depth sign       ``d > 0`` in water, ``d < 0`` on land.  Depth = ``z_w - z_terrain``.
SDF sign         ``s < 0`` in water, ``s > 0`` on land.  **Opposite of depth** --
                 deliberate, so ``s`` reads as "distance inland".  Assert the
                 sign relationship in tests.
FFT convention   numpy default.  ``ifft2`` includes the ``1/N^2``.  Always
                 multiply by ``N^2`` when going from spectral amplitudes to a
                 spatial field.
Time origin      ``t = 0`` is scenario start.  Surface is a pure function of
                 ``t``.  No accumulated state, ever.
Random seed      One integer in config.  Used only in
                 ``spectrum.initial_amplitudes()``.  Nowhere else.
===============  ==============================================================

A note on the two different ``m2`` quantities
---------------------------------------------
The cookbook uses the symbol ``m2`` for two different things, and they are NOT
the same number.  This package keeps them under distinct names:

* ``moments.m2(...)``          -- second moment in *wavenumber*,
  ``int k^2 S(k) d2k`` = total mean square slope.  Sets BSDF roughness.
* ``moments.moment_omega(2)``  -- second moment in *angular frequency*,
  ``int omega^2 S(omega) domega``.  Sets the zero-crossing period
  ``Tz = 2*pi*sqrt(m0/m2_omega)``.

In deep water ``omega^2 = g*k``, so mean square slope is the *fourth* frequency
moment, not the second.  Conflating them produces a Tz that is wrong by orders
of magnitude, so the names never abbreviate to a bare ``m2`` outside
``moments.py``.
"""

from __future__ import annotations

# -- Mechanics ---------------------------------------------------------------

G = 9.81
"""Gravitational acceleration [m/s^2].  Fixed at 9.81 to match the cookbook's
worked examples; do not swap in 9.80665 without re-baselining the gates."""

# -- Spectrum defaults -------------------------------------------------------

JONSWAP_GAMMA = 3.3
"""Default JONSWAP peak enhancement.  gamma = 1 recovers Pierson-Moskowitz."""

JONSWAP_SIGMA_LOW = 0.07
"""Peak width parameter for f <= f_p."""

JONSWAP_SIGMA_HIGH = 0.09
"""Peak width parameter for f > f_p."""

K_CAPILLARY = 400.0
"""Upper wavenumber limit of the gravity-wave slope integral [rad/m].

lambda = 2*pi/400 ~ 1.6 cm, i.e. the gravity-capillary transition.  The
Phillips tail integrates *logarithmically* (see the primer), so mean square
slope depends on where this cut is placed -- it is a modelling choice, it is
stated here once, and it is reported in the validation report.  Physical
surface roughness bottoms out around a millimetre, which is still 100x the
LWIR wavelength, so every sensor band stays in the geometric-optics regime.
"""

# -- Reference wind heights --------------------------------------------------

COX_MUNK_REF_HEIGHT = 12.0
"""Cox & Munk (1954) reported wind speed at 12 m.  Our config is U10."""

WIND_PROFILE_EXPONENT = 1.0 / 7.0
"""Power-law exponent for the neutral wind profile, U(z) ~ z^(1/7).

Used only to convert U10 -> U12 for the Cox-Munk cross-check.  The correction
is ~2.6%, well inside the factor-of-3 tolerance of that check, but doing it
explicitly keeps the comparison honest.
"""


def u10_to_u12(u10: float) -> float:
    """Convert 10 m reference wind speed to Cox & Munk's 12 m reference."""
    return u10 * (COX_MUNK_REF_HEIGHT / 10.0) ** WIND_PROFILE_EXPONENT


# -- Water optical constants -------------------------------------------------
#
# Hale & Querry (1973), "Optical constants of water in the 200-nm to 200-um
# wavelength region", Appl. Opt. 12(3), 555-563.  One citable source covering
# the whole EO/MWIR/LWIR range.  Phase 7/8 will load the full spectral table;
# these are the band-representative anchors quoted in the primer, kept here so
# Phases 1-3 can reason about band behaviour without a table dependency.

WATER_IOR = {
    0.55e-6: (1.333, 1.0e-9),   # EO       -- effectively transparent, bottom visible
    4.0e-6: (1.350, 4.6e-3),    # MWIR     -- weakly absorbing
    10.0e-6: (1.220, 5.1e-2),   # LWIR     -- opaque, penetration depth ~16 um
}
"""Complex refractive index ``n + i*k`` keyed by wavelength [m]."""

__all__ = [
    "G",
    "JONSWAP_GAMMA",
    "JONSWAP_SIGMA_LOW",
    "JONSWAP_SIGMA_HIGH",
    "K_CAPILLARY",
    "COX_MUNK_REF_HEIGHT",
    "WIND_PROFILE_EXPONENT",
    "u10_to_u12",
    "WATER_IOR",
]
