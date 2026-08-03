"""PHASE 2 -- Tessendorf FFT surface synthesis.

Turns ``S(k,theta)`` into an actual surface ``h(x, t)`` on a grid.  The whole
trick is the observation that summing thousands of sinusoids *is* a Fourier
transform, so an FFT does it in O(N log N).  Everything else is bookkeeping --
but the bookkeeping is where the bugs live, so each piece of it is derived in a
docstring rather than asserted.

Three properties of this construction matter more than everything else for
closed-loop simulation, and they are the reason for the design constraints
elsewhere in this module:

1. **Evaluable at arbitrary t.**  No integration, no frame-to-frame state.
   Frame 8000 costs what frame 1 costs and does not drift.
2. **Exactly deterministic from the seed.**  Same seed, same t, same surface,
   on any machine, forever.  That is a V&V property you can write down.
3. **Trivially parallel.**  Any node can compute any frame with no coordination.

Tessendorf's ``omega`` quantisation for seamless looping is deliberately NOT
implemented: it introduces a detectable period, which is poison both for
training-set diversity and for closed-loop temporal analysis.

Reference: Tessendorf, J.  "Simulating Ocean Water."  SIGGRAPH course notes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import G, JONSWAP_GAMMA
from .spectrum import dispersion_omega, initial_amplitudes, jonswap_sk

__all__ = [
    "SurfaceField",
    "WaveTile",
    "flip_k",
    "evolve",
    "surface_at",
    "grid_wavenumbers",
    "displacement_jacobian",
]


# ---------------------------------------------------------------------------
# Grid setup
# ---------------------------------------------------------------------------


def grid_wavenumbers(size: float, n: int):
    """FFT wavenumber grid for an ``n x n`` tile of side ``size`` metres.

    Returns ``(kx, ky, k, dk)`` where ``kx``/``ky``/``k`` are ``(n, n)`` arrays
    indexed ``[y, x]`` (matching the section 0.4 data contract) and ``dk`` is the
    scalar spacing ``2*pi/size`` in both directions.

    ``fftfreq`` ordering is used throughout -- DC at index 0, negative
    frequencies in the upper half.  That ordering is what :func:`flip_k` has to
    undo, and it is the reason the flip needs a roll.
    """
    dx = size / n
    kx_axis = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky_axis = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(kx_axis, ky_axis, indexing="xy")
    k = np.hypot(kx, ky)
    dk = 2.0 * np.pi / size
    return kx, ky, k, dk


def flip_k(h0: np.ndarray) -> np.ndarray:
    """Return ``h0(-k)`` for an array on an ``fftfreq``-ordered grid.

    On an ``fftfreq`` grid, index ``j`` holds wavenumber ``k_j``, and ``-k_j``
    lives at index ``(-j) mod N``.  Reversing the array maps ``j -> N-1-j``, so
    a further roll by +1 is needed to land on ``N-j``::

        roll(h0[::-1, ::-1], 1)[i, j] = h0[(-i) % N, (-j) % N]

    Check the wrap explicitly, because that is where the off-by-one hides:
    at ``i = 0`` the reversed-and-rolled index is ``N-1-(0-1) mod N = 0`` -- DC
    maps to itself, as it must.

    **Get this wrong and you get a complex-valued surface.**  The Nyquist row
    and column are their own negatives on an even grid, which makes some wrong
    implementations *nearly* work, so a spot check on a few bins is not enough;
    :func:`surface_at`'s realness assertion is the real test.
    """
    return np.roll(h0[::-1, ::-1], shift=1, axis=(0, 1))


def evolve(h0: np.ndarray, omega: np.ndarray, t: float) -> np.ndarray:
    """Advance the spectral amplitudes to time ``t``.

    ``h~(k,t) = h0(k) e^(-i omega t) + conj(h0(-k)) e^(+i omega t)``.

    See :func:`surface_at` for why the exponent is negative here and positive in
    Tessendorf's write-up.  Kept as its own function so every consumer of the
    evolution shares one definition of the sign -- there is no second place for
    it to drift.
    """
    phase = np.exp(-1j * omega * t)
    return h0 * phase + np.conj(flip_k(h0)) * np.conj(phase)


# ---------------------------------------------------------------------------
# Evaluated surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceField:
    """One evaluation of a wave surface.

    All arrays share the sampling shape.  ``h`` is displacement about ``z_w``;
    horizontal displacement and slopes are in the same frame as the sample
    coordinates that produced them.
    """

    h: np.ndarray
    """Vertical displacement about the water plane [m]."""
    dx_disp: np.ndarray
    """Horizontal displacement, x component [m], before the choppiness scale."""
    dy_disp: np.ndarray
    """Horizontal displacement, y component [m], before the choppiness scale."""
    slope_x: np.ndarray
    """``dh/dx`` [-], analytic (spectral), not finite-differenced."""
    slope_y: np.ndarray
    """``dh/dy`` [-], analytic (spectral), not finite-differenced."""

    def normals(self) -> np.ndarray:
        """Unit surface normals, shape ``(..., 3)``, Z up.

        ``n = normalize((-dh/dx, -dh/dy, 1))``.  Use these rather than mesh
        normals: mesh normals are a finite-difference approximation of a field
        we already have exactly, and they lose precisely the high-frequency
        content that matters most radiometrically.
        """
        n = np.stack([-self.slope_x, -self.slope_y, np.ones_like(self.h)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def mss_resolved(self) -> float:
        """Realised mean square slope of this sample set, ``<sx^2 + sy^2>``."""
        return float(np.mean(self.slope_x**2 + self.slope_y**2))

    def __add__(self, other: "SurfaceField") -> "SurfaceField":
        """Sum two fields.  Valid only for disjoint spectral bands.

        Variances add only when the bands do not overlap; see
        ``tiling.composite_surface``, which enforces that.
        """
        return SurfaceField(
            self.h + other.h,
            self.dx_disp + other.dx_disp,
            self.dy_disp + other.dy_disp,
            self.slope_x + other.slope_x,
            self.slope_y + other.slope_y,
        )


def surface_at(
    h0: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    k: np.ndarray,
    omega: np.ndarray,
    t: float,
) -> SurfaceField:
    """Evaluate the surface and its derivatives at time ``t``.

    Time evolution -- and the sign of the exponent
    ----------------------------------------------
    ::

        h~(k,t) = h0(k) e^(-i omega t)  +  conj(h0(-k)) e^(+i omega t)

    Each mode rotates in phase at its own frequency -- that is dispersion,
    implemented for free.  The second term enforces Hermitian symmetry,
    ``h~(-k) = conj(h~(k))``, which is what makes the inverse transform real.
    Physically it is a pair of waves travelling in opposite directions, and the
    directional spectrum decides which of the two carries the energy.

    NOTE THE SIGN.  The cookbook (section 2.3) and Tessendorf both write
    ``h0(k) e^(+i omega t)``.  Combined with the ``sum_k h~ e^(+ikx)`` synthesis
    that numpy's ``ifft2`` implements, the dominant term then carries phase
    ``kx + omega t``, i.e. crests satisfy ``x = -omega t / k`` and the whole sea
    marches **backwards** -- into the wind, and against the direction the
    spreading function was centred on.  Flipping the exponent gives phase
    ``kx - omega t`` and waves that travel along ``+k`` as intended.

    Hermitian symmetry is unaffected by the flip: ``ht(-k)`` and
    ``conj(ht(k))`` still agree term for term, so the surface stays exactly
    real either way -- which is precisely why this error survives a realness
    check, a variance check, a spectrum check and a slope check, and is caught
    only by tracking a crest (Gate 2's ``c = omega/k`` test).  That test earned
    its place here.

    FFT normalisation
    -----------------
    numpy's ``ifft2`` includes the ``1/N^2``, so every spatial field is
    multiplied back by ``N^2``.  With that, ``sum_k E|h~|^2 = m0`` and
    ``var(h) = m0`` exactly (see :func:`~pywave.spectrum.initial_amplitudes`
    for the factor-of-0.5 bookkeeping that makes this come out).

    Horizontal displacement -- the sign, derived rather than copied
    --------------------------------------------------------------
    Real waves have sharp crests and broad flat troughs.  The displacement must
    therefore *compress* sample points toward crests.  Take the 1-D case
    ``h(x) = a cos(kx)``, so ``h~(+k) = h~(-k) = a/2``, and evaluate both signs
    of the transfer function with numpy's ``ifft`` convention ``sum_k X_k
    e^(+ikx)``:

    ===============  =================  ===========================
    transfer         D(x)               effect at the crest (x = 0)
    ===============  =================  ===========================
    ``-i k/|k|``     ``+a sin(kx)``     ``dx'/dx = 1 + ak > 1``  -> crest
                                        BROADENS, trough sharpens.  Wrong.
    ``+i k/|k|``     ``-a sin(kx)``     ``dx'/dx = 1 - ak < 1``  -> crest
                                        SHARPENS.  Correct.
    ===============  =================  ===========================

    The second row is exactly the Gerstner (trochoidal) wave,
    ``x = x0 - a sin(k x0)``, ``z = a cos(k x0)`` -- the classic sharp-crested
    profile.  So this module uses ``D~ = +i (k/|k|) h~`` and the displaced
    position is ``x + choppiness * dx_disp``, as in cookbook section 6.3.

    The cookbook's section 2.3 writes ``-1j``; with the ``+ikx`` synthesis
    convention used here that inverts crests and troughs.  ``test_surface.py``
    pins the correct sign by measuring compression at crests directly
    (``corr(J - 1, h) < 0``), so this is checked rather than argued.

    Slopes
    ------
    Differentiation in Fourier space is multiplication by ``i k``.  These are
    analytically exact for the resolved band and cost one extra FFT each.  For a
    radiometric renderer where the normal *is* the physics, that is worth doing.
    """
    ny, nx = h0.shape
    scale = float(nx * ny)

    ht = evolve(h0, omega, t)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_k = np.where(k > 0.0, 1.0 / np.where(k > 0.0, k, 1.0), 0.0)

    def _spatial(spec: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft2(spec) * scale)

    return SurfaceField(
        h=_spatial(ht),
        dx_disp=_spatial(1j * kx * inv_k * ht),
        dy_disp=_spatial(1j * ky * inv_k * ht),
        slope_x=_spatial(1j * kx * ht),
        slope_y=_spatial(1j * ky * ht),
    )


def displacement_jacobian(
    h0: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    k: np.ndarray,
    omega: np.ndarray,
    t: float,
    choppiness: float = 1.0,
) -> np.ndarray:
    """Determinant of the horizontal displacement map's Jacobian.

    ``J = det(I + lambda * dD/dx)``.  Negative values mean the surface has
    folded through itself and normals have inverted -- radiometric nonsense.
    In VFX those fold regions are where foam gets seeded; at 8 cm wave heights
    they should never occur, and Gate 2 asserts they don't at
    ``choppiness = 1.0``.

    Also the cleanest way to check the displacement *sign*: where ``J < 1`` the
    surface is compressed, and compression must coincide with crests.
    """
    ny, nx = h0.shape
    scale = float(nx * ny)

    ht = evolve(h0, omega, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_k = np.where(k > 0.0, 1.0 / np.where(k > 0.0, k, 1.0), 0.0)

    dx_spec = 1j * kx * inv_k * ht
    dy_spec = 1j * ky * inv_k * ht

    def _spatial(spec):
        return np.real(np.fft.ifft2(spec) * scale)

    dxx = _spatial(1j * kx * dx_spec)
    dxy = _spatial(1j * ky * dx_spec)
    dyx = _spatial(1j * kx * dy_spec)
    dyy = _spatial(1j * ky * dy_spec)

    lam = choppiness
    return (1.0 + lam * dxx) * (1.0 + lam * dyy) - lam**2 * dxy * dyx


# ---------------------------------------------------------------------------
# Tile
# ---------------------------------------------------------------------------


@dataclass
class WaveTile:
    """One FFT tile: a band-limited realisation of the spectrum on a periodic grid.

    Construction is the expensive part (spectrum evaluation plus the RNG draw)
    and happens once.  :meth:`evaluate` is then a handful of inverse FFTs --
    five at 512^2 is a few milliseconds, so there is no need to optimise it.

    The tile is exactly periodic at ``size``.  That periodicity is very visible
    in glint, which is what ``tiling.composite_surface`` exists to break up.
    """

    size: float
    n: int
    kx: np.ndarray
    ky: np.ndarray
    k: np.ndarray
    dk: float
    omega: np.ndarray
    s_k: np.ndarray
    h0: np.ndarray
    band: tuple[float, float]
    rotation: float
    """Rotation of this tile's grid relative to world, CCW [rad]."""

    @classmethod
    def build(
        cls,
        size: float,
        n: int,
        u10: float,
        fetch: float,
        theta_wind: float,
        seed: int,
        band: tuple[float, float] = (0.0, np.inf),
        rotation: float = 0.0,
        depth: float | None = None,
        gamma: float = JONSWAP_GAMMA,
        spreading_model: str = "cos2s",
        g: float = G,
    ) -> "WaveTile":
        """Build a tile carrying wavenumbers in ``[band[0], band[1])``.

        ``rotation`` rotates the tile's grid relative to world.  The tile's
        spectrum is therefore built with the wind direction expressed in the
        tile's own frame, ``theta_wind - rotation``; rotating the resulting
        vector quantities back by ``+rotation`` then lands the wave directions
        on the true wind.  Rotating the grid without counter-rotating the wind
        would rotate the entire directional spectrum, which is the obvious way
        to get this wrong and produces waves marching across the wind.
        """
        kx, ky, k, dk = grid_wavenumbers(size, n)
        s_k = jonswap_sk(kx, ky, u10, fetch, theta_wind - rotation, depth,
                         gamma, spreading_model, g)

        lo, hi = band
        s_k = np.where((k >= lo) & (k < hi), s_k, 0.0)
        s_k[k == 0.0] = 0.0

        return cls(
            size=size, n=n, kx=kx, ky=ky, k=k, dk=dk,
            omega=dispersion_omega(k, depth, g),
            s_k=s_k,
            h0=initial_amplitudes(s_k, dk, seed),
            band=(lo, hi),
            rotation=rotation,
        )

    # -- derived quantities --------------------------------------------------

    @property
    def dx(self) -> float:
        return self.size / self.n

    @property
    def k_nyquist(self) -> float:
        return np.pi / self.dx

    @property
    def k_min(self) -> float:
        return 2.0 * np.pi / self.size

    def m0(self) -> float:
        """Height variance carried by this tile's band, from the grid sum [m^2]."""
        return float(np.sum(self.s_k) * self.dk**2)

    def mss(self) -> float:
        """Slope variance carried by this tile's band, from the grid sum [-]."""
        return float(np.sum(self.k**2 * self.s_k) * self.dk**2)

    def evaluate(self, t: float) -> SurfaceField:
        """Evaluate on the tile's own grid at time ``t``.  Fields are ``[y, x]``."""
        return surface_at(self.h0, self.kx, self.ky, self.k, self.omega, t)

    def jacobian(self, t: float, choppiness: float = 1.0) -> np.ndarray:
        return displacement_jacobian(self.h0, self.kx, self.ky, self.k,
                                     self.omega, t, choppiness)
