"""PHASE 2 -- multi-tile composition.

A single FFT tile is exactly periodic at its own size, and the repetition is
far more visible than it looks: specular glint makes it obvious, because the eye
(and a change detector) locks onto repeated sparkle patterns much more readily
than onto repeated displacement.

The fix is to sum several tiles with incommensurate sizes, each carrying a
**disjoint** band of the spectrum, each with its own rotation.  Disjointness is
what makes the sum valid -- variances add only for uncorrelated components, so
overlapping bands would double-count energy and the composite would no longer
integrate to ``m0``.  ``SurfaceConfig`` validates the fractions are contiguous
and non-overlapping; :func:`band_edges` maps them onto real wavenumbers and
validates that each band fits inside the tile that carries it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .constants import G, JONSWAP_GAMMA
from .surface import SurfaceField, WaveTile

__all__ = [
    "GOLDEN_RATIO",
    "GOLDEN_ANGLE",
    "band_edges",
    "derive_tile_sizes",
    "tile_rotations",
    "sample_bilinear_periodic",
    "sample_periodic",
    "TileSet",
    "TileSizing",
    "composite_surface",
]

GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))
"""~137.5 deg.  Successive multiples are maximally non-repeating, which is what
we want for decorrelating tile lattices.

Rotations are derived from the tile *index*, not from the RNG.  Cookbook
section 0.3 says the seed is used in ``initial_amplitudes`` and nowhere else;
drawing rotations from it would quietly violate that and make the surface
depend on the seed through two independent paths.
"""


def tile_rotations(n_tiles: int) -> tuple[float, ...]:
    """Deterministic per-tile grid rotations [rad], CCW."""
    return tuple(float((i * GOLDEN_ANGLE) % (2.0 * np.pi)) for i in range(n_tiles))


#: Minimum ratio of any derived tile's Nyquist to the first tile's.  The first
#: tile has to remain the most restrictive, because :func:`band_edges` takes
#: ``k_ref`` from the minimum and the whole derivation is built on the first
#: tile setting it.  Without this floor a narrow second band (``band_top`` below
#: ``1/headroom``) would quietly take over as ``k_ref`` and the first edge would
#: land nowhere near the target.
_MIN_BIND = 1.05

GOLDEN_RATIO = 0.5 * (1.0 + np.sqrt(5.0))
"""~1.618.  The default Nyquist headroom for derived tiles, chosen irrational
for the same reason :data:`GOLDEN_ANGLE` is: it is what keeps the derived sizes
from sharing a finite joint period.

A rational headroom gives a rational size ratio.  At the obvious ``1.5`` the
first two tiles come out at exactly ``21/10``, so they realign every ten of the
larger tile -- 764 m on the test lake, which is inside its 1000 m domain.  An
irrational ratio never realigns at all.

It also happens to land closer to the hand-tuned configs (19.8 lambda_p for the
second tile against their 19.9 and 20.4) than 1.5 does (21.3)."""


def derive_tile_sizes(lambda_p, n, band_tops, *, pinned=(),
                      target_edge_over_k_p=2.0,
                      headroom=GOLDEN_RATIO) -> tuple[float, ...]:
    """Tile sizes [m] for the leading tiles of a set, derived from ``lambda_p``.

    Sizes are not a scene-independent constant -- see :class:`TileSizing` for
    what goes wrong when they are copied between scenes.  This is the rule that
    replaces the copying.

    Why the first tile is solved exactly
    ------------------------------------
    ``band_edges`` places the first interior edge at ``f0 * k_ref``, and
    ``k_ref`` is the smallest Nyquist in the set -- the first tile's.  Writing
    ``L0 = c0 * lambda_p`` and substituting ``k_p = 2*pi/lambda_p``:

        k_ref      = pi*n0/L0 = n0*k_p / (2*c0)
        first edge = f0*k_ref = f0*n0*k_p / (2*c0)

    so ``c0 = f0*n0 / (2*target)`` puts the edge exactly ``target`` peak
    wavenumbers up, for **any** sea state.  At ``f0 = 0.35``, ``n0 = 512`` and
    ``target = 2`` that is 44.8, which is what ``coastal_bay`` and
    ``straits_crop`` were hand-tuned to (44.84 and 44.67).  The constant is not
    magic; it is what "put the first edge at 2 k_p" evaluates to.

    The consequence worth stating: ``first_edge / k_p`` now contains no
    sea-state term at all, so it holds still across a sweep instead of sliding
    with ``k_p`` under a pinned edge.

    Why the rest are not
    --------------------
    Only the first tile sets the edges.  The others merely have to *carry* their
    band, so each is sized to put its Nyquist ``headroom x`` above its own band
    top, floored by :data:`_MIN_BIND` so the first tile keeps setting ``k_ref``.
    ``band_edges`` then validates the result and raises if any band still does
    not fit; this function proposes, it does not re-check.

    What is deliberately absent
    ---------------------------
    **The last tile.**  It sets ``k_max`` -- the top of the resolved range, and
    therefore where the mesh/BSDF handoff lands -- which is a rendering
    decision, not a property of the sea.  Scaling it with ``lambda_p`` at fixed
    ``n`` coarsens its grid: measured on ``straits``, deriving all three tiles
    dropped ``k_max`` from 35.0 to 9.1 rad/m and destroyed 36% of the resolved
    slope variance, against 0.2% when the top tile was left alone.  So this
    returns ``len(n)`` sizes and expects the caller to hold the top tile itself.

    When the pinned tile does not fit
    ---------------------------------
    A tile held fixed while the derived ones scale will eventually stop being
    compatible with them, at both ends:

    * **Small seas.**  The derived tiles shrink until tile 0's Nyquist rises
      *past* the pinned tile's.  ``k_ref`` then comes from the pinned tile
      instead, and the first edge lands wherever that puts it -- silently.  With
      a 23 m / 256 top tile the crossover is ``lambda_p ~ 1.03 m``; below it,
      measured, the edge slid to 0.97 k_p while every other number stayed right.
    * **Large seas.**  ``k_ref`` falls until the pinned tile's ``k_min`` no
      longer reaches down to its own band.  ``band_edges`` raises for this one;
      the crossover is ``lambda_p ~ 92 m``.

    Between those bounds the derivation is exact.  That interval covers every
    realistic littoral sea, and outside it the answer is not a cleverer
    derivation but a top tile chosen for the render you are actually doing.
    Pass ``pinned`` and the first failure becomes an exception instead of a
    quietly wrong band split.

    Parameters
    ----------
    lambda_p:
        Peak wavelength of the scene's sea [m].
    n:
        Samples per side for each tile being derived, leading order.
    band_tops:
        Upper band fraction of each of those tiles, same order and length.
    pinned:
        ``(size, n)`` for each tile *not* being derived -- in practice the top
        one.  Optional, but supplying it is strongly preferred: it is the only
        way this function can tell that a pinned tile has undercut the derived
        ones.  See "When the pinned tile does not fit" below.
    target_edge_over_k_p:
        Where to put the first interior edge, in units of ``k_p``.  1.5-3 keeps
        the bands carrying meaningfully different physics; the shipped configs
        sit at 2.
    headroom:
        How far above its own band top to put each subsequent tile's Nyquist.
        Defaults to :data:`GOLDEN_RATIO`; keep it irrational unless you have a
        reason not to, or the derived sizes acquire a finite joint period.

    Raises
    ------
    ValueError
        On a non-positive ``lambda_p``, mismatched lengths, or a
        ``target_edge_over_k_p`` that is not positive and finite.
    """
    n = tuple(int(v) for v in n)
    band_tops = tuple(float(v) for v in band_tops)
    if len(n) != len(band_tops):
        raise ValueError(
            f"n and band_tops must be the same length, got {len(n)} and "
            f"{len(band_tops)}")
    if not n:
        raise ValueError("nothing to derive: n is empty")
    if not np.isfinite(lambda_p) or lambda_p <= 0.0:
        raise ValueError(f"lambda_p must be positive and finite, got {lambda_p}")
    if not np.isfinite(target_edge_over_k_p) or target_edge_over_k_p <= 0.0:
        raise ValueError(
            f"target_edge_over_k_p must be positive and finite, got "
            f"{target_edge_over_k_p}")
    if headroom <= 1.0:
        raise ValueError(f"headroom must exceed 1, got {headroom}")
    if any(v <= 0 for v in n):
        raise ValueError(f"every n must be positive, got {n}")
    if any(not 0.0 < f <= 1.0 for f in band_tops):
        raise ValueError(f"every band top must lie in (0, 1], got {band_tops}")

    c0 = band_tops[0] * n[0] / (2.0 * target_edge_over_k_p)
    sizes = [c0 * lambda_p]
    for n_i, top_i in zip(n[1:], band_tops[1:]):
        # Nyquist ratio to tile 0; >= _MIN_BIND keeps tile 0 the minimum.
        ratio = max(headroom * top_i, _MIN_BIND)
        sizes.append(c0 * (n_i / n[0]) / ratio * lambda_p)
    sizes = [float(s) for s in sizes]

    # The derivation is only correct while tile 0 is the most restrictive tile
    # in the *whole* set, pinned tiles included -- that is what makes `k_ref`
    # tile 0's Nyquist and the first edge land on target. A pinned tile that
    # undercuts it does not raise anywhere downstream; it just takes over
    # `k_ref` and moves the edge.
    k_nyq_0 = np.pi * n[0] / sizes[0]
    for size_p, n_p in pinned:
        k_nyq_p = np.pi * int(n_p) / float(size_p)
        if k_nyq_p < k_nyq_0:
            # k_nyq_0 = pi*n0/(c0*lambda_p), so it drops below the pinned tile's
            # Nyquist once lambda_p >= pi*n0/(c0*k_nyq_p).
            lam_min = np.pi * n[0] / (c0 * k_nyq_p)
            raise ValueError(
                f"pinned tile size={size_p} n={n_p} has Nyquist "
                f"{k_nyq_p:.3f} rad/m, below the derived first tile's "
                f"{k_nyq_0:.3f} rad/m. It would silently become k_ref and put "
                f"the first band edge somewhere other than "
                f"{target_edge_over_k_p} k_p. This sea (lambda_p "
                f"{lambda_p:.2f} m) is too small for that top tile, which needs "
                f"lambda_p above {lam_min:.2f} m -- raise the pinned tile's n "
                f"or shrink it.")
    return tuple(sizes)


def band_edges(tiles) -> tuple[tuple[float, float], ...]:
    """Map config band *fractions* onto wavenumber edges [rad/m].

    The cookbook describes ``band`` as "fractions of the radial wavenumber
    range" without pinning what that range is.  It has to be pinned, because the
    obvious reading does not work: taking fractions of ``[0, max_i k_nyq_i]``
    puts the middle band at ``[12.2, 24.5]`` for the shipped test config, and
    the tile assigned to carry it has a Nyquist of 21.7 -- so the top of its own
    band is unrepresentable on its own grid, and that energy vanishes silently.

    The interpretation used here, which does work:

    * ``k_ref = min_i k_nyquist_i`` -- the most restrictive tile sets the scale,
      guaranteeing every interior band fits on the grid that carries it.
    * Interior edges are ``fraction * k_ref``.
    * The topmost band (``fraction == 1.0``) extends to *its own* tile's
      Nyquist rather than to ``k_ref``, so the finest tile contributes all the
      resolution it actually has instead of throwing the top octave away.

    For the shipped test config (Nyquists 25.1 / 21.7 / 35.0 rad/m) this gives
    ``[0, 7.6) [7.6, 15.2) [15.2, 35.0)``, each comfortably inside its tile.

    Raises
    ------
    ValueError
        If any band falls outside the resolvable range of its tile.  Silently
        clipping instead would lose variance and break the Gate 2 composite Hs
        check in a way that is very hard to trace back to here.
    """
    tiles = list(tiles)
    k_ref = min(t.k_nyquist for t in tiles)

    edges = []
    for t in tiles:
        lo_frac, hi_frac = t.band
        lo = lo_frac * k_ref
        hi = t.k_nyquist if hi_frac >= 1.0 else hi_frac * k_ref
        if hi > t.k_nyquist * (1.0 + 1e-12):
            raise ValueError(
                f"tile size={t.size} n={t.n} has Nyquist {t.k_nyquist:.3f} rad/m "
                f"but its band reaches {hi:.3f} rad/m -- it cannot represent its "
                f"own band"
            )
        if lo > 0.0 and lo < t.k_min:
            raise ValueError(
                f"tile size={t.size} n={t.n} has k_min {t.k_min:.3f} rad/m "
                f"but its band starts at {lo:.3f} rad/m"
            )
        edges.append((float(lo), float(hi)))
    return tuple(edges)


# ---------------------------------------------------------------------------
# Periodic sampling
# ---------------------------------------------------------------------------


def sample_bilinear_periodic(field: np.ndarray, x, y, size: float) -> np.ndarray:
    """Bilinear sample of a periodic ``[y, x]`` field at world coordinates.

    Retained for reference and for the interpolator comparison test.  Production
    sampling uses :func:`sample_periodic`, which defaults to cubic -- see there
    for why bilinear is not good enough.
    """
    n = field.shape[0]
    dx = size / n

    u = np.asarray(x, dtype=np.float64) / dx
    v = np.asarray(y, dtype=np.float64) / dx

    j0 = np.floor(u)
    i0 = np.floor(v)
    fu = u - j0
    fv = v - i0

    j0 = j0.astype(np.int64) % n
    i0 = i0.astype(np.int64) % n
    j1 = (j0 + 1) % n
    i1 = (i0 + 1) % n

    return (
        field[i0, j0] * (1 - fu) * (1 - fv)
        + field[i0, j1] * fu * (1 - fv)
        + field[i1, j0] * (1 - fu) * fv
        + field[i1, j1] * fu * fv
    )


def sample_periodic(field: np.ndarray, x, y, size: float, order: int = 3) -> np.ndarray:
    """Sample a periodic ``[y, x]`` field at world coordinates.

    ``field`` is ``(n, n)`` covering ``[0, size)`` in both axes with wraparound.
    Coordinates outside the tile wrap, which is exactly the periodicity the
    rotations and incommensurate sizes exist to hide.

    Why cubic and not bilinear
    --------------------------
    Bilinear interpolation is a low-pass filter, and the tiles carry meaningful
    energy right up to their Nyquist.  For a sinusoid at wavenumber ``k`` on a
    grid of spacing ``d``, sampled at a uniformly-distributed sub-cell offset,
    the retained power is ``int_0^1 |1 - u + u e^(ikd)|^2 du``, which at the
    Nyquist ``kd = pi`` is ``int_0^1 (1-2u)^2 du = 1/3``.  Two thirds of the
    power in the top octave is destroyed.

    That is not an artefact of a test harness -- Phase 6 samples this composite
    at every mesh vertex, so bilinear would quietly shave variance off the
    delivered surface and, worse, shave it preferentially from the *high*
    wavenumbers, which is exactly the content the LOD invariant is accounting
    for.  Measured on the shipped test config, bilinear loses ~6% of Hs and
    cubic ~1%.

    ``order=1`` selects bilinear for comparison; ``order=3`` is an interpolating
    cubic spline (exact at the nodes, far flatter passband).
    """
    from scipy.ndimage import map_coordinates

    n = field.shape[0]
    dx = size / n
    coords = np.stack([
        np.asarray(y, dtype=np.float64) / dx,
        np.asarray(x, dtype=np.float64) / dx,
    ])
    return map_coordinates(field, coords, order=order, mode="grid-wrap")


# ---------------------------------------------------------------------------
# Tile set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileSizing:
    """Whether a tile set is sized for the sea it is carrying.

    Tile sizes are not a scene-independent constant, and the way they go wrong
    is silent: every gate still passes, every number still reconciles, and the
    band machinery simply stops doing anything.

    The documented rule -- "at least one tile's Nyquist must clear ``k_p``" --
    only guards the **top** of the range. What decides whether the bands earn
    their keep is the **first interior edge**, and nothing was checking it.
    Because JONSWAP's shape is universal in ``f/f_p``, the variance split is a
    pure function of where that edge sits in units of ``k_p``:

    ==============  ==================
    first edge      band 1 then holds
    ==============  ==================
    1.5 k_p         71.9%
    2 k_p           82.4%
    3 k_p           91.5%
    8 k_p           98.7%
    16 k_p          99.7%
    ==============  ==================

    So a tile set tuned for 1.7 m chop and then pointed at a 13.6 m sea puts its
    first edge at 16 k_p and collapses the whole spectrum into one band.

    Note what is *not* here: tile size has no measurable effect on spectral
    accuracy. Holding the Nyquist fixed and growing the largest tile from 4.7 to
    75 peak wavelengths moves the realised ``Hs`` by 0.08%, because the build
    integrates the spectrum over each grid cell rather than point-sampling it.
    The reason to want a large tile is spatial repetition, which is a different
    complaint with a different remedy.
    """

    lambda_p: float
    """Peak wavelength of the scene's sea [m]."""
    k_p: float
    """Peak wavenumber [rad/m]."""
    first_edge_over_k_p: float
    """First interior band edge, in units of ``k_p``. Aim for 1.5-3."""
    band_shares: tuple[float, ...]
    """Fraction of the total variance each band carries."""
    largest_tile_m: float
    largest_tile_over_lambda_p: float
    domain_m: float
    notes: tuple[str, ...]
    """Human-readable findings. Empty means nothing looked wrong."""

    @property
    def bands_are_inert(self) -> bool:
        """True when one band holds so much that per-band physics is moot."""
        return bool(self.band_shares) and max(self.band_shares) > 0.95


@dataclass
class TileSet:
    """A set of band-limited tiles that sum to the full-band surface."""

    tiles: tuple[WaveTile, ...]
    #: Kept so :meth:`band_limited` can rebuild with the same seeds. Optional
    #: only so that ``TileSet(tiles)`` still works for hand-assembled sets.
    cfg: "Config | None" = None
    depth: float | None = None

    @classmethod
    def build(
        cls,
        cfg: Config,
        depth: float | None = None,
        k_cut: float | None = None,
    ) -> "TileSet":
        """Build the tile set described by a :class:`~pywave.config.Config`.

        Per-tile RNG streams are spawned from the single config seed with
        ``SeedSequence``.  That keeps "one integer in config" true while giving
        each tile an independent, reproducible stream -- reusing the same
        integer for every tile would correlate the tiles' phases and produce a
        surface whose bands are locked together.

        ``k_cut`` truncates every band at that wavenumber, dropping tiles that
        lie entirely above it.  Seeds are spawned for *all* configured tiles
        before any are dropped, so a truncated set is the same realisation of
        the same sea with its short waves removed -- not a different sea.  See
        :meth:`band_limited` for why anything would want that.
        """
        edges = band_edges(cfg.surface.tiles)
        rotations = tile_rotations(len(cfg.surface.tiles))
        children = np.random.SeedSequence(cfg.spectrum.seed).spawn(
            len(cfg.surface.tiles)
        )

        if k_cut is not None:
            if not np.isfinite(k_cut) or k_cut <= 0.0:
                raise ValueError(f"k_cut must be positive and finite, got {k_cut}")
            edges = tuple((lo, min(hi, float(k_cut))) for lo, hi in edges)

        tiles = tuple(
            WaveTile.build(
                size=tc.size,
                n=tc.n,
                u10=cfg.wind.speed,
                fetch=cfg.wind.fetch,
                theta_wind=cfg.wind.direction_rad,
                seed=child,
                band=band,
                rotation=rot,
                depth=depth,
                gamma=cfg.spectrum.gamma,
                spreading_model=cfg.spectrum.spreading,
            )
            for tc, band, rot, child in zip(
                cfg.surface.tiles, edges, rotations, children
            )
            # An empty band carries nothing; keeping it would cost an FFT per
            # frame to add zero. The lowest tile always survives, because its
            # band starts at k = 0 and k_cut is positive.
            if band[1] > band[0]
        )
        return cls(tiles, cfg=cfg, depth=depth)

    def band_limited(self, k_cut: float) -> "TileSet":
        """This same sea with everything above ``k_cut`` removed.

        A mesh of spacing ``dx`` cannot represent waves shorter than ``2*dx``.
        Sampling them anyway does not discard them -- it *folds* them down onto
        long wavelengths, where they appear as swell that is not there. The
        remedy for aliasing is always the same: remove what cannot be
        represented **before** sampling, not after.

        Nothing is lost physically. That variance is already accounted for: it
        is exactly what :func:`pywave.channels.submesh_mss` hands to the BSDF as
        sub-facet roughness, which is the whole content of the LOD invariant

            ``mss_resolved(dx) + mss_above(pi/dx) = mss_total``.

        Without this, the mesh carries that energy *as well*, aliased, while the
        BSDF is also told about it -- counted twice, and in the wrong place.
        Measured on the straits scene (lambda_p 8.5 m, surface synthesised down
        to lambda 0.18 m), the elevation power at lambda > 6.3 m came out
        +6.4% at 2 m posts and +88.5% at 4 m posts; at 0.25 m it is -0.00%,
        which is why fine-meshed scenes never showed it.
        """
        if self.cfg is None:
            raise ValueError(
                "this TileSet was assembled directly rather than built from a "
                "config, so it cannot be rebuilt; use TileSet.build(cfg, "
                "k_cut=...) instead"
            )
        if not np.isfinite(k_cut) or k_cut <= 0.0:
            raise ValueError(f"k_cut must be positive and finite, got {k_cut}")
        if k_cut >= self.k_max:
            return self          # already inside the cut; nothing to remove
        return TileSet.build(self.cfg, depth=self.depth, k_cut=float(k_cut))

    # -- aggregate spectral properties --------------------------------------

    def m0(self) -> float:
        """Total height variance across all bands [m^2].

        Valid as a plain sum precisely because the bands are disjoint.
        """
        return float(sum(t.m0() for t in self.tiles))

    def mss(self) -> float:
        """Total resolved slope variance across all bands [-]."""
        return float(sum(t.mss() for t in self.tiles))

    def hs(self) -> float:
        return 4.0 * np.sqrt(self.m0())

    @property
    def k_max(self) -> float:
        """Highest wavenumber the composite resolves [rad/m]."""
        return max(t.band[1] for t in self.tiles)

    def sizing(self) -> "TileSizing":
        """Is this tile set sized for *this* scene's sea? See :class:`TileSizing`."""
        if self.cfg is None:
            raise ValueError(
                "sizing() needs the config this set was built from; use "
                "TileSet.build(cfg) rather than assembling tiles directly")
        cfg = self.cfg
        k_p = cfg.k_p
        total = self.m0()
        shares = tuple(float(t.m0() / total) if total > 0 else 0.0
                       for t in self.tiles)
        first_edge = self.tiles[0].band[1] if len(self.tiles) > 1 else float("inf")
        biggest = max(t.size for t in self.tiles)
        domain = float(max(cfg.scene.domain))

        notes = []
        if len(self.tiles) > 1 and shares[0] > 0.95:
            notes.append(
                f"band 1 carries {shares[0]:.1%} of the variance: its top edge "
                f"is at {first_edge / k_p:.1f} k_p, far above the peak. Every "
                f"frequency-dependent nearshore effect is then computed at one "
                f"representative frequency, so the disjoint bands are costing "
                f"{len(self.tiles)} FFTs per frame and buying nothing. Size the "
                f"tiles so the first edge lands near 1.5-3 k_p.")
        # Deliberately measured against the peak wavelength and not against the
        # domain. A tile smaller than the domain is the normal case and not a
        # defect: incommensurate sizes and golden-angle rotations mean the
        # *composite* does not repeat even though each tile does. What a short
        # tile really costs is variety -- it holds only a few wave groups, and
        # those few recur.
        if biggest < 10.0 * cfg.lambda_p:
            notes.append(
                f"the largest tile is {biggest:.0f} m, only "
                f"{biggest / cfg.lambda_p:.1f} peak wavelengths across, so it "
                f"contains just a handful of wave groups and those groups "
                f"recur. This costs variety, not energy: holding the Nyquist "
                f"fixed, tile size moves the realised Hs by under 0.1%.")
        return TileSizing(
            lambda_p=cfg.lambda_p, k_p=k_p,
            first_edge_over_k_p=float(first_edge / k_p),
            band_shares=shares, largest_tile_m=float(biggest),
            largest_tile_over_lambda_p=float(biggest / cfg.lambda_p),
            domain_m=domain, notes=tuple(notes))

    def evaluate_grids(self, t: float) -> tuple[SurfaceField, ...]:
        """Evaluate every tile on its own grid.  Cache this if sampling repeatedly."""
        return tuple(tile.evaluate(t) for tile in self.tiles)

    def sample(self, x, y, t: float, fields=None, order: int = 3) -> SurfaceField:
        """Sample the composite surface at world coordinates ``(x, y)``, time ``t``."""
        return composite_surface(self.tiles, x, y, t, fields=fields, order=order)


def composite_surface(tiles, x, y, t: float, fields=None, order: int = 3,
                      weights=None, rotate=None) -> SurfaceField:
    """Sum tiles sampled at ``(x, y) mod tile size``, each with its own rotation.

    Frames and rotations
    --------------------
    Each tile's grid is rotated by ``phi`` relative to world.  So:

    1. World sample points are rotated *into* the tile frame by ``-phi``.
    2. The tile is evaluated there.
    3. Vector-valued results -- horizontal displacement and slope -- are rotated
       *back* by ``+phi``.

    Step 3 is easy to forget and produces a surface whose normals are subtly
    wrong per tile in a way that averages out in ``mss`` and therefore survives
    every scalar check.  It shows up only as anisotropy pointing the wrong way,
    which in Phase 7 means glint elongated along the wrong axis.

    (The counter-rotation of the wind direction that keeps wave headings correct
    happens at tile *construction*; see ``WaveTile.build``.)

    Nearshore hooks
    ---------------
    ``weights`` and ``rotate`` exist for Phase 5 and default to no-ops.

    ``weights`` is one amplitude scale per tile, broadcastable against the
    sample points -- the per-band shoaling gain, which must be applied per band
    because shoaling is frequency-dependent.

    ``rotate`` is an *additional* per-point rotation applied to the vector
    quantities only, on top of ``phi``: the local refraction angle.  It turns
    the surface normal without moving the sample position, which is exactly the
    approximation Phase 5 documents -- the wave field is not re-solved, so crest
    positions do not move, but everything the BSDF sees is reoriented.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if fields is None:
        fields = [tile.evaluate(t) for tile in tiles]
    if weights is None:
        weights = [1.0] * len(list(tiles))
    if rotate is None:
        rotate = [0.0] * len(list(tiles))

    total = None
    for tile, field, weight, psi in zip(tiles, fields, weights, rotate):
        phi = tile.rotation
        c, s = np.cos(phi), np.sin(phi)

        # world -> tile frame
        xl = c * x + s * y
        yl = -s * x + c * y

        h = sample_periodic(field.h, xl, yl, tile.size, order)
        dxl = sample_periodic(field.dx_disp, xl, yl, tile.size, order)
        dyl = sample_periodic(field.dy_disp, xl, yl, tile.size, order)
        sxl = sample_periodic(field.slope_x, xl, yl, tile.size, order)
        syl = sample_periodic(field.slope_y, xl, yl, tile.size, order)

        # tile frame -> world, for the vector quantities only, plus any
        # per-point refraction rotation.
        cv, sv = np.cos(phi + psi), np.sin(phi + psi)
        contribution = SurfaceField(
            h=weight * h,
            dx_disp=weight * (cv * dxl - sv * dyl),
            dy_disp=weight * (sv * dxl + cv * dyl),
            slope_x=weight * (cv * sxl - sv * syl),
            slope_y=weight * (sv * sxl + cv * syl),
        )
        total = contribution if total is None else total + contribution

    if total is None:
        raise ValueError("composite_surface requires at least one tile")
    return total
