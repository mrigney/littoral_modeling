"""PHASE 3 -- Gate 2: FFT surface synthesis.

Two checks here earn their keep by catching errors nothing else catches:

`test_crest_travels_at_phase_speed`
    The time-evolution sign error leaves the surface exactly real, with the
    right variance, the right spectrum and the right slopes -- the sea simply
    marches backwards into the wind. Tracking a crest is the only check that
    sees it.

`test_displacement_compresses_at_crests`
    The horizontal displacement sign error inverts crests and troughs while
    preserving every scalar statistic. Measuring compression against elevation
    is the only check that sees it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.stats import kurtosis, skew

from pywave import moments, spectrum, surface, tiling

pytestmark = pytest.mark.gate2

REPO_ROOT = Path(__file__).resolve().parent.parent


def _jacobian_weighted_moments(h: np.ndarray, j: np.ndarray):
    """Skewness/kurtosis of elevation over uniform *horizontal area*.

    The displacement map sends a grid cell of area `dA` to `J dA`, so the
    elevation seen at a uniformly random horizontal position is the grid
    elevation weighted by `J`. Validated against explicit resampling of a
    trochoid: for a single mode at `ka = 0.47` both give +0.508.
    """
    w = j.ravel()
    w = w / w.sum()
    x = h.ravel()
    mu = np.sum(w * x)
    var = np.sum(w * (x - mu) ** 2)
    sd = np.sqrt(var)
    return float(np.sum(w * (x - mu) ** 3) / sd**3), float(np.sum(w * (x - mu) ** 4) / var**2 - 3.0)


# ---------------------------------------------------------------------------
# Realness and Hermitian symmetry
# ---------------------------------------------------------------------------


def test_surface_is_real(record, tileset):
    """`max|imag| < 1e-9` -- the Hermitian symmetry `flip_k` enforces."""
    worst = 0.0
    for tile in tileset.tiles:
        for t in (0.0, 3.7):
            raw = np.fft.ifft2(surface.evolve(tile.h0, tile.omega, t)) * tile.n**2
            worst = max(worst, float(np.max(np.abs(raw.imag))))

    record("2", "max |imag(h)| over all tiles, t = 0 and 3.7 s", worst, 0.0, 1e-9,
           unit="m", note="Elevations are ~0.04 m, so this is at the float64 noise floor.",
           passed=worst < 1e-9)
    assert worst < 1e-9


def test_flip_k_maps_each_bin_to_its_negative():
    """`flip_k(a)[i, j] == a[-i % N, -j % N]`, DC included.

    The off-by-one hides in the wrap: reversing maps `j -> N-1-j`, so the roll by
    +1 is what lands it on `N-j`. Nyquist rows are their own negatives on an even
    grid, so a wrong implementation nearly works -- checking every bin is the
    point.
    """
    rng = np.random.default_rng(0)
    for n in (8, 16, 32):
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        flipped = surface.flip_k(a)
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        assert np.array_equal(flipped, a[(-i) % n, (-j) % n])
        assert flipped[0, 0] == a[0, 0]


# ---------------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------------


def test_single_tile_hs_matches_spectrum(record, scene):
    """`4*std(h)` matches the band-limited spectral Hs within 5%.

    Compared against the theory *over the band this tile resolves*, which is the
    only fair comparison: a grid truncated at its Nyquist genuinely carries less
    variance than the untruncated spectrum, and comparing against the latter
    would look like a bug when it is just missing high-frequency content.
    """
    u10, fetch, gamma, theta_wind = scene
    size, n = 128.0, 1024
    tile = surface.WaveTile.build(size, n, u10, fetch, theta_wind, seed=20260801)

    realised = 4.0 * float(np.std(tile.evaluate(0.0).h))
    grid = 4.0 * np.sqrt(tile.m0())
    band = 4.0 * np.sqrt(moments.mss_between(2.0 * np.pi / size, tile.k_nyquist,
                                             u10, fetch, gamma=gamma, order=0))
    full = spectrum.hs_spectral(u10, fetch, gamma)

    rel_band = abs(realised - band) / band
    rel_full = abs(realised - full) / full

    record("2", f"Hs realised, single tile L={size:.0f} m N={n}", realised, band, 0.05,
           unit="m", note=f"Band-limited theory over k in "
                          f"[{2 * np.pi / size:.3f}, {tile.k_nyquist:.2f}] rad/m. "
                          f"Grid sum gives {grid:.5f} m; untruncated spectrum {full:.5f} m "
                          f"({100 * (realised / full - 1):+.1f}%).",
           passed=rel_band < 0.05)
    record("2", "Hs realised vs untruncated spectral Hs", realised, full, 0.05, unit="m",
           passed=rel_full < 0.05)

    assert rel_band < 0.05
    assert rel_full < 0.05


def test_composite_hs_matches_spectrum(record, scene, tileset, tileset_fields, sample_points):
    """The three-tile composite also matches Hs within 5%."""
    u10, fetch, gamma, _ = scene
    x, y = sample_points
    field = tileset.sample(x, y, 0.0, fields=tileset_fields)

    realised = 4.0 * float(np.std(field.h))
    grid = tileset.hs()
    band = 4.0 * np.sqrt(moments.mss_between(moments.K_MIN_DEFAULT, tileset.k_max,
                                             u10, fetch, gamma=gamma, order=0))
    rel = abs(realised - band) / band

    record("2", "Hs realised, 3-tile composite", realised, band, 0.05, unit="m",
           note=f"{x.size} scattered world points, cubic sampling. Grid sum of the "
                f"three disjoint bands gives {grid:.5f} m. The "
                f"{100 * (realised / grid - 1):+.1f}% shortfall is interpolation loss.",
           passed=rel < 0.05)
    record("2", "composite band edges", str([f"[{t.band[0]:.2f}, {t.band[1]:.2f})"
                                             for t in tileset.tiles]), unit="rad/m")
    assert rel < 0.05


def test_disjoint_bands_sum_to_the_total(record, scene, tileset):
    """Tile variances add, because the bands do not overlap."""
    u10, fetch, gamma, _ = scene
    summed = sum(t.m0() for t in tileset.tiles)
    band = moments.mss_between(moments.K_MIN_DEFAULT, tileset.k_max,
                               u10, fetch, gamma=gamma, order=0)
    rel = abs(summed - band) / band

    record("2", "sum of per-tile m0 vs band-limited theory", summed, band, 0.01, unit="m^2",
           note="Variances add only for uncorrelated components; this is what "
                "makes the disjoint-band construction valid.",
           passed=rel < 0.01)
    assert rel < 0.01


# ---------------------------------------------------------------------------
# Slope and the LOD invariant
# ---------------------------------------------------------------------------


def test_resolved_mss_matches_band_limited_theory(record, scene, tileset, tileset_fields,
                                                  sample_points):
    """`mean(slope_x^2 + slope_y^2)` matches `mss_above(0) - mss_above(k_max)` within 10%."""
    u10, fetch, gamma, _ = scene
    theory = (moments.mss_above(0.0, u10, fetch, gamma=gamma)
              - moments.mss_above(tileset.k_max, u10, fetch, gamma=gamma))

    grid = tileset.mss()
    x, y = sample_points
    realised = tileset.sample(x, y, 0.0, fields=tileset_fields).mss_resolved()

    rel_grid = abs(grid - theory) / theory
    rel_real = abs(realised - theory) / theory

    record("2", "resolved mss, grid sum", grid, theory, 0.01, passed=rel_grid < 0.01)
    record("2", "resolved mss, realised from samples", realised, theory, 0.10,
           note=f"{100 * (realised / grid - 1):+.1f}% below the grid sum -- cubic "
                f"interpolation attenuates the top octave, which is where most "
                f"slope variance lives.",
           passed=rel_real < 0.10)
    assert rel_grid < 0.01
    assert rel_real < 0.10


def test_lod_invariant(record, scene, tileset):
    """`mss_resolved(dx) + mss_above(pi/dx) = mss_total`.

    Cookbook section 6.4. This is what keeps appearance constant across LOD
    transitions: geometry lost when the mesh coarsens reappears as BSDF
    roughness, and the total radiometric response is unchanged.
    """
    u10, fetch, gamma, _ = scene
    total = moments.mss_above(0.0, u10, fetch, gamma=gamma)

    resolved = tileset.mss()
    above = moments.mss_above(tileset.k_max, u10, fetch, gamma=gamma)
    rel = abs(resolved + above - total) / total

    record("2", "LOD invariant: mss_resolved + mss_above(k_max)", resolved + above, total, 1e-3,
           note=f"resolved = {resolved:.5f} (k < {tileset.k_max:.2f} rad/m), "
                f"sub-grid = {above:.5f}, i.e. {100 * above / total:.0f}% of the "
                f"total slope variance is below the composite's resolution and is "
                f"carried by the BSDF.",
           passed=rel < 1e-3)
    assert rel < 1e-3

    # ... and across a range of mesh spacings, which is what LOD actually varies.
    worst = 0.0
    for dx in (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0):
        k_cut = np.pi / dx
        lo = moments.mss_between(moments.K_MIN_DEFAULT, k_cut, u10, fetch, gamma=gamma, order=2)
        hi = moments.mss_above(k_cut, u10, fetch, gamma=gamma)
        worst = max(worst, abs(lo + hi - total) / total)

    record("2", "LOD invariant, worst over mesh_dx 0.0625-2 m", worst, 0.0, 1e-6,
           note="Analytic split; exact by construction, so this is a guard against "
                "the two integrals drifting apart in future refactors.",
           passed=worst < 1e-6)
    assert worst < 1e-6


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------


def test_height_distribution_is_gaussian(record, scene, tileset, tileset_fields,
                                         sample_points):
    """The linear surface is Gaussian; the displaced one is reported, not bounded.

    Replaces the cookbook's "skewness in [0, 0.3] with choppiness on", which this
    model cannot produce -- see Gate deviations in the report.
    """
    u10, fetch, gamma, theta_wind = scene
    x, y = sample_points
    h = tileset.sample(x, y, 0.0, fields=tileset_fields).h

    sk = float(skew(h))
    ku = float(kurtosis(h))

    record("2", "skewness of h (linear surface)", sk, 0.0, 0.05,
           note="Gaussian by construction: independent normal amplitude draws "
                "summed over modes. A non-zero value here would mean the "
                "amplitude draw or the Hermitian symmetry is wrong.",
           passed=abs(sk) < 0.05)
    record("2", "excess kurtosis of h (linear surface)", ku, 0.0, 0.1, passed=abs(ku) < 0.1)

    # Displaced surface, area-weighted. Reported, not bounded.
    #
    # Measured on a single *full-band* tile rather than the band-limited members
    # of the composite: elevation skewness is a property of the whole surface,
    # and a tile carrying only [7.6, 15.2) rad/m has no meaningful elevation
    # distribution of its own.
    full = surface.WaveTile.build(64.0, 512, u10, fetch, theta_wind,
                                  seed=20260801, gamma=gamma)
    field = full.evaluate(0.0)
    for chop in (0.0, 1.0, 1.5):
        s, _ = _jacobian_weighted_moments(field.h, full.jacobian(0.0, chop))
        record("2", f"skewness of displaced surface, choppiness = {chop}", s,
               note="Full-band tile, area-weighted by the displacement Jacobian. "
                    "Near zero because the model is linear in elevation; the "
                    "positive skewness of a real sea comes from second-order Stokes "
                    "terms that are not modelled.")

    assert abs(sk) < 0.05
    assert abs(ku) < 0.1


# ---------------------------------------------------------------------------
# Displacement
# ---------------------------------------------------------------------------


def test_jacobian_has_no_folds_at_physical_choppiness(record, tileset):
    """`det(I + lambda dD/dx) > 0` at `choppiness = 1.0`.

    Negative values mean the surface has folded through itself and normals have
    inverted -- radiometric nonsense. At 8 cm wave heights it should not happen.
    """
    worst = np.inf
    for tile in tileset.tiles:
        for t in (0.0, 1.3, 7.9):
            worst = min(worst, float(tile.jacobian(t, 1.0).min()))

    record("2", "min Jacobian determinant, choppiness = 1.0", worst, note=(
        "Over three tiles and three times. Must stay > 0; values below 1 are "
        "compression, which is what sharpens crests."), passed=worst > 0.0)
    assert worst > 0.0


def test_displacement_compresses_at_crests(record, tileset):
    """Compression must coincide with crests, i.e. `corr(J - 1, h) < 0`.

    Pins the sign of the horizontal displacement transfer function. The cookbook
    writes `-1j`; with numpy's `+ikx` synthesis that broadens crests and sharpens
    troughs, inverting the profile while leaving every scalar statistic intact.
    """
    worst = -np.inf
    for tile in tileset.tiles:
        field = tile.evaluate(0.0)
        j = tile.jacobian(0.0, 1.0)
        c = float(np.corrcoef((j - 1.0).ravel(), field.h.ravel())[0, 1])
        worst = max(worst, c)

    record("2", "worst corr(J - 1, h) over tiles", worst, note=(
        "Negative means the surface compresses where elevation is high, i.e. "
        "crests sharpen and troughs broaden, as in a Gerstner trochoid."),
        passed=worst < 0.0)
    assert worst < 0.0


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def test_crest_travels_at_phase_speed(record):
    """A tracked crest moves at `c = omega/k` within 5%, in the +k direction.

    The check that catches the time-evolution sign error. With the cookbook's
    `e^(+i omega t)` the measured speed is `-c`: the surface is still exactly
    real, still has the right variance and spectrum, and marches into the wind.
    """
    size, n, mode = 64.0, 256, 4
    kx, ky, k, _ = surface.grid_wavenumbers(size, n)

    h0 = np.zeros((n, n), dtype=complex)
    h0[0, mode] = 1.0
    omega = spectrum.dispersion_omega(k)

    k_mode = float(k[0, mode])
    c_theory = float(omega[0, mode]) / k_mode
    dx = size / n

    def crest_x(t: float) -> float:
        row = surface.surface_at(h0, kx, ky, k, omega, t).h[0]
        i = int(np.argmax(row))
        a, b, cc = row[(i - 1) % n], row[i], row[(i + 1) % n]
        return (i + 0.5 * (a - cc) / (a - 2 * b + cc)) * dx

    dt = 0.05
    measured = ((crest_x(dt) - crest_x(0.0)) % size) / dt
    rel = abs(measured - c_theory) / c_theory

    record("2", "crest phase speed", measured, c_theory, 0.05, unit="m/s",
           note=f"Single mode, k = {k_mode:.4f} rad/m, along +x. A sign error in the "
                f"time evolution would give {-c_theory:.4f} m/s.",
           passed=rel < 0.05)
    assert rel < 0.05
    assert measured > 0.0


@pytest.mark.slow
def test_zero_crossing_period_from_time_series(record, scene):
    """Tz measured by counting zero crossings matches theory within 10%.

    Band-limited to what the grid resolves. The untruncated Tz is 0.816 s against
    a measured 0.91 s -- an 11.7% "failure" that is entirely the missing
    high-frequency content, which is why the comparison must be banded.
    """
    u10, fetch, gamma, theta_wind = scene
    tile = surface.WaveTile.build(64.0, 256, u10, fetch, theta_wind, seed=20260801)

    dt, duration = 1.0 / 24.0, 48.0
    times = np.arange(0.0, duration, dt)
    probes = np.arange(0, tile.n, 16)

    series = np.empty((times.size, probes.size, probes.size))
    for i, t in enumerate(times):
        h = np.real(np.fft.ifft2(surface.evolve(tile.h0, tile.omega, t)) * tile.n**2)
        series[i] = h[np.ix_(probes, probes)]

    a = series.reshape(times.size, -1)
    a = a - a.mean(axis=0, keepdims=True)
    sign = np.signbit(a)
    crossings = np.count_nonzero(sign[1:] != sign[:-1], axis=0)
    measured = float(np.mean(2.0 * duration / crossings))

    f_lo = float(spectrum.dispersion_omega(2.0 * np.pi / tile.size) / (2.0 * np.pi))
    f_hi = float(spectrum.dispersion_omega(tile.k_nyquist) / (2.0 * np.pi))
    banded = moments.zero_crossing_period(u10, fetch, gamma, f_lo=f_lo, f_hi=f_hi)
    full = moments.zero_crossing_period(u10, fetch, gamma)
    rel = abs(measured - banded) / banded

    record("2", "Tz from zero crossings", measured, banded, 0.10, unit="s",
           note=f"{probes.size**2} probe points, {duration:.0f} s at {1 / dt:.0f} Hz, "
                f"{crossings.sum()} crossings. Theory banded to k in "
                f"[{2 * np.pi / tile.size:.3f}, {tile.k_nyquist:.2f}] rad/m. Against the "
                f"untruncated Tz of {full:.3f} s the error would be "
                f"{100 * abs(measured - full) / full:.1f}%.",
           passed=rel < 0.10)
    assert rel < 0.10


# ---------------------------------------------------------------------------
# Tiling and sampling
# ---------------------------------------------------------------------------


def test_cubic_sampling_beats_bilinear(record, tileset, tileset_fields, sample_points):
    """Bilinear interpolation destroys top-octave variance; cubic mostly does not."""
    x, y = sample_points
    grid = tileset.hs()
    hs_cubic = 4.0 * float(np.std(tileset.sample(x, y, 0.0, fields=tileset_fields, order=3).h))

    # Sample through the standalone bilinear routine rather than `order=1`, so
    # the two implementations are compared rather than two flags on one of them.
    # A single tile is enough: the loss is a property of the interpolator.
    tile, field = tileset.tiles[0], tileset_fields[0]
    lin = tiling.sample_bilinear_periodic(field.h, x, y, tile.size)
    cub = tiling.sample_periodic(field.h, x, y, tile.size, order=3)
    tile_loss = 1.0 - float(np.std(lin)) / float(np.std(cub))

    hs_linear = 4.0 * float(np.std(tileset.sample(x, y, 0.0, fields=tileset_fields, order=1).h))

    loss_cubic = 1.0 - hs_cubic / grid
    loss_linear = 1.0 - hs_linear / grid

    record("2", "Hs loss, cubic sampling (order=3)", loss_cubic, 0.0, 0.03,
           note=f"Bilinear (order=1) loses {100 * loss_linear:.1f}% for comparison, "
                f"and the standalone bilinear routine loses "
                f"{100 * tile_loss:.1f}% of one tile's standard deviation. "
                f"At the Nyquist, bilinear retains only 1/3 of the power.",
           passed=loss_cubic < 0.03)
    assert loss_cubic < 0.03
    assert loss_cubic < loss_linear
    assert tile_loss > 0.0, "the standalone bilinear routine should also lose power"


def test_band_edges_reject_unrepresentable_bands(cfg):
    """A band falling outside its tile's resolvable range raises, not clips.

    Only the *lower* guard is reachable. Because `k_ref` is defined as the
    minimum Nyquist across tiles, an interior edge `frac * k_ref` can never
    exceed the Nyquist of the tile carrying it, and the topmost band is pinned to
    its own tile's Nyquist by construction -- so the upper guard in
    `band_edges` cannot fire as `k_ref` is currently defined. It is left in place
    as a tripwire against a future change to that definition.
    """
    from pywave.config import TileConfig

    good = tiling.band_edges(cfg.surface.tiles)
    assert len(good) == len(cfg.surface.tiles)
    for (lo, hi), tc in zip(good, cfg.surface.tiles):
        assert hi <= tc.k_nyquist * (1 + 1e-12)
        assert lo < hi

    # A tile too small to reach down to the band it was handed: k_min = 6.28
    # rad/m, but the band it is assigned starts at 0.1 * 25.13 = 2.51 rad/m.
    bad = (
        TileConfig(size=64.0, n=512, band=(0.0, 0.1)),
        TileConfig(size=1.0, n=64, band=(0.1, 1.0)),
    )
    with pytest.raises(ValueError, match="but its band starts at"):
        tiling.band_edges(bad)


def test_tile_rotations_are_deterministic_and_spread():
    """Rotations come from the tile index, not the RNG."""
    assert tiling.tile_rotations(3) == tiling.tile_rotations(3)
    r = np.array(tiling.tile_rotations(8))
    assert np.all(np.diff(np.sort(r)) > 0.1)


def test_composite_rotates_vectors_back_to_world(cfg, scene):
    """Slopes must come back out of the tile frame.

    A tile evaluated with rotation phi and sampled through the composite must
    give the same world-frame slope as the same tile at rotation 0. Forgetting
    the back-rotation leaves `mss` correct but points the anisotropy the wrong
    way -- invisible to every scalar check.
    """
    u10, fetch, _, theta_wind = scene
    x = np.linspace(0.0, 40.0, 512)
    y = np.full_like(x, 7.0)

    plain = surface.WaveTile.build(64.0, 256, u10, fetch, theta_wind, seed=7, rotation=0.0)
    turned = surface.WaveTile.build(64.0, 256, u10, fetch, theta_wind, seed=7,
                                    rotation=0.7)

    a = tiling.composite_surface([plain], x, y, 0.0)
    b = tiling.composite_surface([turned], x, y, 0.0)

    # Same wind, same seed, same band: the directional statistics must agree even
    # though the underlying lattices differ.
    ang_a = np.arctan2(np.mean(a.slope_y**2), np.mean(a.slope_x**2))
    ang_b = np.arctan2(np.mean(b.slope_y**2), np.mean(b.slope_x**2))
    assert abs(ang_a - ang_b) < 0.25


# ---------------------------------------------------------------------------
# Tile sizing -- the failure mode that leaves every other number correct
# ---------------------------------------------------------------------------


def test_every_shipped_scene_has_tiles_sized_for_its_own_sea(record):
    """Tile sizes are not a scene-independent constant, and drift is silent.

    Reusing one tile set across scenes passes every existing check: `Hs` is
    right, the bands still sum, the LOD invariant still closes. What quietly
    stops working is the *reason* the bands are disjoint -- per-band shoaling
    and refraction. Push the first band edge far above `k_p` and band 1
    swallows the spectrum, so three FFTs a frame compute one representative
    frequency between them.

    Measured before this was checked: `straits_crop` inherited the test lake's
    64/37/23 m tiles against a peak wavelength eight times longer, putting its
    first edge at 16.4 k_p and 99.7% of the variance in one band.

    Configs are **discovered, not listed**. An earlier version of this test
    iterated a hardcoded triple, which is how `straits.yaml` kept the same
    inherited 64/37/23 m tiles -- at 10.3 k_p, 99.3% in band 1 -- for as long
    as it did. A test named "every shipped scene" that enumerates three of four
    scenes will always drift back out of date the moment a config is added.
    """
    from pywave import load_config

    configs = sorted((REPO_ROOT / "configs").glob("*.yaml"))
    assert configs, "no configs discovered -- the glob is wrong, not the scenes"

    worst_name, worst_share, rows = "", 0.0, []
    for path in configs:
        name = path.stem
        cfg = load_config(path)
        s = tiling.TileSet.build(cfg).sizing()
        rows.append(f"{name} {s.first_edge_over_k_p:.1f} k_p "
                    f"({s.band_shares[0]:.1%} in band 1)")
        if s.band_shares[0] > worst_share:
            worst_name, worst_share = name, s.band_shares[0]
        assert not s.notes, f"{name}: {' '.join(s.notes)}"

    record("2", "worst band-1 variance share across shipped scenes", worst_share,
           tol=0.95, unit="",
           note=f"{'; '.join(rows)}. Above 0.95 the disjoint bands cost three "
                f"FFTs a frame and buy nothing, because every frequency-"
                f"dependent nearshore effect collapses to one representative "
                f"frequency. Worst is {worst_name}.",
           passed=worst_share < 0.95)
    assert worst_share < 0.95


def test_resizing_tiles_redistributes_energy_without_changing_it(record):
    """Moving the band edges must not move `Hs`, `k_max` or resolved `mss`.

    This is what makes a resize safe to apply to a scene that is already
    validated: the split across bands changes, the sea does not. If any of
    these moved, the resize would be a different sea wearing the same config.
    """
    import dataclasses

    from pywave import load_config

    cfg = load_config(REPO_ROOT / "configs" / "straits_crop.yaml")
    good = tiling.TileSet.build(cfg)

    # The old, badly sized set: the test lake's tiles on a 13.6 m sea.
    stale = tuple(dataclasses.replace(t, size=s, n=n) for t, (s, n) in
                  zip(cfg.surface.tiles, ((64.0, 512), (37.0, 256), (23.0, 256))))
    bad = tiling.TileSet.build(
        dataclasses.replace(cfg, surface=dataclasses.replace(cfg.surface,
                                                             tiles=stale)))

    d_hs = abs(good.hs() / bad.hs() - 1)
    d_mss = abs(good.mss() / bad.mss() - 1)
    assert good.k_max == pytest.approx(bad.k_max, rel=1e-12)
    record("2", "Hs shift from resizing tiles", d_hs, reference=0.0, tol=1e-3,
           unit="",
           note=f"band 1 share moves {bad.sizing().band_shares[0]:.1%} -> "
                f"{good.sizing().band_shares[0]:.1%} while Hs moves {d_hs:.2e} "
                f"and resolved mss {d_mss:.2e}, with k_max identical. A resize "
                f"is a redistribution, not a different sea.",
           passed=d_hs < 1e-3 and d_mss < 1e-3)
    assert d_hs < 1e-3 and d_mss < 1e-3


# ---------------------------------------------------------------------------
# Deriving tile sizes from lambda_p
# ---------------------------------------------------------------------------


def test_derived_sizes_put_the_first_edge_on_target_at_every_sea_state(record):
    """The point of the derivation: `first_edge / k_p` stops depending on the sea.

    With sizes fixed, the first band edge is pinned in absolute rad/m by the
    tile geometry while `k_p` slides underneath it -- 2.1 k_p on the test lake
    and 10.3 k_p on `straits`, from the same tile block. Scaling `L` with
    `lambda_p` at fixed `n` cancels the sea-state term out of the ratio
    entirely, so it should hold still to floating-point across the whole
    usable range rather than merely staying inside a tolerance band.
    """
    from pywave.config import TileConfig

    pinned = (23.0, 256)
    lambdas = [1.05, 1.5, 1.705, 3.0, 5.0, 8.48, 13.59, 20.0, 35.3, 50.0,
               75.0, 90.0]
    edges = []
    for lam in lambdas:
        sizes = tiling.derive_tile_sizes(lam, (512, 256), (0.35, 0.7),
                                         pinned=[pinned])
        tiles = (TileConfig(size=sizes[0], n=512, band=(0.0, 0.35)),
                 TileConfig(size=sizes[1], n=256, band=(0.35, 0.7)),
                 TileConfig(size=pinned[0], n=pinned[1], band=(0.7, 1.0)))
        # band_edges is the validator: it raises if any band fails to fit the
        # tile carrying it, so reaching this line is itself part of the check.
        edges.append(tiling.band_edges(tiles)[0][1] / (2.0 * np.pi / lam))

    spread = max(edges) - min(edges)
    record("2", "spread in first band edge over a 86x range of lambda_p", spread,
           reference=0.0, tol=1e-6, unit="k_p",
           note=f"lambda_p {min(lambdas)}-{max(lambdas)} m, all derived sizes "
                f"land the first edge at {np.mean(edges):.6f} k_p. Held fixed "
                f"instead, the same tiles give 2.1 k_p on the test lake and "
                f"10.3 k_p on straits.",
           passed=spread < 1e-6)
    assert spread < 1e-6
    assert all(abs(e - 2.0) < 1e-9 for e in edges)


def test_derivation_refuses_a_pinned_tile_that_would_steal_k_ref():
    """A pinned tile below the derived Nyquist is the silent failure again.

    `band_edges` takes `k_ref` from the *minimum* Nyquist. Shrink the sea far
    enough and the derived tiles overtake the pinned one, which then becomes
    `k_ref` and puts the first edge wherever it likes -- measured at 0.97 k_p
    for a 0.5 m sea against a 23 m top tile, with every other number still
    correct. Nothing downstream raises, so this has to.
    """
    ok = tiling.derive_tile_sizes(1.5, (512, 256), (0.35, 0.7),
                                  pinned=[(23.0, 256)])
    assert ok[0] > 0

    with pytest.raises(ValueError, match="would silently become k_ref"):
        tiling.derive_tile_sizes(0.5, (512, 256), (0.35, 0.7),
                                 pinned=[(23.0, 256)])

    # Without `pinned` there is nothing to check against, so it stays silent --
    # which is exactly why the config path always passes it.
    tiling.derive_tile_sizes(0.5, (512, 256), (0.35, 0.7))


def test_derivation_leaves_the_top_tile_alone(record):
    """`k_max` must not move with the sea state. See TileSizing and 3a.

    Deriving the top tile as well fixes the bands and wrecks the resolved
    slope variance: it is the tile whose Nyquist *is* `k_max`, so growing it at
    fixed `n` drags the mesh/BSDF handoff down with the sea.
    """
    sizes_small = tiling.derive_tile_sizes(2.0, (512, 256), (0.35, 0.7),
                                           pinned=[(23.0, 256)])
    sizes_big = tiling.derive_tile_sizes(40.0, (512, 256), (0.35, 0.7),
                                         pinned=[(23.0, 256)])
    assert len(sizes_small) == 2 and len(sizes_big) == 2
    # A 20x change in sea state scales every derived tile by 20x exactly...
    for a, b in zip(sizes_small, sizes_big):
        assert b / a == pytest.approx(20.0, rel=1e-12)
    # ...and the top tile, not being returned, cannot have moved at all.
    record("2", "derived tiles returned per 3-tile set", len(sizes_small),
           reference=2, tol=0, unit="",
           note="The top tile sets k_max, which is a rendering decision. "
                "Deriving all three on straits dropped k_max 35.0 -> 9.1 rad/m "
                "and 36% of the resolved mss; leaving it pinned moved mss by "
                "1.6e-03.",
           passed=len(sizes_small) == 2)


def test_derived_sizes_do_not_share_a_short_joint_period(record):
    """Derived sizes must not come out in a low-order rational ratio.

    Rotations are the real defence -- a single world translation cannot be a
    lattice vector of two lattices separated by an irrational angle, and the
    golden angle guarantees that. Sizes are the second line, and this pins it
    so a future default cannot quietly make the ratio 2 or 3.
    """
    from fractions import Fraction

    sizes = tiling.derive_tile_sizes(1.705, (512, 256), (0.35, 0.7),
                                     pinned=[(23.0, 256)])
    ratio = Fraction(sizes[0] / sizes[1]).limit_denominator(1000)
    period = ratio.denominator          # in units of the larger tile

    record("2", "joint period of the derived tile pair", period,
           tol=100, unit="x L0",
           note=f"L0/L1 = {sizes[0] / sizes[1]:.4f} = {ratio}, so the pair "
                f"shares a period only after {period} of the larger tile. The "
                f"hand-tuned configs sit at 719x (coastal_bay) and 37x "
                f"(test_lake). Golden-angle rotations remove even this.",
           passed=period >= 100)
    assert period >= 100


def _lake_with_tiles(tmp_path, tiles_block, speed=None):
    """The shipped test lake with its `tiles:` block swapped out."""
    src = (REPO_ROOT / "configs" / "test_lake.yaml").read_text(encoding="utf-8")
    old = ("    - {size: 64.0, n: 512, band: [0.0, 0.35]}\n"
           "    - {size: 37.0, n: 256, band: [0.35, 0.7]}\n"
           "    - {size: 23.0, n: 256, band: [0.7, 1.0]}")
    assert old in src, "test_lake.yaml tiles block moved; update this helper"
    txt = src.replace(old, tiles_block)
    if speed is not None:
        txt = txt.replace("speed: 5.0", f"speed: {speed}")
    p = tmp_path / "probe.yaml"
    p.write_text(txt, encoding="utf-8")
    return p


AUTO_TILES = ("    - {size: auto, n: 512, band: [0.0, 0.35]}\n"
              "    - {size: auto, n: 256, band: [0.35, 0.7]}\n"
              "    - {size: 23.0, n: 256, band: [0.7, 1.0]}")


def test_auto_sized_config_holds_its_band_split_across_a_sweep(record, tmp_path):
    """What a sweep needs: change the sea, keep the band structure.

    This is the whole point of the feature. The same config file, run at three
    wind speeds, must put its first band edge in the same place relative to the
    peak -- and must not move `k_max`, because that is a rendering decision and
    the mesh was built for it.
    """
    from pywave import load_config

    rows, edges, k_maxes = [], [], []
    for speed in (5.0, 9.0, 14.0):
        cfg = load_config(_lake_with_tiles(tmp_path, AUTO_TILES, speed=speed))
        ts = tiling.TileSet.build(cfg)
        s = ts.sizing()
        edges.append(s.first_edge_over_k_p)
        k_maxes.append(ts.k_max)
        rows.append(f"U10 {speed:g} m/s -> lambda_p {cfg.lambda_p:.2f} m, "
                    f"L0 {cfg.surface.tiles[0].size:.0f} m")
        assert not s.notes, f"U10 {speed}: {' '.join(s.notes)}"

    spread = max(edges) - min(edges)
    record("2", "first-edge spread across an auto-sized sweep", spread,
           reference=0.0, tol=1e-9, unit="k_p",
           note=f"{'; '.join(rows)}. The tiles scale with the sea; the band "
                f"split and k_max ({k_maxes[0]:.2f} rad/m) do not move.",
           passed=spread < 1e-9)
    assert spread < 1e-9
    assert len(set(k_maxes)) == 1, "k_max moved with the sea state"


def test_explicit_tile_sizes_are_left_exactly_as_written(tmp_path):
    """A config that pins every size must be untouched by the auto machinery.

    This is what keeps every existing scene reproducible: the derivation is
    opt-in per tile, and a config with no `auto` never reaches it.
    """
    from pywave import load_config

    for name in ("test_lake", "coastal_bay", "straits", "straits_crop"):
        cfg = load_config(REPO_ROOT / "configs" / f"{name}.yaml")
        raw = yaml.safe_load(
            (REPO_ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
        written = [float(t["size"]) for t in raw["surface"]["tiles"]]
        assert [t.size for t in cfg.surface.tiles] == written, name


def test_auto_tiles_are_rejected_where_they_cannot_be_derived(tmp_path):
    """The two arrangements the derivation cannot express, refused by name."""
    from pywave import load_config

    all_auto = AUTO_TILES.replace(
        "{size: 23.0, n: 256, band: [0.7, 1.0]}",
        "{size: auto, n: 256, band: [0.7, 1.0]}")
    with pytest.raises(ValueError, match="highest-band tile cannot be"):
        load_config(_lake_with_tiles(tmp_path, all_auto))

    auto_above_pinned = ("    - {size: 64.0, n: 512, band: [0.0, 0.35]}\n"
                         "    - {size: auto, n: 256, band: [0.35, 0.7]}\n"
                         "    - {size: 23.0, n: 256, band: [0.7, 1.0]}")
    with pytest.raises(ValueError, match="must be the lowest bands"):
        load_config(_lake_with_tiles(tmp_path, auto_above_pinned))
