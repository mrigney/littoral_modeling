"""PHASE 3 -- Gate 3: reproducibility and regression.

Determinism is not a convenience here, it is the V&V property the whole design
is built around: same seed, same `t`, same surface, on any machine, forever. A
scenario can be re-run years later and produce the identical sea, and any frame
can be computed on any node without coordination.

The cookbook's "two machines agree to 1e-12" is what the committed baseline
tests: it was generated on one machine and is checked on every machine that runs
the suite. Within a single process the requirement is stronger -- bitwise.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import make_baseline as mb
from pywave import spectrum, surface, tiling

pytestmark = pytest.mark.gate3

BASELINE_DIR = mb.BASELINE_DIR
TOL = 1e-12


@pytest.fixture(scope="module")
def metadata():
    path = BASELINE_DIR / "metadata.json"
    if not path.exists():
        pytest.fail("baseline metadata missing; run `python tests/make_baseline.py`")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_surface_bitwise(record, cfg):
    """Two independently built tile sets agree bit for bit."""
    a = tiling.TileSet.build(cfg)
    b = tiling.TileSet.build(cfg)

    worst = 0.0
    for ta, tb in zip(a.tiles, b.tiles):
        assert np.array_equal(ta.h0, tb.h0)
        fa, fb = ta.evaluate(2.5), tb.evaluate(2.5)
        for x, y in ((fa.h, fb.h), (fa.slope_x, fb.slope_x), (fa.dx_disp, fb.dx_disp)):
            worst = max(worst, float(np.max(np.abs(x - y))))

    record("3", "max |difference| between two builds at the same seed", worst, 0.0, 0.0,
           note="Bitwise identical within a process. `default_rng` is used "
                "throughout; the legacy global RNG would not guarantee this.",
           passed=worst == 0.0)
    assert worst == 0.0


def test_different_seed_gives_a_different_surface(record, cfg):
    """The seed actually reaches the surface -- and changes only the phases."""
    from dataclasses import replace

    a = tiling.TileSet.build(cfg)
    b = tiling.TileSet.build(replace(cfg, spectrum=replace(cfg.spectrum, seed=987654321)))

    ha = a.tiles[0].evaluate(2.5).h
    hb = b.tiles[0].evaluate(2.5).h
    assert not np.allclose(ha, hb)

    # Same spectrum, so the same variance to sampling error.
    rel = abs(np.std(ha) - np.std(hb)) / np.std(ha)
    record("3", "std(h) change from a different seed", rel, 0.0, 0.10,
           note="A new seed redraws the phases, not the spectrum, so the "
                "variance must be unchanged up to sampling error.",
           passed=rel < 0.10)
    assert rel < 0.10


def test_evaluation_is_stateless(record, cfg):
    """`evaluate(t)` depends on `t` alone -- no accumulated state.

    Frame 8000 must cost what frame 1 costs and must not drift. Evaluating a
    sequence of times before the one under test would perturb the answer if any
    state were being carried.
    """
    tile = tiling.TileSet.build(cfg).tiles[0]

    direct = tile.evaluate(37.5).h
    for t in (0.0, 1.0, 2.0, 5.0, 100.0, 8000.0):
        tile.evaluate(t)
    after = tile.evaluate(37.5).h

    worst = float(np.max(np.abs(direct - after)))
    record("3", "max |difference| evaluating t = 37.5 s cold vs after a sequence",
           worst, 0.0, 0.0, unit="m", passed=worst == 0.0)
    assert worst == 0.0


def test_far_future_frame_is_well_conditioned(record, cfg):
    """A frame 8000 s in has the same statistics as frame 0.

    Phase accumulates as `omega * t`, so at large `t` the argument to `exp` is
    large; this checks nothing degrades.
    """
    tile = tiling.TileSet.build(cfg).tiles[0]
    early = tile.evaluate(0.0).h
    late = tile.evaluate(8000.0).h

    rel = abs(np.std(late) - np.std(early)) / np.std(early)
    record("3", "std(h) at t = 8000 s vs t = 0", float(np.std(late)), float(np.std(early)),
           0.05, unit="m", note="Pure function of t; no integration, so no drift.",
           passed=rel < 0.05)
    assert rel < 0.05
    assert np.all(np.isfinite(late))


def test_seed_does_not_reach_tile_rotations(cfg):
    """Rotations derive from the tile index, not the RNG.

    The seed is used in `initial_amplitudes` and nowhere else; drawing rotations
    from it would make the surface depend on the seed through two paths.
    """
    from dataclasses import replace

    a = tiling.TileSet.build(cfg)
    b = tiling.TileSet.build(replace(cfg, spectrum=replace(cfg.spectrum, seed=13)))
    assert [t.rotation for t in a.tiles] == [t.rotation for t in b.tiles]


# ---------------------------------------------------------------------------
# Regression against the committed baseline
# ---------------------------------------------------------------------------


def test_tile_matches_committed_baseline(record, metadata):
    """One pinned tile reproduces the committed reference to 1e-12.

    This is the cross-machine check. If it fails, a convention changed: the FFT
    normalisation, a sign in the time evolution or the displacement, or the
    `flip_k` index mapping. None of those are visible in the variance or the
    spectrum, which is exactly why this file exists.
    """
    path = BASELINE_DIR / "tile.npy"
    if not path.exists():
        pytest.fail("baseline missing; run `python tests/make_baseline.py`")

    expected = np.load(path)
    actual = mb.build_tile_baseline()

    assert actual.shape == expected.shape
    worst = float(np.max(np.abs(actual - expected)))
    digest = hashlib.sha256(actual.tobytes()).hexdigest()

    record("3", "max |difference| vs committed tile baseline", worst, 0.0, TOL, unit="m",
           note=f"Baseline generated {metadata['generated']} at git "
                f"`{metadata['git_sha'][:12]}` on {metadata['platform']}, "
                f"numpy {metadata['numpy']}. Fields: "
                f"{', '.join(metadata['tile']['fields'])} at t = {metadata['scene']['t']} s.",
           passed=worst < TOL)
    record("3", "tile baseline sha256 match",
           "match" if digest == metadata["tile"]["sha256"] else "differs",
           "match", note="Bitwise on this platform; the 1e-12 tolerance above is "
                         "what portability actually requires.",
           passed=True)
    assert worst < TOL


def test_composite_matches_committed_baseline(record, metadata):
    """The full pipeline -- tiling, rotation, sampling -- reproduces its reference.

    Broader than the tile baseline: it covers band assignment, the golden-angle
    rotations, the world/tile frame round trip and the cubic interpolator.
    """
    path = BASELINE_DIR / "composite.npy"
    if not path.exists():
        pytest.fail("baseline missing; run `python tests/make_baseline.py`")

    expected = np.load(path)
    actual = mb.build_composite_baseline()

    assert actual.shape == expected.shape
    worst = float(np.max(np.abs(actual - expected)))

    record("3", "max |difference| vs committed composite baseline", worst, 0.0, TOL,
           note=f"{metadata['composite']['n_probes']} fixed world points, all five "
                f"fields ({', '.join(metadata['composite']['fields'])}).",
           passed=worst < TOL)
    assert worst < TOL


def test_baseline_metadata_matches_the_arrays(metadata):
    """The recorded hashes describe the committed files."""
    for key, name in (("tile", "tile.npy"), ("composite", "composite.npy")):
        arr = np.load(BASELINE_DIR / name)
        assert list(arr.shape) == metadata[key]["shape"]
        assert hashlib.sha256(arr.tobytes()).hexdigest() == metadata[key]["sha256"], (
            f"{name} does not match the sha256 in metadata.json -- the baseline was "
            f"regenerated without updating its metadata, or vice versa"
        )


# ---------------------------------------------------------------------------
# Config contract
# ---------------------------------------------------------------------------


def test_degrees_are_converted_exactly_once(cfg):
    """Config carries radians; the YAML carried degrees."""
    assert cfg.wind.direction_rad == pytest.approx(np.radians(45.0))
    assert cfg.wind.direction_deg == pytest.approx(45.0)


def test_invalid_configs_are_rejected():
    """Validation happens at construction, not three modules downstream."""
    from pywave.config import SurfaceConfig, TileConfig, WindConfig

    with pytest.raises(ValueError, match="power of two"):
        TileConfig(size=64.0, n=100, band=(0.0, 1.0))
    with pytest.raises(ValueError, match="wind.speed must be positive"):
        WindConfig(speed=0.0, direction_rad=0.0, fetch=1000.0)
    with pytest.raises(ValueError, match="contiguous and disjoint"):
        SurfaceConfig(tiles=(TileConfig(size=64.0, n=64, band=(0.0, 0.3)),
                             TileConfig(size=32.0, n=64, band=(0.5, 1.0))))
    with pytest.raises(ValueError, match="must span"):
        SurfaceConfig(tiles=(TileConfig(size=64.0, n=64, band=(0.0, 0.8)),))


def test_spectrum_model_guard():
    """An unimplemented spectrum model raises rather than falling back."""
    from pywave.config import SpectrumConfig

    with pytest.raises(ValueError, match="only 'jonswap'"):
        SpectrumConfig(model="pierson-moskowitz")


def test_deep_water_dispersion_is_the_tanh_limit():
    """`depth=None` agrees with a large finite depth."""
    k = np.geomspace(0.01, 100.0, 200)
    deep = spectrum.dispersion_omega(k)
    finite = spectrum.dispersion_omega(k, 10_000.0)
    assert np.allclose(deep, finite, rtol=1e-12)


def test_surface_field_addition_is_componentwise():
    """`SurfaceField.__add__` sums every channel -- used by the composite."""
    a = surface.SurfaceField(*[np.full((4, 4), float(i)) for i in range(1, 6)])
    b = surface.SurfaceField(*[np.full((4, 4), float(i)) for i in range(1, 6)])
    c = a + b
    for name in ("h", "dx_disp", "dy_disp", "slope_x", "slope_y"):
        assert np.all(getattr(c, name) == 2 * getattr(a, name))
