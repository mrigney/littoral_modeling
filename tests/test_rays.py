"""PHASE 5b -- Gate 5b: refraction that survives a complex coastline.

The design is in docs/phase5b_refraction.md. This file is written ahead of the
implementation: 5b.6 pins the defect that motivates the whole phase, and the
rest fill in as `pywave/rays.py` grows.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pywave import nearshore
from pywave.bathymetry import Bathymetry

pytestmark = pytest.mark.gate5b


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
