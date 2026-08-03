"""pywave -- spectral water surface synthesis for littoral EO/IR scene generation.

The spectrum is the single source of truth.  Wave height, slope statistics,
BSDF roughness and LOD behaviour are all derived from one ``S(k)``.  Nothing is
tuned independently; if two things disagree, the spectrum wins and the other one
has a bug.

Read ``pywave.constants`` before writing code against this package -- it carries
the coordinate, unit, sign and FFT conventions that everything here assumes.

Phase status
------------
Phase 1 (spectrum, moments)        implemented
Phase 2 (surface, tiling)          implemented
Phase 3 (validation suite)         NOT started -- the Gate 1 and Gate 2 checks
                                   named throughout these docstrings are
                                   specified in the cookbook but are not yet
                                   automated.  ``tests/`` and
                                   ``docs/validation_report.md`` do not exist.
Phase 4+ (nearshore, mesh, BSDF)   not started

See ``docs/users_guide.md`` for the user-facing documentation.
"""

from __future__ import annotations

from . import constants, moments, spectrum, surface, tiling
from .config import Config, load_config

__version__ = "0.1.0"

__all__ = [
    "constants",
    "spectrum",
    "moments",
    "surface",
    "tiling",
    "Config",
    "load_config",
    "__version__",
]
