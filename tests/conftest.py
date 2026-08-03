"""PHASE 3 -- shared fixtures and the validation-report collector.

Tests here do not merely pass or fail: each one records the number it measured
and the reference it was compared against, and the session writes
``docs/validation_report.md`` from those records.  That file is the V&V
artifact; the green checkmarks are a by-product.  A reviewer wants to read
"realised Hs = 0.0839 m vs 0.0853 m, -1.6%", which is a statement about the
physics, rather than "8 passed", which is a statement about pytest.

The report is regenerated on every session.  If you run a subset of the suite
the report will cover only that subset, and the header says so.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from pywave import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "test_lake.yaml"
REPORT_PATH = REPO_ROOT / "docs" / "validation_report.md"
BASELINE_DIR = Path(__file__).resolve().parent / "baseline"


# ---------------------------------------------------------------------------
# Recorded checks
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """One recorded quantity, with the reference it was judged against."""

    gate: str
    name: str
    measured: float | str
    reference: float | str | None = None
    tol: float | None = None
    unit: str = ""
    note: str = ""
    passed: bool = True

    @property
    def rel_error(self) -> float | None:
        if not isinstance(self.measured, (int, float)):
            return None
        if not isinstance(self.reference, (int, float)) or self.reference == 0:
            return None
        return abs(self.measured - self.reference) / abs(self.reference)


@dataclass
class Recorder:
    checks: list[Check] = field(default_factory=list)

    def __call__(self, gate, name, measured, reference=None, tol=None,
                 unit="", note="", passed=True) -> Check:
        c = Check(gate, name, measured, reference, tol, unit, note, passed)
        self.checks.append(c)
        return c


_RECORDER = Recorder()


@pytest.fixture(scope="session")
def recorder() -> Recorder:
    return _RECORDER


@pytest.fixture
def record(recorder):
    """Record a measured quantity into the validation report."""
    return recorder


# ---------------------------------------------------------------------------
# Scene fixtures -- built once, they are not cheap
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def scene(cfg):
    """The bundle every test wants: ``(u10, fetch, gamma, theta_wind)``."""
    return (cfg.wind.speed, cfg.wind.fetch, cfg.spectrum.gamma, cfg.wind.direction_rad)


@pytest.fixture(scope="session")
def tileset(cfg):
    from pywave import tiling

    return tiling.TileSet.build(cfg)


@pytest.fixture(scope="session")
def tileset_fields(tileset):
    """Tile grids evaluated at t = 0, shared across tests that only need t = 0."""
    return tileset.evaluate_grids(0.0)


@pytest.fixture(scope="session")
def beach():
    """Straight Dean beach, 1 m posts. Contours parallel -- Snell is exact here."""
    from pywave.bathymetry import Bathymetry

    return Bathymetry.dean_beach()


@pytest.fixture(scope="session")
def fine_beach():
    """Dean beach at 0.25 m posts.

    The surf zone is about 1 m wide, so a 1 m grid cannot resolve it -- this is
    exactly the terracing problem cookbook section 4.4 warns about, and the
    reason it recommends refining to 0.25 m inside the nearshore band. Foam and
    surf-width checks need this grid; everything else does not.
    """
    from pywave.bathymetry import Bathymetry

    return Bathymetry.dean_beach(nx=256, ny=512, dx=0.25, shoreline_y=100.0)


@pytest.fixture(scope="session")
def onshore_cfg(cfg):
    """The test lake with the wind blowing straight onshore (+Y).

    Normal incidence makes the refraction coefficient exactly 1, which isolates
    shoaling from refraction. With the shipped 45 degree wind the two very
    nearly cancel, so a test that did not control for this would conclude that
    shoaling does nothing.
    """
    from dataclasses import replace

    return replace(cfg, wind=replace(cfg.wind, direction_rad=np.radians(90.0)))


@pytest.fixture(scope="session")
def onshore_tileset(onshore_cfg):
    from pywave import tiling

    return tiling.TileSet.build(onshore_cfg)


@pytest.fixture(scope="session")
def sample_points():
    """A fixed pseudo-random scatter of world points, for realised statistics."""
    rng = np.random.default_rng(12345)
    return rng.uniform(0.0, 500.0, 300_000), rng.uniform(0.0, 500.0, 300_000)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{sha}{' (working tree dirty)' if dirty else ''}" if sha else "unknown"
    except Exception:
        return "unknown"


def _fmt(v, unit: str = "") -> str:
    if v is None:
        return "--"
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    a = abs(v)
    if v == 0:
        s = "0"
    elif a < 1e-3 or a >= 1e5:
        s = f"{v:.4e}"
    elif a < 1:
        s = f"{v:.6f}".rstrip("0").rstrip(".")
    else:
        s = f"{v:.4f}".rstrip("0").rstrip(".")
    return f"{s} {unit}".strip()


_GATE_TITLES = {
    "1": "Gate 1 -- spectrum and moments",
    "2": "Gate 2 -- FFT surface synthesis",
    "3": "Gate 3 -- reproducibility and regression",
    "5": "Gate 5 -- nearshore transformation",
}


def pytest_sessionfinish(session, exitstatus):
    checks = _RECORDER.checks
    if not checks:
        return

    import platform

    import scipy

    # Only a *selection* makes the report partial. Plain flags like
    # `--durations=8` do not change which checks ran.
    ran = getattr(session.config, "invocation_params", None)
    raw = list(ran.args) if ran else []
    selectors = [a for a in raw
                 if not a.startswith("-") or a.startswith(("-k", "-m", "--deselect"))]
    args = " ".join(raw)
    partial = bool(selectors)

    lines: list[str] = []
    lines.append("# Validation report -- `pywave`")
    lines.append("")
    lines.append(
        "**This file is generated by `pytest`. Do not edit it by hand.** It is "
        "regenerated in full on every session; to refresh it, run `pytest`."
    )
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| generated | {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC |")
    lines.append(f"| git_sha | `{_git_sha()}` |")
    lines.append(f"| scene | `{CONFIG_PATH.relative_to(REPO_ROOT).as_posix()}` |")
    lines.append(f"| python | {platform.python_version()} ({platform.system()} {platform.machine()}) |")
    lines.append(f"| numpy / scipy | {np.__version__} / {scipy.__version__} |")
    lines.append(f"| checks recorded | {len(checks)} |")
    lines.append(f"| exit status | {'PASS' if exitstatus == 0 else f'FAIL ({exitstatus})'} |")
    lines.append("")
    if partial:
        lines.append(
            f"> **Partial run.** pytest was invoked with `{args}`, so this report "
            f"covers only the checks that selection executed."
        )
        lines.append("")

    lines.append(
        "Every number below was measured by the test suite against the "
        "implementation in this commit. Tolerances are the gate criteria from "
        "`littoral-water-implementation-cookbook.md`, except where a deviation is "
        "recorded in [Gate deviations](#gate-deviations)."
    )
    lines.append("")

    for gate in ("1", "2", "3", "5"):
        rows = [c for c in checks if c.gate == gate]
        if not rows:
            continue
        lines.append(f"## {_GATE_TITLES[gate]}")
        lines.append("")
        lines.append("| Check | Measured | Reference | Rel. error | Tolerance | Result |")
        lines.append("|---|---|---|---|---|---|")
        for c in rows:
            rel = c.rel_error
            lines.append(
                f"| {c.name} | {_fmt(c.measured, c.unit)} | {_fmt(c.reference, c.unit)} "
                f"| {'--' if rel is None else f'{rel:.2e}'} "
                f"| {'--' if c.tol is None else f'{c.tol:.1e}'} "
                f"| {'PASS' if c.passed else 'FAIL'} |"
            )
        lines.append("")
        notes = [c for c in rows if c.note]
        if notes:
            lines.append("Notes:")
            lines.append("")
            for c in notes:
                lines.append(f"- **{c.name}** -- {c.note}")
            lines.append("")

    lines.append("## Gate deviations")
    lines.append("")
    lines.append(DEVIATIONS.strip())
    lines.append("")
    lines.append("## Cookbook corrections")
    lines.append("")
    lines.append(CORRECTIONS.strip())
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


DEVIATIONS = """
Two cookbook gate criteria are not met, and cannot be met by a correct
implementation of the model as specified. Both are recorded here rather than
silently reinterpreted, and the suite pins a substitute criterion in each case.

### Gate 1: "`4*sqrt(int S df)` matches closed-form `Hs` within 2%"

Not achievable. JONSWAP supplies two independently fitted power laws that are
not mutually consistent: integrating the spectral form gives `m0 ~ X~^1.10`,
while the dimensionless-energy growth law gives `m0 ~ X~^1.00`. Their ratio is
therefore `~ X~^0.05` in `Hs`, with the crossover near `X~ ~ 120` -- so the two
numbers agree at exactly one fetch and nowhere else. For the test lake
(`X~ = 392`) the gap is +6.1%.

Rescaling the spectrum to close the gap would change `alpha`, hence the Phillips
tail, hence mean square slope and every BSDF roughness downstream -- the
independent tuning the design rule forbids.

**Substituted criterion**, which is strictly stronger: pin the *relation*
`hs_spectral / fetch_limited_hs = 0.78696 * X~^0.05` across four decades of
fetch. A normalisation or Jacobian error would show up as a constant offset
(exponent 0); a Jacobian sign or factor error would change the exponent. See
`test_hs_relation_holds_across_fetch`.

### Gate 2: "skewness of `h` in [0, 0.3] with choppiness on"

Not achievable with this model. The construction is *linear in elevation*: the
Gerstner displacement moves sample points horizontally but never changes the set
of elevation values, so it cannot introduce the crest/trough asymmetry that
gives real seas a positive elevation skewness. Measured area-weighted skewness
of the displaced surface is -0.015 at `choppiness = 1.0`, against -0.011 with
the displacement switched off entirely -- so turning choppiness on moves the
skewness by -0.004, in the opposite direction to the gate, and by an amount
indistinguishable from the sampling noise on a single realisation.

This is not a bug in the displacement. Applying the same measurement to a single
steep mode (`ka = 0.47`) recovers +0.508, matching an explicit resampling of the
trochoid to four decimals; the effect is simply negligible once summed over a
broadband spectrum at this sea state, where per-mode steepness is ~0.08.

Positive elevation skewness in a real sea comes from second-order Stokes bound
harmonics, which are not part of a linear spectral model. A second-order
correction would add it, and is not implemented.

**Substituted criterion**: assert the linear surface is Gaussian to tight
tolerance (`|skew| < 0.05`, `|excess kurtosis| < 0.1`), which is what the
amplitude draw and Hermitian symmetry actually predict, and record the displaced
skewness as a reported quantity rather than a pass/fail bound. See
`test_height_distribution_is_gaussian`.
"""


CORRECTIONS = """
Places where the implementation departs from the cookbook's *guidance* (as
opposed to its gate criteria), with the measurement that motivated each.

### Section 5.5: "with a 3 s half-life, 30 frames of spin-up is plenty"

Wrong by more than an order of magnitude, and silently so -- a foam field spun
up for 30 frames looks entirely plausible, it just is not reproducible.

30 frames at 30 fps is **one second**. Against a 3 s half life that leaves
`2^(-1/3)` = **79%** of the discarded initial condition still present. Measured
cold-vs-sequential error at that setting is 2.2% per cell, against the 1% Gate 5
asks for.

The window is set by the half life, not by a frame count:
`T_spin = t_half * log2(1/tol)`. For 0.5% that is 23 s -- 92 steps at the 0.25 s
foam step, or 688 frames at 30 fps. `foam.spinup_steps()` computes it, and
`foam.FoamModel.evaluate` calls it by default rather than accepting a frame
count. Measured error at that setting: 0.61% per cell, 0.12% of peak coverage.

Note also that the foam step need not be the frame step. Advection is
semi-Lagrangian and unconditionally stable, so 0.25 s costs 6x less than 1/30 s
for no visible difference.

### Section 5.3: refraction by depth-weighted blend

The cookbook prescribes blending the wave direction toward the shore normal with
`w = clip(1 - d/d_ref, 0, 1)`, `d_ref ~ 3 lambda_p`. That is implemented as
`nearshore.refraction_angle_blend`, but it is **not** the production path, for
two reasons the tests measure:

* It has no frequency dependence, so every spectral band would turn at the same
  rate. Refraction is dispersive; the tiles carry disjoint bands precisely so
  frequency-dependent effects can be applied per band.
* It reaches full shore-normal alignment at the waterline regardless of the
  incident angle, which Snell does not: at 80 degrees incidence the exact
  answer is still 22 degrees off-normal at the break point.

Production uses `nearshore.refraction_angle`, which solves Snell's law against
the full dispersion relation. Measured disagreement between the two on a test
transect: 38.5 degrees peak, 29.3 degrees mean. Snell's invariant `sin(a)/c` is
conserved to 2.6e-16 across the transect, so the exact path costs nothing in
accuracy and very little in time.

### Section 5.1: "surf zone ~2 m wide"

Not a deviation, a parameter difference, recorded so the numbers reconcile. The
cookbook's 2 m assumes a 5% foreshore slope. The Dean profile used here
(`A = 0.1`, medium sand) is 6.7% at the break point, giving a 0.9 m surf zone
and `xi = 0.30` against the cookbook's 0.23. Both are firmly spilling, and the
runup (2.5 cm) and swash excursion (0.38 m) land inside the cookbook's quoted
ranges.
"""
