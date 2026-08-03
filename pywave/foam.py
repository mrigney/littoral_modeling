"""PHASE 5 -- foam: seeding in the surf band, shoreward advection, exponential decay.

Foam is the one field in this package with frame-to-frame memory, and that is a
problem, because everything else is a pure function of ``t``: same seed, same
time, same answer, on any node, with no coordination.  Losing that would mean
frames could only be generated sequentially.

The fix (cookbook section 5.5, option b) is **bounded spin-up**.  Foam decays
exponentially, so the influence of any initial condition dies with the same half
life.  To evaluate frame ``t``, start from zero at ``t - T_spin`` and integrate
forward.

The spin-up window is set by the *half life*, not by a frame count, and the
cookbook's "30 frames is plenty" is off by more than an order of magnitude: 30
frames at 30 fps is one second against a 3 s half life, leaving ``2^(-1/3)`` =
79% of the initial condition intact.  Reaching the 1% Gate 5 asks for needs
``T_spin = t_half * log2(1/tol)`` = 3 * log2(100) = **20 seconds**.  Getting
this wrong is easy and silent -- the field looks entirely plausible either way,
it just is not reproducible -- so :func:`spinup_steps` derives the count from
the tolerance rather than leaving it to a guess.

The result is a field that has memory over a second or two and no memory at all
beyond the spin-up window, which is both physically right and parallel-safe.
Whether a given frame was computed cold or as part of a sequence cannot be
detected to better than the spin-up residual, and that residual is a stated,
bounded number rather than an accident.

Why foam matters at all when it is sub-pixel
--------------------------------------------
Whitecap coverage at 5 m/s over open water is ~0.1% and is deliberately not
modelled.  Surf-zone foam is different: it is confined to a band about 2 m wide,
so at 1 m GSD it occupies a meaningful fraction of the pixels that contain the
shoreline.  And it is optically nothing like water -- high albedo in EO, and
near-blackbody in LWIR (``eps ~ 0.95-0.98``) against water's strongly angular
emissivity.  A pixel that is 30% foam is radiometrically not a water pixel, so
the *fractional coverage* is what has to be right, not the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FoamModel", "foam_decay_factor", "spinup_residual", "spinup_steps"]


def foam_decay_factor(dt: float, half_life: float) -> float:
    """Multiplicative decay over one step: ``exp(-dt ln2 / t_half)``."""
    if half_life <= 0:
        raise ValueError(f"half_life must be positive, got {half_life}")
    return float(np.exp(-dt * np.log(2.0) / half_life))


def spinup_residual(n_steps: int, dt: float, half_life: float) -> float:
    """Fraction of an initial condition still present after ``n_steps``.

    This is the error bound on treating a cold-started frame as if it had the
    scenario's full history behind it.
    """
    return float(foam_decay_factor(dt, half_life) ** n_steps)


def spinup_steps(tol: float, dt: float, half_life: float) -> int:
    """Steps needed for the discarded history to fall below ``tol``.

    ``n = ceil(t_half * log2(1/tol) / dt)``.  At ``tol = 0.01`` and a 3 s half
    life that is 20 s of simulated time, however many steps that takes -- 80 at
    ``dt = 0.25`` s, or 600 at 30 fps.  Quoting a frame count without saying
    what ``dt`` and ``t_half`` were is meaningless, which is the trap this
    function exists to close.
    """
    if not 0.0 < tol < 1.0:
        raise ValueError(f"tol must be in (0, 1), got {tol}")
    return int(np.ceil(half_life * np.log2(1.0 / tol) / dt))


@dataclass
class FoamModel:
    """Foam coverage on a fixed grid, evaluated by bounded spin-up.

    The grid is the bathymetry grid, so foam is a field over the scene rather
    than over a wave tile -- it lives where the shore is, not where the FFT
    lattice happens to be.

    Parameters
    ----------
    seed_rate : coverage added per second in cells that are breaking [1/s].
        A continuously breaking cell converges to ``seed_rate * t_half / ln2``,
        so the default of 0.2 gives an equilibrium coverage of ~0.87 at a 3 s
        half life. Setting it to 1.0 saturates every breaking cell to full
        coverage, which throws away the distinction between the outer surf and
        the inner swash.
    half_life : decay half life [s]; ``nearshore.foam_halflife`` in config.
    advect : whether to transport foam shoreward at the group velocity.
    """

    bathy: object
    seed_rate: float = 0.2
    half_life: float = 3.0
    advect: bool = True
    max_coverage: float = 1.0

    def step(self, foam: np.ndarray, breaking: np.ndarray, cg: np.ndarray,
             dt: float) -> np.ndarray:
        """Advance one step: advect, decay, seed.

        Advection is semi-Lagrangian -- each cell asks where its foam came from
        one step ago and interpolates there. That is unconditionally stable, so
        the step size is set by how smooth you want the result rather than by a
        CFL limit, which matters because ``Cg`` varies by an order of magnitude
        across the surf zone.
        """
        from scipy.ndimage import map_coordinates

        out = foam
        if self.advect:
            dx = self.bathy.meta.dx
            ny, nx = self.bathy.meta.shape
            jj, ii = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
            # Foam rides shoreward, i.e. along the inland-pointing shore normal.
            n_x, n_y = self.bathy.shore_normal
            src_x = jj - (cg * n_x * dt) / dx
            src_y = ii - (cg * n_y * dt) / dx
            out = map_coordinates(out, np.stack([src_y, src_x]), order=1,
                                  mode="nearest")

        out = out * foam_decay_factor(dt, self.half_life)
        out = out + self.seed_rate * dt * breaking.astype(np.float64)
        return np.clip(out, 0.0, self.max_coverage)

    def evaluate(
        self,
        breaking_at,
        cg: np.ndarray,
        t: float,
        dt: float = 0.25,
        n_spinup: int | None = None,
        tol: float = 0.005,
    ) -> np.ndarray:
        """Foam coverage at time ``t``, cold-started far enough back to be exact.

        ``breaking_at`` is a callable ``t -> bool array`` on the bathymetry
        grid; it is called once per spin-up step, so the breaking mask is
        allowed to move with the waves.

        ``n_spinup`` defaults to whatever :func:`spinup_steps` says is needed to
        get the discarded history below ``tol``.  Pass it explicitly only to
        study the trade-off -- picking a round number by eye is exactly how the
        reproducibility guarantee gets silently lost.

        ``tol`` defaults to 0.005 rather than the 0.01 Gate 5 asks for, because
        the two are not measuring the same thing.  ``tol`` bounds the *initial
        condition's* residual; the gate bounds the *per-cell relative*
        difference between a cold and a sequential frame, and at the foam's
        leading edge -- where coverage is small and made almost entirely of
        advected history -- the latter runs a few times larger.  Measured:
        ``tol = 0.01`` leaves 2.2% per-cell error, ``tol = 0.005`` leaves 0.6%.

        ``dt`` defaults to 0.25 s rather than a frame time.  Advection is
        semi-Lagrangian and therefore unconditionally stable, so the step is set
        by how smoothly foam should move rather than by a CFL limit; at
        ``Cg ~ 1 m/s`` in the surf zone that is a quarter-cell per step on a
        1 m grid.  Using a frame time instead would cost 6x the work for no
        visible difference.

        The returned field depends only on ``t``, ``dt`` and the spin-up window
        -- not on any previously computed frame -- which is what keeps "any
        node, any frame" true.
        """
        if n_spinup is None:
            n_spinup = spinup_steps(tol, dt, self.half_life)

        ny, nx = self.bathy.meta.shape
        foam = np.zeros((ny, nx), dtype=np.float64)
        t0 = t - n_spinup * dt
        for i in range(n_spinup):
            foam = self.step(foam, breaking_at(t0 + i * dt), cg, dt)
        return foam

    def equilibrium_coverage(self, dt: float | None = None) -> float:
        """Steady-state coverage in a cell that breaks continuously [-].

        With ``dt`` given, the exact discrete geometric series
        ``seed_rate * dt / (1 - r)``.  Without it, the continuous limit
        ``seed_rate * t_half / ln2``, which is what to reason about when
        choosing ``seed_rate`` because it does not depend on the step size.
        """
        if dt is None:
            value = self.seed_rate * self.half_life / np.log(2.0)
        else:
            value = self.seed_rate * dt / (1.0 - foam_decay_factor(dt, self.half_life))
        return float(min(value, self.max_coverage))
