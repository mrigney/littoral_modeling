"""Generate the figure set that explains what this package does.

    python scripts/make_figures.py                 # all figures -> docs/figures/
    python scripts/make_figures.py --only spectrum # one figure
    python scripts/make_figures.py --list          # what is available

Written for an audience that has not read the code: each figure carries the
number it is demonstrating, so the picture and the validation report say the
same thing. Every value is computed live from `pywave` at run time -- nothing
here is drawn from memory or hard-coded, so a figure that disagrees with the
report means the code changed.

Output is PNG at 150 dpi into `docs/figures/`, plus `docs/gallery.md` which
embeds them with captions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from pywave import foam as foam_mod  # noqa: E402
from pywave import load_config, moments, nearshore, spectrum, surface, tiling  # noqa: E402
from pywave.bathymetry import Bathymetry  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
CONFIG = ROOT / "configs" / "test_lake.yaml"

# A restrained palette that survives greyscale printing and colour-blind viewers.
INK = "#1b2733"
MUTED = "#7a8794"
ACCENT = "#1f6f8b"
ACCENT2 = "#c1666b"
ACCENT3 = "#5d8a4e"
GRID = "#dde3e8"

WATER = LinearSegmentedColormap.from_list(
    "water", ["#0b2b3f", "#12506e", "#2b86a8", "#78c2d2", "#dff3f6"])
SAND = LinearSegmentedColormap.from_list(
    "sand", ["#0b2b3f", "#17627f", "#63b0c4", "#d9cba3", "#a8895f", "#6d5638"])

_REGISTRY: dict[str, tuple] = {}


def figure(name: str, caption: str):
    """Register a figure builder under ``name``."""
    def deco(fn):
        _REGISTRY[name] = (fn, caption)
        return fn
    return deco


class Scene:
    """Everything a figure might need, built once and shared.

    Tile sets and bathymetry grids are the expensive part; eight figures each
    rebuilding their own turned a 20-second job into a two-minute one. Held
    lazily so `--only` still pays for just what it uses.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._cache: dict = {}

    def _memo(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    @property
    def tileset(self):
        return self._memo("ts", lambda: tiling.TileSet.build(self.cfg))

    @property
    def fields(self):
        return self._memo("fields", lambda: self.tileset.evaluate_grids(0.0))

    @property
    def bathy(self):
        return self._memo("bathy", lambda: Bathymetry.from_config(self.cfg))

    @property
    def fine_bathy(self):
        return self._memo("fine", lambda: Bathymetry.from_config(self.cfg, fine=True))

    @property
    def onshore_cfg(self):
        """The scene with the wind blowing straight onshore (+Y).

        Normal incidence makes the refraction coefficient exactly 1, isolating
        shoaling. At oblique incidence the two very nearly cancel on a Dean
        beach, so a figure that did not control for this would suggest shoaling
        does nothing.
        """
        from dataclasses import replace

        return self._memo("oncfg", lambda: replace(
            self.cfg, wind=replace(self.cfg.wind, direction_rad=np.radians(90.0))))

    @property
    def onshore_tileset(self):
        return self._memo("onts", lambda: tiling.TileSet.build(self.onshore_cfg))

    # -- scene-derived scales, so figures adapt to any config ----------------

    @property
    def view_extent(self) -> float:
        """Side of the surface close-up [m]: enough peaks to read, not so many
        that the finest band aliases into noise."""
        return float(np.clip(24.0 * self.cfg.lambda_p, 8.0, 400.0))

    @property
    def tile_ladder(self):
        """Grid sizes for the Hs-by-resolution bar chart, scaled to the sea state."""
        base = 2.0 ** np.round(np.log2(20.0 * self.cfg.lambda_p))
        return [(float(base * f), n) for f, n in ((0.5, 256), (1, 512),
                                                  (2, 1024), (4, 1024))]

    @property
    def shore_ref(self):
        """``(x, y)`` on a representative stretch of shoreline.

        A synthetic beach has a shoreline at a known Y, so this is arithmetic.
        Real terrain has a closed contour that may be anywhere, so the waterline
        has to be *found* -- and finding it is not optional: every transect,
        camera and animation window in this file is positioned relative to it,
        and a hardcoded Y quietly puts them all in open water.

        Picks the northernmost stretch, because the camera looks from -Y toward
        +Y and that is the orientation where a shore fills the frame.
        """
        def make():
            cfg = self.cfg
            b = self.bathy
            x0, x1, y0, y1 = b.meta.extent
            if not cfg.bathymetry.is_export:
                return 0.5 * (x0 + x1), cfg.bathymetry.shoreline

            xa, ya = b.meta.axes()
            X, Y = np.meshgrid(xa, ya, indexing="xy")
            shore = np.abs(b.sdf) < 1.5 * b.meta.dx
            wet = b.depth > 0.0
            if not shore.any() or not wet.any() or wet.all():
                raise ValueError(
                    f"this bathymetry has no shoreline: "
                    f"{100 * wet.mean():.1f}% of it is wet. Every window, "
                    f"transect and camera in the toolchain is positioned "
                    f"relative to a waterline, so there is nothing sensible to "
                    f"return. An all-water or all-land export is almost always "
                    f"a mistake in the terrain, not a scene worth rendering.")
            # Take the top decile of latitude, then the cell nearest that
            # group's median X. Note the last step: medians of X and Y taken
            # *independently* give a point that need not lie on the contour at
            # all, and on a curved shore it lands metres out to sea -- which is
            # exactly where every transect then starts.
            top = shore & (Y >= np.percentile(Y[shore], 90.0))
            xs, ys = X[top], Y[top]
            k = int(np.argmin(np.abs(xs - np.median(xs))))
            return float(xs[k]), float(ys[k])

        return self._memo("shore_ref", make)

    @property
    def water_extent(self):
        """``(x0, y0, x1, y1)`` bounding box of the wet cells [m]."""
        def make():
            b = self.bathy
            xa, ya = b.meta.axes()
            X, Y = np.meshgrid(xa, ya, indexing="xy")
            wet = b.depth > 0.0
            if not wet.any():
                return b.meta.extent
            return (float(X[wet].min()), float(Y[wet].min()),
                    float(X[wet].max()), float(Y[wet].max()))

        return self._memo("water_extent", make)

    @property
    def offshore(self):
        """Unit vector pointing **offshore** from :attr:`shore_ref`.

        The opposite of the inland shore normal there.  Every window, transect
        and camera in the toolchain used to assume offshore was -Y, which is
        true of the synthetic beaches and of nothing else: a coast facing north
        put the mesh region 86% on dry land, silently, because a rectangle built
        the wrong way from a correct shoreline point is still a valid rectangle.
        """
        def make():
            x, y = self.shore_ref
            _, _, normal = self.bathy.sample(np.array([x]), np.array([y]))
            v = -np.array([float(normal[0][0]), float(normal[1][0])])
            n = float(np.hypot(*v))
            return (v / n) if n > 1e-9 else np.array([0.0, -1.0])

        return self._memo("offshore", make)

    @property
    def offshore_axis(self):
        """``(axis, sign)`` of the dominant offshore direction.

        ``axis`` is 0 for X or 1 for Y; ``sign`` is +1 or -1.  Regions and
        animation windows stay axis-aligned rectangles, so they follow whichever
        axis the coast mostly faces rather than an arbitrary bearing.
        """
        v = self.offshore
        axis = 0 if abs(v[0]) > abs(v[1]) else 1
        return axis, (1.0 if v[axis] >= 0 else -1.0)

    def offshore_line(self, length: float, n: int = 400, start: float = 0.05):
        """``(x, y)`` running from just inside the waterline out to ``length``."""
        x0, y0 = self.shore_ref
        v = self.offshore
        t = np.linspace(start, length, n)
        return x0 + v[0] * t, y0 + v[1] * t

    @property
    def swash_band(self) -> float:
        """Horizontal swash excursion [m], from Hunt runup on this beach."""
        slope = self.bathy.beach_slope()
        l0 = float(nearshore.deep_water_wavelength(1.0 / self.cfg.f_p))
        hs = self.tileset.hs()
        xi = nearshore.iribarren_number(slope, hs, l0)
        return float(nearshore.swash_width(nearshore.hunt_runup(xi, hs), slope))

    def foam_field(self, t: float = 30.0):
        """``(foam, breaking)`` on the refined grid.

        Cached because the spin-up integration is the most expensive thing in
        the figure set, and both the surf-zone figure and the run report want it.
        """
        def make():
            b = self.fine_bathy
            omega = 2.0 * np.pi * self.cfg.f_p
            d = np.maximum(b.depth, 1e-3)
            cg = spectrum.group_velocity(spectrum.dispersion_k(omega, d), d)
            ks = nearshore.shoaling_coefficient(omega, np.maximum(b.depth, 1e-9))
            hs = np.where(b.depth > 0.0, self.onshore_tileset.hs() * ks, 0.0)
            brk = nearshore.breaking_mask(hs, b.depth,
                                          self.cfg.nearshore.breaker_index)
            model = foam_mod.FoamModel(
                bathy=b, half_life=self.cfg.nearshore.foam_halflife,
                equilibrium=self.cfg.nearshore.foam_coverage)
            return model.evaluate(lambda tt: brk, cg, t=t), brk

        return self._memo("foam", make)

    def surf_transect(self, n: int = 1200):
        """``(x, y)`` running from just inside the waterline to well offshore."""
        reach = max(30.0 * self.swash_band, 12.0)
        return self.offshore_line(reach, n)


def _style(ax, title=None, xlabel=None, ylabel=None, legend=False):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
    if legend:
        lg = ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        lg.get_title() and lg.get_title().set_color(INK)


def _annotate(ax, text, xy, xytext, color=ACCENT2):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.9))


def _shade(h, slope_x, slope_y, az_deg=315.0, alt_deg=38.0, cmap=WATER):
    """Colour by elevation, shaded by the *analytic* surface normal.

    Uses `SurfaceField`'s spectral slopes rather than a finite difference of the
    height raster. That is the whole point of carrying them: the normal is the
    physics, and differencing the heightfield would throw away exactly the
    high-frequency content that makes a water surface legible.
    """
    n = np.stack([-slope_x, -slope_y, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    az, alt = np.radians(az_deg), np.radians(alt_deg)
    light = np.array([np.cos(az) * np.cos(alt), np.sin(az) * np.cos(alt), np.sin(alt)])
    lam = np.clip(n @ light, 0.0, 1.0)
    lam = 0.32 + 0.68 * lam ** 0.8                      # lift the shadows

    lim = float(np.abs(h).max()) or 1.0
    rgb = cmap(0.5 + 0.5 * np.clip(h / lim, -1, 1))[..., :3]
    return np.clip(rgb * lam[..., None], 0.0, 1.0)


# ---------------------------------------------------------------------------


@figure("spectrum", "The JONSWAP spectrum and its directional spreading — the "
                    "single source of truth from which every other quantity in "
                    "the package is derived.")
def fig_spectrum(scene):
    cfg = scene.cfg
    u10, fetch, gamma = cfg.wind.speed, cfg.wind.fetch, cfg.spectrum.gamma
    _, f_p, _ = spectrum.jonswap_params(u10, fetch)

    fig = plt.figure(figsize=(11.5, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.34, top=0.76)

    ax = fig.add_subplot(gs[0])
    f = np.linspace(0.05, 6.0, 4000)
    for g_, style, lab in ((gamma, "-", f"gamma = {gamma} (JONSWAP)"),
                           (1.0, "--", "gamma = 1 (Pierson-Moskowitz)")):
        ax.plot(f, spectrum.jonswap_sf(f, u10, fetch, g_), style,
                color=ACCENT if g_ == gamma else MUTED, lw=1.8, label=lab)
    ax.axvline(f_p, color=ACCENT2, lw=1.0, ls=":")
    _annotate(ax, f"$f_p$ = {f_p:.3f} Hz\n$T_p$ = {1/f_p:.2f} s",
              (f_p, spectrum.jonswap_sf(np.array([f_p]), u10, fetch, gamma)[0]),
              (f_p * 1.6, spectrum.jonswap_sf(np.array([f_p]), u10, fetch, gamma)[0] * 0.75))
    _style(ax, "S(f) — energy by frequency", "frequency f [Hz]", "S(f) [m²/Hz]", legend=True)

    ax = fig.add_subplot(gs[1], projection="polar")
    theta = np.linspace(-np.pi, np.pi, 721)
    for ratio, col in ((0.7, "#9dc3d4"), (1.0, ACCENT), (2.0, ACCENT3)):
        d = spectrum.spreading(theta, np.full_like(theta, ratio * f_p), f_p,
                               cfg.wind.direction_rad)
        ax.plot(theta, d, color=col, lw=1.7, label=f"f = {ratio:g} $f_p$")
    ax.set_theta_zero_location("E")
    ax.tick_params(colors=MUTED, labelsize=6.5, pad=-2)
    ax.set_yticklabels([])
    ax.grid(color=GRID)
    ax.set_title("D(f, θ) — energy by direction\nwind toward 45°,  ∫D dθ = 1",
                 color=INK, fontsize=10.5, pad=16)
    ax.legend(frameon=False, fontsize=7, loc="upper left",
              bbox_to_anchor=(-0.22, 0.06), labelcolor=INK)

    ax = fig.add_subplot(gs[2])
    k = np.linspace(-12, 12, 401)
    KX, KY = np.meshgrid(k, k)
    s_k = spectrum.jonswap_sk(KX, KY, u10, fetch, cfg.wind.direction_rad, gamma=gamma)
    # Fourth root rather than log: it keeps the peak legible without giving
    # eight decades of near-zero tail the same visual weight as the energy.
    ax.pcolormesh(KX, KY, np.maximum(s_k, 0.0) ** 0.25, cmap=WATER, shading="auto")
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), cfg.k_p, fill=False, color=ACCENT2, lw=1.0, ls=":"))
    ax.annotate(f"$k_p$ = {cfg.k_p:.2f} rad/m", (0.71 * cfg.k_p, 0.71 * cfg.k_p),
                (4.0, 7.5), fontsize=8, color=ACCENT2,
                arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=0.9))
    _style(ax, "S(kx, ky) — after the Jacobian", "$k_x$ [rad/m]", "$k_y$ [rad/m]")
    ax.grid(False)

    fig.suptitle(
        f"Wind {u10:g} m/s, fetch {fetch:g} m   →   "
        f"Hs = {spectrum.hs_spectral(u10, fetch, gamma):.4f} m, "
        f"T$_p$ = {1 / f_p:.2f} s, λ$_p$ = {cfg.lambda_p:.2f} m",
        color=INK, fontsize=9.5, y=0.97)
    return fig


@figure("lod", "The LOD invariant. Slope variance lost when the mesh coarsens is "
               "not lost — it is handed to the BSDF as sub-facet roughness, so "
               "the total is conserved across every level of detail.")
def fig_lod(scene):
    cfg = scene.cfg
    u10, fetch, gamma = cfg.wind.speed, cfg.wind.fetch, cfg.spectrum.gamma
    total = moments.mss_above(0.0, u10, fetch, gamma=gamma)

    dx = np.geomspace(0.02, 4.0, 200)
    k_cut = np.pi / dx
    resolved = np.array([moments.mss_between(moments.K_MIN_DEFAULT, kc, u10, fetch,
                                             gamma=gamma, order=2) for kc in k_cut])
    above = np.array([moments.mss_above(kc, u10, fetch, gamma=gamma) for kc in k_cut])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    ax = axes[0]
    ax.fill_between(dx, 0, resolved, color=ACCENT, alpha=0.75, label="resolved by the mesh")
    ax.fill_between(dx, resolved, resolved + above, color=ACCENT2, alpha=0.65,
                    label="below the mesh → BSDF roughness")
    ax.plot(dx, resolved + above, color=INK, lw=1.4, label="total mss (invariant)")
    ax.set_xscale("log")
    for ring in cfg.output.lod_rings[:-1]:
        ax.axvline(ring.dx, color=MUTED, ls=":", lw=1.0)
        ax.text(ring.dx, total * 1.02, f" {ring.dx} m", fontsize=7.5, color=MUTED)
    _style(ax, "Where the slope variance lives", "mesh post spacing [m]",
           "mean square slope [-]", legend=True)
    ax.set_ylim(0, total * 1.15)

    ax = axes[1]
    err = np.abs(resolved + above - total) / total
    ax.semilogx(dx, err, color=ACCENT, lw=1.8)
    ax.axhline(1e-6, color=ACCENT2, ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_ylim(err.min() * 0.5, 4e-6)
    ax.text(dx[2], 1.15e-6, "test tolerance 1e-6", fontsize=7.5, color=ACCENT2,
            va="bottom")
    _style(ax, "Closure error of the invariant", "mesh post spacing [m]",
           "|resolved + sub-grid − total| / total")

    fig.suptitle(
        f"mss$_{{total}}$ = {total:.5f}   ·   at the 0.125 m near-field mesh, "
        f"{100 * moments.mss_above(np.pi / 0.125, u10, fetch, gamma=gamma) / total:.0f}% "
        f"of slope variance is sub-mesh",
        color=INK, fontsize=9, y=1.02)
    fig.tight_layout()
    return fig


@figure("surface", "One realisation of the composite surface. Three tiles of "
                   "incommensurate size, each carrying a disjoint band of the "
                   "spectrum and rotated by a multiple of the golden angle, so "
                   "no repeat pattern survives in the sum.")
def fig_surface(scene):
    cfg = scene.cfg
    ts, fields = scene.tileset, scene.fields

    # Scaled to the sea state: ~24 peak wavelengths across at 900 px, which
    # resolves the top of the finest band rather than aliasing it into noise.
    n, extent = 900, scene.view_extent
    ax_ = np.linspace(0.0, extent, n)
    X, Y = np.meshgrid(ax_, ax_)

    fig = plt.figure(figsize=(12.6, 4.1))
    gs = fig.add_gridspec(1, 4, wspace=0.08)

    for i, (tile, field) in enumerate(zip(ts.tiles, fields)):
        ax = fig.add_subplot(gs[i])
        s = tiling.composite_surface([tile], X, Y, 0.0, fields=[field])
        ax.imshow(_shade(s.h, s.slope_x, s.slope_y), origin="lower",
                  extent=(0, extent, 0, extent), interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"tile {i + 1}   L = {tile.size:g} m, N = {tile.n}",
                     color=INK, fontsize=9, loc="left", pad=5)
        ax.text(0.03, 0.965,
                f"k ∈ [{tile.band[0]:.1f}, {tile.band[1]:.1f}) rad/m\n"
                f"λ ∈ [{2 * np.pi / tile.band[1]:.2f}, "
                f"{'∞' if tile.band[0] == 0 else f'{2 * np.pi / tile.band[0]:.1f}'}] m\n"
                f"Hs = {4 * np.sqrt(tile.m0()):.4f} m",
                transform=ax.transAxes, va="top", fontsize=7.5, color="white",
                bbox=dict(boxstyle="round,pad=0.35", fc=INK, ec="none", alpha=0.62))

    ax = fig.add_subplot(gs[3])
    comp = ts.sample(X, Y, 0.0, fields=fields)
    ax.imshow(_shade(comp.h, comp.slope_x, comp.slope_y), origin="lower",
              extent=(0, extent, 0, extent), interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("composite  =  sum of the three", color=INK, fontsize=9,
                 loc="left", pad=5)
    ax.text(0.03, 0.965,
            f"Hs realised {4 * np.std(comp.h):.4f} m\n"
            f"Hs theory   {ts.hs():.4f} m\n"
            f"({100 * (4 * np.std(comp.h) / ts.hs() - 1):+.1f}%)",
            transform=ax.transAxes, va="top", fontsize=7.5, color="white",
            bbox=dict(boxstyle="round,pad=0.35", fc=ACCENT2, ec="none", alpha=0.72))
    ax.plot([2, 12], [2, 2], color="white", lw=2.0, solid_capstyle="butt")
    ax.text(7, 3.0, "10 m", color="white", fontsize=8, ha="center")

    fig.suptitle(
        "40 × 40 m of surface at t = 0, shaded by the analytic spectral normal.  "
        "Incommensurate tile sizes and golden-angle rotations leave no repeat in the sum.",
        color=INK, fontsize=9, y=1.015)
    return fig


@figure("statistics", "The realised surface reproduces the spectrum it was built "
                      "from: Gaussian elevations, the right variance, and crests "
                      "travelling at the right speed.")
def fig_statistics(scene):
    cfg = scene.cfg
    ts, fields = scene.tileset, scene.fields
    rng = np.random.default_rng(3)
    span = 8.0 * scene.view_extent
    X = rng.uniform(0, span, 300_000)
    Y = rng.uniform(0, span, 300_000)
    comp = ts.sample(X, Y, 0.0, fields=fields)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    ax = axes[0]
    h = comp.h
    ax.hist(h, bins=140, density=True, color=ACCENT, alpha=0.65, edgecolor="none",
            label="realised")
    g = np.linspace(h.min(), h.max(), 400)
    ax.plot(g, np.exp(-0.5 * (g / h.std()) ** 2) / (h.std() * np.sqrt(2 * np.pi)),
            color=ACCENT2, lw=1.7, label="Gaussian")
    from scipy.stats import skew
    _style(ax, f"Elevation distribution\nskew {skew(h):+.3f}, σ = {h.std():.4f} m",
           "elevation [m]", "density", legend=True)

    ax = axes[1]
    u10, fetch, gamma = cfg.wind.speed, cfg.wind.fetch, cfg.spectrum.gamma
    sizes = scene.tile_ladder
    realised, theory, labels = [], [], []
    for L, n in sizes:
        tile = surface.WaveTile.build(L, n, u10, fetch, cfg.wind.direction_rad, seed=20260801)
        realised.append(4 * float(np.std(tile.evaluate(0.0).h)))
        theory.append(4 * np.sqrt(moments.mss_between(2 * np.pi / L, tile.k_nyquist,
                                                      u10, fetch, gamma=gamma, order=0)))
        labels.append(f"{L:g} m\nN={n}")
    xpos = np.arange(len(sizes))
    ax.bar(xpos - 0.2, theory, 0.4, color=MUTED, label="band-limited theory")
    ax.bar(xpos + 0.2, realised, 0.4, color=ACCENT, label="realised 4·std(h)")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=7.5, color=MUTED)
    ax.axhline(spectrum.hs_spectral(u10, fetch, gamma), color=ACCENT2, ls="--", lw=1.2)
    ax.text(-0.4, spectrum.hs_spectral(u10, fetch, gamma) * 1.01,
            "untruncated spectrum", fontsize=7.5, color=ACCENT2)
    _style(ax, "Hs, single tile, by grid", None, "Hs [m]", legend=True)
    ax.set_ylim(0, spectrum.hs_spectral(u10, fetch, gamma) * 1.25)

    ax = axes[2]
    L, n, mode = 4.0 * cfg.lambda_p * 8.0, 256, 4
    kx, ky, k, _ = surface.grid_wavenumbers(L, n)
    h0 = np.zeros((n, n), complex); h0[0, mode] = 1.0
    omega = spectrum.dispersion_omega(k)
    c = float(omega[0, mode]) / float(k[0, mode])
    xs = np.arange(n) * (L / n)
    for i, t in enumerate((0.0, 1.0, 2.0)):
        row = surface.surface_at(h0, kx, ky, k, omega, t).h[0]
        off = i * 2.8
        ax.plot(xs, row + off, color=[MUTED, ACCENT, ACCENT3][i], lw=1.5,
                label=f"t = {t:g} s")
        crest = (c * t) % L
        ax.plot([crest], [np.interp(crest, xs, row) + off], "o", color=ACCENT2, ms=5,
                zorder=5)
        if i:
            ax.annotate("", xy=(crest, off - 1.15),
                        xytext=((c * (t - 1)) % L, off - 1.15),
                        arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.1))
    ax.text(0.5, -2.6, f"one crest, moving at c = ω/k = {c:.2f} m/s",
            fontsize=8, color=ACCENT2)
    ax.set_yticks([])
    ax.set_ylim(-3.2, 9.2)
    _style(ax, "A single mode, tracked in time", "x [m]", None, legend=True)

    fig.tight_layout()
    return fig


@figure("bathymetry", "The synthetic Dean beach that Phase 5 is validated "
                      "against. It satisfies the same field contract Houdini "
                      "will export in Phase 4, so swapping in real terrain is a "
                      "loader change rather than a physics change.")
def fig_bathymetry(scene):
    cfg = scene.cfg
    planar = scene.bathy
    bay = Bathymetry.dean_embayment(
        nx=256, ny=256, dx=max(cfg.bathymetry.dx, 1.0),
        shoreline_y=0.75 * 256 * max(cfg.bathymetry.dx, 1.0),
        dean_a=cfg.bathymetry.a, max_depth=cfg.bathymetry.max_depth)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))

    ax = axes[0]
    y = np.linspace(0.0, 300.0, 600)
    dmax = cfg.bathymetry.max_depth
    for a, lab, col in ((0.079, "fine sand  A=0.079", "#9dc3d4"),
                        (cfg.bathymetry.a, f"this scene  A={cfg.bathymetry.a:.3f}", ACCENT),
                        (0.125, "coarse sand  A=0.125", ACCENT3)):
        ax.plot(y, -np.minimum(a * y ** (2 / 3), dmax), color=col, lw=1.8, label=lab)
    ax.plot(y, -np.minimum(dmax / 300.0 * y, dmax), color=ACCENT2, ls="--", lw=1.3,
            label="linear ramp (wrong)")
    ax.axhline(0, color=MUTED, lw=1.0)
    _style(ax, "Dean equilibrium profile  h = A·y$^{2/3}$", "distance offshore [m]",
           "bed elevation about $z_w$ [m]", legend=True)

    ax = axes[1]
    xa, ya = bay.meta.axes()
    m = ax.pcolormesh(xa, ya, np.where(bay.depth > 0, bay.depth, np.nan),
                      cmap=WATER.reversed(), shading="auto")
    ax.contour(xa, ya, bay.sdf, levels=[0.0], colors=[INK], linewidths=1.2)
    # Only in the nearshore band: that is where the normal is used, and drawing
    # it across 400 m of open water just hides the shoreline.
    step = 18
    band = np.abs(bay.sdf) < 70.0
    u = np.where(band, bay.shore_normal[0], np.nan)[::step, ::step]
    v = np.where(band, bay.shore_normal[1], np.nan)[::step, ::step]
    ax.quiver(xa[::step], ya[::step], u, v, color=ACCENT2, scale=24, width=0.005)
    ax.set_ylim(220, 512)
    ax.set_aspect("equal")
    cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("depth [m]", color=INK, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    _style(ax, "Embayment: depth, waterline, shore normal", "x [m]", "y [m]")
    ax.grid(False)

    ax = axes[2]
    xa, ya = planar.meta.axes()
    col = planar.meta.shape[1] // 2
    shore = cfg.bathymetry.shoreline
    ax.plot(planar.sdf[:, col], ya, color=ACCENT, lw=1.8, label="sdf (+ inland)")
    ax.plot(planar.depth[:, col], ya, color=ACCENT3, lw=1.8, label="depth (+ in water)")
    ax.axhline(shore, color=INK, lw=1.0, ls=":")
    ax.text(planar.sdf.min() * 0.9, shore + 0.02 * (ya[-1] - ya[0]),
            "waterline", fontsize=7.5, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    _style(ax, "Sign convention: sdf = −sign(depth)", "field value [m]", "y [m]",
           legend=True)

    fig.suptitle("depth = z$_w$ − z    ·    sdf < 0 in water, > 0 on land    ·    "
                 "shore_normal = ∇s/|∇s| points inland",
                 color=INK, fontsize=9, y=1.03)
    fig.tight_layout()
    return fig


@figure("shoaling", "Shoaling and refraction coefficients, both solved against "
                    "the full dispersion relation and both checked against "
                    "closed-form answers — Green's law and Snell's law.")
def fig_shoaling(scene):
    cfg = scene.cfg
    omega = 2 * np.pi * cfg.f_p
    beach = scene.bathy

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))

    ax = axes[0]
    d = np.geomspace(0.002, max(6.0 * cfg.lambda_p, 20.0), 3000)
    ks = nearshore.shoaling_coefficient(omega, d)
    ax.semilogx(d, ks, color=ACCENT, lw=2.0, label="$K_s$ (full dispersion)")
    green = ks[0] * (d / d[0]) ** -0.25
    ax.semilogx(d, green, color=ACCENT3, ls="--", lw=1.3, label="Green's law  $d^{-1/4}$")
    i = int(np.argmin(ks))
    ax.plot(d[i], ks[i], "o", color=ACCENT2, ms=5)
    _annotate(ax, f"minimum {ks[i]:.4f}\nat kd = 1.20", (d[i], ks[i]), (d[i] * 2.5, 0.80))
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=":")
    ax.set_ylim(0.6, 2.6)
    _style(ax, "Shoaling coefficient", "still-water depth [m]", "$K_s$ [-]", legend=True)

    ax = axes[1]
    x, y = scene.offshore_line(40.0 * cfg.lambda_p, 400, start=0.4)
    depth, _, normal = beach.sample(x, y)
    for wind_deg, col in ((10.0, "#9dc3d4"), (45.0, ACCENT), (135.0, ACCENT3),
                          (170.0, ACCENT2)):
        _, alpha = nearshore.refraction_angle(np.radians(wind_deg), normal, depth, omega)
        ax.plot(depth, np.degrees(alpha), color=col, lw=1.7,
                label=f"wind {wind_deg:g}°")
    ax.axhline(0, color=INK, lw=1.0, ls=":")
    ax.text(3.0, 2, "shore-normal", fontsize=7.5, color=INK)
    ax.invert_xaxis()
    _style(ax, "Refraction: angle from the shore normal\n(shoreward →)",
           "depth [m]", "incidence angle [deg]", legend=True)

    ax = axes[2]
    _, alpha = nearshore.refraction_angle(np.radians(45.0), normal, depth, omega)
    alpha_deep, _ = nearshore._incidence(np.radians(45.0), normal)
    ks_t = nearshore.shoaling_coefficient(omega, depth)
    kr_t = nearshore.refraction_coefficient(alpha_deep, alpha)
    ax.plot(depth, ks_t, color=ACCENT, lw=1.8, label="$K_s$ shoaling")
    ax.plot(depth, kr_t, color=ACCENT2, lw=1.8, label="$K_r$ refraction")
    ax.plot(depth, ks_t * kr_t, color=INK, lw=2.0, label="$K_s K_r$ combined")
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=":")
    ax.invert_xaxis()
    _style(ax, "At 45° incidence the two nearly cancel", "depth [m]",
           "amplitude gain [-]", legend=True)

    fig.tight_layout()
    return fig


@figure("nearshore", "A transect through the surf zone. Wave height saturates at "
                     "the depth-limited breaking criterion, foam is seeded where "
                     "waves break and swept shoreward, and the wetness channel "
                     "carries the sub-pixel swash as a duty cycle.")
def fig_nearshore(scene):
    cfg = scene.cfg
    cfg_on = scene.onshore_cfg
    ts = scene.onshore_tileset
    beach = scene.fine_bathy

    x, y = scene.surf_transect()
    nf = nearshore.transform(ts, beach, cfg_on, x, y, 0.0)
    offshore = np.hypot(x - scene.shore_ref[0], y - scene.shore_ref[1])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4))

    ax = axes[0, 0]
    ax.fill_between(offshore, -nf.depth, 0, color="#bcd9e6", alpha=0.8)
    ax.plot(offshore, -nf.depth, color="#8a6f4a", lw=2.0)
    ax.axhline(0, color=ACCENT, lw=1.2)
    ax.set_ylim(-1.25 * nf.depth.max(), 0.25 * nf.depth.max())
    label = (f"Bed profile (Dean, A = {cfg.bathymetry.a:.3f})"
             if not cfg.bathymetry.is_export
             else f"Bed profile (measured, foreshore "
                  f"{100 * scene.bathy.beach_slope():.0f}%)")
    _style(ax, label, "distance offshore [m]", "elevation about $z_w$ [m]")

    ax = axes[0, 1]
    ax.plot(offshore, nf.hs_local, color=ACCENT, lw=2.0, label="$H_s$ local")
    ax.plot(offshore, 0.78 * nf.depth, color=ACCENT2, ls="--", lw=1.4,
            label="$\\gamma_b d$ = 0.78 d")
    ax.axhline(ts.hs(), color=MUTED, ls=":", lw=1.2)
    ax.text(offshore[-1] * 0.55, ts.hs() * 1.04, f"deep-water Hs = {ts.hs():.4f} m",
            fontsize=7.5, color=MUTED)
    brk = nf.breaking
    if brk.any():
        ax.axvspan(offshore[brk].min(), offshore[brk].max(), color=ACCENT2, alpha=0.13)
        ax.text(offshore[brk].max() * 1.15, ts.hs() * 0.35,
                f"surf zone\n{offshore[brk].max() - offshore[brk].min():.1f} m wide",
                fontsize=7.5, color=ACCENT2)
    ax.set_ylim(0, ts.hs() * 1.45)
    _style(ax, "Wave height saturates at the breaker limit",
           "distance offshore [m]", "$H_s$ [m]", legend=True)

    ax = axes[1, 0]
    band = scene.swash_band
    s = np.linspace(-0.4 * band, 1.45 * band, 500)
    ax.plot(s, nearshore.wetness_fraction(s, band), color=ACCENT, lw=2.0,
            label="duty cycle (thermal)")
    for t, col in ((0.0, "#c7d7df"), (0.25 / cfg.f_p, "#9dc3d4")):
        ax.plot(s, nearshore.wetness(s, band, t, 1.0 / cfg.f_p), color=col, lw=1.2,
                label=f"instantaneous, t = {t:.2f} s")
    ax.axvline(0, color=INK, lw=1.0, ls=":")
    ax.text(band * 0.02, 1.03, "waterline", fontsize=7.5, color=INK)
    ax.axvline(band, color=ACCENT2, lw=1.0, ls=":")
    ax.text(band * 1.03, 0.85, f"swash limit\nR/tanβ = {band:.2f} m",
            fontsize=7.5, color=ACCENT2)
    _style(ax, "Wetness — the sub-pixel swash channel",
           "distance inland (sdf) [m]", "wet fraction [-]", legend=True)

    ax = axes[1, 1]
    f, brk2 = scene.foam_field()
    xa, ya = beach.meta.axes()
    shore = cfg.bathymetry.shoreline
    # Crop to where foam actually is, not to a band around the nominal
    # shoreline: on an embayment the waterline wanders by its full amplitude,
    # and a fixed y-window slices across the bays instead of following them.
    rows = np.flatnonzero(f.max(axis=1) > 0.01 * max(f.max(), 1e-12))
    if rows.size:
        pad = max(int(round(0.5 * max(band, 1.0) / beach.meta.dx)), 2)
        sl = slice(max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad + 1, len(ya)))
    else:
        keep = np.abs(ya - shore) < max(6.0 * band, 3.0)
        sl = slice(int(np.argmax(keep)), int(len(ya) - np.argmax(keep[::-1])))
    m = ax.pcolormesh(xa, ya[sl], f[sl], cmap="Blues", shading="auto", vmin=0)
    ax.contour(xa, ya[sl], beach.sdf[sl], levels=[0.0], colors=[ACCENT2], linewidths=1.2)
    ax.text(xa[len(xa) // 40], ya[sl][int(0.85 * (sl.stop - sl.start))],
            "waterline", fontsize=7.5, color=ACCENT2)
    ax.set_aspect("auto")
    cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("foam coverage [-]", color=INK, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    _style(ax, f"Foam, confined to the surf band\n(peak {f.max():.3f}, "
               f"half life {cfg.nearshore.foam_halflife:g} s)", "x [m]", "y [m]")
    ax.grid(False)

    fig.tight_layout()
    return fig


@figure("refraction_map", "Refraction on a curved shoreline, in the band where it "
                          "actually acts. Two very different wind directions "
                          "produce nearly the same near-shore wave heading, "
                          "because each ray turns toward its own local contour.")
def fig_refraction_map(scene):
    cfg = scene.cfg
    # A finer, shallower basin: refraction is confined to d < ~1 m, which on the
    # 5 m-deep default beach is the last 30 m and invisible at domain scale.
    # A shallow, finely sampled basin: refraction lives in d < ~1 m, which on a
    # 5 m-deep beach is the last few tens of metres and invisible at domain scale.
    span = max(60.0 * cfg.lambda_p, 80.0)
    # Cap the basin at a depth where refraction is still active for *these*
    # waves: it sets in around kd ~ 1, i.e. a depth of order lambda_p/2. Fixing
    # it at a constant would show nothing for long-period swell and nothing but
    # the deep limit for short chop.
    bay = Bathymetry.dean_embayment(
        nx=384, ny=384, dx=span / 384.0, shoreline_y=0.78 * span,
        amplitude=0.13 * span, wavelength=0.73 * span,
        dean_a=cfg.bathymetry.a,
        max_depth=min(max(0.6 * cfg.lambda_p, 1.0), cfg.bathymetry.max_depth))
    omega = 2 * np.pi * cfg.f_p
    xa, ya = bay.meta.axes()
    water = bay.depth > 0

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5))
    residuals, headings = [], []
    winds = (60.0, 130.0)

    for ax, wind_deg in zip(axes, winds):
        _, alpha = nearshore.refraction_angle(np.radians(wind_deg), bay.shore_normal,
                                              bay.depth, omega)
        theta_n = np.arctan2(bay.shore_normal[1], bay.shore_normal[0])
        theta = theta_n + alpha

        m = ax.pcolormesh(xa, ya, np.where(water, bay.depth, np.nan),
                          cmap=WATER.reversed(), shading="auto", alpha=0.9)
        ax.contour(xa, ya, bay.sdf, levels=[0.0], colors=[INK], linewidths=1.4)
        ax.contour(xa, ya, bay.depth,
                   levels=[0.12 * bay.depth.max(), 0.3 * bay.depth.max()],
                   colors=[MUTED], linewidths=0.7, linestyles=":")

        step = 11
        shallow = water & (bay.depth < 0.7 * bay.depth.max())
        u = np.where(shallow, np.cos(theta), np.nan)[::step, ::step]
        v = np.where(shallow, np.sin(theta), np.nan)[::step, ::step]
        ax.quiver(xa[::step], ya[::step], u, v, color=ACCENT2, scale=26, width=0.004)

        # The shallowest few percent of wet cells, rather than a fixed depth:
        # on a coarse grid a fixed threshold can fall between cells and select
        # nothing, which silently turns the reported angle into a NaN.
        near = water & (bay.depth <= np.percentile(bay.depth[water], 3.0))
        residuals.append(float(np.degrees(np.abs(alpha[near]).mean())))
        headings.append(theta[near])

        ax.set_aspect("equal")
        ax.set_ylim(0.5 * span, 0.95 * span)
        _style(ax, f"wind toward {wind_deg:g}°", "x [m]", "y [m]")
        ax.text(0.985, 0.06, f"residual incidence at the break line: "
                             f"{residuals[-1]:.0f}°",
                transform=ax.transAxes, ha="right", fontsize=7.5, color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc=INK, ec="none", alpha=0.65))
        ax.grid(False)

    cb = fig.colorbar(m, ax=axes, fraction=0.014, pad=0.015, aspect=22)
    cb.set_label("depth [m]", color=INK, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    spread = float(np.degrees(np.abs(
        np.mod(headings[0] - headings[1] + np.pi, 2 * np.pi) - np.pi).mean()))
    fig.suptitle(
        f"Snell refraction over a cosine embayment — deep-water headings "
        f"{winds[1] - winds[0]:.0f}° apart are compressed to {spread:.0f}° apart at "
        f"the break line. Refraction narrows the spread; it does not erase it.",
        color=INK, fontsize=9.5, y=1.02)
    return fig


# ---------------------------------------------------------------------------


def build(names, scene, out_dir=None, dpi=150):
    """Render `names` for `scene` into `out_dir`. Returns (name, caption, path)."""
    out_dir = Path(out_dir or FIG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        fn, caption = _REGISTRY[name]
        print(f"  {name} ...", end="", flush=True)
        fig = fn(scene)
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append((name, caption, path))
        print(f" {path.stat().st_size / 1024:.0f} KiB")
    return written


TITLES = {
    "spectrum": "1 · The spectrum, and everything that comes from it",
    "lod": "2 · The LOD invariant",
    "surface": "3 · The synthesised surface",
    "statistics": "4 · Does the surface match the spectrum it was built from?",
    "bathymetry": "5 · The synthetic beach",
    "shoaling": "6 · Shoaling and refraction coefficients",
    "nearshore": "7 · Through the surf zone",
    "refraction_map": "8 · Refraction on a curved shoreline",
}


def write_gallery(written, cfg, path=None, rel="figures"):
    """Emit a markdown gallery so the figures render on GitHub with captions."""
    lines = [
        "# pywave — what the model does, in eight figures",
        "",
        "A visual tour of Phases 1–3 and 5 for readers who will not run the code.",
        "",
        "Everything here is computed live from `pywave` when the script runs — no "
        "number is copied in by hand. A figure that disagrees with "
        "[validation_report.md](validation_report.md) therefore means the code "
        "changed, not that the picture went stale.",
        "",
        "```",
        "python scripts/make_figures.py            # rebuild all of it",
        "python scripts/make_figures.py --list     # what is available",
        "```",
        "",
        f"**Scene throughout:** `{cfg.name}` — {cfg.wind.speed:g} m/s wind over "
        f"{cfg.wind.fetch:g} m of fetch, giving Hs = "
        f"{100 * spectrum.hs_spectral(cfg.wind.speed, cfg.wind.fetch, cfg.spectrum.gamma):.1f}"
        f" cm and a {1 / cfg.f_p:.2f} s peak period.",
        "",
        "---",
        "",
    ]
    # NOT `path` -- that is the output parameter, and shadowing it here silently
    # redirected the gallery to the last figure's PNG path.
    for name, caption, fig_path in written:
        lines += [f"## {TITLES.get(name, name)}", "",
                  f"![{name}]({rel}/{fig_path.name})", "", caption, "", "---", ""]
    lines += [
        "## Where this sits",
        "",
        "| Phase | Status |",
        "|---|---|",
        "| 1 — spectrum, spreading, moments | implemented |",
        "| 2 — FFT synthesis, multi-tile composition | implemented |",
        "| 3 — validation suite and generated V&V report | implemented |",
        "| 4 — Houdini terrain | not started; figure 5 is the synthetic stand-in |",
        "| 5 — shoaling, refraction, breaking, foam | implemented |",
        "| 6–10 — mesh export, BSDF, emissivity, integration | not started |",
        "",
        "See the [user's guide](users_guide.md) for the API and conventions, and "
        "the [validation report](validation_report.md) for every measured number "
        "with the reference it was judged against.",
        "",
    ]
    path = Path(path or ROOT / "docs" / "gallery.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  gallery -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(CONFIG), help="scene YAML to render")
    p.add_argument("--out", default=None, help="output directory for the PNGs")
    p.add_argument("--only", nargs="*", help="figure names to build")
    p.add_argument("--list", action="store_true", help="list available figures")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    if args.list:
        for name, (_, caption) in _REGISTRY.items():
            print(f"{name:16s} {caption}")
        return

    names = args.only or list(_REGISTRY)
    unknown = set(names) - set(_REGISTRY)
    if unknown:
        p.error(f"unknown figure(s): {sorted(unknown)}; try --list")

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else FIG_DIR
    print(f"scene {cfg.name}: building {len(names)} figure(s) into {out_dir}")
    written = build(names, Scene(cfg), out_dir, args.dpi)
    if not args.only:
        write_gallery(written, cfg)


if __name__ == "__main__":
    main()
