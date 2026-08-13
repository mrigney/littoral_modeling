"""Scene configuration -- dataclasses plus YAML loading, validated on construction.

Degrees appear in config files (cookbook section 0.3: "Degrees only in config
files and only for angles a human types") and are converted to radians exactly
once, here, at load.  Nothing downstream of this module sees a degree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .constants import G, JONSWAP_GAMMA
from .spectrum import jonswap_params

__all__ = [
    "SceneConfig",
    "WindConfig",
    "SpectrumConfig",
    "TileConfig",
    "SurfaceConfig",
    "BathymetryConfig",
    "NearshoreConfig",
    "OutputConfig",
    "LodRing",
    "load_config",
]


@dataclass(frozen=True)
class SceneConfig:
    domain: tuple[float, float]
    water_level: float
    epsg: int

    def __post_init__(self) -> None:
        if min(self.domain) <= 0:
            raise ValueError(f"scene.domain must be positive, got {self.domain}")


@dataclass(frozen=True)
class WindConfig:
    speed: float
    """U10, wind speed at 10 m reference height [m/s]."""
    direction_rad: float
    """Direction the wind blows **toward**, CCW from +X [rad]."""
    fetch: float

    def __post_init__(self) -> None:
        if self.speed <= 0:
            raise ValueError(f"wind.speed must be positive, got {self.speed}")
        if self.fetch <= 0:
            raise ValueError(f"wind.fetch must be positive, got {self.fetch}")

    @property
    def direction_deg(self) -> float:
        return float(np.degrees(self.direction_rad))


@dataclass(frozen=True)
class SpectrumConfig:
    model: str = "jonswap"
    gamma: float = JONSWAP_GAMMA
    spreading: str = "cos2s"
    seed: int = 20260801

    def __post_init__(self) -> None:
        if self.model != "jonswap":
            raise ValueError(f"only 'jonswap' is implemented, got {self.model!r}")
        if self.gamma < 1.0:
            raise ValueError(f"spectrum.gamma must be >= 1, got {self.gamma}")


@dataclass(frozen=True)
class TileConfig:
    size: float
    """Tile side length L [m]."""
    n: int
    """Samples per side."""
    band: tuple[float, float]
    """Fraction of the reference wavenumber range this tile carries."""

    def __post_init__(self) -> None:
        if self.n <= 0 or (self.n & (self.n - 1)) != 0:
            raise ValueError(f"tile.n should be a positive power of two, got {self.n}")
        lo, hi = self.band
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(f"tile.band must satisfy 0 <= lo < hi <= 1, got {self.band}")

    @property
    def dx(self) -> float:
        """Sample spacing [m]."""
        return self.size / self.n

    @property
    def k_min(self) -> float:
        """Lowest non-zero wavenumber this tile can represent [rad/m]."""
        return 2.0 * np.pi / self.size

    @property
    def k_nyquist(self) -> float:
        """Highest wavenumber this tile can represent [rad/m]."""
        return np.pi / self.dx


@dataclass(frozen=True)
class SurfaceConfig:
    tiles: tuple[TileConfig, ...]
    choppiness: float = 1.0

    def __post_init__(self) -> None:
        if len(self.tiles) == 0:
            raise ValueError("surface.tiles must not be empty")
        if self.choppiness < 0:
            raise ValueError(f"choppiness must be >= 0, got {self.choppiness}")
        # Bands must tile [0,1] without gaps or overlap: disjointness is what
        # makes summing the tiles valid (variances add), and a gap silently
        # loses variance.  Cookbook section 2.4.
        ordered = sorted(self.tiles, key=lambda t: t.band[0])
        for a, b in zip(ordered, ordered[1:]):
            if abs(a.band[1] - b.band[0]) > 1e-12:
                raise ValueError(
                    f"tile bands must be contiguous and disjoint; "
                    f"{a.band} is followed by {b.band}"
                )
        if abs(ordered[0].band[0]) > 1e-12 or abs(ordered[-1].band[1] - 1.0) > 1e-12:
            raise ValueError("tile bands must span [0, 1]")

    @property
    def sizes(self) -> tuple[float, ...]:
        return tuple(t.size for t in self.tiles)


@dataclass(frozen=True)
class BathymetryConfig:
    """The synthetic basin, standing in for the Phase 4 terrain export.

    Present so a scene file fully determines the nearshore products.  When
    Phase 4 lands, a ``source`` key pointing at exported fields will select real
    bathymetry instead, and everything here becomes the fallback.
    """

    source: str | None = None
    """Path to a Phase 4 terrain export directory.

    When set, the bathymetry is **loaded** and every synthetic key below is
    ignored -- the export carries its own grid, extent and water level.  When
    ``None``, the synthetic basin is built from those keys instead.
    """
    profile: str = "planar"
    """``planar`` (straight shoreline) or ``embayment`` (cosine bays and headlands).

    Synthetic only; ignored when ``source`` is set."""
    shoreline: float = 400.0
    """Y coordinate of the waterline [m]; water lies below it, land above."""
    dean_a: float | None = None
    """Dean scale parameter [m^(1/3)].  ``None`` derives it from ``grain_size``."""
    grain_size: float = 0.25
    """Median sediment diameter D50 [mm].  Ignored when ``dean_a`` is given."""
    max_depth: float = 5.0
    bank_slope: float = 0.08
    dx: float = 1.0
    """Post spacing of the field grid [m]."""
    surf_dx: float = 0.25
    """Refined post spacing for surf-zone products [m].

    The surf zone here is about a metre wide, so a 1 m grid cannot resolve it at
    all -- this is the terracing problem cookbook section 4.4 warns about, and
    the reason it recommends refining inside the nearshore band.
    """
    amplitude: float = 40.0
    """Embayment only: half the bay-to-headland excursion [m]."""
    wavelength: float = 400.0
    """Embayment only: alongshore period [m]."""

    @property
    def is_export(self) -> bool:
        """True when this scene loads real terrain rather than building one."""
        return self.source is not None

    def __post_init__(self) -> None:
        if self.source is not None and not str(self.source).strip():
            raise ValueError("bathymetry.source must be a path, or omitted")
        if self.profile not in ("planar", "embayment"):
            raise ValueError(
                f"bathymetry.profile must be 'planar' or 'embayment', "
                f"got {self.profile!r}")
        if self.dx <= 0 or self.surf_dx <= 0:
            raise ValueError("bathymetry.dx and surf_dx must be positive")
        if self.surf_dx > self.dx:
            raise ValueError(
                f"bathymetry.surf_dx ({self.surf_dx}) must be finer than dx "
                f"({self.dx}); it exists to resolve the surf zone")
        if self.max_depth <= 0:
            raise ValueError(f"bathymetry.max_depth must be positive, got {self.max_depth}")
        if self.dean_a is not None and self.dean_a <= 0:
            raise ValueError(f"bathymetry.dean_a must be positive, got {self.dean_a}")
        if self.grain_size <= 0:
            raise ValueError(f"bathymetry.grain_size must be positive, got {self.grain_size}")

    @property
    def a(self) -> float:
        """The Dean scale parameter actually in force [m^(1/3)]."""
        if self.dean_a is not None:
            return self.dean_a
        from .bathymetry import dean_A_for_grain_size

        return dean_A_for_grain_size(self.grain_size)


@dataclass(frozen=True)
class NearshoreConfig:
    breaker_index: float = 0.78
    foam_halflife: float = 3.0
    foam_coverage: float = 0.85
    """Coverage a continuously breaking cell settles at [0-1].

    The seeding rate is derived from this and ``foam_halflife`` rather than set
    directly, because the two are tied together by the equilibrium and setting
    a rate independently saturates at some half lives and not others.
    """
    refraction: str = "snell"
    """How refraction is applied: ``"snell"``, ``"blend"`` or ``"none"``.

    ``"snell"`` turns the waves *and* scales their height by the ray-convergence
    factor ``Kr``. ``"blend"`` turns them but leaves the height alone. ``"none"``
    does neither.

    The distinction matters on a complex coastline. ``Kr`` is derived for
    straight parallel depth contours, and it is computed here from the direction
    to the nearest shore -- which is discontinuous wherever a different piece of
    coast becomes the nearest one (the medial axis of the water body). Measured
    on a Strait of Hormuz export at 0.25 m, the p99.9 one-cell jump in wave
    amplitude was 0.375 under ``"snell"`` against 0.013 under ``"blend"``: a
    30x difference, visible as hard seams radiating from every shoreline
    concavity. Until refraction is solved properly (see docs/algorithms.md),
    ``"blend"`` is the safe choice for a real coast and ``"snell"`` is right for
    a smooth synthetic one.

    ``"rays"`` is the proper solution: :mod:`pywave.rays` integrates rays
    through the celerity field and measures the energy that arrives, so ``Kr``
    stops being assumed. It never reads ``shore_normal``, which is what removes
    the seams -- 0.0123 against ``snell``'s 0.375 -- while keeping the headland
    focusing that ``blend`` throws away. It is **not the default**: it needs a
    scene-level solve (seconds to minutes, cached), and on a smooth synthetic
    beach ``snell`` is exact and free, so ``rays`` there is equal and slower.

    ``true``/``false`` are still accepted and mean ``"snell"``/``"none"``.
    """
    shoaling: bool = True


@dataclass(frozen=True)
class LodRing:
    r: float
    dx: float


@dataclass(frozen=True)
class OutputConfig:
    fps: float = 30.0
    mesh_dx: float = 0.125
    #: Mesh every wet cell in the domain rather than a shoreline window.
    #: For renders where the camera position is not known in advance, so
    #: there is no window to centre on. `--mesh-full` sets it per run.
    mesh_full: bool = False
    #: Vertex guard. Deliberately low enough that an accidental
    #: whole-domain job at a fine spacing stops rather than swaps.
    mesh_max_vertices: int = 12_000_000
    lod_rings: tuple[LodRing, ...] = ()


@dataclass(frozen=True)
class Config:
    """Top-level scene configuration."""

    scene: SceneConfig
    wind: WindConfig
    spectrum: SpectrumConfig
    surface: SurfaceConfig
    nearshore: NearshoreConfig = field(default_factory=NearshoreConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    bathymetry: BathymetryConfig = field(default_factory=BathymetryConfig)
    source_path: Path | None = None

    @property
    def name(self) -> str:
        """Scene name, from the config filename.  ``unnamed`` if built in code."""
        return self.source_path.stem if self.source_path else "unnamed"

    # -- derived spectral quantities, so callers never re-derive them --------

    @property
    def jonswap(self) -> tuple[float, float, float]:
        """``(alpha, f_p, x_tilde)`` for this scene."""
        return jonswap_params(self.wind.speed, self.wind.fetch)

    @property
    def f_p(self) -> float:
        """Peak frequency [Hz]."""
        return self.jonswap[1]

    @property
    def omega_p(self) -> float:
        return 2.0 * np.pi * self.f_p

    @property
    def k_p(self) -> float:
        """Deep-water peak wavenumber [rad/m]."""
        return self.omega_p**2 / G

    @property
    def lambda_p(self) -> float:
        """Peak wavelength [m]."""
        return peak_wavelength(self.wind.speed, self.wind.fetch)

    def spectrum_kwargs(self) -> dict:
        """Keyword bundle accepted by the ``spectrum``/``moments`` entry points."""
        return {
            "u10": self.wind.speed,
            "fetch": self.wind.fetch,
            "gamma": self.spectrum.gamma,
        }


REFRACTION_MODES = ("snell", "blend", "none", "rays")


def _refraction_mode(raw) -> str:
    """Accept the mode by name, or the older boolean spelling."""
    if isinstance(raw, bool):
        return "snell" if raw else "none"
    mode = str(raw).strip().lower()
    if mode not in REFRACTION_MODES:
        raise ValueError(
            f"nearshore.refraction must be one of {REFRACTION_MODES} "
            f"(or true/false), got {raw!r}")
    return mode


def peak_wavelength(u10: float, fetch: float) -> float:
    """Deep-water peak wavelength [m] for a fetch-limited sea.

    Module level rather than only a ``Config`` property, because ``load_config``
    needs it while the ``Config`` is still being assembled -- ``size: auto``
    tiles are resolved against it before ``SurfaceConfig`` exists.
    """
    omega_p = 2.0 * np.pi * jonswap_params(u10, fetch)[1]
    return 2.0 * np.pi / (omega_p**2 / G)


def _is_auto(size) -> bool:
    return isinstance(size, str) and size.strip().lower() == "auto"


def _tile_from_dict(d: dict) -> TileConfig:
    return TileConfig(size=float(d["size"]), n=int(d["n"]),
                      band=(float(d["band"][0]), float(d["band"][1])))


def _tiles_from_raw(raw_tiles, lambda_p: float) -> tuple[TileConfig, ...]:
    """Build the tile set, resolving any ``size: auto`` against ``lambda_p``.

    A tile written ``{size: auto, n: 512, band: [0.0, 0.35]}`` takes its size
    from the scene's own peak wavelength instead of a number copied from another
    config -- which is how every mis-sized scene in this repository got that way.
    See ``docs/tile_autosizing.md``.

    Two rules, both enforced here rather than left to the reader:

    * **``auto`` tiles must be the low-band ones**, contiguously from the
      bottom.  The derivation solves the first tile exactly (it sets ``k_ref``,
      and therefore every band edge) and sizes the rest relative to it; an
      ``auto`` tile above a pinned one has nothing well-defined to be relative
      to.
    * **The top tile must not be ``auto``.**  It sets ``k_max``, where the
      mesh/BSDF handoff lands, which is a rendering decision and not a property
      of the sea.  Letting it follow ``lambda_p`` at fixed ``n`` coarsens its
      grid: on ``straits`` that cost 36% of the resolved slope variance.

    Configs with every size written out are untouched, which is what keeps
    existing scenes reproducible byte for byte.
    """
    from .tiling import derive_tile_sizes

    entries = [
        {"size": t["size"], "n": int(t["n"]),
         "band": (float(t["band"][0]), float(t["band"][1]))}
        for t in raw_tiles
    ]
    if not any(_is_auto(e["size"]) for e in entries):
        return tuple(_tile_from_dict(e) for e in entries)

    # Work in band order; the file may list tiles in any order, and which tile
    # is "first" is a statement about bands, not about lines in a YAML file.
    order = sorted(range(len(entries)), key=lambda i: entries[i]["band"][0])
    auto = [pos for pos, i in enumerate(order) if _is_auto(entries[i]["size"])]

    if auto != list(range(len(auto))):
        raise ValueError(
            "surface.tiles: `size: auto` tiles must be the lowest bands, with "
            "no pinned tile below an automatic one. The derivation solves the "
            "lowest-band tile exactly and sizes the others relative to it.")
    if len(auto) == len(entries):
        raise ValueError(
            "surface.tiles: the highest-band tile cannot be `size: auto`. It "
            "sets k_max, where the mesh/BSDF handoff lands, which is a "
            "rendering decision rather than a property of the sea -- give it a "
            "size chosen for the finest mesh that will sample it.")

    derived = derive_tile_sizes(
        lambda_p,
        [entries[order[p]]["n"] for p in auto],
        [entries[order[p]]["band"][1] for p in auto],
        pinned=[(float(entries[order[p]]["size"]), entries[order[p]]["n"])
                for p in range(len(auto), len(order))],
    )
    for pos, size in zip(auto, derived):
        entries[order[pos]]["size"] = size
    return tuple(_tile_from_dict(e) for e in entries)


def load_config(path: str | Path) -> Config:
    """Load and validate a scene YAML.

    Angles in the file are degrees; they are converted here and only here.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    scene_raw = raw["scene"]
    wind_raw = raw["wind"]
    spec_raw = raw.get("spectrum", {})
    surf_raw = raw["surface"]
    near_raw = raw.get("nearshore", {})
    out_raw = raw.get("output", {})
    bath_raw = raw.get("bathymetry", {})

    scene = SceneConfig(
        domain=(float(scene_raw["domain"][0]), float(scene_raw["domain"][1])),
        water_level=float(scene_raw["water_level"]),
        epsg=int(scene_raw["epsg"]),
    )
    wind = WindConfig(
        speed=float(wind_raw["speed"]),
        direction_rad=float(np.radians(float(wind_raw["direction"]))),
        fetch=float(wind_raw["fetch"]),
    )
    spectrum = SpectrumConfig(
        model=str(spec_raw.get("model", "jonswap")),
        gamma=float(spec_raw.get("gamma", JONSWAP_GAMMA)),
        spreading=str(spec_raw.get("spreading", "cos2s")),
        seed=int(spec_raw.get("seed", 20260801)),
    )
    surface = SurfaceConfig(
        # `size: auto` is resolved here, against this scene's own peak
        # wavelength, so nothing downstream ever sees a tile without a size.
        tiles=_tiles_from_raw(surf_raw["tiles"],
                              peak_wavelength(wind.speed, wind.fetch)),
        choppiness=float(surf_raw.get("choppiness", 1.0)),
    )
    nearshore = NearshoreConfig(
        breaker_index=float(near_raw.get("breaker_index", 0.78)),
        foam_halflife=float(near_raw.get("foam_halflife", 3.0)),
        foam_coverage=float(near_raw.get("foam_coverage", 0.85)),
        refraction=_refraction_mode(near_raw.get("refraction", "snell")),
        shoaling=bool(near_raw.get("shoaling", True)),
    )
    output = OutputConfig(
        fps=float(out_raw.get("fps", 30.0)),
        mesh_dx=float(out_raw.get("mesh_dx", 0.125)),
        mesh_full=bool(out_raw.get("mesh_full", False)),
        mesh_max_vertices=int(out_raw.get("mesh_max_vertices", 12_000_000)),
        lod_rings=tuple(
            LodRing(r=float(r["r"]), dx=float(r["dx"]))
            for r in out_raw.get("lod_rings", [])
        ),
    )
    bathymetry = BathymetryConfig(
        source=(None if bath_raw.get("source") is None
                else str(bath_raw["source"])),
        profile=str(bath_raw.get("profile", "planar")),
        shoreline=float(bath_raw.get("shoreline", 400.0)),
        dean_a=(None if bath_raw.get("dean_a") is None else float(bath_raw["dean_a"])),
        grain_size=float(bath_raw.get("grain_size", 0.25)),
        max_depth=float(bath_raw.get("max_depth", 5.0)),
        bank_slope=float(bath_raw.get("bank_slope", 0.08)),
        dx=float(bath_raw.get("dx", 1.0)),
        surf_dx=float(bath_raw.get("surf_dx", 0.25)),
        amplitude=float(bath_raw.get("amplitude", 40.0)),
        wavelength=float(bath_raw.get("wavelength", 400.0)),
    )
    return Config(scene, wind, spectrum, surface, nearshore, output, bathymetry,
                  source_path=path)
