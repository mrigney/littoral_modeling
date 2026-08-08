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
        return 2.0 * np.pi / self.k_p

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


def _tile_from_dict(d: dict) -> TileConfig:
    return TileConfig(size=float(d["size"]), n=int(d["n"]),
                      band=(float(d["band"][0]), float(d["band"][1])))


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
        tiles=tuple(_tile_from_dict(t) for t in surf_raw["tiles"]),
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
