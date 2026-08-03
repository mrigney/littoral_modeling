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
class NearshoreConfig:
    breaker_index: float = 0.78
    foam_halflife: float = 3.0
    refraction: bool = True
    shoaling: bool = True


@dataclass(frozen=True)
class LodRing:
    r: float
    dx: float


@dataclass(frozen=True)
class OutputConfig:
    fps: float = 30.0
    mesh_dx: float = 0.125
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
    source_path: Path | None = None

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
        refraction=bool(near_raw.get("refraction", True)),
        shoaling=bool(near_raw.get("shoaling", True)),
    )
    output = OutputConfig(
        fps=float(out_raw.get("fps", 30.0)),
        mesh_dx=float(out_raw.get("mesh_dx", 0.125)),
        lod_rings=tuple(
            LodRing(r=float(r["r"]), dx=float(r["dx"]))
            for r in out_raw.get("lod_rings", [])
        ),
    )
    return Config(scene, wind, spectrum, surface, nearshore, output, source_path=path)
