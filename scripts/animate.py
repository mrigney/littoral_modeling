"""Render short animations of the sea surface.

    python scripts/animate.py                             # both views, test lake
    python scripts/animate.py --mode shore                # just the shoreline
    python scripts/animate.py configs/coastal_bay.yaml    # a different scene
    python scripts/animate.py --seconds 8 --fps 25 --format mp4

Two views:

``open``
    Open water, looking straight down. What the composite surface actually does
    over time. Waves travel along the wind, and because the surface is a pure
    function of ``t`` you can start anywhere -- ``--start 3600`` is an hour in
    and costs the same as ``--start 0``.

``shore``
    A plan view across the waterline: shoaling waves, the breaker line, foam in
    the surf band, and the swash edge breathing at the peak period over wet
    sand. This is the view that shows what the nearshore model is for.

Output is MP4 when ffmpeg is available and animated GIF otherwise; force it with
``--format``. Frames are composited directly as RGB arrays rather than through
matplotlib, so the cost is dominated by sampling the surface.

A note on what animates and what does not
-----------------------------------------
The wave surface, the swash edge and the wet-sand band all evolve with ``t``.
Foam does not pulse, because breaking in this model is a *statistical* condition
(``Hs_local > gamma_b d``) rather than an instantaneous one, so the seeding
region is steady and the foam field sits at its equilibrium. Individual breaking
events would need a wave-by-wave model, which is well outside Phase 5.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_figures as mf  # noqa: E402
from pywave import foam as foam_mod  # noqa: E402
from pywave import load_config, nearshore, spectrum, tiling  # noqa: E402

DRY_SAND = np.array([0.78, 0.71, 0.57])
WET_SAND = np.array([0.42, 0.36, 0.28])
SHALLOW = np.array([0.62, 0.83, 0.85])
FOAM = np.array([0.97, 0.98, 0.99])


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def _burn_scalebar(rgb, metres_per_px, label_px=None):
    """Draw a scale bar into the frame.

    Burned into the pixels rather than drawn as a matplotlib artist so the frame
    is a plain array the whole way through -- one less thing to go wrong between
    the surface and the file.
    """
    h, w = rgb.shape[:2]
    target = 0.18 * w * metres_per_px
    nice = 10.0 ** np.floor(np.log10(max(target, 1e-9)))
    for mult in (1.0, 2.0, 5.0, 10.0):
        if nice * mult >= target:
            nice *= mult
            break
    length = int(round(nice / metres_per_px))
    length = max(min(length, w // 3), 10)

    x0, y0 = int(0.04 * w), int(0.94 * h)
    rgb[y0 - 3:y0, x0:x0 + length] = 1.0
    rgb[y0 - 6:y0, x0:x0 + 2] = 1.0
    rgb[y0 - 6:y0, x0 + length - 2:x0 + length] = 1.0
    return rgb, nice, (x0, y0, length)


def _stamp(rgb, text, origin=(0.04, 0.05), scale=2):
    """Write a short ASCII string into the frame with a 5x7 bitmap font.

    Tiny on purpose: pulling in a font file to print "t = 1.20 s" would be a
    dependency and a licence question for one line of text per frame.
    """
    font = {
        "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
        "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
        "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
        "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
        "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
        "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
        "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
        "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
        "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
        "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
        ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
        "=": ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
        "s": ("00000", "00000", "01111", "10000", "01110", "00001", "11110"),
        "t": ("01000", "01000", "11110", "01000", "01000", "01001", "00110"),
        "m": ("00000", "00000", "11010", "10101", "10101", "10101", "10101"),
        " ": ("00000",) * 7,
        "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    }
    h, w = rgb.shape[:2]
    px, py = int(origin[0] * w), int(origin[1] * h)
    for ch in text:
        glyph = font.get(ch)
        if glyph is None:
            px += 6 * scale
            continue
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit == "1":
                    y, x = py + r * scale, px + c * scale
                    if 0 <= y < h - scale and 0 <= x < w - scale:
                        rgb[y:y + scale, x:x + scale] = 1.0
        px += 6 * scale
    return rgb


def _to_u8(rgb):
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Frame producers
# ---------------------------------------------------------------------------


class OpenWater:
    """Looking straight down at open water."""

    def __init__(self, scene, px=560):
        self.scene = scene
        self.extent = scene.view_extent
        ax = np.linspace(0.0, self.extent, px)
        self.X, self.Y = np.meshgrid(ax, ax)
        self.mpp = self.extent / px
        self.ts = scene.tileset

    def frame(self, t: float):
        fields = self.ts.evaluate_grids(t)
        f = self.ts.sample(self.X, self.Y, t, fields=fields)
        # Flip to image orientation (row 0 = top) *before* anything is drawn
        # into the pixels: stamping first and flipping afterwards mirrors the
        # text, which is exactly what it looks like.
        rgb = mf._shade(f.h, f.slope_x, f.slope_y)[::-1]
        rgb, nice, _ = _burn_scalebar(rgb, self.mpp)
        _stamp(rgb, f"t={t:5.2f}s")
        _stamp(rgb, f"{nice:.0f}m", origin=(0.04, 0.885), scale=2)
        return _to_u8(rgb)


class Shoreline:
    """Plan view across the waterline: shoaling, breaking, foam, swash."""

    def __init__(self, scene, px=560, fps=20.0):
        self.scene = scene
        cfg = scene.cfg
        self.cfg = cfg
        self.bathy = scene.fine_bathy

        # Size the window on the *nearshore* scales, not on the wavelength.
        # The surf zone here is about a metre wide and the swash 0.4 m; a window
        # sized at fourteen peak wavelengths puts both below one pixel, which is
        # the physical point of the project but makes a useless animation.
        # Wide enough to show waves arriving, tight enough to see them break.
        #
        # Floored at forty bathymetry cells, which matters for loaded terrain:
        # a 10 m window on 1 m posts is ten cells across, and the shoreline
        # renders as a smooth bilinear blob rather than a coastline. The
        # synthetic beach at 0.25 m posts already clears this comfortably.
        cross = max(25.0 * scene.swash_band, 6.0 * cfg.lambda_p,
                    40.0 * self.bathy.meta.dx)
        along = cross * 1.5
        x0, shore = scene.shore_ref

        nx = px
        ny = max(int(px * cross / along), 64)
        xs = np.linspace(x0 - 0.5 * along, x0 + 0.5 * along, nx)
        # Put the waterline about a third of the way down rather than hard
        # against the top edge: the surf band and the swash are the subject, and
        # they sit within a metre or two of it.
        ys = np.linspace(shore - 0.68 * cross, shore + 0.32 * cross, ny)
        self.X, self.Y = np.meshgrid(xs, ys)
        self.mpp = along / nx

        # Crop the bathymetry to the window before anything iterates over it.
        # Foam advection touches every cell every frame; on a 4 km scene that is
        # 3.5 M cells to animate a 200 m view, and it dominated the run time.
        pad = 0.15 * cross
        self.bathy = self.bathy.crop((xs[0] - pad, xs[-1] + pad),
                                     (ys[0] - pad, ys[-1] + pad))

        self.depth, self.sdf, _ = self.bathy.sample(self.X, self.Y)
        self.water = self.depth > 0.0
        self.swash = scene.swash_band
        self.period = 1.0 / cfg.f_p

        # Foam: spun up once on the cropped grid, then stepped with the
        # animation. Sequential generation is exactly the case where that is
        # legitimate -- the bounded-spin-up machinery exists for random access.
        omega = 2.0 * np.pi * cfg.f_p
        d = np.maximum(self.bathy.depth, 1e-3)
        self.cg = spectrum.group_velocity(spectrum.dispersion_k(omega, d), d)
        ks = nearshore.shoaling_coefficient(omega, np.maximum(self.bathy.depth, 1e-9))
        hs = np.where(self.bathy.depth > 0.0, scene.onshore_tileset.hs() * ks, 0.0)
        self.brk = nearshore.breaking_mask(hs, self.bathy.depth,
                                           cfg.nearshore.breaker_index)
        self.model = foam_mod.FoamModel(
            bathy=self.bathy, half_life=cfg.nearshore.foam_halflife,
            equilibrium=cfg.nearshore.foam_coverage)
        self.foam_state = self.model.evaluate(lambda tt: self.brk, self.cg, t=0.0)
        self.dt = 1.0 / fps

    def _sample_grid(self, field):
        """Bilinear sample of a bathymetry-grid field at the view coordinates."""
        from scipy.ndimage import map_coordinates

        m = self.bathy.meta
        x0, y0 = m.origin
        coords = np.stack([(self.Y - y0) / m.dx - 0.5, (self.X - x0) / m.dx - 0.5])
        return map_coordinates(field, coords, order=1, mode="nearest")

    def frame(self, t: float):
        cfg = self.cfg
        nf = nearshore.transform(self.scene.onshore_tileset, self.bathy,
                                 self.scene.onshore_cfg, self.X, self.Y, t)

        # Land: dry sand darkened by the instantaneous swash, so the wet band
        # advances and retreats at the peak period.
        wet = nearshore.wetness(self.sdf, self.swash, t, self.period)
        rgb = DRY_SAND + (WET_SAND - DRY_SAND) * wet[..., None]

        # Water: shaded surface, lightened as it shallows so the beach reads.
        water_rgb = mf._shade(nf.surface.h, nf.surface.slope_x, nf.surface.slope_y)
        # Lighten as the water shallows so the beach reads as a beach, scaled
        # by depth relative to the breaker line rather than the wavelength --
        # and kept gentle, because a strong tint flattens the wave shading that
        # is the whole point of the frame.
        d_break = max(self.scene.tileset.hs() / cfg.nearshore.breaker_index, 1e-6)
        shallow_f = np.clip(1.0 - self.depth / (6.0 * d_break), 0.0, 1.0)
        water_rgb = water_rgb + (SHALLOW - water_rgb) * (0.30 * shallow_f)[..., None]
        rgb = np.where(self.water[..., None], water_rgb, rgb)

        # Foam, advected and decayed one step per frame.
        self.foam_state = self.model.step(self.foam_state, self.brk, self.cg, self.dt)
        cover = np.clip(self._sample_grid(self.foam_state), 0.0, 1.0)
        rgb = rgb + (FOAM - rgb) * cover[..., None]

        rgb = rgb[::-1]                     # -> image orientation, see OpenWater
        rgb, nice, _ = _burn_scalebar(rgb, self.mpp)
        _stamp(rgb, f"t={t:5.2f}s")
        _stamp(rgb, f"{nice:.0f}m", origin=(0.04, 0.885), scale=2)
        return _to_u8(rgb)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _ffmpeg_path() -> str | None:
    """ffmpeg on PATH, or the copy `imageio-ffmpeg` bundles, or nothing."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def write_frames(frames, path: Path, fps: float, fmt: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "mp4":
        # h.264 requires even dimensions; an odd --px silently fails otherwise.
        h, w = frames[0].shape[:2]
        if h % 2 or w % 2:
            frames = [f[:h - (h % 2), :w - (w % 2)] for f in frames]
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.animation as anim
        import matplotlib.pyplot as plt

        exe = _ffmpeg_path()
        if exe:
            matplotlib.rcParams["animation.ffmpeg_path"] = exe

        h, w = frames[0].shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        im = ax.imshow(frames[0], interpolation="nearest")

        def upd(i):
            im.set_data(frames[i])
            return (im,)

        a = anim.FuncAnimation(fig, upd, frames=len(frames), blit=True)
        # CRF rather than a fixed bitrate: wave texture is high-entropy, so a
        # constant bitrate either starves the busy frames or wastes space on the
        # calm ones. `yuv420p` is not optional -- without it ffmpeg picks a
        # 4:4:4 profile that QuickTime, PowerPoint and Safari all refuse to
        # play, which is a miserable thing to discover in a meeting.
        writer = anim.FFMpegWriter(
            fps=fps, codec="libx264",
            extra_args=["-crf", "20", "-preset", "medium",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        a.save(str(path), writer=writer)
        plt.close(fig)
        return path

    import numpy as _np
    from PIL import Image

    # One palette for the whole clip, built from frames sampled across it, and
    # no dithering. A per-frame palette costs ~40% more because consecutive
    # frames stop sharing byte runs, and dithering adds noise that defeats LZW
    # entirely. Wave texture is close to worst case for GIF either way.
    imgs = [Image.fromarray(f) for f in frames]
    step = max(len(frames) // 8, 1)
    sample = Image.fromarray(_np.concatenate(frames[::step], axis=0))
    palette = sample.quantize(colors=64, method=Image.MEDIANCUT)
    quantised = [im.quantize(palette=palette, dither=Image.NONE) for im in imgs]
    quantised[0].save(path, save_all=True, append_images=quantised[1:],
                      duration=int(round(1000.0 / fps)), loop=0, optimize=True)
    return path


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=str(mf.CONFIG))
    ap.add_argument("--mode", choices=("open", "shore", "both"), default="both")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--start", type=float, default=0.0,
                    help="scenario time of the first frame [s]")
    ap.add_argument("--px", type=int, default=480, help="frame width in pixels")
    ap.add_argument("--format", choices=("auto", "mp4", "gif"), default="auto")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        ap.error(f"no such config: {cfg_path}")
    cfg = load_config(cfg_path)
    scene = mf.Scene(cfg)

    out = Path(args.out) if args.out else ROOT / "runs" / cfg.name
    fmt = args.format
    if fmt == "auto":
        fmt = "mp4" if _ffmpeg_path() else "gif"
        if fmt == "gif":
            print("no ffmpeg -> writing GIF. Wave texture compresses badly in "
                  "GIF, so expect a few MB; `pip install imageio-ffmpeg` gets "
                  "you MP4 at roughly a tenth the size.")

    n = max(int(round(args.seconds * args.fps)), 2)
    times = args.start + np.arange(n) / args.fps
    modes = ("open", "shore") if args.mode == "both" else (args.mode,)

    print(f"scene {cfg.name}: {n} frames at {args.fps:g} fps "
          f"({args.seconds:g} s from t = {args.start:g} s)")

    for mode in modes:
        t0 = time.perf_counter()
        print(f"  {mode}: building ...", end="", flush=True)
        producer = (OpenWater(scene, args.px) if mode == "open"
                    else Shoreline(scene, args.px, args.fps))
        frames = []
        for i, t in enumerate(times):
            frames.append(producer.frame(float(t)))
            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)
        path = write_frames(frames, out / f"{mode}.{fmt}", args.fps, fmt)
        print(f" {path.name}  {path.stat().st_size / 1024 / 1024:.2f} MiB  "
              f"({time.perf_counter() - t0:.0f} s)")

    print(f"\nwrote to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
