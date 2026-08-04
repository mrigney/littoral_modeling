"""Run a scene end to end and write every artifact it produces.

    python scripts/run_scene.py                              # the shipped test lake
    python scripts/run_scene.py configs/my_scene.yaml        # your own
    python scripts/run_scene.py my_scene.yaml --out /tmp/run # somewhere else
    python scripts/run_scene.py my_scene.yaml --quick        # figures only, no data
    python scripts/run_scene.py my_scene.yaml --animate      # add the video clips

Point it at a scene YAML and it produces, in one directory:

    summary.md          human-readable scene report, every number derived live
    summary.json        the same numbers, machine-readable
    gallery.md          the figures with captions, renders on GitHub
    overview.html       one self-contained page you can mail to someone
    figures/*.png       eight figures
    channels/*.npy      the per-cell nearshore fields, plus a manifest
    mesh/water_0000.ply   displaced water mesh + per-vertex channels  (--mesh)
    mesh/terrain_0000.ply the bed, co-registered with it               (--mesh)
    mesh/scene.py         a starter Mitsuba scene dict, loads both     (--mesh)
    mesh/scene.xml        the same scene as XML, for the CLI           (--mesh)
    open.mp4            open-water animation      (--animate)
    shore.mp4           shoreline animation       (--animate)

Nothing here is hard-coded to the shipped scene: change the wind, the fetch, the
beach, and every figure, number and channel follows. That is the point -- a
colleague should be able to write a config and see what it looks like without
reading any of the code.

Copy `configs/test_lake.yaml` as a starting point; every key is documented there
and in `docs/users_guide.md` section 6.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Windows consoles still default to cp1252, which raises on any non-ASCII
# character the moment a scene name or a unit symbol reaches stdout. Files are
# always written as UTF-8 explicitly; this only guards the terminal.
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
import make_overview as mo  # noqa: E402
from pywave import __version__, load_config, moments, nearshore, spectrum  # noqa: E402


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def scene_summary(scene) -> dict:
    """Every derived quantity worth reporting, computed from the config alone."""
    cfg = scene.cfg
    u10, fetch, gamma = cfg.wind.speed, cfg.wind.fetch, cfg.spectrum.gamma
    ts = scene.tileset
    beach = scene.bathy
    alpha, f_p, x_tilde = cfg.jonswap

    total = moments.mss_above(0.0, u10, fetch, gamma=gamma)
    above = moments.mss_above(ts.k_max, u10, fetch, gamma=gamma)
    up, cross = moments.mss_anisotropic(u10, fetch, cfg.wind.direction_rad, gamma=gamma)

    slope = beach.beach_slope()
    l0 = float(nearshore.deep_water_wavelength(1.0 / f_p))
    xi = float(nearshore.iribarren_number(slope, ts.hs(), l0))
    runup = float(nearshore.hunt_runup(xi, ts.hs()))

    mesh_dx = cfg.output.mesh_dx
    return {
        "scene": {
            "name": cfg.name,
            "domain_m": list(cfg.scene.domain),
            "water_level_m": cfg.scene.water_level,
            "epsg": cfg.scene.epsg,
            "wind_speed_ms": u10,
            "wind_direction_deg": cfg.wind.direction_deg,
            "fetch_m": fetch,
            "gamma": gamma,
            "spreading": cfg.spectrum.spreading,
            "seed": cfg.spectrum.seed,
        },
        "spectrum": {
            "dimensionless_fetch": x_tilde,
            "alpha": alpha,
            "f_p_hz": f_p,
            "T_p_s": 1.0 / f_p,
            "k_p_rad_m": cfg.k_p,
            "lambda_p_m": cfg.lambda_p,
            "L0_m": l0,
            "Hs_spectral_m": spectrum.hs_spectral(u10, fetch, gamma),
            "Hs_fetch_fit_m": spectrum.fetch_limited_hs(u10, fetch),
            "Tz_s": moments.zero_crossing_period(u10, fetch, gamma),
        },
        "slope": {
            "mss_total": total,
            "rms_slope_deg": float(np.degrees(np.arctan(np.sqrt(total)))),
            "mss_upwind": up,
            "mss_crosswind": cross,
            "anisotropy_ratio": up / cross,
            "cox_munk_mss": moments.cox_munk_from_u10(u10),
            "cox_munk_ratio": total / moments.cox_munk_from_u10(u10),
        },
        "surface": {
            "n_tiles": len(ts.tiles),
            "tiles": [
                {"size_m": t.size, "n": t.n,
                 "band_rad_m": [t.band[0], t.band[1]],
                 "k_nyquist_rad_m": t.k_nyquist,
                 "m0_m2": t.m0(), "mss": t.mss(),
                 "rotation_deg": float(np.degrees(t.rotation))}
                for t in ts.tiles
            ],
            "Hs_composite_m": ts.hs(),
            "k_max_rad_m": ts.k_max,
            "mss_resolved": ts.mss(),
            "mss_sub_grid": above,
            "lod_closure_rel": abs(ts.mss() + above - total) / total,
        },
        "lod": {
            "mesh_dx_m": mesh_dx,
            "mss_resolved_at_mesh": float(moments.mss_between(
                moments.K_MIN_DEFAULT, np.pi / mesh_dx, u10, fetch,
                gamma=gamma, order=2)),
            "mss_sub_mesh": float(moments.mss_above(np.pi / mesh_dx, u10, fetch,
                                                    gamma=gamma)),
            "beckmann_alpha_at_mesh": float(np.sqrt(moments.mss_above(
                np.pi / mesh_dx, u10, fetch, gamma=gamma))),
            "rings": [{"r_m": r.r, "dx_m": r.dx} for r in cfg.output.lod_rings],
        },
        "nearshore": {
            "profile": cfg.bathymetry.profile,
            "dean_A": cfg.bathymetry.a,
            "shoreline_y_m": cfg.bathymetry.shoreline,
            "max_depth_m": cfg.bathymetry.max_depth,
            "foreshore_slope": slope,
            "iribarren_xi": xi,
            "breaker_type": str(nearshore.breaker_type(xi)),
            "breaker_index": cfg.nearshore.breaker_index,
            "breaker_depth_m": float(nearshore.breaker_depth(
                ts.hs(), cfg.nearshore.breaker_index)),
            "runup_m": runup,
            "swash_width_m": float(nearshore.swash_width(runup, slope)),
            "foam_halflife_s": cfg.nearshore.foam_halflife,
        },
        "provenance": {
            "pywave_version": __version__,
            "generated_utc": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}",
            "git_sha": mo._git_sha(),
            "config": str(cfg.source_path) if cfg.source_path else None,
        },
    }


def _fmt(v):
    if isinstance(v, float):
        a = abs(v)
        if v == 0:
            return "0"
        if a < 1e-3 or a >= 1e5:
            return f"{v:.4e}"
        return f"{v:.5g}"
    if isinstance(v, list):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


_UNITS = {"_m": "m", "_s": "s", "_hz": "Hz", "_deg": "deg", "_ms": "m/s",
          "_m2": "m²", "_rad_m": "rad/m"}


def write_summary_md(summary: dict, path: Path) -> None:
    """A readable scene report -- the thing to skim before opening any figure."""
    s = summary
    lines = [
        f"# Scene report — `{s['scene']['name']}`",
        "",
        f"Generated {s['provenance']['generated_utc']} UTC by "
        f"`scripts/run_scene.py`, pywave {s['provenance']['pywave_version']}, "
        f"commit `{s['provenance']['git_sha']}`.",
        "",
        "Every number below is derived from the config at run time. Nothing is "
        "tuned independently: wave height, slope statistics and the nearshore "
        "channels all come from one spectrum.",
        "",
        "## The sea state",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Wind | {_fmt(s['scene']['wind_speed_ms'])} m/s toward "
        f"{_fmt(s['scene']['wind_direction_deg'])}° |",
        f"| Fetch | {_fmt(s['scene']['fetch_m'])} m |",
        f"| Dimensionless fetch X̃ | {_fmt(s['spectrum']['dimensionless_fetch'])} |",
        f"| Significant height Hs | **{_fmt(s['spectrum']['Hs_spectral_m'])} m** |",
        f"| Peak period Tp | {_fmt(s['spectrum']['T_p_s'])} s |",
        f"| Peak wavelength λp | {_fmt(s['spectrum']['lambda_p_m'])} m |",
        f"| Zero-crossing period Tz | {_fmt(s['spectrum']['Tz_s'])} s |",
        f"| Mean square slope | {_fmt(s['slope']['mss_total'])} "
        f"(RMS slope {_fmt(s['slope']['rms_slope_deg'])}°) |",
        f"| Upwind/crosswind anisotropy | {_fmt(s['slope']['anisotropy_ratio'])} |",
        f"| vs Cox–Munk | ×{_fmt(s['slope']['cox_munk_ratio'])} |",
        "",
        "`Hs` is the value obtained by integrating the spectrum, which is what a "
        "realised surface reproduces. The fetch-limited energy fit gives "
        f"{_fmt(s['spectrum']['Hs_fetch_fit_m'])} m; the two are not expected to "
        "agree, and the validation report explains why.",
        "",
        "## The surface",
        "",
        "| Tile | Size | N | Band [rad/m] | Wavelengths [m] | Hs [m] |",
        "|---|---|---|---|---|---|",
    ]
    for i, t in enumerate(s["surface"]["tiles"], 1):
        lo, hi = t["band_rad_m"]
        lam_hi = "∞" if lo <= 0 else f"{2 * np.pi / lo:.2f}"
        lines.append(
            f"| {i} | {_fmt(t['size_m'])} m | {t['n']} | "
            f"{lo:.2f} – {hi:.2f} | {2 * np.pi / hi:.2f} – {lam_hi} | "
            f"{4 * np.sqrt(t['m0_m2']):.4f} |")
    lines += [
        "",
        f"Composite Hs **{_fmt(s['surface']['Hs_composite_m'])} m**, resolving "
        f"wavenumbers to {_fmt(s['surface']['k_max_rad_m'])} rad/m.",
        "",
        "## Level of detail",
        "",
        f"At the configured mesh spacing of {_fmt(s['lod']['mesh_dx_m'])} m, "
        f"{100 * s['lod']['mss_sub_mesh'] / s['slope']['mss_total']:.0f}% of the "
        "slope variance is below the mesh and must be carried by the BSDF as "
        f"roughness — Beckmann α = {_fmt(s['lod']['beckmann_alpha_at_mesh'])}.",
        "",
        f"The invariant `mss_resolved + mss_above(k_max) = mss_total` closes to "
        f"{s['surface']['lod_closure_rel']:.2e}.",
        "",
        "## The nearshore",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Profile | {s['nearshore']['profile']}, Dean A = "
        f"{_fmt(s['nearshore']['dean_A'])} |",
        f"| Foreshore slope | {100 * s['nearshore']['foreshore_slope']:.1f}% |",
        f"| Iribarren number ξ | {_fmt(s['nearshore']['iribarren_xi'])} "
        f"(**{s['nearshore']['breaker_type']}** breakers) |",
        f"| Breaking depth | {_fmt(s['nearshore']['breaker_depth_m'])} m |",
        f"| Runup (Hunt) | {_fmt(s['nearshore']['runup_m'])} m vertical |",
        f"| Swash excursion | {_fmt(s['nearshore']['swash_width_m'])} m horizontal |",
        f"| Foam half life | {_fmt(s['nearshore']['foam_halflife_s'])} s |",
        "",
        "## Files in this directory",
        "",
        "| File | What it is |",
        "|---|---|",
        "| `overview.html` | One self-contained page. Open it in a browser; mail it if useful. |",
        "| `gallery.md` | The same figures with captions, renders on GitHub. |",
        "| `figures/` | Eight PNGs at 150 dpi. |",
        "| `channels/` | Per-cell nearshore fields as `.npy`, with `manifest.json`. |",
        "| `summary.json` | Everything in this report, machine-readable. |",
        "| `mesh/` | Water + terrain PLYs and a starter Mitsuba scene, if `--mesh` was passed. |",
        "| `open.*` / `shore.*` | Animations, if `--animate` was passed. |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Data products
# ---------------------------------------------------------------------------


def write_channels(scene, out_dir: Path) -> dict:
    """Per-cell nearshore fields -- what Phase 6 will pack into mesh attributes.

    Written on the refined grid, because the surf zone is about a metre wide and
    a 1 m grid cannot resolve it (cookbook section 4.4's terracing warning).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = scene.cfg
    b = scene.fine_bathy
    ny, nx = b.meta.shape

    ax_x, ax_y = b.meta.axes()
    X, Y = np.meshgrid(ax_x, ax_y, indexing="xy")
    nf = nearshore.transform(scene.onshore_tileset, b, scene.onshore_cfg,
                             X.ravel(), Y.ravel(), 0.0)
    foam, _ = scene.foam_field()

    def r(a):
        return np.asarray(a, dtype=np.float32).reshape(ny, nx)

    channels = {
        "depth": (r(b.depth), "m", "still-water depth; positive in water"),
        "sdf": (r(b.sdf), "m", "signed distance to the waterline; positive inland"),
        "shore_normal_x": (r(b.shore_normal[0]), "-", "unit shore normal, X, inland"),
        "shore_normal_y": (r(b.shore_normal[1]), "-", "unit shore normal, Y, inland"),
        "hs_local": (r(nf.hs_local), "m", "local significant height after transform"),
        "shoaling_gain": (r(nf.shoaling), "-", "Hs_local / Hs_deep"),
        "breaking": (r(nf.breaking), "-", "1 where the depth-limited height is exceeded"),
        "wetness": (r(nf.wetness), "-", "fraction of a wave period submerged"),
        "foam": (foam.astype(np.float32), "-", "surf-zone foam coverage"),
        "elevation": (r(nf.surface.h), "m", "wave elevation about z_w at t = 0"),
        "slope_x": (r(nf.surface.slope_x), "-", "dh/dx at t = 0"),
        "slope_y": (r(nf.surface.slope_y), "-", "dh/dy at t = 0"),
    }

    manifest = {
        "grid": {
            "shape": [ny, nx],
            "dx_m": b.meta.dx,
            "origin_xy_m": list(b.meta.origin),
            "extent_m": list(b.meta.extent),
            "water_level_m": b.meta.water_level,
            "epsg": b.meta.epsg,
            "index_order": "[y, x]",
        },
        "time_s": 0.0,
        "wind_direction_deg": float(np.degrees(scene.onshore_cfg.wind.direction_rad)),
        "note": ("Channels are evaluated with the wind blowing straight onshore so "
                 "shoaling is isolated from refraction; the deep-water surface in "
                 "the figures uses the config's own wind direction."),
        "channels": {},
    }
    for name, (arr, unit, desc) in channels.items():
        np.save(out_dir / f"{name}.npy", arr)
        manifest["channels"][name] = {
            "file": f"{name}.npy", "unit": unit, "description": desc,
            "dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr)),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------


def _default_region(scene, posts: int = 900):
    """A shoreline window sized so the default spacing stays affordable.

    Meshing the whole domain at `output.mesh_dx` is usually millions of posts
    (the shipped lake is 64 M), so the default is a window on the waterline --
    which is the interesting part and the part LOD rings would otherwise be
    protecting.
    """
    cfg = scene.cfg
    dx = cfg.output.mesh_dx
    span = posts * dx
    b = scene.fine_bathy
    x0b, x1b, y0b, y1b = b.meta.extent
    cx = 0.5 * (x0b + x1b)
    shore = cfg.bathymetry.shoreline
    return (max(cx - 0.5 * span, x0b), max(shore - 0.75 * span, y0b),
            min(cx + 0.5 * span, x1b), min(shore + 0.1 * span, y1b))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=str(mf.CONFIG),
                    help="scene YAML (default: configs/test_lake.yaml)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: runs/<scene name>)")
    ap.add_argument("--quick", action="store_true",
                    help="skip the channel export and the HTML page")
    ap.add_argument("--mesh", action="store_true",
                    help="also build and export the water mesh (PLY + JSON)")
    ap.add_argument("--mesh-dx", type=float, default=None,
                    help="mesh post spacing [m]; defaults to output.mesh_dx")
    ap.add_argument("--mesh-region", type=float, nargs=4, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"),
                    help="bound the mesh to a region; spacing enters "
                         "quadratically, so the whole domain at a fine spacing "
                         "is usually far too much geometry")
    ap.add_argument("--mesh-obj", action="store_true",
                    help="also write an OBJ (geometry only -- no channels)")
    ap.add_argument("--animate", action="store_true",
                    help="also render the open-water and shoreline clips "
                         "(adds a few minutes)")
    ap.add_argument("--animate-seconds", type=float, default=5.0)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        ap.error(f"no such config: {cfg_path}")

    t_start = time.perf_counter()
    print(f"loading {cfg_path}")
    try:
        cfg = load_config(cfg_path)
    except Exception as exc:                      # config errors are user errors
        print(f"\n  config rejected: {exc}\n\n"
              f"  See docs/users_guide.md section 6 for every key, or copy\n"
              f"  configs/test_lake.yaml and edit it.", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else ROOT / "runs" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    scene = mf.Scene(cfg)

    print(f"  {cfg.wind.speed:g} m/s toward {cfg.wind.direction_deg:g} deg, "
          f"fetch {cfg.wind.fetch:g} m")
    print(f"  output -> {out}")

    print("checking the bathymetry contract ...", flush=True)
    stats = scene.bathy.validate()
    print(f"  ok: |shore_normal| error {stats['shore_normal_magnitude_error']:.1e}, "
          f"sdf sign disagreement {stats['sdf_sign_disagreement']:.0%}")

    print("computing scene summary ...", flush=True)
    summary = scene_summary(scene)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")
    write_summary_md(summary, out / "summary.md")
    sp, ns = summary["spectrum"], summary["nearshore"]
    print(f"  Hs {sp['Hs_spectral_m']:.4f} m, Tp {sp['T_p_s']:.2f} s, "
          f"lambda_p {sp['lambda_p_m']:.2f} m, {ns['breaker_type']} breakers")

    print(f"rendering {len(mf._REGISTRY)} figures ...", flush=True)
    written = mf.build(list(mf._REGISTRY), scene, out / "figures", args.dpi)
    mf.write_gallery(written, cfg, out / "gallery.md")

    if not args.quick:
        print("exporting nearshore channels ...", flush=True)
        man = write_channels(scene, out / "channels")
        print(f"  {len(man['channels'])} channels on a "
              f"{man['grid']['shape'][0]}x{man['grid']['shape'][1]} grid "
              f"at {man['grid']['dx_m']:g} m")

        print("building the self-contained page ...", flush=True)
        mo.build_page(scene, out / "overview.html")

    if args.mesh:
        from pywave import export as pw_export
        from pywave import mesh as pw_mesh

        print("building the water mesh ...", flush=True)
        region = tuple(args.mesh_region) if args.mesh_region else _default_region(scene)
        foam_field, _ = scene.foam_field()
        mesh_dir = out / "mesh"

        wm = pw_mesh.build_water_mesh(
            scene.tileset, scene.fine_bathy, cfg, t=0.0,
            dx=args.mesh_dx, region=region,
            foam=foam_field, foam_bathy=scene.fine_bathy)
        wstats = wm.validate()
        written = pw_export.export_frame(wm, mesh_dir, "water_0000",
                                         obj=args.mesh_obj)
        print(f"  water   {wstats['n_vertices']:>9,} v {wstats['n_faces']:>9,} f "
              f"at {wm.meta['mesh_dx']:g} m over "
              f"{region[2] - region[0]:.0f} x {region[3] - region[1]:.0f} m")

        # The bed, on the same grid and slightly larger, so the water has
        # something to sit on and nothing shows through at the edges.
        print("building the terrain mesh ...", flush=True)
        tm = pw_mesh.build_terrain_mesh(scene.fine_bathy, cfg, dx=args.mesh_dx,
                                        region=region, tileset=scene.tileset)
        tstats = tm.validate()
        written["terrain_ply"] = pw_export.write_ply(
            tm, mesh_dir / "terrain_0000.ply")
        written["terrain_json"] = pw_export.write_frame_metadata(
            tm, mesh_dir / "terrain_0000.json")
        if args.mesh_obj:
            written["terrain_obj"] = pw_export.write_obj(
                tm, mesh_dir / "terrain_0000.obj", quiet=True)
        print(f"  terrain {tstats['n_vertices']:>9,} v {tstats['n_faces']:>9,} f")

        params = pw_export.mitsuba_scene_params(
            "water_0000.ply", "terrain_0000.ply", water_mesh=wm, terrain_mesh=tm)
        written["scene_py"] = pw_export.write_mitsuba_scene(
            mesh_dir / "scene.py", "water_0000.ply", "terrain_0000.ply",
            params=params)
        written["scene_xml"] = pw_export.write_mitsuba_xml(
            mesh_dir / "scene.xml", "water_0000.ply", "terrain_0000.ply",
            params=params)

        for kind, path in written.items():
            print(f"  {kind:12s} {path.stat().st_size / 1024 / 1024:7.2f} MiB  "
                  f"{path.name}")

    if args.animate:
        import animate as anm

        fmt = "mp4" if anm._ffmpeg_path() else "gif"
        fps, px = 20.0, 640
        n = max(int(round(args.animate_seconds * fps)), 2)
        times = np.arange(n) / fps
        print(f"rendering animations ({n} frames, {fmt}) ...", flush=True)
        for mode in ("open", "shore"):
            producer = (anm.OpenWater(scene, px) if mode == "open"
                        else anm.Shoreline(scene, px, fps))
            frames = [producer.frame(float(t)) for t in times]
            path = anm.write_frames(frames, out / f"{mode}.{fmt}", fps, fmt)
            print(f"  {path.name}  {path.stat().st_size / 1024 / 1024:.2f} MiB")

    print(f"\ndone in {time.perf_counter() - t_start:.0f} s")
    print(f"  start here:  {out / 'summary.md'}")
    if not args.quick:
        print(f"  or open:     {out / 'overview.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
