"""Render a time sequence of the water surface, in one process.

The naive way to make a movie is to run ``run_scene.py`` once per frame. Don't:
each invocation reloads the bathymetry, rebuilds the tile set, redoes the foam
spin-up and rewrites a terrain mesh that never changes. On a 700 m domain at
0.25 m that is ~94 s and 365 MB of PLY *per frame*, almost none of which is the
frame.

Four facts make this cheap instead:

1. **Terrain is static.** Written once.
2. **Water topology is constant.** ``water_extent_mask`` depends on depth, not
   on time, so the faces are identical every frame. The Mitsuba scene is built
   once and only its vertex buffers are overwritten -- no PLY write, no PLY
   parse, no scene rebuild.
3. **The surface is a pure function of t.** Frames are independent.
4. **Foam is the only thing with state.** Spun up once, then stepped by ``dt``.
   Re-spinning per frame would cost more than everything else combined.

Usage
-----
::

    python scripts/animate_frames.py configs/straits_crop.yaml \\
        --seconds 5 --fps 30 --mesh-dx 0.5 \\
        --render-module path/to/scene_lwir.py --variant llvm_ad_mono \\
        --out runs/straits_crop/anim

``--render-module`` is your own scene script. It must expose ``scene_dict()``
returning a Mitsuba scene dictionary -- the generated ``mesh/scene.py`` already
does, and an LWIR variant of it works unchanged. It is called **after** the
variant is set, so ``mi.ScalarTransform4f`` is available to it.

``--dry-run`` does the whole mesh pipeline and skips Mitsuba entirely, which is
the way to check timings and vertex counts on a machine with no renderer.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pywave import config as pw_config          # noqa: E402
from pywave import export as pw_export          # noqa: E402
from pywave import foam as pw_foam              # noqa: E402
from pywave import mesh as pw_mesh              # noqa: E402
from pywave import nearshore, spectrum, tiling  # noqa: E402
from pywave.bathymetry import Bathymetry        # noqa: E402


# ---------------------------------------------------------------------------
# Foam: spun up once, then marched
# ---------------------------------------------------------------------------


class FoamTrack:
    """Foam coverage carried across frames rather than recomputed each one.

    The spin-up is the expensive part (a minute or so) and it only has to happen
    once: after it, each frame is a single semi-Lagrangian advect/decay/seed
    step, which is milliseconds.
    """

    def __init__(self, cfg, bathy, tileset, t0: float = 30.0):
        omega = 2.0 * np.pi * cfg.f_p
        d = np.maximum(bathy.depth, 1e-3)
        self.cg = spectrum.group_velocity(spectrum.dispersion_k(omega, d), d)
        ks = nearshore.shoaling_coefficient(omega, np.maximum(bathy.depth, 1e-9))
        hs = np.where(bathy.depth > 0.0, tileset.hs() * ks, 0.0)
        self.breaking = nearshore.breaking_mask(hs, bathy.depth,
                                                cfg.nearshore.breaker_index)
        self.model = pw_foam.FoamModel(
            bathy=bathy, half_life=cfg.nearshore.foam_halflife,
            equilibrium=cfg.nearshore.foam_coverage)
        self.field = self.model.evaluate(lambda tt: self.breaking, self.cg, t=t0)

    def step(self, dt: float) -> np.ndarray:
        self.field = self.model.step(self.field, self.breaking, self.cg, dt)
        return self.field


# ---------------------------------------------------------------------------
# Mitsuba plumbing
# ---------------------------------------------------------------------------


def load_render_module(path: Path):
    spec = importlib.util.spec_from_file_location("user_scene", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import a scene module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_scene"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "scene_dict"):
        raise RuntimeError(
            f"{path} does not define scene_dict(); this script needs a function "
            f"returning a Mitsuba scene dictionary. The generated mesh/scene.py "
            f"is the reference shape.")
    return mod


def find_keys(params, shape_id: str) -> dict:
    """Locate the buffers we have to overwrite each frame.

    Mitsuba's key naming for mesh attributes has moved between versions, and
    this cannot be tested without a renderer present -- so discover the keys
    rather than assume them, and say plainly what was found.
    """
    keys = [str(k) for k in params.keys()]
    mine = [k for k in keys if k.startswith(shape_id + ".")]
    if not mine:
        raise RuntimeError(
            f"no scene parameters start with {shape_id!r}. The water shape in "
            f"your scene dict must use that key. Available: "
            f"{sorted({k.split('.')[0] for k in keys})}")

    def pick(*cands):
        for c in cands:
            full = f"{shape_id}.{c}"
            if full in keys:
                return full
        return None

    found = {
        "positions": pick("vertex_positions", "vertex_positions_buf"),
        "normals": pick("vertex_normals", "vertex_normals_buf"),
        "foam": pick("foam", "mesh_attribute_foam", "vertex_foam"),
    }
    if found["positions"] is None:
        raise RuntimeError(
            f"could not find a vertex position buffer among: {mine}")
    return found


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--t0", type=float, default=0.0,
                    help="scenario time of the first frame [s]")
    ap.add_argument("--mesh-dx", type=float, default=None)
    ap.add_argument("--mesh-region", type=float, nargs=4, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"),
                    help="bound the water mesh; strongly recommended -- mesh "
                         "only what the camera can see")
    ap.add_argument("--mesh-max-vertices", type=int, default=None)
    ap.add_argument("--terrain-ply", type=str, default=None,
                    help="use this bed instead of building one")
    ap.add_argument("--no-foam", action="store_true",
                    help="skip the foam spin-up; sensible if your BSDF does not "
                         "read the foam channel yet, and it saves ~1 min")
    ap.add_argument("--render-module", type=Path, default=None)
    ap.add_argument("--variant", default="llvm_ad_rgb")
    ap.add_argument("--spp", type=int, default=None)
    ap.add_argument("--shape-id", default="water",
                    help="key of the water shape in your scene dict")
    ap.add_argument("--ext", default="exr", help="output image extension")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every frame's geometry, render nothing")
    ap.add_argument("--emit-ply", action="store_true",
                    help="also write a PLY per frame (large; for debugging)")
    args = ap.parse_args()

    cfg = pw_config.load_config(args.config)
    scene_name = args.config.stem
    out = args.out or Path("runs") / scene_name / "anim"
    out.mkdir(parents=True, exist_ok=True)

    n_frames = max(int(round(args.seconds * args.fps)), 1)
    dt = 1.0 / args.fps
    dx = args.mesh_dx if args.mesh_dx else cfg.output.mesh_dx

    print(f"scene {scene_name}: {n_frames} frames at {args.fps:g} fps "
          f"({args.seconds:g} s), mesh {dx:g} m")
    print(f"  lambda_p {cfg.lambda_p:.2f} m -> {cfg.lambda_p / dx:.1f} posts per "
          f"peak wave; Tp {2 * np.pi / cfg.omega_p:.2f} s -> "
          f"{2 * np.pi / cfg.omega_p * args.fps:.0f} frames per wave period")

    # ---- setup, once -----------------------------------------------------
    t_setup = time.time()
    ts = tiling.TileSet.build(cfg)
    bathy = Bathymetry.from_config(cfg, fine=True)
    region = tuple(args.mesh_region) if args.mesh_region else \
        tuple(bathy.meta.extent[i] for i in (0, 2, 1, 3))
    budget = args.mesh_max_vertices or cfg.output.mesh_max_vertices

    foam_track = None
    if not args.no_foam:
        print("  foam spin-up ...", flush=True)
        foam_track = FoamTrack(cfg, bathy, ts)

    mesh_dir = out / "mesh"
    mesh_dir.mkdir(exist_ok=True)
    if args.terrain_ply:
        terrain_ref = str(args.terrain_ply)
        print(f"  terrain: supplied, {terrain_ref}")
    else:
        print("  terrain (once) ...", flush=True)
        tm = pw_mesh.build_terrain_mesh(bathy, cfg, dx=dx, region=region,
                                        tileset=ts, max_vertices=budget)
        pw_export.write_ply(tm, mesh_dir / "terrain.ply")
        terrain_ref = "terrain.ply"
        print(f"    {tm.n_vertices:,} v")

    def build(t: float):
        foam = foam_track.field if foam_track else None
        return pw_mesh.build_water_mesh(
            ts, bathy, cfg, t=t, dx=dx, region=region, max_vertices=budget,
            foam=foam, foam_bathy=bathy if foam is not None else None)

    m0 = build(args.t0)
    pw_export.write_ply(m0, mesh_dir / "water.ply")
    faces0 = m0.faces
    print(f"  water: {m0.n_vertices:,} v {m0.n_faces:,} f")
    print(f"  setup took {time.time() - t_setup:.1f} s")

    # ---- Mitsuba, once ---------------------------------------------------
    mi = params = keys = scene = None
    if not args.dry_run:
        if args.render_module is None:
            print("\nno --render-module given; use --dry-run, or point at your "
                  "scene script")
            return 2
        import mitsuba as mi_
        mi = mi_
        if args.variant.startswith("scalar"):
            # In a scalar variant mi.Float is a plain Python float, so the whole
            # buffer-overwrite scheme silently does not exist. Fail loudly here
            # rather than at the first frame.
            print(f"variant {args.variant!r} is scalar; per-frame buffer updates "
                  f"need a JIT variant (llvm_* or cuda_*). Use one of those, or "
                  f"--emit-ply --dry-run and render the PLYs separately.")
            return 2
        mi.set_variant(args.variant)            # before the dict is built
        mod = load_render_module(args.render_module)
        try:
            sd = mod.scene_dict(spp=args.spp)
        except TypeError:
            sd = mod.scene_dict()
        scene = mi.load_dict(sd)
        params = mi.traverse(scene)
        keys = find_keys(params, args.shape_id)
        print(f"\nMitsuba {args.variant}; buffers to update each frame:")
        for k, v in keys.items():
            print(f"  {k:10s} -> {v}")
        if keys["foam"] is None and foam_track is not None:
            print("  note: no foam attribute found in the scene; foam will "
                  "still advance but will not reach the renderer")

    # ---- frames ----------------------------------------------------------
    print(f"\nrendering {n_frames} frames -> {out}")
    t_start = time.time()
    for n in range(n_frames):
        t = args.t0 + n * dt
        t_frame = time.time()

        m = m0 if n == 0 else build(t)

        # Topology must not move: the whole scheme rests on overwriting buffers
        # in place. It cannot change (the extent mask is time-independent), but
        # a silent mismatch here would corrupt geometry in a way that is very
        # hard to read back from a picture.
        if m.n_vertices != m0.n_vertices or not np.array_equal(m.faces, faces0):
            raise AssertionError(
                f"frame {n}: topology changed ({m.n_vertices:,} vertices vs "
                f"{m0.n_vertices:,}). The vertex-buffer update is only valid "
                f"while the water extent is fixed.")

        if args.emit_ply:
            pw_export.write_ply(m, mesh_dir / f"water_{n:04d}.ply")

        if not args.dry_run:
            import drjit as dr

            v = np.ascontiguousarray(m.vertices, dtype=np.float32)
            params[keys["positions"]] = mi.Float(v.ravel())
            if keys["normals"] is not None:
                nrm = np.ascontiguousarray(m.normals, dtype=np.float32)
                params[keys["normals"]] = mi.Float(nrm.ravel())
            if keys["foam"] is not None and "foam" in m.channels:
                params[keys["foam"]] = mi.Float(
                    np.ascontiguousarray(m.channels["foam"], dtype=np.float32))
            params.update()                     # refits the acceleration structure

            img = mi.render(scene, spp=args.spp) if args.spp else mi.render(scene)
            mi.util.write_bitmap(str(out / f"frame_{n:04d}.{args.ext}"), img)
            dr.eval()

        if foam_track is not None:
            foam_track.step(dt)

        el = time.time() - t_frame
        done = time.time() - t_start
        eta = done / (n + 1) * (n_frames - n - 1)
        print(f"  frame {n:4d}/{n_frames}  t={t:6.3f}s  {el:6.2f}s  "
              f"eta {eta / 60:5.1f} min", flush=True)

    total = time.time() - t_start
    print(f"\n{n_frames} frames in {total / 60:.1f} min "
          f"({total / n_frames:.2f} s/frame)")
    if not args.dry_run:
        print(f"  ffmpeg -framerate {args.fps:g} -i {out}/frame_%04d.{args.ext} "
              f"-c:v libx264 -pix_fmt yuv420p {out}/movie.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
