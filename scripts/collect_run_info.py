"""Harvest every run in a directory into one markdown report.

Half of a debugging session gets lost to not knowing which `mesh_dx` produced
which picture. All of that is already written down -- `summary.json` and the
mesh sidecars carry the sea state, the spacing, the vertex counts, the channel
ranges and the commit -- it is just spread across files. This gathers it.

    python scripts/collect_run_info.py                       # all of runs/
    python scripts/collect_run_info.py -o ~/test_log_data.md
    python scripts/collect_run_info.py runs/straits           # just one

Paste the output next to your render notes and the pairing is unambiguous.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def _runs(root: Path):
    if (root / "summary.json").exists():
        return [root]
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and (d / "summary.json").exists())


def _rows(run: Path):
    """One row per meshed frame; one row anyway if the run has no mesh."""
    s = _load(run / "summary.json")
    if s is None:
        return []
    # Runs made by older commits carry fewer sections, and a run is still worth
    # reporting when one is missing -- so every lookup has a fallback.
    sc, sp = s.get("scene", {}), s.get("spectrum", {})
    ba = s.get("bathymetry", {})
    lam = sp.get("lambda_p_m")
    if lam is None:
        return []
    base = {
        "run": run.name,
        "scene": sc.get("name", run.name),
        "wind": (f"{sc.get('wind_speed_ms', float('nan')):g} m/s @ "
                 f"{sc.get('wind_direction_deg', float('nan')):g}deg"),
        "fetch_m": sc.get("fetch_m", float("nan")),
        "lambda_p_m": lam,
        "Hs_m": sp.get("Hs_spectral_m", float("nan")),
        "Tp_s": sp.get("T_p_s", float("nan")),
        "bathy_dx": ba.get("dx"),
        "source": ba.get("source", "?"),
    }
    metas = sorted((run / "mesh").glob("water_*.json")) if (run / "mesh").exists() else []
    if not metas:
        return [dict(base, mesh_dx=None, posts_per_lambda=None, n_vertices=None,
                     alpha=None, git_sha=None, t=None, region=None)]

    out = []
    for m in metas:
        d = _load(m)
        if d is None:
            continue
        dx = d["mesh_dx"]
        mss = d.get("channels", {}).get("mss", {})
        x0, y0, x1, y1 = d["region"]
        out.append(dict(
            base,
            frame=m.stem,
            t=d["t"],
            mesh_dx=dx,
            posts_per_lambda=lam / dx if dx else None,
            n_vertices=d["n_vertices"],
            n_faces=d["n_faces"],
            alpha=(mss.get("mean", 0.0) ** 0.5) if mss else None,
            mss_min=mss.get("min"), mss_max=mss.get("max"),
            region=f"{x1 - x0:.0f} x {y1 - y0:.0f} m",
            git_sha=(d.get("git_sha") or "")[:7],
            written=d.get("written_utc"),
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", nargs="?", default="runs", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.root.exists():
        print(f"no such directory: {args.root}")
        return 1
    runs = _runs(args.root)
    if not runs:
        print(f"no runs with a summary.json under {args.root}")
        return 1

    rows = [r for run in runs for r in _rows(run)]
    L = []
    L.append(f"# Run data — {len(runs)} run(s) under `{args.root}`\n")

    L.append("## Sea state and mesh\n")
    L.append("| run | wind | fetch | Tp | **λp** | Hs | **mesh_dx** | "
             "**posts/λp** | vertices | α=√mss̄ | commit |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        dx = f"{r['mesh_dx']:g} m" if r["mesh_dx"] else "— (no mesh)"
        ppl = f"**{r['posts_per_lambda']:.1f}**" if r["posts_per_lambda"] else "—"
        nv = f"{r['n_vertices']:,}" if r.get("n_vertices") else "—"
        al = f"{r['alpha']:.4f}" if r.get("alpha") else "—"
        L.append(f"| {r['run']} | {r['wind']} | {r['fetch_m']:g} m | "
                 f"{r['Tp_s']:.2f} s | **{r['lambda_p_m']:.2f} m** | "
                 f"{r['Hs_m']:.3f} m | {dx} | {ppl} | {nv} | {al} | "
                 f"`{r.get('git_sha') or '—'}` |")

    L.append("\n> `posts/λp` is the number that governs how the water looks: "
             "≳16 good, ~8 acceptable, ~4 visibly patterned, ≲2 pure moiré.\n")

    L.append("## Detail\n")
    for r in rows:
        L.append(f"### {r['run']}"
                 + (f" — `{r.get('frame')}` at t = {r['t']} s" if r.get("frame") else ""))
        bdx = f"{r['bathy_dx']:g} m" if r["bathy_dx"] else "unknown spacing"
        L.append(f"- bathymetry: `{r['source']}` at {bdx}")
        if r["mesh_dx"]:
            L.append(f"- mesh: {r['mesh_dx']:g} m over {r['region']}, "
                     f"{r['n_vertices']:,} v / {r['n_faces']:,} f")
            L.append(f"- mss channel: {r['mss_min']:.5f} – {r['mss_max']:.5f} "
                     f"(α {r['mss_min'] ** 0.5:.4f} – {r['mss_max'] ** 0.5:.4f})")
            if r["bathy_dx"] and r["mesh_dx"] < r["bathy_dx"]:
                L.append(f"- **note:** mesh ({r['mesh_dx']:g} m) is finer than the "
                         f"fields ({r['bathy_dx']:g} m) — check clearance")
            L.append(f"- written {r.get('written')} UTC")
        L.append("")

    text = "\n".join(L)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(rows)} rows)")
    else:
        # The report uses lambda/alpha, which a Windows console codepage cannot
        # encode. The file path always could; only stdout needs persuading.
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
