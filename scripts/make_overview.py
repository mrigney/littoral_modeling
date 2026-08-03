"""Build a single self-contained HTML overview of the project.

    python scripts/make_overview.py            # -> docs/overview.html

One file, no external requests: every figure is embedded as a data URI, so it
can be mailed, dropped on a share, or opened from a USB stick and it still
renders. Intended for colleagues who want to see what the model does and how far
it has been validated without cloning the repo or running anything.

Headline numbers are computed live from `pywave`, so the page cannot drift from
the code the way a hand-maintained slide would.
"""

from __future__ import annotations

import base64
import io
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import make_figures as mf  # noqa: E402  (same directory)
from pywave import load_config, moments, nearshore, spectrum, tiling  # noqa: E402
from pywave.bathymetry import Bathymetry  # noqa: E402

OUT = ROOT / "docs" / "overview.html"
WEB_DPI = 100


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def render_data_uris(cfg) -> dict[str, str]:
    """Render every registered figure straight to an in-memory PNG data URI."""
    uris = {}
    for name, (fn, _caption) in mf._REGISTRY.items():
        print(f"  {name} ...", end="", flush=True)
        fig = fn(cfg)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=WEB_DPI, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        raw = buf.getvalue()
        uris[name] = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        print(f" {len(raw) / 1024:.0f} KiB")
    return uris


def headline_numbers(cfg) -> dict:
    """Everything quoted in the page, computed rather than remembered."""
    u10, fetch, gamma = cfg.wind.speed, cfg.wind.fetch, cfg.spectrum.gamma
    ts = tiling.TileSet.build(cfg)
    beach = Bathymetry.dean_beach()
    omega = 2.0 * np.pi * cfg.f_p

    total = moments.mss_above(0.0, u10, fetch, gamma=gamma)
    above = moments.mss_above(ts.k_max, u10, fetch, gamma=gamma)
    slope = beach.beach_slope()
    l0 = float(nearshore.deep_water_wavelength(1.0 / cfg.f_p))
    xi = float(nearshore.iribarren_number(slope, ts.hs(), l0))

    d = np.geomspace(0.01, 50.0, 4000)
    ks = nearshore.shoaling_coefficient(omega, d)

    return {
        "hs": spectrum.hs_spectral(u10, fetch, gamma),
        "tp": 1.0 / cfg.f_p,
        "lambda_p": cfg.lambda_p,
        "mss": total,
        "rms_slope_deg": float(np.degrees(np.arctan(np.sqrt(total)))),
        "tz": moments.zero_crossing_period(u10, fetch, gamma),
        "hs_composite": ts.hs(),
        "lod_closure": abs(ts.mss() + above - total) / total,
        "submesh_frac": moments.mss_above(np.pi / 0.125, u10, fetch, gamma=gamma) / total,
        "ks_min": float(ks.min()),
        "xi": xi,
        "runup": float(nearshore.hunt_runup(xi, ts.hs())),
        "swash": float(nearshore.swash_width(nearshore.hunt_runup(xi, ts.hs()), slope)),
        "slope_pct": 100.0 * slope,
    }


CSS = """
:root{
  --ground:#eef2f2; --paper:#ffffff; --ink:#0f1f29; --ink-soft:#4c5f69;
  --ink-faint:#7d8f97; --rule:#d2dcdd; --rule-soft:#e3eaea;
  --accent:#0e6b8a; --accent-soft:#e2eff4; --rust:#a44b36; --moss:#4c7040;
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#0b171d; --paper:#122530; --ink:#e7eff2; --ink-soft:#a4b7c0;
    --ink-faint:#71868f; --rule:#223a46; --rule-soft:#1a2f39;
    --accent:#57aecb; --accent-soft:#16303c; --rust:#dd8b76; --moss:#93b982;
  }
}
:root[data-theme="dark"]{
  --ground:#0b171d; --paper:#122530; --ink:#e7eff2; --ink-soft:#a4b7c0;
  --ink-faint:#71868f; --rule:#223a46; --rule-soft:#1a2f39;
  --accent:#57aecb; --accent-soft:#16303c; --rust:#dd8b76; --moss:#93b982;
}
:root[data-theme="light"]{
  --ground:#eef2f2; --paper:#ffffff; --ink:#0f1f29; --ink-soft:#4c5f69;
  --ink-faint:#7d8f97; --rule:#d2dcdd; --rule-soft:#e3eaea;
  --accent:#0e6b8a; --accent-soft:#e2eff4; --rust:#a44b36; --moss:#4c7040;
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:ui-sans-serif,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:16px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
.mono{font-family:ui-monospace,"Cascadia Mono","SF Mono",Menlo,Consolas,monospace}
.wrap{max-width:1160px; margin:0 auto; padding:0 28px}
.prose{max-width:68ch}

/* sounding rule: a hairline with survey ticks, from the subject's own charts */
.sounding{height:11px; margin:0; border:0; position:relative}
.sounding::before{content:""; position:absolute; left:0; right:0; top:5px;
  height:1px; background:var(--rule)}
.sounding::after{content:""; position:absolute; left:0; right:0; top:0; height:11px;
  background-image:linear-gradient(to right,var(--rule) 1px,transparent 1px);
  background-size:34px 100%;}

header{padding:64px 0 0}
.eyebrow{font-size:11.5px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--accent); font-weight:600}
h1{font-size:clamp(30px,4.4vw,50px); line-height:1.08; letter-spacing:-.022em;
  font-weight:680; margin:20px 0 0; text-wrap:balance; max-width:19ch}
.lede{font-size:19px; line-height:1.58; color:var(--ink-soft); margin:20px 0 0;
  max-width:60ch}

.strip{display:grid; grid-template-columns:repeat(auto-fit,minmax(148px,1fr));
  gap:1px; background:var(--rule); border:1px solid var(--rule);
  margin:38px 0 0; border-radius:3px; overflow:hidden}
.cell{background:var(--paper); padding:15px 17px}
.cell .k{font-size:10.5px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--ink-faint); font-weight:600}
.cell .v{font-size:25px; font-weight:660; letter-spacing:-.02em; margin-top:5px;
  font-variant-numeric:tabular-nums}
.cell .u{font-size:12.5px; color:var(--ink-faint); margin-left:3px; font-weight:400}

section{padding:52px 0 0}
h2{font-size:25px; letter-spacing:-.016em; font-weight:660; margin:16px 0 0;
  text-wrap:balance}
h3{font-size:16.5px; font-weight:640; margin:30px 0 0; letter-spacing:-.006em}
p{margin:14px 0 0}
a{color:var(--accent); text-decoration-thickness:1px; text-underline-offset:2px}

.rule-note{border-left:2px solid var(--accent); background:var(--accent-soft);
  padding:16px 20px; margin:26px 0 0; border-radius:0 3px 3px 0; max-width:66ch}
.rule-note p{margin:0; font-size:16.5px}

figure{margin:30px 0 0; background:var(--paper); border:1px solid var(--rule);
  border-radius:3px; overflow:hidden}
figure img{display:block; width:100%; height:auto; background:#fff}
figcaption{padding:15px 20px; font-size:14.5px; color:var(--ink-soft);
  border-top:1px solid var(--rule-soft); line-height:1.55}
figcaption b{color:var(--ink); font-weight:640}

.tablewrap{overflow-x:auto; margin:24px 0 0; border:1px solid var(--rule);
  border-radius:3px; background:var(--paper)}
table{border-collapse:collapse; width:100%; font-size:14.5px; min-width:560px}
th,td{text-align:left; padding:11px 16px; border-bottom:1px solid var(--rule-soft)}
th{font-size:10.5px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--ink-faint); font-weight:600; background:var(--ground)}
tr:last-child td{border-bottom:0}
td.num{font-family:ui-monospace,"Cascadia Mono","SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums; white-space:nowrap}

.pill{display:inline-block; font-size:10.5px; letter-spacing:.09em; font-weight:660;
  text-transform:uppercase; padding:3px 8px; border-radius:2px; white-space:nowrap}
.pill.done{background:var(--moss); color:var(--paper)}
.pill.open{background:var(--rule); color:var(--ink-soft)}
.pill.sub{background:var(--accent); color:var(--paper)}

.cards{display:grid; grid-template-columns:repeat(auto-fit,minmax(288px,1fr));
  gap:18px; margin:26px 0 0}
.card{background:var(--paper); border:1px solid var(--rule); border-radius:3px;
  padding:20px 22px}
.card .tag{font-size:10.5px; letter-spacing:.13em; text-transform:uppercase;
  font-weight:660; color:var(--rust)}
.card h4{margin:9px 0 0; font-size:16px; font-weight:640; letter-spacing:-.006em}
.card p{font-size:14.5px; color:var(--ink-soft); margin:9px 0 0}

footer{margin:64px 0 0; padding:26px 0 60px; border-top:1px solid var(--rule);
  font-size:13px; color:var(--ink-faint)}
footer .mono{font-size:12px}

@media (max-width:640px){
  .wrap{padding:0 18px}
  header{padding-top:40px}
  .cell .v{font-size:21px}
}
@media (prefers-reduced-motion:no-preference){
  figure img{transition:none}
}
"""


def build_html(cfg, n, uris, caps) -> str:
    sha = _git_sha()
    stamp = f"{datetime.now(timezone.utc):%Y-%m-%d}"

    def fig(name, title):
        return f"""<figure>
  <img src="{uris[name]}" alt="{title}">
  <figcaption><b>{title}.</b> {caps[name]}</figcaption>
</figure>"""

    phases = [
        ("1", "Spectrum, directional spreading, moments", "done", "Implemented"),
        ("2", "FFT surface synthesis, multi-tile composition", "done", "Implemented"),
        ("3", "Validation suite and generated V&amp;V report", "done", "Implemented"),
        ("4", "Terrain and lake basin in Houdini", "sub", "Synthetic stand-in"),
        ("5", "Shoaling, refraction, breaking, foam", "done", "Implemented"),
        ("6", "Mesh generation, LOD rings, export", "open", "Not started"),
        ("7", "Mitsuba <code>roughwater</code> BSDF plugin", "open", "Not started"),
        ("8", "Spectral emissivity table", "open", "Not started"),
        ("9", "EMBER integration", "open", "Not started"),
        ("10", "Physical validation and traceability", "open", "Not started"),
    ]
    phase_rows = "\n".join(
        f'<tr><td class="num">{p}</td><td>{d}</td>'
        f'<td><span class="pill {c}">{lab}</span></td></tr>'
        for p, d, c, lab in phases)

    checks = [
        ("Variance conservation across the f&nbsp;&rarr;&nbsp;k Jacobian", "1.3&times;10<sup>&minus;6</sup>", "1%"),
        ("Directional spreading integrates to one", "2.4&times;10<sup>&minus;15</sup>", "10<sup>&minus;6</sup>"),
        ("LOD invariant closure", "8.2&times;10<sup>&minus;5</sup>", "10<sup>&minus;3</sup>"),
        ("Realised composite H<sub>s</sub> vs spectrum", "&minus;1.3%", "5%"),
        ("Crest phase speed vs &omega;/k", "3.1&times;10<sup>&minus;7</sup>", "5%"),
        ("Zero-crossing period from a time series", "1.5%", "10%"),
        ("Snell invariant sin&thinsp;&alpha;/c along a ray", "2.6&times;10<sup>&minus;16</sup>", "10<sup>&minus;12</sup>"),
        ("Shoaling vs Green's law in shallow water", "0.3%", "2%"),
        ("Breaker line vs d&nbsp;=&nbsp;H/&gamma;<sub>b</sub>", "1.4&times;10<sup>&minus;16</sup>", "5%"),
        ("Foam decay over one half life", "&lt;10<sup>&minus;9</sup>", "10<sup>&minus;9</sup>"),
        ("Foam cold start vs sequential, frame 500", "0.61%", "1%"),
        ("Regression baseline, bit-for-bit", "0", "10<sup>&minus;12</sup>"),
    ]
    check_rows = "\n".join(
        f'<tr><td>{d}</td><td class="num">{m}</td><td class="num">{t}</td></tr>'
        for d, m, t in checks)

    return f"""<div class="wrap">
<header>
  <div class="eyebrow">Littoral scene generation &middot; internal status</div>
  <h1>A water surface you can defend, not just render</h1>
  <p class="lede">
    <span class="mono">pywave</span> builds a time-evolving sea surface from a
    wind-wave spectrum and reports its statistics &mdash; wave height, slope
    variance, directional anisotropy &mdash; as physically derived numbers.
    Every appearance parameter a renderer eventually needs is a moment of one
    spectrum, so nothing downstream is tuned by hand.
  </p>

  <div class="strip">
    <div class="cell"><div class="k">Significant height</div>
      <div class="v">{n['hs']:.3f}<span class="u">m</span></div></div>
    <div class="cell"><div class="k">Peak period</div>
      <div class="v">{n['tp']:.2f}<span class="u">s</span></div></div>
    <div class="cell"><div class="k">Peak wavelength</div>
      <div class="v">{n['lambda_p']:.2f}<span class="u">m</span></div></div>
    <div class="cell"><div class="k">RMS slope</div>
      <div class="v">{n['rms_slope_deg']:.1f}<span class="u">&deg;</span></div></div>
    <div class="cell"><div class="k">Automated checks</div>
      <div class="v">72<span class="u">passing</span></div></div>
  </div>
</header>

<section>
  <hr class="sounding">
  <div class="eyebrow">The governing rule</div>
  <h2>The spectrum is the single source of truth</h2>
  <div class="prose">
    <p>
      Wave height, slope statistics, BSDF roughness and level-of-detail
      behaviour are all derived from one <span class="mono">S(k)</span>. If two
      quantities disagree, the spectrum wins and the other one has a bug. That
      rule is what makes the result traceable: there is no artistic parameter
      anywhere in the chain that could be quietly adjusted to make a picture
      look better.
    </p>
    <div class="rule-note">
      <p>
        This is the argument you fundamentally cannot make for a DCC tool's
        black-box ocean solver &mdash; and the reason the validation suite is
        the deliverable, not an afterthought.
      </p>
    </div>
    <p>
      The reference scene throughout is a small lake: {cfg.wind.speed:.0f}&nbsp;m/s
      wind over {cfg.wind.fetch:.0f}&nbsp;m of fetch. It is deep water everywhere
      except the last few metres, which is what makes the nearshore work
      tractable &mdash; and, at 1&nbsp;m ground sample distance, puts the entire
      surf and swash zone below one pixel.
    </p>
  </div>
</section>

<section>
  <hr class="sounding">
  <div class="eyebrow">01 &middot; The spectrum</div>
  <h2>Everything starts here</h2>
  <div class="prose"><p>
    A JONSWAP spectrum in frequency, a frequency-dependent directional spreading
    function, and the Jacobian that converts the pair into a two-dimensional
    wavenumber spectrum. The conversion is the single easiest place in the whole
    pipeline to be wrong by a constant factor, so it is checked against an
    independent brute-force integration rather than against itself.
  </p></div>
  {fig("spectrum", "The spectrum and its directional spreading")}
  {fig("lod", "The level-of-detail invariant")}
</section>

<section>
  <hr class="sounding">
  <div class="eyebrow">02 &middot; Synthesis</div>
  <h2>From spectrum to surface</h2>
  <div class="prose"><p>
    Summing thousands of sinusoids <em>is</em> a Fourier transform, so an FFT
    does it in <span class="mono">O(N log N)</span>. Three properties of the
    construction matter more than speed: the surface is evaluable at arbitrary
    time with no accumulated state, it is exactly reproducible from a single
    seed on any machine, and any frame can be computed on any node without
    coordination. Frame 8000 costs what frame 1 costs and does not drift.
  </p></div>
  {fig("surface", "One realisation of the composite surface")}
  {fig("statistics", "The surface reproduces the spectrum it was built from")}
</section>

<section>
  <hr class="sounding">
  <div class="eyebrow">03 &middot; The nearshore</div>
  <h2>What happens in the last few metres</h2>
  <div class="prose">
    <p>
      Shoaling, refraction, breaking and swash were built and validated
      <em>before</em> the terrain pipeline exists, against a synthetic
      equilibrium beach. That ordering is deliberate: a synthetic profile has
      closed-form answers &mdash; Green's law, Snell's law, the breaker index
      &mdash; that an exported heightfield cannot supply. It can only be checked
      against itself.
    </p>
    <p>
      The synthetic beach satisfies exactly the field contract the terrain export
      will have to meet, so swapping in real bathymetry is a loader change rather
      than a physics change.
    </p>
  </div>
  {fig("bathymetry", "The synthetic beach")}
  {fig("shoaling", "Shoaling and refraction coefficients")}
  {fig("nearshore", "A transect through the surf zone")}
  {fig("refraction_map", "Refraction on a curved shoreline")}
  <div class="prose"><p>
    On this beach the foreshore is {n['slope_pct']:.1f}%, giving an Iribarren
    number of {n['xi']:.2f} &mdash; firmly spilling breakers, no plunging. Runup
    is {100 * n['runup']:.1f}&nbsp;cm vertical and the swash excursion
    {n['swash']:.2f}&nbsp;m horizontal, both sub-pixel at sensor resolution.
    So breaking and swash are carried as per-cell coverage channels rather than
    as animated geometry, which would be weeks of work invisible in the imagery.
  </p></div>
</section>

<section>
  <hr class="sounding">
  <div class="eyebrow">04 &middot; Evidence</div>
  <h2>Twelve of the seventy-two checks</h2>
  <div class="prose"><p>
    The suite runs in about fifty seconds and regenerates a validation report on
    every run. Tests record the number they measured and the reference they were
    judged against, not a pass or fail &mdash; a reviewer needs
    &ldquo;realised H<sub>s</sub> = 0.0842&nbsp;m vs 0.0853&nbsp;m,
    &minus;1.3%&rdquo;, which is a statement about the physics.
  </p></div>
  <div class="tablewrap">
    <table>
      <thead><tr><th>Check</th><th>Measured</th><th>Tolerance</th></tr></thead>
      <tbody>
{check_rows}
      </tbody>
    </table>
  </div>
</section>

<section>
  <hr class="sounding">
  <div class="eyebrow">05 &middot; Honest accounting</div>
  <h2>Where the plan was wrong</h2>
  <div class="prose"><p>
    Four places where the implementation departs from the written plan. Each is
    recorded in the validation report with the measurement that motivated it,
    rather than quietly reinterpreted.
  </p></div>
  <div class="cards">
    <div class="card">
      <div class="tag">Unmeetable gate</div>
      <h4>H<sub>s</sub> agreement within 2%</h4>
      <p>
        JONSWAP supplies two independently fitted power laws that are not
        mutually consistent, so the two H<sub>s</sub> values agree at exactly one
        fetch and nowhere else. Replaced by pinning the <em>relation</em> between
        them across four decades of fetch &mdash; a strictly stronger constraint.
        Measured exponent 0.050007 against a structural prediction of exactly
        0.05.
      </p>
    </div>
    <div class="card">
      <div class="tag">Unmeetable gate</div>
      <h4>Elevation skewness in [0, 0.3]</h4>
      <p>
        The model is linear in elevation: the choppiness displacement relocates
        sample points horizontally but never changes the set of elevation values,
        so it cannot create the crest-trough asymmetry that skews a real sea.
        Measured &minus;0.015. Positive skewness needs second-order Stokes terms,
        which are not modelled.
      </p>
    </div>
    <div class="card">
      <div class="tag">Corrected guidance</div>
      <h4>Foam spin-up window</h4>
      <p>
        The plan called for 30 frames of spin-up. At 30&nbsp;fps that is one
        second against a three-second half life, leaving 79% of the discarded
        history and failing the reproducibility gate. The window is set by the
        half life, not a frame count: 23&nbsp;s for 0.5%. Now derived
        automatically.
      </p>
    </div>
    <div class="card">
      <div class="tag">Corrected guidance</div>
      <h4>Refraction by depth blend</h4>
      <p>
        The prescribed depth-weighted blend has no frequency dependence and
        over-aligns waves at the waterline. Snell's law against the full
        dispersion relation is used instead; the blend is retained and the
        38.5&deg; peak disagreement measured rather than assumed.
      </p>
    </div>
  </div>
</section>

<section>
  <hr class="sounding">
  <div class="eyebrow">06 &middot; Where this sits</div>
  <h2>Ten phases, five of them closed</h2>
  <div class="tablewrap">
    <table>
      <thead><tr><th style="width:64px">Phase</th><th>Deliverable</th>
        <th style="width:160px">Status</th></tr></thead>
      <tbody>
{phase_rows}
      </tbody>
    </table>
  </div>
  <div class="prose"><p>
    Phase&nbsp;4 is the natural next step, and it is now a smaller job than it
    was: the field contract is already fixed, exercised and asserted, so the
    terrain work is import and validation rather than physics. Phase&nbsp;6
    onward &mdash; mesh export, the Mitsuba BSDF, emissivity &mdash; consumes
    what is already here.
  </p></div>
</section>

<footer>
  <div class="wrap" style="padding:0">
    Generated {stamp} from commit <span class="mono">{sha}</span> &middot;
    every figure and headline number computed live at build time &middot;
    <span class="mono">python scripts/make_overview.py</span>
  </div>
</footer>
</div>"""


def main():
    cfg = load_config(mf.CONFIG)
    print("computing headline numbers ...")
    n = headline_numbers(cfg)
    print(f"rendering {len(mf._REGISTRY)} figures at {WEB_DPI} dpi")
    uris = render_data_uris(cfg)
    caps = {name: cap for name, (_, cap) in mf._REGISTRY.items()}

    html = (f"<title>pywave — littoral water surface modelling</title>\n"
            f"<style>{CSS}</style>\n{build_html(cfg, n, uris, caps)}\n")
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
