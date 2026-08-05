# Algorithm Description Document — `pywave`

**Spectral water surface synthesis for physics-based EO/IR littoral scene generation.**

This document works through each physical phenomenon the model represents: the
governing physics, the discrete algorithm that implements it, what is exact and
what is approximated, and how each is verified.

It is the companion to two other documents and overlaps neither:

| Document | Answers |
|---|---|
| [users_guide.md](users_guide.md) | How do I run it? |
| **this** | How does it work, and why is it right? |
| [validation_report.md](validation_report.md) | What did it measure on this commit? |

Every number quoted here is reproduced by `pytest`, which regenerates the
validation report. Where a number appears in prose it is a measurement, not an
expectation.

---

## Contents

1. [The governing principle](#1-the-governing-principle)
2. [Conventions](#2-conventions)
3. [The wave spectrum](#3-the-wave-spectrum)
4. [Directional spreading](#4-directional-spreading)
5. [Frequency to wavenumber: the Jacobian](#5-frequency-to-wavenumber-the-jacobian)
6. [Spectral moments](#6-spectral-moments)
7. [Surface synthesis](#7-surface-synthesis)
8. [Multi-tile composition](#8-multi-tile-composition)
9. [Bathymetry](#9-bathymetry)
10. [Shoaling](#10-shoaling)
11. [Refraction](#11-refraction)
12. [Breaking and depth limiting](#12-breaking-and-depth-limiting)
13. [Swash and wetness](#13-swash-and-wetness)
14. [Foam](#14-foam)
15. [Mesh generation](#15-mesh-generation)
16. [Per-vertex channels and the LOD invariant](#16-per-vertex-channels-and-the-lod-invariant)
17. [Summary of approximations](#17-summary-of-approximations)
18. [References](#18-references)

---

## 1. The governing principle

> The spectrum is the single source of truth. Wave height, slope statistics,
> BSDF roughness and level-of-detail behaviour are all derived from one `S(k)`.
> Nothing is tuned independently; if two quantities disagree, the spectrum wins
> and the other one has a bug.

This is not a style preference. A rendered water surface for EO/IR simulation
has to be *defensible*: every appearance parameter that reaches the renderer
must trace back to a measured physical quantity through a chain someone can
audit. The alternative — a roughness slider tuned until the picture looks right
— produces imagery that cannot be justified in review and cannot be trusted when
the sensor geometry changes.

The practical consequence is that this codebase has no artistic parameters. The
one number that looks like a knob, `surface.choppiness`, has a physical value of
1.0 and the config says not to tune it.

```
                    JONSWAP S(f)  ×  spreading D(f, θ)
                                 │
                       Jacobian  │  S(k,θ) = S(f)·D(f,θ)·Cg / (2πk)
                                 ▼
                              S(kx, ky)
                    ┌────────────┴────────────┐
                    ▼                         ▼
             moments (analytic)         FFT synthesis
             m₀ → Hs                    h(x,y,t)
             m₂ → mss → BSDF α          slopes → normals
             mss_above(k) → sub-grid    displacement
                    │                         │
                    └───────────┬─────────────┘
                                ▼
                     nearshore transformation
                     shoaling · refraction · breaking
                                ▼
                    mesh + per-vertex channels → renderer
```

The left branch is exact and unbounded in wavenumber. The right branch is a
discrete grid truncated at its Nyquist. Section 16 explains how the two are
reconciled.

---

## 2. Conventions

Fixed once, in `pywave/constants.py`, which every other module imports. Violating
any of these silently is the most common source of multi-day debugging.

| Item | Convention |
|---|---|
| Coordinates | Right-handed, **Z up**. X east, Y north. |
| Units | Metres, seconds, radians internally. Degrees only in config files. |
| Angular frequency | `ω` (rad/s) everywhere; `f` (Hz) only inside `spectrum.py`. Never mixed in one expression. |
| Wavenumber | `k` in rad/m (`k = 2π/λ`), not cycles/m. |
| Direction | `θ` is the direction waves travel **toward**, CCW from +X. Oceanographers often use "coming from"; this package never does. |
| Water plane | `z = z_w`, a scene constant. Displacement is relative to it. |
| Depth sign | `d > 0` in water. `depth = z_w − z_terrain`. |
| SDF sign | `s < 0` in water, `s > 0` on land — deliberately **opposite** to depth, so `s` reads as "distance inland". |
| FFT | numpy default. `ifft2` includes the `1/N²`, so spatial fields are multiplied back by `N²`. |
| Time origin | `t = 0` is scenario start. The surface is a pure function of `t`. |
| Random seed | One integer, used in `spectrum.initial_amplitudes()` and nowhere else. |

Degrees are converted to radians exactly once, in `config.load_config`. Nothing
downstream of that function sees a degree.

### Two quantities are both called `m₂`

This trips people badly enough that the package refuses to abbreviate:

| Symbol | Definition | Units | Sets |
|---|---|---|---|
| `moments.m2` | `∫ k² S(k) d²k` — second moment in **wavenumber** | dimensionless | mean square slope, BSDF roughness |
| `moments.moment_omega(2)` | `∫ ω² S(ω) dω` — second moment in **angular frequency** | rad²/s² | zero-crossing period `Tz = 2π√(m₀/m₂ω)` |

In deep water `ω² = gk`, so **mean square slope is the fourth frequency moment**,
not the second. Conflating them gives a `Tz` wrong by orders of magnitude.

---

## 3. The wave spectrum

**Module:** `pywave/spectrum.py`

### Physics

A wind-driven sea is a superposition of many wave components. The JONSWAP
spectrum (Hasselmann et al. 1973) describes how energy is distributed across
frequency for a fetch-limited sea:

```
S(f) = α g² (2π)⁻⁴ f⁻⁵ · exp[−1.25 (f_p/f)⁴] · γ^r

r = exp[ −(f − f_p)² / (2 σ² f_p²) ],   σ = 0.07 (f ≤ f_p), 0.09 (f > f_p)
```

The `f⁻⁵` tail is Phillips' equilibrium range; the exponential cuts off energy
below the peak; `γ` sharpens the peak relative to a fully developed
Pierson–Moskowitz sea (`γ = 1` recovers it).

Fetch-limited parameters, with `X̃ = gX/U₁₀²` the dimensionless fetch:

```
α   = 0.076 · X̃^(−0.22)
f_p = 3.5 (g/U₁₀) · X̃^(−0.33)
```

### Implementation

Evaluated **in log space** so the `f⁻⁵` singularity is never formed:

```python
log S = log(α g² (2π)⁻⁴) − 5 log f − 1.25 (f_p/f)⁴ + r log γ
```

Below `f_p/8` the exponential has already underflowed to zero in float64, so the
function returns exactly 0 there rather than `inf × 0`. That is the correct
limit as well as the safe one.

### Hs has two values, and neither is wrong

JONSWAP supplies two **independently fitted** power laws that are not mutually
consistent:

| Quantity | Fit | Implied `m₀` scaling |
|---|---|---|
| scale parameter | `α = 0.076 X̃^(−0.22)` | `m₀ ~ α f_p⁻⁴ ~ X̃^1.10` |
| peak frequency | `f_p ~ X̃^(−0.33)` | |
| dimensionless energy | `ε = g²m₀/U₁₀⁴ = 1.6×10⁻⁷ X̃` | `m₀ ~ X̃^1.00` |

Integrating the spectral form uses the first pair; the energy growth law uses the
second. Their ratio is therefore `~X̃^0.10` in energy and `~X̃^0.05` in `Hs`:

```
hs_spectral / fetch_limited_hs = 0.78696 · X̃^0.05
```

**This was measured, not assumed.** Fitting `c·X̃^p` over U₁₀ = 3–20 m/s and
fetch = 300 m – 5000 km gives **p = 0.050007** against a structural prediction of
exactly 0.05, with worst residual 2.5×10⁻⁴ across four decades.

`hs_spectral` is the value to quote: it is what a realised FFT surface
reproduces. Rescaling the spectrum to hit the energy fit would change `α`, hence
the Phillips tail, hence mean square slope and every BSDF roughness downstream —
which is exactly the independent tuning the governing principle forbids.

### Verification

| Check | Measured | Tolerance |
|---|---|---|
| Relation exponent `p` | 0.050007 | 0.05 ± 10⁻³ |
| Relation coefficient `c` | 0.786967 | 0.78696 ± 1% |
| Worst residual over 30 scenes | 2.5×10⁻⁴ | 10⁻³ |
| `f_p` from numerical argmax | 8.8×10⁻⁵ rel. | 2% |

---

## 4. Directional spreading

**Module:** `pywave/spectrum.py`

### Physics

Energy is not all travelling in the wind direction. The spreading function
`D(f, θ)` distributes it in angle, normalised so `∫D dθ = 1` at every frequency —
which is what makes spreading affect the *directional* distribution without
changing `m₀`.

```
D(f, θ) = N(s) · cos^(2s)[ (θ − θ_wind)/2 ]
```

The exponent is frequency-dependent: waves near the peak are tightly aligned
with the wind, high-frequency waves spread broadly.

```
s = 9.77 (f/f_p)^4.06     f ≤ f_p
s = 11.5 (f/f_p)^(−2.5)   f > f_p
```

### Implementation

**The normalisation has two implementations, and they must agree.** The closed
form comes from substituting `u = θ/2`, giving a Wallis integral:

```
N(s) = Γ(s+1) / (2√π · Γ(s+½))
```

evaluated via `gammaln` so large `s` cannot overflow. The cookbook prescribes
numerical quadrature instead, arguing the closed form is error-prone and this is
not a hot path. The first half is true; the second is not — `jonswap_sk` runs on
512² grids, where per-element quadrature would be a 262144 × 8192 temporary,
about 17 GB. So the closed form is the production path and the quadrature is
retained as its **test oracle**, which keeps the cookbook's actual intent (verify
the normalisation, don't trust it) at 1/1000 the cost.

The angular difference is wrapped into `[−π, π]` before halving. Without the
wrap, directions more than π from the wind produce a negative base raised to a
fractional power — silent NaNs over half the domain.

### A known discontinuity, deliberately preserved

The two branches do not meet at `f = f_p`: the low branch gives `s = 9.77`, the
high branch `s = 11.5`, an ~18% step in spreading width exactly at the peak. The
published source (Hasselmann, Dunckel & Ewing 1980) uses 6.97 and 9.77, also
discontinuous but by a different amount — so the step is a property of the fitted
model, not a transcription error.

It does **not** affect `Hs` or any Gate 1 quantity, because `D` integrates to 1
at every frequency and divides out of `m₀`. It **does** affect directional
quantities — the upwind/crosswind anisotropy, hence the BSDF tangent frame in
Phase 7. `model="hasselmann"` selects the published coefficients if that matters.

### Verification

| Check | Measured | Tolerance |
|---|---|---|
| `max │∫D dθ − 1│` over f/f_p ∈ [0.125, 100] | 2.4×10⁻¹⁵ | 10⁻⁶ |
| Closed form vs quadrature oracle | 2.1×10⁻¹¹ | 10⁻¹⁰ |

The integral is checked with **adaptive** quadrature, deliberately: uniform
sampling of `cos^(2s)` converges only algebraically, and a rectangle rule at
n = 8192 still shows 5×10⁻⁵ error that has nothing to do with the normalisation
under test.

---

## 5. Frequency to wavenumber: the Jacobian

**Module:** `pywave/spectrum.py` → `jonswap_sk`

This is the single easiest place in the pipeline to be wrong by a constant
factor, and the error looks like a plausible sea state.

### Derivation

Variance must be conserved between the two parameterisations:

```
∫ S(f) df  =  ∫∫ S(k,θ) d²k        with  d²k = k dk dθ
```

Matching integrands element by element:

```
S(f)·D(f,θ) df dθ = S(k,θ)·k dk dθ
```

so

```
S(k,θ) = S(f)·D(f,θ) · (df/dk) / k
```

and since `f = ω/2π`, `df/dk = (1/2π)·dω/dk = Cg/(2π)`:

```
S(k,θ) = S(f) · D(f,θ) · Cg / (2πk)
```

**The division by `k` comes from the polar area element, not from the frequency
Jacobian.** Confusing the two is the most common way to be wrong here, and it
surfaces as an `Hs` off by a factor you will spend days hunting.

### Finite depth

`ω(k)` uses the full `ω² = gk·tanh(kd)`, so a given wavenumber maps to a lower
frequency in shallow water and both `S(f)` and `Cg` are evaluated consistently
there. Carried from the start rather than retrofitted, because Phase 5 needs it.

### Verification

Checked against a **brute-force 2-D polar integration** that calls `jonswap_sk`
on Cartesian `(kx, ky)` rather than reusing the collapsed radial form. That makes
it an independent evaluation of both the Jacobian and the spreading
normalisation — an oracle sharing the analytic shortcut would agree with it no
matter how wrong both were.

| Check | Measured | Tolerance |
|---|---|---|
| `∫S(k)d²k` vs `∫S(f)df`, deep water | 1.3×10⁻⁶ rel. | 1% |
| Same, finite depth d = 3 m | 8.8×10⁻⁷ rel. | 1% |

---

## 6. Spectral moments

**Module:** `pywave/moments.py`

### Physics

Moments are the bridge between the abstract spectrum and measurable quantities.

```
m₀  = ∫ S(k) d²k              height variance      Hs = 4√m₀
m₂  = ∫ k² S(k) d²k           mean square slope    α_Beckmann = √mss
```

Mean square slope is the quantity that sets microfacet roughness in the BSDF. It
is not a free parameter and not an artistic choice.

### The logarithmic tail, and why the grid must be log-spaced

The Phillips tail makes the slope integrand fall as `1/k`:

```
k² S(k) · k dk  ~  dk/k
```

so **equal contributions come from equal ratios of k**, and every decade
contributes equally. A linear grid puts essentially all its samples in the last
octave and systematically under-reports slope variance. The radial integrals are
therefore log-spaced — a correctness requirement, not an optimisation.

The same property makes the total depend on where the upper cut is placed:
`mss ~ log(k_max/k_p)`. `K_CAPILLARY = 400 rad/m` (λ ≈ 1.6 cm, the
gravity–capillary transition) is a **stated modelling choice**, declared once in
`constants.py` and reported alongside every mss figure so the number is never
silently rescaled.

### The directional integral collapses

Because `S(k,θ) = S(f)·D·Cg/(2πk)` and `∫D dθ = 1` by construction:

```
∫∫ kⁿ S(k,θ) k dk dθ  =  ∫ kⁿ S(f(k)) Cg(k)/(2π) dk
```

This is exact, not an approximation — and it is also why spreading cannot affect
`m₀` or total `mss`, only their directional split.

### Verification

| Check | Measured | Tolerance |
|---|---|---|
| `mss_above(0)` vs direct radial integral | 0 | 10⁻⁹ |
| `mss_above(k)` strictly decreasing over 80 cuts | true | — |
| `mss_up + mss_cross` vs `mss_above(0)` | 1.0×10⁻⁴ rel. | 10⁻³ |
| `Tz = 2π√(m₀/m₂ω)` | exact | 10⁻⁹ |
| mss vs Cox–Munk (1954), U₁₀ 3–15 m/s | worst factor 2.07 | 3 |

The Cox–Munk comparison is a **sanity bound, not ground truth**: it is an
open-ocean, long-fetch fit, and a 1 km freshwater fetch has a different
high-frequency balance. A factor of 10 would mean a normalisation bug; a factor
of 2 means the ocean and the lake are different bodies of water, which they are.

---

## 7. Surface synthesis

**Module:** `pywave/surface.py`

### Physics

Summing thousands of sinusoids *is* a Fourier transform, so an FFT does it in
`O(N log N)` (Tessendorf). Complex amplitudes are drawn once at `t = 0` and
rotated in phase thereafter; each mode rotates at its own frequency, which is
dispersion implemented for free.

### Initial amplitudes

```
h₀(k) = ½ (ξ_r + i ξ_i) √(S(k) · dk_x dk_y),    ξ ~ N(0,1) independent
```

Conceptually this is spectral filtering of white noise: take white noise, shape
it with the square root of the spectrum. The square root appears because `S` is a
*power* density and this is an *amplitude*; the `dk²` converts spectral density
into per-mode variance.

**The factor of ½ is a notorious trap.** With this form `E|h₀|² = ½ S dk²`; the
two-term time evolution below sums two independent contributions, giving
`E|h̃|² = S dk²` per mode and hence `Σ E|h̃|² = m₀` as required. Tessendorf's
written formula uses `1/√2` because he folds a factor of 2 into his definition of
`P_h`. If `Hs` comes out `√2` too large, this is why.

### Time evolution, and the sign

```
h̃(k,t) = h₀(k)·e^(−iωt) + conj(h₀(−k))·e^(+iωt)
```

The second term enforces Hermitian symmetry `h̃(−k) = conj(h̃(k))`, which is what
makes the inverse transform real. Physically it is a pair of waves travelling in
opposite directions, with the directional spectrum deciding which carries the
energy.

**Note the sign.** The cookbook and Tessendorf both write `h₀(k)·e^(+iωt)`.
Combined with the `Σ h̃ e^(+ikx)` synthesis that numpy's `ifft2` implements, the
dominant term then carries phase `kx + ωt` — crests satisfy `x = −ωt/k` and the
whole sea marches **backwards**, into the wind and against the direction the
spreading function was centred on. Flipping the exponent gives phase `kx − ωt`
and waves that travel along `+k` as intended.

Hermitian symmetry is unaffected by the flip, so the surface stays exactly real
either way. That is precisely why this error survives a realness check, a
variance check, a spectrum check *and* a slope check, and is caught only by
tracking a crest.

### Horizontal displacement, derived rather than copied

Real waves have sharp crests and broad flat troughs, so the displacement must
*compress* sample points toward crests. Take `h(x) = a cos(kx)` and evaluate both
signs of the transfer function under numpy's `Σ X_k e^(+ikx)` convention:

| Transfer | `D(x)` | Effect at the crest |
|---|---|---|
| `−i k/│k│` | `+a sin(kx)` | `dx'/dx = 1 + ak > 1` → crest **broadens**. Wrong. |
| `+i k/│k│` | `−a sin(kx)` | `dx'/dx = 1 − ak < 1` → crest **sharpens**. Correct. |

The second row is exactly the Gerstner trochoid, `x = x₀ − a sin(kx₀)`,
`z = a cos(kx₀)`. The cookbook writes `−1j`; with `+ikx` synthesis that inverts
crests and troughs while preserving every scalar statistic. Pinned by measuring
compression against elevation directly: `corr(J − 1, h) = −0.96`.

### Slopes are analytic, never differenced

Differentiation in Fourier space is multiplication by `ik`:

```python
h       = ifft2(h̃)              × N²
slope_x = ifft2(1j · kx · h̃)    × N²
slope_y = ifft2(1j · ky · h̃)    × N²
```

Two extra inverse FFTs, and the slopes are **exact for every mode the grid
carries**. This matters more than it sounds. A central difference is a low-pass
filter, and slope scales as `k·h` — so the slope spectrum is the height spectrum
weighted by `k²`, and the top of the band dominates, which is precisely what
differencing destroys:

| | mean square slope | vs band-limited theory |
|---|---|---|
| Analytic (spectral) | 0.01906 | +6% |
| Finite difference of the same heights | 0.01121 | **−38%** |

Differencing throws away **41% of the slope variance** and swings the normals by
2.8° on average, up to 12°. In a microfacet BSDF that feeds straight into
roughness: glint too tight and too bright, and the LWIR emissivity error follows.

(The analytic value sits slightly *above* the disc-integrated theory because an
FFT grid carries modes out to the corners of its k-space square, past the Nyquist
radius. Expected, and small.)

### Three properties that drive the design

1. **Evaluable at arbitrary `t`.** No integration, no frame-to-frame state.
   Frame 8000 costs what frame 1 costs and does not drift.
2. **Exactly deterministic from the seed.** Same seed, same `t`, same surface, on
   any machine, forever. That is a V&V property you can write down.
3. **Trivially parallel.** Any node can compute any frame with no coordination.

Tessendorf's `ω` quantisation for seamless looping is deliberately **not**
implemented: it introduces a detectable period, which is poison both for
training-set diversity and for closed-loop temporal analysis.

### Verification

| Check | Measured | Tolerance |
|---|---|---|
| `max│imag(h)│` | 1.7×10⁻¹⁷ m | 10⁻⁹ |
| Realised `Hs`, single tile | +0.3% vs band theory | 5% |
| Crest phase speed vs `ω/k` | 3.1×10⁻⁷ rel. | 5% |
| `Tz` from zero crossings | 1.5% | 10% |
| `min(J)` at choppiness 1.0 | 0.644 | > 0 |
| `corr(J − 1, h)` | −0.96 | < 0 |
| Elevation skewness | +0.012 | │·│ < 0.05 |

The `Tz` check is band-limited to what the grid resolves. Against the
untruncated spectrum the error would be 11.7% — not a bug, just missing
high-frequency content, and the reason the comparison must be banded.

---

## 8. Multi-tile composition

**Module:** `pywave/tiling.py`

### The problem

A single FFT tile is exactly periodic at its side length, and the repetition is
far more visible than it sounds: specular glint makes it obvious, because the eye
and a change detector both lock onto repeated sparkle long before they notice
repeated displacement.

### The construction

Sum several tiles with **incommensurate sizes**, each carrying a **disjoint band**
of the spectrum, each **rotated** by a multiple of the golden angle (≈137.5°).

Disjointness is what makes the sum valid: variances add only for uncorrelated
components, so overlapping bands would double-count energy and the composite
would no longer integrate to `m₀`. `SurfaceConfig` validates that the bands are
contiguous, non-overlapping and span `[0, 1]` at load time.

Rotations derive from the tile **index**, not the RNG — drawing them from the
seed would make the surface depend on it through two independent paths,
violating "the seed is used in `initial_amplitudes` and nowhere else".

### Band fractions to wavenumbers

Config bands are *fractions*, and the reference range has to be pinned because
the obvious reading does not work. Taking fractions of `[0, max_i k_nyq_i]` puts
the middle band at `[12.2, 24.5]` for the shipped config, while the tile carrying
it has a Nyquist of 21.7 — so the top of its own band is unrepresentable and that
energy vanishes silently.

The interpretation used:

- `k_ref = min_i k_nyquist_i` — the most restrictive tile sets the scale,
  guaranteeing every interior band fits on the grid carrying it.
- Interior edges are `fraction × k_ref`.
- The topmost band extends to *its own* tile's Nyquist, so the finest tile
  contributes all the resolution it has instead of discarding its top octave.

For the shipped config (Nyquists 25.1 / 21.7 / 35.0 rad/m) that gives
`[0, 7.61) [7.61, 15.22) [15.22, 34.97)`.

### Frames and rotations

Each tile's grid is rotated by `φ` relative to world. So:

1. World sample points are rotated *into* the tile frame by `−φ`.
2. The tile is evaluated there.
3. Vector results — displacement and slope — are rotated *back* by `+φ`.

Step 3 is easy to forget and produces normals that are subtly wrong per tile in a
way that averages out in `mss` and survives every scalar check. It shows up only
as anisotropy pointing the wrong way — in Phase 7, glint elongated along the
wrong axis.

The counter-rotation of the *wind* direction that keeps wave headings correct
happens at tile construction, not here.

### Sampling must be cubic

Bilinear interpolation is a low-pass filter, and the tiles carry meaningful
energy right to their Nyquist. For a sinusoid at wavenumber `k` on a grid of
spacing `d`, sampled at a uniformly distributed sub-cell offset, retained power
is

```
∫₀¹ │1 − u + u e^(ikd)│² du
```

which at the Nyquist (`kd = π`) is `∫₀¹(1−2u)² du = 1/3`. **Two thirds of the
top-octave power destroyed.** Measured on the shipped config: bilinear loses 5.8%
of `Hs`, cubic 1.3%.

That is not a test-harness artefact — Phase 6 samples this composite at every
mesh vertex, so bilinear would quietly shave variance off the delivered surface
and shave it preferentially from the *high* wavenumbers, which is exactly the
content the LOD invariant is accounting for.

### Verification

| Check | Measured | Tolerance |
|---|---|---|
| Composite `Hs` vs band theory | 1.3% | 5% |
| Sum of per-tile `m₀` vs theory | 2.4×10⁻⁴ rel. | 1% |
| Cubic sampling `Hs` loss | 1.3% | 3% |

---

## 9. Bathymetry

**Module:** `pywave/bathymetry.py`

Three fields, and Phase 5 needs nothing else:

| Field | Definition |
|---|---|
| `depth` | `z_w − z_terrain`. Positive in water. |
| `sdf` | Signed distance to the waterline. **Negative in water**, positive inland. |
| `shore_normal` | `∇s/│∇s│`, unit, pointing **inland**. |

Two constructors produce the same object: `from_export` reads a Phase 4 terrain
export; `dean_beach`/`dean_embayment` manufacture the fields analytically.

### Why the synthetic path exists, and still matters

A synthetic profile has **closed-form answers**. Shoaling on a Dean beach can be
checked against Green's law, refraction against Snell's law, the breaker line
against `d = H/γ_b` — all exactly, at every cell. An exported heightfield gives
none of that; it can only be checked against itself.

So Phase 5 was written and validated against synthetic bathymetry satisfying the
identical §4.5 contract, which made Phase 4 a loader swap rather than a physics
change. The synthetic path is not a placeholder that got thrown away — it is the
oracle real bathymetry is checked against.

### Dean's equilibrium profile

```
h(y) = A · y^(2/3)
```

with `y` the distance offshore and `A` set by sediment grain size. The exponent
follows from assuming wave energy dissipation per unit *volume* is uniform across
the surf zone — the profile that neither erodes nor accretes on average.
Concave-up, which real beaches are and a linear ramp is not: a ramp puts the
breaker line in the wrong place and gives the wrong surf-zone width.

### Constructing the fields, in the right order

The order is not the obvious one. Dean's profile is a function of distance
offshore, which *is* `│sdf│` — so the signed distance must exist before the depth
does. Deriving the sdf from a depth field that was itself built from a distance
would be circular.

1. Define the shoreline geometrically as a curve.
2. Compute `sdf` from an **exact Euclidean distance transform** of the resulting
   water mask. (§4.5 specifically calls for EDT rather than an approximate SDF,
   because Phase 5 differentiates it.)
3. `depth` follows from `sdf`.
4. Smooth `sdf` slightly before differentiating for the normal — raw EDT is noisy
   at the pixel level and that noise propagates straight into refraction.

### Foreshore slope: analytic or measured

Everything downstream (breaker type, runup, swash width) needs one representative
foreshore slope, and "the beach slope" is only meaningful once you say *where*.

- **Synthetic** (`dean_a` set): evaluate the profile exactly at a stated depth.
- **Loaded** (`dean_a is None`): measure the bed — the **median** `│∇z│` over wet
  cells within a band of the target depth. Median rather than mean because a real
  shoreline includes cliffs and gullies that drag a mean around.

The band is **relative** to the target depth (half of it), and that detail is
load-bearing. A fixed band of 0.5 m around a 0.1 m target admits everything out
to 0.6 m depth — tens of metres of seabed on a concave profile — and the median
then reports the slope at the median *distance offshore*, not at the target
depth. Measured against a Dean beach that reads **0.51× the analytic answer, at
every resolution**, so it does not announce itself as a discretisation artefact.

A single number for a whole coastline is a real simplification: a scene with both
cliffs and mudflats has no one foreshore slope. Making it per-cell is a sensible
later refinement.

### Verification

| Check | Measured | Tolerance |
|---|---|---|
| `depth == z_w − terrain_z` | 0 | 10⁻¹² |
| `sign(sdf) == −sign(depth)` beyond a cell of the contour | 0% disagree | 0 |
| `│shore_normal│ − 1` in the nearshore band | 6×10⁻⁸ | 10⁻⁶ |
| Normal points inland | 100% of cells | > 99% |
| Export round trip | 2.3×10⁻⁷ m | 10⁻⁵ |
| Measured vs analytic foreshore slope | within 10% at 3 resolutions | 20% |

A loaded export must also agree with the scene on `water_level` and `epsg`, and
is refused if not. A mismatch builds every mesh at the config's value while
measuring every depth from the export's, so water and terrain agree *with each
other* and both sit at the wrong absolute height — perfect in isolation, metres
out against anything else.

---

## 10. Shoaling

**Module:** `pywave/nearshore.py`

### Physics

Energy flux `E·Cg` is conserved as a wave train moves into shallow water, and
`E ~ H²`, so:

```
Ks = H/H_deep = √(Cg_deep / Cg_local)
```

with `Cg_local` from solving `ω² = gk·tanh(kd)` at the local depth.

**`ω` is the invariant**, not `k`: frequency does not change as a wave shoals,
wavenumber does. The function signature takes `ω` only, so passing a wavenumber
is impossible rather than subtly wrong.

### `Ks` is not monotonic

It dips to a minimum of **0.913 near `kd = 1.2`** before rising, because `n = Cg/c`
grows faster than `c` falls at first. That dip is real physics. An implementation
that clamps `Ks ≥ 1` on the assumption that "shoaling makes waves bigger" would
pass a shallow-water check and still be wrong through the whole
intermediate-depth band — which, for 1-second wind chop, is where the beach
actually is.

In the shallow limit it recovers **Green's law**, `Ks ~ d^(−1/4)`.

### Applied per spectral band

Shoaling is frequency-dependent, so it is applied **per tile** rather than as one
scalar. Each tile is reduced to the frequency carrying its energy centroid:

```
k_rep = Σ(k·S) / Σ(S),    ω_rep = √(g·k_rep)
```

This is a concrete reason the tiles carry *disjoint* bands rather than merely
different sizes.

### Verification

| Check | Measured | Tolerance |
|---|---|---|
| Green's law ratio `Ks(d)/Ks(2d)` | 1.1859 | 2^¼ = 1.1892, ±2% |
| Minimum `Ks` | 0.912993 | 0.913 ± 0.01 |
| `kd` at that minimum | 1.1998 | 1.2 ± 0.15 |
| `Ks` in deep water | 1.0 | ±10⁻³ |
| Max relative step over 6000 depths | 3.6×10⁻⁴ | 5×10⁻³ |

The continuity check exists because an Eckart seed that failed to converge in
shallow water would show up as a kink here and nowhere else.

---

## 11. Refraction

**Module:** `pywave/nearshore.py`

### Physics

Snell's law: `sin α / c = const` along a ray, with `α` measured from the shore
normal. With `c_deep = g/ω` and `c = ω/k`:

```
sin α_local = sin α_deep · c_local / c_deep
```

Exact for straight parallel depth contours. On a curved shoreline it is applied
locally with the local shore normal — standard ray theory.

Ray spacing scales as `cos α`, so as a wave turns toward the normal the rays
spread along the shore and the height drops:

```
Kr = √( │cos α_deep│ / │cos α_local│ )
```

`Kr ≤ 1` on straight parallel contours, always: oblique waves deliver their
energy over a longer stretch of shoreline than normal-incident ones. Total local
height is `H = H_deep · Ks · Kr`.

### Waves travelling offshore

A wave heading *away* from the shore is returned **unrefracted**, so
`α_local = α_deep` and the ray-spacing ratio is exactly 1 — no convergence, no
divergence. The absolute values above are what make that come out right.

Clamping `cos α_deep` at zero instead sends `Kr` to **zero**, which does not
attenuate the wave so much as delete it. That is invisible on an open coast,
where the wind blows onshore everywhere and the case never arises. On a closed
basin it removes the sea from the entire downwind half of the shoreline —
measured on a real lake export, **47.4% of wet cells had exactly zero wave
height**, silently, because everything downstream of `Kr` is a product and zero
propagates without complaint.

### What is approximated

The FFT surface is translation-invariant; refraction is not. A rigorous treatment
needs a mild-slope or Boussinesq solver, which is a much larger project and
unnecessary at 8 cm wave heights.

So refraction is a **per-cell post-process**: it rotates the local wave direction
— the displacement and slope vectors, hence the surface normal, hence everything
the BSDF sees — but it does not re-solve the wave field, so crest *positions* are
unchanged. Shoaling likewise is a per-band amplitude scale, exact for the
amplitude and ignoring the accompanying wavelength shortening.

What is **not** approximated is the coefficient physics: `shoaling_coefficient`
and `refraction_angle` solve the full dispersion relation and Snell's law, and
are checked against closed-form answers at every cell.

### The blend alternative, and why it is not used

The cookbook prescribes a cheaper depth-weighted interpolation toward the shore
normal, `w = clip(1 − d/d_ref, 0, 1)`. It is implemented as
`refraction_angle_blend` but is **not** the production path:

- it has no frequency dependence, so every spectral band would turn at the same
  rate, when refraction is dispersive;
- it reaches full shore-normal alignment at the waterline regardless of incident
  angle, which Snell does not.

Measured disagreement on a test transect: **38.5° peak, 29.3° mean**. Snell's
invariant is conserved to 2.6×10⁻¹⁶, so the exact path costs nothing in accuracy
and very little in time.

### Verification

| Check | Measured | Tolerance |
|---|---|---|
| `sin α / c` spread along a ray | 2.6×10⁻¹⁶ | 10⁻¹² |
| Residual incidence at the waterline, winds 10–170° | 22.3° worst | < 25° |
| `Kr` range at 45° incidence | 0.865 … 1.000 | ≤ 1 |
| `Kr` at normal incidence | 1.0 | exact |
| `Kr` for offshore-bound waves | 1.0 | exact |
| Wet samples with zero height, four wind quadrants | 0% | 0 |

---

## 12. Breaking and depth limiting

**Module:** `pywave/nearshore.py`

### Scope, decided by the numbers

For the reference lake: `Hs = 0.086 m`, `Tp = 1.05 s`, `λp = 1.7 m`, max depth
5 m. The deep-water cutoff is `λ/2 = 0.85 m`, so **the lake is deep water
everywhere except the last few metres**. Breaking happens in ~11 cm of water; the
surf zone is about a metre wide and the swash excursion 0.4 m.

**At 1 m GSD that entire zone is sub-pixel.** Hence the split the module
implements:

- **Shoaling and refraction are geometry** — they act over tens of metres, are
  resolved at sensor scale, and change the surface far outside the surf zone.
- **Breaking and swash are channels** — sub-pixel, so modelled as per-cell
  fractional coverage. Building animated swash geometry would be weeks of work
  invisible at this resolution.

### Criterion and saturation

```
breaking  ⟺  Hs_local > γ_b · d          γ_b = 0.78
```

Inside the surf zone the height is **depth-limited**: a single limiter is derived
from the unlimited `Hs` and applied to every band, so the zone saturates rather
than letting `Ks` diverge at the waterline.

The limiter is shared across bands deliberately. Applying it per band
independently would change the spectral shape inside the surf zone, which is not
what depth limiting does.

A useful consequence: with `Hs ≤ γ_b d` the elevation standard deviation is
`0.195 d`, so even a three-sigma trough is `0.6 d` — comfortably inside the
depth. **That is why the water surface cannot punch through the bed.** Measured
minimum clearance on the shipped scene: +2.8 mm at a vertex in 7.5 mm of water.
Without the limiter the shoaling gain would drive the surface straight through
the bottom in the last few centimetres.

### Breaker type

The Iribarren number (surf similarity parameter):

```
ξ = tan β / √(Hs / L₀),     L₀ = g T² / 2π
```

`ξ < 0.5` spilling, `0.5 ≤ ξ < 3.3` plunging, `ξ ≥ 3.3` surging (Battjes 1974).

| Scene | Foreshore | ξ | Type | Surf zone |
|---|---|---|---|---|
| test lake (synthetic) | 6.7% | 0.33 | spilling | 1.0 m |
| coastal bay (synthetic) | 12.3% | 0.61 | plunging | 48.6 m |
| Houdini export | 31.5% | 1.41 | plunging | 0.35 m |

### Verification

| Check | Measured | Tolerance |
|---|---|---|
| Depth at the outer breaker vs `H/γ_b` | 1.4×10⁻¹⁶ rel. | 5% |
| `Hs / (γ_b d)` inside the surf zone | 1.0 exactly | 10⁻⁶ |
| Water-surface clearance above the bed | +2.8 mm | > 0 |

---

## 13. Swash and wetness

**Module:** `pywave/nearshore.py`

### Runup

Hunt's (1959) formula gives the vertical runup, and the horizontal excursion
follows from the slope:

```
R = ξ · Hs
W = R / tan β
```

A pleasing identity falls out: substituting `ξ = tan β/√(Hs/L₀)` gives

```
W = √(Hs · L₀)
```

— **the swash excursion depends only on the sea state, not the beach slope.**
Which is why the test lake (6.7% foreshore) and the Houdini export (31.5%) both
give 0.38 m despite a five-fold difference in steepness. Worth writing down,
because it otherwise looks like a coincidence or a bug.

### Two wetness functions, deliberately

**`wetness_fraction`** — the fraction of a wave period a point spends submerged.
With the swash edge oscillating as `e(t) = W(1 + sin 2πt/T)/2`, a point at
distance `s` inland is wet whenever `e(t) > s`, for a fraction

```
f(s) = ½ − arcsin(2s/W − 1)/π  =  (1/π)·arccos(2s/W − 1)
```

This is the **thermal** channel. Wet sand has much higher thermal inertia than
dry, so the damp band reads as a cold line in daytime LWIR — one of the most
diagnostic features in a littoral thermal image. That line is set by the *duty
cycle* of wetting, not by where the water happens to be in any one frame, and a
per-frame binary mask would be the wrong input to a thermal solver.

**`wetness`** — the instantaneous smoothstepped field, for rendering. A hard edge
aliases badly.

The two are **not** time-averages of each other: the closed form is the duty cycle
of a *hard* waterline crossing, while the instantaneous function smoothsteps at
each instant and averages to a softer curve. They answer different questions.

### Verification

| Check | Measured | Tolerance |
|---|---|---|
| Closed form vs sampled hard-waterline duty cycle | 5.0×10⁻⁵ | 5×10⁻³ |
| Wetness at mid-swash | 0.5 exactly | 10⁻⁹ |
| Hunt runup (test lake) | 2.5 cm | cookbook ~2 cm |
| Swash excursion | 0.38 m | cookbook 0.3–0.5 m |

---

## 14. Foam

**Module:** `pywave/foam.py`

### Why track it when it is sub-pixel

Whitecap coverage over open water at 5 m/s is ~0.1% and is deliberately not
modelled. Surf-zone foam is different: it is confined to a narrow band, so at 1 m
GSD it occupies a meaningful fraction of the pixels containing the shoreline —
and it is optically nothing like water. High albedo in EO; near-blackbody in LWIR
(`ε ≈ 0.95–0.98`) against water's strongly angular emissivity, and notably
*warmer-looking* than water, which is mostly mirroring cold sky.

A pixel that is 30% foam is radiometrically not a water pixel. So the **fractional
coverage** is what has to be right, not the geometry.

### The model

```
foam[t+dt] = advect(foam[t]) · exp(−dt·ln2/t_half) + rate · dt · breaking
```

Advection is semi-Lagrangian — each cell asks where its foam came from one step
ago and interpolates there — which is unconditionally stable, so the step size is
set by how smooth the result should be rather than by a CFL limit. That matters
because `Cg` varies by an order of magnitude across the surf zone.

### Parameterised by coverage, not rate

The knob is `equilibrium`: the coverage a *continuously breaking* cell settles
at. The seeding rate follows from it and the half life:

```
rate = equilibrium · ln2 / t_half
```

**rather than the other way round.** Seeding and decay are not independent — a
cell converges to `rate · t_half / ln2` — so a rate that behaves at one half life
saturates at another. A fixed `rate = 0.2/s` gives 0.87 coverage at a 3 s half
life and **clips at 1.0** by 6 s, which pinned an entire 50 m surf band at full
coverage with no internal structure.

That is not cosmetic once Phase 7 blends BSDFs by this fraction: a saturated band
renders as pure foam with zero water contribution — no Fresnel, no glint, across
the whole surf zone.

### Bounded spin-up: the one stateful field

Foam has frame-to-frame memory, which would otherwise destroy the "any node, any
frame" property. The fix is **bounded spin-up**: to evaluate time `t`, start from
zero far enough back that the discarded history has decayed below tolerance.

The window is set by the half life, **not** by a frame count:

```
T_spin = t_half · log₂(1/tol)
```

The cookbook's "with a 3 s half-life, 30 frames of spin-up is plenty" is off by
more than an order of magnitude — 30 frames at 30 fps is *one second*, leaving
`2^(−1/3)` = **79%** of the initial condition intact. Reaching 0.5% needs 23 s.
`spinup_steps()` computes it and `evaluate` calls it by default, so the guarantee
cannot be lost by picking a round number.

Note the default foam step is 0.25 s, not a frame time: advection is
unconditionally stable, so a frame-rate step costs 6× more for no visible
difference.

### What does not animate

Breaking here is a **statistical** condition (`Hs_local > γ_b d`), not an
instantaneous one, so the seeding region is steady and foam sits near its
equilibrium. Individual breaking events would need a wave-by-wave model, well
outside this scope.

### Verification

| Check | Measured | Tolerance |
|---|---|---|
| Remaining after one half life | 0.500000 | 10⁻⁹ |
| After two half lives | 0.250000 | 10⁻⁹ |
| Equilibrium vs geometric series | 9×10⁻¹¹ | 10⁻⁶ |
| Equilibrium across half lives 1–30 s | flat at 0.850 | 10⁻⁹ |
| Cold start vs sequential, frame 500 | 0.61% per cell | 1% |

---

## 15. Mesh generation

**Module:** `pywave/mesh.py`

### Constant post spacing, deliberately

Section 6.2 specifies concentric LOD rings, each with its own spacing, stitched
at the boundaries. **Not implemented.** One spacing everywhere makes the geometry
trivial — a regular grid is two triangles per cell, no seams to stitch and no
transition rows to get wrong — and collapses the LOD invariant from per-vertex
bookkeeping into a single global check.

The cost is vertex count, and spacing enters **quadratically**: the reference lake
at its configured 0.125 m is 64 M posts over a 1 km domain. So either the spacing
is coarsened or the mesh is bounded to a region, and **bounding the region is what
stands in for LOD rings** until 6.2 exists.

### Water extent

Meshed where `depth > 0.02 m`, then dilated landward by the swash excursion.

Trimming at a small positive depth rather than exactly zero is not fussiness: the
band around `depth = 0` is where a depth-thresholded mesh goes slivery and where
it z-fights against the terrain it sits on. Cutting at 2 cm and dilating gives a
clean edge that still covers the swash, and the onshore vertices end up *below*
the bed, so the water is hidden rather than fighting for pixels.

A cell becomes two triangles only when **all four** corners are wet. Emitting a
triangle whenever any three are present would chase the waterline more closely,
at the cost of exactly the slivers §6.1 warns about.

### Displacement and normals

```
pos = (x + λ·D_x,  y + λ·D_y,  z_w + h)
n   = normalize(−∂h/∂x, −∂h/∂y, 1)
```

Normals come from the **analytic** spectral slopes, never from the triangles —
see §7. `face_normals()` exists only as a check; Gate 6 compares the two and
expects a few degrees of disagreement (4.3° mean at 0.25 m posts). Agreement to
machine precision would mean the analytic normals had been silently replaced.

Terrain is the opposite case and correctly uses **geometric** normals: the bed
exists only as a sampled grid, with a shoreline from a distance transform rather
than a closed form, so there is no analytic slope to prefer and a central
difference is the honest answer.

### Verification

| Check | Measured | Tolerance |
|---|---|---|
| `│normal│ − 1` | 6×10⁻⁸ | 10⁻⁵ |
| Minimum normal `z` | 0.88 | > 0 |
| Wet posts not meshed | 0 | 0 |
| Onshore clearance below the bed | ≥ 0 | ≥ 0 |
| Analytic vs face normals | 4.3° mean | < 15° |
| Rebuilt frame difference | 0 | 0 |

---

## 16. Per-vertex channels and the LOD invariant

**Modules:** `pywave/channels.py`, `pywave/export.py`

### The invariant

An FFT grid or a mesh of spacing `dx` cannot represent anything shorter than
`2dx`. The slope variance it is missing is not lost — it is *accounted for* and
handed to the BSDF as sub-facet roughness:

```
mss_resolved(dx)  +  mss_above(π/dx)  =  mss_total
```

This is what keeps appearance constant across LOD transitions: geometry lost when
the mesh coarsens reappears as roughness, and the total radiometric response is
unchanged. Skipping it produces the classic failure — distant water rendered
mirror-smooth because its roughness was baked at the fine spacing.

Measured closure: **8.2×10⁻⁵** at the mesh spacing, and better than 10⁻⁶ across
mesh spacings from 0.0625 to 2 m.

### The channels

| Channel | Meaning |
|---|---|
| `mss` | Sub-mesh mean square slope at this spacing **and local depth** → `α = √mss` |
| `wdir_x`, `wdir_y` | Local **refracted** wave direction, unit |
| `aniso` | Crosswind/upwind slope variance ratio |
| `depth` | Local still-water depth |
| `foam` | Foam coverage fraction |
| `wetness` | Submergence duty cycle |

### `mss` is depth-dependent, which is not obvious

The tempting argument is that sub-mesh waves are short — at 0.125 m posts the cut
is `k = 25 rad/m`, wavelengths under 25 cm — so they are deep-water waves
everywhere and the sub-mesh share is a scene constant. True over almost the whole
domain and false exactly where it matters:

| Depth | `kd` | Sub-mesh mss vs deep |
|---|---|---|
| 3 cm | 0.75 | **+44%** |
| 5 cm | 1.26 | +11% |
| 10 cm | 2.51 | +0.6% |
| 20 cm+ | > 5 | +0.0% |

Converged past `kd ≈ 2.5`, badly wrong below it — and below it is the surf and
swash band. A coarser mesh is worse: +142% at 3 cm for 0.25 m posts. So the depth
is used, via a log-spaced lookup interpolated per vertex, because `mss_above` is
a radial quadrature far too slow to call millions of times.

### What Phase 7 does with them

The baked constants in the starter scene become per-vertex reads. Cookbook §7.3
gives the mapping, warning that a stray `√2` is the likeliest single bug in the
plugin. Beckmann's slope distribution is Gaussian with per-axis variance `α²/2`,
so summing both axes gives `mss = α²`:

```
isotropic     α   = √mss
anisotropic   α_u = √(2·mss_up)
              α_v = √(2·mss_cross)
```

The two shipped channels recover the anisotropic pair, since
`mss = mss_up + mss_cross` and `aniso = mss_cross/mss_up`:

```
mss_up    = mss / (1 + aniso)
mss_cross = mss · aniso / (1 + aniso)
```

Verified against direct computation to 1.2×10⁻⁵.

§7.4 adds a trap: those live in the **tangent frame**, which Mitsuba derives from
UV parameterisation — and on a displaced water mesh UVs are arbitrary, so the
anisotropy axis would rotate randomly and would not follow refraction nearshore.
Build the frame from `wdir` and ignore UV tangents. That is what `wdir_x/y` are
carried for.

Foam blends by coverage, `f = (1 − foam)·f_water + foam·f_foam`, which is the
correct form for an unresolved sub-pixel mixture: radiance adds linearly weighted
by area fraction.

### Delivery format

Binary PLY, because it has custom per-vertex properties and Mitsuba's loader
passes unknown ones through as mesh attributes. OBJ has positions, normals and
texture coordinates and no mechanism for arbitrary named scalars, so it carries
**none** of the above — everything that makes the mesh worth generating. The
round trip is verified bit-exact.

---

## 17. Summary of approximations

Everything the model does *not* do exactly, in one place.

| # | Approximation | Why acceptable here | Where it would bite |
|---|---|---|---|
| 1 | Linear wave theory; no Stokes bound harmonics | 8 cm waves, `ka ≈ 0.08` | Steep seas; elevation skewness is ~0 rather than positive |
| 2 | Refraction rotates vectors, does not re-solve the field | Crest positions matter less than normals for a BSDF | Strong focusing behind a headland |
| 3 | Shoaling scales amplitude, ignores wavelength shortening | Amplitude drives the radiometry | Detailed surf-zone geometry |
| 4 | One foreshore slope per scene | Uniform coasts | A scene with both cliffs and mudflats |
| 5 | One fetch per scene | Open coast with a dominant wind | Closed basins, where fetch is direction-dependent |
| 6 | Breaking is statistical, not wave-by-wave | Sub-pixel at 1 m GSD | Resolving individual breakers |
| 7 | `K_CAPILLARY = 400 rad/m` cut on the slope integral | Stated once, reported with every mss | Any comparison against a different cut |
| 8 | Constant mesh post spacing (no LOD rings) | Bounded region substitutes | Whole-domain meshes at fine spacing |
| 9 | Foam spin-up truncates history at 0.5% | Bounded and stated | Nothing; the residual is a declared number |

### Deviations from the cookbook

Four, each with the measurement that motivated it — all recorded in the
validation report.

1. **Gate 1's "Hs within 2%"** is unmeetable; JONSWAP's two fits agree at exactly
   one fetch. Replaced by pinning the *relation* across four decades of fetch,
   which is strictly stronger.
2. **Gate 2's "skewness in [0, 0.3]"** is unmeetable; the model is linear in
   elevation. Replaced by asserting Gaussianity, with the displaced skewness
   reported unbounded.
3. **§5.5's "30 frames of spin-up"** leaves 79% of the initial condition.
   Replaced by deriving the window from the half life.
4. **§5.3's refraction blend** has no frequency dependence and over-aligns at the
   waterline. Snell is used instead; the blend is retained and its disagreement
   measured.

---

## 18. References

**Wave spectra**

- Hasselmann, K. et al. (1973). "Measurements of wind-wave growth and swell decay
  during the Joint North Sea Wave Project (JONSWAP)." *Deutsche Hydrographische
  Zeitschrift*, Suppl. A8.
- Hasselmann, D., Dunckel, M. & Ewing, J. (1980). "Directional wave spectra
  observed during JONSWAP 1973." *J. Phys. Oceanogr.* **10**, 1264–1280.

**Surface statistics**

- Cox, C. & Munk, W. (1954). "Measurement of the roughness of the sea surface from
  photographs of the sun's glitter." *J. Opt. Soc. Am.* **44**(11), 838–850.

**Synthesis**

- Tessendorf, J. "Simulating Ocean Water." SIGGRAPH course notes.

**Nearshore**

- Dean, R.G. (1977). "Equilibrium beach profiles: U.S. Atlantic and Gulf coasts."
  Ocean Engineering Report No. 12, University of Delaware.
- Dean, R.G. & Dalrymple, R.A. (1991). *Water Wave Mechanics for Engineers and
  Scientists.* World Scientific.
- Hunt, I.A. (1959). "Design of seawalls and breakwaters." *J. Waterways and
  Harbors Division*, ASCE **85**, 123–152.
- Battjes, J.A. (1974). "Surf similarity." *Proc. 14th Coastal Engineering Conf.*
- Eckart, C. (1952). Approximate dispersion solution, used to seed the Newton
  iteration in `dispersion_k`.

**Optics**

- Hale, G. & Querry, M. (1973). "Optical constants of water in the 200-nm to
  200-µm wavelength region." *Appl. Opt.* **12**(3), 555–563.

**Project internal**

- `water-surface-modeling-primer.md` — physics rationale.
- `littoral-water-implementation-cookbook.md` — phase plan and gate checklists.
