# Water Surface Modeling for Physics-Based EO/IR Rendering

*A primer on wave spectra, FFT surface synthesis, nearshore transformation, and the optical link — oriented toward littoral scene generation.*

---

## 0. How the pieces fit together

Before any of the individual topics make sense, it helps to see the chain they form. There are five links, and each one has its own literature that mostly doesn't talk to the others:

```
   wind speed, fetch, depth          "sea state"
              ↓
        WAVE SPECTRUM  S(k,θ)        oceanography  (JONSWAP, PM, TMA)
              ↓
      SURFACE REALIZATION  h(x,t)    graphics      (Tessendorf FFT)
              ↓
   NEARSHORE TRANSFORMATION          coastal eng.  (shoaling, refraction, breaking)
              ↓
     OPTICAL / RADIOMETRIC           optics        (Cox-Munk, Fresnel, complex IOR)
```

The single most important conceptual distinction in this whole document:

> **The spectrum describes what the water is *statistically*. The realization is one particular surface consistent with those statistics.**

A spectrum doesn't tell you where any wave crest is. It tells you how much variance lives at each wavelength and direction. An infinite number of different-looking surfaces satisfy the same spectrum, and they are all equally correct. You pick one by choosing a random seed.

This matters practically because it tells you where to put your rigor. **Defensibility lives in the spectrum; appearance lives in the realization.** If your spectrum is right, any realization drawn from it is physically valid — you don't have to justify why one wave is where it is. That's the entire argument for owning the spectrum code and being relaxed about everything downstream of it.

---

## 1. Linear wave theory: the minimum you need

### A single wave

A sinusoidal water wave is described by:

| Symbol | Name | Definition |
|---|---|---|
| `a` | amplitude | crest height above mean level |
| `H` | wave height | `2a`, crest to trough |
| `λ` | wavelength | crest to crest |
| `k` | wavenumber | `2π/λ` |
| `T` | period | time between crests at a fixed point |
| `ω` | angular frequency | `2π/T` |
| `c` | phase speed | `ω/k` — speed a crest moves |
| `d` | depth | still-water depth |

### The dispersion relation

This one equation governs almost everything:

```
ω² = g·k·tanh(k·d)
```

It links frequency to wavelength, and the `tanh(kd)` term is where depth enters. Two limits:

**Deep water** (`kd > π`, i.e. `d > λ/2`): `tanh(kd) → 1`, so

```
ω² = g·k        c = √(g/k) = √(gλ/2π)
```

Phase speed depends on wavelength. Longer waves move faster. This is *dispersion*, and it's why a distant storm produces swell that arrives sorted by period — the long stuff gets there first.

**Shallow water** (`kd < π/10`, i.e. `d < λ/20`): `tanh(kd) → kd`, so

```
ω² = g·k²·d     c = √(g·d)
```

Phase speed depends only on depth, not wavelength. Waves stop dispersing and all travel at the same speed. This is also why waves slow down as they approach a beach — and slowing down is what causes shoaling and refraction.

**Worked example for a small lake.** A 1 km fetch at 5 m/s wind gives roughly a 1.7 m wavelength (we'll derive this in §2). Deep-water cutoff is `λ/2` = 0.85 m. So in a lake with a 5 m maximum depth, *the entire lake is deep water* except the final few meters near the shoreline. The nearshore transition band is narrow. That is a very useful thing to know before you spend a month building a shallow-water solver.

### Group velocity

Energy travels at the *group* velocity, not the phase velocity:

```
Cg = dω/dk
```

In deep water `Cg = c/2` (energy moves at half the speed of the crests — individual waves appear at the back of a group, move forward through it, and vanish at the front). In shallow water `Cg = c`. The transition between these is what drives shoaling amplification, so this is not a trivia item.

---

## 2. The wave spectrum

### Why a spectrum at all

Real wind-driven water is not a wave. It's thousands of waves of different wavelengths and directions superposed with random phases. By the central limit theorem, the resulting surface elevation is very nearly a **Gaussian random field** — and a Gaussian random field is *completely* described by its power spectrum. Nothing else is needed. This is a remarkably strong result and it's what makes the whole spectral approach work.

(Real water is not exactly Gaussian. Crests are slightly sharper and troughs slightly flatter than a Gaussian would predict — a skewness of roughly +0.1 to +0.2. Tessendorf's "choppiness" term in §3 is a first-order correction for exactly this.)

### Spectral moments — the bridge to observables

Define the n-th moment of the spectrum:

```
mₙ = ∫ ωⁿ · S(ω) dω
```

These connect the abstract spectrum to things you can measure:

| Moment | Meaning | Use |
|---|---|---|
| `m₀` | variance of surface elevation | **`Hs = 4√m₀`** — significant wave height |
| `m₂` | related to slope / zero-crossings | mean zero-crossing period `Tz = 2π√(m₀/m₂)` |
| `m₄` | related to curvature | crest statistics, spectral bandwidth |

And in wavenumber space, the one that matters most for rendering:

```
mean square slope   σ² = ∫ k² · S(k) d²k
```

**Read that again.** Mean square slope — the quantity that sets microfacet roughness in your BSDF — is a *moment of the wave spectrum*. It is not a free parameter and it is not an artistic choice. If you own the spectrum, you can compute it. If you don't, you're guessing.

This is also why the wavenumber integration limits matter so much. Slopes from wavelengths your mesh resolves are already in the geometry. Slopes from wavelengths below your mesh Nyquist must go into the BSDF. The cutoff between them is the boundary of that integral, and — critically — it can be different for different sensor bands.

### Pierson-Moskowitz (1964)

The simplest useful spectrum. It describes a **fully developed sea**: wind has blown at constant speed over unlimited fetch for long enough that the waves stop growing. One parameter — wind speed.

```
S(f) = α g² (2π)⁻⁴ f⁻⁵ exp[ −(5/4)(f_p/f)⁴ ]
```

with `α = 0.0081` (the Phillips constant) and `f_p = 0.13 g / U₁₀`.

The `f⁻⁵` tail is the Phillips equilibrium range — the idea that at high frequencies, waves are limited by breaking rather than by wind input, so the spectrum takes a universal shape. (Modern work often prefers `f⁻⁴` over part of this range. This matters for you, because the tail is exactly where your mean square slope lives.)

### JONSWAP (Hasselmann et al., 1973)

The Joint North Sea Wave Project measured wave growth along a fetch line in the North Sea. The finding: a **fetch-limited** sea — still growing, not yet fully developed — has a *sharper, more peaked* spectrum than Pierson-Moskowitz. Energy piles up near the peak frequency because nonlinear wave-wave interactions haven't yet spread it out.

JONSWAP is PM multiplied by a peak enhancement factor:

```
S(f) = α g² (2π)⁻⁴ f⁻⁵ exp[ −(5/4)(f_p/f)⁴ ] · γ^r

where   r = exp[ −(f − f_p)² / (2 σ² f_p²) ]
        σ = 0.07  for f ≤ f_p
        σ = 0.09  for f > f_p
        γ = 3.3   (typical; γ = 1 recovers Pierson-Moskowitz)
```

The fetch dependence enters through dimensionless fetch:

```
X̃ = g·X / U₁₀²          (X = fetch in metres)

α   = 0.076 · X̃^(−0.22)
f_p = 3.5 (g/U₁₀) · X̃^(−0.33)
```

**This is the right spectrum for a lake.** A small water body is fetch-limited essentially by definition — the wind runs out of water before the waves finish growing.

**Worked example, 1 km fetch, 5 m/s wind:**

```
X̃   = 9.81 × 1000 / 5²  = 392
f_p = 3.5 × (9.81/5) × 392^(−0.33) ≈ 0.95 Hz    →  T_p ≈ 1.05 s
Hs  ≈ 8 cm    (from g·Hs/U² ≈ 0.0016·X̃^0.5)
λ_p = g T_p² / 2π ≈ 1.7 m
```

Eight-centimetre waves with a one-second period. Get comfortable with how small this is — it drives every downstream decision. It means the surf zone is about 2 m wide, the swash band is sub-metre, and whitecap coverage is around 0.1% (negligible). Almost all of the visual and radiometric character of this water lives in *sub-mesh-scale roughness*, not in resolved wave geometry.

### TMA (Bouws et al., 1985)

JONSWAP extended to finite depth. It multiplies JONSWAP by a depth-dependent factor (the Kitaigorodskii function) that suppresses the high-frequency tail in shallow water. Worth knowing exists — for a lake with a broad shallow shelf it's more correct than plain JONSWAP, though for a mostly-deep 5 m lake the difference is small.

### Directional spreading

Everything above is 1-D — energy per frequency. Real seas spread over direction:

```
S(f, θ) = S(f) · D(f, θ)      with   ∫ D dθ = 1
```

The common form is `D(θ) ∝ cos^(2s)((θ − θ_w)/2)`, where `s` controls the narrowness. Larger `s` = more tightly aligned with the wind. Two things to know:

- Spreading is **frequency-dependent**. Waves near the spectral peak are tightly aligned with the wind; high-frequency waves spread much more broadly.
- Short fetch → narrower spreading. Lake waves are more directional than ocean waves.

For a better model, look at Donelan-Banner spreading or Horvath (2015), which is written specifically for graphics implementers.

**Converting S(f) to S(k):** you'll need this, and it's where implementations most often go wrong. Use the dispersion relation and the Jacobian:

```
S(k) = S(ω) · (dω/dk) / k
```

Get the normalization wrong here and your Hs will be off by a factor you'll spend days hunting. See the validation tests in §6 — this is exactly what they catch.

---

## 3. Tessendorf FFT synthesis

### The problem

You have `S(k,θ)`. You want an actual surface `h(x, t)` on a grid, evolving in time, that is a valid realization of that spectrum.

The naive approach — sum thousands of sinusoids at every grid point — is O(N_gridpoints × N_waves). For a 512×512 grid that's hopeless. The Tessendorf method is just the observation that **this sum is a Fourier transform**, so an FFT does it in O(N log N). That's the whole trick. Everything else is bookkeeping.

Reference: Jerry Tessendorf, *"Simulating Ocean Water,"* SIGGRAPH course notes (1999–2004). Freely available and still the standard reference. It's written for graphics people, so it's readable, but it is loose about some of the oceanography.

### Step 1: the initial spectrum grid

On a `N×N` grid of wavevectors `k`, draw:

```
h̃₀(k) = (1/√2) · (ξ_r + i·ξ_i) · √( S(k) · Δk_x · Δk_y )
```

where `ξ_r, ξ_i ~ N(0,1)` are independent standard normals.

Conceptually: **take white noise, and shape it with the square root of the spectrum.** That's all this is — spectral filtering of noise. The `√` appears because the spectrum is a *power* density and you're building an *amplitude*. The `Δk_x Δk_y` converts spectral density to per-mode variance.

This array is computed once. The random seed lives here and nowhere else, which is what makes the whole thing reproducible.

### Step 2: time evolution

```
h̃(k, t) = h̃₀(k) · e^(iω(k)t)  +  h̃₀*(−k) · e^(−iω(k)t)
```

Two things are happening:

**Each mode rotates in phase at its own frequency**, given by the dispersion relation. Long waves rotate slowly, short waves rapidly. That's dispersion, implemented for free.

**The Hermitian pairing (the second term) makes the output real.** A complex field with `h̃(−k) = h̃*(k)` inverse-transforms to a purely real signal. Physically, it's a pair of waves travelling in opposite directions.

This structure has three consequences that matter enormously for your use case:

1. **It's evaluable at arbitrary `t`.** No integration, no frame-to-frame state. Frame 8,000 costs exactly what frame 1 costs, and it doesn't drift.
2. **It's exactly deterministic** from the seed. Same seed, same `t`, same surface, on any machine, forever. That's a V&V property you can write in a document.
3. **It's trivially parallel.** Any node can compute any frame with no coordination.

For closed-loop simulation over minutes, these three properties are worth more than everything else in this document combined.

*One caution:* Tessendorf describes quantizing the `ω` values to integer multiples of a base frequency `ω₀ = 2π/T_loop`, which makes the animation seamlessly loop. **Don't do this.** It introduces a detectable period into your imagery — poison for training data and for closed-loop temporal analysis.

### Step 3: inverse FFT

```
h(x, t) = IFFT2[ h̃(k, t) ]
```

Done. That's your displacement field.

### Choppiness (horizontal displacement)

Linear theory gives sinusoidal waves — symmetric crests and troughs. Real waves have sharp crests and broad flat troughs (Stokes waves). Tessendorf approximates this by *also* displacing points horizontally:

```
D̃(k, t) = −i · (k / |k|) · h̃(k, t)     →   IFFT →   D(x, t)
```

Each surface point moves to `x + λ_c·D(x,t)` and up to `h(x,t)`.

**The magnitude is not free.** `λ_c = 1` is the physically correct first-order value; it's the Hilbert transform of the elevation field, fully determined by the spectrum. DCC tools expose it as an artist slider, which is convenient and physically wrong — it changes your slope distribution, and slope distribution is your BRDF.

Push it too far and the mapping becomes non-injective: the surface folds through itself and normals invert. Detect this by checking the Jacobian determinant of the displacement map for negative values. (In VFX, those fold regions are where foam gets seeded — a nice trick, though for a lake at 8 cm wave height you'll rarely trigger it.)

### Getting slopes for free

Don't finite-difference the mesh. Differentiation in Fourier space is multiplication by `ik`:

```
slope_x = IFFT2[ i·k_x · h̃(k,t) ]
slope_y = IFFT2[ i·k_y · h̃(k,t) ]
```

These are analytically exact for the band you've resolved, cost one extra FFT each, and are noticeably better than mesh normals. For a radiometric renderer where the normal *is* the physics, this is worth doing.

### Tiling and periodicity

The FFT surface is exactly periodic at the tile size. This is the main practical annoyance, and it's worse than it looks — specular glint makes repetition far more visible than displacement does, because the eye (and a change detector) locks onto repeated sparkle patterns.

Mitigations, in order of effectiveness:
- **Sum several incommensurate tiles** (e.g. 64 m, 37 m, 23 m), each carrying a different band of the spectrum. This is the standard approach and it works well.
- Rotate tiles relative to each other.
- Make the largest tile bigger than anything in frame.

---

## 4. Nearshore transformation

Everything above assumes constant depth. Near the shore, depth varies, and four things happen. This is classical coastal engineering — a mature, well-documented field.

### Shoaling

As depth decreases, `Cg` changes. Energy flux `E·Cg` is conserved, and since `E ∝ H²`:

```
Ks = √( Cg_deep / Cg_local )       H_local = Ks · H_deep
```

Waves initially get slightly *shorter* as `Cg` rises through the transition zone, then grow as `Cg` falls in shallow water.

### Refraction

Waves slow in shallow water, so a crest arriving obliquely pivots — the shoreward end slows first. Snell's law applies directly:

```
sin(θ) / c = constant
```

The practical consequence: **waves always end up nearly parallel to the shoreline**, regardless of wind direction. If your render shows waves hitting a beach at 40°, it's wrong. Your SDF gradient gives you the shore normal for free, so this is cheap to approximate well.

### Breaking

Depth-limited breaking:

```
H_b ≈ γ_b · d       with γ_b ≈ 0.78  (range 0.6–1.2)
```

For 8 cm waves that's breaking in ~10 cm of water. On a 5% slope, a surf zone about **2 m wide**.

The **Iribarren number** classifies the breaker type:

```
ξ = tan(β) / √(H/L₀)         β = beach slope, L₀ = deepwater wavelength

ξ < 0.5      spilling   (foam cascades down the face — gentle)
0.5 < ξ < 3.3  plunging (the curling barrel)
ξ > 3.3      surging    (wave runs up without breaking)
```

Small lake: `ξ ≈ 0.23` → spilling. No barrels.

### Runup and swash

The waterline oscillates. Hunt's formula gives vertical runup `R ≈ ξ · H`, which for our lake is about 2 cm vertical — a horizontal swash excursion of roughly 0.3–0.5 m.

**This is sub-pixel at any realistic GSD.** Which means it should be a *wetness fraction channel*, not animated geometry. Recognizing this early saves a lot of wasted effort.

---

## 5. The optical link

### Cox and Munk (1954)

The foundational measurement. Cox and Munk photographed sun glitter from an aircraft and inferred the distribution of sea surface slopes from the glitter pattern's shape. They found the slope distribution is nearly Gaussian with variance linear in wind speed:

```
σ² = 0.003 + 0.00512 · U₁₂          (isotropic, clean surface)
```

with an anisotropic version splitting up-wind and cross-wind components.

**Here's the connection that makes everything click:** recall from §2 that `σ² = ∫k²S(k)d²k`. Cox-Munk is an *empirical measurement of a moment of the wave spectrum.* Oceanography predicts it; Cox and Munk measured it. They should agree.

Which gives you a free validation test: compute mean square slope from your JONSWAP spectrum, compare to Cox-Munk at the same wind speed. If they're wildly different, your spectrum normalization is wrong.

*Caveat for your case:* the Cox-Munk fit is from open ocean at long fetch. A small freshwater body at short fetch has a different high-frequency balance, and this relation will likely over-predict. Treat it as a starting point and a sanity bound, not ground truth.

### The multi-scale cutoff problem

This is the deepest idea in the optical link, and it's the one that determines whether your EO/IR bands stay physically consistent.

Surface slope structure spans wavelengths from hundreds of metres down to millimetres. Your mesh resolves down to some cutoff `k_max`. Everything above that cutoff must be represented statistically in the BSDF:

```
resolved geometry       slopes from  k < k_max     →  mesh normals
sub-mesh roughness      slopes from  k > k_max     →  BSDF microfacet σ²
```

Both integrals come from the *same spectrum*. This guarantees they sum correctly and don't double-count.

**And the cutoff is band-dependent.** A microfacet model assumes facets are large compared to the wavelength of light. At 0.5 µm, essentially all capillary structure qualifies. At 10 µm it does not — structure comparable to the wavelength enters a diffraction regime the model doesn't cover.

The practical upshot for co-registered EO/MWIR/LWIR: **one mesh, several roughness channels**, each computed by integrating the same spectrum with a different lower limit. That's what keeps your bands genuinely registered rather than independently tuned to look right.

Bruneton, Neyret & Holzschuch (2010) treat this geometry-to-BRDF transition carefully and are worth reading closely.

### Fresnel, complex IOR, and emissivity

Water's optical constants change dramatically across your bands:

| λ | n | k | Notes |
|---|---|---|---|
| 0.55 µm | 1.333 | ~1e-9 | essentially transparent; bottom visible |
| 4 µm | ~1.35 | ~0.0046 | weakly absorbing |
| 10 µm | ~1.22 | ~0.051 | effectively opaque (penetration ~16 µm) |

Consequences:

- **In the visible you need the bottom.** Beer-Lambert attenuation through the water column with a turbid-lake coefficient of roughly 1–3 m⁻¹. For a littoral scene this is the whole point.
- **In LWIR the water is opaque**, so `ε = 1 − ρ` and everything is a surface effect.
- **Angular emissivity is the dominant LWIR signature driver.** Water is ~0.98 at nadir but falls sharply beyond about 60° incidence. Oblique-look IR scenes live or die on this. A Lambertian shortcut fails immediately.
- **Complex IOR breaks most dielectric BSDFs.** Standard `roughdielectric` implementations assume real IOR. For the IR bands you'll likely need a conductor-style BSDF with water's complex constants, or a custom plugin.

The standard citation for the optical constants — covering 200 nm to 200 µm, so your whole range in one paper — is Hale & Querry (1973).

### The thermal side

Worth flagging even though it isn't wave physics: a water body's **skin temperature** runs typically 0.1–0.5 K below the bulk, from evaporative and radiative cooling at the surface. That's well within microbolometer NEDT, so it's a real signature feature, and it's a thermal solver problem rather than a renderer problem.

Similarly, the **capillary fringe** — the damp soil band just above the waterline — has much higher thermal inertia than dry soil, so it reads as a distinct cold line in daytime LWIR. On a littoral scene this is one of the most visually diagnostic features in the whole image, and it comes entirely from the thermal model, not the water surface.

---

## 6. Validation experiments

These build intuition and double as your unit tests. Run them in order; each catches a specific class of error.

**1. Does the realized surface have the right height?**
Generate `h(x,t)`, compute `4·std(h)`, compare to the JONSWAP closed-form `Hs`. Should match within a few percent for a large enough grid. *Catches: spectrum normalization errors, Jacobian mistakes in the S(f)→S(k) conversion.* This is the single most valuable test in the list.

**2. Does it have the right period?**
Extract a time series at one point, count zero-crossings, compare `Tz` to `2π√(m₀/m₂)`. *Catches: dispersion relation errors, frequency-axis scaling.*

**3. Is the slope variance sane?**
Compute `mean(slope_x² + slope_y²)` and compare to Cox-Munk at the same wind speed. Expect the same order of magnitude, not exact agreement. *Catches: high-frequency tail truncation, grid resolution too coarse.*

**4. Is the elevation distribution Gaussian?**
Histogram `h`. Should be near-Gaussian with slight positive skew once choppiness is on. *Catches: bugs in the Hermitian pairing.*

**5. Does the surface fold?**
Compute the Jacobian determinant of the displacement map; check for negatives. *Catches: excessive choppiness — which produces inverted normals and radiometric nonsense.*

**6. Do waves move at the right speed?**
Track a single crest across frames; compare to `c = ω/k`. *Catches: sign errors and time-scaling bugs in the evolution term.*

**7. Is it reproducible?**
Same seed, same `t`, two different machines. Should be bit-identical or very near. *This is your V&V artifact.*

---

## 7. Suggested reading path

**Week 1 — foundations.** Holthuijsen, *Waves in Oceanic and Coastal Waters*, chapters 1–5. This is the best single book for your purpose: rigorous but genuinely readable, and it covers spectra and nearshore transformation in one volume. If you read only one thing, read this.

**Week 2 — synthesis.** Tessendorf's SIGGRAPH notes, then Horvath (2015) for the more careful spectral treatment. Implement the basic FFT surface alongside the reading — it's short enough that coding it *is* the study method, and validation test #1 will teach you more than any amount of reading.

**Week 3 — nearshore.** USACE *Coastal Engineering Manual*, Part II. It's free, comprehensive, and being a Corps of Engineers publication it carries useful weight in a DoD traceability argument. Dean & Dalrymple if you want the derivations.

**Week 4 — optics.** Cox & Munk (1954) — it's short and surprisingly readable for a 1954 paper. Then Bruneton et al. (2010) for the scale-transition treatment, and Mobley's *Ocean Optics Web Book* (free online) for the radiometry.

---

## 8. References

### Spectra
- Pierson, W. J. & Moskowitz, L. (1964). "A proposed spectral form for fully developed wind seas." *J. Geophys. Res.* 69(24), 5181–5190.
- Hasselmann, K. et al. (1973). "Measurements of wind-wave growth and swell decay during the Joint North Sea Wave Project (JONSWAP)." *Deutsche Hydrographische Zeitschrift*, Suppl. A8. — *The JONSWAP paper.*
- Bouws, E. et al. (1985). "Similarity of the wind wave spectrum in finite depth water." *J. Geophys. Res.* 90(C1). — *TMA.*
- Donelan, M., Hamilton, J. & Hui, W. (1985). "Directional spectra of wind-generated waves." *Phil. Trans. R. Soc. A* 315.

### Synthesis
- Tessendorf, J. "Simulating Ocean Water." SIGGRAPH course notes. — *Free online. The standard graphics reference.*
- Horvath, C. (2015). "Empirical directional wave spectra for computer graphics." *DigiPro 2015.* — *Bridges the oceanography and graphics literatures; very practical.*

### Waves and coastal engineering
- Holthuijsen, L. (2007). *Waves in Oceanic and Coastal Waters.* Cambridge UP. — **Best single reference for this work.**
- Dean, R. G. & Dalrymple, R. A. (1991). *Water Wave Mechanics for Engineers and Scientists.* World Scientific.
- USACE. *Coastal Engineering Manual*, EM 1110-2-1100. — *Free. Authoritative. DoD-friendly provenance.*

### Optics and radiometry
- Cox, C. & Munk, W. (1954). "Measurement of the roughness of the sea surface from photographs of the sun's glitter." *J. Opt. Soc. Am.* 44(11), 838–850.
- Hale, G. M. & Querry, M. R. (1973). "Optical constants of water in the 200-nm to 200-µm wavelength region." *Appl. Opt.* 12(3), 555–563. — *The complex IOR source; covers your entire band range.*
- Bruneton, E., Neyret, F. & Holzschuch, N. (2010). "Real-time realistic ocean lighting using seamless transitions from geometry to BRDF." *Computer Graphics Forum* 29(2). — *Directly on your multi-scale cutoff problem.*
- Ross, V., Dion, D. & Potvin, G. (2005). "Detailed analytical approach to the Gaussian surface BRDF specular component applied to the sea surface." *J. Opt. Soc. Am. A* 22(11).
- Mobley, C. *Ocean Optics Web Book* — *Free, excellent, actively maintained.*

---

## Appendix: symbols

| Symbol | Meaning |
|---|---|
| `a`, `H` | amplitude; wave height (`H = 2a`) |
| `Hs` | significant wave height, `4√m₀` |
| `λ`, `k` | wavelength; wavenumber `2π/λ` |
| `T`, `ω` | period; angular frequency `2π/T` |
| `f_p`, `T_p` | peak frequency; peak period |
| `c`, `Cg` | phase speed; group velocity |
| `d` | still-water depth |
| `g` | 9.81 m/s² |
| `S(f)`, `S(k)` | spectral density in frequency / wavenumber |
| `mₙ` | n-th spectral moment |
| `σ²` | mean square slope |
| `X`, `X̃` | fetch; dimensionless fetch `gX/U²` |
| `U₁₀` | wind speed at 10 m reference height |
| `γ` | JONSWAP peak enhancement (≈3.3) |
| `α` | Phillips constant |
| `β` | beach slope |
| `ξ` | Iribarren number |
| `γ_b` | breaker index `H/d` (≈0.78) |
| `Ks` | shoaling coefficient |
| `n`, `k` | real and imaginary refractive index |
| `ε` | emissivity |
