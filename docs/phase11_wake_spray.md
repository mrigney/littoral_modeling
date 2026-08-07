# Phase 11 — boat wake, turbulent wake and bow spray

**Status: architecture only. Nothing here is built.**

Depicting a boat's disturbance of the water, given an external 6-DOF motion
model. Target: small fast craft (Boghammar class, ~10 m LOA, 6–40 kn) in a
littoral scene of 2–20 m depth.

---

## 0. Nomenclature

Everything used below, defined once. Angles are radians internally and degrees
only where a human reads them, as everywhere else in this package
(`algorithms.md` §2).

### Symbols

| Symbol | Meaning | Units |
|---|---|---|
| `g` | acceleration due to gravity, 9.81 | m/s² |
| `x, y` | horizontal scene coordinates, Z up | m |
| `z_w` | still-water level — the mean free surface the scene is built on | m |
| `d` | water depth below `z_w`, positive in water | m |
| `h` | surface elevation relative to `z_w`; subscripted by which system produces it | m |
| `ζ` (zeta) | the Kelvin wake's contribution to `h` | m |
| `t` | scenario time | s |
| `U` | boat speed through the water | m/s |
| `L` | boat length overall (LOA) | m |
| `k` | wavenumber, `k = 2π/λ` | rad/m |
| `λ` (lambda) | wavelength | m |
| `λ_T` | **transverse** wake wavelength — the waves running across the track, `λ_T = 2πU²/g` in deep water | m |
| `λp` | peak wavelength of the **ambient** sea (not the wake); from `summary.md` | m |
| `ω` (omega) | angular frequency, `ω = 2πf` | rad/s |
| `θ` (theta) | in the Kelvin integral, the direction a given wave component travels, measured from the boat's track | rad |
| `A(θ)` | **free-wave spectrum** — the amplitude the hull radiates into direction `θ`. This is where hull shape enters | m·s (per unit `θ`) |
| `kd` | depth times wavenumber; the deep-water limit is `kd → ∞` | — |
| `Fr_L` | **length** Froude number, `U/√(gL)`. Sets whether the hull is displacing or planing | — |
| `Fr_h` | **depth** Froude number, `U/√(gd)`. Sets the wake regime — see §4 | — |
| `mss` | mean square slope of the surface; the BSDF's roughness comes from it as `α = √mss` | — |
| `N` | droplet number density in a spray plume | m⁻³ |
| `r` | droplet radius | m |
| `Q_ext` | extinction efficiency — how much light a droplet removes relative to its geometric cross-section | — |
| `σ_ext` | extinction coefficient of the spray medium; optical depth is `σ_ext` × path length | m⁻¹ |
| `mesh_dx` | water-mesh post spacing | m |
| `Hs` | significant wave height of the ambient sea, `4√m₀`; the sea state's headline number | m |
| `α` (alpha) | BSDF surface roughness, `α = √mss` | — |
| kn | knots, 1 kn = 0.5144 m/s. Used only where a human reads a boat speed | — |

Two easy confusions worth naming:

- **`λ_T` and `λp` are different things.** `λ_T` belongs to the boat and scales as
  `U²`; `λp` belongs to the wind sea and comes from the spectrum. §8 depends on
  keeping them apart.
- **`Fr_L` and `Fr_h` are different things.** `Fr_L` says whether the boat is
  planing. `Fr_h` says what shape the wake is. A boat can be planing
  (`Fr_L > 1`) while still subcritical in depth (`Fr_h < 1`), and vice versa.

### Terms

| Term | Meaning |
|---|---|
| **6-DOF** | six degrees of freedom: three translations (surge, sway, heave) and three rotations (roll, pitch, yaw) |
| **LOA** | length overall — the boat's full length, the `L` in `Fr_L` |
| **Draft** | how deep the hull sits below the waterline |
| **Wetted surface** | the part of the hull actually in the water; it changes with speed, which is why sinkage and trim matter |
| **Sinkage / trim** | how much a hull sinks and tilts when underway, relative to its at-rest attitude |
| **Slender / centreplane** | a hull long compared with its beam and draft; thin-ship theory collapses such a hull onto its vertical centre-plane and puts wave sources there |
| **Orbital velocity** | the circular motion of water *particles* under a passing wave, distinct from the wave's own travel speed. What a floating boat actually responds to |
| **Optical depth** | `σ_ext` × path length through a medium; dimensionless. About 3 reads as opaque |
| **Displacing / planing** | at low speed a hull pushes water aside; above roughly `Fr_L ≈ 1` it rides on hydrodynamic lift, with a very different wetted shape |
| **Kelvin wake** | the steady V-shaped wave pattern a body radiates by moving; 19.47° half-angle in deep water |
| **Transverse / divergent waves** | the two families inside the wake — transverse run across the track, divergent feather out toward the wedge edge |
| **Cusp line** | the wedge boundary, where the two families meet. The brightest feature of a wake |
| **Caustic** | a place where a ray theory predicts infinite amplitude because neighbouring rays converge to zero spacing. Cusp lines are caustics |
| **Stationary phase** | the approximation that only wave components whose phase varies slowly contribute; it gives the wake geometry and it is what fails at a caustic |
| **Airy function** | the correct amplitude shape near a caustic, replacing the infinity with a finite peak and a decaying oscillation |
| **Thin-ship (Michell) theory** | linearized ship-wave theory that represents a slender hull as sources on its centreplane, yielding `A(θ)` analytically |
| **Havelock source** | the elementary free-surface wave source those theories are built from |
| **Wigley hull** | a simple analytic hull form with published resistance curves, used as a validation reference |
| **Seakeeping: radiation / diffraction** | radiation = waves made by the hull *moving*; diffraction = incident waves *scattered* by a hull held still. Distinct problems, both distinct from the Kelvin wake |
| **Wave resistance** | drag from making waves. Its "hollows and humps" against speed come from bow/stern interference |
| **Transcritical / supercritical** | `Fr_h ≈ 1` / `Fr_h > 1`; see §4 |
| **Participating medium** | a volume that absorbs, scatters and emits along a ray — how a renderer represents fog, smoke or spray, as opposed to a surface |
| **Phase function** | the angular distribution of scattering within such a medium |
| **Panel method / BEM** | boundary-element method: solve for flow by discretizing the hull surface. What a proper hull-diffraction treatment needs |
| **Geometric-optics limit** | when droplets are much larger than the wavelength, so `Q_ext → 2` and no Mie computation is required |

---

## 1. It is four wave systems, not one

Linear theory decomposes the disturbance exactly:

```
h_total = h_ambient        the incident sea            -- built
        + h_kelvin         steady forward motion       -- this phase
        + h_radiation      hull heaving/pitching/rolling
        + h_diffraction    ambient waves scattered off the fixed hull
```

Plus two things that are not surface elevation at all:

```
turbulent wake             surface CHANNELS (mss, foam, temperature)
bow spray                  a VOLUMETRIC medium above the surface
```

Keeping these separate is the whole architecture. They have different physics,
different cost, and different representations, and conflating them is how a wake
implementation turns into a mess.

### Why superposition is legitimate

The model is already linear in elevation (`algorithms.md` §17, approximation 1),
so:

```
h_total   = h_ambient + h_wake
grad h    = grad h_ambient + grad h_wake     -- analytic normals still work
mss_total = mss_ambient + mss_wake           -- independent, so variances add
```

A wake therefore drops in without disturbing anything downstream. It does not
violate "the spectrum is the single source of truth" either: the wake has its own
defensible physics, it simply is not stochastic.

**Where it stops being defensible:** within about a hull length of the boat, and
for any breaking bow wave. Bow waves reach a large fraction of the draft, and
breaking is emphatically non-linear. State this as a bounded approximation rather
than pretend otherwise; the existing depth limiter will at least stop the sum
punching through the bed.

---

## 2. What surface the wake is computed against

**The flat mean water plane `z_w`** — not the instantaneous wavy surface.

This is not a shortcut, it is what makes the sum valid. Classical Michell/Havelock
theory linearizes the free-surface boundary condition about the undisturbed
plane. Our ambient sea is linearized about the same plane. Computing the wake
against the instantaneous surface would mix a non-linear formulation into a linear
one, and superposition would stop being justified.

---

## 3. Where the 6-DOF pose enters — three distinct places

A common mistake is to assume the pose only orients the boat model. It does three
separate things, with different consequences for what can be precomputed.

### 3a. Mean sinkage and trim — steady, changes the wake itself

Even at constant speed a hull squats and trims, changing the wetted shape, which
*is* the free-wave spectrum `A(θ)`. For a planing hull this is not a correction —
a Boghammar on plane has a wholly different running attitude and wetted surface
than at rest.

**Consequence:** the Kelvin lookup is keyed on `(speed, depth, running attitude)`,
not `(speed, depth)`. Still precomputable, because attitude is a function of speed.

### 3b. Oscillatory motion radiates its own waves

The seakeeping *radiation* problem: a hull heaving, pitching and rolling makes
waves independently of forward motion. A 10 m boat in a sea with `λp` of 8–14 m is
in the resonant band, so this is not negligible.

**Consequence:** cannot be precomputed in the boat frame — it depends on
instantaneous motion. Second-order for the picture next to the Kelvin wake and
the spray, so it is staged late.

### 3c. Spray generation — first-order, and the reason 6-DOF matters most

Spray is thrown by **relative** vertical velocity at the bow: hull motion *minus*
local wave orbital velocity. A boat slamming a wave throws far more spray than the
same boat at the same speed in calm water.

**Consequence:** the spray *rate* must be evaluated per frame from the pose
against our surface. Only the plume *shape* is precomputable, then scaled.

---

## 4. The finding that reorders everything: depth Froude number

The 19.47° Kelvin half-angle is a **deep-water** result. The governing parameter
is the depth Froude number `Fr_h = U/√(gd)`:

| `Fr_h` | Regime | Wake |
|---|---|---|
| < 0.9 | subcritical | 19.47° wedge, transverse + divergent |
| ≈ 1 | **transcritical** | wedge opens toward 90°; wave resistance peaks; largest wake |
| > 1.1 | **supercritical** | Mach cone at `arcsin(1/Fr_h)`; **transverse waves vanish** |

Computed for a 10 m boat:

| U (kn) | `Fr_L` | `λ_T` | `Fr_h` @ 18 m | Regime at 18 m |
|---|---|---|---|---|
| 9.7 | 0.50 | 16 m | 0.38 | subcritical |
| 19.4 | 1.01 | 64 m | 0.75 | subcritical |
| 25.3 | 1.31 | 108 m | **0.98** | **transcritical** |
| 35.0 | 1.82 | 208 m | **1.35** | supercritical, cone 47.6° |
| 38.9 | 2.02 | 256 m | 1.51 | supercritical, cone 41.6° |

Critical speed `√(gd)`: **8.6 kn in 2 m, 13.6 kn in 5 m, 25.8 kn in 18 m.**

**A fast small boat in the littoral is transcritical or supercritical most of the
time.** Implementing deep-water Kelvin first would produce a wake that is wrong
for the actual use case. Finite depth is therefore in from the start, not a later
refinement — and it is a genuine differentiator, since we already carry `depth`
everywhere.

---

## 5. The Kelvin field

Waves that keep station with the hull satisfy a stationary-phase condition. In
finite depth `k(θ)` solves

```
omega(k) = U k cos(theta),        the wave keeps station with the boat
omega^2  = g k tanh(k d)          the dispersion relation, finite depth
```

The first line says a wave only persists in the pattern if its own phase speed
along the track matches the boat's speed; the second is the standard relation
between frequency and wavenumber at depth `d`. Solving them together gives the
`k(θ)` that survives. It reduces to `k = g/(U cos θ)²` as `kd → ∞`, which is the
deep-water Kelvin result. The elevation is a 1-D
integral over `θ`:

```
zeta(x, y) = Re  int  A(theta) exp[ i k(theta) (x cos theta + y sin theta) ] dtheta
```

`A(θ)` is the hull's free-wave spectrum. Thin-ship (Michell) theory gives it
analytically for a simple hull form; that is where the boat geometry and running
attitude enter.

### Interference comes free

`A(θ)` for a real hull carries the ship's length scale, so the **bow and stern
wave systems interfere automatically** — that is exactly what produces the
hollows and humps in wave resistance. Do not simplify `A(θ)` to a point source
beyond the first milestone.

### Caustics at the cusp lines — the piece most likely to be got wrong

The wedge boundary is a **caustic**, where stationary phase predicts **infinite
amplitude**. The cusp line is also the brightest and most recognisable feature of
a wake, so it cannot simply be clamped.

The correct treatment is the standard uniform asymptotic one: near a caustic the
amplitude is an **Airy function**, not a cosine (Ursell 1960). This is a cheap
substitution, not a solver, and it is what makes a wake *look* like a wake.

### What is not included

Diffraction of ambient waves around the hull needs a boundary-element/panel
method. Genuinely second-order **unless the hull is large compared with `λp`** —
and at `λp` = 13.6 m against a 10 m boat, that condition is not comfortably met.
Measure it before dismissing it.

---

## 6. Turbulent wake — surface channels, not geometry

The churned region astern is not elevation; it is a change in surface *state*:

| Channel | Effect | Notes |
|---|---|---|
| `mss` | raised — the wake is rougher than the ambient sea | drives BSDF alpha |
| `foam` | seeded along the track, decaying | reuses the existing foam advect/decay |
| `temperature` | **new channel** | subsurface water brought up; often the strongest LWIR signature |

The existing foam model already advects, decays and seeds — the wake is another
seeding term, not new machinery.

---

## 7. Spray — a volume, not a channel

**Spray is above the surface, so no surface property can represent it.** `mss`,
`foam` and `wetness` are all things a BSDF does *at* a surface; spray is a medium
between the surface and the sensor. This is the one part of the phase that does
not fit the current pipeline.

Representation: a **heterogeneous participating medium** (Mitsuba supports this
directly) — a density grid, extinction, a phase function, and emission.

### LWIR makes this tractable

Droplets are tens to hundreds of microns; at `lambda` = 10 um that is the
geometric-optics limit, so `Q_ext ~ 2` and extinction is twice the projected area
per unit volume:

```
sigma_ext ~ 2 N pi r^2        extinction coefficient [1/m]
                              N = droplets per m^3, r = droplet radius [m]
                              the leading 2 is Q_ext in the geometric-optics limit
```

Optical depth along a ray of length `L_path` is then `σ_ext · L_path`, and a plume
is visually opaque once that exceeds about 3.

No Mie computation needed. Droplets sit near water temperature with emissivity
~1, so a plume reads as a **warm, semi-opaque volume against a cold sky** — often
a stronger LWIR signature than the wake itself.

### Generating the density field

1. Generation rate at the bow from relative vertical velocity (§3c) and bow
   geometry.
2. Ballistic droplets with drag, advected by the wind, with a size distribution.
3. Accumulate into a density grid **in boat coordinates**.

At steady speed and attitude the plume shape is stationary in that frame, so the
shape is precomputed and only the *rate* is evaluated per frame.

---

## 8. Cost, and what can be precomputed

The same structure that makes the foam spin-up and the Phase 5b ray solve
affordable: **heavy lifting up front, cheap frames.**

| Field | Frame | Precomputable? |
|---|---|---|
| Kelvin elevation | boat | **Yes**, keyed on (speed, depth, attitude) |
| Turbulent wake channels | boat | **Yes**, same key |
| Spray plume shape | boat | **Yes**, same key |
| Spray rate | — | **No** — per frame from the pose |
| Radiation waves | — | **No** — instantaneous motion |

Per frame, the precomputed parts cost a pose transform and a few interpolations.

**Manoeuvring** breaks the rigid-pattern assumption: a turning boat's wake is a
*history*, not a fixed shape. Standard treatment is to lay contributions along the
track and let each age and decay — more bookkeeping, not more physics.

### Mesh resolution

`λ_T` runs 16 m at 10 kn to 208 m at 35 kn — easy. But the **divergent waves and
the cusp region are much shorter**, and that is the visually important part. The
band limiting added in `algorithms.md` §15 will silently *remove* wake detail the
mesh cannot hold. That is correct behaviour, but it means **`mesh_dx` must be
sized for the wake, not for `λp`.**

---

## 9. Gate 11

| # | Check | Criterion |
|---|---|---|
| 11.1 | Deep-water wedge half-angle, `Fr_h < 0.5` | 19.47° ± 0.5° |
| 11.2 | Transverse wavelength vs `2πU²/g` | within 2% |
| 11.3 | Supercritical cone vs `arcsin(1/Fr_h)` | within 1° |
| 11.4 | Transverse waves absent for `Fr_h > 1.2` | energy < 5% of divergent |
| 11.5 | Cusp amplitude finite and Airy-shaped | no `inf`; peak within 20% of Ursell |
| 11.6 | Superposition | `h_total − h_ambient` equals the wake alone to 1e-12 |
| 11.7 | Wake vanishes at `U = 0` | identically zero |
| 11.8 | Energy: wave resistance vs a published Wigley-hull curve | within 20% |
| 11.9 | Spray rate is zero in calm water at zero relative velocity | exact |
| 11.10 | LWIR plume optical depth vs hand-computed `2Nπr²L` | within 5% |
| 11.11 | Reproducibility: same pose track twice | bitwise identical |
| 11.12 | Cost: precompute and per-frame, on the straits scene | recorded |

11.3 and 11.4 are the littoral-specific ones and the reason finite depth is not
deferred.

---

## 10. Staging

1. **Finite-depth Kelvin field, point-source `A(θ)`, straight track.** Gates
   11.1–11.4, 11.7. Gets the geometry right across all three regimes.
2. **Airy uniform asymptotics at the cusp.** Gate 11.5. This is what makes it look
   like a wake.
3. **Hull form factor `A(θ)`** from the boat's geometry and running attitude →
   bow/stern interference. Gate 11.8.
4. **Turbulent wake channels**, including the new `temperature` channel.
5. **Spray as a volumetric medium.** Gates 11.9–11.10.
6. **Track-history wake** for manoeuvring.
7. **Radiation waves** from oscillatory 6-DOF.
8. *(Only if measured to matter)* hull diffraction of ambient waves.

Steps 1–3 are analytic — no solver — and compose with the Phase 5b ray work
rather than conflicting with it.

---

## 11. References

- Kelvin, W. Thomson (1887). *On ship waves.* The original 19.47° wedge.
- Michell, J.H. (1898). *The wave resistance of a ship.* Thin-ship theory; the
  origin of `A(θ)`.
- Havelock, T.H. (1908, 1932). Free-surface source potentials; wave resistance.
- **Ursell, F. (1960).** *On Kelvin's ship-wave pattern.* J. Fluid Mech. 8,
  418–431. The uniform asymptotic (Airy) treatment of the cusp caustic — the
  reference for milestone 2 and Gate 11.5.
- Havelock (1908) / Lighthill (1978), *Waves in Fluids* ch. 3, for the
  finite-depth and transcritical behaviour in §4.
- Newman, J.N. (1977). *Marine Hydrodynamics.* Standard reference for the
  seakeeping radiation and diffraction problems in §1.

---

## 12. Integration risk: which sea is the boat responding to?

**Resolve this before writing code.** The motion model presumably takes a sea
state as input. If it has its own internal wave field rather than consuming ours,
the boat will pitch and heave to waves that are *not the ones being rendered* —
and at a 10 m boat in 8–14 m waves that mismatch is visible, not subtle.

The contract needs to be explicit and two-way:

```
pywave -> motion model   surface elevation, slope and orbital velocity
                         at the hull, per timestep
motion model -> pywave   6-DOF pose and velocity, per frame
```

Sampling elevation and slope at arbitrary points is already cheap
(`TileSet.sample`); orbital velocity would be a small addition. If the motion
model cannot accept an external sea, that is a scoping constraint worth knowing
now rather than discovering at integration.
