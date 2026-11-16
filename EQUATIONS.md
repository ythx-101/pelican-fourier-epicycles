# EQUATIONS.md: generator equations for "Pelican Cycling in 3D — Spatio-Temporal Fourier Epicycles (N = 1024)"

## How to read this

This file specifies the generator that Claude Fable 5.2 found by reverse-engineering the original video (fofr's "Pelican Cycling in 3D", made with Gemini 4 Argon) from the video alone.
Backticked names are keys of the solver's parameter file `params.json`, or arrays of the coefficient file and the glyph atlas. These files are described here (shapes, counts, meaning, how they were measured) but are NOT distributed, out of respect for the original author. Scalar appearance constants are listed in full.
`src/render.py` in this repository is DeepSeek V4 Pro's independent re-implementation, written from this file alone.
The document is the solver's own, with internal file names removed; "the solver's renderer" means its reference renderer. Section numbers are kept as in the original, so section 10.7 appears before section 10.6.
The text below is unchanged except for those removals; apparent ambiguities (for example the transpose convention in 6.7 versus 6.8) are left as written, because the reproduction was scored against this exact text.

Conventions used throughout:

- Output is 1280×720 (W×H), 60 fps, frames f = 0..3599, RGB float 0–255 until the final quantisation.
- Screen points are complex numbers z = x + i·y, x to the right, y down, pixel centres at integer coordinates.
- `name` in backticks is a key of `params.json`; `hud.name` is a key inside its `hud` object.
- G_σ(·) is `scipy.ndimage.gaussian_filter(·, σ)` with defaults (mode `reflect`, truncate 4.0). The constant 2.5066 is √(2π).
- Colours are RGB triples. "over" compositing means `img = img·(1−a) + a·colour`.

Keys that are present in `params.json` but have no effect with the current values are listed in section 8.

---

## 1. Overview: what is drawn when

| Frames | N | Pen parameter s_pen/2π | Curve shown | ψ | View |
|---|---|---|---|---|---|
| 0–288 | 3 | f/288 | traced part only | 0 | 2D |
| 288–576 | 10 | (f−288)/288 | traced part only | 0 | 2D |
| 576–1008 | 36 | (f−576)/432 | traced part only | 0 | 2D |
| 1008–1584 | 1024 | (f+δ(f)−1008)/576 | traced part only | 0 until 1296, then ease-in | 2D |
| 1584–2160 (ride) | 1024 | ((f−1584) mod 288)/288 | whole curve | 2.5°/frame | 2D |
| 2160–3600 (3D) | 1024 | ((f−2160) mod 576)/576 | whole curve | 2.5°/frame | turntable + pitch + perspective |

Ranges are half-open [f0, f1). Values come from `stages`, `ride_start` = 1584, `ride_period` = 288, `f3d` = 2160, `period3d` = 576.

Other timed elements:

| Element | Timing | Keys |
|---|---|---|
| Epicycle chain and pen dot | full strength until 3350, linear fade to 0 at 3370 | `chain_end` = 3370, `chain_fade` = 20 |
| Chain tail colour (main view) | cyan before 1584, gold from 1584 | `gold_start` |
| Inset window | fades in 1076→1114, out 2056→2094 | `hud.win_in`, `hud.win_out` |
| Window view | pen-follow 5× → crank view 3.1×, smoothstep over 1524.5→1608 | `hud.win_switch` |
| Window title bar text | "5× Stylus Tip Macro" before 1584, "3.1× Crankset & IK Feet" from 1584 | `ride_start` |
| Turntable ring | absent before 2180, linear fade-in 2180→2244 | `ring_in` |
| Depth shading | fades in 2172→2232 | `look_in` |
| Title (top) | fades in 3195→3315 | `hud.title_in` |
| HUD text layer | one static layer per segment (0, 288, 576, 1008, 1584, 2160 boundaries) | atlas `meta.modes` |

Per-frame draw order (`render`): background copy → turntable (3D only) → main view (chain, curve, pen) → lead line and inset window → output quantisation → atlas text layers.

---

## 2. The curve

```
r(s,ψ)   = Σ_{|k|≤N} C_k(ψ) e^{iks}
c_k(ψ)   = Σ_{p=−P..P} C[k,p] e^{ipψ}        screen-plane part, x + i·y (pixels)
Z_k(ψ)   = Σ_{p=−P..P} Z[k,p] e^{ipψ}
dep(s,ψ) = Re Σ_{|k|≤N} Z_k(ψ) e^{iks}       depth (pixels), positive towards the viewer at θ = 0
```

Stage curves (N = 3, 10, 36, frames 0–1008): the coefficients with |k| ≤ 36 are taken from `stage_c` (73 complex numbers in `params.json`,
rows k = −36..36 as [Re, Im]) instead of Σ_p C[k,p]; see section 10.3. The N = 1024 figure uses `C` only.

`coeffs_r3.npz` (file name from `coeffs`):

| Array | Shape / type | Meaning |
|---|---|---|
| `k` | int64[2049] | k values, contiguous −1024..1024; row index = k + 1024 |
| `C` | complex128[2049, 33] | column j ↔ p = j − 16 (p = 0 is column 16); P = (33−1)/2 = 16 (earlier fits: 17 columns, P = 8) |
| `Z` | complex128[2049, 33] | same layout, depth (columns |p| > 8 are zero) |

`Z` satisfies Z[−k,−p] = conj(Z[k,p]) to 2e-14, in an earlier fit. In R3 the depth corrections were added on columns p ≥ 0 only, so that symmetry no longer holds in the file; the code takes `.real` of the synthesis, which is the definition of depth. Non-zero columns of `Z`: p = −2..8.

Evaluation at a given ψ (`coef`):

```
e   = exp(i·ψ·[−P, …, P])        # P = 16
c   = C @ e          # complex[2049]
zc  = Z @ e          # complex[2049]
```

Sampling (`synth(c, N, M)`), M = `M` = 8192:

```
F = zeros(M, complex)
F[k mod M] = c_k   for |k| ≤ N        (all other bins 0)
samples = M · ifft(F)                 # numpy ifft has 1/M, so samples[i] = Σ c_k e^{2πi·k·i/M}
```

Sample i corresponds to s = 2π·i/M. No half-sample phase is applied at render time. (An earlier analysis suggested a half-bin offset; if it exists it is already inside the stored coefficients.)

Depth use:

- 2D frames (f < `f3d`): `persp2d` = 0, so the code sets `zc = None`; depth is zero everywhere and the projection is the identity. `Z` is not used at all.
- 3D frames: `dep = synth(zc, 1024, M).real`, fed to the projection (section 4) together with the screen-plane samples.

---

## 3. Time functions

### 3.1 Stage and pen parameter (`stage(f)` → N, sp, full)

`sp` is s_pen/2π in [0, 1).

```
if spen_warp present and 1008 ≤ f < 1584:
    f ← f + δ(f)                              # used only inside this function
for (f0, f1, N) in stages:                    # [[0,288,3],[288,576,10],[576,1008,36],[1008,1584,1024]]
    if f < f1: return N, (f − f0)/(f1 − f0), full=False
if f ≥ f3d:  return 1024, ((f − f3d) mod period3d)/period3d, full=True
else:        return 1024, ((f − ride_start) mod ride_period)/ride_period, full=True
```

`spen_warp` table δ(f):

- A list of [frame, δ] pairs. The current table has 470 entries, one for every integer frame 1114..1583, 466 of them non-zero.
- δ(f) = `np.interp(f, frames, deltas, left=0, right=0)`: linear interpolation between knots, zero outside [1114, 1583].
- The last entry is at frame 1583 with δ < 1, so f + δ stays < 1584 and the frame stays in the tracing stage.
- The numbers 1008 and 1584 in the warp condition are hard-coded in the solver's renderer, not read from `stages`.
- ψ(f) is not warped.

### 3.2 Crank phase ψ(f) (`psi`)

```
f ≥ ride_start:  ψ = psi_rate · (f − ride_start)                       degrees
f < ride_start:  ψ = psi_rate / (2·(ride_start − f0)) · max(f − f0, 0)²   degrees,  f0 = psi_ease_start
```

With `psi_rate` = 2.5, `psi_ease_start` = 1296, `ride_start` = 1584: ψ = 0 for f ≤ 1296, ψ = (2.5/576)·(f−1296)² degrees for 1296..1584 (reaching exactly 360° at 1584), then 2.5°/frame (period 144 frames). The result is converted to radians and is not reduced mod 360.

### 3.3 Turntable angle θ(f) and pitch(f)

Both are `scipy.interpolate.PchipInterpolator(knots_f, knots_deg)(f)`, converted to radians. They are evaluated only for f ≥ `f3d`; for 2D frames θ = pitch = 0.

- θ: `theta_knots_f` / `theta_knots_deg`, 52 knots. Frames 2160, 2184, …, 3336 (every 24 frames, 50 knots), then 3340 and 3600. The angle rises monotonically from 0° to about 392° (just over one turn) and then holds.
- pitch: `pitch_knots_f` / `pitch_knots_deg`, 24 knots, denser near 2160; pitch rises from 0° to about 17° and settles near 14° (knot values not published). All knots were measured from the HUD readout transitions.

### 3.4 Visible range of the curve

```
n_vis = M                 if full
n_vis = floor(sp · M)     otherwise
```

Samples 0..n_vis−1 are drawn, as a closed polyline if `full`, open otherwise. Nothing is drawn when n_vis ≤ 1 (the first frame of each stage).

`tail_to_pen` = 1 (R3): when not `full`, the pen-tip point (section 5, evaluated at the continuous `sp`) is appended to the polyline as its last vertex, with the brightness weight of sample n_vis−1. The drawn arc is then exactly s ∈ [0, s_pen]. Without this key the line stopped at the last grid sample before the pen, 2–4 px short of it during the N = 1024 stage.

---

## 4. Projection and depth shading

### 4.1 Projection (`project(z, dep, θ, φ)`)

```
x  = Re z − pivot_x          y  = Im z − pivot_y
x1 =  x·cosθ + dep·sinθ      d1 = −x·sinθ + dep·cosθ      # turntable: rotation about the vertical axis through the pivot
y1 =  y·cosφ + d1·sinφ       d2 = −y·sinφ + d1·cosφ       # pitch
κ  = 1 / (1 − d2/focal)                                    # perspective
screen = (pivot_x + x1·κ) + i·(pivot_y + y1·κ)
returns (screen, d2)
```

`pivot` = [639.4970, 292.4311], `focal` = 1408.5297.

- 2D frames: θ = φ = 0 and dep = 0, so screen = z (orthographic) and d2 = 0.
- 3D frames: θ = θ(f), φ = pitch(f), dep from `Z`. Perspective is applied from frame 2160 on, even while θ and pitch are still ≈ 0.

### 4.2 Depth → brightness (3D frames only; 2D frames use bright = 1)

Applied to the per-sample d2 of the projected curve, in this order:

```
bright = (d2 − min(d2)) / (max(d2) − min(d2) + 1e−9)                  # depth_norm = 1: per-frame normalisation, far = 0, near = 1
bright = interp(bright, linspace(0, 1, 6), bright_lut)                # piecewise linear, 6 knots
t      = clip((f − look_in[0]) / (look_in[1] − look_in[0]), 0, 1)     # look_in = [2172, 2232]
bright = 1 − t·(1 − bright)
```

`bright_lut` = [0.01, 0.04, 0.30, 0.55, 0.96, 1.36] at normalised depth 0, 0.2, 0.4, 0.6, 0.8, 1. The last value exceeds 1, so `bright` can exceed 1 and the colour interpolation of section 6.4 extrapolates past the near colour.

The min and max are taken over all M samples of the frame.

---

## 5. Epicycle chain

Origin O = `origin` = 639.5 + 291.125i.

### 5.1 2D frames

```
terms k with |k| ≤ N                     # 7, 21, 73, 2049 terms for N = 3, 10, 36, 1024
cc_k = c_k(ψ),  except cc_0 = c_0(ψ) − O
order = stable argsort of −|cc|          # descending magnitude
term_j = cc_j · exp(i·k_j·2π·sp)
cen_0 = O,  cen_j = O + Σ_{m<j} term_m   # n+1 points for n terms
rad_j = |cc_j|                           # circle j: centre cen_j, radius rad_j
pen   = cen_n = Σ_{|k|≤N} c_k e^{ik·2π·sp} = r(s_pen, ψ)
```

The code also carries a depth chain and projects `cen`, but in 2D both are trivial (depth 0, identity projection).

`trunc2d` = 1 names a "truncate in screen space" branch (c := FFT of the projected N = 1024 curve). That branch requires depth to be present, and `persp2d` = 0 removes depth before it is tested, so it does not execute. Under orthographic projection it would be a no-op anyway: the 2D series is truncated directly on c_k(ψ).

### 5.2 3D frames (`chain3d_screen` = 1)

The chain is rebuilt each frame from the projected curve `zs` (all M samples, N = 1024):

```
c2_k = (fft(zs) / M)[k mod M]   for k = −1024..1024
c2_0 ← c2_0 − O
order = stable argsort of −|c2|
cen_0 = O,  cen_j = O + Σ_{m<j} c2_m · exp(i·k_m·2π·sp)
rad_j = |c2_j|
pen   = cen_n
```

These are screen-space coefficients c_k(ψ; θ, φ); the chain is a flat 2D chain on the screen and is not projected again.

---

## 6. Rasterisation and appearance

### 6.1 Primitives

`splat(acc, z, w, clip)`: bilinear deposit of weighted points into a float32 density image.

```
keep points with 1 ≤ x < width−2 and 1 ≤ y < height−2, and (if clip) x0 ≤ x ≤ x1, y0 ≤ y ≤ y1
ix = floor(x), fx = x − ix (same for y)
acc[iy, ix] += w(1−fx)(1−fy);  acc[iy, ix+1] += w·fx(1−fy);  acc[iy+1, ix] += w(1−fx)fy;  acc[iy+1, ix+1] += w·fx·fy
```

`dense(z, w=None, step=0.6, closed=False)`: polyline → points with arc-length weights.

```
if closed: append the first point (and first weight)
for each segment a→b of length d:  n = max(ceil(d/step), 1)
    points a + (b−a)·(j+0.5)/n, j = 0..n−1
    weight of each = d/n · (linear interpolation of w along the segment, or 1 if w is None)
```

`stroke(img, den, σ, colour, gain=1, glow=None, sat=None)`: density → additive line.

```
v = min( G_σ(den) · σ·2.5066 · gain,  sat if given else params `sat` )        # core; a straight isolated line peaks at `gain`
img += v · colour
for each glow layer (σg, colour_g[, Gg]):
    g = G_σg(den) · σg·2.5066 · gain
    with Gg:    img += min(g·Gg, 1) · colour_g      # flat-top "wide stroke", half-width σg·sqrt(2·ln Gg), overlaps do not add
    without Gg: img += min(g, 1.5) · colour_g
```

A circle of centre c and radius r (r already multiplied by the view zoom) is sampled with `npt = clip(int(2πr/0.5), nmin, 4000)` points at angles 2π·j/npt, each with weight 2πr/npt (nmin = 10 in the main-view chain, 8 for the window hue circles).

### 6.2 Views

`draw_view(img, scene, f, view)` draws chain, curve and pen either into the full frame or into the window sub-image.

| | Main view (`view=None`) | Window view (`view=(cx, cy, zoom, rect)`) |
|---|---|---|
| Transform T(z) | z | (z − (cx + i·cy))·zoom + wc, wc = centre of rect |
| Clip | none | rect (inclusive) |
| Line core σ, colour | `line_sigma`, `line_color` | `win_line_sigma`, `win_line_color` |
| Flat-top glow (σ, colour, G) | `glow_sigma`, `glow_color`, `glow_gain` | `win_glow_sigma`, `win_glow_color`, `win_glow_gain` |
| Circle colour | `circle_color` | `win_circle_color` |
| Cap colour (2D) | `cap2d` | `win_cap` |

Chain strength: `chain_a = clip((chain_end − f)/chain_fade, 0, 1)`. Chain and pen are skipped when chain_a = 0.

### 6.3 Chain circles, arms and tail

Notation: `rad` = unzoomed radii (descending), `R = rad·zoom`, `cen = T(scene.cen)`, n = number of terms, `i0` = first index with rad < `gold_r` (n if none).

Main view:

```
big   = indices with rad > circle_min_r
large = big with rad ≥ gold_r;   small = big with rad < gold_r

# large circles + first arms: additive
den = Σ circles(large), weight 2πR/npt
den += dense(cen[0 .. min(n, arm_count)]) · arm_gain            # the first arm_count links, starting at O
stroke(img, den, circle_sigma, circle_color, gain = circle_alpha·chain_a, sat = circle_sat)

# tail (small circles + all links from circle i0 to the pen): density → alpha, "over"
den = Σ circles(small), weight 2πR/npt · tail_circle_gain
den += dense(cen[i0 .. n])
v   = G_{tail_sigma}(den) · gold_alpha · chain_a                # no σ·2.5066 factor here
a   = (1 − exp(−v)) · gold_sat
img = img·(1−a) + a·(gold_color if f ≥ gold_start else tail_color)
```

Window view (`win_hues` is set, so the window uses its own circle model; the tail is always gold-toned there):

```
selw = indices with R ≥ win_rmin                                 # zoomed radius
fw_j = clip(R_j / win_fade_r, 0, 1) ^ win_fade_g                 # small circles fade out
for gi, hue in enumerate(win_hues):                              # 3 hues, circle index j belongs to group j mod 3
    den = Σ circles(selw with j mod 3 == gi), weight 2πR/npt · fw_j
    if gi == 0: den += dense(cen[0 .. min(n, win_arm_count)]) · win_arm_gain
    a = 1 − exp(−G_{win_circle_sigma}(den) · win_circle_sigma·2.5066 · chain_a · win_circle_alpha)
    img = img·(1−a) + a·hue

# tail links, dim gold
den = dense(cen[i0 .. n], weights fw[i0..n−1] with the last value repeated)
a   = (1 − exp(−G_{win_tail_sigma}(den) · chain_a · win_gold_alpha)) · gold_sat
img = img·(1−a) + a·win_tail_color
```

After that the main-view code path still runs in the window with empty circle selections, which adds two more layers:

```
# first win_arm_count links again, additive
den = dense(cen[0 .. min(n, win_arm_count)]) · arm_gain
stroke(img, den, circle_sigma·win_circle_scale, win_circle_color,
       gain = circle_alpha·chain_a·win_circle_alpha, sat = win_circle_sat)

# tail links again, "over", bright gold
den = dense(cen[i0 .. n])
v = G_{tail_sigma·win_circle_scale}(den) · gold_alpha · chain_a · win_gold_alpha
a = (1 − exp(−v)) · gold_sat
img = img·(1−a) + a·gold_color
```

Whether drawing the window tail links twice (once in `win_tail_color`, once in `gold_color`) is intended is not clear from the code; it is what the code does. `win_circle_min_r` only affects the (emptied) main-style selection and has no visible effect.

### 6.4 Main curve

Drawn after the chain, only if n_vis > 1. `zs = T(scene.zs)`; "visible polyline" = `zs[0..n_vis−1]`, closed if `full`.

2D frames (main and window):

```
lay = 0
den = splat(dense(visible polyline, weights 1))
stroke(lay, den, lw, line colour, gain 1,
       glow = [(glow σ, glow colour, glow gain)]                # flat-top layer
            + [(mid_sigma, mid_color)]  in the main view only)  # faint wide additive halo, min(g, 1.5)
```

3D frames (`comp3d` = 1; the window no longer exists in 3D):

```
dT = splat(dense(visible polyline, weights 1))
dB = splat(dense(visible polyline, weights bright))
lay = 0
for (σ, c_near, c_far, top) in [
        (line_sigma·sigma3d, near_color,  far_color, sat),
        (glow_sigma3d,       near_glow,   far_glow,  1.5),
        (mid3d_sigma,        mid3d_color, 0,         1.5)]:
    gT = G_σ(dT)
    b  = G_σ(dB) / max(gT, 1e−6)                 # local mean nearness
    v  = min(gT · σ·2.5066, top)                 # coverage; overlapping lines do not add beyond `top`
    lay += v · (b·c_near + (1−b)·c_far)
```

Soft cap and compositing (2D main: c = `cap2d`; window: c = `win_cap`; 3D: c = `cap3d`; q = `cap_q`):

```
lay  = lay / (1 + (lay/c)^q)^(1/q)                         # per channel
cov  = clip( max over channels(lay/c) · line_over, 0, 1 )
base = background image (main view)  or  window background colour (window view)
img  = img − (img − base)·cov                              # the line hides what is under it (chain, turntable)
img += lay
```

In the main view `base` is the precomputed background (gradient + stars), so in 3D the turntable ring is also pushed back towards the background under lines.

### 6.5 Pen dot and halo

Drawn last in each view, only if chain_a > 0. One unit-weight point is splatted at T(pen); `ps` = 1 (main) or `win_pen_scale` (window).

```
σ = pen_sigma·ps
a = min( G_σ(den) · 2πσ² · pen_gain, 1 ) · chain_a
img = img·(1−a) + a·pen_color
σh = pen_halo_sigma·ps
img += G_σh(den) · 2πσh² · pen_halo_color · chain_a
```

### 6.6 Turntable (3D frames, drawn before the main view)

R = `ring_r`, yg = `ring_y`, x0 = pivot_x, θ and φ from section 3.3. All points go through `project` of section 4.1.

```
fa = clip((f − ring_in[0]) / (ring_in[1] − ring_in[0]), 0, 1);  nothing is drawn if fa ≤ 0

# ring: 720 points, a = 2πj/720, not rotated
ring = project(x0 + R·cos a + i·yg, depth R·sin a, 0, φ)            → dense(closed), weight 1
# dashed diameter: 400 points u ∈ [−1, 1], rotated by θ
on   = 1 if floor((u+1)·ring_dashes) is even else 0
diam = project(x0 + R·u + i·yg, depth 0, θ, φ)                      → dense with weights on·ring_dash_gain
stroke(img, den_ring + den_diam, ring_sigma, ring_color, gain = ring_alpha·fa)      # sat = `sat`

# 4 ticks at q·90°, radial, rotated by θ: rr = R·linspace(0.95, 1.05, 20)
tick_q = project(x0 + rr·cos(qπ/2) + i·yg, depth rr·sin(qπ/2), θ, φ)   → dense
stroke(img, den_ticks, circle_sigma·1.5, gold_color, gain = tick_gain·fa)

# marker ball on the ring at ring angle am = ring_mark_phase − 2θ (not rotated by θ again)
m = project(x0 + R·cos am + i·yg, depth R·sin am, 0, φ)
in the 25×25 patch starting at (int(mx)−12, int(my)−12), r = distance to m (skipped if the patch leaves the frame):
    a = clip((mark_disc_r − r)/mark_disc_soft + 0.5, 0, 1) · fa
    patch = patch·(1−a) + a·mark_disc_color
    patch += exp(−r²/(2·mark_sigma²)) · mark_gain · fa · mark_color
```

The dash weights are linearly interpolated between the 400 samples by `dense`, so dash ends are soft.

### 6.7 Background

`bg_yuv` is present, so the background is defined in limited-range BT.601 YUV and the polynomial `bg` key is overwritten (it is computed and discarded).

```
basis(n, t): the n cardinal cubic splines on knots linspace(0, 1, n)
             = scipy CubicSpline(linspace(0,1,n), eye(n))(t), default not-a-knot ends
Bx = basis(NX, |2(x+0.5)/W − 1|)     x = 0..W−1      # left/right symmetric
By = basis(NY, (y+0.5)/H)            y = 0..H−1
S_c(y, x) = Σ_{a<NY} Σ_{b<NX} By[y,a] · bg_yuv[c][b][a] · Bx[x,b]          # c ∈ Y, U, V; node array is [NX][NY] = [8][14]
yuv = (S_Y − 16, S_U − 128, S_V − 128)
bg  = yuv @ inverse(YUV) − codec_bias
```

with the forward matrix (RGB 0–255 → Y−16, U−128, V−128):

```
YUV = [[ 65.481, 128.553,  24.966],
       [−37.797, −74.203, 112.0  ],
       [112.0,   −93.786, −18.214]] / 255
```

Stars (`stars`, 66 entries [x, y, aR, aG, aB]; `star_sigma` = 0.8): each adds a small Gaussian on the 5×5 patch starting at (int(x)−2, int(y)−2):

```
bg[patch] += exp(−((px − x)² + (py − y)²) / (2·star_sigma²)) · (aR, aG, aB)
```

Seven coordinates occur twice in the list (59 distinct positions); every entry is added, so those stars are drawn twice.

### 6.8 Output quantisation

After everything except the atlas text has been drawn:

```
img += codec_bias                                  # per channel; cancels the subtraction in 6.7 for pure background
if snap_y:  yuv = img @ YUVᵀ;  yuv[0] = round(yuv[0]);  img = yuv @ inverse(YUV)ᵀ     # luma snapped to integer levels
out = uint8( clip(img + 0.5, 0, 255) )             # truncation after +0.5
out = apply_atlas(out)                             # section 7.3, integer addition in 8 bit
```

### 6.9 Main appearance parameters (current values, float noise rounded)

| Key | Value | Role |
|---|---|---|
| `sat` | 0.85 | core saturation of `stroke`; top of the 3D core layer |
| `line_sigma` / `line_color` | 0.602 / [224.6, 237.6, 227.8] | 2D main line core |
| `glow_sigma` / `glow_color` / `glow_gain` | 1.035 / [1.16, 14.285, 12.24] / 9.335 | 2D main flat-top layer |
| `mid_sigma` / `mid_color` | 3.668 / [0.045, 5.34, 7.965] | 2D main wide halo |
| `cap2d` / `win_cap` / `cap3d` / `cap_q` | [210, 226, 235.5] / [199, 211, 238] / [210, 170, 175] / 8 | soft cap |
| `line_over` | 1.0 | line-over-chain coverage factor |
| `win_line_sigma` / `win_line_color` | 0.72 / [206, 222, 208] | window line core |
| `win_glow_sigma` / `win_glow_color` / `win_glow_gain` | 1.23 / [7.2, 21.6, 22.1] / 10.5 | window flat-top layer |
| `sigma3d` | 0.974 | 3D core σ = `line_sigma`·`sigma3d` |
| `near_color` / `far_color` | [129.5, 140, 124.5] / [45, 49, 47.5] | 3D core colours |
| `glow_sigma3d` / `near_glow` / `far_glow` | 2.406 / [−1.25, 19.75, 26.95] / [−1.87, 10.875, 18.9] | 3D glow layer |
| `mid3d_sigma` / `mid3d_color` | 0.615 / [23.75, 28.5, 31.5] | 3D third layer (near only) |
| `bright_lut` | [0.01, 0.04, 0.30, 0.55, 0.96, 1.36] | depth → brightness |
| `circle_min_r` / `gold_r` | 1.2 / 8.0 | circle drawn if rad > 1.2; tail if rad < 8 |
| `circle_sigma` / `circle_color` / `circle_alpha` / `circle_sat` | 0.425 / [77.5, 96, 95] / 0.98 / 1.1 | large circles and first arms |
| `arm_count` / `arm_gain` | 12 / 1.475 | first links of the chain |
| `gold_color` / `tail_color` | [248, 205, 120] / [105, 200, 235] | tail colour from / before `gold_start` |
| `gold_alpha` / `gold_sat` / `tail_sigma` / `tail_circle_gain` | 0.465 / 0.8 / 0.36 / 1.07 | tail density → alpha |
| `win_hues` | [118,114,134], [128,126,100], [138,118,112] | window circle hues |
| `win_circle_sigma` / `win_circle_alpha` / `win_rmin` | 0.825 / 0.7 / 1.0 | window circles |
| `win_fade_r` / `win_fade_g` | 40 / 1.0 | window small-circle fade |
| `win_arm_count` / `win_arm_gain` | 80 / 0.15 | window arms |
| `win_tail_sigma` / `win_tail_color` / `win_gold_alpha` | 0.8 / [150, 125, 85] / 0.09 | window tail |
| `win_circle_color` / `win_circle_scale` / `win_circle_sat` | [33, 30, 7] / 0.7 / 2.0 | window additive arm layer |
| `pen_sigma` / `pen_gain` / `pen_color` | 3.1 / 1.15 / [178, 224, 244] | pen dot |
| `pen_halo_sigma` / `pen_halo_color` / `win_pen_scale` | 4.5 / [6, 35, 40] / 1.375 | pen halo; window scale |
| `ring_r` / `ring_y` / `ring_sigma` / `ring_color` / `ring_alpha` | 208.5 / 503.1 / 0.65 / [63, 78, 82] / 0.8 | turntable ring |
| `ring_dashes` / `ring_dash_gain` / `tick_gain` | 12 / 0.075 / 0.37 | diameter dashes, ticks |
| `ring_mark_phase` | 90 | marker ring angle = 90° − 2θ |
| `mark_disc_r` / `mark_disc_soft` / `mark_disc_color` | 5.8 / 2.0 / [60, 68, 63] | marker disc |
| `mark_sigma` / `mark_gain` / `mark_color` | 2.35 / 1.075 / [150, 135, 125] | marker core |
| `codec_bias` / `snap_y` | [1.07, 1.25, 0.9] / 1 | output quantisation |

---

## 7. Inset window, lead line, HUD and title (the HUD module of the solver's renderer)

### 7.1 Window

Rectangle `hud.win_rect` = [x0, y0, x1, y1] = [963, 323, 1223, 533], inclusive, so the sub-image is 261×211. Title bar height `hud.win_title_h` = 25.

```
wa = clip((f − 1076)/38, 0, 1) · clip((2094 − f)/38, 0, 1)         # hud.win_in, hud.win_out; skip the window if wa = 0
u0 = clip((f − 1524.5)/83.5, 0, 1);   u = u0²(3 − 2u0)             # hud.win_switch, smoothstep
winbg = win_bg·(1−u) + win_bg_ride·u                               # [1.0, 8.8, 27.1] → [−1.0, 9.8, 31.7]
sub   = winbg everywhere
```

View centre (in main-view coordinates) and zoom:

```
cx   = (Re pen + pen_view_off[0])·(1−u) + crank_view[0]·u
cy   = (Im pen + pen_view_off[1])·(1−u) + crank_view[1]·u
zoom = pen_zoom·(1−u) + crank_view[2]·u                            # linear blend 5 → 3.1
```

If `hud.pen_view_tab` is present (R3, section 10.7), `pen_view_off` is replaced in these two lines by `pen_view_off + τ(f)`, where τ(f) = (τx, τy) is the
table [[frame, τx, τy], …] interpolated linearly in f (`np.interp`, zero outside the table's frame range). Current table: 436 integer frames in 1114..1571,
|τ| median 0.23 px, 90th percentile about 0.9 px, in main-view pixels.

`hud.pen_view_off` = [0.1, 0.78], `hud.crank_view` = [624.82, 420.62, 3.1], `hud.pen_zoom` = 5.0. `hud.win_zoom_blend` is absent, so the default (linear blend) applies; an inline comment in the source that speaks of an instant zoom switch at the ride start does not describe the current behaviour.

The scene is drawn into `sub` with `draw_view(sub, scene, f, (cx, cy, zoom, rect))`, where rect = (0, 25, 260, 210) in sub-image coordinates; so wc = (130, 117.5) and content is clipped to the area below the title bar.

Then, in this order:

```
sub[rows 0..25] = win_title_bg                                              # [2.5, 5.0, 19.5]
sub[row 25]     = sub[row 25]·(1 − win_sep_alpha) + win_border·win_sep_alpha   # separator, alpha 0.18

# rounded border: signed distance d to a rounded rectangle (negative inside), r = win_radius = 7
dx = |qx − 130| − (130 − r);   dy = |qy − 105| − (105 − r)                  # qx = 0..260, qy = 0..210
d  = hypot(max(dx,0), max(dy,0)) + min(max(dx,dy), 0) − r
bl = clip(win_border_w − |d + win_border_off|, 0, 1)                        # 0.975, 0.075
sub = sub·(1−bl) + win_border·bl                                            # win_border = [244, 224, 170]
wa_px = wa · clip(0.5 − d, 0, 1)                                            # per-pixel alpha; transparent outside the rounded rect

# title bar from the atlas (28 rows × 261 columns, replaces rows 0..27 of sub including their border pixels)
spr = atlas wt_31 if f ≥ ride_start else atlas wt_5
if f ≥ ride_start: for digit i of '%03d' % psiread(f):
    if atlas has wdig_{i}_{digit}: spr[:, wslots[i][0]:wslots[i][1]] = that sprite
sub[rows 0..27] = spr
```

Finally, after the lead line (7.2): `img[y0..y1, x0..x1] = img·(1 − wa_px) + sub·wa_px`.

The title bar text switches at `ride_start` (1584), while the view blend runs from 1524.5 to 1608 (u ≈ 0.80 at frame 1584).

### 7.2 Lead line

A grey dashed line in the main image from the view centre to the middle of the window's left edge, drawn before the window is composited:

```
L0 = cx + i·cy;   L1 = x0 + i·lead_y = 963 + 437i
nL = int(|L1 − L0|);   t_j = j + 0.5, j = 0..nL−1
on_j = ((nL − t_j) mod lead_period) < lead_on                    # 7.0, 4.0: dashes anchored at the window end
splat unit-weight points L0 + (L1 − L0)·t_j/nL for on_j
stroke(img, den, 0.45, lead_color, gain = max(wa_px))            # lead_color = [44, 45, 43], sat = `sat`
```

### 7.3 HUD and title text: glyph atlas

With `atlas` set (current state), all HUD rows, the title and the window title bar are pixel layers measured from the original video. This is typesetting, not an equation. Only the numeric readouts are driven by the model.

`atlas.npz` contents:

| Key | Shape | Use |
|---|---|---|
| `meta` | JSON string | `rect` = [Y0, Y1, X0, X1] = [544, 704, 96, 1184]; `modes` = [name, f_start, f_end) list: s3 0–288, s10 288–576, s36 576–1008, draw 1008–1584, ride 1584–2160, 3d 2160–3600; `slots`; `band2` = [595, 631]; `title` = [0, 60, 196, 1084]; `wslots` |
| `hud_<mode>` | int16[160, 1088, 3] | signed difference layer for the HUD rectangle, one per mode |
| `dig_<mode>_<i>_<d>` | int16 sprites | digit d at readout slot i (modes ride, 3d); slot rectangle `slots["<mode>_<i>"]` = [xa, xb, ya, yb] in frame coordinates |
| `p2_<n>` | int16 rows | HUD row 2 band (frame rows 595..630) for pitch readout n = 0..17 (3D only) |
| `title` | int16[60, 888, 3] | title difference layer |
| `wt_5`, `wt_31` | float32[28, 261, 3] | window title bar colours (absolute, not differences) |
| `wdig_<i>_<d>` | float32 sprites | ψ digits in the 3.1× window title bar, columns `wslots[i]` |

`apply_atlas(out, scene, f)`, on the 8-bit frame:

```
name = mode whose [a, b) contains f
d = copy of hud_<name>
if name is ride or 3d:
    digits = ψ readout (3 digits)                       for ride
           = ψ readout (3 digits) + θ readout (3 digits) for 3d
    if name is 3d and atlas has p2_<pitch readout>:  d[rows 595..630 of the frame] = that band
    for slot i, digit ch:  if atlas has dig_<name>_<i>_<ch>:  d[slot rectangle] = that sprite
out[544:704, 96:1184] += d                              # int16 arithmetic
ta = clip((f − 3195)/120, 0, 1)                         # hud.title_in
if ta > 0:  out[0:60, 196:1084] += round(title · ta)
out = clip(out, 0, 255) as uint8
```

Order matters: the pitch band is pasted first, then the digit sprites. A digit whose sprite is missing keeps whatever the static layer shows (for example the last digit slot only has sprites for 0, 2, 3, 5, 7, 8).

Readouts:

```
ψ readout (f ≥ 1584, HUD and window title):
    psiread(f) = round( degrees(2π·((f/60 − 26.4)/2.4)) mod 360 ) mod 360
    evaluated in double precision with Python's round() (half-to-even on the float value);
    it is a function of the frame number, not of the ψ value used for the curve
θ readout (3D):      int(floor(degrees(θ(f)) + 0.5 + 1e−6)) mod 360, printed as %03d
pitch readout (3D):  int(floor(degrees(pitch(f)) + 0.5))
```

The digit list is produced by a helper of the solver's renderer, which takes the first HUD row built by `hud.rows` and collects the digits of every run matching `\d\d\d°`, in order. N, the error percentage and the circle count are not computed at render time with the atlas; they are part of the static layers.

### 7.4 Font fallback (not used in the current state)

If the `atlas` key is removed, the HUD module of the solver's renderer renders the same texts from fonts (Latin Modern for text, Nimbus Roman for ψ, Latin Modern Math fallback) with its own scanline rasteriser. Row strings, sizes, positions, kerning and spacing come from the `hud` object (`rows`, `adj`, `eq1`, `eq2`, `sep_em*`, `gap3*`, `kern`, `text_kernel`, `sup`, `sub`, `title_*`, `win_title_*`). This path is typesetting and is not specified further here; it is not needed to reproduce the video. The texts it would print:

- Row 1: `N = <N>`, plus `· ψ = ddd°` from 1584, plus `· θ = ddd°` in 3D.
- Row 2 (tracing): `error = 26.12% / 16.55% / 10.57% / 0.33%` and `7 / 21 / 73 / 2,049 spinning circles` for N = 3 / 10 / 36 / 1024.
- Row 2 (ride): `2,049 Spatio-Temporal modes · cadence = 100 BPM · error = 0.33%`.
- Row 2 (3D): `2,049 3D Spatio-Temporal modes · pitch = <n>° · error = 0.33%`.
- Row 3: the formulas z(s, ψ) = Σ c_k(ψ) e^{iks} etc.

---

## 8. Measured data versus closed form

Measured from the original video (data, not formulas):

- `C`, `Z` coefficient arrays in `coeffs_r3.npz` (the curve and its ψ dependence).
- θ knots (52) and pitch knots (24), from HUD readout transitions.
- `spen_warp` table (470 entries, per-frame pen timing 1114–1583).
- `hud.pen_view_tab` (436 entries, per-frame centre offset of the 5× window, tracing stage; no law found, section 10.7).
- `stage_c` (73 complex numbers: the coefficients c_k, |k| ≤ 36, used by the stage curves N = 3, 10, 36; section 10.3).
- `atlas.npz` (HUD, title and window title-bar pixels, digit sprites).
- `stars` (66 entries) and the `bg_yuv` node grids (3 × 8 × 14).
- All appearance constants in section 6.9, window geometry and colours in `hud`, `codec_bias`, `origin`, `pivot`, `focal`, ring geometry.

Closed form:

- The series r(s, ψ) and its truncation to N; sampling by inverse FFT.
- Stage schedule and s_pen(f) apart from δ(f); ψ(f) (quadratic ease-in, then uniform).
- Projection (rotation, pitch, perspective) and per-frame depth normalisation.
- Epicycle chain construction (ordering, partial sums, per-frame screen FFT in 3D).
- Rasterisation model (splat, Gaussian layers, soft cap, compositing rules), window view blend, lead line, fades.
- Readout rounding rules.

Keys present in `params.json` that have no effect with the current values:

| Key | Why |
|---|---|
| `bg` | computed, then overwritten because `bg_yuv` is present |
| `stars_old` | never read |
| `depth_b0`, `depth_range`, `depth_gamma` | the fixed-range brightness is overwritten because `depth_norm` = 1 |
| `trunc2d` | its branch needs depth, which `persp2d` = 0 removes (section 5.1) |
| `win_circle_min_r` | the selection it controls is emptied when `win_hues` is set |
| `hud` text keys (`rows`, `size*`, `base*`, `alpha*`, `adj`, `emb`, `font*`, `title_size`, `title_x`, `win_title_size`, …) and `hud.white/cyan/gold/dim/red` | only used by the font fallback (7.4) |

The 3D branch of the layered `stroke` path in `draw_view` (non-`comp3d`) is likewise not executed.

---

## 9. How to reproduce

Files needed (described in this document, not distributed):

| File | Role |
|---|---|
| `params.json` | all parameters (keys named throughout this document) |
| `coeffs_r3.npz` | `k`, `C`, `Z` (section 2) |
| `atlas.npz` | text layers, digit sprites (section 7.3) |

`params.json` refers to `coeffs_r3.npz` and `atlas.npz` by name; they are loaded from the directory of the params file.

`src/render.py` in this repository is DeepSeek V4 Pro's re-implementation of this document. With the three files above it renders single frames or a range:

```
python3 src/render.py --params path/to/params.json --frames A B --out x.mp4     # B = A+1 with --out x.png gives a single PNG
```

Requirements: Python 3 with numpy and scipy, and `ffmpeg` for video output.

Encoding: the solver's submitted video used crf 10 on frames 0–1008 and crf 18 elsewhere. The lower crf on the stage frames raises their score by about 0.01 (thin sparse lines suffer most from compression) and changes nothing or lowers the score slightly elsewhere. With crf 18 everywhere the score is 0.8102 instead of 0.8132 (self-reported).

---

## 10. Measurement and fitting methods (round R3, 2026-11-16)

The equation set of sections 2–7 is unchanged by the last refinement round. It changed one drawing rule and re-measured data (coefficients and the pen timing table).

### 10.1 Drawing rule: the traced line ends at the pen (`tail_to_pen`)
See section 3.4. Found by looking at the 5× window: the original's line runs into the pen dot, ours stopped 2–4 px short of it.

### 10.2 Five route corrections of the static curve
Reverse check on frame 1584 ("lit pixels of the original with no line of ours within 1.5 px"): 1.52 % → 0.99 % of the original's line pixels.
For each case the ψ-independent part of the samples with t ∈ [t_a, t_b] (t = frame at which the pen passes the sample) is replaced by a polyline
read off the original's ridge, the pen slowing near the tip (dt/ds ∝ 1 + 6·exp(−(s−s_tip)²/(2·6²))), then transformed back to |k| ≤ 1024:

| Where | t range | What was wrong |
|---|---|---|
| rear fender tip (792, 322) | 1393.3–1395.8 | U-turn cut 17 px short |
| top-tube end (722, 298) | 1077.5–1079.0 | U-turn cut short |
| wing hatch stroke (583–597, 191) | 1185.1–1186.0 | a 19 px out-and-back stroke was missing (model pen just dwelt) |
| hatch tip (508, 264) | 1331.5–1332.5 | tip cut short |
| small V at (419, 372) | 1036.6–1037.6 | one branch missing |

### 10.3 The stage curves have their own low-order coefficients (`stage_c`)
Structure finding of R3: the stage curves (N = 3, 10, 36) are NOT the low-order part of the N = 1024 figure.

- The 73 numbers c_k, |k| ≤ 36, were measured on the stage frames (the ridge measurement of 10.4 on 24 frames of each stage, last 65 % of
  the stage, samples at least 120 behind the pen; one linear least-squares problem for all 73 complex corrections, d_jf = Re(conj(n_j)·Σ_{|k|≤N} Δc_k e^{iks_j}),
  two passes; ridge residual rms 0.055 / 0.095 / 0.138 px → 0.032 / 0.049 / 0.074 px for N = 3 / 10 / 36).
- The same correction applied to the N = 1024 figure makes it worse: tracing frames at rest (ψ = 0) lose 0.008–0.024, ride frames lose 0.006 after refitting.
  At the ride optimum the residual asks for a static low-order change of 0.054 px, the stage frames for 0.42 px, and the two directions are unrelated (cosine 0.08).
  A perspective explanation ("truncate x, y, depth in world space, then project") predicts 0.29 px with cosine ±0.08 to the measured difference: rejected.
- Model: `params.json` key `stage_c` = [[Re c_k, Im c_k] for k = −36..36]. In `scene()`, when N < 1024 the coefficients with |k| ≤ 36 are replaced by
  `stage_c` before the truncated synthesis (and before the epicycle chain is built). For N = 1024, `C` alone is used and its low-order row sums are free
  in the fits of 10.4. Why the original's two sets differ by 0.4 px is not known.
- History: the first official score of R3 lost 0.02–0.03 on the stage frames because the route corrections 10.2 had moved the low-order part; an
  intermediate version pinned the row sums Σ_p C[k,p]; `stage_c` replaces that rule.

### 10.4 Ridge-snapping fit of all coefficients C[k,p] and Z[k,p]
Reason: the score is very sensitive to sub-pixel line position (a 0.12 px offset of a σ = 0.6 px line costs about 0.15 of its J), and the earlier
curve sat a median 0.13 px (ride) and 0.23 px (3D) off the original's line ridge.

Measurement, for curve sample j (M = 8192 samples) in frame f, with screen position z_jf and unit normal n_jf from the model:

```
off(I) = Σ_o o·w(o) / Σ_o w(o),   o = −2 … 2 px step 0.25,   w(o) = max(I(z_jf + o·n_jf) − 60, 0)²      (I = gray image)
d_jf   = off(original frame f) − off(our rendered frame f)            used if both Σw > 2·10⁴ and |d| < 1.2 px
```

Model (Gauss–Newton step), per sample j, with the frame's ψ_f, θ_f, pitch_f:

```
d_jf = Re( conj(n_jf) · ( J^x_jf·Δx_j(ψ_f) + J^y_jf·Δy_j(ψ_f) + J^d_jf·Δdep_j(ψ_f) ) )
Δx + iΔy = Σ_p A_jp e^{ipψ},      Δdep = Re Σ_{p≥0} W_jp e^{ipψ}
J^x, J^y, J^d = derivative of the projection (section 4.1) with respect to x, y, depth; in 2D frames J^x = 1, J^y = i, J^d = 0
```

- Frames: 192 ride frames (1584 + 3i; ψ covers four pedal turns, θ = 0) pin the 2D part; 240 3D frames (2160 + 6i) add depth. Fitting the 2D
  part from 3D frames alone is wrong (ψ and θ are tied along the one camera path; it lowered the ride score by 0.016).
- In ride frames the 3.1× crank window (section 7.1) is a second view of the same samples: z_w = (z − crank centre)·3.1 + window centre;
  its offsets, divided by 3.1, enter as extra rows with weight 2.
- Harmonics allowed per sample: |p| ≤ P_free (2, later 4) everywhere; 5 ≤ p ≤ 8 only where that harmonic already has amplitude > 0.1 px.
- Last 6 iterations (P_max = 16): harmonics 9 ≤ |p| ≤ 16 of the 2D part are also free, on samples whose summed harmonic amplitude over p = 1..8 exceeds 1 px
  (97 % of the samples); depth stays at p ≤ 8 because the 3D frames sample ψ only every 15°. Reason: a per-sample harmonic fit of the ride residual leaves
  0.107 px rms with P = 8, 0.082 with P = 16 and 0.068 with P = 30; the single-frame measurement noise is 0.067 px.
- Damped least squares per sample (ridge 4 on 2D terms, 0.3 on depth terms), step clipped to 0.3 px (2D) / 1.5 px (depth), Gaussian smoothing
  σ = 0.7 samples along s, gain 0.8, result transformed back to |k| ≤ 1024.
- Final 6 iterations (the final fit) ran from an earlier fit's coefficients with P_max = 16, P_free = 4 (low-order row sums free, see 10.3): ridge residual
  median 0.094 px (ride), 0.164 px (3D).
- All test frames used to accept or reject a variant lie off both fit grids.

### 10.5 Pen timing table δ(f) (`spen_warp`)
Re-derived twice: (a) from an earlier fit's measured pen positions with adoption threshold 0.3 px instead of 1 px;
(b) from the registration of the original's 5× window content against ours (step 1 of the registration procedure: time only).
Step 2 of the registration procedure (bending the curve through the registered pen positions) was rejected: window J +0.034 but main-image J −0.007 and ride J −0.019.
(c) Final pass against the final curve, together with the window-centre table: section 10.7.

### 10.7 Centre of the 5× window in the tracing stage (`hud.pen_view_tab`)
Measurement: for every frame 1114..1583, the shift s(f) (window pixels, sub-pixel) that maximises the normalised cross-correlation between the original's
window content and ours (both Gaussian σ = 1 minus Gaussian σ = 8; search ±14 px). With u the pen→crank blend of section 7.1, the original's view centre
is at pen + e, e = −s / (5·(1−u)) main-view pixels. Frames with NCC < 0.3, a shift at the search border, or u > 0.6 are left unchanged.

```
v   = d pen / d f                                   # model pen velocity, px/frame
Δδ  = clip( Re(e·conj(v)) / |v|², −0.5, 0.5 )       # tangential part → pen timing table: δ(f) += Δδ
τ(f) += e − v·Δδ                                    # remainder → view-centre table (section 7.1)
```

Two passes (the second measured a remainder of 0.026 px median, i.e. converged). Before: median |e| 0.52 px (2.6 window px), 108 of 470 frames above 1.2 px.
The timing part is small (|Δδ| median 0.015 frame); the remainder is mostly normal to the path (median 0.24 px, 90th percentile 0.87 px).
Effect on 36 tracing frames (1116 + 13i): J 0.692 → 0.715, window J 0.425 → 0.490, main-view J unchanged.

τ(f) is measured data without a law. Tested and rejected as explanations of e(f): curvature, pen speed; a pixel grid of 1, 0.5, 0.25 or 0.2 px
for the view centre (phasor test); the tip of a truncated chain, |k| ≤ N′ or the N′ largest circles (|r| < 0.11); a camera that filters
the pen path in time, lags −4..+8 frames (held-out residual power 0.98–1.02 of the total). The 1× curve itself is right (10.5 b, step 2).

### 10.6 Not solved, and what bounds the score
- Ceiling of a perfect model (our own frames re-encoded with the original's x264 settings, scored against our crf 18 video):
  stage 0.943, tracing 0.921, ride 0.888, 3D 0.904. Final values are in BENCHMARK.md; the gap is model error, not codec noise.
- Main-view line appearance (σ, colours, glow, cap, saturation) was re-swept after the geometry fit on ride and stage frames:
  flat, no change above +0.001. The remaining main-view loss is positional.
- Ride ridge residual after the fit: about 0.10 px median, 0.19 px rms per frame. Its static mean (0.11 px rms) is white from one curve sample to
  the next and cannot be reproduced by any band-limited displacement on held-out samples; it is the frozen codec noise of the
  original (the source has one key frame, so static lines keep one noise pattern for the whole segment). It must not be fitted; a per-sample table
  (`hk` key: a per-sample table, not used) would gain +0.006 on ride by copying noise.
- 5× window in the tracing stage (window J ≈ 0.5 after 10.7): line geometry near the pen mostly agrees at 5×; the chain does not look the same. The original
  shows hairline crisp rings in gold / olive / blue-grey and compact bright golden clumps of tiny circles; the model draws wide grey rings and a
  brown haze. Making the model rings thinner or stronger lowers the score: ring positions are not accurate enough at 5×
  (0.1 px in the main image is half a window pixel). The displacement of the original's window content against ours (median 0.26 px normal
  to the path in main-image units, white in time, uncorrelated with curvature, speed or pen history) is now reproduced by the measured table of 10.7 (window J in the tracing stage 0.43 → 0.49), but its law is still unexplained. Tested and rejected as explanations: a coarser polyline through M₀ < 8192 samples
  (M = 4096 −0.009, M = 16384 / 32768 ±0), a vertex grid, a pen offset periodic in the sample phase.
- 3D: ridge residual 0.17 px median, usable samples 65 % (far side is too dark to measure).
