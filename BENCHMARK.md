# Benchmark: solve the equations, then let another model redraw them

This repository doubles as a small, repeatable benchmark for frontier models.
The task tests two different skills: **reverse-engineering** (recover a generator from a video) and **writing a specification someone else can follow** (can a second model rebuild the renderer from the text alone?).

## 1. The two stages

| Stage | Who | Input | Output | Score reported |
|---|---|---|---|---|
| A. Solve | the model under test | the original 60 s video (1280×720, 60 fps, 3600 frames); no source code | an equation document (like [EQUATIONS.md](EQUATIONS.md)), a parameter/coefficient set, and its own renderer | **self-render score**: SCORE2 of the solver's own full-length render |
| B. Reproduce | a fixed reproducer model (currently DeepSeek V4 Pro) | only the stage-A equation document plus the parameter/coefficient files; not the solver's code | a new renderer written from scratch, and its full-length render | **reproduction score**: SCORE2 of the reproducer's render, measured by us |

The gap between the two numbers measures how complete and unambiguous the document is. A small gap means the equations, not hidden code, carry the result.

## 2. The metric: SCORE2

[`src/score2.py`](src/score2.py) compares a candidate video with the original, frame by frame, at full resolution (1280×720, RGB, no downscaling, no grayscale).

- BG = a static background plate estimated once from the original (per-pixel 15th percentile over frames 4–999, then a 31×31 spatial median).
- Per pixel and channel, with A = original, B = candidate and a dead zone δ = 2 levels for codec noise:
  - error `e = max(|A − B| − δ, 0)`
  - content `m = max(max(|A − BG|, |B − BG|, |A − B|) − δ, 0)`
- Frame score `J = 1 − Σe / Σm` (a weighted Jaccard index). Background-only pixels count for nothing, so a darker or blurrier image does not score higher.
- **SCORE2 = mean J over 514 frames** (frames 3, 10, 17, …, 3598: step 7, offset 3).
- Diagnostics that do not enter the score: per-segment and per-region J, energy ratio, sharpness ratio, and J4 (J after 4×4 pooling).

A perfect model would still not reach 1.0, because the original was encoded with lossy x264. Re-encoding the solver's own frames with the original's x264 settings and scoring them against its crf 18 render gives about **0.92** overall (stages 0.943, tracing 0.921, ride 0.888, 3D 0.904). We treat that as the practical ceiling.

The original video is **not** included. To score, obtain it from the original post and pass it with `--src`:

```
python3 src/score2.py --src original.mp4 --video your_render.mp4 --tag mine
```

## 3. Results (round 1, November 2026)

| | SCORE2 (514 frames) | stages N=3/10/36 | tracing | ride | 3D | ending | 5× / 3.1× window | file |
|---|---|---|---|---|---|---|---|---|
| Starting point (solver's previous round) | 0.7842 (self-reported) | .8839 | .6973 | .7572 | .7541 | .788 | .4878 | — |
| **Claude Fable 5.2, self-render** | **0.8132** (self-reported) | .9027 | .7349 | .7933 | .7825 | .8258 | .5532 | [results/claude_fable_5_2_self_render.json](results/claude_fable_5_2_self_render.json) |
| Claude Fable 5.2, self-render, crf 18 throughout | 0.8102 (self-reported) | .8920 | .7349 | .7933 | .7825 | .8258 | .5532 | [results/claude_fable_5_2_self_render_crf18.json](results/claude_fable_5_2_self_render_crf18.json) |
| **DeepSeek V4 Pro, reproduction from EQUATIONS.md** | **0.8089** (measured by us with `score2.py`) | .8919 | .7351 | .7934 | .7787 | .8254 | .5532 | [results/deepseek_v4_pro_reproduction.json](results/deepseek_v4_pro_reproduction.json) |
| Practical ceiling (perfect model, original's codec) | ≈ 0.92 | .943 | .921 | .888 | .904 | — | — | — |

Segments are frame ranges: stages 0–1008, tracing 1008–1584, ride 1584–2160, 3D 2160–3360, ending 3360–3600.

How to read the gap:

- Headline: the reproduction is **0.0043** below the solver's own render (0.8132 → 0.8089).
- The solver's submitted render used crf 10 for frames 0–1008 and crf 18 elsewhere; the reproducer used crf 18 everywhere. With the same encoding (crf 18 throughout) the gap is **0.0013** (0.8102 → 0.8089). Most of it is in the 3D segment (.7825 vs .7787).
- "Self-reported" means the solver ran `score2.py` on its own render. The reproduction score was measured by us with the same script on DeepSeek's video.

What the solver did **not** solve (details in EQUATIONS.md §10.6–10.7):

- The 5× inset window during tracing: its view centre wanders by sub-pixel amounts from frame to frame with no law found, so it is reproduced by a measured per-frame table; the chain inside the window does not look like the original (window J ≈ 0.55).
- Why the stage curves (N = 3, 10, 36) use their own low-order coefficients instead of the low-order part of the N = 1024 figure.
- The remaining sub-pixel line offsets in the ride and 3D segments.

So this is a close reconstruction, not a proof that these are fofr's exact equations.

## 4. Ambiguities the reproducer reported

DeepSeek V4 Pro was asked to list every place where the document was unclear and what it chose. Its notes, summarised:

1. **YUV inverse transpose.** §6.7 writes `yuv @ inverse(YUV)`, §6.8 writes `yuv @ inverse(YUV)ᵀ`. The mathematical inverse of `yuv = rgb @ YUVᵀ` is `inverse(YUV)ᵀ`, and §6.8's `codec_bias` cancellation requires both to match, so it used the transpose in both places.
2. **Turntable ticks (§6.6).** It drew the 4 ticks as 4 separate radial segments rather than one connected polyline (which would draw a diamond).
3. **Window title bar rows (§7.1).** It followed the text order: title colour on rows 0..25, separator on row 25, rounded border, then the 28-row atlas sprite over rows 0..27.
4. **Window clip.** The 2 px safety margin of `splat` intersected with the window rect leaves content in rows 25..208.
5. **Precision.** Density images in float32, image accumulation in float64.
6. **`stage_c` range.** The 73 coefficients replace rows k = −36..36 before truncation.
7. **`tail_to_pen`.** The appended pen point takes the weight of sample n_vis−1.
8. **Window tail weights.** `[fw[i0], …, fw[n−1], fw[n−1]]`.
9. **Chain circle centres.** n circles use centres cen_0..cen_{n−1}; all n+1 points are used for the arms.
10. **Readout rounding.** ψ uses Python `round` (half to even); θ uses floor(x + 0.5 + 1e−6); pitch uses floor(x + 0.5).

EQUATIONS.md is published exactly as the reproducer received it (apart from removing internal file names, paths and the measured tables listed in the README), so these ambiguities are left in place on purpose: they are part of the benchmark record. A future version of the document should resolve them.

DeepSeek's render took about 27 minutes of wall time for 3600 frames with two processes (about 0.34–0.50 s per frame on one core).

## 5. Adding a new model

For each new model generation we plan to:

1. Give the model the original video and the stage-A task (recover the generator; report a self-render SCORE2 on the 514-frame set).
2. Hand its equation document and parameter files, without its code, to the reproducer (DeepSeek V4 Pro, kept fixed so results stay comparable).
3. Score the reproduction with `score2.py`, and add a row to the table above with both numbers and the gap, plus the per-segment breakdown.

Results files follow the format of `results/*.json`: overall `SCORE2`, `J4`, `energy`, `sharp`, `n`, per-segment `segs`, per-region `regions` (J and share of the total loss), the eight `worst` frames, and `per_frame` = frame → [J, J4, energy, sharp].

Because the original video and the fitted coefficients are not redistributed, runs are coordinated by the maintainer. If you want to run the benchmark with your model, open an issue or contact [@YuLin807](https://x.com/YuLin807) on X.
