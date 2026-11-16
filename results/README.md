# results/

SCORE2 output files (see [BENCHMARK.md](../BENCHMARK.md)), 514 frames each (frames 3, 10, …, 3598).

| File | Render | SCORE2 |
|---|---|---|
| `claude_fable_5_2_self_render.json` | Claude Fable 5.2's own render of its equations (crf 10 on frames 0–1008, crf 18 elsewhere) | 0.8132 (self-reported) |
| `claude_fable_5_2_self_render_crf18.json` | same model, crf 18 throughout | 0.8102 (self-reported) |
| `deepseek_v4_pro_reproduction.json` | DeepSeek V4 Pro's renderer written from EQUATIONS.md, crf 18 | 0.8089 (measured with `src/score2.py`) |

Fields: `SCORE2` (mean frame J), `J4` (J after 4×4 pooling), `energy` (content energy ratio render/original, < 1 = darker), `sharp` (gradient ratio, < 1 = blurrier), `n` (frames), `segs` (mean J per segment), `regions` (J and share of total loss for the inset window, HUD, title and main image), `worst` (eight lowest frames), `per_frame` (frame → [J, J4, energy, sharp]), `tag`.

The files are the scorer's output with the segment and region labels translated to English; the values are unchanged.
