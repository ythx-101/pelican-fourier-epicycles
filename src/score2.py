#!/usr/bin/env python3
"""SCORE2: scoring script (full-resolution 1280x720, RGB, no downscaling, dense frame sampling).

SPDX-License-Identifier: MIT

The original video is NOT included in this repository. Obtain it from the original
post and pass it with --src. The background-plate cache (score2_bg.npy) is derived
from the original video; do not redistribute it.

Definition (per frame, per pixel, per channel):
  A = original frame, B = our frame (both decoded mp4, rgb24),
  BG = static background of the original (temporal low percentile + spatial median, computed once).
     e = max(|A-B| - delta, 0)                                       error (delta = 2 levels: codec-noise dead zone)
     m = max(max(|A-BG|, |B-BG|, |A-B|) - delta, 0)               "content amount": how much non-background there is at this pixel
  frame score J = 1 - sum(e) / sum(m)      (weighted Jaccard: signed generalisation of sum(min)/sum(max))
  SCORE2 = mean of J over all sampled frames.
Properties: only pixel-exact agreement scores 1; blurred lines (energy spread out), too dark, too bright,
      shifted, missing or extra drawing are all penalised; pure-background pixels do not count (the score is not
      inflated by the ~85% background), so "darker and blurrier scores higher" does not hold.
Also reported (diagnostics only, not part of the total): energy ratio sum(B')/sum(A') (<1 = too dark),
      sharpness ratio sum|grad B|/sum|grad A| (<1 = too blurry), J4 (J after 4x4 pooling: high J4 with low J
      means right structure, 1-2 px offset), per-segment and per-region (main/window/HUD/title) scores, loss heat map.

Usage:
  python3 score2.py --src orig.mp4 --video full.mp4 [--step 7 --off 3] [--tag x]          # score a whole video (official setting: step 7 -> 514 frames)
  python3 score2.py --src orig.mp4 --gen gen.py --params params.json --frames 1100,1500  # iterate: render given frames -> x264 crf18 round trip -> score
  python3 score2.py --src orig.mp4 --gen gen.py --params params.json --step 30 --off 5   # iterate: evenly spaced frames
  python3 score2.py --src orig.mp4 --ceiling                                              # score of the original re-encoded once at crf18 (noise ceiling)
--gen is any generator script (not src/render.py, which has a different CLI) accepting `--params P --frames a,b,c --outdir D` and writing D/f*.png.
"""
import argparse, subprocess, os, json, sys, time, tempfile, struct, zlib
import numpy as np
SRC = None; CACHE = './score2_cache'; W, H = 1280, 720; N = W * H * 3; DELTA = 2.0
SEGS = [('stages N=3/10/36 0-1008', 0, 1008), ('tracing 1008-1584', 1008, 1584), ('ride 1584-2160', 1584, 2160), ('3D 2160-3360', 2160, 3360), ('ending 3360-3600', 3360, 3600)]
REG = {'window': (950, 310, 1236, 546), 'HUD': (0, 548, 1280, 700), 'title': (200, 0, 1080, 56)}   # x0,y0,x1,y1; everything else = main

def savepng(path, arr):
    """Minimal RGB PNG writer (zlib/struct)."""
    arr = np.clip(arr, 0, 255).astype(np.uint8); h, w, _ = arr.shape
    def chunk(tag, data): return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    raw = b''.join(b'\x00' + arr[y].tobytes() for y in range(h))
    with open(path, 'wb') as fp:
        fp.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))

def reader(path, fs):
    """Yield frames from the video for the (ascending) frame-number list fs."""
    fs = list(fs); out = {}
    for i in range(0, len(fs), 80):
        ch = fs[i:i + 80]; sel = '+'.join(f'eq(n\\,{f})' for f in ch)
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', path, '-vf', f'select={sel}', '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
        for f in ch:
            b = p.stdout.read(N)
            if len(b) < N: raise SystemExit(f'{path}: cannot read frame {f}')
            yield f, np.frombuffer(b, np.uint8).reshape(H, W, 3)
        p.stdout.close(); p.wait()

def background():
    """Static background of the original: per-pixel temporal 15th percentile over frames 0-1000 (before the window appears), then a 31x31 spatial median (removes the static HUD text). Cached in CACHE, keyed by source size."""
    os.makedirs(CACHE, exist_ok=True)
    path = f'{CACHE}/score2_bg_{os.path.getsize(SRC)}.npy'
    if os.path.exists(path): return np.load(path)
    from scipy.ndimage import median_filter
    st = np.stack([im for _, im in reader(SRC, range(4, 1000, 12))]); q = np.percentile(st, 15, axis=0).astype(np.float32); del st
    bg = np.stack([median_filter(q[..., c], size=31, mode='nearest') for c in range(3)], 2); np.save(path, bg); return bg

def frame_terms(A, B, BG):
    A = A.astype(np.float32); B = B.astype(np.float32); d = np.abs(A - B); a = np.abs(A - BG); b = np.abs(B - BG)
    e = np.maximum(d - DELTA, 0); m = np.maximum(np.maximum(np.maximum(a, b), d) - DELTA, 0)
    return e.sum(2), m.sum(2), a, b

def grad(x): g = x.sum(2); return np.abs(np.diff(g, axis=0)).sum() + np.abs(np.diff(g, axis=1)).sum()
def pool4(x): return x.reshape(H // 4, 4, W // 4, 4).sum((1, 3))

def score(pairs, BG, heat=None):
    rows = []; reg = {k: [0.0, 0.0] for k in list(REG) + ['main']}; hm_e = np.zeros((H, W)); hm_m = np.zeros((H, W))
    mask = np.zeros((H, W), np.int8); names = list(REG)
    for i, k in enumerate(names): x0, y0, x1, y1 = REG[k]; mask[y0:y1, x0:x1] = i + 1
    for f, A, B in pairs:
        e, m, a, b = frame_terms(A, B, BG); E = e.sum(); M = m.sum() + 1e-9
        Af = A.astype(np.float32) - BG; Bf = B.astype(np.float32) - BG
        d4 = np.abs(np.stack([pool4(Af[..., c] - Bf[..., c]) for c in range(3)])) / 16; a4 = np.abs(np.stack([pool4(Af[..., c]) for c in range(3)])) / 16; b4 = np.abs(np.stack([pool4(Bf[..., c]) for c in range(3)])) / 16
        e4 = np.maximum(d4 - DELTA / 2, 0).sum(); m4 = np.maximum(np.maximum(np.maximum(a4, b4), d4) - DELTA / 2, 0).sum() + 1e-9
        am = np.maximum(a - DELTA, 0); bm = np.maximum(b - DELTA, 0)
        rows.append(dict(f=f, J=float(1 - E / M), J4=float(1 - e4 / m4), energy=float(bm.sum() / (am.sum() + 1e-9)), sharp=float(grad(bm) / (grad(am) + 1e-9)), E=float(E), M=float(M)))
        hm_e += e; hm_m += m
        for i, k in enumerate(names + ['main']):
            sel = mask == ((i + 1) if i < len(names) else 0); reg[k][0] += float(e[sel].sum()); reg[k][1] += float(m[sel].sum())
    J = np.array([r['J'] for r in rows]); fs = np.array([r['f'] for r in rows])
    out = dict(SCORE2=round(float(J.mean()), 4), J4=round(float(np.mean([r['J4'] for r in rows])), 4), energy=round(float(np.mean([r['energy'] for r in rows])), 3),
               sharp=round(float(np.mean([r['sharp'] for r in rows])), 3), n=len(rows))
    out['segs'] = {n: round(float(J[(fs >= a) & (fs < b)].mean()), 4) for n, a, b in SEGS if ((fs >= a) & (fs < b)).any()}
    tot = sum(v[0] for v in reg.values()) + 1e-9
    out['regions'] = {k: dict(J=round(1 - v[0] / (v[1] + 1e-9), 4), loss_share=round(v[0] / tot, 3)) for k, v in reg.items()}
    out['worst'] = [(r['f'], round(r['J'], 4)) for r in sorted(rows, key=lambda r: r['J'])[:8]]
    out['per_frame'] = {r['f']: [round(r['J'], 4), round(r['J4'], 4), round(r['energy'], 3), round(r['sharp'], 3)] for r in rows}
    if heat:
        g = np.clip(hm_e / (np.percentile(hm_e, 99.7) + 1e-9), 0, 1) ** 0.5 * 255; mm = np.clip(hm_m / (np.percentile(hm_m, 99.7) + 1e-9), 0, 1) ** 0.5 * 255
        savepng(heat, np.stack([g, mm * 0.5, mm * 0.5], 2))      # red = loss, grey-cyan = content
    return out

# x264 settings of the original video (read from its SEI: crf 28, vbv 2048, psy off, aq-mode 2, subme 2 ...)
X264SRC = 'ref=5:deblock=0,0:me=hex:subme=2:psy=0:mixed-refs=1:trellis=1:8x8dct=1:bframes=3:b-pyramid=normal:b-adapt=1:direct=spatial:weightb=1:weightp=2:keyint=infinite:min-keyint=30:scenecut=0:rc-lookahead=40:mbtree=0:vbv-maxrate=2048:vbv-bufsize=2048:aq-mode=2:aq-strength=1.0:fast-pskip=1:chroma-qp-offset=0:qcomp=0.6'
ENC = {'crf18': ['-crf', '18', '-g', '1'], 'crf18seq': ['-crf', '18'], 'src': ['-crf', '28', '-x264-params', X264SRC], 'srci': ['-crf', '28', '-x264-params', X264SRC + ':keyint=1:min-keyint=1']}
def roundtrip_frames(frames, enc='crf18'):
    """libx264 yuv420p round trip. enc: crf18 (all-intra) / crf18seq / src (original encoder settings, for consecutive frames) / srci (original settings, all-intra, for sparse sampling)."""
    tmp = tempfile.mktemp(suffix='.mp4', dir=CACHE)
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '60', '-i', '-', '-c:v', 'libx264'] + ENC[enc] + ['-pix_fmt', 'yuv420p', tmp], stdin=subprocess.PIPE)
    for _, im in frames: ff.stdin.write(im.tobytes())
    ff.stdin.close(); ff.wait()
    out = [(f, im.copy()) for (f, _), (_, im) in zip(frames, reader(tmp, range(len(frames))))]; os.remove(tmp); return out

def render(gen, params, fs, procs=2):
    d = tempfile.mkdtemp(prefix='s2_', dir=CACHE); ps = []
    for i in range(procs):
        sub = fs[i::procs]
        if sub: ps.append(subprocess.Popen([sys.executable, gen, '--params', params, '--frames', ','.join(map(str, sub)), '--outdir', d]))
    if any(p.wait() for p in ps): raise SystemExit('generator failed')
    p = subprocess.run(['ffmpeg', '-v', 'error', '-pattern_type', 'glob', '-i', f'{d}/f*.png', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
    subprocess.run(['rm', '-rf', d]); return [(f, np.frombuffer(p[i * N:(i + 1) * N], np.uint8).reshape(H, W, 3)) for i, f in enumerate(sorted(fs))]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--src', required=True, help='original video (not included; obtain from the original post)'); ap.add_argument('--cache', default='./score2_cache', help='cache/temp dir'); ap.add_argument('--video'); ap.add_argument('--gen'); ap.add_argument('--params'); ap.add_argument('--ceiling', action='store_true')
    ap.add_argument('--frames'); ap.add_argument('--step', type=int, default=7); ap.add_argument('--off', type=int, default=3); ap.add_argument('--range', nargs=2, type=int, default=[0, 3600])
    ap.add_argument('--tag', default='s2'); ap.add_argument('--outdir', default='./eval'); ap.add_argument('--heat', action='store_true'); ap.add_argument('--raw', action='store_true', help='gen mode: skip the x264 round trip'); ap.add_argument('--enc', default='crf18'); ap.add_argument('--keep'); ap.add_argument('--npy', help='use saved raw rendered frames (as stored by --keep)'); a = ap.parse_args(); global SRC, CACHE; SRC = a.src; CACHE = a.cache; os.makedirs(CACHE, exist_ok=True)
    if not os.path.exists(SRC): raise SystemExit(f'original video not found: {SRC} (not included in this repository; obtain it from the original post and pass it with --src)')
    fs = sorted(int(x) for x in a.frames.split(',')) if a.frames else list(range(a.range[0] + a.off, a.range[1], a.step))
    BG = background(); t0 = time.time(); os.makedirs(a.outdir, exist_ok=True)
    if a.video: ours = reader(a.video, fs)
    elif a.ceiling: ours = iter(roundtrip_frames(list(reader(SRC, fs)), a.enc))
    else:
        fr = list(zip(fs, np.load(a.npy))) if a.npy else render(a.gen, a.params, fs); ours = iter(fr if a.raw else roundtrip_frames(fr, a.enc))
        if a.keep: np.save(a.keep, np.stack([im for _, im in fr]))
    pairs = ((f, A, B) for (f, A), (_, B) in zip(reader(SRC, fs), ours))
    out = score(pairs, BG, f'{a.outdir}/{a.tag}_heat.png' if a.heat else None); out['tag'] = a.tag; out['sec'] = round(time.time() - t0, 1)
    json.dump(out, open(f'{a.outdir}/{a.tag}.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'per_frame'}, ensure_ascii=False))
    print(f"SCORE2 = {out['SCORE2']:.4f}")
if __name__ == '__main__': main()
