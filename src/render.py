#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render.py -- independent re-implementation of the "Pelican Cycling in 3D"
generator, written by DeepSeek V4 Pro from EQUATIONS.md alone (plus the fitted
params file, coefficients and measured atlas, which are NOT distributed here).

SPDX-License-Identifier: MIT

Usage:
    python3 render.py --params path/to/params.json --frames A B --out x.mp4   # frames [A, B) to H.264 (needs ffmpeg)
    python3 render.py --params path/to/params.json --frames A B --out x.png   # single frame (B == A+1) to PNG

The coefficient file (P['coeffs']) and atlas file (P['atlas']) named inside
params.json are resolved relative to the directory of params.json.
"""
import os, sys, json, math, struct, zlib, subprocess, argparse
import numpy as np
from scipy import ndimage, interpolate

NOT_FOUND_MSG = ('params.json / coeffs_r3.npz / atlas.npz not found. The fitted coefficients and '
                 'measured tables are not distributed with this repository out of respect for the '
                 'original work; see README ("Why the coefficients are not included"). EQUATIONS.md documents their '
                 'shapes and meaning.')

# ---------------------------------------------------------------- loading
# Filled in by load(params_path); everything below that is derived from the
# params / coefficients / atlas is computed there.
P = C = AT = AT_FILES = None
K = CC = ZZ = M = META = None
pivot_x = pivot_y = focal = O = None
STAGE_C = WARP_F = WARP_D = TAU_F = TAU_X = TAU_Y = None
THETA_INT = PITCH_INT = None
bg = None
W, H = 1280, 720

YUV_MAT = np.array([[65.481, 128.553, 24.966],
                    [-37.797, -74.203, 112.0],
                    [112.0, -93.786, -18.214]]) / 255.0
YUV_INV_T = np.linalg.inv(YUV_MAT).T

SQRT2PI = 2.5066

def load(params_path):
    global P, C, AT, AT_FILES, K, CC, ZZ, M, META
    global pivot_x, pivot_y, focal, O, STAGE_C, WARP_F, WARP_D
    global TAU_F, TAU_X, TAU_Y, THETA_INT, PITCH_INT, bg
    try:
        base = os.path.dirname(os.path.abspath(params_path))
        P = json.load(open(params_path))
        C = np.load(os.path.join(base, P['coeffs']))
        AT = np.load(os.path.join(base, P['atlas']))
        AT_FILES = set(AT.files)

        K = C['k'].astype(np.int64)          # 2049, k = -1024..1024
        CC = C['C']                          # (2049, 33)  col j <-> p = j-16
        ZZ = C['Z']
        M = int(P['M'])

        META = json.loads(str(AT['meta']))

        pivot_x = float(P['pivot'][0]); pivot_y = float(P['pivot'][1])
        focal = float(P['focal'])
        O = complex(float(P['origin'][0]), float(P['origin'][1]))

        STAGE_C = np.array([re + 1j*im for (re, im) in P['stage_c']], dtype=np.complex128)  # 73, k=-36..36

        WARP_F = np.array([r[0] for r in P['spen_warp']], dtype=float)
        WARP_D = np.array([r[1] for r in P['spen_warp']], dtype=float)

        TAU_F = np.array([r[0] for r in P['hud']['pen_view_tab']], dtype=float)
        TAU_X = np.array([r[1] for r in P['hud']['pen_view_tab']], dtype=float)
        TAU_Y = np.array([r[2] for r in P['hud']['pen_view_tab']], dtype=float)

        THETA_INT = interpolate.PchipInterpolator(P['theta_knots_f'], P['theta_knots_deg'])
        PITCH_INT = interpolate.PchipInterpolator(P['pitch_knots_f'], P['pitch_knots_deg'])
        bg = make_background()
    except FileNotFoundError:
        raise SystemExit(NOT_FOUND_MSG)
    except KeyError as e:
        raise SystemExit('missing required key %s in params / coefficients / atlas (%s)' % (e, params_path))

# ---------------------------------------------------------------- helpers
def gauss(den, s):
    return ndimage.gaussian_filter(den, s)

def dense(z, w=None, step=0.6, closed=False):
    """polyline -> arc-length points and weights (section 6.1)."""
    z = np.asarray(z, dtype=np.complex128)
    if closed:
        z = np.concatenate([z, z[:1]])
        if w is not None:
            w = np.concatenate([w, w[:1]])
    if len(z) < 2:
        return np.zeros(0, np.complex128), np.zeros(0)
    dz = np.diff(z)
    d = np.abs(dz)
    n = np.maximum(np.ceil(d / step).astype(np.int64), 1)
    total = int(n.sum())
    seg = np.repeat(np.arange(len(n)), n)
    n_r = np.repeat(n, n)
    j = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
    t = (j + 0.5) / n_r
    pts = z[seg] + dz[seg] * t
    d_on_n = np.repeat(d / n, n)
    if w is None:
        wts = d_on_n
    else:
        w = np.asarray(w, dtype=np.float64)
        wa = w[seg]; wb = w[seg + 1]
        wts = d_on_n * (wa + t * (wb - wa))
    return pts, wts

def circle_points(cens, rads, weights, nmin=10):
    cens = np.asarray(cens, np.complex128); rads = np.asarray(rads, np.float64)
    weights = np.asarray(weights, np.float64)
    if len(rads) == 0:
        return np.zeros(0, np.complex128), np.zeros(0)
    npt = np.clip(np.rint(2 * np.pi * rads / 0.5).astype(np.int64), nmin, 4000)
    total = int(npt.sum())
    seg = np.repeat(np.arange(len(npt)), npt)
    npt_r = np.repeat(npt, npt)
    j = np.arange(total) - np.repeat(np.cumsum(npt) - npt, npt)
    ang = 2 * np.pi * j / npt_r
    pts = cens[seg] + rads[seg] * np.exp(1j * ang)
    wts = (2 * np.pi * rads[seg] / npt_r) * weights[seg]
    return pts, wts

def deposit(Hh, Ww, pts, wts, clip=None):
    """bilinear splat of weighted points into a float32 density image."""
    pts = np.asarray(pts, np.complex128); wts = np.asarray(wts, np.float64)
    x = pts.real; y = pts.imag
    keep = (x >= 1) & (x < Ww - 2) & (y >= 1) & (y < Hh - 2)
    if clip is not None:
        x0, y0, x1, y1 = clip
        keep &= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    x = x[keep]; y = y[keep]; wts = wts[keep]
    if len(x) == 0:
        return np.zeros((Hh, Ww), np.float32)
    ix = np.floor(x).astype(np.int64); iy = np.floor(y).astype(np.int64)
    fx = x - ix; fy = y - iy
    flat = iy * Ww + ix
    idx = np.concatenate([flat, flat + 1, flat + Ww, flat + Ww + 1])
    wt = np.concatenate([wts * (1 - fx) * (1 - fy), wts * fx * (1 - fy),
                         wts * (1 - fx) * fy, wts * fx * fy])
    acc = np.bincount(idx, weights=wt, minlength=Hh * Ww).reshape(Hh, Ww).astype(np.float32)
    return acc

def stroke(img, den, s, colour, gain=1.0, glow=None, sat=None):
    satv = P['sat'] if sat is None else sat
    colour = np.asarray(colour, np.float64)
    g = gauss(den, s)
    v = np.minimum(g * (s * SQRT2PI) * gain, satv)
    img += v[..., None] * colour
    if glow:
        for item in glow:
            sg = item[0]; cg = np.asarray(item[1], np.float64)
            Gg = item[2] if len(item) > 2 else None
            gg = gauss(den, sg) * (sg * SQRT2PI) * gain
            if Gg is not None:
                img += np.minimum(gg * Gg, 1.0)[..., None] * cg
            else:
                img += np.minimum(gg, 1.5)[..., None] * cg

# ---------------------------------------------------------------- time fns
def spen_delta(f):
    if f < 1008 or f >= 1584:
        return 0.0
    return float(np.interp(f, WARP_F, WARP_D, left=0, right=0))

def stage(f):
    ff = f + spen_delta(f) if (1008 <= f < 1584) else float(f)
    for (f0, f1, N) in P['stages']:
        if ff < f1:
            return N, (ff - f0) / (f1 - f0), False
    if ff >= P['f3d']:
        return 1024, ((ff - P['f3d']) % P['period3d']) / P['period3d'], True
    return 1024, ((ff - P['ride_start']) % P['ride_period']) / P['ride_period'], True

def psi_rad(f):
    rs = P['ride_start']; pr = P['psi_rate']
    if f >= rs:
        deg = pr * (f - rs)
    else:
        f0 = P['psi_ease_start']
        deg = pr / (2.0 * (rs - f0)) * (max(f - f0, 0.0) ** 2)
    return math.radians(deg)

def theta_deg(f):
    return float(THETA_INT(f)) if f >= P['f3d'] else 0.0

def pitch_deg(f):
    return float(PITCH_INT(f)) if f >= P['f3d'] else 0.0

def tau(f):
    if f < TAU_F[0] or f > TAU_F[-1]:
        return (0.0, 0.0)
    return (float(np.interp(f, TAU_F, TAU_X, left=0.0, right=0.0)),
            float(np.interp(f, TAU_F, TAU_Y, left=0.0, right=0.0)))

def psiread(f):
    v = math.degrees(2 * math.pi * ((f / 60.0 - 26.4) / 2.4)) % 360.0
    return round(v) % 360

def theta_read(f):
    return int(math.floor(theta_deg(f) + 0.5 + 1e-6)) % 360

def pitch_read(f):
    return int(math.floor(pitch_deg(f) + 0.5))

# ---------------------------------------------------------------- curve
def synth(c, N):
    F = np.zeros(M, np.complex128)
    ks = np.arange(-N, N + 1, dtype=np.int64)
    F[ks % M] = c[ks + 1024]
    return M * np.fft.ifft(F)

def project(z, dep, th, ph):
    x = z.real - pivot_x
    y = z.imag - pivot_y
    ct = math.cos(th); st = math.sin(th)
    cp = math.cos(ph); sp = math.sin(ph)
    x1 = x * ct + dep * st
    d1 = -x * st + dep * ct
    y1 = y * cp + d1 * sp
    d2 = -y * sp + d1 * cp
    kap = 1.0 / (1.0 - d2 / focal)
    scr = (pivot_x + x1 * kap) + 1j * (pivot_y + y1 * kap)
    return scr, d2

def build_chain(cc, N, sp):
    sel = np.abs(K) <= N
    kk = K[sel]
    cc = cc[sel].copy()
    cc[kk == 0] -= O
    order = np.argsort(-np.abs(cc), kind='stable')
    kk = kk[order]; cc = cc[order]
    terms = cc * np.exp(1j * kk * (2 * np.pi * sp))
    partial = np.concatenate([[0 + 0j], np.cumsum(terms)])
    cen = O + partial
    rad = np.abs(cc)
    pen = cen[-1]
    return cen, rad, kk

def build_scene(f):
    N, sp, full = stage(f)
    psir = psi_rad(f)
    e = np.exp(1j * psir * np.arange(-16, 17))
    c = CC @ e
    if N < 1024:
        c[988:1061] = STAGE_C          # rows k = -36..36
    zs = synth(c, N)
    bright = None
    if f >= P['f3d']:
        zc = ZZ @ e
        dep = synth(zc, 1024).real
        th = math.radians(theta_deg(f)); ph = math.radians(pitch_deg(f))
        zs, d2 = project(zs, dep, th, ph)
        b = (d2 - d2.min()) / (d2.max() - d2.min() + 1e-9)
        b = np.interp(b, np.linspace(0, 1, 6), P['bright_lut'])
        t = np.clip((f - P['look_in'][0]) / (P['look_in'][1] - P['look_in'][0]), 0, 1)
        bright = 1.0 - t * (1.0 - b)
        cch = np.fft.fft(zs) / M
        cen, rad, kk = build_chain(cch[K % M], 1024, sp)
    else:
        cen, rad, kk = build_chain(c, N, sp)
    n_vis = M if full else int(math.floor(sp * M))
    return dict(N=N, sp=sp, full=full, zs=zs, bright=bright,
                cen=cen, rad=rad, kk=kk, n=len(rad), pen=cen[-1], n_vis=n_vis)

# ---------------------------------------------------------------- draw view
def draw_view(img, scene, f, view, base):
    Hh, Ww = img.shape[:2]
    chain_a = np.clip((P['chain_end'] - f) / P['chain_fade'], 0.0, 1.0)

    if view is None:
        win = False
        zoom = 1.0; clip = None
        T = lambda z: z
        line_sigma = P['line_sigma']; line_color = P['line_color']
        glow = [(P['glow_sigma'], P['glow_color'], P['glow_gain']),
                (P['mid_sigma'], P['mid_color'])]
        cap = P['cap2d']
    else:
        win = True
        cx, cy, zoom, rect = view
        clip = rect
        wc = ((rect[0] + rect[2]) / 2.0) + 1j * ((rect[1] + rect[3]) / 2.0)
        def T(z):
            return (z - (cx + 1j * cy)) * zoom + wc
        line_sigma = P['win_line_sigma']; line_color = P['win_line_color']
        glow = [(P['win_glow_sigma'], P['win_glow_color'], P['win_glow_gain'])]
        cap = P['win_cap']

    cen = scene['cen']; rad = scene['rad']; n = scene['n']
    cen_t = T(cen)
    cen_c = cen_t[:-1]            # circle centres (n of them)
    R = rad * zoom

    if chain_a > 0:
        # i0 = first index with rad < gold_r (n if none)
        below = np.nonzero(rad < P['gold_r'])[0]
        i0 = int(below[0]) if len(below) else n

        if win:
            fw = np.clip(R / P['win_fade_r'], 0.0, 1.0) ** P['win_fade_g']
            selw = np.nonzero(R >= P['win_rmin'])[0]
            for gi, hue in enumerate(P['win_hues']):
                sel = selw[selw % 3 == gi]
                pts, wts = circle_points(cen_c[sel], R[sel], fw[sel], nmin=8)
                den = deposit(Hh, Ww, pts, wts, clip)
                if gi == 0:
                    ap, aw = dense(cen_t[:min(n, P['win_arm_count']) + 1])
                    den += deposit(Hh, Ww, ap, aw * P['win_arm_gain'], clip)
                g = gauss(den, P['win_circle_sigma'])
                a = 1.0 - np.exp(-g * (P['win_circle_sigma'] * SQRT2PI) * chain_a * P['win_circle_alpha'])
                img[:] = img * (1 - a[..., None]) + a[..., None] * np.asarray(hue, np.float64)

            # window tail links, dim gold
            tp, tw = dense(cen_t[i0:], w=np.concatenate([fw[i0:], fw[n - 1:n]]))
            den = deposit(Hh, Ww, tp, tw, clip)
            a = (1.0 - np.exp(-gauss(den, P['win_tail_sigma']) * chain_a * P['win_gold_alpha'])) * P['gold_sat']
            img[:] = img * (1 - a[..., None]) + a[..., None] * np.asarray(P['win_tail_color'], np.float64)

            # main-style arm layer again (additive), empty circle selection
            ap, aw = dense(cen_t[:min(n, P['win_arm_count']) + 1])
            den = deposit(Hh, Ww, ap, aw * P['arm_gain'], clip)
            stroke(img, den, P['circle_sigma'] * P['win_circle_scale'], P['win_circle_color'],
                   gain=P['circle_alpha'] * chain_a * P['win_circle_alpha'], sat=P['win_circle_sat'])

            # tail links again, bright gold
            tp, tw = dense(cen_t[i0:])
            den = deposit(Hh, Ww, tp, tw, clip)
            v = gauss(den, P['tail_sigma'] * P['win_circle_scale']) * P['gold_alpha'] * chain_a * P['win_gold_alpha']
            a = (1.0 - np.exp(-v)) * P['gold_sat']
            img[:] = img * (1 - a[..., None]) + a[..., None] * np.asarray(P['gold_color'], np.float64)
        else:
            big = rad > P['circle_min_r']
            large = big & (rad >= P['gold_r'])
            small = big & (rad < P['gold_r'])

            # large circles + first arms: additive
            pts, wts = circle_points(cen_c[large], R[large], np.ones(int(large.sum())), nmin=10)
            den = deposit(Hh, Ww, pts, wts, clip)
            ap, aw = dense(cen_t[:min(n, P['arm_count']) + 1])
            den += deposit(Hh, Ww, ap, aw * P['arm_gain'], clip)
            stroke(img, den, P['circle_sigma'], P['circle_color'],
                   gain=P['circle_alpha'] * chain_a, sat=P['circle_sat'])

            # tail (small circles + links i0..pen): over
            pts, wts = circle_points(cen_c[small], R[small],
                                     np.full(int(small.sum()), P['tail_circle_gain']), nmin=10)
            den = deposit(Hh, Ww, pts, wts, clip)
            tp, tw = dense(cen_t[i0:])
            den += deposit(Hh, Ww, tp, tw, clip)
            v = gauss(den, P['tail_sigma']) * P['gold_alpha'] * chain_a
            a = (1.0 - np.exp(-v)) * P['gold_sat']
            colour = P['gold_color'] if f >= P['gold_start'] else P['tail_color']
            img[:] = img * (1 - a[..., None]) + a[..., None] * np.asarray(colour, np.float64)

    # ---- main curve
    n_vis = scene['n_vis']; full = scene['full']
    if n_vis > 1:
        zs = T(scene['zs'][:n_vis])
        if (not full) and P['tail_to_pen']:
            zs = np.concatenate([zs, T(scene['pen'])[None]])
        if f < P['f3d']:
            lay = np.zeros((Hh, Ww, 3), np.float64)
            pts, wts = dense(zs, closed=full)
            den = deposit(Hh, Ww, pts, wts, clip)
            stroke(lay, den, line_sigma, line_color, gain=1.0, glow=glow, sat=None)
        else:
            bright = scene['bright'][:n_vis]
            pts, wts = dense(zs, closed=full)
            dT = deposit(Hh, Ww, pts, wts, clip)
            pts2, wts2 = dense(zs, w=bright, closed=full)
            dB = deposit(Hh, Ww, pts2, wts2, clip)
            lay = np.zeros((Hh, Ww, 3), np.float64)
            for (s, cn, cf, top) in [(P['line_sigma'] * P['sigma3d'], P['near_color'], P['far_color'], P['sat']),
                                     (P['glow_sigma3d'], P['near_glow'], P['far_glow'], 1.5),
                                     (P['mid3d_sigma'], P['mid3d_color'], 0.0, 1.5)]:
                cn = np.asarray(cn, np.float64); cf = np.asarray(cf, np.float64)
                gT = gauss(dT, s)
                b = gauss(dB, s) / np.maximum(gT, 1e-6)
                v = np.minimum(gT * (s * SQRT2PI), top)
                lay += v[..., None] * (b[..., None] * cn + (1 - b[..., None]) * cf)
        capc = np.asarray(cap, np.float64)
        lay = lay / (1.0 + (lay / capc) ** P['cap_q']) ** (1.0 / P['cap_q'])
        cov = np.clip(np.max(lay / capc, axis=-1) * P['line_over'], 0.0, 1.0)
        img[:] = img - (img - base) * cov[..., None] + lay

    # ---- pen dot and halo
    if chain_a > 0:
        p = T(scene['pen'])
        den = deposit(Hh, Ww, np.array([p]), np.array([1.0]), clip)
        ps = 1.0 if not win else P['win_pen_scale']
        s = P['pen_sigma'] * ps
        a = np.minimum(gauss(den, s) * (2 * np.pi * s * s) * P['pen_gain'], 1.0) * chain_a
        img[:] = img * (1 - a[..., None]) + a[..., None] * np.asarray(P['pen_color'], np.float64)
        sh = P['pen_halo_sigma'] * ps
        img += gauss(den, sh)[..., None] * (2 * np.pi * sh * sh) * np.asarray(P['pen_halo_color'], np.float64) * chain_a

# ---------------------------------------------------------------- turntable
def draw_turntable(img, f):
    fa = np.clip((f - P['ring_in'][0]) / (P['ring_in'][1] - P['ring_in'][0]), 0.0, 1.0)
    if fa <= 0:
        return
    th = math.radians(theta_deg(f)); ph = math.radians(pitch_deg(f))
    Rr = P['ring_r']; yg = P['ring_y']; x0 = pivot_x

    a_ang = 2 * np.pi * np.arange(720) / 720.0
    zr = (x0 + Rr * np.cos(a_ang)) + 1j * yg
    ring_scr, _ = project(zr, Rr * np.sin(a_ang), 0.0, ph)
    rp, rw = dense(ring_scr, closed=True)
    den_ring = deposit(H, W, rp, rw, None)

    u = np.linspace(-1.0, 1.0, 400)
    on = np.where(np.floor((u + 1.0) * P['ring_dashes']).astype(np.int64) % 2 == 0, 1.0, 0.0)
    zd = (x0 + Rr * u) + 1j * yg
    diam_scr, _ = project(zd, np.zeros(400), th, ph)
    dp, dw = dense(diam_scr, w=on * P['ring_dash_gain'])
    den_diam = deposit(H, W, dp, dw, None)

    stroke(img, den_ring + den_diam, P['ring_sigma'], P['ring_color'], gain=P['ring_alpha'] * fa, sat=None)

    rr = Rr * np.linspace(0.95, 1.05, 20)
    den_ticks = np.zeros((H, W), np.float32)
    for q in range(4):
        aq = q * math.pi / 2.0
        zt = (x0 + rr * np.cos(aq)) + 1j * yg
        tick_scr, _ = project(zt, rr * np.sin(aq), th, ph)
        tp, tw = dense(tick_scr)
        den_ticks += deposit(H, W, tp, tw, None)
    stroke(img, den_ticks, P['circle_sigma'] * 1.5, P['gold_color'], gain=P['tick_gain'] * fa, sat=None)

    am = math.radians(P['ring_mark_phase']) - 2.0 * th
    zm = (x0 + Rr * np.cos(am)) + 1j * yg
    m, _ = project(zm, Rr * np.sin(am), 0.0, ph)
    mx = int(m.real); my = int(m.imag)
    xp = mx - 12; yp = my - 12
    if xp >= 0 and yp >= 0 and xp + 25 <= W and yp + 25 <= H:
        px = np.arange(xp, xp + 25)[None, :].astype(np.float64)
        py = np.arange(yp, yp + 25)[:, None].astype(np.float64)
        r = np.sqrt((px - m.real) ** 2 + (py - m.imag) ** 2)
        a = np.clip((P['mark_disc_r'] - r) / P['mark_disc_soft'] + 0.5, 0.0, 1.0) * fa
        patch = img[yp:yp + 25, xp:xp + 25]
        patch[:] = patch * (1 - a[..., None]) + a[..., None] * np.asarray(P['mark_disc_color'], np.float64)
        patch += np.exp(-r ** 2 / (2 * P['mark_sigma'] ** 2))[..., None] * (P['mark_gain'] * fa) * np.asarray(P['mark_color'], np.float64)

# ---------------------------------------------------------------- window + lead line
def draw_window_and_lead(img, scene, f, bg):
    wa = np.clip((f - P['hud']['win_in'][0]) / 38.0, 0, 1) * np.clip((P['hud']['win_out'][1] - f) / 38.0, 0, 1)
    if wa <= 0:
        return
    pen = scene['pen']
    u0 = np.clip((f - P['hud']['win_switch'][0]) / (P['hud']['win_switch'][1] - P['hud']['win_switch'][0]), 0.0, 1.0)
    u = u0 * u0 * (3 - 2 * u0)
    winbg = np.asarray(P['hud']['win_bg'], np.float64) * (1 - u) + np.asarray(P['hud']['win_bg_ride'], np.float64) * u

    tx, ty = tau(f)
    off = np.asarray(P['hud']['pen_view_off'], np.float64) + np.array([tx, ty])
    cv = np.asarray(P['hud']['crank_view'], np.float64)
    cx = (pen.real + off[0]) * (1 - u) + cv[0] * u
    cy = (pen.imag + off[1]) * (1 - u) + cv[1] * u
    zoom = P['hud']['pen_zoom'] * (1 - u) + cv[2] * u
    rect = (0, 25, 260, 210)

    sH, sW = 211, 261
    sub = np.zeros((sH, sW, 3), np.float64) + winbg
    draw_view(sub, scene, f, (cx, cy, zoom, rect), winbg)

    # title background + separator
    sub[0:26] = np.asarray(P['hud']['win_title_bg'], np.float64)
    sub[25] = sub[25] * (1 - P['hud']['win_sep_alpha']) + np.asarray(P['hud']['win_border'], np.float64) * P['hud']['win_sep_alpha']

    # rounded border
    qx = np.arange(sW)[None, :].astype(np.float64)
    qy = np.arange(sH)[:, None].astype(np.float64)
    rr = P['hud']['win_radius']
    dx = np.abs(qx - 130.0) - (130.0 - rr)
    dy = np.abs(qy - 105.0) - (105.0 - rr)
    d = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0)) + np.minimum(np.maximum(dx, dy), 0.0) - rr
    bl = np.clip(P['hud']['win_border_w'] - np.abs(d + P['hud']['win_border_off']), 0.0, 1.0)
    sub = sub * (1 - bl[..., None]) + np.asarray(P['hud']['win_border'], np.float64) * bl[..., None]
    wa_px = wa * np.clip(0.5 - d, 0.0, 1.0)

    # title bar sprite
    spr = AT['wt_31'] if f >= P['ride_start'] else AT['wt_5']
    spr = spr.astype(np.float64).copy()
    if f >= P['ride_start']:
        ds = '%03d' % psiread(f)
        for i in range(3):
            key = 'wdig_%d_%s' % (i, ds[i])
            if key in AT_FILES:
                ws = META['wslots'][i]
                spr[:, ws[0]:ws[1]] = AT[key]
    sub[0:28] = spr

    # lead line
    L0 = cx + 1j * cy
    L1 = 963 + 1j * P['hud']['lead_y']
    nL = int(abs(L1 - L0))
    tj = np.arange(nL) + 0.5
    onj = ((nL - tj) % P['hud']['lead_period']) < P['hud']['lead_on']
    pts = L0 + (L1 - L0) * (tj / nL)
    den = deposit(H, W, pts[onj], np.ones(int(onj.sum())), None)
    stroke(img, den, 0.45, P['hud']['lead_color'], gain=float(wa_px.max()), sat=None)

    x0, y0, x1, y1 = P['hud']['win_rect']
    img[y0:y1 + 1, x0:x1 + 1] = img[y0:y1 + 1, x0:x1 + 1] * (1 - wa_px[..., None]) + sub * wa_px[..., None]

# ---------------------------------------------------------------- atlas text
def apply_atlas(out, f):
    out = out.astype(np.int32)
    name = None
    for (nm, a, b) in META['modes']:
        if a <= f < b:
            name = nm
            break
    d = AT['hud_' + name].astype(np.int32).copy()
    if name in ('ride', '3d'):
        if name == '3d':
            key = 'p2_%d' % pitch_read(f)
            if key in AT_FILES:
                d[51:87] = AT[key]
        ps = psiread(f)
        if name == 'ride':
            digits = ['%03d' % ps]
        else:
            digits = ['%03d' % ps, '%03d' % theta_read(f)]
        for i, ds in enumerate(digits):
            for j, ch in enumerate(ds):
                slot = '%s_%d' % (name, i * 3 + j)
                key = 'dig_%s_%s' % (slot, ch)
                if key in AT_FILES:
                    xa, xb, ya, yb = META['slots'][slot]
                    d[ya - 544:yb - 544, xa - 96:xb - 96] = AT[key]
    out[544:704, 96:1184] += d
    ta = np.clip((f - 3195) / 120.0, 0.0, 1.0)
    if ta > 0:
        out[0:60, 196:1084] += np.round(AT['title'] * ta).astype(np.int32)
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------- background
def make_background():
    NX = P['bg_yuv']['NX']; NY = P['bg_yuv']['NY']
    def basis(n, t):
        xk = np.linspace(0.0, 1.0, n)
        return interpolate.CubicSpline(xk, np.eye(n))(t)
    xs = np.arange(W)
    Bx = basis(NX, np.abs(2 * (xs + 0.5) / W - 1.0))
    ys = np.arange(H)
    By = basis(NY, (ys + 0.5) / H)
    yuv = np.zeros((H, W, 3), np.float64)
    for c in range(3):
        key = ['Y', 'U', 'V'][c]
        nodes = np.asarray(P['bg_yuv'][key], np.float64)   # (NX, NY)
        S = By @ nodes.T @ Bx.T
        yuv[..., c] = S
    yuv[..., 0] -= 16.0
    yuv[..., 1] -= 128.0
    yuv[..., 2] -= 128.0
    bg = yuv @ YUV_INV_T - np.asarray(P['codec_bias'], np.float64)
    for (x, y, aR, aG, aB) in P['stars']:
        ix, iy = int(x) - 2, int(y) - 2
        if ix < 0 or iy < 0 or ix + 5 > W or iy + 5 > H:
            continue
        px = np.arange(ix, ix + 5)[None, :].astype(np.float64)
        py = np.arange(iy, iy + 5)[:, None].astype(np.float64)
        g = np.exp(-((px - x) ** 2 + (py - y) ** 2) / (2 * P['star_sigma'] ** 2))
        bg[iy:iy + 5, ix:ix + 5, 0] += g * aR
        bg[iy:iy + 5, ix:ix + 5, 1] += g * aG
        bg[iy:iy + 5, ix:ix + 5, 2] += g * aB
    return bg

# ---------------------------------------------------------------- render frame
def render_frame(f):
    scene = build_scene(f)
    img = bg.copy()
    if f >= P['f3d']:
        draw_turntable(img, f)
    draw_view(img, scene, f, None, bg)
    draw_window_and_lead(img, scene, f, bg)
    img = img + np.asarray(P['codec_bias'], np.float64)
    if P['snap_y']:
        yuv = img @ YUV_MAT.T
        yuv[..., 0] = np.round(yuv[..., 0])
        img = yuv @ YUV_INV_T
    out = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    out = apply_atlas(out, f)
    return out

# ---------------------------------------------------------------- output
def write_png(arr, path):
    h, w, c = arr.shape
    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    raw = b''
    for yy in range(h):
        raw += b'\x00' + arr[yy].tobytes()
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b'')
    with open(path, 'wb') as fp:
        fp.write(png)

def render_range(A, B, out):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', '60',
           '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
           '-pix_fmt', 'yuv420p', out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in range(A, B):
            frame = render_frame(f)
            proc.stdin.write(frame.tobytes())
            if (f - A) % 120 == 0:
                print('frame', f, flush=True)
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit('ffmpeg failed rc=%d' % rc)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True, help='path to params.json (not distributed)')
    ap.add_argument('--frames', nargs=2, type=int, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    load(args.params)
    A, B = args.frames
    if args.out.endswith('.png'):
        if B - A != 1:
            raise SystemExit('PNG output only for a single frame')
        write_png(render_frame(A), args.out)
        print('wrote', args.out, flush=True)
    else:
        render_range(A, B, args.out)
