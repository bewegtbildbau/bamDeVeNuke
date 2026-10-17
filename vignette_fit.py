"""Numpy-only vignette-from-motion fit for use inside Nuke (no scipy / no torch).

A portable port of the container's classical solver
(``src/baseline/lsq_vignette.py`` + ``src/infer/run_classical.py``). The only
dependency is numpy, so it runs inside Nuke without installing scipy or torch.

Layers:
  * ``fit_vignette_anamorphic_np`` — the solver (checked against scipy), plus
    ``fit_with_uncertainty`` / ``choose_order`` / ``choose_aniso``.
  * ``FrameCache``, ``collect_direct``, ``collect_chain`` — read the plate and a
    pre-rendered flow EXR straight off disk (no per-pixel Nuke sampling) and
    build the log-ratio correspondences.
  * ``solve_from_reads`` (direct fXX layer) / ``solve_from_stride1`` (chained
    f01 layer) — the drivers; ``scan_direct`` / ``scan_chain`` /
    ``detect_flow_v_sign`` — the quick scoring used by Analyse.

Vignette model (gauge V(center)=1, elliptical radius for anamorphic):
    V(r) = 1 + k1 r^2 + k2 r^4 (+ k3 r^6),   r^2 = (a*gx)^2 + (gy/a)^2
    a = exp(log_aniso),  gx,gy = (pixel - center) / half_diagonal
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import os
import sys

import numpy as np

# Pure-Python EXR reader (numpy-only) — the ONLY reader used inside Nuke, since
# native EXR libs (pip OpenEXR / OpenImageIO) clash with Nuke's bundled copies
# and segfault. Sits next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import exr_read
except Exception:  # pragma: no cover
    exr_read = None

AP1_TO_Y = np.array([0.2722287168, 0.6740817658, 0.0536895174], dtype=np.float32)


def _in_nuke() -> bool:
    return "nuke" in sys.modules


# --------------------------------------------------------------------------- #
# model helpers
# --------------------------------------------------------------------------- #
def _poly_gain(r2: np.ndarray, k) -> np.ndarray:
    return 1.0 + k[0] * r2 + k[1] * r2 * r2 + k[2] * r2 * r2 * r2


def _norm_coords(x, y, height, width):
    cx0 = (width - 1) / 2.0
    cy0 = (height - 1) / 2.0
    hd = 0.5 * np.hypot(width, height)
    return (x - cx0) / hd, (y - cy0) / hd


# --------------------------------------------------------------------------- #
# robust Levenberg-Marquardt (soft-L1 via IRLS) — pure numpy
# --------------------------------------------------------------------------- #
def _soft_l1_cost(r: np.ndarray, f: float) -> float:
    z = (r / f) ** 2
    return float(np.sum(f * f * (np.sqrt(1.0 + z) - 1.0)))


def _num_jac(resid: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, r0: np.ndarray) -> np.ndarray:
    n = theta.size
    J = np.empty((r0.size, n), dtype=np.float64)
    for j in range(n):
        eps = 1e-6 + 1e-6 * abs(theta[j])
        tp = theta.copy()
        tp[j] += eps
        J[:, j] = (resid(tp) - r0) / eps
    return J


def robust_lm(
    resid: Callable[[np.ndarray], np.ndarray],
    n_params: int,
    f_scale: float = 0.05,
    max_iter: int = 80,
    tol: float = 1e-8,
) -> np.ndarray:
    theta = np.zeros(n_params, dtype=np.float64)
    cost = _soft_l1_cost(resid(theta), f_scale)
    lam = 1e-3
    for _ in range(max_iter):
        r = resid(theta)
        J = _num_jac(resid, theta, r)
        # soft-L1 IRLS weight is rho'(z) = 1/sqrt(1+z); rows are scaled by its
        # square root so J^T W J / J^T W r are the true Gauss-Newton terms
        w = (1.0 + (r / f_scale) ** 2) ** -0.25
        Jw = J * w[:, None]
        rw = r * w
        A = Jw.T @ Jw
        g = Jw.T @ rw
        improved = False
        for _inner in range(30):
            H = A + lam * (np.diag(np.diag(A)) + 1e-12 * np.eye(n_params))
            try:
                step = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            new = theta + step
            cnew = _soft_l1_cost(resid(new), f_scale)
            if cnew < cost:
                theta, cost = new, cnew
                lam = max(lam * 0.5, 1e-12)
                improved = True
                break
            lam *= 4.0
            if lam > 1e10:
                break
        if not improved or np.linalg.norm(step) < tol:
            break
    return theta


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #
def _make_resid(xp, yp, xq, yq, d, height, width, fit_aniso, order, reg_strength, floor):
    """Residual function for the fit: data rows first (``d.size`` of them), then the prior."""
    gxp, gyp = _norm_coords(np.asarray(xp, float), np.asarray(yp, float), height, width)
    gxq, gyq = _norm_coords(np.asarray(xq, float), np.asarray(yq, float), height, width)
    d = np.asarray(d, dtype=np.float64)

    rho = np.linspace(0.0, 1.0, 33)
    t = rho * rho
    n_data = max(d.size, 1)
    pen = np.sqrt(reg_strength * n_data / rho.size)

    def unpack(theta):
        k = [theta[0], theta[1], theta[2] if order >= 3 else 0.0]
        la = float(theta[order]) if fit_aniso else 0.0
        return k, la

    def resid(theta: np.ndarray) -> np.ndarray:
        k, la = unpack(theta)
        a = np.exp(la)
        r2p = (a * gxp) ** 2 + (gyp / a) ** 2
        r2q = (a * gxq) ** 2 + (gyq / a) ** 2
        gp = np.clip(_poly_gain(r2p, k), 1e-3, None)
        gq = np.clip(_poly_gain(r2q, k), 1e-3, None)
        data = (np.log(gq) - np.log(gp)) - d

        g = 1.0 + k[0] * t + k[1] * t * t + k[2] * t ** 3
        dgdrho = 2 * k[0] * rho + 4 * k[1] * rho ** 3 + 6 * k[2] * rho ** 5
        mono = pen * np.maximum(0.0, dgdrho)
        flo = pen * np.maximum(0.0, floor - g)
        l2 = pen * np.array([0.5 * k[2], 0.5 * la])
        return np.concatenate([data, mono, flo, l2])

    return resid, unpack, d.size


def fit_vignette_anamorphic_np(
    xp, yp, xq, yq, d, height, width,
    fit_aniso: bool = True,
    order: int = 2,
    reg_strength: float = 0.5,
    floor: float = 0.08,
    f_scale: float = 0.05,
) -> dict:
    """Fit (k1,k2[,k3])[,log_aniso] from flow correspondences + log-ratios.

    Mirrors the scipy version: robust soft-L1 data term + monotonic/floor/L2
    physical prior, elliptical radius, center fixed at frame center.
    """
    resid, unpack, n_data = _make_resid(xp, yp, xq, yq, d, height, width,
                                        fit_aniso, order, reg_strength, floor)
    nparam = order + (1 if fit_aniso else 0)
    theta = robust_lm(resid, nparam, f_scale=f_scale)
    k, la = unpack(theta)
    return {
        "k1": float(k[0]), "k2": float(k[1]), "k3": float(k[2]),
        "log_aniso": float(la), "n_constraints": int(n_data),
    }


def fit_with_uncertainty(
    xp, yp, xq, yq, d, height, width,
    sid=None,
    order: int = 2,
    reg_strength: float = 0.5,
    fit_aniso: bool = True,
    floor: float = 0.08,
) -> dict:
    """Fit, then predict how well the data pins down the vignette.

    ``sigma_v`` is the predicted 1-sigma error of V in the frame corner, relative
    to V there: ``Cov = s^2 (J^T J)^-1`` from the data rows only (the prior would
    make weak data look confident), with ``s`` a robust residual scale measured
    about 0, so flow drift and lighting/exposure changes count as noise.
    ``sid`` (one id per start pixel) accounts for several offsets sharing a start
    point: they are correlated, so the covariance is scaled by rows / unique ids.
    """
    res = fit_vignette_anamorphic_np(xp, yp, xq, yq, d, height, width, fit_aniso=fit_aniso,
                                     order=order, reg_strength=reg_strength, floor=floor)
    resid, unpack, n = _make_resid(xp, yp, xq, yq, d, height, width,
                                   fit_aniso, order, reg_strength, floor)
    theta = _theta(res, order, fit_aniso)
    r0 = resid(theta)
    J = _num_jac(resid, theta, r0)[:n]
    sigma = 1.4826 * float(np.median(np.abs(r0[:n])))
    A = J.T @ J
    A += (1e-12 * np.trace(A) + 1e-30) * np.eye(theta.size)
    cov = sigma * sigma * np.linalg.inv(A)
    if sid is not None and n:
        cov *= n / float(np.unique(sid).size)

    def v_corner(th):
        k, la = unpack(th)
        return _v_corner(k, la, height, width)

    vc = v_corner(theta)
    g = np.empty(theta.size)
    for j in range(theta.size):
        tp = theta.copy()
        tp[j] += 1e-6
        g[j] = (v_corner(tp) - vc) / 1e-6
    gxp, gyp = _norm_coords(np.asarray(xp, float), np.asarray(yp, float), height, width)
    gxq, gyq = _norm_coords(np.asarray(xq, float), np.asarray(yq, float), height, width)
    radial = np.abs((gxq ** 2 + gyq ** 2) - (gxp ** 2 + gyp ** 2))
    res.update({
        "sigma": sigma,
        "sigma_v": float(np.sqrt(max(g @ cov @ g, 0.0))) / max(vc, 1e-3),
        "v_corner": vc,
        "radial": float(np.median(radial)) if n else 0.0,
        "n_points": int(n),
    })
    return res


def _theta(res, order, fit_aniso=True):
    """Parameter vector (as the solver packs it) from a fit result dict."""
    return np.array([res["k1"], res["k2"]] + ([res["k3"]] if order >= 3 else [])
                    + ([res["log_aniso"]] if fit_aniso else []))


def _v_corner(k, la, height, width):
    """V in a frame corner (the largest radius the fit ever sees)."""
    gxc, gyc = _norm_coords(0.0, 0.0, height, width)
    a = np.exp(la)
    return float(_poly_gain((a * gxc) ** 2 + (gyc / a) ** 2, k))


def _v_edges(k, la, height, width):
    """V at a corner, the middle of the left edge and the middle of the top edge."""
    gx, gy = _norm_coords(np.array([0.0, 0.0, (width - 1) / 2.0]),
                          np.array([0.0, (height - 1) / 2.0, 0.0]), height, width)
    a = np.exp(la)
    return _poly_gain((a * gx) ** 2 + (gy / a) ** 2, k)


def _heldout_compare(corr, height, width, simple, full, sigma_v, reg_strength=0.5,
                     min_gain=0.03, f_scale=0.05) -> Tuple[Optional[bool], dict]:
    """Does the ``full`` model (a dict of fit kwargs: order, fit_aniso) beat ``simple``?

    Hold-out validation: the start frames are split into two halves; each model is
    fitted on one half and scored (soft-L1 cost, as the fit uses) on the other,
    both ways round. A term that only fits noise does not predict unseen frames
    better, so ``full`` wins only if its held-out cost is at least ``min_gain``
    lower AND the two full-data fits differ at the frame edge (corner, side, top)
    by more than twice ``sigma_v`` (the predicted corner error), i.e. the extra
    term actually changes the result. Returns (None, info) if it can't be tested.
    """
    start = corr["sid"].astype(np.int64) // (height * width)
    ids = np.unique(start)
    if ids.size < 2:
        return None, {"reason": "needs at least two start frames"}
    half = np.isin(start, ids[0::2])

    def sub(m):
        return {k: v[m] for k, v in corr.items()}

    def fit(c, cfg):
        return fit_vignette_anamorphic_np(c["xp"], c["yp"], c["xq"], c["yq"], c["d"], height, width,
                                          reg_strength=reg_strength, **cfg)

    cost, fits = {}, {}
    for name, cfg in (("simple", simple), ("full", full)):
        total = 0.0
        for train, test in ((half, ~half), (~half, half)):
            tr, te = sub(train), sub(test)
            res = fit(tr, cfg)
            resid, _, n = _make_resid(te["xp"], te["yp"], te["xq"], te["yq"], te["d"], height, width,
                                      cfg["fit_aniso"], cfg["order"], reg_strength, 0.08)
            total += _soft_l1_cost(resid(_theta(res, cfg["order"], cfg["fit_aniso"]))[:n], f_scale)
        cost[name] = total / max(1, corr["d"].size)
        fits[name] = fit(corr, cfg)
    ve = {n: _v_edges([r["k1"], r["k2"], r["k3"]], r["log_aniso"], height, width)
          for n, r in fits.items()}
    diff = float(np.max(np.abs(ve["full"] - ve["simple"]) / np.maximum(ve["simple"], 1e-3)))
    use = cost["full"] < (1.0 - min_gain) * cost["simple"] and diff > 2.0 * sigma_v
    return bool(use), {"cost_simple": cost["simple"], "cost_full": cost["full"], "diff": diff,
                       "fit_full": fits["full"]}


def choose_order(corr, height, width, sigma_v, fit_aniso=True, reg_strength=0.5,
                 min_gain=0.03, f_scale=0.05) -> Tuple[int, dict]:
    """Order 2 (k1,k2) or 3 (+k3), by hold-out validation (see ``_heldout_compare``)."""
    use, info = _heldout_compare(corr, height, width, {"order": 2, "fit_aniso": fit_aniso},
                                 {"order": 3, "fit_aniso": fit_aniso}, sigma_v, reg_strength,
                                 min_gain, f_scale)
    if use is None:
        return 2, info
    return (3 if use else 2), {"cost2": info["cost_simple"], "cost3": info["cost_full"],
                               "corner_diff": info["diff"], "k3": info["fit_full"]["k3"]}


def choose_aniso(corr, height, width, sigma_v, order=2, reg_strength=0.5,
                 min_gain=0.03, f_scale=0.05) -> Tuple[Optional[bool], dict]:
    """Anamorphic (fit log_aniso) or spherical (log_aniso = 0), by hold-out validation.

    Returns (None, info) if it can't be tested; keep the current setting then.
    """
    use, info = _heldout_compare(corr, height, width, {"order": order, "fit_aniso": False},
                                 {"order": order, "fit_aniso": True}, sigma_v, reg_strength,
                                 min_gain, f_scale)
    if use is None:
        return None, info
    return use, {"cost_sph": info["cost_simple"], "cost_ana": info["cost_full"],
                 "edge_diff": info["diff"], "log_aniso": info["fit_full"]["log_aniso"]}


# --------------------------------------------------------------------------- #
# EXR reading: pure-Python exr_read (the only reader inside Nuke); outside Nuke
# OpenImageIO -> OpenEXR -> OpenCV as fallbacks for files exr_read can't decode
# --------------------------------------------------------------------------- #
def _read_rgb_np(path: str) -> np.ndarray:
    """Read an EXR as (H,W,3) float32 RGB, top-row-first (numpy convention).

    Pure-Python reader first (numpy-only, safe in Nuke; returns the display
    window, NaN outside the data window). Falls back to native readers only
    OUTSIDE Nuke (those return the data window).
    """
    # 0) Pure-Python (Nuke-safe)
    if exr_read is None and _in_nuke():
        raise ImportError(
            "exr_read.py not importable — ensure it sits next to vignette_fit.py on NUKE_PATH."
        )
    if exr_read is not None:
        try:
            names = exr_read.list_channels(path)
            want = [c for c in ("R", "G", "B") if c in names] or names[:1]
            ch, _, _ = exr_read.read_channels(path, want)
            arrs = [ch[c] for c in want]
            if len(arrs) >= 3:
                return np.ascontiguousarray(np.stack(arrs[:3], -1))
            return np.ascontiguousarray(arrs[0])
        except Exception:
            if _in_nuke():
                raise  # never load native EXR libs inside Nuke
    # 1) OpenImageIO
    try:
        import OpenImageIO as oiio  # facility-standard, reads to numpy
        buf = oiio.ImageBuf(path)
        arr = buf.get_pixels(oiio.FLOAT)  # (H, W, C), row 0 = top
        return np.ascontiguousarray(arr[:, :, :3], dtype=np.float32)
    except Exception:
        pass
    # 2) OpenEXR (ASWF PyPI package — easiest to add to Nuke's Python)
    try:
        import OpenEXR
        chans = OpenEXR.File(path).channels()

        def get(nm):
            return np.asarray(chans[nm].pixels, dtype=np.float32)

        names = list(chans.keys())
        if all(c in chans for c in ("R", "G", "B")):
            return np.ascontiguousarray(np.stack([get("R"), get("G"), get("B")], -1))
        return np.ascontiguousarray(get(names[0]))  # single channel (e.g. luminance)
    except Exception:
        pass
    # 3) OpenCV
    try:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(path)
        img = img.astype(np.float32)
        if img.ndim == 3 and img.shape[2] >= 3:
            img = img[:, :, :3][:, :, ::-1]  # BGR->RGB
        return np.ascontiguousarray(img)
    except ImportError:
        pass
    raise ImportError(
        "No EXR reader available. Install one into this Python (Nuke bundles none), "
        "e.g.  <nuke_python> -m pip install --target <dir> OpenEXR  and add <dir> to "
        "PYTHONPATH. OpenImageIO or opencv-python-headless also work."
    )


def _match_channel(names, want: str):
    """Index of the channel named ``want`` or ending in ``want`` (layer prefix).

    Lets the caller pass just ``n_u`` and match ``smartvector_f01_v01.n_u``.
    """
    if want in names:
        return names.index(want)
    for i, nm in enumerate(names):
        if nm.endswith("." + want) or nm.endswith(want):
            return i
    return None


def _read_flow_np(path: str, u_channel: str = "", v_channel: str = "") -> np.ndarray:
    """Read a flow EXR as (H,W,2) pixel displacement (u=x, v=y), top-row-first.

    u_channel/v_channel may be full names or suffixes (e.g. ``n_u`` matches
    ``smartvector_f01_v01.n_u``). Pure-Python reader first (Nuke-safe), then
    native readers outside Nuke.
    """
    # 0) Pure-Python (Nuke-safe), selects channels by name/suffix
    if exr_read is None and _in_nuke():
        raise ImportError(
            "exr_read.py not importable — ensure it sits next to vignette_fit.py on NUKE_PATH."
        )
    if exr_read is not None:
        try:
            ch, _, _ = exr_read.read_channels(path, [u_channel, v_channel])
            if u_channel in ch and v_channel in ch:
                return np.ascontiguousarray(np.stack([ch[u_channel], ch[v_channel]], -1), dtype=np.float32)
            if _in_nuke():
                # Channel names didn't resolve — raise before falling through to native libs,
                # which segfault Nuke (see README). Give actionable channel list.
                available = exr_read.list_channels(path)
                raise RuntimeError(
                    "Flow channels %r / %r not found in:\n  %s\n"
                    "Available channels: %s\n"
                    "Check the n_u / n_v / p_u / p_v channel name knobs."
                    % (u_channel, v_channel, path, available)
                )
        except Exception:
            if _in_nuke():
                raise
    # OpenImageIO — reads channels by name (facility standard)
    try:
        import OpenImageIO as oiio
        buf = oiio.ImageBuf(path)
        names = list(buf.spec().channelnames)
        arr = buf.get_pixels(oiio.FLOAT)  # (H,W,C)
        iu = _match_channel(names, u_channel) if u_channel else 0
        iv = _match_channel(names, v_channel) if v_channel else 1
        if iu is None or iv is None:
            iu, iv = 0, 1
        return np.ascontiguousarray(np.stack([arr[:, :, iu], arr[:, :, iv]], -1), dtype=np.float32)
    except Exception:
        pass
    # OpenEXR — also selects channels by name
    try:
        import OpenEXR
        chans = OpenEXR.File(path).channels()
        names = list(chans.keys())
        iu = u_channel if u_channel in names else (
            next((n for n in names if n.endswith(u_channel)), None) if u_channel else names[0])
        iv = v_channel if v_channel in names else (
            next((n for n in names if n.endswith(v_channel)), None) if v_channel else names[min(1, len(names) - 1)])
        u = np.asarray(chans[iu].pixels, dtype=np.float32)
        v = np.asarray(chans[iv].pixels, dtype=np.float32)
        return np.ascontiguousarray(np.stack([u, v], -1), dtype=np.float32)
    except Exception:
        pass
    # OpenCV fallback — cannot select by name, uses first two channels
    img = _read_rgb_np(path)
    return np.ascontiguousarray(img[:, :, :2], dtype=np.float32)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    return (rgb[:, :, :3] @ AP1_TO_Y).astype(np.float32)


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = np.where(finite, xs, -1.0)              # NaN tracks -> out of bounds
    ys = np.where(finite, ys, -1.0)
    x0 = np.floor(xs).astype(np.int64); y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1; y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    xc0 = np.clip(x0, 0, w - 1); xc1 = np.clip(x1, 0, w - 1)
    yc0 = np.clip(y0, 0, h - 1); yc1 = np.clip(y1, 0, h - 1)
    wx = xs - x0; wy = ys - y0
    top = img[yc0, xc0] * (1 - wx) + img[yc0, xc1] * wx
    bot = img[yc1, xc0] * (1 - wx) + img[yc1, xc1] * wx
    return np.where(valid, top * (1 - wy) + bot * wy, np.nan)



# --------------------------------------------------------------------------- #
# Frame cache: downsampled plate luminance + flow, each EXR decoded once
# --------------------------------------------------------------------------- #
class Cancelled(Exception):
    """Raised when a progress callback asks to stop."""


def _tick(progress, msg):
    if progress is not None and progress(msg):
        raise Cancelled(msg)


def _block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Numpy-only area downsample by an integer factor (no cv2 needed)."""
    if factor <= 1:
        return arr
    h, w = arr.shape[:2]
    hc, wc = (h // factor) * factor, (w // factor) * factor
    a = arr[:hc, :wc]
    if a.ndim == 2:
        return a.reshape(hc // factor, factor, wc // factor, factor).mean((1, 3))
    return a.reshape(hc // factor, factor, wc // factor, factor, a.shape[2]).mean((1, 3))


def _read_flow_layers(path: str, layers: dict, required: str) -> dict:
    """Read several flow layers from one EXR with a single decode.

    ``layers`` is ``{name: (n_u, n_v, p_u, p_v)}`` (p_* may be empty). Returns
    ``{name: (fwd (H,W,2), bwd (H,W,2) or None)}`` for every layer found; only a
    missing ``required`` layer is an error.
    """
    if exr_read is not None:
        wanted = [c for chs in layers.values() for c in chs if c]
        try:
            ch, _, _ = exr_read.read_channels(path, wanted)
        except Exception:
            if _in_nuke():
                raise
            ch = None
        if ch is not None:
            out = {}
            for name, (nu, nv, pu, pv) in layers.items():
                if nu in ch and nv in ch and (not (pu and pv) or (pu in ch and pv in ch)):
                    fwd = np.stack([ch[nu], ch[nv]], -1)
                    bwd = np.stack([ch[pu], ch[pv]], -1) if pu and pv else None
                    out[name] = (fwd, bwd)
            if required in out:
                return out
            if _in_nuke():
                nu, nv, pu, pv = layers[required]
                available = exr_read.list_channels(path)
                raise RuntimeError(
                    "Flow channels %s not found in:\n  %s\nAvailable channels: %s\n"
                    "Check the n_u / n_v / p_u / p_v channel name knobs."
                    % ([c for c in (nu, nv, pu, pv) if c], path, available))
    nu, nv, pu, pv = layers[required]
    fwd = _read_flow_np(path, nu, nv)
    bwd = _read_flow_np(path, pu, pv) if pu and pv else None
    return {required: (fwd, bwd)}


class FrameCache:
    """Plate luminance + flow per frame, block-mean downsampled to ``long_side``.

    ``layers`` maps a layer name to its ``(n_u, n_v, p_u, p_v)`` channels. A flow
    EXR is decoded once for every layer that lives on that frame (per
    ``layer_frames``), so candidates sharing a frame share the read. The least
    recently used frames are dropped beyond ``max_frames``. Flow is returned in
    downsampled pixels (divided by the factor).
    """

    def __init__(self, plate_path_fn, flow_path_fn, layers, long_side=768,
                 layer_frames=None, max_frames=16):
        from collections import OrderedDict
        self.plate_path_fn = plate_path_fn
        self.flow_path_fn = flow_path_fn
        self.layers = dict(layers)
        self.layer_frames = {k: set(v) for k, v in (layer_frames or {}).items()}
        self.long_side = long_side
        self.max_frames = max_frames
        self.fac = None
        self.full_shape = None
        self.reads = 0
        self._lum = OrderedDict()
        self._flow = OrderedDict()

    def factor_for(self, long_side):
        if not long_side or self.full_shape is None:
            return 1
        return max(1, round(max(self.full_shape) / long_side))

    def _set_shape(self, shape):
        if self.fac is None:
            self.full_shape = tuple(shape[:2])
            self.fac = self.factor_for(self.long_side)

    def _put(self, od, key, val):
        od[key] = val
        od.move_to_end(key)
        while len(od) > self.max_frames:
            od.popitem(last=False)

    def lum(self, fr):
        if fr in self._lum:
            self._lum.move_to_end(fr)
            return self._lum[fr]
        img = _luminance(_read_rgb_np(self.plate_path_fn(fr)))
        self._set_shape(img.shape)
        img = _block_mean(img, self.fac)
        self.reads += 1
        self._put(self._lum, fr, img)
        return img

    def flow(self, fr, layer):
        got = self._flow.get(fr)
        if got is not None and layer in got:
            self._flow.move_to_end(fr)
            return got[layer]
        want = {L: chs for L, chs in self.layers.items()
                if L == layer or fr in self.layer_frames.get(L, ())}
        raw = _read_flow_layers(self.flow_path_fn(fr), want, layer)
        self._set_shape(next(iter(raw.values()))[0].shape)
        got = dict(got or {})
        for L, (fwd, bwd) in raw.items():
            got[L] = (_block_mean(fwd, self.fac) / self.fac,
                      None if bwd is None else _block_mean(bwd, self.fac) / self.fac)
        self.reads += 1
        self._put(self._flow, fr, got)
        return got[layer]


# --------------------------------------------------------------------------- #
# correspondences: (p in frame f, q in frame f+offset, log-ratio d)
# --------------------------------------------------------------------------- #
def _grid(h, w):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    return xs.ravel(), ys.ravel()


class _Corr:
    KEYS = ("xp", "yp", "xq", "yq", "d", "sid", "off")

    def __init__(self):
        self.parts = {k: [] for k in self.KEYS}

    def add(self, xp, yp, xq, yq, d, sid, off):
        for k, v in zip(self.KEYS, (xp, yp, xq, yq, d, sid, np.full(sid.size, off))):
            self.parts[k].append(v)

    def arrays(self):
        return {k: (np.concatenate(v) if v else np.empty(0)) for k, v in self.parts.items()}


def chain_offsets(stride: int) -> List[int]:
    """Baselines a chain of length ``stride`` emits correspondences at."""
    return sorted({min(stride, max(4, stride // 2)), stride})


def collect_direct(src: FrameCache, layer, starts, stride, flow_v_sign=-1.0, fb_thresh=1.5,
                   points=30000, eps=1e-4, rng=None, bwd_fn=None, progress=None):
    """Pairs (f, f+stride) from a flow layer whose n_* at f points to f+stride.

    The backward check uses p_* at f+stride (``bwd_fn(f)`` if the layer has none).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    c = _Corr()
    h = w = None
    for i, f in enumerate(starts):
        _tick(progress, "pair %d/%d  (%d -> %d)" % (i + 1, len(starts), f, f + stride))
        lum0 = src.lum(f)
        lum1 = src.lum(f + stride)
        fwd = src.flow(f, layer)[0]
        h, w = lum0.shape
        xr, yr = _grid(h, w)
        u = fwd[:, :, 0].ravel()
        v = flow_v_sign * fwd[:, :, 1].ravel()
        qx, qy = xr + u, yr + v
        i0 = _bilinear(lum0, xr, yr)
        i1 = _bilinear(lum1, qx, qy)
        ok = np.isfinite(i0) & np.isfinite(i1) & (i0 > eps) & (i1 > eps)

        fb = src.flow(f + stride, layer)[1]
        if fb is None and bwd_fn is not None:
            fb = bwd_fn(f)
        if fb is not None:
            bu = _bilinear(fb[:, :, 0], qx, qy)
            bv = flow_v_sign * _bilinear(fb[:, :, 1], qx, qy)
            err = np.hypot(u + bu, v + bv)          # fwd(p) + bwd(q) ~ 0
            ok &= np.isfinite(err) & (err < fb_thresh)

        idx = np.flatnonzero(ok)
        if idx.size > points:
            idx = rng.choice(idx, points, replace=False)
        c.add(xr[idx], yr[idx], qx[idx], qy[idx], np.log(i1[idx]) - np.log(i0[idx]),
              i * h * w + idx, stride)
    return c.arrays(), (h, w)


def collect_chain(src: FrameCache, layer, starts, offsets, flow_v_sign=-1.0, fb_thresh=1.0,
                  points=20000, eps=1e-4, rng=None, progress=None):
    """Walk stride-1 flow forward from each start frame, emitting at ``offsets``.

    Each step samples the next frame's n_* at the tracked position; p_* at the new
    frame gives a per-step forward/backward check that drops a point for good the
    moment its track drifts or occludes.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    offsets = sorted(set(int(o) for o in offsets))
    max_off = max(offsets)
    c = _Corr()
    h = w = None
    for i, f in enumerate(starts):
        lum0 = src.lum(f)
        h, w = lum0.shape
        px, py = _grid(h, w)                     # tracked positions
        p0x, p0y = px.copy(), py.copy()          # fixed start positions
        valid = np.ones(px.shape, dtype=bool)
        i0 = _bilinear(lum0, p0x, p0y)

        for k in range(1, max_off + 1):
            _tick(progress, "chain %d/%d  frame %d" % (i + 1, len(starts), f + k))
            fr = f + k - 1
            n_fwd = src.flow(fr, layer)[0]
            du = _bilinear(n_fwd[:, :, 0], px, py)
            dv = flow_v_sign * _bilinear(n_fwd[:, :, 1], px, py)
            nx, ny = px + du, py + dv
            valid &= (np.isfinite(nx) & np.isfinite(ny)
                      & (nx >= 0) & (nx <= w - 1) & (ny >= 0) & (ny <= h - 1))
            p_bwd = src.flow(fr + 1, layer)[1]
            if p_bwd is not None:
                pu = _bilinear(p_bwd[:, :, 0], nx, ny)
                pv = flow_v_sign * _bilinear(p_bwd[:, :, 1], nx, ny)
                step_err = np.hypot(du + pu, dv + pv)
                valid &= np.isfinite(step_err) & (step_err < fb_thresh)
            px, py = nx, ny

            if k in offsets:
                i1 = _bilinear(src.lum(f + k), px, py)
                ok = valid & np.isfinite(i0) & np.isfinite(i1) & (i0 > eps) & (i1 > eps)
                idx = np.flatnonzero(ok)
                if idx.size > points:
                    idx = rng.choice(idx, points, replace=False)
                c.add(p0x[idx], p0y[idx], px[idx], py[idx], np.log(i1[idx]) - np.log(i0[idx]),
                      i * h * w + idx, k)
    return c.arrays(), (h, w)


def _fit_corr(corr, h, w, order, reg_strength, fit_aniso=True):
    if corr["d"].size == 0:
        raise RuntimeError(
            "No usable correspondences: every point was rejected. Check the flow v sign, "
            "the fb thresh and that the flow channels belong to this plate.")
    return fit_vignette_anamorphic_np(corr["xp"], corr["yp"], corr["xq"], corr["yq"], corr["d"],
                                      h, w, fit_aniso=fit_aniso, order=order, reg_strength=reg_strength)


# --------------------------------------------------------------------------- #
# Nuke drivers: read plate + pre-rendered flow off disk, build & fit
# --------------------------------------------------------------------------- #
def solve_from_reads(
    plate_path_fn: Callable[[int], str],
    flow_path_fn: Callable[[int], str],
    frames: List[int],
    stride: int,
    flow_bwd_path_fn: Optional[Callable[[int], str]] = None,
    fb_thresh: float = 1.5,
    flow_v_sign: float = -1.0,       # Nuke y-up vs numpy row order (see notes)
    points_per_frame: int = 30000,
    eps: float = 1e-4,
    seed: int = 0,
    order: int = 2,
    reg_strength: float = 0.5,
    long_side: int = 768,
    u_channel: str = "n_u",
    v_channel: str = "n_v",
    bwd_u_channel: str = "p_u",
    bwd_v_channel: str = "p_v",
    progress: Optional[Callable[[str], bool]] = None,
    fit_aniso: bool = True,
) -> dict:
    """Solve the vignette from pre-rendered plate + flow EXR sequences.

    ``plate_path_fn(frame)`` / ``flow_path_fn(frame)`` return the EXR path for a
    frame (Nuke: ``read['file'].evaluate()`` at that frame).

    Flow convention (RAFT / SmartVector): one EXR carries motion to the **next**
    frame in ``.n_u/.n_v`` and to the **previous** frame in ``.p_u/.p_v``. So the
    forward flow for a pair (f, f+stride) is ``n_*`` at frame ``f``; the backward
    flow (f+stride -> f) is ``p_*`` at frame ``f+stride`` — both read from the
    same sequence. The backward read enables the forward/backward consistency
    mask (rejects occluded / mis-tracked pixels — strongly recommended).

    IMPORTANT: n_/p_ vectors are relative to a fixed temporal gap. For stride>1
    you must render the vectors at that same gap (``n`` = f -> f+stride), or the
    ``p`` at f+stride will not point back to f. See the README.

    If ``bwd_*_channel`` are empty, falls back to ``flow_bwd_path_fn`` (separate
    backward-flow file) or no confidence mask.

    ``flow_v_sign`` handles the y-orientation mismatch: Nuke motion vectors are
    y-up, numpy arrays are row-0-at-top, so the v component is typically negated.
    Verify with a one-frame warp check before trusting the sign.

    ``progress(msg)`` is called per pair; returning True raises ``Cancelled``.
    ``fit_aniso=False`` fits a spherical (circular) vignette: log_aniso stays 0.
    """
    src = FrameCache(plate_path_fn, flow_path_fn,
                     {"flow": (u_channel, v_channel, bwd_u_channel, bwd_v_channel)},
                     long_side=long_side, max_frames=4)
    bwd_fn = None
    if not (bwd_u_channel and bwd_v_channel) and flow_bwd_path_fn is not None:
        def bwd_fn(f):
            return _block_mean(_read_flow_np(flow_bwd_path_fn(f), u_channel, v_channel), src.fac) / src.fac
    corr, (h, w) = collect_direct(src, "flow", frames, stride, flow_v_sign, fb_thresh,
                                  points_per_frame, eps, np.random.default_rng(seed), bwd_fn, progress)
    res = _fit_corr(corr, h, w, order, reg_strength, fit_aniso)
    res["resolution"] = [h, w]
    return res


def solve_from_stride1(
    plate_path_fn: Callable[[int], str],
    flow_path_fn: Callable[[int], str],
    frames: List[int],
    stride: int = 48,
    offsets: Optional[List[int]] = None,
    long_side: int = 640,
    flow_v_sign: float = -1.0,
    fb_thresh: float = 1.0,
    points_per_offset: int = 20000,
    eps: float = 1e-4,
    seed: int = 0,
    order: int = 2,
    reg_strength: float = 0.5,
    n_u_channel: str = "n_u",
    n_v_channel: str = "n_v",
    p_u_channel: str = "p_u",
    p_v_channel: str = "p_v",
    progress: Optional[Callable[[str], bool]] = None,
    fit_aniso: bool = True,
) -> dict:
    """Solve the vignette from **stride-1** (neighbour-frame) flow by chaining.

    RAFT/SmartVector is rendered once at stride 1 (``.n_*`` = f->f+1, ``.p_*`` =
    f->f-1). For each start frame we walk the chain forward, sampling each next
    frame's ``.n_*`` at the current warped position, accumulating a large
    baseline. Correspondences are emitted at each of ``offsets`` (default
    ``chain_offsets(stride)``), so a single chain yields several baselines. A
    per-step forward/backward check (``.p_*`` at the new position) drops a point
    the moment its track drifts or occludes.

    ``long_side`` downsamples reads (numpy block-mean) — the vignette is
    low-frequency, so this is lossless for the fit and keeps memory/time sane.
    Frames are cached across start frames, so overlapping chains read each EXR once.
    ``fit_aniso=False`` fits a spherical (circular) vignette: log_aniso stays 0.
    """
    if offsets is None:
        offsets = chain_offsets(stride)
    src = FrameCache(plate_path_fn, flow_path_fn,
                     {"flow": (n_u_channel, n_v_channel, p_u_channel, p_v_channel)},
                     long_side=long_side, max_frames=max(offsets) + 2)
    corr, (h, w) = collect_chain(src, "flow", sorted(frames), offsets, flow_v_sign, fb_thresh,
                                 points_per_offset, eps, np.random.default_rng(seed), progress)
    res = _fit_corr(corr, h, w, order, reg_strength, fit_aniso)
    res["resolution"] = [h, w]
    return res


# --------------------------------------------------------------------------- #
# auto setup: flow v sign + scoring candidate flow sources / strides
# --------------------------------------------------------------------------- #
def detect_flow_v_sign(src: FrameCache, layer, starts, gap=1, steps=1, eps=1e-4,
                       progress=None):
    """Find the sign of the flow's v component by which one lines the frames up.

    Every pixel of a start frame is pushed ``steps`` hops of ``gap`` frames along
    the flow, once with v as stored and once negated; the right sign lands on the
    same surface point, so the log brightness difference to the target frame is
    smaller. Returns ``(sign, info)``; sign is None when it can't be told apart
    (too little vertical motion, or no clear winner).
    """
    errs = {1.0: [], -1.0: []}
    vdisp = []
    for f in starts:
        lum0 = src.lum(f)
        h, w = lum0.shape
        xr, yr = _grid(h, w)
        i0 = lum0.ravel()
        for s in (1.0, -1.0):
            _tick(progress, "flow v sign  frame %d" % f)
            px, py = xr.copy(), yr.copy()
            for k in range(steps):
                fwd = src.flow(f + k * gap, layer)[0]
                du = _bilinear(fwd[:, :, 0], px, py)
                dv = s * _bilinear(fwd[:, :, 1], px, py)
                px, py = px + du, py + dv
            i1 = _bilinear(src.lum(f + steps * gap), px, py)
            ok = np.isfinite(i1) & (i1 > eps) & (i0 > eps)
            errs[s].append(np.abs(np.log(i1[ok]) - np.log(i0[ok])))
            if s == 1.0:
                vdisp.append(np.abs(py - yr)[np.isfinite(py)])
    e = {s: (float(np.median(np.concatenate(v))) if sum(a.size for a in v) else np.inf)
         for s, v in errs.items()}
    vd = float(np.median(np.concatenate(vdisp))) if vdisp else 0.0
    lo = min(e, key=e.get)
    decided = vd >= 0.5 and e[lo] < 0.9 * e[-lo]
    return (lo if decided else None), {"err_pos": e[1.0], "err_neg": e[-1.0], "v_disp": vd}


MIN_SCAN_POINTS = 1000


def _score(corr, h, w, order, reg_strength):
    n = int(corr["d"].size)
    if n < MIN_SCAN_POINTS:
        return {"ok": False, "n_points": n, "sigma_v": float("inf")}
    res = fit_with_uncertainty(corr["xp"], corr["yp"], corr["xq"], corr["yq"], corr["d"], h, w,
                               sid=corr["sid"], order=order, reg_strength=reg_strength)
    res["ok"] = bool(np.isfinite(res["sigma_v"]))
    res["corr"], res["hw"] = corr, (h, w)          # for choose_order (not JSON-safe)
    return res


def scan_direct(src: FrameCache, layer, starts, stride, flow_v_sign=-1.0, fb_thresh=1.5,
                points=8000, order=2, reg_strength=0.5, seed=0, progress=None) -> dict:
    """Score one direct (pre-baked gap) flow layer on a few pairs."""
    corr, (h, w) = collect_direct(src, layer, starts, stride, flow_v_sign, fb_thresh, points,
                                  rng=np.random.default_rng(seed), progress=progress)
    res = _score(corr, h, w, order, reg_strength)
    res["n_starts"] = len(starts)
    return res


def scan_chain(src: FrameCache, layer, starts, lengths, flow_v_sign=-1.0, fb_thresh=1.0,
               points=6000, order=2, reg_strength=0.5, seed=0, progress=None) -> dict:
    """Score several chain lengths with ONE walk per start frame.

    The walk runs to the longest length and emits at every offset any candidate
    uses; each length is then scored on just its own offsets. Returns
    ``{length: score}``.
    """
    offs = {L: chain_offsets(L) for L in lengths}
    union = sorted(set(o for v in offs.values() for o in v))
    corr, (h, w) = collect_chain(src, layer, sorted(starts), union, flow_v_sign, fb_thresh, points,
                                 rng=np.random.default_rng(seed), progress=progress)
    out = {}
    for L in lengths:
        m = np.isin(corr["off"], offs[L])
        res = _score({k: v[m] for k, v in corr.items()}, h, w, order, reg_strength)
        res["n_starts"] = len(starts)
        out[L] = res
    return out
