"""Nuke-side glue for the VignetteFromMotion gizmo.

Runs *inside* Nuke's Python. Two buttons:
  * **Analyse** (``analyse(nuke.thisNode())``) — probes the connected Reads, then
    runs a quick low-res scan that scores every usable flow source and chain
    length, checks the flow's v sign and the lens model, and fills every knob
    under *advanced* with the best guess.
  * **Solve** (``solve(nuke.thisNode())``) — runs the numpy-only fit with those
    settings, writes k1/k2/k3/log_aniso onto the group (driving its procedural V
    output) and creates a NEW Vignette node each time. Divisor nodes made
    by earlier solves are never touched.

Inputs to the group (by index):
  0: plate  (Read)  — the shot
  1: flow   (Read)  — motion vectors (RAFT / SmartVector). One EXR holds both
                      directions: .n_u/.n_v (to next frame) and .p_u/.p_v
                      (to previous). Stride-1 (f01) layers are CHAINED to a
                      longer baseline; SmartVector fXX layers are used directly.

The probe / scan / solve steps (``probe``, ``auto_setup``, ``source_frames``,
``run_solve``) do not touch Nuke, so ``solve_cli.py`` runs the same logic
headless.

Dependencies in Nuke's Python: numpy (bundled). EXRs are read by the pure-Python
``exr_read`` next to this file. No scipy, no torch, no native EXR library.
"""

from __future__ import annotations

import math
import os
import re
import sys
import time

# Make vignette_fit importable when this file sits alongside it on NUKE_PATH.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vignette_fit as vf  # noqa: E402
import exr_read  # noqa: E402

if not hasattr(exr_read, "read_header"):  # stale copy cached by an earlier Nuke import
    import importlib
    exr_read = importlib.reload(exr_read)
    vf.exr_read = exr_read

try:
    import nuke  # only available inside Nuke
except Exception:  # pragma: no cover
    nuke = None


def _upstream_read(group, input_index):
    """Return the Read node feeding a group input, walking up through Dots etc."""
    n = group.input(input_index)
    for _ in range(64):
        if n is None or n.Class() == "Read":
            return n
        n = n.input(0)
    return None


def substitute_frame(pattern, frame):
    """Resolve a printf (``%04d``) or hash (``####``) frame pattern; None if it has neither."""
    if "%" in pattern:                                   # e.g. plate.%04d.exr
        try:
            return pattern % frame
        except (TypeError, ValueError):
            pass
    m = re.search(r"#+", pattern)                        # e.g. plate.####.exr
    if m:
        return pattern[:m.start()] + str(int(frame)).zfill(len(m.group(0))) + pattern[m.end():]
    return None


def pattern_path_fn(pattern):
    """f(frame) -> path for a file pattern (a path without a pattern is returned as is)."""
    def fn(frame):
        path = substitute_frame(pattern, frame)
        return pattern if path is None else path
    return fn


def _path_fn(read_node):
    """Return f(frame) -> resolved EXR path for that frame.

    Resolves the printf/# pattern of the raw file knob (most reliable across Nuke
    versions); falls back to File_Knob.evaluate(frame).
    """
    knob = read_node["file"]

    def fn(frame):
        raw = knob.getValue()
        path = substitute_frame(raw, frame)
        if path is not None:
            return path
        try:
            return knob.evaluate(frame)                  # last resort
        except TypeError:
            return raw
    return fn


def sample_frames(first, last, stride, n):
    """``n`` start frames evenly over [first, last - stride] (no analysis available)."""
    hi = max(first, last - stride)
    if n >= (hi - first + 1):
        return list(range(first, hi + 1))
    import numpy as np
    return [int(round(v)) for v in np.linspace(first, hi, n)]


class _Progress:
    """nuke.ProgressTask wrapper usable as a vignette_fit ``progress`` callback."""

    def __init__(self, title, total):
        self.task = nuke.ProgressTask(title)
        self.total = max(1, total)
        self.done = 0

    def __call__(self, msg):
        self.done += 1
        self.task.setProgress(min(99, int(100 * self.done / self.total)))
        self.task.setMessage(msg)
        return self.task.isCancelled()

    def close(self):
        del self.task


class NullProgress:
    """Progress callback that never cancels (headless use)."""

    def __init__(self, title="", total=1):
        pass

    def __call__(self, msg):
        return False

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# Probe: which flow layers exist on which frames (headers only)
# --------------------------------------------------------------------------- #
# SmartVector writes layer fN (vectors to f+N / f-N) only on every N-th frame,
# so a layer's availability is per frame. We scan every flow header once.
_FLOW_COMPS = ("n_u", "n_v", "p_u", "p_v")
_LAYER_RE = re.compile(r"^(?P<layer>.+)\.(?P<comp>[np]_[uv])$")
_GAP_RE = re.compile(r"_f(\d+)(?:_|$)")
# Nuke's own motion layers (VectorGenerator / a Write of the motion channels):
# forward.u/v = to the next frame, backward.u/v = to the previous one. Treated
# as one stride-1 layer named "<prefix>forward/backward".
_FWD_BWD = "forward/backward"
_FWD_BWD_RE = re.compile(r"^(?P<pre>.*)(?P<dir>forward|backward)\.(?P<c>[uv])$")


def _flow_comp(ch):
    """(layer, comp) for a flow channel name (comp in _FLOW_COMPS), or None."""
    m = _LAYER_RE.match(ch)
    if m:
        return m.group("layer"), m.group("comp")
    m = _FWD_BWD_RE.match(ch)
    if m:
        return m.group("pre") + _FWD_BWD, ("n_" if m.group("dir") == "forward" else "p_") + m.group("c")
    return None

# Auto setup (Analyse). The scan reads frames at SCAN_LONG_SIDE and scores each
# candidate by the predicted error of V in the frame corner (vf.fit_with_uncertainty).
DEFAULT_SAMPLES = 12                          # start frames / pairs a Solve uses
CHAIN_LENGTHS = (8, 16, 24, 32, 48, 64, 96)   # chain lengths tried for f01 layers
SCAN_LONG_SIDE = 384
SCAN_PAIRS = 3                                # pairs scored per direct layer
SCAN_CHAINS = 3                               # overlapping chains scored per f01 layer
SOLVE_LONG_SIDE = {0: 768, 1: 640}            # solve_from_reads / solve_from_stride1 defaults


def _layer_gap(layer, frames):
    """Temporal gap N of a flow layer: from its fNN name, else its frame spacing."""
    m = _GAP_RE.search(layer)
    if m:
        return int(m.group(1))
    steps = sorted(b - a for a, b in zip(frames, frames[1:]))
    return steps[len(steps) // 2] if steps else 1


def _read_range(read):
    return int(read["first"].value()), int(read["last"].value())


def _load_analysis(group):
    import json
    try:
        raw = group["analysis_json"].value()
        return json.loads(raw) if raw else None
    except Exception:
        return None


def probe(plate_fn, flow_fn, plate_range, flow_range, names=("plate", "flow"), tick=None):
    """Read every flow EXR header (not the pixels) and check which plate frames exist.

    Returns ``(an, lines)`` — the analysis dict (plate frames, per-layer frames and
    gaps) and report lines — or None if ``tick(msg)`` asked to cancel.
    """
    pf, pl_ = plate_range
    ff, fl = flow_range
    plate_frames, plate_missing = [], []
    for fr in range(pf, pl_ + 1):
        if tick is not None and tick("plate %d" % fr):
            return None
        (plate_frames if os.path.isfile(plate_fn(fr)) else plate_missing).append(fr)

    layer_frames = {}           # layer -> frames where n_u/n_v/p_u/p_v all exist
    flow_missing, flow_bad, flow_hdr = [], [], None
    for fr in range(ff, fl + 1):
        if tick is not None and tick("flow %d" % fr):
            return None
        path = flow_fn(fr)
        if not os.path.isfile(path):
            flow_missing.append(fr)
            continue
        try:
            hdr = exr_read.read_header(path)
        except Exception as e:
            flow_bad.append((fr, str(e)))
            continue
        flow_hdr = flow_hdr or hdr
        comps = {}
        for ch in hdr["channels"]:
            lc = _flow_comp(ch)
            if lc:
                comps.setdefault(lc[0], set()).add(lc[1])
        for layer, c in comps.items():
            if c.issuperset(_FLOW_COMPS):
                layer_frames.setdefault(layer, []).append(fr)

    plate_hdr = None
    if plate_frames:
        try:
            plate_hdr = exr_read.read_header(plate_fn(plate_frames[0]))
        except Exception as e:
            plate_hdr = {"readable": False, "problems": [str(e)], "channels": [],
                         "width": 0, "height": 0, "compression": "?"}

    layers = {L: {"frames": fr, "gap": _layer_gap(L, fr)} for L, fr in layer_frames.items()}
    an = {
        "plate": plate_frames,
        "flow_range": [ff, fl],
        "plate_range": [pf, pl_],
        "layers": {L: v["frames"] for L, v in layers.items()},
        "gaps": {L: v["gap"] for L, v in layers.items()},
    }

    def span(frames):
        return "%d-%d" % (frames[0], frames[-1]) if frames else "-"

    def size(hdr):
        s = "   %dx%d  %s" % (hdr["width"], hdr["height"], hdr["compression"])
        dw = hdr.get("data_window")
        if dw and dw != [0, 0, hdr["width"] - 1, hdr["height"] - 1]:
            s += "  (data window %d,%d - %d,%d)" % tuple(dw)
        return s + ("" if hdr["readable"] else "  NOT READABLE: " + ", ".join(hdr["problems"]))

    lines = ["plate: %s  %d frames (%s)" % (names[0], len(plate_frames), span(plate_frames))]
    if plate_hdr:
        lines.append(size(plate_hdr))
    if plate_missing:
        lines.append("   missing: %d frames (first %d)" % (len(plate_missing), plate_missing[0]))
    lines.append("flow: %s  range %d-%d" % (names[1], ff, fl))
    if flow_hdr:
        lines.append(size(flow_hdr))
        if plate_hdr and (plate_hdr["width"], plate_hdr["height"]) != (flow_hdr["width"], flow_hdr["height"]):
            lines.append("   WARNING: flow format differs from plate")
    if flow_missing:
        lines.append("   missing: %d frames (first %d)" % (len(flow_missing), flow_missing[0]))
    if flow_bad:
        lines.append("   unreadable headers: %d (first %d: %s)" % (len(flow_bad), flow_bad[0][0], flow_bad[0][1]))
    for L in sorted(layers, key=lambda L: layers[L]["gap"]):
        fr = layers[L]["frames"]
        lines.append("   %s  gap %d: %d frames %s" % (L, layers[L]["gap"], len(fr), span(fr)))
    if not layers:
        lines.append("   no layer with .n_u/.n_v/.p_u/.p_v (or forward.u/v + backward.u/v) found")
    return an, lines


def solve_range(an):
    """(first, last) of the frames where both plate and flow exist, or None."""
    both = sorted(set(an["plate"]) & {f for fr in an["layers"].values() for f in fr})
    return (both[0], both[-1]) if both else None


# --------------------------------------------------------------------------- #
# Frame selection
# --------------------------------------------------------------------------- #
def _valid_frames(an, mode, layer, stride, first, last):
    """Start frames f in [first, last] usable for the chosen layer/mode/stride."""
    fs = set(an["layers"].get(layer, []))
    pl = set(an["plate"])
    if mode == 0:   # direct: n_* at f, p_* at f+stride, plate at both
        return [f for f in sorted(fs)
                if first <= f and f + stride <= last
                and f + stride in fs and f in pl and f + stride in pl]
    # chain: flow + plate on every frame f..f+stride
    return [f for f in range(first, last - stride + 1)
            if all(f + k in fs and f + k in pl for k in range(stride + 1))]


def _pick_evenly(frames, n):
    if n <= 0 or n >= len(frames):
        return list(frames)
    idx = sorted({int(round(i * (len(frames) - 1) / float(n - 1))) for i in range(n)}) if n > 1 else [0]
    return [frames[i] for i in idx]


def _spread_starts(valid, n, spacing):
    """``n`` start frames ~``spacing`` apart around the middle of ``valid``.

    Keeps the scan's chains overlapping, so they share most of their EXR reads.
    """
    if len(valid) <= n:
        return list(valid)
    span = spacing * (n - 1)
    mid = valid[len(valid) // 2]
    lo = max(valid[0], min(mid - span // 2, valid[-1] - span))
    return sorted({min(valid, key=lambda v: abs(v - (lo + i * spacing))) for i in range(n)})


def effective_starts(starts, length):
    """Roughly how many independent pairs / chains ``starts`` are worth.

    Chains of ``length`` frames that start close together follow the same scene
    content through mostly the same flow, so their errors are correlated. Each
    start counts in proportion to its distance from the previous one, up to 1
    (no overlap). Direct pairs one gap apart count fully.
    """
    s = sorted(starts)
    if not s:
        return 0.0
    return 1.0 + sum(min(1.0, (b - a) / float(max(1, length))) for a, b in zip(s, s[1:]))


def source_frames(an, mode, layer, stride, first, last, samples):
    """Start frames a solve uses for this flow source; ValueError if it can't run."""
    gap = an["gaps"][layer]
    if mode == 0 and stride != gap:
        raise ValueError("Direct mode with %s needs stride %d (the layer's frame gap), "
                         "but stride is %d." % (layer, gap, stride))
    if mode == 1 and gap != 1:
        raise ValueError("Chain mode needs a layer present on every frame (gap 1); "
                         "%s has gap %d." % (layer, gap))
    valid = _valid_frames(an, mode, layer, stride, first, last)
    if not valid:
        raise ValueError("No usable frames for %s (%s, stride %d) in %d-%d.\n"
                         "Widen the solve range or pick another flow source."
                         % (layer, "direct" if mode == 0 else "chain", stride, first, last))
    return _pick_evenly(valid, samples)


def _channels(layer):
    """(n_u, n_v, p_u, p_v) channel names of a flow layer."""
    if layer.endswith(_FWD_BWD):
        pre = layer[:-len(_FWD_BWD)]
        return tuple(pre + c for c in ("forward.u", "forward.v", "backward.u", "backward.v"))
    return tuple("%s.%s" % (layer, c) for c in _FLOW_COMPS)


def _layer_of(channels):
    """The flow layer whose channels these are (the n_u knob's layer part as a fallback)."""
    lc = _flow_comp(channels[0])
    if lc and _channels(lc[0]) == tuple(channels):
        return lc[0]
    nu = channels[0]
    return nu.rsplit(".", 1)[0] if "." in nu else None


# --------------------------------------------------------------------------- #
# Auto setup: score every flow source / stride on the footage
# --------------------------------------------------------------------------- #
def fallback_options(an, first, last, stride):
    """Unscored options (the scan was cancelled or failed): largest gap first, but
    the first direct layer with at least 3 pairs in range goes to the top."""
    layers, plate = an["layers"], an["plate"]
    opts = []
    for layer in sorted(layers, key=lambda L: -an["gaps"][L]):
        gap, frames = an["gaps"][layer], layers[layer]
        sub = {"layers": {layer: frames}, "plate": plate}
        pairs = _valid_frames(sub, 0, layer, gap, frames[0], frames[-1] + gap)
        if pairs:
            opts.append({"label": "direct  %s  stride %d  (%d pairs)" % (layer, gap, len(pairs)),
                         "mode": 0, "layer": layer, "stride": gap})
        if gap == 1 and len(frames) >= 2:
            opts.append({"label": "chain  %s  (%d frames, stride = chain length)" % (layer, len(frames)),
                         "mode": 1, "layer": layer, "stride": None})

    def pairs_in_range(o):
        return len(_valid_frames(an, o["mode"], o["layer"], o["stride"] or stride, first, last))
    best = next((o for o in opts if o["mode"] == 0 and pairs_in_range(o) >= 3), None)
    if best is not None:
        opts.remove(best)
        opts.insert(0, best)
    return opts


def _candidates(an, first, last):
    """What the scan tries: each fXX layer directly, each f01 layer at several chain lengths."""
    cands = []
    for layer in sorted(an["gaps"], key=lambda L: an["gaps"][L]):
        gap = an["gaps"][layer]
        if gap == 1:
            lengths = [L for L in CHAIN_LENGTHS if len(_valid_frames(an, 1, layer, L, first, last)) >= 2]
            if lengths:
                cands.append({"mode": 1, "layer": layer, "lengths": lengths})
                continue
        valid = _valid_frames(an, 0, layer, gap, first, last)
        if valid:
            cands.append({"mode": 0, "layer": layer, "stride": gap, "valid": valid})
    return cands


def _detect_v_sign(src, an, cands, first, last, progress):
    """Flow v sign from the data: chain f01 a few steps, else one hop of the widest fXX."""
    tries = []
    for c in cands:
        if c["mode"] == 1:
            steps = min(8, max(c["lengths"]))
            tries.append((c["layer"], 1, steps, _valid_frames(an, 1, c["layer"], steps, first, last)))
    for c in sorted((c for c in cands if c["mode"] == 0), key=lambda c: -c["stride"]):
        tries.append((c["layer"], c["stride"], 1, c["valid"]))
    info = None
    for layer, gap, steps, valid in tries[:2]:
        if not valid:
            continue
        sign, info = vf.detect_flow_v_sign(src, layer, _pick_evenly(valid, 2), gap, steps,
                                           progress=progress)
        info["layer"] = layer
        if sign is not None:
            return sign, info
    return None, info


def auto_setup(an, plate_fn, flow_fn, first, last, flow_v_sign=-1.0, fb_thresh=1.5,
               reg_strength=0.5, fit_aniso=True, progress_factory=None):
    """Scan the footage; return (options best-first, picks, report lines).

    ``flow_v_sign`` / ``fit_aniso`` are the current settings, used where the
    footage can't decide. Picks are ``{"flow_v_sign", "fit_aniso", "order"}``;
    None means "could not tell, keep the current setting".
    ``progress_factory(title, total)`` returns a vignette_fit progress callback
    with a ``close()``.
    """
    import numpy as np
    if progress_factory is None:
        progress_factory = _Progress if nuke is not None else NullProgress
    cands = _candidates(an, first, last)
    if not cands:
        raise RuntimeError("no flow layer has usable frames in %d-%d" % (first, last))
    max_len = max([max(c["lengths"]) for c in cands if c["mode"] == 1] or [0])
    src = vf.FrameCache(plate_fn, flow_fn, {L: _channels(L) for L in an["gaps"]},
                        long_side=SCAN_LONG_SIDE, layer_frames=an["layers"],
                        max_frames=max(8, max_len + 3))
    src.lum(next(f for f in an["plate"] if first <= f <= last))   # learn the resolution

    # rough step count for the progress bar: v-sign walks + pairs + chain steps
    total = 40
    for c in cands:
        if c["mode"] == 0:
            total += SCAN_PAIRS
        else:
            total += SCAN_CHAINS * max(c["lengths"])
    prog = progress_factory("VignetteFromMotion: finding the best settings", total)
    t0 = time.time()
    try:
        sign, sign_info = _detect_v_sign(src, an, cands, first, last, prog)
        v_sign = sign if sign is not None else float(flow_v_sign)
        order = 2                       # candidates are ranked with k1,k2; k3 is checked after
        reg = float(reg_strength)

        opts = []
        for c in cands:
            layer, ch = c["layer"], _channels(c["layer"])
            # fb thresh is in the solve's downsampled pixels; express it at scan resolution
            fb = float(fb_thresh) * src.factor_for(SOLVE_LONG_SIDE[c["mode"]]) / float(src.fac or 1)
            if c["mode"] == 0:
                starts = _pick_evenly(c["valid"], SCAN_PAIRS)
                res = vf.scan_direct(src, layer, starts, c["stride"], v_sign, fb,
                                     order=order, reg_strength=reg, progress=prog)
                scored = [(c["stride"], res, c["valid"])]
            else:
                longest = max(c["lengths"])
                valid = _valid_frames(an, 1, layer, longest, first, last)
                starts = _spread_starts(valid, SCAN_CHAINS, max(1, longest // 4))
                by_len = vf.scan_chain(src, layer, starts, c["lengths"], v_sign, fb,
                                       order=order, reg_strength=reg, progress=prog)
                scored = [(L, by_len[L], _valid_frames(an, 1, layer, L, first, last))
                          for L in c["lengths"]]
            for stride, res, valid in scored:
                samples = min(DEFAULT_SAMPLES, len(valid))
                # sigma_v treats the scan's n_starts as independent; the solve's
                # starts are worth effective_starts() independent ones (overlap)
                n_eff = effective_starts(_pick_evenly(valid, samples), stride)
                pred = res["sigma_v"] * np.sqrt(res["n_starts"] / max(1.0, n_eff))
                opts.append({"mode": c["mode"], "layer": layer, "stride": stride, "samples": samples,
                             "n_eff": float(n_eff), "channels": ch, "pred": float(pred), "ok": res["ok"],
                             "n_points": res["n_points"], "radial": res.get("radial", 0.0),
                             "k1": res.get("k1"), "k2": res.get("k2"),
                             "_corr": res.get("corr"), "_hw": res.get("hw"), "_sigma_v": res["sigma_v"]})
        opts.sort(key=lambda o: (not o["ok"], o["pred"]))

        # lens model on the best candidate's data, by hold-out validation:
        # does an ellipse (anamorphic) or k3 predict unseen frames better?
        aniso_pick = order_pick = aniso_info = order_info = None
        if opts and opts[0]["ok"]:
            best = opts[0]
            h, w = best["_hw"]
            prog("spherical / anamorphic check")
            aniso_pick, aniso_info = vf.choose_aniso(best["_corr"], h, w, best["_sigma_v"],
                                                     order=2, reg_strength=reg)
            use_aniso = aniso_pick if aniso_pick is not None else bool(fit_aniso)
            prog("poly order check")
            order_pick, order_info = vf.choose_order(best["_corr"], h, w, best["_sigma_v"],
                                                     fit_aniso=use_aniso, reg_strength=reg)
    finally:
        prog.close()
    for o in opts:                      # scan data is not JSON-safe (and big)
        for k in ("_corr", "_hw", "_sigma_v"):
            o.pop(k, None)
    for o in opts:
        kind = "direct" if o["mode"] == 0 else "chain "
        what = "stride %d" % o["stride"] if o["mode"] == 0 else "length %d" % o["stride"]
        score = "+-%.2g%%" % (100 * o["pred"]) if o["ok"] else "too few points"
        o["label"] = "%s  %s  %s   %s  (%d %s)" % (
            kind, o["layer"], what, score, o["samples"], "pairs" if o["mode"] == 0 else "chains")

    lines = ["", "auto setup  (scan %.0fs, %d frame reads)" % (time.time() - t0, src.reads)]
    if sign is not None:
        lines.append("   flow v sign %+d  (from %s: log error %.3f vs %.3f)" % (
            sign, sign_info["layer"], min(sign_info["err_pos"], sign_info["err_neg"]),
            max(sign_info["err_pos"], sign_info["err_neg"])))
    else:
        lines.append("   flow v sign: could not tell from the footage (too little vertical motion) "
                     "- kept %+d" % v_sign)
    if aniso_pick is not None:
        lines.append("   lens %s  (%s: held-out error %.4g with ellipse vs %.4g circular, "
                     "V at the edges differs by %.2g%%, ellipse w:h %.3f)" % (
                         "anamorphic" if aniso_pick else "spherical",
                         "ellipse helps" if aniso_pick else "ellipse not supported by the data",
                         aniso_info["cost_ana"], aniso_info["cost_sph"], 100 * aniso_info["edge_diff"],
                         math.exp(-2.0 * aniso_info["log_aniso"])))
    elif aniso_info is not None:
        lines.append("   lens: kept the anamorphic setting (%s)" % aniso_info["reason"])
    if order_pick is not None and "cost2" in order_info:
        lines.append("   poly order %d  (%s: held-out error %.4g with k3 vs %.4g without, "
                     "corner V differs by %.2g%%)" % (
                         order_pick, "k3 helps" if order_pick == 3 else "k3 not supported by the data",
                         order_info["cost3"], order_info["cost2"], 100 * order_info["corner_diff"]))
    elif order_pick is not None:
        lines.append("   poly order 2  (%s)" % order_info["reason"])
    lines.append("   %-44s %10s %6s %8s %7s %7s %7s" % (
        "candidate", "corner err", "indep", "points", "|dr2|", "k1", "k2"))
    for o in opts:
        what = "%s %s %d" % ("direct" if o["mode"] == 0 else "chain", o["layer"], o["stride"])
        if o["ok"]:
            lines.append("   %-44s %9.2g%% %6.1f %8d %7.3f %7.3f %7.3f" % (
                what, 100 * o["pred"], o["n_eff"], o["n_points"], o["radial"], o["k1"], o["k2"]))
        else:
            lines.append("   %-44s %10s %6.1f %8d" % (what, "-", o["n_eff"], o["n_points"]))
    return opts, {"flow_v_sign": sign, "fit_aniso": aniso_pick, "order": order_pick}, lines


def run_solve(plate_fn, flow_fn, mode, frames, stride, channels, flow_v_sign=-1.0, order=2,
              reg_strength=0.5, fb_thresh=1.5, fit_aniso=True, long_side=None, progress=None):
    """Fit one flow source: mode 0 = direct (pre-baked gap), 1 = chain (stride-1 flow).

    ``channels`` is ``(n_u, n_v, p_u, p_v)``; ``long_side`` defaults per mode
    (``SOLVE_LONG_SIDE``) so the gizmo and the CLI solve at the same resolution.
    """
    nu, nv, pu, pv = channels
    common = dict(plate_path_fn=plate_fn, flow_path_fn=flow_fn, frames=frames, stride=stride,
                  flow_v_sign=float(flow_v_sign), order=int(order), reg_strength=float(reg_strength),
                  fb_thresh=float(fb_thresh), fit_aniso=bool(fit_aniso), progress=progress,
                  long_side=int(long_side or SOLVE_LONG_SIDE[mode]))
    if mode == 0:
        return vf.solve_from_reads(u_channel=nu, v_channel=nv, bwd_u_channel=pu, bwd_v_channel=pv,
                                   **common)
    return vf.solve_from_stride1(n_u_channel=nu, n_v_channel=nv, p_u_channel=pu, p_v_channel=pv,
                                 **common)


# --------------------------------------------------------------------------- #
# Nuke buttons
# --------------------------------------------------------------------------- #
def _knob_fit_aniso(group):
    """The anamorphic toggle (gizmos made before it existed always fit the ellipse)."""
    k = group.knobs().get("fit_aniso")
    return True if k is None else bool(k.value())


def analyse(group, quiet=False):
    """Probe the Reads, scan the footage, and set every advanced knob to the best guess."""
    import json
    if nuke is None:
        raise RuntimeError("analyse() must run inside Nuke")
    plate = _upstream_read(group, 0)
    flow = _upstream_read(group, 1)
    if plate is None or flow is None:
        nuke.message("Connect a plate Read (input 0) and a flow Read (input 1).")
        return None

    plate_fn, flow_fn = _path_fn(plate), _path_fn(flow)
    pf, pl_ = _read_range(plate)
    ff, fl = _read_range(flow)
    total = (pl_ - pf + 1) + (fl - ff + 1)
    task = nuke.ProgressTask("VignetteFromMotion: analysing")
    done = 0

    def tick(msg):
        nonlocal done
        done += 1
        task.setProgress(int(100 * done / max(1, total)))
        task.setMessage(msg)
        return task.isCancelled()

    try:
        probed = probe(plate_fn, flow_fn, (pf, pl_), (ff, fl), (plate.name(), flow.name()), tick)
    finally:
        del task
    if probed is None:
        return None
    an, lines = probed
    an["sources"] = [plate["file"].value(), flow["file"].value()]

    rng = solve_range(an)
    if rng:
        group["solve_first"].setValue(rng[0])
        group["solve_last"].setValue(rng[1])
    first = int(group["solve_first"].value()); last = int(group["solve_last"].value())

    # ---- auto setup: score every flow source / stride on the footage ------ #
    options, picks = [], {}
    if an["layers"]:
        try:
            options, picks, scan_lines = auto_setup(
                an, plate_fn, flow_fn, first, last,
                flow_v_sign=float(group["flow_v_sign"].value()),
                fb_thresh=float(group["fb_thresh"].value()),
                reg_strength=float(group["reg_strength"].value()),
                fit_aniso=_knob_fit_aniso(group))
            lines += scan_lines
        except vf.Cancelled:
            lines += ["", "auto setup cancelled - using the largest flow gap"]
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            lines += ["", "auto setup failed (%s) - using the largest flow gap" % e]
    if not options:                             # cancelled / failed: old heuristic
        options = fallback_options(an, first, last, int(group["stride"].value()))
    lines.append("%d solve option(s)" % len(options))

    an["options"] = options
    group["analysis_json"].setValue(json.dumps(an, separators=(",", ":")))
    report = "\n".join(lines)
    print("[vignette] analysis\n" + report)
    group["analysis_info"].setValue(report)
    if "analysis_status" in group.knobs():
        group["analysis_status"].setValue(
            "analysed  (%s)  -  details under 'advanced'" % time.strftime("%Y-%m-%d %H:%M"))

    # ---- fill the menu with the best guess selected ---------------------- #
    k = group["flow_source"]
    labels = [o["label"] for o in options] or ["(no usable flow - see analysis)"]
    k.setValues(labels)
    k.setValue(0)
    if options:
        _apply_option(group, options[0])
    for name, value in picks.items():           # flow_v_sign, fit_aniso, order
        if value is not None and name in group.knobs():
            group[name].setValue(value)

    if not quiet:
        best = ("\n\nBest guess:  " + options[0]["label"]) if options else ""
        nuke.message("Analysis done.%s\n\n%s" % (best, report))
    return an


def _apply_option(group, opt):
    """Push a flow-source option into the effective knobs (mode, channels, stride, samples)."""
    group["flow_mode"].setValue(opt["mode"])
    for comp, ch in zip(_FLOW_COMPS, _channels(opt["layer"])):
        group["%s_channel" % comp].setValue(ch)
    if opt.get("stride"):
        group["stride"].setValue(opt["stride"])
    if opt.get("samples"):
        group["num_samples"].setValue(opt["samples"])


def knob_changed(group, knob):
    """Group knobChanged hook (wired in the .nk)."""
    name = knob.name()
    if name == "flow_source":
        an = _load_analysis(group)
        if not an:
            return
        opts = an.get("options", [])
        i = int(knob.getValue())
        if 0 <= i < len(opts):
            _apply_option(group, opts[i])


def solve(group):
    if nuke is None:
        raise RuntimeError("solve() must run inside Nuke")

    plate = _upstream_read(group, 0)
    flow = _upstream_read(group, 1)
    if plate is None or flow is None:
        nuke.message("Connect a plate (input 0) and a flow (input 1).")
        return

    # Analysis tells us on which frames each flow layer exists; (re)run it if
    # missing or if the Reads were repointed since.
    an = _load_analysis(group)
    if not an or an.get("sources") != [plate["file"].value(), flow["file"].value()]:
        an = analyse(group, quiet=True)
        if an is None:
            return

    channels = tuple(group["%s_channel" % c].value() for c in _FLOW_COMPS)
    # 0 = direct (SmartVector: a pre-baked stride layer, e.g. f64; stride = gap).
    # 1 = chain  (a stride-1 / f01 layer, chained to the stride baseline).
    mode = int(group["flow_mode"].getValue())     # Enumeration: .getValue() = index
    stride = int(group["stride"].value())
    first = int(group["solve_first"].value())
    last = int(group["solve_last"].value())
    samples = int(group["num_samples"].value())
    layer = _layer_of(channels)
    if layer in an["layers"]:
        try:
            frames = source_frames(an, mode, layer, stride, first, last, samples)
        except ValueError as e:
            nuke.message(str(e))
            return
    else:
        print("[vignette] layer %r not in analysis - sampling blindly" % layer)
        frames = sample_frames(first, last, stride, samples)

    steps = len(frames) if mode == 0 else len(frames) * max(vf.chain_offsets(stride))
    prog = _Progress("VignetteFromMotion: solving", steps)
    try:
        print("[vignette] solving: mode=%d stride=%d frames=%s" % (mode, stride, frames))
        res = run_solve(_path_fn(plate), _path_fn(flow), mode, frames, stride, channels,
                        flow_v_sign=float(group["flow_v_sign"].value()),
                        order=int(group["order"].value()),
                        reg_strength=float(group["reg_strength"].value()),
                        fb_thresh=float(group["fb_thresh"].value()),
                        fit_aniso=_knob_fit_aniso(group), progress=prog)
    except vf.Cancelled:
        print("[vignette] solve cancelled")
        return
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        nuke.message("Vignette solve failed:\n%s\n\n(traceback in the Script Editor)" % e)
        return
    finally:
        prog.close()

    group["k1"].setValue(res["k1"])
    group["k2"].setValue(res["k2"])
    if "k3" in group.knobs():
        group["k3"].setValue(res["k3"])
    elif res["k3"]:
        print("[vignette] WARNING: this gizmo has no k3 knob (made before k3 support); "
              "its own V output ignores k3. The new divisor node includes it. Re-import the gizmo.")
    group["log_aniso"].setValue(res["log_aniso"])
    if "result_info" in group.knobs():
        group["result_info"].setValue("%s  %s" % (group["flow_source"].value(),
                                                  time.strftime("%Y-%m-%d %H:%M")))
    print("[vignette] done: k1=%.4f k2=%.4f k3=%.4f log_aniso=%.4f"
          % (res["k1"], res["k2"], res["k3"], res["log_aniso"]))
    div = make_divisor(group, res)
    nuke.message(
        "Vignette solved:\n  k1 = %.4f\n  k2 = %.4f\n  k3 = %.4f\n  log_aniso = %.4f\n  pairs = %d\n\n"
        "New Vignette node: %s"
        % (res["k1"], res["k2"], res["k3"], res["log_aniso"], len(frames), div.name())
    )


def create_vignette_node(group):
    """'Create Vignette node' button: a new node from the last solve's values."""
    info = group.knobs().get("result_info")
    if info is not None and info.value() == "not solved yet":
        nuke.message("Nothing solved yet - press Solve first.")
        return None
    div = make_divisor(group)
    div.setSelected(True)
    return div


# --------------------------------------------------------------------------- #
# Standalone divisor node: plain Nuke nodes, baked values, no Python needed
# --------------------------------------------------------------------------- #
# V is normalised by the half-diagonal, so it is resolution independent (same
# aspect). Temp vars: s = 1/half-diagonal, asp = anamorphic squeeze, r2 = radius^2.
# V is floored at V_MIN (as the container's render_vignette does): with an
# ellipse the frame corners lie beyond the radius the fit's prior covers.
V_MIN = 1e-3
_DIVISOR_TEMPS = [
    ("s", "2/sqrt(width*width+height*height)"),
    ("asp", "exp(parent.log_aniso)"),
    ("r2", "pow(asp*(x+0.5-width/2)*s,2)+pow((y+0.5-height/2)*s/asp,2)"),
    ("V", "max(%g,1+r2*(parent.k1+r2*(parent.k2+r2*parent.k3)))" % V_MIN),
]


_DIVISOR_MODES = ["divisor", "remove (input / V)", "apply (input * V)"]


def _parent_of(node):
    """The Group/Root that contains `node` (button callbacks run *inside* the group)."""
    path = node.fullName().rsplit(".", 1)
    return nuke.toNode("root." + path[0]) if len(path) > 1 else nuke.root()


def make_divisor(group, res=None):
    """Create a NEW 'Vignette' group next to this node with k1/k2/k3/aniso baked in.

    Every call makes a fresh node, so divisors from earlier solves stay as they
    were. Values come from ``res`` (the fit result), else from the group's knobs.
    Mode 'divisor' outputs V; 'remove' outputs input / V; 'apply' outputs
    input * V (re-vignette a comp after working on the flattened plate).
    """
    parent = _parent_of(group)
    with parent:
        taken = {(n.xpos(), n.ypos()) for n in nuke.allNodes()}
        x, y = group.xpos() + 150, group.ypos() + 60
        while any(abs(x - tx) < 100 and abs(y - ty) < 40 for tx, ty in taken):
            x += 120                              # next to the previous solves
        div = nuke.nodes.Group(tile_color=0x7f5f00ff)
        div.setName("Vignette", uncollide=True)
        div.setXYpos(x, y)
        div.addKnob(nuke.Text_Knob("about", "", "Lens vignette V = 1 + k1 r^2 + k2 r^4 + k3 r^6  (anamorphic).\n"
                                   "divisor: outputs V.   remove: input / V.   apply: input * V."))
        div.addKnob(nuke.Enumeration_Knob("mode", "output", _DIVISOR_MODES))
        div.addKnob(nuke.Text_Knob("div_res", "", "values from VignetteFromMotion"))
        for kn, lbl in (("k1", "k1"), ("k2", "k2"), ("k3", "k3"), ("log_aniso", "log aniso")):
            k = nuke.Double_Knob(kn, lbl)
            k.setRange(-1, 1)
            div.addKnob(k)
            if res is not None:
                k.setValue(res[kn])
            elif kn in group.knobs():
                k.setValue(group[kn].value())
        info = group.knobs().get("result_info")       # "<flow source>  <solve date>"
        solved = info.value() if info is not None else "%s  %s" % (
            group["flow_source"].value(), time.strftime("%Y-%m-%d %H:%M"))
        div.addKnob(nuke.Text_Knob("source", "solved by", "%s  (%s)" % (group.name(), solved)))
        with div:
            inp = nuke.nodes.Input(name="img")
            ex = nuke.nodes.Expression(name="Vignette", inputs=[inp])
            for i, (n, e) in enumerate(_DIVISOR_TEMPS):
                ex["temp_name%d" % i].setValue(n)
                ex["temp_expr%d" % i].setValue(e)
            for i, c in enumerate("rgb"):
                ex["expr%d" % i].setValue("parent.mode==0 ? V : parent.mode==1 ? %s/V : %s*V" % (c, c))
            ex["expr3"].setValue("parent.mode ? a : 1")
            nuke.nodes.Output(inputs=[ex])
        plate = group.input(0)
        if plate is not None:
            div.setInput(0, plate)
    return div
