"""Headless vignette solve: the gizmo's Analyse + Solve without Nuke (farm / shell).

Same code path as the gizmo (``vignette_solve.probe`` / ``auto_setup`` /
``run_solve``): probes the flow headers, scans every flow source, picks the flow
v sign, lens model and poly order, then solves. Anything given on the command
line overrides the automatic pick. Prints one JSON line (the fit plus the
settings used) to stdout; the analysis report goes to stderr.

    # fully automatic
    python solve_cli.py --plate /p/plate.%04d.exr --flow /p/flow.%04d.exr --first 1001 --last 1200

    # a fixed flow source (mode follows the layer: gap 1 = chain, fXX = direct)
    python solve_cli.py --plate ... --flow ... --first 1001 --last 1200 \
        --layer smartvector_f01_v01 --stride 48

The scan is skipped when the layer (plus --stride for a chain), --flow-v-sign,
--order and --spherical/--anamorphic are all given. Needs only numpy (EXRs are
read by the pure-Python ``exr_read``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vignette_solve as vs  # noqa: E402


class _StderrProgress:
    """Progress callback printing every ~10 % to stderr."""

    def __init__(self, title, total):
        self.title, self.total, self.done, self.shown = title, max(1, total), 0, -1
        print("[vignette] %s" % title, file=sys.stderr)

    def __call__(self, msg):
        self.done += 1
        pct = min(99, int(100 * self.done / self.total)) // 10 * 10
        if pct != self.shown:
            self.shown = pct
            print("[vignette]   %3d%%  %s" % (pct, msg), file=sys.stderr)
        return False

    def close(self):
        pass


def _log(lines):
    for line in lines:
        print("[vignette] " + line, file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plate", required=True, help="plate path pattern (%%04d or ####)")
    ap.add_argument("--flow", required=True, help="flow path pattern")
    ap.add_argument("--first", type=int, required=True, help="first frame to look at")
    ap.add_argument("--last", type=int, required=True, help="last frame to look at")
    ap.add_argument("--layer", help="flow layer, e.g. smartvector_f01_v01 (default: best from the scan)")
    ap.add_argument("--mode", choices=["direct", "chain"],
                    help="default: chain for a gap-1 layer, direct otherwise")
    ap.add_argument("--stride", type=int, help="chain length (chain) / layer gap (direct)")
    ap.add_argument("--samples", type=int, default=vs.DEFAULT_SAMPLES)
    ap.add_argument("--flow-v-sign", type=float, help="default: detected from the footage")
    ap.add_argument("--order", type=int, choices=[2, 3], help="default: hold-out validation")
    lens = ap.add_mutually_exclusive_group()
    lens.add_argument("--spherical", dest="fit_aniso", action="store_false", default=None,
                      help="circular vignette (log_aniso = 0)")
    lens.add_argument("--anamorphic", dest="fit_aniso", action="store_true",
                      help="elliptical vignette (default: hold-out validation)")
    ap.add_argument("--reg", type=float, default=0.5)
    ap.add_argument("--fb-thresh", type=float, default=1.5)
    ap.add_argument("--long-side", type=int, help="default: 768 direct / 640 chain (as the gizmo)")
    args = ap.parse_args(argv)

    plate_fn = vs.pattern_path_fn(args.plate)
    flow_fn = vs.pattern_path_fn(args.flow)
    rng = (args.first, args.last)
    an, lines = vs.probe(plate_fn, flow_fn, rng, rng, (args.plate, args.flow))
    _log(lines)
    if not an["layers"]:
        raise SystemExit("no flow layer with .n_u/.n_v/.p_u/.p_v found")
    both = vs.solve_range(an)
    if both is None:
        raise SystemExit("no frame has both plate and flow in %d-%d" % rng)
    first, last = both
    if args.layer is not None and args.layer not in an["layers"]:
        raise SystemExit("layer %r not found; have: %s" % (args.layer, ", ".join(sorted(an["layers"]))))

    fixed_source = args.layer is not None and (args.stride is not None or an["gaps"][args.layer] > 1)
    picks, option = {}, None
    if not (fixed_source and args.flow_v_sign is not None and args.order is not None
            and args.fit_aniso is not None):
        opts, picks, scan_lines = vs.auto_setup(
            an, plate_fn, flow_fn, first, last,
            flow_v_sign=-1.0 if args.flow_v_sign is None else args.flow_v_sign,
            fb_thresh=args.fb_thresh, reg_strength=args.reg,
            fit_aniso=True if args.fit_aniso is None else args.fit_aniso,
            progress_factory=_StderrProgress)
        _log(scan_lines)
        usable = [o for o in opts if o["ok"] and (args.layer is None or o["layer"] == args.layer)]
        option = usable[0] if usable else None

    layer = args.layer or (option and option["layer"])
    if layer is None:
        raise SystemExit("the scan found no usable flow source")
    gap = an["gaps"][layer]
    mode = {"direct": 0, "chain": 1}[args.mode] if args.mode else (1 if gap == 1 else 0)
    stride = args.stride or (option["stride"] if option and option["mode"] == mode else None)
    if stride is None:
        if mode == 1:
            raise SystemExit("the scan found no usable chain length for %s - give --stride" % layer)
        stride = gap
    settings = {
        "flow_v_sign": args.flow_v_sign if args.flow_v_sign is not None else picks.get("flow_v_sign"),
        "order": args.order or picks.get("order"),
        "fit_aniso": args.fit_aniso if args.fit_aniso is not None else picks.get("fit_aniso"),
    }
    defaults = {"flow_v_sign": -1.0, "order": 2, "fit_aniso": True}
    settings = {k: (defaults[k] if v is None else v) for k, v in settings.items()}

    try:
        frames = vs.source_frames(an, mode, layer, stride, first, last, args.samples)
    except ValueError as e:
        raise SystemExit(str(e))
    steps = len(frames) if mode == 0 else len(frames) * stride
    res = vs.run_solve(plate_fn, flow_fn, mode, frames, stride, vs._channels(layer),
                       flow_v_sign=settings["flow_v_sign"], order=settings["order"],
                       reg_strength=args.reg, fb_thresh=args.fb_thresh,
                       fit_aniso=settings["fit_aniso"], long_side=args.long_side,
                       progress=_StderrProgress("solving", steps))
    res.update({"layer": layer, "mode": "direct" if mode == 0 else "chain", "stride": stride,
                "frames": frames, "n_pairs": len(frames), **settings})
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    main()
