"""Minimal pure-Python EXR reader — stdlib + numpy only (no pip, no native libs).

Runs safely inside Nuke: it does NOT load any OpenEXR/OpenImageIO native library
(those clash with Nuke's bundled copies and segfault). numpy ships with Nuke.

Supports single-part scanline EXRs with HALF or FLOAT channels and NONE / RLE /
ZIPS / ZIP compression (Nuke's default Zip and ARRI/ACES ZIPS plates). Tiled,
multi-part, deep, and PIZ/PXR24/B44/DWA compression are not supported — for those,
re-write the flow/plate through a Nuke Write set to "Zip (16 scanline)".

Images are returned on the **display window** (the format), top row first, so a
plate and a flow written with different bounding boxes / overscan still line up
pixel for pixel. Pixels of the display window outside the data window are NaN.
"""

from __future__ import annotations

import struct
import zlib
from typing import Dict, List, Tuple

import numpy as np

__all__ = ["read_channels", "list_channels", "read_header"]

_MAGIC = 20000630
_LINES_PER_BLOCK = {0: 1, 1: 1, 2: 1, 3: 16}          # NONE, RLE, ZIPS, ZIP
_HALF, _FLOAT, _UINT = 1, 2, 0
_COMPRESSION_NAMES = {0: "none", 1: "rle", 2: "zips", 3: "zip", 4: "piz", 5: "pxr24",
                      6: "b44", 7: "b44a", 8: "dwaa", 9: "dwab"}


class _Short(Exception):
    """The header runs past the bytes read so far."""


def _attrs(data: bytes):
    """Parse the header attributes -> (version, {name: raw bytes}, offset past the header)."""
    if len(data) < 8:
        raise _Short()
    if struct.unpack_from("<I", data, 0)[0] != _MAGIC:
        raise ValueError("not an EXR file")
    version = struct.unpack_from("<I", data, 4)[0]
    off = 8
    attrs = {}
    try:
        while True:
            e = data.index(b"\x00", off)
            name = data[off:e].decode("latin-1"); off = e + 1
            if name == "":
                return version, attrs, off
            off = data.index(b"\x00", off) + 1   # attribute type (unused)
            size = struct.unpack_from("<I", data, off)[0]; off += 4
            if off + size > len(data):
                raise _Short()
            attrs[name] = data[off:off + size]; off += size
    except (ValueError, struct.error):          # index() / unpack ran off the end
        raise _Short()


def _header(fh):
    """Read just enough of an open file to parse the whole header (any size)."""
    data = fh.read(1 << 14)
    while True:
        try:
            return _attrs(data)
        except _Short:
            more = fh.read(max(1 << 16, len(data)))
            if not more:
                raise ValueError("truncated EXR header")
            data += more


def _channel_list(attrs):
    """[(name, pixtype, bytes per sample)] (alphabetical, as stored)."""
    chd = attrs.get("channels", b""); p = 0
    chans = []
    while p < len(chd):
        e = chd.index(b"\x00", p)
        cn = chd[p:e].decode("latin-1"); p = e + 1
        if cn == "":
            break
        pixtype = struct.unpack_from("<i", chd, p)[0]
        p += 16  # pixtype + pLinear + reserved(3) + xSampling + ySampling
        chans.append((cn, pixtype, 2 if pixtype == _HALF else 4))
    return chans


def _box(attrs, name):
    return struct.unpack_from("<iiii", attrs[name], 0)   # xmin, ymin, xmax, ymax


def _problems(version, comp):
    problems = []
    if version & (1 << 9):
        problems.append("tiled")
    if version & ((1 << 11) | (1 << 12)):
        problems.append("deep/multi-part")
    if comp not in _LINES_PER_BLOCK:
        problems.append("compression %s" % _COMPRESSION_NAMES.get(comp, comp))
    return problems


def read_header(path: str) -> dict:
    """Header-only probe: channels, resolution, compression, and whether this
    reader can decode the file. Reads only the header bytes (cheap over network).

    ``width``/``height`` are the display window (the format); ``data_window`` is
    ``[xmin, ymin, xmax, ymax]`` relative to it.
    """
    with open(path, "rb") as fh:
        version, attrs, _ = _header(fh)
    dw = _box(attrs, "dataWindow")
    disp = _box(attrs, "displayWindow") if "displayWindow" in attrs else dw
    comp = attrs["compression"][0] if "compression" in attrs else -1
    problems = _problems(version, comp)
    return {
        "channels": [c[0] for c in _channel_list(attrs)],
        "width": disp[2] - disp[0] + 1,
        "height": disp[3] - disp[1] + 1,
        "data_window": [dw[0] - disp[0], dw[1] - disp[1], dw[2] - disp[0], dw[3] - disp[1]],
        "compression": _COMPRESSION_NAMES.get(comp, str(comp)),
        "readable": not problems,
        "problems": problems,
    }


def list_channels(path: str) -> List[str]:
    return read_header(path)["channels"]


def _rle_decompress(b: bytes) -> bytes:
    out = bytearray(); i = 0; n = len(b)
    while i < n:
        c = b[i] - 256 if b[i] > 127 else b[i]; i += 1
        if c < 0:
            cnt = -c
            out += b[i:i + cnt]; i += cnt
        else:
            out += bytes([b[i]]) * (c + 1); i += 1
    return bytes(out)


def _unpredict_reorder(dec: bytes) -> np.ndarray:
    a = np.frombuffer(dec, dtype=np.uint8).copy()
    a[1:] -= np.uint8(128)
    a = np.cumsum(a, dtype=np.uint8)                     # predictor (uint8 wraps mod 256)
    size = a.size; half = (size + 1) // 2
    out = np.empty(size, np.uint8)                       # reorder (deinterleave)
    out[0::2] = a[:half]
    out[1::2] = a[half:]
    return out


def _resolve(wanted: List[str], names: List[str]) -> Dict[str, str]:
    """Map each requested name (full or suffix like n_u) to an actual channel."""
    out = {}
    for w in wanted:
        if w in names:
            out[w] = w
        else:
            m = [n for n in names if n == w or n.endswith("." + w) or n.endswith(w)]
            if m:
                out[w] = m[0]
    return out


def read_channels(path: str, wanted: List[str]) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Read the requested channels as {requested_name: (H,W) float32}, top-first.

    (H, W) is the display window; pixels outside the data window are NaN.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    version, attrs, off = _attrs(data)
    comp = attrs["compression"][0]
    problems = _problems(version, comp)
    if problems:
        raise ValueError("EXR not supported (%s) - re-write it as Zip (16 scanline): %s"
                         % (", ".join(problems), path))
    chans = _channel_list(attrs)
    xmin, ymin, xmax, ymax = _box(attrs, "dataWindow")
    W, H = xmax - xmin + 1, ymax - ymin + 1
    names = [c[0] for c in chans]
    resolved = _resolve(wanted, names)                   # requested -> actual
    want_actual = set(resolved.values())

    lines_per = _LINES_PER_BLOCK[comp]
    n_blocks = (H + lines_per - 1) // lines_per
    offsets = struct.unpack_from("<%dQ" % n_blocks, data, off)

    row_bytes = sum(c[2] * W for c in chans)             # all channels, one scanline
    out = {a: np.empty((H, W), np.float32) for a in want_actual}

    for b in range(n_blocks):
        bo = offsets[b]
        y0 = struct.unpack_from("<i", data, bo)[0]
        ds = struct.unpack_from("<i", data, bo + 4)[0]
        raw = data[bo + 8:bo + 8 + ds]
        nlines = min(lines_per, ymax - y0 + 1)
        expected = row_bytes * nlines
        if ds >= expected:                               # block stored uncompressed
            buf = raw
        elif comp == 1:
            buf = _unpredict_reorder(_rle_decompress(raw))
        else:                                            # ZIPS / ZIP
            buf = _unpredict_reorder(zlib.decompress(raw))

        # Line order only sets the order of the blocks in the file; each block
        # carries its own y and its lines are always stored top to bottom.
        pos = 0
        for li in range(nlines):
            row = (y0 - ymin) + li
            for cn, pixtype, size in chans:
                nb = W * size
                if cn in out:
                    seg = buf[pos:pos + nb]
                    if pixtype == _HALF:
                        out[cn][row] = np.frombuffer(seg, np.float16).astype(np.float32)
                    elif pixtype == _FLOAT:
                        out[cn][row] = np.frombuffer(seg, np.float32)
                    else:
                        out[cn][row] = np.frombuffer(seg, np.uint32).astype(np.float32)
                pos += nb

    # place the data window into the display window (crop overscan, NaN outside)
    dx0, dy0, dx1, dy1 = _box(attrs, "displayWindow") if "displayWindow" in attrs else (xmin, ymin, xmax, ymax)
    DW, DH = dx1 - dx0 + 1, dy1 - dy0 + 1
    if (xmin, ymin, xmax, ymax) != (dx0, dy0, dx1, dy1):
        x0, x1 = max(xmin, dx0), min(xmax, dx1)
        y0, y1 = max(ymin, dy0), min(ymax, dy1)
        for a in list(out):
            full = np.full((DH, DW), np.nan, np.float32)
            if x0 <= x1 and y0 <= y1:
                full[y0 - dy0:y1 - dy0 + 1, x0 - dx0:x1 - dx0 + 1] = \
                    out[a][y0 - ymin:y1 - ymin + 1, x0 - xmin:x1 - xmin + 1]
            out[a] = full
    return {req: out[act] for req, act in resolved.items()}, DW, DH
