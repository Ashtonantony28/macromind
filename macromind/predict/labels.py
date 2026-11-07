"""Label parsers and label summaries for the predictor datasets.

* :func:`parse_endpoint_slack` -- OpenSTA ``report_checks -format end`` output
  (written by ``grt_labels.tcl``) -> ``{endpoint: slack_ns}``.
* :func:`congestion_maps` -- ``gcells.npz`` (GRT capacity/usage per gcell,
  summed over H and V layers) -> per-bin H/V utilisation maps on a coarse grid.
* :func:`congestion_summary` -- scalar congestion labels (mean / top-k utilisation,
  overflowing gcells) used for ranking placements.
"""

from __future__ import annotations

import re

import numpy as np

_EP_LINE = re.compile(r"^(\S+)\s+\(([^)]*)\)\s+(-?[\d.]+(?:e-?\d+)?)\s+(-?[\d.]+(?:e-?\d+)?)\s+(-?[\d.]+(?:e-?\d+)?)")
_EP_NOTYPE = re.compile(r"^(\S+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+\((?:MET|VIOLATED)\)")


def parse_endpoint_slack(text: str) -> dict[str, float]:
    """``{endpoint pin: worst setup slack (ns)}`` from ``report_checks -format end``.

    Handles several path groups (the worst slack per endpoint wins) and lines
    whose endpoint name is too long for the column (OpenSTA then just pads with
    one space). Unconstrained / header lines are skipped.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("Endpoint", "---", "max_delay", "min_delay", "Required")):
            continue
        m = _EP_LINE.match(line)
        if m:
            name, slack = m.group(1), float(m.group(5))
        else:
            m = _EP_NOTYPE.match(line)
            if not m:
                continue
            name, slack = m.group(1), float(m.group(4))
        if name not in out or slack < out[name]:
            out[name] = slack
    return out


def endpoint_instance(pin: str) -> tuple[str, bool]:
    """``(instance name, is_port)`` of an endpoint pin name (``inst/D`` or a top-level port)."""
    if "/" not in pin:
        return pin, True
    return pin.rsplit("/", 1)[0], False


def _pool(a: np.ndarray, edges_y: np.ndarray, edges_x: np.ndarray) -> np.ndarray:
    """Sum ``a`` (ny, nx) into bins given by index edges."""
    s = np.add.reduceat(a, edges_x[:-1], axis=1)
    return np.add.reduceat(s, edges_y[:-1], axis=0)


def congestion_maps(g: dict, nx: int = 32, ny: int = 32, core: tuple[float, float, float, float] | None = None) -> dict[str, np.ndarray]:
    """Coarse ``(ny, nx)`` H/V utilisation (usage / capacity) and overflow maps from a gcells.npz dict.

    Bins cover ``core`` (DBU, ``(xl, yl, xh, yh)``; default: the die). Gcells are
    assigned to bins by centre; empty bins get 0.
    """
    gx = np.asarray(g["grid_x"], np.float64)
    gy = np.asarray(g["grid_y"], np.float64)
    die = np.asarray(g["die"], np.float64)
    xl, yl, xh, yh = core if core is not None else tuple(die)
    xr = np.append(gx, die[2])
    yr = np.append(gy, die[3])
    cx = 0.5 * (xr[:-1] + xr[1:])
    cy = 0.5 * (yr[:-1] + yr[1:])
    bx = np.clip(((cx - xl) / (xh - xl) * nx).astype(int), 0, nx - 1)
    by = np.clip(((cy - yl) / (yh - yl) * ny).astype(int), 0, ny - 1)
    inside = ((cx >= xl) & (cx < xh))[None, :] & ((cy >= yl) & (cy < yh))[:, None]
    out = {}
    for d in ("h", "v"):
        cap = np.where(inside, g[f"cap_{d}"], 0.0)
        use = np.where(inside, g[f"use_{d}"], 0.0)
        over = np.where(inside, np.maximum(g[f"use_{d}"] - g[f"cap_{d}"], 0.0), 0.0)
        C = np.zeros((ny, nx))
        U = np.zeros((ny, nx))
        O = np.zeros((ny, nx))
        np.add.at(C, (by[:, None].repeat(len(bx), 1), bx[None, :].repeat(len(by), 0)), cap)
        np.add.at(U, (by[:, None].repeat(len(bx), 1), bx[None, :].repeat(len(by), 0)), use)
        np.add.at(O, (by[:, None].repeat(len(bx), 1), bx[None, :].repeat(len(by), 0)), over)
        out[f"util_{d}"] = np.where(C > 0, U / np.maximum(C, 1e-9), 0.0).astype(np.float32)
        out[f"overflow_{d}"] = O.astype(np.float32)
        # usage relative to the mean bin capacity: sums to (total usage / mean capacity), a wirelength-like quantity
        out[f"use_{d}"] = (U / max(C.mean(), 1e-9)).astype(np.float32)
    return out


def congestion_summary(g: dict, topk: float = 0.05) -> dict[str, float]:
    """Scalar congestion labels at gcell resolution: mean util, mean of the top-``topk`` gcells, overflow."""
    res = {}
    for d in ("h", "v"):
        cap, use = np.asarray(g[f"cap_{d}"], np.float64), np.asarray(g[f"use_{d}"], np.float64)
        ok = cap > 0
        u = use[ok] / cap[ok]
        k = max(1, int(round(topk * u.size)))
        res[f"util_mean_{d}"] = float(u.mean()) if u.size else 0.0
        res[f"util_top_{d}"] = float(np.sort(u)[-k:].mean()) if u.size else 0.0
        res[f"overflow_{d}"] = float(np.maximum(use - cap, 0).sum())
        res[f"hot_frac_{d}"] = float((u >= 0.9).mean()) if u.size else 0.0
    res["util_top"] = 0.5 * (res["util_top_h"] + res["util_top_v"])
    res["util_mean"] = 0.5 * (res["util_mean_h"] + res["util_mean_v"])
    res["overflow"] = res["overflow_h"] + res["overflow_v"]
    return res
