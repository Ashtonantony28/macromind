"""Cost functions vs. brute-force (loop-based) references."""

import math

import numpy as np
import pytest

from macromind.core import Orient, Placement, generate_synthetic
from macromind.cost import (
    GridSpec, IncrementalProxy, ProxyConfig, axis_overlap, congestion_score, density_map,
    density_overflow, hpwl, proxy_cost, proxy_maps, rudy_maps, topk_mean,
)
from macromind.cost.wirelength import net_bboxes, pin_positions


def _ref_offset(dx, dy, o):
    z = complex(dx, dy)
    name = Orient(o).name
    if name in ("FN", "FE"):
        z = complex(-z.real, z.imag)
    if name in ("FS", "FW"):
        z = complex(z.real, -z.imag)
    z *= 1j ** {"N": 0, "W": 1, "S": 2, "E": 3, "FN": 0, "FE": 1, "FS": 0, "FW": 1}[name]
    return z.real, z.imag


def _random_state(design, rng, all_orients=True):
    """Random macro placement (overlaps allowed) inside the core."""
    c = design.core
    n = len(design.macros)
    o = rng.integers(0, 8 if all_orients else 1, n)
    p = Placement(rng.uniform(c.xl, c.xh - 1, n), rng.uniform(c.yl, c.yh - 1, n), o)
    return p


def _ref_pin_xy(design, placement):
    """Brute-force absolute pin positions, per net, from the object model."""
    nodes = {m.name: m for m in design.macros} | {c.name: c for c in design.cells}
    for i, m in enumerate(design.macros):
        nodes[m.name] = type(m)(**{**m.__dict__, "x": int(placement.x[i]), "y": int(placement.y[i]), "orient": int(placement.orient[i])})
    ports = {p.name: p for p in design.ports}
    out = []
    for net in design.nets:
        pts = []
        for r in net.pins:
            if r.is_port:
                if r.node in ports:
                    pts.append((ports[r.node].x, ports[r.node].y))
                continue
            m = nodes.get(r.node)
            if m is None:
                continue
            w, h = (m.height, m.width) if m.orient & 1 else (m.width, m.height)
            cx, cy = m.x + w / 2, m.y + h / 2
            off = m.pins.get(r.pin, (m.width / 2, m.height / 2))
            ox, oy = _ref_offset(off[0] - m.width / 2, off[1] - m.height / 2, m.orient)
            pts.append((cx + ox, cy + oy))
        if len(pts) >= 2:
            out.append((pts, net.weight))
    return out


@pytest.mark.parametrize("seed", range(5))
def test_hpwl_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    d = generate_synthetic(12, 60, 8, seed=seed)
    p = _random_state(d, rng)
    p = Placement(p.x, p.y, p.orient)  # rounds to int
    ref = 0.0
    for pts, w in _ref_pin_xy(d, p):
        xs, ys = zip(*pts)
        ref += w * (max(xs) - min(xs) + max(ys) - min(ys))
    a = d.arrays()
    cx, cy, o = a.node_state(p)
    assert hpwl(a, cx, cy, o) == pytest.approx(ref, rel=1e-12)
    assert proxy_cost(d, p).hpwl == pytest.approx(ref, rel=1e-12)


def _ref_overlap_area(r, bx0, by0, bx1, by1):
    return max(0.0, min(r[2], bx1) - max(r[0], bx0)) * max(0.0, min(r[3], by1) - max(r[1], by0))


def test_axis_overlap_bruteforce():
    rng = np.random.default_rng(0)
    edges = np.linspace(0, 10, 6)
    lo = rng.uniform(-2, 10, 30)
    hi = lo + rng.uniform(0, 6, 30)
    ov = axis_overlap(lo, hi, edges)
    for k in range(30):
        for b in range(5):
            assert ov[k, b] == pytest.approx(max(0.0, min(hi[k], edges[b + 1]) - max(lo[k], edges[b])))


@pytest.mark.parametrize("seed", range(3))
def test_density_map_bruteforce(seed):
    rng = np.random.default_rng(seed)
    d = generate_synthetic(10, 40, 4, seed=seed)
    p = _random_state(d, rng)
    p = Placement(p.x, p.y, p.orient)
    g = GridSpec.over(d.core, 7, 5)
    a = d.arrays()
    cx, cy, o = a.node_state(p)
    util = density_map(a, cx, cy, o, g)
    rects = [m.rect() for m in d.apply_placement(p).macros] + [c.rect() for c in d.cells]
    ref = np.zeros((7, 5))
    for i in range(7):
        for j in range(5):
            bx0, by0 = g.x0 + i * g.bin_w, g.y0 + j * g.bin_h
            for r in rects:
                ref[i, j] += _ref_overlap_area((r.xl, r.yl, r.xh, r.yh), bx0, by0, bx0 + g.bin_w, by0 + g.bin_h)
    np.testing.assert_allclose(util, ref / g.bin_area, rtol=1e-9, atol=1e-12)
    total = sum(r.area for r in rects)
    ov = density_overflow(util, 0.6, g.bin_area, total)
    ref_ov = sum(max(0.0, u - 0.6) for u in (ref / g.bin_area).ravel()) * g.bin_area / total
    assert ov == pytest.approx(ref_ov)


def test_rudy_bruteforce():
    rng = np.random.default_rng(1)
    g = GridSpec(6, 4, 0.0, 0.0, 60.0, 40.0)
    k = 25
    # nets inside the grid (so total demand is conserved), with min-extent room
    x0 = rng.uniform(5, 30, k)
    x1 = x0 + rng.uniform(0, 25, k)
    y0 = rng.uniform(3, 15, k)
    y1 = y0 + rng.uniform(0, 20, k)
    x1[:3] = x0[:3]  # degenerate (zero-width) nets
    w = rng.uniform(1, 4, k)
    h_map, v_map = rudy_maps(x0, x1, y0, y1, w, g)
    ref_h = np.zeros((6, 4))
    ref_v = np.zeros((6, 4))
    mw, mh = 0.5 * g.bin_w, 0.5 * g.bin_h
    for n in range(k):
        a0, a1, b0, b1 = x0[n], x1[n], y0[n], y1[n]
        if a1 - a0 < mw:
            c = (a0 + a1) / 2
            a0, a1 = c - mw / 2, c + mw / 2
        if b1 - b0 < mh:
            c = (b0 + b1) / 2
            b0, b1 = c - mh / 2, c + mh / 2
        for i in range(6):
            for j in range(4):
                ar = _ref_overlap_area((a0, b0, a1, b1), i * 10, j * 10, i * 10 + 10, j * 10 + 10)
                ref_h[i, j] += w[n] * ar / (b1 - b0)
                ref_v[i, j] += w[n] * ar / (a1 - a0)
    np.testing.assert_allclose(h_map, ref_h, rtol=1e-9)
    np.testing.assert_allclose(v_map, ref_v, rtol=1e-9)
    # total demand is conserved: sum of H demand == sum of weighted (expanded) widths
    assert h_map.sum() == pytest.approx(sum(w[n] * max(x1[n] - x0[n], mw) for n in range(k)))
    cong = congestion_score(h_map, v_map, 2.0, 3.0, topk=0.25)
    allv = np.sort(np.concatenate([(ref_h / 2.0).ravel(), (ref_v / 3.0).ravel()]))[::-1]
    assert cong == pytest.approx(allv[: math.ceil(0.25 * 48)].mean())


def test_topk_mean():
    v = np.arange(10.0)
    assert topk_mean(v, 0.2) == pytest.approx(8.5)
    assert topk_mean(v, 0.01) == 9.0
    assert topk_mean(v, 1.0) == 4.5


def test_proxy_combination_and_maps():
    d = generate_synthetic(10, 80, seed=0)
    cfg = ProxyConfig(grid_nx=16, w_wirelength=1.0, w_density=0.3, w_congestion=0.7)
    r = proxy_cost(d, None, cfg)
    assert r.proxy == pytest.approx(r.wirelength + 0.3 * r.density + 0.7 * r.congestion)
    a = d.arrays()
    assert r.wirelength == pytest.approx(r.hpwl / ((d.core.width + d.core.height) * a.net_weight.sum()))
    maps = proxy_maps(d, None, cfg)
    assert maps["utilization"].shape == (16, 16)
    allv = np.concatenate([maps["congestion_h"].ravel(), maps["congestion_v"].ravel()])
    assert r.congestion == pytest.approx(topk_mean(allv, cfg.congestion_topk))
    rt = proxy_cost(d, None, ProxyConfig(grid_nx=16, density_mode="topk"))
    assert rt.density == pytest.approx(topk_mean(maps["utilization"], 0.1))
    with pytest.raises(ValueError):
        proxy_cost(d, None, ProxyConfig(density_mode="bogus"))


@pytest.mark.parametrize("seed", range(4))
def test_incremental_matches_full(seed):
    """Random sequences of shift/swap/flip proposals (accepted or rejected) stay
    in agreement with a full re-evaluation."""
    rng = np.random.default_rng(seed)
    d = generate_synthetic(15, 120, 12, seed=seed)
    cfg = ProxyConfig(grid_nx=12)
    p = _random_state(d, rng)
    p = Placement(p.x, p.y, p.orient)
    inc = IncrementalProxy(d, p, cfg)
    c = d.core
    for step in range(300):
        kind = rng.integers(3)
        if kind == 0:
            nodes = [int(rng.integers(15))]
            cx, cy, o = [rng.uniform(c.xl, c.xh)], [rng.uniform(c.yl, c.yh)], [int(inc.orient[nodes[0]])]
        elif kind == 1:
            a, b = (int(v) for v in rng.choice(15, 2, replace=False))
            nodes = [a, b]
            cx, cy, o = [inc.cx[b], inc.cx[a]], [inc.cy[b], inc.cy[a]], [int(inc.orient[a]), int(inc.orient[b])]
        else:
            nodes = [int(rng.integers(15))]
            cx, cy, o = [inc.cx[nodes[0]]], [inc.cy[nodes[0]]], [int(rng.integers(8))]
        new = inc.propose(nodes, cx, cy, o)
        # the proposed value equals a from-scratch evaluation of the proposed state
        if step % 25 == 0:
            ncx, ncy, no = inc.cx.copy(), inc.cy.copy(), inc.orient.copy()
            ncx[nodes], ncy[nodes], no[nodes] = cx, cy, o
            ref = _full_from_state(d, cfg, ncx, ncy, no)
            assert new == pytest.approx(ref.proxy, rel=1e-9, abs=1e-12)
        if rng.random() < 0.5:
            inc.accept()
        else:
            inc.reject()
    cur = inc.result
    ref = _full_from_state(d, cfg, inc.cx, inc.cy, inc.orient)
    for k in ("hpwl", "wirelength", "density", "congestion", "proxy"):
        assert getattr(cur, k) == pytest.approx(getattr(ref, k), rel=1e-9, abs=1e-12), k
    # pin array was correctly restored after rejects
    a = d.arrays()
    px, py = pin_positions(a, inc.cx, inc.cy, inc.orient)
    np.testing.assert_allclose(inc.px, px)
    np.testing.assert_allclose(inc.py, py)
    x0, x1, y0, y1 = net_bboxes(a, px, py)
    np.testing.assert_allclose(inc.bx0, x0)
    np.testing.assert_allclose(inc.by1, y1)


def _full_from_state(d, cfg, cx, cy, o):
    """Full evaluation for arbitrary (float) centres, via a fresh evaluator."""
    inc = IncrementalProxy(d, None, cfg)
    inc.cx, inc.cy, inc.orient = cx.copy(), cy.copy(), o.copy()
    return inc.resync()


def test_incremental_protocol_errors():
    d = generate_synthetic(3, 10, seed=0)
    inc = IncrementalProxy(d)
    with pytest.raises(RuntimeError):
        inc.accept()
    with pytest.raises(RuntimeError):
        inc.reject()
    inc.propose([0], [inc.cx[0]], [inc.cy[0]], [0])
    with pytest.raises(RuntimeError):
        inc.propose([0], [inc.cx[0]], [inc.cy[0]], [0])
    inc.reject()


def test_zero_weight_components_skip_work():
    d = generate_synthetic(5, 30, seed=0)
    inc = IncrementalProxy(d, None, ProxyConfig(w_density=0, w_congestion=0))
    before = inc.rh.copy()
    v = inc.propose([0], [inc.cx[0] + 1000], [inc.cy[0]], [0])
    inc.accept()
    assert v == pytest.approx(inc.hpwl / inc.wl_norm)
    np.testing.assert_array_equal(inc.rh, before)
