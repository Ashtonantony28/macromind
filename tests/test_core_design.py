import numpy as np
import pytest

from macromind.core import (
    Cell, Design, Macro, Net, Orient, PinRef, Placement, Port, Rect, Site,
    generate_synthetic, oriented_dims, transform_offsets,
)


def _ref_transform(dx, dy, o):
    """Reference: mirror (if flipped) then rotate counter-clockwise, via complex numbers."""
    z = complex(dx, dy)
    name = Orient(o).name
    rot = {"N": 0, "W": 1, "S": 2, "E": 3, "FN": 0, "FE": 1, "FS": 0, "FW": 1}[name]
    if name in ("FN", "FE"):
        z = complex(-z.real, z.imag)  # MY
    if name in ("FS", "FW"):
        z = complex(z.real, -z.imag)  # MX
    z *= 1j**rot
    return z.real, z.imag


@pytest.mark.parametrize("o", list(range(8)))
def test_orientation_transforms_match_reference(o):
    rng = np.random.default_rng(o)
    dx, dy = rng.normal(size=20), rng.normal(size=20)
    tx, ty = transform_offsets(dx, dy, o)
    for a, b, x, y in zip(dx, dy, tx, ty):
        rx, ry = _ref_transform(a, b, o)
        assert x == pytest.approx(rx) and y == pytest.approx(ry)


def test_orient_parse_and_dims():
    assert Orient.parse("FS") == Orient.FS
    assert Orient.parse("R90") == Orient.W
    assert Orient.parse("MX") == Orient.FS
    assert Orient.parse(4) == Orient.FN
    w, h = oriented_dims(np.array([3, 3]), np.array([5, 5]), np.array([0, 1]))
    assert list(w) == [3, 5] and list(h) == [5, 3]
    assert Orient.E.swaps_dims and not Orient.FS.swaps_dims


def test_rect_ops():
    a, b = Rect(0, 0, 10, 10), Rect(10, 0, 20, 10)
    assert not a.overlaps(b)  # touching
    assert a.expanded(1).overlaps(b)
    assert Rect(-1, -1, 30, 30).contains(a)
    assert a.area == 100 and a.cx == 5


def _tiny_design():
    m = Macro("m0", 10, 20, 0, 0, Orient.N, pins={"a": (0.0, 0.0), "b": (10.0, 20.0)})
    c = Cell("c0", 4, 4, 50, 50, is_cluster=True)
    p = Port("io", 100.0, 0.0)
    nets = [
        Net("n0", [PinRef("m0", "a"), PinRef("c0", "")]),
        Net("n1", [PinRef("m0", "b"), PinRef("io", "", True)], weight=2.0),
        Net("single", [PinRef("c0", "")]),  # dropped (< 2 pins)
        Net("dangling", [PinRef("nope", "x"), PinRef("c0", "")]),  # unknown ref dropped -> 1 pin -> dropped
    ]
    return Design("t", 1000, Rect(0, 0, 100, 100), Rect(0, 0, 100, 100), Site("s", 1, 1), [], [m], [c], [p], nets)


def test_arrays_topology():
    d = _tiny_design()
    a = d.arrays()
    assert (a.n_macros, a.n_cells, a.n_ports) == (1, 1, 1)
    assert a.n_nets == 2 and list(a.net_orig) == [0, 1]
    assert list(a.net_ptr) == [0, 2, 4]
    assert list(a.pin_node) == [0, 1, 0, 2]
    # centre-relative offsets
    assert (a.pin_dx[0], a.pin_dy[0]) == (-5.0, -10.0)
    assert (a.pin_dx[2], a.pin_dy[2]) == (5.0, 10.0)
    assert list(a.net_weight) == [1.0, 2.0]
    # node->pins and node->nets CSR
    pins_m0 = a.node_pin_idx[a.node_pin_ptr[0] : a.node_pin_ptr[1]]
    assert sorted(pins_m0) == [0, 2]
    assert list(a.nets_of_nodes([0])) == [0, 1]
    assert list(a.nets_of_nodes([1, 2])) == [0, 1]
    idx, seg = a.pins_of_nets(np.array([1, 0]))
    assert list(idx) == [2, 3, 0, 1] and list(seg) == [0, 2]


def test_placement_roundtrip_and_apply():
    d = generate_synthetic(8, 20, seed=3)
    a = d.arrays()
    rng = np.random.default_rng(0)
    p = Placement(rng.integers(0, 10**6, 8), rng.integers(0, 10**6, 8), rng.integers(0, 8, 8))
    cx, cy, o = a.node_state(p)
    q = a.macro_centres_to_placement(cx, cy, o)
    assert np.array_equal(p.x, q.x) and np.array_equal(p.y, q.y) and np.array_equal(p.orient, q.orient)
    d2 = d.apply_placement(p)
    assert np.array_equal(d2.placement().x, p.x)
    assert d.macros[0].x != d2.macros[0].x or d.macros[0].y != d2.macros[0].y or p.x[0] == d.macros[0].x
    assert d2.arrays() is a  # topology shared
    with pytest.raises(ValueError):
        d.apply_placement(Placement([0], [0], [0]))


def test_macro_rect_oriented():
    m = Macro("m", 10, 30, 5, 7, Orient.E)
    r = m.rect()
    assert (r.width, r.height) == (30, 10) and (r.xl, r.yl) == (5, 7)
