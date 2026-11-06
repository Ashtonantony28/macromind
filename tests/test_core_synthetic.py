import numpy as np

from macromind.core import generate_synthetic


def test_deterministic_and_seed_dependent():
    a = generate_synthetic(10, 50, seed=7)
    b = generate_synthetic(10, 50, seed=7)
    c = generate_synthetic(10, 50, seed=8)
    assert [m.width for m in a.macros] == [m.width for m in b.macros]
    assert [[tuple(p) for p in n.pins] for n in a.nets] == [[tuple(p) for p in n.pins] for n in b.nets]
    assert [m.width for m in a.macros] != [m.width for m in c.macros]


def test_counts_geometry_and_connectivity():
    d = generate_synthetic(30, 300, 24, seed=1)
    assert (len(d.macros), len(d.cells), len(d.ports)) == (30, 300, 24)
    core, die = d.core, d.die
    assert die.contains(core)
    for m in d.macros:
        assert m.width <= core.width and m.height <= core.height
        assert m.width % d.site_width == 0 and m.height % d.row_height == 0
        for dx, dy in m.pins.values():
            assert 0 <= dx <= m.width and 0 <= dy <= m.height
    for c in d.cells:
        assert core.contains(c.rect())
    for p in d.ports:
        assert p.x in (die.xl, die.xh) or p.y in (die.yl, die.yh)
    a = d.arrays()
    deg = np.diff(a.node_net_ptr)
    assert (deg > 0).all(), "every node is connected"
    assert a.n_nets >= 300
    # utilisation is roughly as requested
    area = sum(m.area for m in d.macros) + sum(c.area for c in d.cells)
    assert 0.45 < area / core.area < 0.7
    assert d.hroutes_per_micron > 0 and d.vroutes_per_micron > 0


def test_connectivity_is_local():
    """Rent-style locality: connected clusters are closer than random pairs."""
    d = generate_synthetic(0, 400, 0, seed=2, macro_area_fraction=0.0)
    a = d.arrays()
    cx, cy = a.node_cx, a.node_cy
    spans = []
    for k in range(a.n_nets):
        nodes = a.pin_node[a.net_ptr[k] : a.net_ptr[k + 1]]
        spans.append(np.ptp(cx[nodes]) + np.ptp(cy[nodes]))
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, 400, (2, 2000))
    rand = np.abs(cx[i] - cx[j]) + np.abs(cy[i] - cy[j])
    assert np.median(spans) < 0.6 * np.median(rand)


def test_fixed_macros():
    d = generate_synthetic(6, 40, seed=0, n_fixed_macros=2)
    assert [m.fixed for m in d.macros].count(True) == 2
    assert all(d.core.contains(m.rect()) for m in d.macros if m.fixed)
