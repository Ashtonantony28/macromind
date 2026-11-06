import numpy as np
import pytest

from macromind.core import generate_synthetic
from macromind.cost import proxy_cost
from macromind.io import read_def, read_lef
from macromind.legalize import LegalizationError, SnapGrid, check_legality, choose_orientations, is_legal, legalize


def _random_input(d, rng):
    c = d.core
    p = d.placement()
    mov = np.array([not m.fixed for m in d.macros])
    # anywhere, including partly or fully outside the core, off-grid
    p.x[mov] = rng.uniform(c.xl - 0.2 * c.width, c.xh, mov.sum()).astype(np.int64)
    p.y[mov] = rng.uniform(c.yl - 0.2 * c.height, c.yh, mov.sum()).astype(np.int64)
    p.orient[mov] = rng.choice([0, 2, 4, 6], mov.sum())
    return p


@pytest.mark.parametrize("seed", range(40))
def test_property_overlap_free_in_bounds_on_grid(seed):
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(2, 40))
    d = generate_synthetic(
        n, int(rng.integers(10, 100)), 8, seed=seed,
        utilization=float(rng.uniform(0.4, 0.75)), macro_area_fraction=float(rng.uniform(0.2, 0.6)),
        halo_um=float(rng.choice([0.0, 1.0, 5.0])), n_fixed_macros=int(rng.integers(0, 3)),
    )
    before_fixed = [(m.x, m.y, m.orient) for m in d.macros if m.fixed]
    p = _random_input(d, rng)
    out = legalize(d, p)
    assert check_legality(d, out) == []
    c = d.core
    w = np.array([m.width for m in d.macros])
    h = np.array([m.height for m in d.macros])
    assert (out.x >= c.xl).all() and (out.x + w <= c.xh).all()
    assert (out.y >= c.yl).all() and (out.y + h <= c.yh).all()
    g = SnapGrid.of(d)
    mov = np.array([not m.fixed for m in d.macros])
    assert ((out.x[mov] - g.x0) % g.px == 0).all() and ((out.y[mov] - g.y0) % g.py == 0).all()
    fixed_after = [(int(out.x[i]), int(out.y[i]), int(out.orient[i])) for i, m in enumerate(d.macros) if m.fixed]
    assert fixed_after == before_fixed
    # footprint-preserving orientations only
    assert ((out.orient & 1) == (p.orient & 1)).all()


def test_legal_input_is_unchanged():
    d = generate_synthetic(20, 50, seed=4)
    p = legalize(d)
    q = legalize(d, p, choose_orientation=False)
    assert np.array_equal(p.x, q.x) and np.array_equal(p.y, q.y)


def test_minimal_displacement_two_macros():
    d = generate_synthetic(2, 10, seed=0, halo_um=1.0)
    g = SnapGrid.of(d)
    c = d.core
    big = 0 if d.macros[0].area >= d.macros[1].area else 1
    small = 1 - big
    mb, ms = d.macros[big], d.macros[small]
    p = d.placement()
    x0, y0 = int(g.near_x(c.cx - mb.width / 2)), int(g.near_y(c.cy - mb.height / 2))
    p.x[:] = x0
    p.y[:] = y0
    out = legalize(d, p, choose_orientation=False)
    assert (out.x[big], out.y[big]) == (x0, y0)  # larger macro is committed first and stays
    disp = np.hypot(out.x[small] - x0, out.y[small] - y0)
    # nearest escape is along one axis: past the big macro or its own size, plus halo
    bound = min(mb.width, mb.height, ms.width, ms.height) + d.halo + max(g.px, g.py)
    assert disp <= bound
    assert is_legal(d, out)


def test_impossible_raises():
    d = generate_synthetic(4, 10, seed=0, utilization=0.6)
    for m in d.macros:  # make every macro nearly core-sized
        m.width = int(d.core.width * 0.6) // d.site_width * d.site_width
        m.height = int(d.core.height * 0.6) // d.row_height * d.row_height
    d.touch()
    with pytest.raises(LegalizationError):
        legalize(d)


def test_orientation_choice_never_increases_hpwl():
    for seed in range(5):
        d = generate_synthetic(15, 80, seed=seed)
        p = legalize(d, choose_orientation=False)
        q = choose_orientations(d, p)
        assert proxy_cost(d, q).hpwl <= proxy_cost(d, p).hpwl + 1e-6
        assert np.array_equal(p.x, q.x) and np.array_equal(p.y, q.y)


def test_legalize_def_fixture(fixtures):
    d = read_def(fixtures / "tiny.def", read_lef(fixtures / "tiny.lef"), halo=2000)
    out = legalize(d)
    assert check_legality(d, out) == []
    assert (out.x[1], out.y[1], out.orient[1]) == (100000, 40000, 6)  # fixed macro untouched


def test_check_legality_reports():
    d = generate_synthetic(3, 10, seed=0)
    p = d.placement()
    p.x[:] = int(d.core.xl)
    p.y[:] = int(d.core.yl)
    errs = check_legality(d, p)
    assert any("overlaps" in e for e in errs)
    p.x[0] = int(d.core.xl) - 10
    assert any("outside core" in e for e in check_legality(d, p))
    p.x[0] = int(d.core.xl) + 1
    assert any("off grid" in e for e in check_legality(d, p))
