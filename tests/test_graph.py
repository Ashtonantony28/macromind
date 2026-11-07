"""SG5: clustering, quadratic cluster placement and graph tensors."""

import copy
import glob
from pathlib import Path

import numpy as np
import pytest
import torch

from macromind.baselines import grid_place, random_place
from macromind.core import generate_synthetic
from macromind.cost import proxy_cost
from macromind.graph import (
    N_FEATURES, QPConfig, build_graph, cluster_cells, cluster_hpwl, hierarchy_groups, hierarchy_path, place_clusters,
)


def flat_design(n_macros=4, n_cells=600, seed=0, unplaced=True):
    """Synthetic design whose 'clusters' act as flat std cells with hierarchical names."""
    d = generate_synthetic(n_macros, n_cells, seed=seed)
    d = copy.copy(d)
    d.macros = [copy.copy(m) for m in d.macros]
    p = grid_place(d)
    d = d.apply_placement(p)
    mapping = {}
    cells = []
    rng = np.random.default_rng(seed)
    core = d.core
    for j, c in enumerate(d.cells):
        c = copy.copy(c)
        # hierarchy correlated with (generator) locality, with 25% noise
        q = int(c.x + c.width / 2 > core.cx) + 2 * int(c.y + c.height / 2 > core.cy)
        if rng.random() < 0.25:
            q = int(rng.integers(4))
        new = f"u_top/u_blk{q}/u_sub{j % 2}/_{j}_"
        mapping[c.name] = new
        c.name, c.is_cluster = new, False
        if unplaced:
            c.x, c.y, c.placed = 0, 0, False
        cells.append(c)
    d.cells = cells
    for net in d.nets:
        net.pins = [r._replace(node=mapping.get(r.node, r.node)) if not r.is_port else r for r in net.pins]
    d.touch()
    return d


def test_hierarchy_helpers():
    assert hierarchy_path("a/b/c") == ("a", "b")
    assert hierarchy_path("\\a.b._1_") == ("a", "b")
    assert hierarchy_path("leaf") == ()
    g = hierarchy_groups(["a/x/1", "a/y/2", "b/z/3", "b/z/4"], max_groups=2)
    assert g[0] != g[2] and g[2] == g[3]
    g3 = hierarchy_groups(["a/x/1", "a/y/2", "b/z/3", "b/z/4"], max_groups=3)
    assert len(set(g3.tolist())) == 3


def test_cluster_assigns_all_and_preserves_area():
    d = flat_design()
    cl = cluster_cells(d, 40, seed=0)
    assert np.all(cl.assignment >= 0)  # every synthetic cell is connected
    assert 30 <= cl.n_clusters <= 40
    area = np.array([c.width * c.height for c in d.cells], dtype=float)
    np.testing.assert_allclose(np.bincount(cl.assignment, weights=area), cl.cluster_area)
    assert cl.cluster_area.sum() == pytest.approx(area.sum())
    red_area = sum(c.width * c.height for c in cl.design.cells)
    assert red_area == pytest.approx(area.sum(), rel=0.03)
    assert len(cl.design.macros) == len(d.macros) and len(cl.design.ports) == len(d.ports)
    # area balance
    assert cl.cluster_area.max() <= 2.5 * area.sum() / 40 * 8 + 1


def test_cluster_follows_hierarchy():
    # names are only partly correlated with connectivity; hierarchy affinity must raise purity
    d = flat_design()
    blk = np.array([int(c.name.split("/")[1][5:]) for c in d.cells])

    def purity(hw):
        cl = cluster_cells(d, 24, hier_weight=hw, seed=1, initial_place=False)
        return np.mean([np.bincount(blk[cl.members(k)]).max() / len(cl.members(k)) for k in range(cl.n_clusters)])

    p0, p1 = purity(0.0), purity(20.0)
    assert p1 > p0 + 0.08 and p1 > 0.74, (p0, p1)  # ceiling ~0.81 with 25% label noise


def test_unplaced_cells_get_real_positions():
    d = flat_design(unplaced=True)
    cl = cluster_cells(d, 40)
    r = cl.design
    c = r.core
    xs = np.array([cc.x + cc.width / 2 for cc in r.cells])
    ys = np.array([cc.y + cc.height / 2 for cc in r.cells])
    assert all(cc.placed for cc in r.cells)
    assert xs.std() > 0.05 * c.width and ys.std() > 0.05 * c.height  # not piled at a point
    for cc in r.cells:
        assert c.xl <= cc.x and cc.x + cc.width <= c.xh and c.yl <= cc.y and cc.y + cc.height <= c.yh
    assert np.isfinite(proxy_cost(r).proxy)


def test_dropped_unconnected_cells():
    d = flat_design(n_cells=100)
    from macromind.core import Cell

    d.cells = d.cells + [Cell("TAP_1", 380, 2800, 0, 0, fixed=True)]
    d.touch()
    cl = cluster_cells(d, 20)
    assert cl.assignment[-1] == -1 and cl.stats["dropped"] == 1


def test_identity_when_target_exceeds_cells():
    d = generate_synthetic(5, 30, seed=3)
    cl = cluster_cells(d, 100, initial_place=False)
    assert cl.n_clusters == 30
    assert sorted(cl.assignment.tolist()) == list(range(30))
    assert [c.width for c in cl.design.cells] == [d.cells[i].width for i in np.argsort(cl.assignment)]


def test_place_clusters_in_core_and_lowers_hpwl():
    d = generate_synthetic(10, 120, seed=4)
    p = random_place(d, seed=0)
    rng = np.random.default_rng(0)
    d = copy.copy(d)
    c = d.core
    d.cells = [copy.copy(x) for x in d.cells]
    for x in d.cells:  # scramble clusters
        x.x = int(rng.uniform(c.xl, c.xh - x.width))
        x.y = int(rng.uniform(c.yl, c.yh - x.height))
    d.touch()
    before = cluster_hpwl(d, p)
    out = place_clusters(d, p)
    after = cluster_hpwl(out, p)
    assert after < 0.7 * before
    for x in out.cells:
        assert c.xl <= x.x and x.x + x.width <= c.xh and c.yl <= x.y and x.y + x.height <= c.yh
    # macros untouched, input design untouched
    assert np.array_equal(out.placement().x, p.x)
    assert d.cells[0].x != out.cells[0].x or d.cells[1].x != out.cells[1].x
    # arrays reflect new positions
    assert out.arrays().node_cx[d.n_macros] == pytest.approx(out.cells[0].x + out.cells[0].width / 2)


def test_spreading_reduces_density():
    d = generate_synthetic(12, 150, seed=5)
    p = grid_place(d)
    tight = proxy_cost(place_clusters(d, p, spread=False), p)
    spread = proxy_cost(place_clusters(d, p), p)
    assert spread.density < tight.density
    assert spread.hpwl < 1.5 * tight.hpwl


def test_place_clusters_deterministic_and_fast():
    import time

    d = generate_synthetic(20, 200, seed=6)
    p = grid_place(d)
    t = time.perf_counter()
    a = place_clusters(d, p)
    assert time.perf_counter() - t < 2.0
    b = place_clusters(d, p)
    assert [x.x for x in a.cells] == [x.x for x in b.cells]


@pytest.mark.parametrize("model", ["clique", "star"])
def test_graph_tensors(model):
    d = generate_synthetic(8, 60, seed=7)
    g = build_graph(d, model)
    arr = d.arrays()
    n = arr.n_nodes + (arr.n_nets if model == "star" else 0)
    assert g.x.shape == (n, N_FEATURES) and g.x.dtype == torch.float32
    assert g.edge_index.shape[0] == 2 and g.edge_index.shape[1] == g.edge_weight.shape[0] > 0
    assert int(g.edge_index.max()) < n and bool((g.edge_weight > 0).all())
    # symmetric
    e = set(map(tuple, g.edge_index.T.tolist()))
    assert all((b, a) in e for a, b in e)
    assert not bool((g.edge_index[0] == g.edge_index[1]).any())
    assert torch.isfinite(g.x).all()
    # movable macro positions hidden
    assert float(g.x[: g.n_macros, 12].sum()) == 0.0
    assert g.n_macros == 8 and g.n_cells == 60


def test_clique_weights_sum():
    d = generate_synthetic(4, 20, seed=8)
    g = build_graph(d, "clique", clique_max=1000)
    arr = d.arrays()
    k = np.diff(arr.net_ptr)
    # each net contributes w * k/2 per direction (k(k-1)/2 pairs * w/(k-1)), minus self pairs (same node twice)
    expected = float((arr.net_weight * k / 2).sum()) * 2
    assert float(g.edge_weight.sum()) <= expected + 1e-3
    assert float(g.edge_weight.sum()) > 0.8 * expected


ROOT = Path(__file__).resolve().parents[1]
GCD = ROOT / "runs/orfs/smoke/results/nangate45/gcd/base/6_final.def"


@pytest.mark.skipif(not GCD.exists(), reason="gcd ORFS run not available")
def test_real_def_unplaced_cells():
    from macromind.io.defio import read_def

    lefs = glob.glob(str(ROOT / "tools/ORFS/flow/platforms/nangate45/lef/Nangate*.lef"))
    if not lefs:
        pytest.skip("nangate45 LEFs not available")
    d = read_def(GCD, lefs)
    for c in d.cells:
        if not c.fixed:
            c.x, c.y, c.placed = 0, 0, False
    d.touch()
    cl = cluster_cells(d, 32)
    assert cl.stats["dropped"] > 0  # fillers / tapcells
    xs = np.array([c.x for c in cl.design.cells])
    assert (xs > 0).mean() > 0.9
