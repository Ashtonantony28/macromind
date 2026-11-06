import subprocess
import sys
import time

import numpy as np
import pytest

from macromind.baselines import PLACERS, SAConfig, get_placer, place, sa_place
from macromind.baselines.__main__ import main as cli_main
from macromind.core import generate_synthetic
from macromind.cost import proxy_cost
from macromind.legalize import check_legality


@pytest.fixture(scope="module")
def design():
    return generate_synthetic(20, 200, 16, seed=11)


@pytest.mark.parametrize("method", sorted(PLACERS))
def test_common_interface_and_legality(design, method):
    p = get_placer(method)(design, seed=1, budget=500)
    assert len(p) == len(design.macros)
    assert check_legality(design, p) == []
    assert p.meta["method"] == method and p.meta["legalized"]


@pytest.mark.parametrize("method", sorted(PLACERS))
def test_deterministic_per_seed(design, method):
    a = place(design, method, seed=3, budget=300)
    b = place(design, method, seed=3, budget=300)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y) and np.array_equal(a.orient, b.orient)


def test_unknown_placer():
    with pytest.raises(KeyError):
        get_placer("nope")


def test_sa_clearly_beats_random():
    d = generate_synthetic(25, 250, 16, seed=5)
    rnd = [proxy_cost(d, place(d, "random", seed=s)).proxy for s in range(5)]
    sa = [proxy_cost(d, sa_place(d, seed=s, budget=4000)).proxy for s in range(3)]
    assert max(sa) < min(rnd), (sa, rnd)
    assert np.mean(sa) < 0.8 * np.mean(rnd), (sa, rnd)


def test_sa_improves_on_its_init_and_reports_meta(design):
    init = proxy_cost(design, place(design, "grid", seed=0)).proxy
    p = sa_place(design, seed=0, budget=3000)
    assert proxy_cost(design, p).proxy < init
    assert p.meta["evaluations"] == 3000
    assert 0 < p.meta["accepted"] <= 3000


def test_sa_with_fixed_macros_and_random_init():
    d = generate_synthetic(10, 60, seed=2, n_fixed_macros=3)
    fixed = [(m.x, m.y) for m in d.macros if m.fixed]
    p = sa_place(d, seed=0, budget=1500, config=SAConfig(init="random"))
    assert check_legality(d, p) == []
    assert [(int(p.x[i]), int(p.y[i])) for i, m in enumerate(d.macros) if m.fixed] == fixed


def test_cli_inprocess(capsys):
    assert cli_main(["--synthetic", "6", "--clusters", "40", "--method", "sa", "--seed", "0", "--budget", "300"]) == 0
    out = capsys.readouterr().out
    for k in ("proxy", "wirelength", "density", "congestion", "evaluations"):
        assert k in out


def test_cli_def_roundtrip(fixtures, tmp_path, capsys):
    out = tmp_path / "placed.def"
    rc = cli_main(["--def", str(fixtures / "tiny.def"), "--lef", str(fixtures / "tiny.lef"), "--method", "grid", "--out", str(out), "--json"])
    assert rc == 0 and out.exists()
    assert '"legal": true' in capsys.readouterr().out


def test_cli_module_subprocess():
    r = subprocess.run(
        [sys.executable, "-m", "macromind.baselines", "--synthetic", "5", "--clusters", "30", "--method", "random", "--json"],
        capture_output=True, text=True, check=True,
    )
    assert '"method": "random"' in r.stdout


@pytest.mark.slow
def test_sa_performance_50_macros_500_clusters():
    """Quality bar: ~50k move evaluations on 50 macros / 500 clusters within a minute."""
    d = generate_synthetic(50, 500, 32, seed=0)
    t = time.perf_counter()
    p = sa_place(d, seed=0, budget=50_000)
    dt = time.perf_counter() - t
    assert p.meta["evaluations"] == 50_000
    assert dt < 60, dt
