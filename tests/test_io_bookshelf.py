import pytest

from macromind.core import Orient
from macromind.io import read_bookshelf


def test_read_bookshelf_fixture(fixtures):
    d = read_bookshelf(fixtures / "bookshelf" / "tiny.aux")
    assert d.name == "tiny"
    assert (d.core.xl, d.core.yl, d.core.xh, d.core.yh) == (0, 0, 40, 20)
    assert d.site.height == 4 and d.site.width == 1 and len(d.rows) == 5
    macros = {m.name: m for m in d.macros}
    cells = {c.name: c for c in d.cells}
    assert set(macros) == {"a0", "a3", "p1"}  # tall movable + in-core terminal
    assert set(cells) == {"a1", "a2"}
    assert [p.name for p in d.ports] == ["p0"]  # pad outside the core
    assert macros["p1"].fixed and not macros["a0"].fixed
    assert cells["a2"].orient == Orient.FS
    # centre-relative offset (1, -2) on 4x16 node -> lower-left-relative (3, 6)
    n0 = d.nets[0]
    assert n0.weight == 2.0 and len(n0.pins) == 3
    assert macros["a0"].pins[n0.pins[0].pin] == pytest.approx((3.0, 6.0))
    assert d.nets[2].name == "net2"  # unnamed net
    assert d.arrays().n_nets == 3


def test_bookshelf_missing_file(tmp_path):
    (tmp_path / "x.aux").write_text("RowBasedPlacement : x.nodes x.nets\n")
    with pytest.raises(ValueError):
        read_bookshelf(tmp_path / "x.aux")
