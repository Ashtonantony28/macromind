import re

import pytest

from macromind.core import Orient
from macromind.io import parse_def, read_def, read_lef, update_def_components, write_def_placement


@pytest.fixture
def design(fixtures):
    return read_def(fixtures / "tiny.def", read_lef(fixtures / "tiny.lef"))


def test_parse_def_raw(fixtures):
    d = parse_def((fixtures / "tiny.def").read_text())
    assert d.design == "tiny" and d.dbu_per_micron == 2000
    assert (d.die.xh, d.die.yh) == (300000, 300000)
    assert len(d.rows) == 95 and d.rows[1].orient == Orient.FS and d.rows[0].num_x == 700 and d.rows[0].step_x == 380
    comps = {c.name: c for c in d.components}
    assert comps["ram0"].status == "PLACED" and (comps["ram0"].x, comps["ram0"].y) == (30000, 40000)
    assert comps["ram1"].status == "FIXED" and comps["ram1"].orient == Orient.FS
    assert comps["ram2"].status == "UNPLACED"
    pins = {p.name: p for p in d.pins}
    assert pins["out"].orient == Orient.W and pins["out"].centre() == (299930.0, 100000.0)
    assert pins["clk"].centre() == (100000.0, 70.0)
    nets = {n.name: n for n in d.nets}
    assert nets["clk"].use == "CLOCK" and len(nets["clk"].pins) == 4
    assert nets["n1"].pins == [("ram0", "rd_out[0]"), ("u1", "A")]  # routing ignored
    assert nets["n_out"].pins[1] == ("u2", "A") and nets["n_out"].weight == 3.0


def test_def_to_design(design):
    assert [m.name for m in design.macros] == ["ram0", "ram1", "ram2"]
    assert [c.name for c in design.cells] == ["u1", "u2"]
    assert design.core.xl == 20000 and design.core.yl == 20000
    assert design.core.xh == 20000 + 700 * 380 and design.core.yh == 20000 + 95 * 2800
    assert design.site.width == 380 and design.site.height == 2800
    ram1 = design.macros[1]
    assert ram1.fixed and ram1.width == 40000 and ram1.height == 80000
    assert ram1.pins["clk"] == pytest.approx((70.0, 40000.0))
    assert not design.macros[2].placed
    assert {p.name for p in design.ports} == {"clk", "out"}  # power pin skipped
    a = design.arrays()
    assert a.n_nets == 3


def test_def_writer_roundtrip(fixtures, design, tmp_path):
    src = fixtures / "tiny.def"
    p = design.placement()
    p.x[:] = [40000, 0, 120000]
    p.y[:] = [100000, 0, 20000]
    p.orient[:] = [Orient.FN, 0, Orient.S]
    out = tmp_path / "out.def"
    write_def_placement(src, out, design, p)
    d2 = read_def(out, read_lef(fixtures / "tiny.lef"))
    got = {m.name: (m.x, m.y, m.orient, m.fixed, m.placed) for m in d2.macros}
    assert got["ram0"] == (40000, 100000, Orient.FN, False, True)
    assert got["ram1"] == (100000, 40000, Orient.FS, True, True)  # fixed: untouched
    assert got["ram2"] == (120000, 20000, Orient.S, False, True)  # was UNPLACED
    # only the changed component statements differ
    a, b = src.read_text().splitlines(), out.read_text().splitlines()
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2 and all(re.match(r"- ram[02] ", x) for x, _ in diff)
    assert "+ SOURCE DIST + PLACED ( 120000 20000 ) S" in out.read_text()


def test_def_writer_status_and_errors(fixtures):
    text = (fixtures / "tiny.def").read_text()
    new = update_def_components(text, {"ram1": (1, 2, 0)}, status="PLACED")
    assert "- ram1 fakeram45_64x32\n      + PLACED ( 1 2 ) N ;" in new
    with pytest.raises(KeyError):
        update_def_components(text, {"nope": (0, 0, 0)})
    with pytest.raises(ValueError):
        update_def_components("DESIGN x ;", {})
