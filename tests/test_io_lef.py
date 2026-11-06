import pytest

from macromind.io import parse_lef, read_lef
from macromind.io.lef import tokenize


def test_tokenize_comments_and_quotes():
    assert tokenize('A "b ; # c" ; # comment\nD ;') == ["A", '"b ; # c"', ";", "D", ";"]


def test_read_lef_fixture(fixtures):
    lib = read_lef(fixtures / "tiny.lef")
    assert lib.dbu_per_micron == 2000
    site = lib.sites["FreePDK45_38x28_10R_NP_162NW_34O"]
    assert (site.width, site.height) == (0.19, 1.4)
    assert set(lib.macros) == {"INV_X1", "fakeram45_64x32", "BLK_ORIGIN"}
    ram = lib.macros["fakeram45_64x32"]
    assert ram.is_block and (ram.width, ram.height) == (20.0, 40.0)
    offs = ram.pin_offsets()
    assert offs["clk"] == pytest.approx((0.035, 20.0))
    # two shapes on two layers (one with MASK): bbox centre
    assert offs["rd_out[0]"] == pytest.approx((0.1, 1.05))
    inv = lib.macros["INV_X1"]
    assert not inv.is_block and inv.macro_class == "CORE"
    assert inv.pin_offsets()["A"] == pytest.approx((0.105, 0.6125))  # polygon bbox
    assert inv.pins["VDD"].use.lower() == "power"
    # ORIGIN is applied to shapes
    assert lib.macros["BLK_ORIGIN"].pin_offsets()["A"] == pytest.approx((1.0, 1.0))


def test_lef_merge_and_minimal():
    lib = parse_lef("MACRO X CLASS BLOCK ; SIZE 1 BY 2 ; END X END LIBRARY")
    assert lib.macros["X"].height == 2.0 and lib.dbu_per_micron is None
