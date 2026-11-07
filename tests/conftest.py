from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


def pytest_collection_modifyitems(config, items):
    """`eda` tests run real ORFS flows (minutes); run them with `pytest -m eda` or MACROMIND_EDA=1."""
    import os

    if "eda" in (config.getoption("-m") or "") or os.environ.get("MACROMIND_EDA") == "1":
        return
    skip = pytest.mark.skip(reason="EDA integration test: run with `pytest -m eda` (or MACROMIND_EDA=1)")
    for it in items:
        if "eda" in it.keywords:
            it.add_marker(skip)
