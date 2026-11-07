"""Dump the GRT gcell grid (per-layer capacity and usage) of an ODB as JSON.

Runs inside OpenROAD's embedded Python (stdlib + ``odb`` only), not the venv::

    openroad -python -no_splash -exit macromind/predict/odb_gcells.py <in.odb> <out.json>

The global router saves its final congestion data (``capacity``/``usage`` per
routing layer per gcell, 8-bit) into the block's ``dbGCellGrid`` when the stage
ODB is written, so ``5_1_grt.odb`` carries the full routing-resource map, also
for runs that never had an explicit dump. The SWIG binding returns these
``uint8_t`` values as pointers, which we read with ctypes.
"""

import ctypes
import json
import sys


def _u8(p):
    return ctypes.c_uint8.from_address(int(p)).value


def main(odb_path, out_path):
    from openroad import Design, Tech  # noqa: PLC0415 -- only importable inside OpenROAD

    tech = Tech()
    design = Design(tech)
    design.readDb(odb_path)
    db = tech.getDB()
    block = design.getBlock()
    gg = block.getGCellGrid()
    die = block.getDieArea()
    out = {"odb": odb_path, "dbu_per_micron": block.getDbUnitsPerMicron(),
           "die": [die.xMin(), die.yMin(), die.xMax(), die.yMax()], "layers": {}}
    if gg is None:
        out["error"] = "no gcell grid"
    else:
        gx, gy = list(gg.getGridX()), list(gg.getGridY())
        out["grid_x"], out["grid_y"] = gx, gy
        nx, ny = len(gx), len(gy)
        for layer in db.getTech().getLayers():
            if layer.getType() != "ROUTING":
                continue
            cap, use = [], []
            for j in range(ny):
                cap.append([_u8(gg.getCapacity(layer, i, j)) for i in range(nx)])
                use.append([_u8(gg.getUsage(layer, i, j)) for i in range(nx)])
            if any(any(r) for r in cap):
                out["layers"][layer.getName()] = {"direction": layer.getDirection(), "capacity": cap, "usage": use}
    with open(out_path, "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
