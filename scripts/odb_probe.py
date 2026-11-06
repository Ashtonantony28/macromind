"""Probe an OpenROAD .odb with the odb Python API.

Run with OpenROAD's embedded interpreter (not the venv python):
    source scripts/env.sh
    openroad -python -exit scripts/odb_probe.py <design.odb>
Prints die/core area, instance/net counts and every macro (block master)
with its location, size and orientation.
"""
import sys

import odb  # provided by openroad -python

path = sys.argv[1]
db = odb.dbDatabase.create()
odb.read_db(db, path)
blk = db.getChip().getBlock()
dbu = blk.getDbUnitsPerMicron()
die, core = blk.getDieArea(), blk.getCoreArea()
um = lambda v: v / dbu  # noqa: E731
print(f"block={blk.getName()} dbu/um={dbu}")
print(f"die=({um(die.xMin())},{um(die.yMin())})-({um(die.xMax())},{um(die.yMax())}) um "
      f"core=({um(core.xMin())},{um(core.yMin())})-({um(core.xMax())},{um(core.yMax())}) um")
insts = blk.getInsts()
macros = [i for i in insts if i.getMaster().isBlock()]
print(f"insts={len(insts)} nets={len(blk.getNets())} macros={len(macros)} bterms={len(blk.getBTerms())}")
for m in macros:
    x, y = m.getLocation()
    mst = m.getMaster()
    print(f"  {m.getName()} master={mst.getName()} loc=({um(x)},{um(y)}) "
          f"size=({um(mst.getWidth())}x{um(mst.getHeight())}) orient={m.getOrient()} "
          f"status={m.getPlacementStatus()}")
