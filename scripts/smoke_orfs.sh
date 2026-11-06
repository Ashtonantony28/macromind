#!/usr/bin/env bash
# Run ORFS end to end for nangate45/<design> (or <platform>/<design>) and print
# key metrics from the ORFS per-stage metric JSON files.
#
# Usage: scripts/smoke_orfs.sh <design> [make target] [extra make VAR=val ...]
#   scripts/smoke_orfs.sh gcd
#   scripts/smoke_orfs.sh tinyRocket
#   scripts/smoke_orfs.sh asap7/gcd
#   scripts/smoke_orfs.sh ariane133 floorplan      # stop after a stage
# Env: WORK_HOME (default runs/orfs/smoke), FLOW_VARIANT (default base),
#      NUM_CORES (default nproc), CLEAN=0 to reuse previous results.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$here/env.sh"

[ $# -ge 1 ] || { echo "usage: $0 <design> [target] [VAR=val ...]" >&2; exit 2; }
spec="$1"; shift
target="${1:-finish}"; [ $# -ge 1 ] && shift
if [[ "$spec" == */* ]]; then platform="${spec%%/*}"; design="${spec#*/}"; else platform=nangate45; design="$spec"; fi
cfg="designs/$platform/$design/config.mk"
[ -f "$FLOW_HOME/$cfg" ] || { echo "no such ORFS design: $FLOW_HOME/$cfg" >&2; exit 2; }
[ -x "$OPENROAD_EXE" ] || { echo "OpenROAD missing; run scripts/setup_toolchain.sh" >&2; exit 2; }

export WORK_HOME="${WORK_HOME:-$MACROMIND_ROOT/runs/orfs/smoke}"
variant="${FLOW_VARIANT:-base}"
mkdir -p "$WORK_HOME"
log="$WORK_HOME/${platform}_${design}_${variant}.make.log"
mk=(make -C "$FLOW_HOME" DESIGN_CONFIG="$cfg" WORK_HOME="$WORK_HOME" FLOW_VARIANT="$variant" NUM_CORES="$NUM_CORES" "$@")

if [ "${CLEAN:-1}" = 1 ]; then "${mk[@]}" clean_all >/dev/null 2>&1 || true; fi
echo "== ORFS $platform/$design target=$target variant=$variant cores=$NUM_CORES"
echo "== log: $log"
t0=$(date +%s)
set +e
"${mk[@]}" "$target" >"$log" 2>&1
rc=$?
set -e
t1=$(date +%s)
echo "== make exit code: $rc, wall time: $((t1 - t0)) s"
if [ $rc -ne 0 ]; then echo "== last 25 log lines:"; tail -25 "$log"; fi

logdir="$WORK_HOME/logs/$platform/$design/$variant"
python3 -I - "$logdir" "$rc" "$((t1 - t0))" <<'PY'
import glob, json, os, sys
logdir, rc, wall = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = {}
for f in sorted(glob.glob(os.path.join(logdir, "*.json"))):
    try:
        m.update(json.load(open(f)))
    except Exception as e:  # noqa: BLE001
        print(f"warn: cannot parse {f}: {e}")
def first(*keys):
    for k in keys:
        if k in m:
            return m[k]
    return None
# per-stage runtime from the *.log "Elapsed time" lines
stages = []
for f in sorted(glob.glob(os.path.join(logdir, "*.log"))):
    el = None
    for line in open(f, errors="replace"):
        if line.startswith("Elapsed time:"):
            t = line.split()[2].replace("[h:]min:sec.", "")
            parts = [float(p) for p in t.split(":")]
            s = 0.0
            for p in parts:
                s = s * 60 + p
            el = s
    if el is not None:
        stages.append((os.path.basename(f)[:-4], el))
macros = first("floorplan__design__instance__count__macros", "finish__design__instance__count__macros",
               "globalplace__design__instance__count__macros")
rows = [
    ("stdcell instances", first("finish__design__instance__count__stdcell", "detailedplace__design__instance__count__stdcell")),
    ("macros", macros),
    ("core util", first("finish__design__instance__utilization")),
    ("routed wirelength (um)", first("detailedroute__route__wirelength")),
    ("vias", first("detailedroute__route__vias")),
    ("DRC errors (detailed route)", first("detailedroute__route__drc_errors")),
    ("antenna violating nets", first("detailedroute__antenna__violating__nets")),
    ("setup WNS (ns)", first("finish__timing__setup__ws", "detailedroute__timing__setup__ws", "globalroute__timing__setup__ws", "cts__timing__setup__ws")),
    ("setup TNS (ns)", first("finish__timing__setup__tns", "detailedroute__timing__setup__tns", "globalroute__timing__setup__tns", "cts__timing__setup__tns")),
    ("hold WNS (ns)", first("finish__timing__hold__ws")),
    ("total power (W)", first("finish__power__total")),
    ("die area (um^2)", first("finish__design__die__area", "floorplan__design__die__area")),
    ("GRT antenna diodes", first("globalroute__antenna_diodes_count")),
]
print("== metrics")
for k, v in rows:
    print(f"  {k:30s} {v if v is not None else 'n/a'}")
print("== stage runtimes (s)")
tot = 0.0
for s, e in stages:
    tot += e
    print(f"  {s:28s} {e:8.1f}")
print(f"  {'sum of stages':28s} {tot:8.1f}   (make wall {wall} s)")
ok = rc == 0 and m.get("detailedroute__route__drc_errors", None) == 0
print("== RESULT:", "PASS (routed, 0 DRC)" if ok else ("FAIL" if rc else "finished stage(s) without detailed route metrics"))
sys.exit(0 if rc == 0 else 1)
PY
