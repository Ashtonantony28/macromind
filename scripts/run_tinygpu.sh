#!/usr/bin/env bash
# SG9: tiny-gpu-extended macro variant through ORFS (NanGate45 + fakeram45).
#
# Usage: scripts/run_tinygpu.sh [config] [make target] [VAR=val ...]
#   scripts/run_tinygpu.sh fetch                 # clone the pinned upstream GPU RTL only
#   scripts/run_tinygpu.sh sim                   # RTL sim: upstream vs macro variant vs ISA model
#   scripts/run_tinygpu.sh c8 synth              # Yosys (yosys-slang front end) only
#   scripts/run_tinygpu.sh c8 grt                # floorplan + RTLMP macro placement + global route
#   scripts/run_tinygpu.sh c4 route              # + detailed route
#   scripts/run_tinygpu.sh c4                    # full flow (finish)
# Configs: ff2 (upstream default, flops), c4, c8, c16 (see designs/tinygpu/configs.mk).
# Env: WORK_HOME (default runs/orfs/tinygpu), NUM_CORES (default 2), CLEAN=0 to resume.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NUM_CORES="${NUM_CORES:-2}"
# shellcheck source=env.sh
source "$here/env.sh"
tg="$MACROMIND_ROOT/designs/tinygpu"
upstream_url="https://github.com/Ashtonantony28/Didactic_GPU_Simulator"

fetch() {
  local pin; pin="$(cat "$tg/UPSTREAM_PIN")"
  if [ ! -d "$tg/upstream/.git" ]; then
    git clone -q "$upstream_url" "$tg/upstream"
  fi
  if [ "$(git -C "$tg/upstream" rev-parse HEAD)" != "$pin" ]; then
    git -C "$tg/upstream" fetch -q origin "$pin" 2>/dev/null || git -C "$tg/upstream" fetch -q origin
    git -C "$tg/upstream" -c advice.detachedHead=false checkout -q "$pin"
  fi
  python3 "$tg/derive_rtl.py" --check >/dev/null || { echo "derived RTL is stale: run python3 $tg/derive_rtl.py" >&2; exit 1; }
}

cfg="${1:-c8}"; [ $# -ge 1 ] && shift
fetch
case "$cfg" in
  fetch) echo "upstream at $(git -C "$tg/upstream" rev-parse --short HEAD)"; exit 0 ;;
  sim)   exec "$MACROMIND_ROOT/.venv/bin/python" "$tg/sim/run_sim.py" "$@" ;;
esac
target="${1:-finish}"; [ $# -ge 1 ] && shift
case "$target" in grt) target=globalroute ;; esac

export WORK_HOME="${WORK_HOME:-$MACROMIND_ROOT/runs/orfs/tinygpu}"
mkdir -p "$WORK_HOME"
log="$WORK_HOME/tinygpu_${cfg}_${target}.make.log"
mk=(make -C "$FLOW_HOME" DESIGN_CONFIG="$tg/config.mk" TG_CONFIG="$cfg" WORK_HOME="$WORK_HOME" NUM_CORES="$NUM_CORES" "$@")
if [ "${CLEAN:-1}" = 1 ]; then "${mk[@]}" clean_all >/dev/null 2>&1 || true; fi
echo "== ORFS tinygpu config=$cfg target=$target cores=$NUM_CORES log=$log"
t0=$(date +%s)
set +e
"${mk[@]}" "$target" >"$log" 2>&1
rc=$?
set -e
t1=$(date +%s)
echo "== make exit code: $rc, wall time: $((t1 - t0)) s"
[ $rc -eq 0 ] || tail -25 "$log"
python3 -I "$tg/report.py" "$WORK_HOME" "$cfg" --wall "$((t1 - t0))" --json "$WORK_HOME/tinygpu_${cfg}.metrics.json" || true
exit $rc
