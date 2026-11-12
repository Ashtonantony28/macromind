#!/usr/bin/env bash
# SpikeMesh neurocore macro variant through ORFS (sky130hd + OpenRAM sky130_sram_1rw1r_64x256_8).
#
# Usage: scripts/run_spikemesh.sh [config] [make target] [VAR=val ...]
#   scripts/run_spikemesh.sh fetch               # clone the pinned upstream SpikeMesh RTL only
#   scripts/run_spikemesh.sh sim [cfg ...]       # upstream core cocotb tests on the macro variant (n128 n128_l2 big)
#   scripts/run_spikemesh.sh n128 synth          # Yosys (yosys-slang front end) only
#   scripts/run_spikemesh.sh n128 grt            # floorplan + RTLMP macro placement + global route
# Configs: n128 (10 macros), n128_l2 (on-chip learning) - see designs/spikemesh/configs.mk.
# Env: WORK_HOME (default runs/orfs/spikemesh), NUM_CORES (default 2), CLEAN=0 to resume.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NUM_CORES="${NUM_CORES:-2}"
# shellcheck source=env.sh
source "$here/env.sh"
sm="$MACROMIND_ROOT/designs/spikemesh"
upstream_url="https://github.com/Ashtonantony28/spikemesh"

fetch() {
  local pin; pin="$(cat "$sm/UPSTREAM_PIN")"
  if [ ! -d "$sm/upstream/.git" ]; then
    git clone -q "$upstream_url" "$sm/upstream"
  fi
  if [ "$(git -C "$sm/upstream" rev-parse HEAD)" != "$pin" ]; then
    git -C "$sm/upstream" fetch -q origin "$pin" 2>/dev/null || git -C "$sm/upstream" fetch -q origin
    git -C "$sm/upstream" -c advice.detachedHead=false checkout -q "$pin"
  fi
}

cfg="${1:-n128}"; [ $# -ge 1 ] && shift
fetch
case "$cfg" in
  fetch) echo "upstream at $(git -C "$sm/upstream" rev-parse --short HEAD)"; exit 0 ;;
  sim)   exec "$MACROMIND_ROOT/.venv/bin/python" "$sm/sim/run_sim.py" "$@" ;;
esac
target="${1:-globalroute}"; [ $# -ge 1 ] && shift
case "$target" in grt) target=globalroute ;; esac

export WORK_HOME="${WORK_HOME:-$MACROMIND_ROOT/runs/orfs/spikemesh}"
mkdir -p "$WORK_HOME"
log="$WORK_HOME/spikemesh_${cfg}_${target}.make.log"
mk=(make -C "$FLOW_HOME" DESIGN_CONFIG="$sm/config.mk" SM_CONFIG="$cfg" WORK_HOME="$WORK_HOME" NUM_CORES="$NUM_CORES" "$@")
if [ "${CLEAN:-1}" = 1 ]; then "${mk[@]}" clean_all >/dev/null 2>&1 || true; fi
echo "== ORFS spikemesh config=$cfg target=$target cores=$NUM_CORES log=$log"
t0=$(date +%s)
set +e
"${mk[@]}" "$target" >"$log" 2>&1
rc=$?
set -e
echo "== make exit code: $rc, wall time: $(( $(date +%s) - t0 )) s"
[ $rc -eq 0 ] || tail -25 "$log"
exit $rc
