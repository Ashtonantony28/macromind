#!/usr/bin/env bash
# Source this file:  source scripts/env.sh
# Sets PATH and ORFS variables for the MacroMind EDA toolchain installed by
# scripts/setup_toolchain.sh (native OpenROAD/Yosys binaries extracted from the
# openroad/orfs:v3.0-3468-gc673f615 image, running on the host).
_mm_src="${BASH_SOURCE[0]:-$0}"
export MACROMIND_ROOT="$(cd "$(dirname "$_mm_src")/.." && pwd)"
unset _mm_src
export ORFS_ROOT="$MACROMIND_ROOT/tools/ORFS"
export FLOW_HOME="$ORFS_ROOT/flow"
_mm_t="$ORFS_ROOT/tools"
export OPENROAD_EXE="$MACROMIND_ROOT/tools/bin/openroad"   # wrapper around $_mm_t/install/OpenROAD/bin/openroad
export OPENSTA_EXE="$_mm_t/install/OpenROAD/bin/sta"
export YOSYS_EXE="$_mm_t/install/yosys/bin/yosys"
# Bundled libs not available on Ubuntu 24.04: OR-Tools 9 and libpython3.10
# (openroad -python embeds CPython 3.10; its stdlib lives in py310/).
case ":${LD_LIBRARY_PATH:-}:" in *":$_mm_t/lib:"*) ;; *) export LD_LIBRARY_PATH="$_mm_t/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}";; esac
# PYTHONHOME only affects the embedded interpreter when set for openroad; we
# do not export it globally (it would break the host python3/.venv).  Use the
# `openroad` wrapper in tools/bin, which sets it per invocation.
export MM_OPENROAD_PYTHONHOME="$_mm_t/py310"
case ":$PATH:" in *":$MACROMIND_ROOT/tools/bin:"*) ;; *) export PATH="$MACROMIND_ROOT/tools/bin:$_mm_t/install/yosys/bin:$PATH";; esac
export NUM_CORES="${NUM_CORES:-$(nproc)}"
# Headless Qt (openroad is built with +GUI)
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
unset _mm_t
