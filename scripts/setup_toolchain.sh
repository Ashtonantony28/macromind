#!/usr/bin/env bash
# Idempotent installer for the MacroMind EDA toolchain (SG1).
#
# What it does (see docs/toolchain.md for why):
#   1. apt-installs the host runtime libs the prebuilt binaries need
#      (Tcl 8.6, tclreadline, Qt5, KLayout, GNU time/make).
#   2. Fetches the OpenROAD-flow-scripts image  openroad/orfs:$ORFS_TAG  from
#      the mirror.gcr.io pull-through cache (Docker Hub itself is rate limited
#      here), via docker if a daemon is available, else via the registry HTTP
#      API directly (scripts/_oci_extract.py).
#   3. Extracts from it, into tools/ORFS/:
#        flow/                      ORFS Makefile, scripts, platforms, designs
#        tools/install/{OpenROAD,yosys,...}   prebuilt binaries (Ubuntu 22.04 build)
#        tools/lib/libortools.so.9, libpython3.10.so.1.0   (not in Ubuntu 24.04)
#        tools/py310/lib/python3.10 stdlib for `openroad -python`
#   4. Writes tools/bin/openroad (wrapper) and runs a version check.
#
# Usage: scripts/setup_toolchain.sh [--force]
# Env:   ORFS_TAG (default v3.0-3468-gc673f615), ORFS_REGISTRY (default mirror.gcr.io),
#        KEEP_IMAGE=1 to keep the docker image after extraction.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
ORFS_TAG="${ORFS_TAG:-v3.0-3468-gc673f615}"
ORFS_REGISTRY="${ORFS_REGISTRY:-mirror.gcr.io}"
image="$ORFS_REGISTRY/openroad/orfs:$ORFS_TAG"
dest="$root/tools/ORFS"
stamp="$dest/.installed-$ORFS_TAG"
force=0; [ "${1:-}" = "--force" ] && force=1

log() { echo "[setup_toolchain] $*"; }

# ---------------------------------------------------------------- 1. apt deps
apt_pkgs=(tcl8.6 tcl-tclreadline libqt5charts5 libqt5widgets5t64 libqt5gui5t64 libqt5core5t64 klayout time make)
missing=()
for p in "${apt_pkgs[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
if [ ${#missing[@]} -gt 0 ]; then
  log "apt installing: ${missing[*]}"
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends "${missing[@]}"
else
  log "apt deps present"
fi

# ------------------------------------------------------ 2+3. fetch + extract
if [ -f "$stamp" ] && [ $force = 0 ]; then
  log "ORFS $ORFS_TAG already installed in $dest (use --force to redo)"
else
  rm -rf "$dest"; mkdir -p "$dest/tools/lib" "$dest/tools/py310/lib"
  have_docker=0
  if command -v docker >/dev/null; then
    if ! docker info >/dev/null 2>&1 && command -v dockerd >/dev/null && [ "$(id -u)" -eq 0 ]; then
      log "starting dockerd"
      (nohup dockerd >/tmp/dockerd.log 2>&1 &)
      for _ in $(seq 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
    fi
    docker info >/dev/null 2>&1 && have_docker=1
  fi
  if [ $have_docker = 1 ] && docker pull -q "$image" >/dev/null; then
    log "extracting from docker image $image"
    cid=$(docker create "$image")
    trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
    docker cp "$cid:/OpenROAD-flow-scripts/flow" "$dest/"
    docker cp "$cid:/OpenROAD-flow-scripts/env.sh" "$dest/"
    docker cp "$cid:/OpenROAD-flow-scripts/tools/install" "$dest/tools/"
    docker cp -L "$cid:/opt/or-tools/lib/libortools.so.9" "$dest/tools/lib/"
    docker cp -L "$cid:/lib/x86_64-linux-gnu/libpython3.10.so.1.0" "$dest/tools/lib/"
    docker cp "$cid:/usr/lib/python3.10" "$dest/tools/py310/lib/"
    docker rm -f "$cid" >/dev/null; trap - EXIT
    [ "${KEEP_IMAGE:-0}" = 1 ] || docker rmi "$image" >/dev/null 2>&1 || true
  else
    log "docker unavailable; pulling $image layers over the registry API"
    python3 -I "$here/_oci_extract.py" "$ORFS_REGISTRY" openroad/orfs "$ORFS_TAG" "$dest/_rootfs" \
      OpenROAD-flow-scripts/flow OpenROAD-flow-scripts/env.sh OpenROAD-flow-scripts/tools/install \
      opt/or-tools/lib usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0 usr/lib/python3.10
    r="$dest/_rootfs"
    mv "$r/OpenROAD-flow-scripts/flow" "$r/OpenROAD-flow-scripts/env.sh" "$dest/"
    mv "$r/OpenROAD-flow-scripts/tools/install" "$dest/tools/"
    cp -L "$r/opt/or-tools/lib/libortools.so.9" "$dest/tools/lib/"
    cp -L "$r/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0" "$dest/tools/lib/"
    mv "$r/usr/lib/python3.10" "$dest/tools/py310/lib/"
    rm -rf "$r"
  fi
  touch "$stamp"
fi

# ------------------------------------------------------------- 4. wrapper
mkdir -p "$root/tools/bin"
cat >"$root/tools/bin/openroad" <<'EOF'
#!/usr/bin/env bash
# Wrapper: run bundled OpenROAD (ORFS image build) on the host.
# PYTHONHOME is set only for `-python` mode: OpenROAD's embedded CPython 3.10
# needs the bundled stdlib, but in Tcl mode ORFS spawns host python3 children
# that must not inherit it.
T="$(cd "$(dirname "${BASH_SOURCE[0]}")/../ORFS/tools" && pwd)"
export LD_LIBRARY_PATH="$T/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
for a in "$@"; do
  if [ "$a" = "-python" ]; then export PYTHONHOME="$T/py310"; unset PYTHONPATH; break; fi
done
exec "$T/install/OpenROAD/bin/openroad" "$@"
EOF
chmod +x "$root/tools/bin/openroad"

# ------------------------------------------------------------- 5. check
# shellcheck source=env.sh
source "$here/env.sh"
missing_libs=$(ldd "$dest/tools/install/OpenROAD/bin/openroad" | grep "not found" || true)
if [ -n "$missing_libs" ]; then log "ERROR: unresolved libs:"; echo "$missing_libs"; exit 1; fi
log "openroad: $(command -v openroad)  ($(openroad -version 2>&1 | tail -1); ORFS tag $ORFS_TAG)"
log "yosys:    $(yosys -V)"
log "FLOW_HOME=$FLOW_HOME"
tmp=$(mktemp -d)
printf 'import odb\nfrom openroad import Design, Tech\ndb = odb.dbDatabase.create()\nprint("odb python OK, schema-capable db:", db is not None)\n' >"$tmp/t.py"
openroad -python -exit "$tmp/t.py" 2>&1 | tail -1
rm -rf "$tmp"
log "done. Next: source scripts/env.sh && scripts/smoke_orfs.sh gcd"
