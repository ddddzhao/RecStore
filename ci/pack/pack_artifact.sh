#!/bin/bash
set -euo pipefail

# RecStore CI packer: collects executables or .so libraries and their runtime deps.
# Usage:
#   ci/pack/pack_artifact.sh <output-tar.gz> <artifact1> [artifact2 ...]
# Examples:
#   ci/pack/pack_artifact.sh build/packed-bin.tar.gz build/bin/grpc_ps_server
#   ci/pack/pack_artifact.sh build/packed-lib.tar.gz build/lib/lib_recstore_ops.so

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <output-tar.gz> <artifact1> [artifact2 ...]" >&2
  exit 2
fi

OUTPUT_TAR="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORK_DIR="$(mktemp -d)"
PACKAGE_ROOT="${WORK_DIR}/package"
BIN_DIR="${PACKAGE_ROOT}/bin"
LIB_DIR="${PACKAGE_ROOT}/lib"
DEPS_DIR="${PACKAGE_ROOT}/deps/lib"
MANIFEST="${PACKAGE_ROOT}/manifest.txt"

mkdir -p "${BIN_DIR}" "${LIB_DIR}" "${DEPS_DIR}"
echo "RecStore Pack Manifest" > "${MANIFEST}"
date >> "${MANIFEST}"

copy_unique() {
  local src="$1"
  local dest_dir="$2"
  local base
  base="$(basename "$src")"
  if [[ -f "${dest_dir}/${base}" ]]; then
    return 0
  fi
  if [[ -L "$src" ]]; then
    local target
    target="$(readlink -f "$src")"
    if [[ -f "$target" ]]; then
      local tbase
      tbase="$(basename "$target")"
      if [[ ! -f "${dest_dir}/${tbase}" ]]; then
        cp -a "$target" "${dest_dir}/"
      fi
      ln -sf "$tbase" "${dest_dir}/${base}"
    else
      cp -L "$src" "${dest_dir}/"
    fi
  else
    cp -a "$src" "${dest_dir}/"
  fi
}

parse_ldd_and_copy_deps() {
  local target="$1"
  echo "\n[lddtree] ${target}" >> "${MANIFEST}"
  local listed=0
  if command -v lddtree >/dev/null 2>&1; then
    lddtree -l "$target" | tee -a "${MANIFEST}" | while read -r dep; do
      listed=1
      [[ -z "$dep" ]] && continue
      if [[ ! -f "$dep" ]]; then continue; fi
      copy_unique "$dep" "${DEPS_DIR}"
    done
  else
    echo "lddtree not available; falling back to ldd parsing" >> "${MANIFEST}"
    ldd "$target" | tee -a "${MANIFEST}" | awk '{ for(i=1;i<=NF;i++){ if ($i ~ /^\//) print $i; } }' | while read -r dep; do
      [[ -z "$dep" ]] && continue
      if [[ ! -f "$dep" ]]; then continue; fi
      copy_unique "$dep" "${DEPS_DIR}"
    done
  fi
}

detect_type() {
  local path="$1"
  # detect ELF type
  local info
  info="$(file -Lb "$path" || true)"
  if [[ "$info" == *"ELF"* && "$info" == *"executable"* ]]; then
    echo "exe"
  elif [[ "$path" == *.so* || ( "$info" == *"ELF"* && "$info" == *"shared object"* ) ]]; then
    echo "so"
  else
    echo "unknown"
  fi
}

for artifact in "$@"; do
  if [[ ! -f "$artifact" ]]; then
    echo "Artifact not found: $artifact" >&2
    exit 1
  fi
  type="$(detect_type "$artifact")"
  case "$type" in
    exe)
      copy_unique "$artifact" "${BIN_DIR}"
      parse_ldd_and_copy_deps "$artifact"
      ;;
    so)
      copy_unique "$artifact" "${LIB_DIR}"
      parse_ldd_and_copy_deps "$artifact"
      ;;
    *)
      echo "Skipping unknown type: $artifact" >&2
      ;;
  esac
done

# Inject RPATH
inject_rpath() {
  local target="$1"
  local origin_rpath="$2"
  if command -v patchelf >/dev/null 2>&1; then
    patchelf --set-rpath "$origin_rpath" "$target" || true
  else
    echo "patchelf not available; skip rpath injection for $target" >&2
  fi
}

# For executables: $ORIGIN/../deps/lib:$ORIGIN/../lib
# For shared libs: $ORIGIN/../deps/lib:$ORIGIN
if command -v file >/dev/null 2>&1; then
  for f in "${BIN_DIR}"/*; do
    [[ -f "$f" ]] || continue
    if file -Lb "$f" | grep -q "ELF"; then
      inject_rpath "$f" "\$ORIGIN/../deps/lib:\$ORIGIN/../lib"
    fi
  done
  for f in "${LIB_DIR}"/*; do
    [[ -f "$f" ]] || continue
    if file -Lb "$f" | grep -q "ELF"; then
      inject_rpath "$f" "\$ORIGIN/../deps/lib:\$ORIGIN"
    fi
  done
else
  echo "file command not available; cannot detect ELF for rpath injection" >&2
fi

cat > "${PACKAGE_ROOT}/README.txt" <<EOF
This package contains selected RecStore binaries and/or shared libraries
and their runtime dependencies as resolved by ldd on the build host.

Structure:
- bin/: executables
- lib/: shared libraries (.so)
- deps/lib/: copied shared library dependencies from ldd
- manifest.txt: ldd outputs and timestamp

Additionally, when available, dependencies are resolved via lddtree (pax-utils)
and recorded here. For each packaged target, the lddtree listing is appended
to manifest.txt for diagnostic purposes.

Note: System loader (ld-linux) and linux-vdso are not included.
EOF

tar -C "${WORK_DIR}" --numeric-owner --owner=0 --group=0 -czf "${OUTPUT_TAR}" package

echo "Packed artifacts to ${OUTPUT_TAR}"
echo "Contents:" && tar -tzf "${OUTPUT_TAR}" | sed 's/^/  /'

rm -rf "${WORK_DIR}"
