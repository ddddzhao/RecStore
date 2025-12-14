#!/usr/bin/env bash
set -euo pipefail

# Sync CI scripts from external repo and overwrite local ci/{env,pack,test}
# Repo: https://github.com/Choimoe/recstore_ci_test.git

REPO_URL="https://github.com/Choimoe/recstore_ci_test.git"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${REPO_ROOT}/.tmp_ci_sync"

echo "[CI Sync] Repo root: ${REPO_ROOT}"
echo "[CI Sync] Temporary work dir: ${WORK_DIR}"

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

echo "[CI Sync] Cloning ${REPO_URL}..."
git clone --depth 1 "${REPO_URL}" "${WORK_DIR}/recstore_ci_test"

SRC_ROOT="${WORK_DIR}/recstore_ci_test"
for d in env pack test; do
	if [[ -d "${SRC_ROOT}/${d}" ]]; then
		echo "[CI Sync] Updating ci/${d}"
		mkdir -p "${REPO_ROOT}/ci/${d}"
		rsync -a --delete "${SRC_ROOT}/${d}/" "${REPO_ROOT}/ci/${d}/"
	else
		echo "[CI Sync] Warning: ${d} not found in source repo" >&2
	fi
done

echo "[CI Sync] Done. Cleaning up..."
rm -rf "${WORK_DIR}"
echo "[CI Sync] Completed successfully."

