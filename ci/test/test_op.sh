#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LIB_PATH="${REPO_ROOT}/build/lib/lib_recstore_ops.so"
PY_CLIENT_DIR="${REPO_ROOT}/src/framework/pytorch/python_client"
PY_PKG_ROOT="${REPO_ROOT}/src/python/pytorch"

GRPC_SERVER_PATH="${REPO_ROOT}/build/bin/grpc_ps_server"

export LD_LIBRARY_PATH="${REPO_ROOT}/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PY_PKG_ROOT}:${PYTHONPATH:-}"

if [[ "$@" == *"--mock"* ]]; then
    echo "Mock mode enabled; skipping this script."
    exit 0
fi

# Ensure ctest is available (fallback to pip cmake bin)
PY_CMAKE_BIN=$(python3 - <<'PY'
import cmake, os
print(os.path.join(os.path.dirname(cmake.__file__), 'data', 'bin'))
PY
)
export PATH="${PY_CMAKE_BIN}:$PATH"

LOG_DIR="${REPO_ROOT}/build/logs"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/grpc_ps_server.log"

echo "Starting grpc_ps_server... (log: ${SERVER_LOG})"
"${GRPC_SERVER_PATH}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 ${SERVER_PID} >/dev/null 2>&1; then
        kill ${SERVER_PID} || true
        wait ${SERVER_PID} 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for server ready signal
READY_TIMEOUT=600
echo "Waiting for grpc_ps_server to be ready..."
if ! timeout ${READY_TIMEOUT} bash -c "until grep -q 'listening on' '${SERVER_LOG}'; do sleep 1; done"; then
    echo "grpc_ps_server did not report 'listening on' within ${READY_TIMEOUT}s"
    exit 1
fi
echo "grpc_ps_server is ready. Running ctest --verbose"

cd "${REPO_ROOT}/build"
export LD_LIBRARY_PATH="${REPO_ROOT}/build/lib:${LD_LIBRARY_PATH:-}"
ctest --verbose

echo "All tests finished successfully."