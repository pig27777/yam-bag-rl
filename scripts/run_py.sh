#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
export PYTHONPATH="$project/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [[ ${1:-} == --bootstrap ]]; then
    shift
    exec python3.11 -B -P "$@"
fi
if [[ ${1:-} == --local ]]; then
    shift
fi
exec "$project/.venv/bin/python" -B -P "$@"
