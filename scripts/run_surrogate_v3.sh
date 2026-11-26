#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$project"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec flock /tmp/claude-gpu.lock "$project/scripts/run_py.sh" --local -m yam_bag_rl.envs.surrogate_run "$@"
