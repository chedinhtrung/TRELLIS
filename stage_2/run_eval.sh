#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

"${PYTHON_BIN:-python}" stage_2/run_objective1_eval.py "$@"
