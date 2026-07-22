#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "[data-preparation] Pipeline failed with exit code $status"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export REPO_ROOT

exec bash "$REPO_ROOT/dataset_toolkits/prepare_shapenet.sh"
