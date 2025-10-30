#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

usage() {
  cat <<'USAGE'
Usage: build-local.sh [--clean]

Runs the configured build command inside the repository root to verify the
release build. Optionally deletes the dist/ directory first when --clean is
provided.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

clean_flag=false
if [[ "${1:-}" == "--clean" ]]; then
  clean_flag=true
fi

cd "$REPO_ROOT"

if [[ "$clean_flag" == true && -d dist ]]; then
  rm -rf dist
fi

# shellcheck disable=SC2086
if [[ "$BUILD_COMMAND" == ":" ]]; then
  echo "No build command configured (BUILD_COMMAND). Skipping local build."
else
  $BUILD_COMMAND
fi

echo "✅ Local build finished. Output: ${REPO_ROOT}/dist"
