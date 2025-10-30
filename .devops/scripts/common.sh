#!/usr/bin/env bash

set -euo pipefail

# Resolve script dir and repo root robustly (works when called from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Try git to find the top-level; fallback to two levels up from scripts dir
if REPO_TOP=$(git -C "$SCRIPT_DIR/../.." rev-parse --show-toplevel 2>/dev/null); then
  REPO_ROOT="${REPO_ROOT:-$REPO_TOP}"
else
  REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
fi

# Branch defaults (overridable via env)
DEV_BRANCH="${DEV_BRANCH:-dev}"
MAIN_BRANCH="${MAIN_BRANCH:-main}"

# Build command default (no-op unless provided by CI or project)
BUILD_COMMAND="${BUILD_COMMAND:-:}"

die() { echo "Error: $*" >&2; exit 1; }


