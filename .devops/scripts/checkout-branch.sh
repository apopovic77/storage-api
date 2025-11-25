#!/usr/bin/env bash

set -euo pipefail

# Resolve repository root relative to this script so it works everywhere.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
DEV_BRANCH="dev"
MAIN_BRANCH="main"

usage() {
  cat <<'USAGE'
Usage: checkout-branch.sh <branch>

Switches the working tree to the requested branch and fast-forwards it from origin.

Examples:
  checkout-branch.sh dev
  checkout-branch.sh main
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

branch="$1"

if [[ "$branch" != "$DEV_BRANCH" && "$branch" != "$MAIN_BRANCH" ]]; then
  echo "Error: branch must be '$DEV_BRANCH' or '$MAIN_BRANCH'." >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working tree has uncommitted changes. Please commit or stash them first." >&2
  exit 1
fi

git fetch origin "$branch"
git checkout "$branch"
git pull --ff-only origin "$branch"

echo "✅ Checked out $(git rev-parse --abbrev-ref HEAD) at $(git rev-parse --short HEAD)"
