#!/usr/bin/env bash

set -euo pipefail

# Resolve repository root relative to this script so it works on any machine.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
DEV_BRANCH="dev"

usage() {
  cat <<'USAGE'
Usage: push-dev.sh <commit-message>

Stages all changes, commits them to the development branch with the provided
message, fast-forwards from origin, and pushes to origin dev.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

commit_msg="$*"

cd "$REPO_ROOT"

git checkout "$DEV_BRANCH"
git fetch origin "$DEV_BRANCH"
git pull --ff-only origin "$DEV_BRANCH"

git add -A

if git diff --cached --quiet; then
  echo "No staged changes to commit." >&2
  exit 1
fi

git commit -m "$commit_msg"

git push origin "$DEV_BRANCH"

echo "✅ Pushed $DEV_BRANCH with commit: $commit_msg"
