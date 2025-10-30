#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

usage() {
  cat <<'USAGE'
Usage: push-dev.sh <commit-message>

Stages all changes, commits them to the development branch with the provided
message, fast-forwards from origin, and pushes to origin {{DEV_BRANCH}}.
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

git fetch origin "$DEV_BRANCH"
if ! git checkout "$DEV_BRANCH"; then
  die "Branch $DEV_BRANCH does not exist"
fi
git pull --ff-only origin "$DEV_BRANCH" || true

git add -A

if git diff --cached --quiet; then
  echo "No staged changes to commit. Staging working tree..."
fi

git commit -m "$commit_msg" || echo "No changes to commit"

git push origin "$DEV_BRANCH"

echo "✅ Pushed $DEV_BRANCH with commit: $commit_msg"
