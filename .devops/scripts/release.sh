#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/Volumes/DatenAP/Code/storage-api"
DEV_BRANCH="dev"
MAIN_BRANCH="main"
CHECKOUT_SCRIPT="$(dirname "$0")/checkout-branch.sh"
BUILD_SCRIPT="$(dirname "$0")/build-local.sh"

usage() {
  cat <<'USAGE'
Usage: release.sh [--no-build]

Fast-forwards main from dev, pushes to origin, and
triggers the GitHub Actions deployment workflow. By default the local build
script runs beforehand.

Options:
  --no-build   Skip running the local build before pushing
USAGE
}

run_build=true

for arg in "$@"; do
  case "$arg" in
    --no-build)
      run_build=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage
      exit 1
      ;;
  esac
done

cd "$REPO_ROOT"

# Check for uncommitted changes and auto-commit them
if [[ -n "$(git status --porcelain)" ]]; then
  printf '\n==> Auto-committing changes for release\n'
  
  # Generate informative commit message
  changed_files=$(git status --porcelain | wc -l | xargs)
  timestamp=$(date '+%Y-%m-%d %H:%M')
  commit_msg="Release preparation - ${changed_files} files updated [${timestamp}]"
  
  git add -A
  git commit -m "$commit_msg"
  
  echo "✅ Auto-committed: $commit_msg"
fi

printf '\n==> Syncing %s branch\n' "$DEV_BRANCH"
"$CHECKOUT_SCRIPT" "$DEV_BRANCH"

git push origin "$DEV_BRANCH"

if [[ "$run_build" == true ]]; then
  printf '\n==> Running local production build\n'
  "$BUILD_SCRIPT"
fi

printf '\n==> Syncing %s branch\n' "$MAIN_BRANCH"
"$CHECKOUT_SCRIPT" "$MAIN_BRANCH"

printf '\n==> Fast-forwarding %s from %s\n' "$MAIN_BRANCH" "$DEV_BRANCH"
if ! git merge --ff-only "$DEV_BRANCH"; then
  echo "⚠️  Cannot fast-forward - branches have diverged."
  echo "🔄 Auto-syncing: merging $MAIN_BRANCH into $DEV_BRANCH first..."
  
  # Switch back to dev and merge main
  "$CHECKOUT_SCRIPT" "$DEV_BRANCH"
  git merge "$MAIN_BRANCH" -m "Auto-sync: Merge $MAIN_BRANCH into $DEV_BRANCH before release"
  
  echo "✅ Branches synced. Pushing $DEV_BRANCH..."
  git push origin "$DEV_BRANCH"
  
  # Now switch to main and try fast-forward again
  "$CHECKOUT_SCRIPT" "$MAIN_BRANCH"
  echo "🔄 Retrying fast-forward..."
  
  if ! git merge --ff-only "$DEV_BRANCH"; then
    echo "❌ Error: Still cannot fast-forward after sync. Manual intervention needed." >&2
    exit 1
  fi
  
  echo "✅ Fast-forward successful after sync!"
fi

git push origin "$MAIN_BRANCH"

printf '\n✅ Release pushed. GitHub Actions will build & deploy from %s.\n' "$MAIN_BRANCH"

printf '\n==> Switching back to %s\n' "$DEV_BRANCH"
"$CHECKOUT_SCRIPT" "$DEV_BRANCH"
