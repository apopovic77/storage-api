#!/usr/bin/env bash

set -euo pipefail

# Resolve repository root relative to this script so it works anywhere.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
CONFIG_FILE="$REPO_ROOT/.devops/starter-config.json"
DEFAULT_STARTER_PATH="/Volumes/DatenAP/Code/github-starterpack"

usage() {
  cat <<'USAGE'
Usage: update-devops.sh [--starter-path <path>] [--skip-fetch]

Fetches the starter pack remote (starter-devops) and reruns the template
installer with --update so the latest helper scripts and workflows are
applied to this repository.
USAGE
}

starter_path="$DEFAULT_STARTER_PATH"
skip_fetch=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --starter-path)
      starter_path="${2:-}"
      shift 2
      ;;
    --skip-fetch)
      skip_fetch=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -f "$CONFIG_FILE" ]]; then
  config_path=$(python3 - <<'PY' "$CONFIG_FILE"
import json, sys
try:
    data = json.load(open(sys.argv[1]))
    print(data.get("STARTER_PATH", ""))
except Exception:
    print("")
PY
)
  if [[ -n "$config_path" ]]; then
    starter_path="$config_path"
  fi
fi

if [[ -z "$starter_path" ]]; then
  starter_path="$DEFAULT_STARTER_PATH"
fi

if [[ ! -x "$starter_path/scripts/setup-devops.sh" ]]; then
  echo "Error: starter pack script not found at $starter_path/scripts/setup-devops.sh" >&2
  echo "Pass --starter-path <path> to point at the correct starter repo." >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working tree has changes. Commit or stash before updating DevOps templates." >&2
  exit 1
fi

if git remote get-url starter-devops >/dev/null 2>&1; then
  if [[ "$skip_fetch" == false ]]; then
    git fetch starter-devops
  fi
else
  echo "Warning: remote 'starter-devops' not configured. Skipping fetch." >&2
fi

"$starter_path/scripts/setup-devops.sh" --target "$REPO_ROOT" --update

echo
echo "✅ DevOps templates refreshed. Review changes with 'git diff' and commit when ready."
