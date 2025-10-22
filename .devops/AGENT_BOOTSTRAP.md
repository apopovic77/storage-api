# Agent Bootstrap Notes

1. **Repository context**
   - Project lives at `/Volumes/DatenAP/Code/storage-api`.
   - Default working branch is `dev`; production branch is `main`.

2. **Git helper scripts** (under `.devops/scripts/` and already executable):
   - `checkout-branch.sh <dev|main>` – validate clean tree → fetch → checkout → fast-forward.
   - `push-dev.sh "commit message"` – checkout dev → fast-forward from origin → stage all → commit → push `origin dev`.
   - `build-local.sh [--clean]` – optional clean dist, run `echo 'No build needed for FastAPI'`.
   - `release.sh [--no-build]` – sync & push dev, optional local build, fast-forward main from dev, push main, switch back to dev (triggers GitHub Actions deploy).
   - `update-devops.sh [--starter-path <path>]` – optionally fetches `starter-devops` and re-applies the starter templates with `--update` using `.devops/starter-config.json`.

   Shortcuts: the repository root includes a dispatcher `./devops` so you can run `./devops checkout dev`, `./devops push "msg"`, `./devops release`, or `./devops update` without remembering the full paths.

3. **Deployment pipeline**
   - GitHub Actions workflows are in `.github/workflows/`.
   - `dev.yml` runs on pushes to `dev` or PRs targeting `main`; it builds but does *not* deploy.
   - `deploy.yml` runs on pushes to `main`; it builds and deploys via SSH using repo secrets (`DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`, `DEPLOY_PORT`).

4. **Deployment target**
   - Built files go to `/var/www/api-storage.arkturian.com`.
   - Backups are stored with prefix `storage-api` under `/var/backups/` (configurable via templates).

5. **Startup checklist for new session**
   - `cd /Volumes/DatenAP/Code/storage-api`.
   - `git status -sb` and note if tree is clean.
   - Use `.devops/scripts/checkout-branch.sh dev` to sync branch (if tree clean).
   - When ready to release, follow: `.devops/scripts/push-dev.sh "msg"` → optional `build-local.sh` → `release.sh`.

6. **GitHub CLI**
   - Ensure `gh auth status` reports a logged-in user prior to running helper scripts that push.
   - After copying templates, add deployment secrets via `gh secret set`.

Read this file at session start so you remember the workflow.
