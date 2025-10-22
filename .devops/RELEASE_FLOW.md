# 🚀 Release Flow Overview

## Branches
- **`dev`** – Integration branch for day-to-day work.
- **`main`** – Production branch; pushing here triggers the deployment workflow.

## Typical Release Steps
1. Develop features on short-lived branches branched off `dev`.
2. Merge into `dev` via pull request. CI (`dev.yml`) runs automatically.
3. When ready to ship, ensure `dev` is green and run `.devops/scripts/release.sh`.
4. The script fast-forwards `main`, pushes to GitHub, and GitHub Actions (`deploy.yml`) handles build + deploy.
5. Monitor the workflow run with `gh run watch` or via the Actions tab.
6. Validate the site at `https://api-storage.arkturian.com` (if applicable) and confirm fresh assets reached `/var/www/api-storage.arkturian.com`.

## Rollback Strategy
- Use `.devops/rollback.sh` on the server to restore from `/var/backups/storage-api-*`.
- Alternatively revert the offending commit on `main` and push to trigger a redeploy.
