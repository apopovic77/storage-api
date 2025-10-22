# Helper Scripts

The following helper scripts live in `.devops/scripts/` and automate the local release workflow:

- `checkout-branch.sh <dev|main>` – Checks for a clean working tree, fetches from `origin`, and checks out the requested branch.
- `push-dev.sh "commit message"` – Switches to `dev`, fast-forwards from `origin`, stages all changes, commits with the given message, and pushes to `origin dev`.
- `build-local.sh [--clean]` – Runs `echo 'No build needed for FastAPI'` in the repo (optional `dist/` cleanup first) for a production build smoke test.
- `release.sh [--no-build]` – Synchronises `dev`, pushes it, optionally runs a local build, fast-forwards `main` from `dev`, pushes `main`, and switches back to `dev`. Pushing `main` triggers the GitHub Actions deploy.
- `update-devops.sh [--starter-path <path>]` – Optionally fetches `starter-devops` and reruns the starter templates with `--update` so `.devops/` and `.github/workflows/` stay current.

After copying the scripts into a project run `chmod +x .devops/scripts/*.sh` if your Git checkout does not preserve executable bits.

Shortcuts: the repository root includes a `./devops` helper. Example calls:

```bash
./devops checkout dev
./devops push "feat: add new helmet grid"
./devops release
./devops update
```
