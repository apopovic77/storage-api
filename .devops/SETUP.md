# Setup Checklist

1. **Copy templates** – Use `scripts/setup-devops.sh` from the starter pack to install `.devops/` and `.github/workflows/` into your project.
2. **Install dependencies** – Ensure the target project can run `echo 'No build needed for FastAPI'` successfully.
3. **Configure GitHub secrets**
   - `DEPLOY_HOST`
   - `DEPLOY_USER`
   - `DEPLOY_SSH_KEY`
   - `DEPLOY_PORT` *(optional, default 22)*
4. **Ensure server access** – The SSH key must be authorised to run deployments on the remote host (append the public key to `authorized_keys`).
5. **Validate CI** – Push to `dev` and confirm `dev.yml` runs successfully.
6. **Test release** – Run `.devops/scripts/release.sh --no-build` (or full build) to fast-forward `main` and trigger the deploy workflow. Verify the GitHub Action can SSH to the server and publish files to `/var/www/api-storage.arkturian.com`.
