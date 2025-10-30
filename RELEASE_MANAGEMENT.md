# Release Management & CI/CD Strategy

## Overview

Multi-Environment Deployment Strategy für Storage API System mit Development, Staging und Production Environments.

---

## Server Architecture

### Development/Staging Server
- **Domain**: arkturian.com
- **Purpose**: Feature development, testing, QA
- **Branch**: `dev` (auto-deploy)
- **Database**: Separate dev database
- **OpenAI API**: Dev API key mit niedrigeren rate limits

### Production Servers
- **Customer 1**: oneal.arkturian.com (Beispiel)
- **Customer 2**: customer2.com (zukünftig)
- **Purpose**: Live customer deployments
- **Branch**: `main` / `release/*` (manual deploy after QA)
- **Database**: Customer-specific production databases
- **OpenAI API**: Production API keys

---

## Branch Strategy

```
main (production)
  ├── release/v1.0.0 (tagged releases)
  ├── release/v1.1.0
  │
dev (staging/development)
  ├── feature/vibe-search
  ├── feature/build-your-kit
  ├── fix/embedding-quality
```

### Branch Rules

**`main` Branch**:
- Protected branch
- Nur via Pull Request + Review
- Alle Tests müssen grün sein
- Tagged releases (v1.0.0, v1.1.0, etc.)
- **Deploys zu Production Servers**

**`dev` Branch**:
- Feature integration branch
- Auto-deploy zu arkturian.com (staging)
- Feature branches mergen hier
- CI/CD pipeline runs automatisch

**Feature Branches**:
- `feature/*` - neue Features
- `fix/*` - Bug fixes
- `hotfix/*` - Critical production fixes (direkt von main)

---

## CI/CD Pipeline

### GitHub Actions Workflow

**File**: `.github/workflows/deploy.yml`

```yaml
name: Deploy Storage API

on:
  push:
    branches:
      - dev          # Auto-deploy to staging
      - main         # Manual deploy to production
  pull_request:
    branches:
      - main
      - dev

env:
  PYTHON_VERSION: "3.11"

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: pytest tests/ -v --cov=. --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  deploy-staging:
    name: Deploy to Staging (arkturian.com)
    needs: test
    if: github.ref == 'refs/heads/dev' && github.event_name == 'push'
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: https://api-storage.arkturian.com

    steps:
      - uses: actions/checkout@v3

      - name: Deploy to Staging Server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.STAGING_HOST }}
          username: ${{ secrets.STAGING_USER }}
          key: ${{ secrets.STAGING_SSH_KEY }}
          script: |
            cd /var/www/api-storage.arkturian.com
            git fetch origin
            git checkout dev
            git pull origin dev
            /root/.pyenv/versions/3.11.9/bin/pip install -r requirements.txt
            systemctl restart storage-api.service
            sleep 3
            systemctl status storage-api.service --no-pager

      - name: Health Check
        run: |
          sleep 5
          curl -f https://api-storage.arkturian.com/health || exit 1

  deploy-production:
    name: Deploy to Production
    needs: test
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://api.customer.com

    steps:
      - uses: actions/checkout@v3

      - name: Create Release
        id: create_release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: v${{ github.run_number }}
          release_name: Release v${{ github.run_number }}
          draft: false
          prerelease: false

      - name: Deploy to Production Servers
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.PRODUCTION_HOSTS }}  # Comma-separated list
          username: ${{ secrets.PRODUCTION_USER }}
          key: ${{ secrets.PRODUCTION_SSH_KEY }}
          script: |
            cd /var/www/api-storage
            git fetch origin
            git checkout main
            git pull origin main
            /root/.pyenv/versions/3.11.9/bin/pip install -r requirements.txt
            systemctl restart storage-api.service
            sleep 3
            systemctl status storage-api.service --no-pager

      - name: Production Health Check
        run: |
          sleep 5
          for host in ${{ secrets.PRODUCTION_URLS }}; do
            curl -f $host/health || exit 1
          done

      - name: Rollback on Failure
        if: failure()
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.PRODUCTION_HOSTS }}
          username: ${{ secrets.PRODUCTION_USER }}
          key: ${{ secrets.PRODUCTION_SSH_KEY }}
          script: |
            cd /var/www/api-storage
            git checkout main^  # Previous commit
            systemctl restart storage-api.service
```

---

## Release Process

### 1. Development Phase

```bash
# Create feature branch
git checkout dev
git pull origin dev
git checkout -b feature/my-feature

# Work on feature
git add .
git commit -m "Add my feature"
git push origin feature/my-feature

# Create PR to dev
# → Auto-deploy to arkturian.com (staging) after merge
```

### 2. Testing & QA on Staging

- Test auf arkturian.com
- Manual QA
- Performance testing
- Security checks

### 3. Production Release

```bash
# Merge dev to main via PR
git checkout main
git pull origin main
git merge dev

# Create release tag
git tag -a v1.1.0 -m "Release v1.1.0: Vibe-search and improved embeddings"
git push origin main --tags

# → Auto-deploy to all production servers
```

### 4. Hotfix Process (Critical Bug in Production)

```bash
# Branch from main
git checkout main
git checkout -b hotfix/critical-bug

# Fix the bug
git add .
git commit -m "Fix critical bug"

# Merge to main
git checkout main
git merge hotfix/critical-bug
git push origin main

# Backport to dev
git checkout dev
git merge main
git push origin dev
```

---

## GitHub Secrets Configuration

Set up diese secrets in GitHub Settings → Secrets and variables → Actions:

### Staging (arkturian.com)
- `STAGING_HOST`: arkturian.com
- `STAGING_USER`: root
- `STAGING_SSH_KEY`: SSH private key

### Production
- `PRODUCTION_HOSTS`: "server1.com,server2.com"
- `PRODUCTION_URLS`: "https://api1.com,https://api2.com"
- `PRODUCTION_USER`: root
- `PRODUCTION_SSH_KEY`: SSH private key

---

## Deployment Checklist

### Before Deployment

- [ ] All tests passing
- [ ] Code reviewed and approved
- [ ] Changelog updated
- [ ] Documentation updated
- [ ] Database migrations tested
- [ ] Backward compatibility verified

### After Deployment

- [ ] Health check passes
- [ ] API endpoints responding
- [ ] Error logs checked
- [ ] Performance metrics normal
- [ ] Customer notification (if required)

---

## Monitoring & Rollback

### Health Checks

```bash
# Quick health check
curl https://api-storage.arkturian.com/health

# Detailed status
ssh root@arkturian.com "systemctl status storage-api.service"
```

### Rollback Strategy

**Automated Rollback** (in CI/CD):
- Health check fails → automatic rollback to previous commit

**Manual Rollback**:
```bash
ssh root@server.com
cd /var/www/api-storage
git log --oneline -n 5  # Find previous commit
git checkout <previous-commit-hash>
systemctl restart storage-api.service
```

---

## Multi-Tenant Deployment Considerations

### Customer-Specific Configuration

Jeder Production Server hat eigene `.env` file:

**arkturian.com (staging)**:
```env
ENVIRONMENT=staging
TENANT_ID=arkturian
DATABASE_URL=sqlite:////var/www/api-storage.arkturian.com/storage.db
OPENAI_API_KEY=sk-...dev-key...
```

**customer1.com (production)**:
```env
ENVIRONMENT=production
TENANT_ID=customer1
DATABASE_URL=postgresql://...customer1_db...
OPENAI_API_KEY=sk-...prod-key...
SENTRY_DSN=https://...customer1...
```

### Tenant Isolation

- Separate databases per customer
- Tenant-specific ChromaDB collections
- API rate limits per tenant
- Separate error tracking (Sentry projects)

---

## Next Steps

1. ✅ Create GitHub repository (if not exists)
2. ⏳ Set up `.github/workflows/deploy.yml`
3. ⏳ Configure GitHub Secrets
4. ⏳ Test CI/CD with dev branch
5. ⏳ Document customer-specific setup process
6. ⏳ Create server provisioning script
