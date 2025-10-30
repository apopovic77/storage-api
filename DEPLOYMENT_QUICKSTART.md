# Deployment Quick Start Guide

## Setup Overview

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  GitHub Repository (storage-api)                            │
│  ├── main branch    → Production servers                    │
│  └── dev branch     → Staging (arkturian.com)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │      CI/CD (GitHub Actions)           │
        │  ✓ Tests                              │
        │  ✓ Auto-deploy                        │
        │  ✓ Health checks                      │
        │  ✓ Rollback on failure                │
        └───────────────────────────────────────┘
                 │                     │
                 ▼                     ▼
    ┌─────────────────────┐  ┌─────────────────────┐
    │  Staging Server     │  │ Production Servers  │
    │  arkturian.com      │  │ customer1.com       │
    │  (dev branch)       │  │ customer2.com       │
    │                     │  │ (main branch)       │
    └─────────────────────┘  └─────────────────────┘
```

---

## Initial Setup (One Time)

### 1. Create GitHub Repository

```bash
cd /Volumes/DatenAP/Code/storage-api

# Initialize git (if not already)
git init

# Create branches
git checkout -b main
git add .
git commit -m "Initial commit"

git checkout -b dev
git push -u origin main
git push -u origin dev
```

### 2. Configure GitHub Secrets

Go to: **Settings → Secrets and variables → Actions**

Add these secrets:

**Staging Secrets**:
- `STAGING_HOST` = `arkturian.com`
- `STAGING_USER` = `root`
- `STAGING_SSH_KEY` = `<your-ssh-private-key>`

**Production Secrets**:
- `PRODUCTION_HOSTS` = `server1.com,server2.com` (comma-separated)
- `PRODUCTION_USER` = `root`
- `PRODUCTION_SSH_KEY` = `<your-ssh-private-key>`
- `PRODUCTION_URLS` = `https://api1.com,https://api2.com` (for health checks)

### 3. Enable GitHub Actions

- Go to **Actions** tab
- Enable workflows
- CI/CD pipeline will run automatically on push

---

## Daily Workflow

### Feature Development

```bash
# Start from dev
git checkout dev
git pull origin dev

# Create feature branch
git checkout -b feature/my-feature

# Work on feature
# ... make changes ...

git add .
git commit -m "Add my feature"
git push origin feature/my-feature

# Create Pull Request to dev branch
# → After merge: Auto-deploys to arkturian.com (staging)
```

### Testing on Staging

After merge to `dev`, CI/CD automatically:
1. Runs tests
2. Deploys to arkturian.com
3. Runs health checks
4. Notifies you of success/failure

Test manually:
```bash
# Health check
curl https://api-storage.arkturian.com/health

# Test API endpoints
curl https://api-storage.arkturian.com/storage/kg/search?query=helmet

# Check logs
ssh root@arkturian.com
tail -f /var/log/storage-api-error.log
```

### Production Release

```bash
# After testing on staging, merge to main
git checkout main
git pull origin main
git merge dev

# Push to production
git push origin main

# → Auto-deploys to all production servers
```

---

## New Customer Server Setup

### Option 1: Automated Setup Script

```bash
# Copy script to new server
scp setup-new-server.sh root@newserver.com:/tmp/

# Run on server
ssh root@newserver.com
chmod +x /tmp/setup-new-server.sh
/tmp/setup-new-server.sh customer-name api.customer.com
```

The script will:
- ✅ Install system dependencies
- ✅ Setup Python environment
- ✅ Clone repository
- ✅ Configure .env
- ✅ Initialize database
- ✅ Setup systemd service
- ✅ Configure Nginx + SSL
- ✅ Start services

### Option 2: Manual Setup

See `RELEASE_MANAGEMENT.md` for detailed manual setup instructions.

---

## Monitoring & Troubleshooting

### Check Service Status

```bash
# Staging
ssh root@arkturian.com "systemctl status storage-api.service"

# Production
ssh root@customer.com "systemctl status storage-api-customer.service"
```

### View Logs

```bash
# Real-time logs
ssh root@server.com "tail -f /var/log/storage-api-error.log"

# Last 100 lines
ssh root@server.com "tail -100 /var/log/storage-api-error.log"
```

### Restart Service

```bash
ssh root@server.com "systemctl restart storage-api.service"
```

### Manual Rollback

```bash
ssh root@server.com
cd /var/www/api-storage
git log --oneline -n 5
git checkout <previous-commit>
systemctl restart storage-api.service
```

---

## Common Tasks

### Update Dependencies

```bash
# Staging
ssh root@arkturian.com
cd /var/www/api-storage.arkturian.com
git pull origin dev
/root/.pyenv/versions/3.11.9/bin/pip install -r requirements.txt
systemctl restart storage-api.service
```

### Database Migration

```bash
# Staging test first
ssh root@arkturian.com
cd /var/www/api-storage.arkturian.com
/root/.pyenv/versions/3.11.9/bin/python migrate.py

# Then production
ssh root@production.com
cd /var/www/api-storage
/root/.pyenv/versions/3.11.9/bin/python migrate.py
```

### Regenerate Embeddings (after AI updates)

```bash
# Use the regeneration script
scp regenerate-embeddings-server.py root@server.com:/tmp/
ssh root@server.com
cd /var/www/api-storage
export OPENAI_API_KEY="<key>"
export DATABASE_URL="sqlite:////var/www/api-storage/storage.db"
python3 /tmp/regenerate-embeddings-server.py
```

---

## Emergency Procedures

### Service Down

```bash
# 1. Check status
ssh root@server.com "systemctl status storage-api.service"

# 2. Check logs
ssh root@server.com "tail -100 /var/log/storage-api-error.log"

# 3. Restart
ssh root@server.com "systemctl restart storage-api.service"

# 4. If still failing, rollback
ssh root@server.com
cd /var/www/api-storage
git checkout main^  # Previous commit
systemctl restart storage-api.service
```

### High CPU/Memory

```bash
# Check resource usage
ssh root@server.com "htop"

# Check gunicorn workers
ssh root@server.com "ps aux | grep gunicorn"

# Restart with fewer workers
ssh root@server.com
nano /etc/systemd/system/storage-api.service
# Change -w 4 to -w 2
systemctl daemon-reload
systemctl restart storage-api.service
```

---

## Resources

- **Full Documentation**: `RELEASE_MANAGEMENT.md`
- **Setup Script**: `setup-new-server.sh`
- **CI/CD Config**: `.github/workflows/deploy.yml`
- **Health Check**: `https://<domain>/health`
- **API Docs**: `https://<domain>/docs`

---

## Support

For issues or questions:
1. Check logs first
2. Review GitHub Actions run
3. Test health endpoint
4. Check systemd service status
5. Review RELEASE_MANAGEMENT.md

**Emergency Contact**: [Your contact info]
