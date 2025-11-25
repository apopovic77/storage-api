# Server Configurations

This directory contains YAML configuration files for deployment targets.

## Structure

```
.devops/servers/
├── production.yaml    # Production server
├── staging.yaml       # Staging server
├── arkserver.yaml     # Internal test server
└── README.md          # This file
```

## Configuration Format

Each server config is a YAML file with the following structure:

```yaml
server:
  name: "my-server"
  type: "python-api"  # python-api | node | php | static
  host: "example.com"
  user: "root"
  deploy_path: "/var/www/my-app"

service:
  type: "systemd"
  name: "my-app"
  port: 8000

nginx:
  enabled: true
  server_name: "app.example.com"
  port: 80

environment:
  # Your app's environment variables
  DATABASE_URL: "..."
  API_KEY: "{{SECRET}}"

python:  # For Python APIs
  version: "3.11"
  requirements: "requirements.txt"
  venv_path: "venv"

backup:
  enabled: true
  dir: "/var/backups"
  prefix: "my-app"
```

## Usage

### Initial Server Setup

```bash
# Setup a new server
./.devops/scripts/setup-server.sh --server production

# Force reinstall
./.devops/scripts/setup-server.sh --server staging --force
```

### Deploy to Server

```bash
# Deploy to specific server
./.devops/scripts/deploy-to-server.sh --server production

# Deploy without build step
./.devops/scripts/deploy-to-server.sh --server staging --skip-build
```

### Cleanup Server

```bash
# Remove deployment completely
./.devops/scripts/cleanup-server.sh --server staging

# Keep backups
./.devops/scripts/cleanup-server.sh --server staging --keep-backups

# Skip confirmation
./.devops/scripts/cleanup-server.sh --server test --force
```

## Server Types

### python-api
- Creates Python venv
- Installs from requirements.txt
- Runs with uvicorn
- Systemd service management

### node
- Runs npm build
- Deploys dist/ folder
- Static file serving or Node.js app

### php
- Direct file deployment
- No build step
- Works with Apache/Nginx + PHP-FPM

### static
- Plain static files
- No build, no service
- Just nginx serving files

## Environment Variables

For secrets, use `{{SECRET}}` placeholders in the config:

```yaml
environment:
  API_KEY: "{{SECRET}}"
  DATABASE_PASSWORD: "{{SECRET}}"
```

Then provide them at deploy time or manage via server-side .env files.

## Multiple Environments

Create separate configs for each environment:

```
.devops/servers/
├── production.yaml
├── staging.yaml
└── dev.yaml
```

Then deploy to specific environments:

```bash
./.devops/scripts/deploy-to-server.sh --server production
./.devops/scripts/deploy-to-server.sh --server staging
```

## Testing Deployment Cycle

Test that your deployment is idempotent:

```bash
# 1. Clean deployment
./.devops/scripts/cleanup-server.sh --server test --force

# 2. Initial setup
./.devops/scripts/setup-server.sh --server test

# 3. Verify works
curl http://test.example.com/health

# 4. Cleanup again
./.devops/scripts/cleanup-server.sh --server test --force

# 5. Setup again (should be smooth)
./.devops/scripts/setup-server.sh --server test

# 6. Verify works again
curl http://test.example.com/health
```

## Best Practices

1. **Don't commit secrets** - Use {{SECRET}} placeholders
2. **Test on staging first** - Always test before production
3. **Keep backups** - Use --keep-backups for important servers
4. **Document your servers** - Add comments to configs
5. **Version control configs** - Commit config files to git

## Troubleshooting

### Setup fails

```bash
# Check SSH connection
ssh user@server "echo 'Connection OK'"

# Check Python version
ssh user@server "python3 --version"

# Check disk space
ssh user@server "df -h"
```

### Service won't start

```bash
# Check service logs
ssh user@server "journalctl -u my-service -n 100"

# Check service status
ssh user@server "systemctl status my-service"

# Manual start
ssh user@server "systemctl start my-service"
```

### Deployment fails

```bash
# Check deploy path exists
ssh user@server "ls -la /var/www/my-app"

# Check permissions
ssh user@server "ls -la /var/www/"

# Check venv
ssh user@server "ls -la /var/www/my-app/venv/bin/"
```

## Examples

See the example configs:
- `example-python-api.yaml` - Python/FastAPI app
- Check your project's specific configs for real examples
