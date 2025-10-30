#!/usr/bin/env bash

#############################################
# Server Cleanup Script
#
# Removes deployment from a server completely.
# Stops service, removes files, nginx config, backups (optional).
#############################################

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
SERVER_CONFIG=""
SERVER_NAME=""
KEEP_BACKUPS=false
FORCE=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVOPS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Remove deployment from a server completely.

OPTIONS:
    --server NAME          Server name (reads .devops/servers/NAME.yaml)
    --config PATH         Direct path to server config YAML
    --keep-backups        Don't remove backup directory
    --force               Skip confirmation prompt
    -h, --help            Show this help message

EXAMPLES:
    $(basename "$0") --server arkserver
    $(basename "$0") --server staging --force
    $(basename "$0") --server prod --keep-backups

WARNING: This will stop the service and delete all deployed files!

EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server)
            SERVER_NAME="$2"
            SERVER_CONFIG="$DEVOPS_DIR/servers/$2.yaml"
            shift 2
            ;;
        --config)
            SERVER_CONFIG="$2"
            shift 2
            ;;
        --keep-backups)
            KEEP_BACKUPS=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validation
if [ -z "$SERVER_CONFIG" ]; then
    echo -e "${RED}❌ Error: --server or --config required${NC}"
    usage
fi

if [ ! -f "$SERVER_CONFIG" ]; then
    echo -e "${RED}❌ Error: Server config not found: $SERVER_CONFIG${NC}"
    exit 1
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}⚠️  Server Cleanup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "Config: ${YELLOW}$SERVER_CONFIG${NC}"
echo ""

# Simple YAML parser
parse_yaml() {
    local file=$1
    local prefix=$2
    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
    sed -ne "s|^\($s\):|\1|" \
         -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
         -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" "$file" |
    awk -F$fs '{
        indent = length($1)/2;
        vname[indent] = $2;
        for (i in vname) {if (i > indent) {delete vname[i]}}
        if (length($3) > 0) {
            vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
            printf("%s%s%s=\"%s\"\n", "'$prefix'", vn, $2, $3);
        }
    }'
}

# Load configuration
echo -e "${YELLOW}📋 Loading server configuration...${NC}"
eval $(parse_yaml "$SERVER_CONFIG" "CFG_")

SERVER_HOST="${CFG_server_host:-}"
SERVER_USER="${CFG_server_user:-root}"
DEPLOY_PATH="${CFG_server_deploy_path:-}"
SERVICE_NAME="${CFG_service_name:-}"
NGINX_ENABLED="${CFG_nginx_enabled:-true}"
BACKUP_DIR="${CFG_backup_dir:-/var/backups}"
BACKUP_PREFIX="${CFG_backup_prefix:-$SERVICE_NAME}"

# Validate
if [ -z "$SERVER_HOST" ] || [ -z "$DEPLOY_PATH" ] || [ -z "$SERVICE_NAME" ]; then
    echo -e "${RED}❌ Error: Missing required config fields${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Configuration loaded${NC}"
echo -e "  Server: ${YELLOW}$SERVER_USER@$SERVER_HOST${NC}"
echo -e "  Deploy path: ${YELLOW}$DEPLOY_PATH${NC}"
echo -e "  Service: ${YELLOW}$SERVICE_NAME${NC}"
echo ""

# Warning and confirmation
echo -e "${RED}⚠️  WARNING: This will:${NC}"
echo -e "  ${RED}•${NC} Stop and disable the systemd service"
echo -e "  ${RED}•${NC} Remove the service file"
echo -e "  ${RED}•${NC} Delete all deployed files: $DEPLOY_PATH"
if [ "$NGINX_ENABLED" = "true" ]; then
    echo -e "  ${RED}•${NC} Remove nginx configuration"
fi
if [ "$KEEP_BACKUPS" = false ]; then
    echo -e "  ${RED}•${NC} Delete all backups in $BACKUP_DIR/$BACKUP_PREFIX-*"
else
    echo -e "  ${GREEN}•${NC} Keep backups (--keep-backups specified)"
fi
echo ""

if [ "$FORCE" = false ]; then
    read -p "Are you sure you want to proceed? Type 'yes' to confirm: " -r
    echo
    if [ "$REPLY" != "yes" ]; then
        echo -e "${YELLOW}Cleanup cancelled${NC}"
        exit 0
    fi
fi

# Stop and disable service
echo -e "${YELLOW}🛑 Stopping service...${NC}"
if ssh "$SERVER_USER@$SERVER_HOST" "systemctl is-active --quiet $SERVICE_NAME" 2>/dev/null; then
    ssh "$SERVER_USER@$SERVER_HOST" "systemctl stop $SERVICE_NAME"
    echo -e "${GREEN}✅ Service stopped${NC}"
else
    echo -e "${YELLOW}⚠️  Service was not running${NC}"
fi

echo -e "${YELLOW}🔧 Disabling service...${NC}"
if ssh "$SERVER_USER@$SERVER_HOST" "systemctl is-enabled --quiet $SERVICE_NAME" 2>/dev/null; then
    ssh "$SERVER_USER@$SERVER_HOST" "systemctl disable $SERVICE_NAME"
    echo -e "${GREEN}✅ Service disabled${NC}"
else
    echo -e "${YELLOW}⚠️  Service was not enabled${NC}"
fi

# Remove service file
echo -e "${YELLOW}🗑️  Removing service file...${NC}"
if ssh "$SERVER_USER@$SERVER_HOST" "[ -f '/etc/systemd/system/$SERVICE_NAME.service' ]"; then
    ssh "$SERVER_USER@$SERVER_HOST" "rm -f /etc/systemd/system/$SERVICE_NAME.service && systemctl daemon-reload"
    echo -e "${GREEN}✅ Service file removed${NC}"
else
    echo -e "${YELLOW}⚠️  Service file not found${NC}"
fi

# Remove nginx config
if [ "$NGINX_ENABLED" = "true" ]; then
    echo -e "${YELLOW}🌐 Removing nginx configuration...${NC}"
    if ssh "$SERVER_USER@$SERVER_HOST" "[ -f '/etc/nginx/sites-enabled/$SERVICE_NAME' ]"; then
        ssh "$SERVER_USER@$SERVER_HOST" "rm -f /etc/nginx/sites-enabled/$SERVICE_NAME /etc/nginx/sites-available/$SERVICE_NAME && nginx -t && systemctl reload nginx"
        echo -e "${GREEN}✅ Nginx configuration removed${NC}"
    else
        echo -e "${YELLOW}⚠️  Nginx config not found${NC}"
    fi
fi

# Remove deployed files
echo -e "${YELLOW}📁 Removing deployed files...${NC}"
if ssh "$SERVER_USER@$SERVER_HOST" "[ -d '$DEPLOY_PATH' ]"; then
    ssh "$SERVER_USER@$SERVER_HOST" "rm -rf '$DEPLOY_PATH'"
    echo -e "${GREEN}✅ Deployed files removed${NC}"
else
    echo -e "${YELLOW}⚠️  Deploy directory not found${NC}"
fi

# Remove backups (optional)
if [ "$KEEP_BACKUPS" = false ]; then
    echo -e "${YELLOW}🗑️  Removing backups...${NC}"
    BACKUP_COUNT=$(ssh "$SERVER_USER@$SERVER_HOST" "ls -d $BACKUP_DIR/$BACKUP_PREFIX-* 2>/dev/null | wc -l" || echo "0")

    if [ "$BACKUP_COUNT" -gt 0 ]; then
        ssh "$SERVER_USER@$SERVER_HOST" "rm -rf $BACKUP_DIR/$BACKUP_PREFIX-*"
        echo -e "${GREEN}✅ Removed $BACKUP_COUNT backup(s)${NC}"
    else
        echo -e "${YELLOW}⚠️  No backups found${NC}"
    fi
else
    echo -e "${BLUE}ℹ️  Keeping backups as requested${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Cleanup completed successfully!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}📊 Summary:${NC}"
echo -e "  Server: ${YELLOW}$SERVER_USER@$SERVER_HOST${NC}"
echo -e "  Service: ${YELLOW}$SERVICE_NAME${NC} ${GREEN}(removed)${NC}"
echo -e "  Deploy path: ${YELLOW}$DEPLOY_PATH${NC} ${GREEN}(deleted)${NC}"
if [ "$KEEP_BACKUPS" = true ]; then
    echo -e "  Backups: ${GREEN}Kept${NC}"
else
    echo -e "  Backups: ${GREEN}Deleted${NC}"
fi
echo ""
echo -e "${YELLOW}To redeploy:${NC}"
echo -e "  ${BLUE}./.devops/scripts/setup-server.sh --server $SERVER_NAME${NC}"
echo ""
