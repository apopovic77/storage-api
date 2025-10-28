#!/usr/bin/env bash
#############################################
# Setup Storage Cache Cleanup Service
#
# Installs systemd timer for automatic cache cleanup
#############################################

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

REPO_PATH="${REPO_PATH:-/var/www/api-storage.arkturian.com}"
SYSTEMD_PATH="/etc/systemd/system"

echo -e "${GREEN}🔧 Setting up Storage Cache Cleanup Service${NC}"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}❌ This script must be run as root${NC}"
   exit 1
fi

# Copy service files
echo -e "${YELLOW}📋 Installing systemd service files...${NC}"
cp "$REPO_PATH/systemd/storage-cache-cleanup.service" "$SYSTEMD_PATH/"
cp "$REPO_PATH/systemd/storage-cache-cleanup.timer" "$SYSTEMD_PATH/"

# Reload systemd
echo -e "${YELLOW}🔄 Reloading systemd daemon...${NC}"
systemctl daemon-reload

# Enable and start timer
echo -e "${YELLOW}▶️  Enabling and starting timer...${NC}"
systemctl enable storage-cache-cleanup.timer
systemctl start storage-cache-cleanup.timer

# Show status
echo -e "\n${GREEN}✅ Cache cleanup service installed successfully!${NC}\n"

echo -e "${GREEN}📊 Timer Status:${NC}"
systemctl status storage-cache-cleanup.timer --no-pager

echo -e "\n${GREEN}⏰ Next scheduled run:${NC}"
systemctl list-timers storage-cache-cleanup.timer --no-pager

echo -e "\n${GREEN}📝 Useful Commands:${NC}"
echo -e "  Check timer status:  systemctl status storage-cache-cleanup.timer"
echo -e "  View logs:           journalctl -u storage-cache-cleanup.service"
echo -e "  Run manually:        systemctl start storage-cache-cleanup.service"
echo -e "  Stop timer:          systemctl stop storage-cache-cleanup.timer"
echo -e "  Disable timer:       systemctl disable storage-cache-cleanup.timer"
