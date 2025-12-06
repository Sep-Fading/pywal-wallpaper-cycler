#!/bin/bash
# Wallpaper Rotation Service - Automated Installation Script

set -e

echo "=== Wallpaper Rotation Service Installer ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if script file exists
if [ ! -f "wallpaper_rotation_improved.py" ]; then
    echo -e "${RED}Error: wallpaper_rotation_improved.py not found in current directory${NC}"
    exit 1
fi

if [ ! -f "wallpaper-rotation.service" ]; then
    echo -e "${RED}Error: wallpaper-rotation.service not found in current directory${NC}"
    exit 1
fi

if [ ! -f "wallpaper-rotation.timer" ]; then
    echo -e "${RED}Error: wallpaper-rotation.timer not found in current directory${NC}"
    exit 1
fi

# Step 1: Install script
echo -e "${GREEN}[1/5]${NC} Installing script to ~/.local/bin/"
mkdir -p ~/.local/bin
cp wallpaper_rotation_improved.py ~/.local/bin/wallpaper-rotation.py
chmod +x ~/.local/bin/wallpaper-rotation.py
echo "  ✓ Script installed"

# Step 2: Check PATH
echo -e "${GREEN}[2/5]${NC} Checking PATH configuration"
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo -e "${YELLOW}  ⚠ ~/.local/bin is not in PATH${NC}"
    echo "  Add this to your ~/.bashrc or ~/.zshrc:"
    echo '    export PATH="$HOME/.local/bin:$PATH"'
else
    echo "  ✓ PATH is configured correctly"
fi

# Step 3: Install systemd service files
echo -e "${GREEN}[3/5]${NC} Installing systemd service files"
mkdir -p ~/.config/systemd/user
cp wallpaper-rotation.service ~/.config/systemd/user/
cp wallpaper-rotation.timer ~/.config/systemd/user/
echo "  ✓ Service files installed"

# Step 4: Reload systemd and enable
echo -e "${GREEN}[4/5]${NC} Enabling systemd timer"
systemctl --user daemon-reload
systemctl --user enable wallpaper-rotation.timer
systemctl --user start wallpaper-rotation.timer
echo "  ✓ Timer enabled and started"

# Step 5: Check status
echo -e "${GREEN}[5/5]${NC} Checking status"
sleep 2
systemctl --user status wallpaper-rotation.timer --no-pager | head -n 10
echo ""

echo -e "${GREEN}=== Installation Complete! ===${NC}"
echo ""
echo "Next steps:"
echo "  • Check logs: journalctl --user -u wallpaper-rotation.service -f"
echo "  • Manual trigger: systemctl --user start wallpaper-rotation.service"
echo "  • Check timer: systemctl --user list-timers"
echo ""
echo "Optional: Add to Hyprland config (~/.config/hypr/hyprland.conf):"
echo "  exec-once = systemctl --user start wallpaper-rotation.timer"
echo ""
echo "Configuration:"
echo "  • Script location: ~/.local/bin/wallpaper-rotation.py"
echo "  • Service files: ~/.config/systemd/user/"
echo "  • Logs: journalctl --user -u wallpaper-rotation.service"
echo ""
echo "The timer will run:"
echo "  • 2 minutes after boot"
echo "  • Every 30 minutes after that"
echo ""
echo "To adjust timing, edit: ~/.config/systemd/user/wallpaper-rotation.timer"
echo "Then run: systemctl --user daemon-reload && systemctl --user restart wallpaper-rotation.timer"
