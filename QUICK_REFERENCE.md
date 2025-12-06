# Wallpaper Rotation - Quick Reference Card

## Common Commands

### Check Status

```bash
# See if timer is running
systemctl --user status wallpaper-rotation.timer

# See when next rotation will occur
systemctl --user list-timers wallpaper-rotation.timer

# View recent logs (last 20 lines)
journalctl --user -u wallpaper-rotation.service -n 20
```

### Manual Control

```bash
# Trigger rotation now (don't wait for timer)
systemctl --user start wallpaper-rotation.service

# Stop the timer
systemctl --user stop wallpaper-rotation.timer

# Restart the timer (after config changes)
systemctl --user restart wallpaper-rotation.timer

# Enable on boot
systemctl --user enable wallpaper-rotation.timer

# Disable on boot
systemctl --user disable wallpaper-rotation.timer
```

### Logs

```bash
# Follow logs in real-time
journalctl --user -u wallpaper-rotation.service -f

# Last 50 lines
journalctl --user -u wallpaper-rotation.service -n 50

# Logs from last hour
journalctl --user -u wallpaper-rotation.service --since "1 hour ago"

# Logs from today
journalctl --user -u wallpaper-rotation.service --since today

# All logs
journalctl --user -u wallpaper-rotation.service
```

### Configuration

#### Change rotation interval

```bash
# Edit timer
nano ~/.config/systemd/user/wallpaper-rotation.timer

# Change these lines:
#   OnUnitActiveSec=30min    # How often to run
#   OnBootSec=2min           # Delay after boot

# Apply changes
systemctl --user daemon-reload
systemctl --user restart wallpaper-rotation.timer
```

#### Adjust color matching

```bash
# Edit script
nano ~/.local/bin/wallpaper-rotation.py

# Find and adjust:
#   MAX_MATCH_DIST = 45.0    # Lower = more strict
#   DRIFT_TOLERANCE = 25.0   # Theme change detection
#   QUEUE_SIZE = 10          # How many wallpapers to queue
```

### Troubleshooting

```bash
# Test script manually
~/.local/bin/wallpaper-rotation.py

# Check if ImageMagick is installed
which magick

# Check if pywal is active
cat ~/.cache/wal/colors

# Check wallpaper directory
ls ~/Pictures/Wallpapers

# View cached colors
cat ~/.cache/wallpaper_colors.json | head -n 20

# Reset queue (forces recalculation)
rm ~/.cache/wallpaper_queue_state.json

# Reset history (allows reusing old wallpapers)
rm ~/.cache/wallpaper_global_history.txt

# Full reset
rm ~/.cache/wallpaper_*.{json,txt}
```

## Hyprland Integration (Optional)

Add to `~/.config/hypr/hyprland.conf`:

```bash
# Start timer on Hyprland startup
exec-once = systemctl --user start wallpaper-rotation.timer
```

## Common Timer Intervals

```ini
# Every 15 minutes
OnUnitActiveSec=15min

# Every 30 minutes (default)
OnUnitActiveSec=30min

# Every hour
OnUnitActiveSec=1h

# Every 2 hours
OnUnitActiveSec=2h

# Once per day at startup
OnBootSec=5min
OnUnitActiveSec=1d

# Cron-style (every 15 minutes)
OnCalendar=*:0/15

# Hourly (on the hour)
OnCalendar=hourly

# Daily at 9 AM
OnCalendar=09:00
```

## Quick Diagnostics

```bash
# Is everything working?
systemctl --user is-active wallpaper-rotation.timer && echo "✓ Timer is active"
systemctl --user is-enabled wallpaper-rotation.timer && echo "✓ Timer is enabled"

# When is next run?
systemctl --user list-timers wallpaper-rotation.timer | grep -A1 NEXT

# Last few rotations
journalctl --user -u wallpaper-rotation.service -n 5 --output=short-precise

# Any errors?
journalctl --user -u wallpaper-rotation.service -p err -n 10
```

## File Locations

| File                                                | Purpose        |
| --------------------------------------------------- | -------------- |
| `~/.local/bin/wallpaper-rotation.py`                | Main script    |
| `~/.config/systemd/user/wallpaper-rotation.timer`   | Timer config   |
| `~/.config/systemd/user/wallpaper-rotation.service` | Service config |
| `~/.cache/wallpaper_colors.json`                    | Color cache    |
| `~/.cache/wallpaper_queue_state.json`               | Current queue  |
| `~/.cache/wallpaper_global_history.txt`             | Usage history  |
| `~/.cache/wal/colors`                               | Pywal colors   |

## Uninstall

```bash
# Stop and disable
systemctl --user stop wallpaper-rotation.timer
systemctl --user disable wallpaper-rotation.timer

# Remove files
rm ~/.config/systemd/user/wallpaper-rotation.{service,timer}
rm ~/.local/bin/wallpaper-rotation.py

# Reload systemd
systemctl --user daemon-reload

# Remove cache (optional)
rm ~/.cache/wallpaper_*.{json,txt}
```
