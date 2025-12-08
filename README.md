# Subtle Wallpaper Rotation System

A smart wallpaper rotation system that seamlessly integrates with pywal, changing wallpapers with imperceptibly subtle color transitions.

## 🎯 What This Does

- Rotates through your wallpapers with **color-matched transitions**
- Queues wallpapers that are visually similar to your current theme
- Prevents jarring color changes (adjustable threshold)
- Tracks history to avoid repetition
- Runs as a systemd service for reliability
- Survives reboots and integrates with Hyprland
- It's kind of slow so I might just remake it at somepoint.

### Installation

```bash
# 1. Install script
mkdir -p ~/.local/bin
cp wallpaper_rotation_improved.py ~/.local/bin/wallpaper-rotation.py
chmod +x ~/.local/bin/wallpaper-rotation.py

# 2. Install systemd files
mkdir -p ~/.config/systemd/user
cp wallpaper-rotation.{service,timer} ~/.config/systemd/user/

# 3. Enable and start
systemctl --user daemon-reload
systemctl --user enable --now wallpaper-rotation.timer

# 4. Check status
systemctl --user status wallpaper-rotation.timer
```

### Hyprland Integration (Recommended)

Add to `~/.config/hypr/hyprland.conf`:

```bash
exec-once = systemctl --user start wallpaper-rotation.timer
```

This ensures the timer starts when you log in.

## 📋 Prerequisites

- **Python 3.6+**
- **ImageMagick** (for `magick` command)
- **pywal** (for color theming)
- **Systemd** (for service management)
- Wallpapers in `~/Pictures/Wallpapers/`

Install missing dependencies:

```bash
# Arch Linux
sudo pacman -S python imagemagick python-pywal

# Ubuntu/Debian
sudo apt install python3 imagemagick

# pywal (via pip)
pip install pywal
```

## 🎛️ Configuration

### Rotation Frequency

Edit `~/.config/systemd/user/wallpaper-rotation.timer`:

```ini
[Timer]
OnBootSec=2min        # Wait 2 minutes after boot
OnUnitActiveSec=30min # Run every 30 minutes
```

Common intervals:

- `15min` - Every 15 minutes (frequent)
- `30min` - Every 30 minutes (default)
- `1h` - Every hour
- `4h` - Every 4 hours (infrequent)

After changes:

```bash
systemctl --user daemon-reload
systemctl --user restart wallpaper-rotation.timer
```

### Color Matching Strictness

Edit `~/.local/bin/wallpaper-rotation.py`:

```python
MAX_MATCH_DIST = 45.0   # Lower = more strict (try 35.0)
                        # Higher = less strict (try 60.0)

DRIFT_TOLERANCE = 25.0  # How much theme can change
                        # before rebuilding queue

QUEUE_SIZE = 10         # How many wallpapers to queue
```

## 📊 Common Commands

```bash
# Check status
systemctl --user status wallpaper-rotation.timer

# See when next rotation occurs
systemctl --user list-timers wallpaper-rotation.timer

# Trigger rotation now
systemctl --user start wallpaper-rotation.service

# View logs
journalctl --user -u wallpaper-rotation.service -f

# Stop/start timer
systemctl --user stop wallpaper-rotation.timer
systemctl --user start wallpaper-rotation.timer
```

See `QUICK_REFERENCE.md` for more commands.

## 🔍 How It Works

1. **Reads current pywal theme** - Gets your background color
2. **Scans wallpaper directory** - Extracts dominant color from each image
3. **Calculates color distance** - Finds wallpapers similar to current theme
4. **Builds a queue** - Selects top 10 closest matches
5. **Rotates through queue** - Applies next wallpaper
6. **Tracks history** - Avoids repetition until full cycle complete
7. **Detects drift** - Rebuilds queue if theme changes significantly

### Queue System

- Wallpapers are pre-calculated into a queue based on color similarity
- Queue is only rebuilt when empty or theme drifts significantly
- This prevents repeated processing and ensures smooth transitions

### Drift Detection

If you manually change your pywal theme, the script detects this "drift" and recalculates the queue based on the new color.

## 🐛 Troubleshooting

### Timer not running

```bash
# Check if enabled
systemctl --user is-enabled wallpaper-rotation.timer

# Enable if needed
systemctl --user enable wallpaper-rotation.timer

# Start if stopped
systemctl --user start wallpaper-rotation.timer
```

### No wallpapers changing

```bash
# Check logs for errors
journalctl --user -u wallpaper-rotation.service -n 20

# Common issues:
# - "No wallpapers found within distance" → Increase MAX_MATCH_DIST
# - "Pywal not active" → Make sure pywal is running
# - "ImageMagick not found" → Install ImageMagick
```

### Script errors

```bash
# Test manually
~/.local/bin/wallpaper-rotation.py

# Check dependencies
which magick  # Should show ImageMagick path
cat ~/.cache/wal/colors  # Should show pywal colors
ls ~/Pictures/Wallpapers  # Should list wallpapers
```

### Reset everything

```bash
# Delete all cache files (forces fresh start)
rm ~/.cache/wallpaper_*.{json,txt}

# Trigger rotation
systemctl --user start wallpaper-rotation.service
```

## 📚 Documentation

- **QUICK_REFERENCE.md** - Command cheat sheet

## 🔧 Advanced Usage

### Custom wallpaper directory

Edit the script:

```python
WALLPAPER_DIR = Path.home() / "Pictures/Wallpapers"
# Change to your directory
```

### Custom theme command

The script uses `wall` by default, I just copied it into this repo if you want inspiration for your own setup or copying it. To change:

```python
THEME_COMMAND = "wall"  # Change to your command
# e.g., "feh --bg-scale", "nitrogen --set-auto", etc.
```

## 🗑️ Uninstallation

```bash
# Stop and disable
systemctl --user stop wallpaper-rotation.timer
systemctl --user disable wallpaper-rotation.timer

# Remove files
rm ~/.config/systemd/user/wallpaper-rotation.{service,timer}
rm ~/.local/bin/wallpaper-rotation.py

# Reload systemd
systemctl --user daemon-reload

# Optional: Remove cache
rm ~/.cache/wallpaper_*.{json,txt}
```

## 📝 Notes

- The script uses ImageMagick to extract dominant colors (1x1 resize method)
- Color distance is calculated in RGB space (Euclidean distance)
- Queue system prevents repeated processing of all wallpapers
- History tracking ensures variety until full cycle complete
- File locking prevents concurrent runs from corrupting state

## 🤝 Support

If you encounter issues:

1. Check logs: `journalctl --user -u wallpaper-rotation.service -n 50`
2. Test manually: `~/.local/bin/wallpaper-rotation.py`
3. Verify prerequisites are installed
4. Check file permissions
5. See troubleshooting section above
