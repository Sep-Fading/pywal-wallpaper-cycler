#!/usr/bin/env python3
import json
import subprocess
import math
import sys
import shutil
import logging
from pathlib import Path
from typing import Optional, Set, List, Tuple, Dict
import fcntl
import time

# --- CONFIGURATION ---
THEME_COMMAND = "wall"
WALLPAPER_DIR = Path.home() / "Pictures/Wallpapers"

# Files
CACHE_FILE = Path.home() / ".cache/wallpaper_colors.json"
CYCLE_LOG = Path.home() / ".cache/wallpaper_global_history.txt"
STATE_FILE = Path.home() / ".cache/wallpaper_queue_state.json"
PYWAL_COLORS = Path.home() / ".cache/wal/colors"
LOCK_FILE = Path.home() / ".cache/wallpaper_rotation.lock"

# --- TUNING FOR "IMPERCEPTIBLE" CHANGES ---
QUEUE_SIZE = 10         

# STRICTNESS: Lower = More subtle. 
# 45.0 is a good balance. 
# If you still notice it, lower to 35.0. 
# If it never changes wallpaper, raise to 60.0.
MAX_MATCH_DIST = 45.0   

# If the current theme drifts more than this (e.g. you picked a new color manually),
# we scrap the old queue and recalculate.
DRIFT_TOLERANCE = 25.0  
# ---------------------

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FileLock:
    """Context manager for file-based locking to prevent concurrent runs"""
    def __init__(self, lock_file: Path, timeout: int = 5):
        self.lock_file = lock_file
        self.timeout = timeout
        self.fd = None
    
    def __enter__(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.fd = open(self.lock_file, 'w')
        
        start_time = time.time()
        while True:
            try:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Could not acquire lock on {self.lock_file}")
                time.sleep(0.1)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd:
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            self.fd.close()
        return False


def get_files() -> List[Path]:
    """Get all image files from wallpaper directory"""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    try:
        return [f for f in WALLPAPER_DIR.rglob('*') 
                if f.suffix.lower() in extensions and f.is_file()]
    except (PermissionError, OSError) as e:
        logger.error(f"Error accessing wallpaper directory: {e}")
        return []


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple"""
    hex_str = hex_str.lstrip('#')
    return (
        int(hex_str[0:2], 16),
        int(hex_str[2:4], 16),
        int(hex_str[4:6], 16)
    )


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """Calculate Euclidean distance between two RGB colors"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def get_dominant_color(image_path: Path) -> Optional[str]:
    """
    Extract dominant color from image using ImageMagick.
    Returns hex color string without '#' prefix, or None on failure.
    """
    try:
        path_str = str(image_path)
        # For GIFs, only analyze first frame
        if image_path.suffix.lower() == '.gif':
            path_str += "[0]"
        
        # Resize to 1x1 to get average color
        cmd = ['magick', path_str, '-resize', '1x1', '-depth', '8', 'txt:-']
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=10
        )
        
        for line in result.stdout.splitlines():
            if '#' in line:
                return line.split('#')[1].split()[0]
        
        logger.warning(f"Could not parse color from: {image_path.name}")
        return None
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout processing image: {image_path.name}")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"ImageMagick error for {image_path.name}: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error("ImageMagick not found. Please install it.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing {image_path.name}: {e}")
        return None


def load_json(path: Path) -> dict:
    """Safely load JSON file with error handling"""
    if not path.exists():
        return {}
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        # Backup corrupted file
        backup = path.with_suffix('.json.backup')
        shutil.copy2(path, backup)
        logger.info(f"Backed up corrupted file to {backup}")
        return {}
    except (PermissionError, OSError) as e:
        logger.error(f"Error reading {path}: {e}")
        return {}


def save_json(path: Path, data: dict) -> bool:
    """Safely save JSON file with atomic write"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file first, then rename (atomic on Unix)
        temp_path = path.with_suffix('.json.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        temp_path.replace(path)
        return True
        
    except (PermissionError, OSError) as e:
        logger.error(f"Error writing {path}: {e}")
        return False


def load_history() -> Set[str]:
    """Load history of used wallpapers"""
    if not CYCLE_LOG.exists():
        return set()
    
    try:
        with open(CYCLE_LOG, 'r') as f:
            return set(line.strip() for line in f.readlines() if line.strip())
    except (PermissionError, OSError) as e:
        logger.error(f"Error reading history: {e}")
        return set()


def append_history(path: str) -> None:
    """Append wallpaper path to history"""
    try:
        CYCLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CYCLE_LOG, 'a') as f:
            f.write(f"{path}\n")
    except (PermissionError, OSError) as e:
        logger.error(f"Error writing history: {e}")


def reset_history() -> None:
    """Clear wallpaper history"""
    try:
        CYCLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CYCLE_LOG, 'w') as f:
            f.write("")
        logger.info("History reset")
    except (PermissionError, OSError) as e:
        logger.error(f"Error resetting history: {e}")


def validate_cache(cache: Dict[str, str], existing_files: List[Path]) -> Dict[str, str]:
    """
    Remove stale entries from cache and check for modified files.
    Returns cleaned cache.
    """
    existing_paths = {str(p): p for p in existing_files}
    cleaned_cache = {}
    removed_count = 0
    
    for path_str, color in cache.items():
        path_obj = Path(path_str)
        
        # Remove if file no longer exists
        if not path_obj.exists():
            removed_count += 1
            continue
        
        # Check if file was modified (simple check - could be enhanced with mtime)
        if path_str in existing_paths:
            cleaned_cache[path_str] = color
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} stale cache entries")
    
    return cleaned_cache


def main():
    # Use file lock to prevent concurrent runs
    try:
        with FileLock(LOCK_FILE):
            run_wallpaper_rotation()
    except TimeoutError as e:
        logger.error(f"Another instance is running: {e}")
        sys.exit(1)


def run_wallpaper_rotation():
    """Main wallpaper rotation logic"""
    
    # 1. Get Current Pywal Color
    if not PYWAL_COLORS.exists():
        logger.error("Pywal not active. Colors file not found.")
        sys.exit(1)
    
    try:
        with open(PYWAL_COLORS, 'r') as f:
            lines = f.readlines()
            if not lines:
                logger.error("Pywal colors file is empty")
                sys.exit(1)
            current_hex = lines[0].strip()
            current_rgb = hex_to_rgb(current_hex)
    except (PermissionError, OSError) as e:
        logger.error(f"Error reading pywal colors: {e}")
        sys.exit(1)
    except (ValueError, IndexError) as e:
        logger.error(f"Invalid color format in pywal file: {e}")
        sys.exit(1)

    # 2. Check Existing Queue State
    state = load_json(STATE_FILE)
    queue = state.get('queue', [])
    last_trigger_hex = state.get('trigger_color', "#000000")
    
    last_trigger_rgb = hex_to_rgb(last_trigger_hex)
    drift = color_distance(current_rgb, last_trigger_rgb)
    
    force_refresh = False
    
    # Determine if queue needs rebuilding
    if not queue:
        logger.info("Queue empty.")
        force_refresh = True
    elif drift > DRIFT_TOLERANCE:
        logger.info(f"Theme drift detected ({drift:.1f} > {DRIFT_TOLERANCE}). Recalculating.")
        force_refresh = True
    else:
        logger.info(f"Theme stable (Drift: {drift:.1f}).")

    # 3. Rebuild Queue if needed
    if force_refresh:
        cache = load_json(CACHE_FILE)
        files = get_files()
        
        if not files:
            logger.error(f"No wallpaper files found in {WALLPAPER_DIR}")
            sys.exit(1)
        
        # Validate and clean cache
        cache = validate_cache(cache, files)
        dirty = False
        
        # Update Cache with missing files
        valid_map = {}
        for path_obj in files:
            path_str = str(path_obj)
            
            if path_str not in cache:
                logger.info(f"Indexing: {path_obj.name}")
                color = get_dominant_color(path_obj)
                if color:
                    cache[path_str] = color
                    dirty = True
            
            if path_str in cache:
                valid_map[path_str] = cache[path_str]
        
        if dirty:
            if save_json(CACHE_FILE, cache):
                logger.info(f"Cache updated with {len(cache)} entries")

        # Filter by History
        used = load_history()
        candidates = [p for p in valid_map if p not in used]
        
        if not candidates:
            logger.info("Cycle complete. Resetting history.")
            reset_history()
            candidates = list(valid_map.keys())

        # Calculate Distances and Filter by Threshold
        scored: List[Tuple[float, str]] = []
        for path_str in candidates:
            w_rgb = hex_to_rgb(valid_map[path_str])
            distance = color_distance(current_rgb, w_rgb)
            
            # STRICT FILTER: Only include if within threshold
            if distance <= MAX_MATCH_DIST:
                scored.append((distance, path_str))
        
        # Sort by distance (closest first)
        scored.sort(key=lambda x: x[0])
        
        # Take top N matches
        queue = [x[1] for x in scored[:QUEUE_SIZE]]
        
        logger.info(f"Built queue with {len(queue)} wallpapers")
        
        # Save updated state
        state['trigger_color'] = current_hex
        state['queue'] = queue
        save_json(STATE_FILE, state)

    # 4. Safety Check & Execute
    if not queue:
        logger.warning(f"No wallpapers found within distance {MAX_MATCH_DIST}. Staying put.")
        return

    chosen = queue.pop(0)
    
    # Update State
    state['queue'] = queue
    save_json(STATE_FILE, state)
    
    # Update History
    append_history(chosen)
    
    # Apply Wallpaper
    logger.info(f"Applying: {Path(chosen).name}")
    cmd_path = shutil.which(THEME_COMMAND)
    
    if not cmd_path:
        logger.error(f"Command '{THEME_COMMAND}' not found in PATH")
        sys.exit(1)
    
    try:
        subprocess.run(
            [cmd_path, chosen],
            check=True,
            timeout=30,
            capture_output=True
        )
        logger.info("Wallpaper applied successfully")
    except subprocess.TimeoutExpired:
        logger.error("Timeout applying wallpaper")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error applying wallpaper: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error applying wallpaper: {e}")


if __name__ == "__main__":
    main()
