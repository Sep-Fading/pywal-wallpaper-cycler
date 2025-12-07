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
WALLPAPER_DIR = Path.home() / "Pictures/Wallpapers"

# Set to True to update theme colors (pywal) on rotation, False for wallpaper-only
UPDATE_THEME_COLORS = True

# Cache and state files
CACHE_FILE = Path.home() / ".cache/wallpaper_colors.json"
CYCLE_LOG = Path.home() / ".cache/wallpaper_global_history.txt"
STATE_FILE = Path.home() / ".cache/wallpaper_queue_state.json"
CURRENT_WALLPAPER_FILE = Path.home() / ".cache/current_wallpaper.txt"
LOCK_FILE = Path.home() / ".cache/wallpaper_rotation.lock"

# SWWW transition settings
SWWW_TRANSITION_TYPE = "grow"
SWWW_TRANSITION_POS = "0.5,0.5"
SWWW_TRANSITION_STEP = "45"
SWWW_TRANSITION_FPS = "240"
SWWW_TRANSITION_DURATION = "0.75"

# --- COLOR MATCHING TUNING ---
NUM_COLORS = 6             # Number of dominant colors to extract per image
QUEUE_SIZE = 10
MAX_MATCH_DIST = 35.0      # Lower values = stricter color matching
DRIFT_TOLERANCE = 35.0     # Threshold for queue rebuild (using average distance)  
# ---------------------

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FileLock:
    """File-based locking to prevent concurrent script execution"""
    
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


def palette_distance(palette1: List[str], palette2: List[str]) -> float:
    """
    Calculate minimum Euclidean distance between two color palettes.
    Returns the smallest distance between any color pair.
    """
    min_dist = float('inf')
    
    for color1_hex in palette1:
        rgb1 = hex_to_rgb(color1_hex)
        for color2_hex in palette2:
            rgb2 = hex_to_rgb(color2_hex)
            dist = color_distance(rgb1, rgb2)
            min_dist = min(min_dist, dist)
    
    return min_dist


def palette_average_distance(palette1: List[str], palette2: List[str]) -> float:
    """
    Calculate average minimum distance between two color palettes.
    For each color in palette1, finds its closest match in palette2,
    then returns the average of all these minimum distances.
    This gives a better overall similarity metric than single minimum.
    """
    if not palette1 or not palette2:
        return float('inf')
    
    total_dist = 0.0
    
    for color1_hex in palette1:
        rgb1 = hex_to_rgb(color1_hex)
        min_dist_for_this_color = float('inf')
        
        for color2_hex in palette2:
            rgb2 = hex_to_rgb(color2_hex)
            dist = color_distance(rgb1, rgb2)
            min_dist_for_this_color = min(min_dist_for_this_color, dist)
        
        total_dist += min_dist_for_this_color
    
    return total_dist / len(palette1)


def get_dominant_colors(image_path: Path, num_colors: Optional[int] = None) -> Optional[List[str]]:
    """
    Extract top N dominant colors from image using ImageMagick color quantization.
    Returns list of hex color strings (without '#'), or None on failure.
    """
    if num_colors is None:
        num_colors = NUM_COLORS
    
    try:
        path_str = str(image_path)
        if image_path.suffix.lower() == '.gif':
            path_str += "[0]"
        
        cmd = [
            'magick', path_str,
            '-resize', '400x400>',
            '-colors', str(num_colors),
            '-depth', '8',
            '-format', '%c',
            'histogram:info:-'
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=15
        )
        
        colors = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '#' in line and ':' in line:
                hex_part = line.split('#')[1].split()[0]
                if len(hex_part) == 6 and all(c in '0123456789ABCDEFabcdef' for c in hex_part):
                    colors.append(hex_part.upper())
                    if len(colors) >= num_colors:
                        break
        
        if colors:
            return colors
        
        logger.warning(f"Could not parse colors from: {image_path.name}")
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
    """Load JSON file with error handling"""
    if not path.exists():
        return {}
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        backup = path.with_suffix('.json.backup')
        shutil.copy2(path, backup)
        logger.info(f"Backed up corrupted file to {backup}")
        return {}
    except (PermissionError, OSError) as e:
        logger.error(f"Error reading {path}: {e}")
        return {}


def save_json(path: Path, data: dict) -> bool:
    """Save JSON file with atomic write"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
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
    """Remove stale cache entries for deleted files"""
    existing_paths = {str(p): p for p in existing_files}
    cleaned_cache = {}
    removed_count = 0
    
    for path_str, color in cache.items():
        path_obj = Path(path_str)
        
        if not path_obj.exists():
            removed_count += 1
            continue
        
        if path_str in existing_paths:
            cleaned_cache[path_str] = color
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} stale cache entries")
    
    return cleaned_cache


def main():
    """Entry point with file locking"""
    try:
        with FileLock(LOCK_FILE):
            run_wallpaper_rotation()
    except TimeoutError as e:
        logger.error(f"Another instance is running: {e}")
        sys.exit(1)


def run_wallpaper_rotation():
    """Main wallpaper rotation logic"""
    
    # 1. Load or initialize current wallpaper
    if not CURRENT_WALLPAPER_FILE.exists():
        logger.warning("No current wallpaper file found. Initializing with random wallpaper.")
        files = get_files()
        if not files:
            logger.error(f"No wallpaper files found in {WALLPAPER_DIR}")
            sys.exit(1)
        current_wallpaper = str(files[0])
        CURRENT_WALLPAPER_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CURRENT_WALLPAPER_FILE, 'w') as f:
            f.write(current_wallpaper)
    else:
        try:
            with open(CURRENT_WALLPAPER_FILE, 'r') as f:
                current_wallpaper = f.read().strip()
            
            if not current_wallpaper or not Path(current_wallpaper).exists():
                logger.warning(f"Current wallpaper file invalid or missing: {current_wallpaper}")
                files = get_files()
                if files:
                    current_wallpaper = str(files[0])
                    with open(CURRENT_WALLPAPER_FILE, 'w') as f:
                        f.write(current_wallpaper)
                else:
                    logger.error(f"No wallpaper files found in {WALLPAPER_DIR}")
                    sys.exit(1)
        except (PermissionError, OSError) as e:
            logger.error(f"Error reading current wallpaper file: {e}")
            sys.exit(1)
    
    # Extract dominant color palette
    logger.info(f"Current wallpaper: {Path(current_wallpaper).name}")
    current_colors = get_dominant_colors(Path(current_wallpaper))
    if not current_colors:
        logger.error(f"Could not extract colors from current wallpaper: {current_wallpaper}")
        sys.exit(1)
    
    logger.info(f"Current palette: {', '.join(['#' + c for c in current_colors])}")

    # 2. Check queue state and drift
    state = load_json(STATE_FILE)
    queue = state.get('queue', [])
    last_trigger_colors = state.get('trigger_colors', [])
    
    drift = 0.0
    if last_trigger_colors:
        drift = palette_average_distance(current_colors, last_trigger_colors)
    
    force_refresh = False
    
    if not queue:
        logger.info("Queue empty.")
        force_refresh = True
    elif drift > DRIFT_TOLERANCE:
        logger.info(f"Theme drift detected ({drift:.1f} > {DRIFT_TOLERANCE}). Recalculating.")
        force_refresh = True
    else:
        logger.info(f"Theme stable (Drift: {drift:.1f}).")

    # 3. Rebuild queue if needed
    if force_refresh:
        cache = load_json(CACHE_FILE)
        files = get_files()
        
        if not files:
            logger.error(f"No wallpaper files found in {WALLPAPER_DIR}")
            sys.exit(1)
        
        cache = validate_cache(cache, files)
        dirty = False
        
        # Index new or outdated files
        valid_map = {}
        for path_obj in files:
            path_str = str(path_obj)
            
            if path_str not in cache or not isinstance(cache[path_str], list):
                logger.info(f"Indexing: {path_obj.name}")
                colors = get_dominant_colors(path_obj)
                if colors:
                    cache[path_str] = colors
                    dirty = True
            
            if path_str in cache and isinstance(cache[path_str], list):
                valid_map[path_str] = cache[path_str]
        
        if dirty:
            if save_json(CACHE_FILE, cache):
                logger.info(f"Cache updated with {len(cache)} entries")

        # Filter by history
        used = load_history()
        candidates = [p for p in valid_map if p not in used and p != current_wallpaper]
        
        if not candidates:
            logger.info("Cycle complete. Resetting history.")
            reset_history()
            candidates = [p for p in valid_map.keys() if p != current_wallpaper]

        # Calculate palette distances and filter
        scored: List[Tuple[float, str]] = []
        for path_str in candidates:
            wallpaper_colors = valid_map[path_str]
            distance = palette_distance(current_colors, wallpaper_colors)
            
            if distance <= MAX_MATCH_DIST:
                scored.append((distance, path_str))
        
        scored.sort(key=lambda x: x[0])
        queue = [x[1] for x in scored[:QUEUE_SIZE]]
        
        logger.info(f"Found {len(scored)} wallpapers within distance {MAX_MATCH_DIST}")
        logger.info(f"Built queue with {len(queue)} wallpapers")
        
        if queue:
            logger.info(f"Queue preview (closest 3):")
            for i, (dist, path) in enumerate(scored[:3]):
                wall_colors = valid_map[path]
                logger.info(f"  {i+1}. {Path(path).name} - {', '.join(['#' + c for c in wall_colors])} (distance: {dist:.1f})")
        
        state['trigger_colors'] = current_colors
        state['queue'] = queue
        save_json(STATE_FILE, state)

    # 4. Apply next wallpaper
    if not queue:
        logger.warning(f"No wallpapers found within distance {MAX_MATCH_DIST}. Staying put.")
        return

    chosen = queue.pop(0)
    
    state['queue'] = queue
    save_json(STATE_FILE, state)
    append_history(chosen)
    
    logger.info(f"Applying: {Path(chosen).name}")
    
    swww_path = shutil.which('swww')
    if not swww_path:
        logger.error("swww command not found in PATH")
        sys.exit(1)
    
    try:
        subprocess.run(
            [
                swww_path, 'img', chosen,
                '--transition-type', SWWW_TRANSITION_TYPE,
                '--transition-pos', SWWW_TRANSITION_POS,
                '--transition-step', SWWW_TRANSITION_STEP,
                '--transition-fps', SWWW_TRANSITION_FPS,
                '--transition-duration', SWWW_TRANSITION_DURATION
            ],
            check=True,
            timeout=30,
            capture_output=True
        )
        
        with open(CURRENT_WALLPAPER_FILE, 'w') as f:
            f.write(chosen)
        
        logger.info("Wallpaper applied successfully")
        
        # Update theme colors if enabled
        if UPDATE_THEME_COLORS:
            wal_path = shutil.which('wal')
            if wal_path:
                try:
                    logger.info("Updating theme colors with pywal...")
                    subprocess.run(
                        [wal_path, '-i', chosen, '-n', '-q'],
                        check=True,
                        timeout=10,
                        capture_output=True
                    )
                    logger.info("Theme colors updated")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to update theme colors: {e.stderr}")
                except Exception as e:
                    logger.warning(f"Error updating theme: {e}")
            else:
                logger.warning("pywal not found, cannot update theme colors")
        
    except subprocess.TimeoutExpired:
        logger.error("Timeout applying wallpaper")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error applying wallpaper: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error applying wallpaper: {e}")


if __name__ == "__main__":
    main()
