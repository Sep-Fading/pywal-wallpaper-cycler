#!/usr/bin/env python3
import json
import subprocess
import math
import sys
import shutil
import logging
import re
from pathlib import Path
from typing import Optional, Set, List, Tuple, Dict, Any
import fcntl
import time

# --- CONFIGURATION ---
WALLPAPER_DIR = Path.home() / "Pictures/Wallpapers"

# Set to True to update theme colors (pywal) on rotation, False for wallpaper-only
UPDATE_THEME_COLORS = False

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
NUM_COLORS = 6  # Number of dominant colors to extract per image
QUEUE_SIZE = 10  # Maximum number of matching wallpapers in queue

# Queue behavior
CIRCULAR_QUEUE = True  # If True, loop through queue infinitely until externally changed

# Using LAB color space (perceptually uniform) instead of RGB
# LAB Delta-E reference: <2 = imperceptible, <5 = close, <10 = similar, >10 = different
MAX_MATCH_DIST = 15.0  # Stricter matching threshold - prevents poor color shifts

# Weighting for different color properties when matching
WEIGHT_HUE = 2.0  # Strongly prioritize hue similarity - prevents green→red shifts
WEIGHT_SATURATION = 1.2  # Higher weight to prevent vibrancy jumps (muted vs vibrant)
WEIGHT_LIGHTNESS = 0.8  # Moderate weight on lightness

# Mood matching - prefer wallpapers with similar overall feel
ENABLE_MOOD_MATCHING = True
MOOD_WEIGHT = (
    0.25  # Higher mood weight to ensure similar "feel" (warmth/vibrance/brightness)
)
# ---------------------

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",  # Simplified - journalctl adds timestamp
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
        self.fd = open(self.lock_file, "w")

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
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    try:
        return [
            f
            for f in WALLPAPER_DIR.rglob("*")
            if f.suffix.lower() in extensions and f.is_file()
        ]
    except (PermissionError, OSError) as e:
        logger.error(f"Error accessing wallpaper directory: {e}")
        return []


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple"""
    hex_str = hex_str.lstrip("#")
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert RGB to LAB color space for perceptually uniform color comparison.
    LAB is designed to approximate human vision - equal distances in LAB space
    correspond to equal perceived color differences.

    L* = Lightness (0-100)
    a* = Green-Red axis (-128 to 127)
    b* = Blue-Yellow axis (-128 to 127)
    """
    # Normalize RGB to 0-1 range
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    # Apply gamma correction (sRGB to linear RGB)
    def gamma_correct(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92
        else:
            return ((channel + 0.055) / 1.055) ** 2.4

    r_linear = gamma_correct(r_norm)
    g_linear = gamma_correct(g_norm)
    b_linear = gamma_correct(b_norm)

    # Convert to XYZ color space (D65 illuminant)
    x = r_linear * 0.4124564 + g_linear * 0.3575761 + b_linear * 0.1804375
    y = r_linear * 0.2126729 + g_linear * 0.7151522 + b_linear * 0.0721750
    z = r_linear * 0.0193339 + g_linear * 0.1191920 + b_linear * 0.9503041

    # Normalize by D65 white point
    x_norm = x / 0.95047
    y_norm = y / 1.00000
    z_norm = z / 1.08883

    # Convert XYZ to LAB
    def f(t: float) -> float:
        delta = 6.0 / 29.0
        if t > delta**3:
            return t ** (1.0 / 3.0)
        else:
            return t / (3.0 * delta**2) + 4.0 / 29.0

    fx = f(x_norm)
    fy = f(y_norm)
    fz = f(z_norm)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)

    return (L, a, b_lab)


def lab_distance(
    lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]
) -> float:
    """
    Calculate Delta-E (CIE76) distance between two LAB colors.
    This is the Euclidean distance in LAB space and approximates
    perceived color difference much better than RGB distance.

    Delta-E interpretation:
    < 1.0 = Not perceptible by human eyes
    1-2   = Perceptible through close observation
    2-10  = Perceptible at a glance
    10+   = Colors are more different than similar
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def weighted_lab_distance(
    lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]
) -> float:
    """
    Calculate weighted LAB distance with adjustable emphasis on different properties.
    Allows fine-tuning of what makes colors "match" - e.g., prioritizing hue over brightness.

    In LAB space:
    - L = Lightness
    - a = green-red axis (also relates to chroma/saturation)
    - b = blue-yellow axis (also relates to chroma/saturation)
    """
    L1, a1, b1_lab = lab1
    L2, a2, b2_lab = lab2

    # Calculate chroma (saturation) for each color in LAB space
    # Chroma = sqrt(a^2 + b^2) - distance from neutral axis
    chroma1 = math.sqrt(a1**2 + b1_lab**2)
    chroma2 = math.sqrt(a2**2 + b2_lab**2)

    # Calculate distance for each component
    delta_L = (L1 - L2) ** 2 * WEIGHT_LIGHTNESS
    delta_a = (a1 - a2) ** 2 * WEIGHT_HUE
    delta_b = (b1_lab - b2_lab) ** 2 * WEIGHT_HUE

    # Add saturation (chroma) difference with its own weight
    delta_chroma = (chroma1 - chroma2) ** 2 * WEIGHT_SATURATION

    return math.sqrt(delta_L + delta_a + delta_b + delta_chroma)


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert RGB to HSV (Hue, Saturation, Value) for mood analysis.
    HSV better represents color "feel" - warm vs cool, vibrant vs muted.
    """
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    max_c = max(r_norm, g_norm, b_norm)
    min_c = min(r_norm, g_norm, b_norm)
    delta = max_c - min_c

    # Calculate hue
    if delta == 0:
        h = 0.0
    elif max_c == r_norm:
        h = 60.0 * (((g_norm - b_norm) / delta) % 6)
    elif max_c == g_norm:
        h = 60.0 * (((b_norm - r_norm) / delta) + 2)
    else:
        h = 60.0 * (((r_norm - g_norm) / delta) + 4)

    # Calculate saturation
    s = 0.0 if max_c == 0 else delta / max_c

    # Value is just max
    v = max_c

    return (h, s, v)


def calculate_mood(palette: List[str]) -> Dict[str, float]:
    """
    Calculate overall mood/feel of a color palette.
    Returns metrics for:
    - warmth: cool (blues/greens) vs warm (reds/oranges/yellows)
    - vibrance: muted vs saturated
    - brightness: dark vs light

    This helps match wallpapers that "feel" similar even if exact colors differ.
    """
    if not palette:
        return {"warmth": 0.5, "vibrance": 0.5, "brightness": 0.5}

    warmth_sum = 0.0
    saturation_sum = 0.0
    value_sum = 0.0

    for hex_color in palette:
        r, g, b = hex_to_rgb(hex_color)
        h, s, v = rgb_to_hsv(r, g, b)

        # Warmth: hue-based (red/orange/yellow = warm, blue/green = cool)
        # Hue circle: 0=red, 60=yellow, 120=green, 180=cyan, 240=blue, 300=magenta
        if h < 60 or h > 300:  # Reds and magentas
            warmth = 1.0
        elif h < 180:  # Yellows and greens
            warmth = 0.7 if h < 120 else 0.3
        else:  # Cyans and blues
            warmth = 0.0

        warmth_sum += warmth
        saturation_sum += s
        value_sum += v

    count = len(palette)
    return {
        "warmth": warmth_sum / count,
        "vibrance": saturation_sum / count,
        "brightness": value_sum / count,
    }


def mood_distance(mood1: Dict[str, float], mood2: Dict[str, float]) -> float:
    """
    Calculate how different two color moods are.
    Returns normalized distance (0 = identical mood, 1 = completely different)

    Vibrancy differences are weighted more heavily to prevent jarring
    transitions between muted and saturated wallpapers.
    """
    warmth_diff = abs(mood1["warmth"] - mood2["warmth"])
    vibrance_diff = abs(mood1["vibrance"] - mood2["vibrance"])
    brightness_diff = abs(mood1["brightness"] - mood2["brightness"])

    # Weight vibrancy more heavily (1.5x) to reduce saturation jumps
    weighted_sum = warmth_diff + (vibrance_diff * 1.5) + brightness_diff
    weighted_avg = weighted_sum / 3.5  # Normalize (1 + 1.5 + 1 = 3.5)

    return weighted_avg


def palette_weighted_distance(
    palette1: List[str],
    palette2: List[str],
    weights1: Optional[List[float]] = None,
    weights2: Optional[List[float]] = None,
) -> float:
    """
    Calculate weighted bidirectional distance between two color palettes using LAB.
    Each color can have a weight representing its prominence in the image.

    Uses bidirectional matching: checks both palette1→palette2 AND palette2→palette1.
    This prevents cases where black+green matches black+red just because they share black.
    Both palettes must have good matching colors.

    This ensures all dominant colors matter, not just finding some overlap.
    """
    if not palette1 or not palette2:
        return float("inf")

    # If no weights provided, assume equal prominence
    if weights1 is None:
        weights1 = [1.0] * len(palette1)
    if weights2 is None:
        weights2 = [1.0] * len(palette2)

    # Convert all colors to LAB once
    lab1 = [rgb_to_lab(*hex_to_rgb(c)) for c in palette1]
    lab2 = [rgb_to_lab(*hex_to_rgb(c)) for c in palette2]

    # Calculate distance from palette1 to palette2
    total_weighted_dist_1to2 = 0.0
    total_weight_1 = 0.0

    for i, color1_lab in enumerate(lab1):
        min_dist = float("inf")

        # Find closest matching color in palette2
        for color2_lab in lab2:
            dist = weighted_lab_distance(color1_lab, color2_lab)
            min_dist = min(min_dist, dist)

        # Weight this distance by the prominence of color1
        weight = weights1[i]
        total_weighted_dist_1to2 += min_dist * weight
        total_weight_1 += weight

    dist_1to2 = (
        total_weighted_dist_1to2 / total_weight_1
        if total_weight_1 > 0
        else float("inf")
    )

    # Calculate distance from palette2 to palette1 (reverse direction)
    total_weighted_dist_2to1 = 0.0
    total_weight_2 = 0.0

    for i, color2_lab in enumerate(lab2):
        min_dist = float("inf")

        # Find closest matching color in palette1
        for color1_lab in lab1:
            dist = weighted_lab_distance(color2_lab, color1_lab)
            min_dist = min(min_dist, dist)

        # Weight this distance by the prominence of color2
        weight = weights2[i]
        total_weighted_dist_2to1 += min_dist * weight
        total_weight_2 += weight

    dist_2to1 = (
        total_weighted_dist_2to1 / total_weight_2
        if total_weight_2 > 0
        else float("inf")
    )

    # Return the maximum of the two directions
    # This ensures BOTH palettes must match well
    # If black+green tries to match black+red:
    # - Direction 1: green finds black (bad match, high distance)
    # - Direction 2: red finds black (bad match, high distance)
    # - Maximum will be high, preventing the match
    return max(dist_1to2, dist_2to1)


def palette_distance_with_mood(
    palette1: List[str],
    palette2: List[str],
    weights1: Optional[List[float]] = None,
    weights2: Optional[List[float]] = None,
) -> float:
    """
    Combined distance metric incorporating both color matching and mood matching.
    This provides the most perceptually accurate similarity measure.

    Returns weighted combination of:
    - Color distance (LAB-based, weighted by prominence)
    - Mood distance (warmth, vibrance, brightness)
    """
    # Calculate color distance
    color_dist = palette_weighted_distance(palette1, palette2, weights1, weights2)

    # If mood matching disabled, return pure color distance
    if not ENABLE_MOOD_MATCHING:
        return color_dist

    # Calculate mood distance
    mood1 = calculate_mood(palette1)
    mood2 = calculate_mood(palette2)
    mood_dist = mood_distance(mood1, mood2)

    # Normalize mood distance to same scale as color distance
    # Mood distance is 0-1, scale it to approximate LAB scale
    mood_dist_scaled = mood_dist * 100

    # Combine with weighting
    combined = (color_dist * (1 - MOOD_WEIGHT)) + (mood_dist_scaled * MOOD_WEIGHT)

    return combined


def get_dominant_colors(
    image_path: Path, num_colors: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Extract top N dominant colors from image using ImageMagick color quantization.

    Returns dict with:
    - 'colors': list of hex color strings (without '#')
    - 'weights': list of weights (0-1) representing each color's prominence
    - 'mood': dict with warmth/vibrance/brightness metrics

    Returns None on failure.
    """
    if num_colors is None:
        num_colors = NUM_COLORS

    try:
        path_str = str(image_path)
        if image_path.suffix.lower() == ".gif":
            path_str += "[0]"

        # Get color histogram with pixel counts
        cmd = [
            "magick",
            path_str,
            "-resize",
            "400x400>",
            "-colors",
            str(num_colors),
            "-depth",
            "8",
            "-format",
            "%c",
            "histogram:info:-",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=15
        )

        colors = []
        counts = []

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse format: "count: (r,g,b) #HEXCODE ..."
            if "#" in line and ":" in line:
                # Extract count (number before colon)
                try:
                    count_str = line.split(":")[0].strip()
                    count = int(count_str)
                except (ValueError, IndexError):
                    count = 1

                # Extract hex color
                hex_part = line.split("#")[1].split()[0]
                if len(hex_part) == 6 and all(
                    c in "0123456789ABCDEFabcdef" for c in hex_part
                ):
                    colors.append(hex_part.upper())
                    counts.append(count)
                    if len(colors) >= num_colors:
                        break

        if colors:
            # Normalize counts to weights (0-1)
            total_count = sum(counts)
            weights = (
                [c / total_count for c in counts]
                if total_count > 0
                else [1.0 / len(colors)] * len(colors)
            )

            # Calculate mood
            mood = calculate_mood(colors)

            return {"colors": colors, "weights": weights, "mood": mood}

        # If histogram failed, try simpler method as fallback
        return get_dominant_colors_fallback(image_path, num_colors)

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout processing image: {image_path.name}")
        return None
    except subprocess.CalledProcessError as e:
        return get_dominant_colors_fallback(image_path, num_colors)
    except FileNotFoundError:
        logger.error("ImageMagick not found. Please install it.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing {image_path.name}: {e}")
        return None


def get_dominant_colors_fallback(
    image_path: Path, num_colors: int
) -> Optional[Dict[str, Any]]:
    """
    Fallback method using multiple strategies to extract colors from difficult images.
    Tries progressively simpler approaches until one works.
    """

    # Strategy 1: Sample grid of pixels
    try:
        path_str = str(image_path)
        if image_path.suffix.lower() == ".gif":
            path_str += "[0]"

        grid_size = min(num_colors, 3)
        cmd = [
            "magick",
            path_str,
            "-resize",
            f"{grid_size}x{grid_size}!",
            "-depth",
            "8",
            "txt:-",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=10
        )

        colors = []
        seen = set()

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("ImageMagick"):
                continue

            # Look for hex color in various formats
            # Format: "0,0: (r,g,b) #HEXCODE ..."
            if "#" in line:
                parts = line.split("#")
                if len(parts) > 1:
                    hex_part = parts[1].split()[0] if parts[1].split() else parts[1][:6]
                    hex_part = hex_part[:6]  # Take only first 6 chars

                    if len(hex_part) == 6 and all(
                        c in "0123456789ABCDEFabcdef" for c in hex_part
                    ):
                        hex_upper = hex_part.upper()
                        if hex_upper not in seen:
                            colors.append(hex_upper)
                            seen.add(hex_upper)
                            if len(colors) >= num_colors:
                                break

        if colors:
            logger.info(
                f"Fallback (grid): {len(colors)} colors from {image_path.name[:40]}"
            )
            weights = [1.0 / len(colors)] * len(colors)
            mood = calculate_mood(colors)
            return {"colors": colors, "weights": weights, "mood": mood}

    except Exception as e:
        logger.debug(f"Grid fallback failed for {image_path.name}: {e}")

    # Strategy 2: Simple average color
    try:
        path_str = str(image_path)
        if image_path.suffix.lower() == ".gif":
            path_str += "[0]"

        cmd = ["magick", path_str, "-resize", "1x1", "-format", "%[pixel:u]", "info:-"]

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=5
        )

        # Parse output like "srgb(R,G,B)" or "#HEXCODE"
        output = result.stdout.strip()
        colors = []

        # Try to extract hex or RGB
        if "#" in output:
            hex_match = output.split("#")[1][:6]
            if len(hex_match) == 6 and all(
                c in "0123456789ABCDEFabcdef" for c in hex_match
            ):
                colors = [hex_match.upper()]
        elif "srgb(" in output or "rgb(" in output:
            # Parse rgb(R,G,B) format
            rgb_match = re.search(r"rgb\((\d+),(\d+),(\d+)\)", output)
            if rgb_match:
                r, g, b = map(int, rgb_match.groups())
                hex_color = f"{r:02X}{g:02X}{b:02X}"
                colors = [hex_color]

        if colors:
            logger.info(f"Fallback (avg): 1 color from {image_path.name[:40]}")
            # Create variations by adjusting lightness slightly
            # This gives us multiple colors for better matching
            expanded_colors = [colors[0]]

            # Add slight variations if we need more colors
            if num_colors > 1 and len(colors) == 1:
                r, g, b = hex_to_rgb(colors[0])

                # Slightly darker version
                r2 = max(0, int(r * 0.8))
                g2 = max(0, int(g * 0.8))
                b2 = max(0, int(b * 0.8))
                expanded_colors.append(f"{r2:02X}{g2:02X}{b2:02X}")

                # Slightly lighter version
                r3 = min(255, int(r * 1.2))
                g3 = min(255, int(g * 1.2))
                b3 = min(255, int(b * 1.2))
                expanded_colors.append(f"{r3:02X}{g3:02X}{b3:02X}")

            weights = [1.0 / len(expanded_colors)] * len(expanded_colors)
            mood = calculate_mood(expanded_colors)
            return {
                "colors": expanded_colors[:num_colors],
                "weights": weights[:num_colors],
                "mood": mood,
            }

    except Exception as e:
        logger.debug(f"Average fallback failed for {image_path.name}: {e}")

    # Strategy 3: Last resort - grayscale sampling
    try:
        path_str = str(image_path)
        if image_path.suffix.lower() == ".gif":
            path_str += "[0]"

        # Convert to grayscale and get average
        cmd = [
            "magick",
            path_str,
            "-colorspace",
            "Gray",
            "-resize",
            "1x1",
            "-format",
            "%[fx:int(u*255)]",
            "info:-",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=5
        )

        gray_value = int(result.stdout.strip())
        gray_hex = f"{gray_value:02X}{gray_value:02X}{gray_value:02X}"

        # Create gradient of grays
        colors = []
        for i in range(min(num_colors, 3)):
            offset = (i - 1) * 30
            val = max(0, min(255, gray_value + offset))
            colors.append(f"{val:02X}{val:02X}{val:02X}")

        if colors:
            logger.info(
                f"Fallback (gray): {len(colors)} colors from {image_path.name[:40]}"
            )
            weights = [1.0 / len(colors)] * len(colors)
            mood = calculate_mood(colors)
            return {"colors": colors, "weights": weights, "mood": mood}

    except Exception as e:
        logger.debug(f"Grayscale fallback failed for {image_path.name}: {e}")

    logger.warning(f"All fallbacks failed: {image_path.name[:40]}")
    return None


def load_json(path: Path) -> dict:
    """Load JSON file with error handling"""
    if not path.exists():
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        backup = path.with_suffix(".json.backup")
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
        temp_path = path.with_suffix(".json.tmp")

        with open(temp_path, "w") as f:
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
        with open(CYCLE_LOG, "r") as f:
            return set(line.strip() for line in f.readlines() if line.strip())
    except (PermissionError, OSError) as e:
        logger.error(f"Error reading history: {e}")
        return set()


def append_history(path: str) -> None:
    """Append wallpaper path to history"""
    try:
        CYCLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CYCLE_LOG, "a") as f:
            f.write(f"{path}\n")
    except (PermissionError, OSError) as e:
        logger.error(f"Error writing history: {e}")


def reset_history() -> None:
    """Clear wallpaper history"""
    try:
        CYCLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CYCLE_LOG, "w") as f:
            f.write("")
        logger.info("History reset")
    except (PermissionError, OSError) as e:
        logger.error(f"Error resetting history: {e}")


def validate_cache(
    cache: Dict[str, dict], existing_files: List[Path]
) -> Dict[str, dict]:
    """
    Remove stale cache entries for deleted files.
    Also migrates old cache format (simple color lists) to new format (with weights and mood).
    """
    existing_paths = {str(p): p for p in existing_files}
    cleaned_cache = {}
    removed_count = 0
    migrated_count = 0

    for path_str, entry in cache.items():
        path_obj = Path(path_str)

        if not path_obj.exists():
            removed_count += 1
            continue

        if path_str in existing_paths:
            # Check if entry needs migration (old format: just list of colors)
            if isinstance(entry, list):
                # Migrate to new format
                mood = calculate_mood(entry)
                weights = [1.0 / len(entry)] * len(entry)
                cleaned_cache[path_str] = {
                    "colors": entry,
                    "weights": weights,
                    "mood": mood,
                }
                migrated_count += 1
            elif isinstance(entry, dict) and "colors" in entry:
                # Already in new format
                cleaned_cache[path_str] = entry
            else:
                # Invalid entry, skip
                removed_count += 1

    if removed_count > 0:
        logger.info(f"Removed {removed_count} stale cache entries")
    if migrated_count > 0:
        logger.info(f"Migrated {migrated_count} cache entries to new format")

    return cleaned_cache


def format_mood(mood: Dict[str, float]) -> str:
    """Format mood metrics for human-readable logging"""
    warmth = mood["warmth"]
    vibrance = mood["vibrance"]
    brightness = mood["brightness"]

    # Warmth descriptor
    if warmth > 0.7:
        warmth_str = "warm"
    elif warmth < 0.3:
        warmth_str = "cool"
    else:
        warmth_str = "neutral"

    # Vibrance descriptor
    if vibrance > 0.6:
        vibrance_str = "vibrant"
    elif vibrance < 0.3:
        vibrance_str = "muted"
    else:
        vibrance_str = "moderate"

    # Brightness descriptor
    if brightness > 0.7:
        brightness_str = "light"
    elif brightness < 0.3:
        brightness_str = "dark"
    else:
        brightness_str = "medium"

    return f"{warmth_str}, {vibrance_str}, {brightness_str}"


def main():
    """Entry point with file locking"""
    try:
        with FileLock(LOCK_FILE):
            run_wallpaper_rotation()
    except TimeoutError as e:
        logger.error(f"Another instance is running: {e}")
        sys.exit(1)


def run_wallpaper_rotation():
    """Main wallpaper rotation logic with improved perceptual color matching"""

    # 1. Load or initialize current wallpaper
    if not CURRENT_WALLPAPER_FILE.exists():
        logger.warning(
            "No current wallpaper file found. Initializing with random wallpaper."
        )
        files = get_files()
        if not files:
            logger.error(f"No wallpaper files found in {WALLPAPER_DIR}")
            sys.exit(1)
        current_wallpaper = str(files[0])
        CURRENT_WALLPAPER_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CURRENT_WALLPAPER_FILE, "w") as f:
            f.write(current_wallpaper)
    else:
        try:
            with open(CURRENT_WALLPAPER_FILE, "r") as f:
                current_wallpaper = f.read().strip()

            if not current_wallpaper or not Path(current_wallpaper).exists():
                logger.warning(
                    f"Current wallpaper file invalid or missing: {current_wallpaper}"
                )
                files = get_files()
                if files:
                    current_wallpaper = str(files[0])
                    with open(CURRENT_WALLPAPER_FILE, "w") as f:
                        f.write(current_wallpaper)
                else:
                    logger.error(f"No wallpaper files found in {WALLPAPER_DIR}")
                    sys.exit(1)
        except (PermissionError, OSError) as e:
            logger.error(f"Error reading current wallpaper file: {e}")
            sys.exit(1)

    # 2. Extract dominant color palette with weights and mood
    logger.info(f"Current wallpaper: {Path(current_wallpaper).name}")
    current_data = get_dominant_colors(Path(current_wallpaper))
    if not current_data:
        logger.error(
            f"Could not extract colors from current wallpaper: {current_wallpaper}"
        )
        sys.exit(1)

    current_colors = current_data["colors"]
    current_weights = current_data["weights"]
    current_mood = current_data["mood"]

    logger.info(f"Current palette: {', '.join(['#' + c for c in current_colors])}")
    logger.info(f"Current mood: {format_mood(current_mood)}")

    # 3. Load state and check if rebuild needed
    state = load_json(STATE_FILE)
    queue = state.get("queue", [])
    original_queue = state.get("original_queue", [])

    force_refresh = False

    # Rebuild queue if:
    # - No queue exists yet (first run)
    # - Current wallpaper not in original queue (changed externally)
    if not original_queue:
        logger.info("No queue found - building initial queue.")
        force_refresh = True
    elif current_wallpaper not in original_queue:
        logger.info("Wallpaper changed externally - rebuilding queue.")
        force_refresh = True
    elif not queue:
        # Queue exhausted, loop back to start
        logger.info("Queue exhausted. Looping back to start of circular queue.")
        queue = original_queue.copy()
        state["queue"] = queue
        save_json(STATE_FILE, state)

    # 4. Build queue if needed
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

            # Check if entry is missing or in old format (needs re-indexing)
            needs_indexing = (
                path_str not in cache
                or not isinstance(cache[path_str], dict)
                or "colors" not in cache[path_str]
                or "weights" not in cache[path_str]
                or "mood" not in cache[path_str]
            )

            if needs_indexing:
                logger.info(f"Indexing: {path_obj.name[:50]}")
                data = get_dominant_colors(path_obj)
                if data:
                    cache[path_str] = data
                    dirty = True
                else:
                    logger.warning(f"Skip (no colors): {path_obj.name[:40]}")

            if (
                path_str in cache
                and isinstance(cache[path_str], dict)
                and "colors" in cache[path_str]
            ):
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

        # Calculate palette distances with mood and filter by MAX_MATCH_DIST
        scored: List[Tuple[float, str]] = []
        for path_str in candidates:
            wallpaper_data = valid_map[path_str]
            wallpaper_colors = wallpaper_data["colors"]
            wallpaper_weights = wallpaper_data.get("weights")

            distance = palette_distance_with_mood(
                current_colors, wallpaper_colors, current_weights, wallpaper_weights
            )

            if distance <= MAX_MATCH_DIST:
                scored.append((distance, path_str))

        scored.sort(key=lambda x: x[0])
        queue = [x[1] for x in scored[:QUEUE_SIZE]]

        logger.info(f"Found {len(scored)} wallpapers within distance {MAX_MATCH_DIST}")
        logger.info(f"Built queue with {len(queue)} wallpapers")

        if queue:
            logger.info(f"Queue preview (closest 3):")
            for i, (dist, path) in enumerate(scored[:3]):
                wall_data = valid_map[path]
                wall_colors = wall_data["colors"]
                wall_mood = wall_data["mood"]
                logger.info(f"  {i+1}. {Path(path).name}")
                logger.info(
                    f"      Colors: {', '.join(['#' + c for c in wall_colors])}"
                )
                logger.info(f"      Mood: {format_mood(wall_mood)}")
                logger.info(f"      Distance: {dist:.1f}")

        # Store current state as reference point
        state["trigger_data"] = current_data
        state["queue"] = queue
        state["original_queue"] = queue.copy()
        save_json(STATE_FILE, state)

    # 5. Apply next wallpaper
    if not queue:
        logger.warning(
            f"No wallpapers found within distance {MAX_MATCH_DIST}. Staying put."
        )
        return

    # Get next wallpaper from queue
    chosen = queue.pop(0)

    # Log selection details
    cache = load_json(CACHE_FILE)
    if (
        chosen in cache
        and isinstance(cache[chosen], dict)
        and "colors" in cache[chosen]
    ):
        chosen_data = cache[chosen]
        chosen_colors = chosen_data["colors"]
        chosen_mood = chosen_data["mood"]

        logger.info(f"Next wallpaper: {Path(chosen).name}")
        logger.info(f"  Colors: {', '.join(['#' + c for c in chosen_colors])}")
        logger.info(f"  Mood: {format_mood(chosen_mood)}")

    logger.info(f"Queue has {len(queue)} remaining")

    state["queue"] = queue
    save_json(STATE_FILE, state)
    append_history(chosen)

    logger.info(f"Applying: {Path(chosen).name}")

    swww_path = shutil.which("swww")
    if not swww_path:
        logger.error("swww command not found in PATH")
        sys.exit(1)

    try:
        subprocess.run(
            [
                swww_path,
                "img",
                chosen,
                "--transition-type",
                SWWW_TRANSITION_TYPE,
                "--transition-pos",
                SWWW_TRANSITION_POS,
                "--transition-step",
                SWWW_TRANSITION_STEP,
                "--transition-fps",
                SWWW_TRANSITION_FPS,
                "--transition-duration",
                SWWW_TRANSITION_DURATION,
            ],
            check=True,
            timeout=30,
            capture_output=True,
        )

        with open(CURRENT_WALLPAPER_FILE, "w") as f:
            f.write(chosen)

        logger.info("Wallpaper applied successfully")

        # Update theme colors if enabled
        if UPDATE_THEME_COLORS:
            wal_path = shutil.which("wal")
            if wal_path:
                try:
                    logger.info("Updating theme colors with pywal...")
                    subprocess.run(
                        [wal_path, "-i", chosen, "-n", "-q"],
                        check=True,
                        timeout=30,
                        capture_output=True,
                    )
                    logger.info("Theme colors updated")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Pywal timed out after 30s - continuing anyway")
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
