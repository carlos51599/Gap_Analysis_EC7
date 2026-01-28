#!/usr/bin/env python3
"""
Satellite Imagery Fetching and Caching Module

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Fetch, cache, and serve satellite tile imagery for map backgrounds.
Handles API requests to Esri WorldImagery with caching and SSL workarounds.

Key Features:
1. Tile fetching via contextily from Esri WorldImagery
2. File-based cache with deterministic hash keys
3. Base64 encoding for HTML embedding
4. SSL workarounds for corporate networks

Dependencies:
- contextily (for tile fetching)
- matplotlib (for rendering)
- PIL (for image processing)

Navigation Guide:
- Use VS Code outline (Ctrl+Shift+O) to jump between functions
"""

import base64
import hashlib
import logging
import os
import ssl
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

# ===========================================================================
# SSL WORKAROUNDS FOR CORPORATE NETWORKS
# ===========================================================================
# Disable SSL verification for corporate networks with proxy/certificate issues
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Monkey-patch requests to disable SSL verification
import requests

_original_request = requests.Session.request


def _patched_request(self, *args, **kwargs) -> requests.Response:
    """Patched request method that disables SSL verification."""
    kwargs["verify"] = False
    return _original_request(self, *args, **kwargs)


requests.Session.request = _patched_request

# Suppress SSL warnings
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ===========================================================================
# CACHE PATH UTILITIES
# ===========================================================================


def get_cache_path(
    bounds: Tuple[float, float, float, float],
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Generate a cache file path based on bounds hash.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy)
        cache_dir: Directory for cache (default: Output/satellite_cache)

    Returns:
        Path to cache file
    """
    # Create a deterministic hash from bounds (rounded to nearest meter)
    bounds_str = f"{int(bounds[0])}_{int(bounds[1])}_{int(bounds[2])}_{int(bounds[3])}"
    bounds_hash = hashlib.md5(bounds_str.encode()).hexdigest()[:12]

    # Cache path in Output/satellite_cache/
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "Output" / "satellite_cache"

    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / f"satellite_{bounds_hash}.b64"


def load_from_cache(
    bounds: Tuple[float, float, float, float],
    cache_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """
    Load satellite imagery from cache if available.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy)
        cache_dir: Directory for cache files
        logger: Optional logger

    Returns:
        Base64 encoded string if cache exists, None otherwise
    """
    cache_path = get_cache_path(bounds, cache_dir)

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = f.read()
            if logger:
                logger.info(
                    f"📦 Loaded satellite imagery from cache: {cache_path.name}"
                )
            return cached_data
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Failed to load satellite cache: {e}")

    return None


def save_to_cache(
    bounds: Tuple[float, float, float, float],
    base64_data: str,
    cache_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Save satellite imagery to cache.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy)
        base64_data: Base64 encoded image string
        cache_dir: Directory for cache files
        logger: Optional logger
    """
    cache_path = get_cache_path(bounds, cache_dir)

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(base64_data)
        if logger:
            logger.info(f"💾 Saved satellite imagery to cache: {cache_path.name}")
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Failed to save satellite cache: {e}")


# ===========================================================================
# MAIN FETCH FUNCTION
# ===========================================================================


def fetch_satellite_tiles_base64(
    bounds: Tuple[float, float, float, float],
    crs: str = "EPSG:27700",
    alpha: float = 0.8,
    logger: Optional[logging.Logger] = None,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    Fetch satellite tiles from Esri WorldImagery and return as base64 string.
    Uses local cache to avoid repeated fetches for same extent.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy) in the given CRS
        crs: Coordinate reference system (default: British National Grid)
        alpha: Opacity for the satellite image (0-1)
        logger: Optional logger instance
        use_cache: Whether to use caching (default True)
        cache_dir: Directory for cache files

    Returns:
        Base64 encoded PNG string with 'data:image/png;base64,' prefix, or None on failure
    """
    # Try to load from cache first
    if use_cache:
        cached = load_from_cache(bounds, cache_dir, logger)
        if cached:
            return cached

    try:
        import contextily as ctx
        import matplotlib.pyplot as plt

        if logger:
            logger.info("🛰️ Fetching satellite imagery (this may take a moment)...")

        minx, miny, maxx, maxy = bounds

        # Create a figure with exact extent
        fig, ax = plt.subplots(figsize=(20, 20), dpi=100)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ax.axis("off")

        # Fetch satellite tiles
        ctx.add_basemap(
            ax,
            crs=crs,
            source=ctx.providers.Esri.WorldImagery,
            alpha=alpha,
        )

        # Save to bytes buffer
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)

        # Encode as base64
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        result = f"data:image/png;base64,{img_base64}"

        if logger:
            logger.info("✅ Satellite imagery fetched successfully")

        # Save to cache for future runs
        if use_cache:
            save_to_cache(bounds, result, cache_dir, logger)

        return result

    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Could not fetch satellite imagery: {e}")
        return None


# ===========================================================================
# CONVENIENCE ALIASES (for backward compatibility)
# ===========================================================================

# Alias for backward compatibility with html_builder_EC7.py internal naming
_fetch_satellite_tiles_base64 = fetch_satellite_tiles_base64
_get_satellite_cache_path = get_cache_path
_load_satellite_cache = load_from_cache
_save_satellite_cache = save_to_cache
