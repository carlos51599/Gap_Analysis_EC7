#!/usr/bin/env python3
"""
Geometry Utility Functions

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Pure geometry and color computations for visualization.
No external dependencies on Plotly or HTML generation.

Key Functions:
1. Color conversions (hex to RGB, RGB to hex)
2. Color interpolation for gradients
3. Diverging colorscale generation
4. Bounds manipulation (center calculation, expansion)

Dependencies:
- numpy
- geopandas (for GeoDataFrame operations)

Navigation Guide:
- Use VS Code outline (Ctrl+Shift+O) to jump between functions
"""

from typing import Dict, List, Tuple, Union

import numpy as np
from geopandas import GeoDataFrame
from scipy.spatial import distance


# ===========================================================================
# COLOR UTILITIES
# ===========================================================================


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color string to RGB tuple.

    Args:
        hex_color: Color string like "#FF5733" or "FF5733"

    Returns:
        Tuple of (R, G, B) integers 0-255
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to hex color string.

    Args:
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)

    Returns:
        Hex color string with # prefix
    """
    return f"#{r:02x}{g:02x}{b:02x}"


def interpolate_color(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    """
    Interpolate between two RGB colors.

    Args:
        color1: Start color (RGB tuple)
        color2: End color (RGB tuple)
        t: Interpolation factor 0-1

    Returns:
        Interpolated RGB color
    """
    return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))


# ===========================================================================
# COLORSCALE GENERATION
# ===========================================================================


def generate_diverging_colorscale(
    low_color: str = "#2166AC",  # Blue
    mid_color: str = "#F7F7F7",  # White
    high_color: str = "#B2182B",  # Red
    n_steps: int = 11,
) -> List[List]:
    """
    Generate Plotly-compatible diverging colorscale.

    Args:
        low_color: Color for minimum values
        mid_color: Color for middle values
        high_color: Color for maximum values
        n_steps: Number of steps in colorscale

    Returns:
        List of [position, color] pairs for Plotly
    """
    low_rgb = hex_to_rgb(low_color)
    mid_rgb = hex_to_rgb(mid_color)
    high_rgb = hex_to_rgb(high_color)

    colorscale = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        if t <= 0.5:
            # Interpolate low -> mid
            color = interpolate_color(low_rgb, mid_rgb, t * 2)
        else:
            # Interpolate mid -> high
            color = interpolate_color(mid_rgb, high_rgb, (t - 0.5) * 2)

        colorscale.append([t, f"rgb{color}"])

    return colorscale


# ===========================================================================
# COORDINATE UTILITIES
# ===========================================================================


def bounds_to_center(
    bounds: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """
    Calculate center point from bounds.

    Args:
        bounds: (minx, miny, maxx, maxy)

    Returns:
        Tuple of (center_x, center_y)
    """
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    return (center_x, center_y)


def expand_bounds(
    bounds: Tuple[float, float, float, float],
    factor: float = 0.1,
) -> Tuple[float, float, float, float]:
    """
    Expand bounds by a percentage factor.

    Args:
        bounds: (minx, miny, maxx, maxy)
        factor: Expansion factor (0.1 = 10% on each side)

    Returns:
        Expanded bounds tuple
    """
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny

    return (
        minx - width * factor,
        miny - height * factor,
        maxx + width * factor,
        maxy + height * factor,
    )


# ===========================================================================
# BACKWARD COMPATIBILITY ALIASES
# ===========================================================================

# Aliases matching internal naming in html_builder_EC7.py
_hex_to_rgb = hex_to_rgb
