#!/usr/bin/env python3
"""
Zone Coverage Visualization - Configuration

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Centralized configuration for all zone coverage visualization
settings including colors, sizes, and formatting.

This file contains all configurable parameters used across the visualization
application. Edit values here to customize the appearance and behavior.

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

from typing import Dict, Any

# ═══════════════════════════════════════════════════════════════════════════
# 🗺️ MAP SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

MAP_CONFIG: Dict[str, Any] = {
    # Initial map center (lat, lon in WGS84)
    "center": [51.5, -1.0],
    
    # Initial zoom level
    "zoom": 14,
    
    # Base tile layer opacity (0-1)
    "base_layer_opacity": 0.25,
}


# ═══════════════════════════════════════════════════════════════════════════
# 📍 BOREHOLE MARKER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

BOREHOLE_MARKER_CONFIG: Dict[str, Any] = {
    # Visible dot radius in meters (scales with zoom)
    "visible_radius_m": 8,
    
    # Invisible grab radius in meters (for easier clicking/dragging)
    "grab_radius_m": 24,
    
    # Visible dot color
    "color": "#000000",
    
    # Visible dot fill color
    "fill_color": "#000000",
    
    # Visible dot fill opacity (0-1)
    "fill_opacity": 1,
    
    # Visible dot border weight
    "weight": 0,
}


# ═══════════════════════════════════════════════════════════════════════════
# 🎨 ZONE COLORS
# ═══════════════════════════════════════════════════════════════════════════

ZONE_COLORS: Dict[str, str] = {
    "Embankment": "#e74c3c",  # Red
    "Highways": "#3498db",    # Blue
    # Add more zone types as needed:
    # "Foundation": "#27ae60",  # Green
    # "Structural": "#9b59b6",  # Purple
}

# Default zone color if zone type not in ZONE_COLORS
DEFAULT_ZONE_COLOR: str = "#3498db"


# ═══════════════════════════════════════════════════════════════════════════
# 🟦 ZONE POLYGON STYLE
# ═══════════════════════════════════════════════════════════════════════════

ZONE_POLYGON_STYLE: Dict[str, Any] = {
    "color": "#666666",        # Border color
    "weight": 2,               # Border width in pixels
    "opacity": 0.8,            # Border opacity
    "fill_color": "#ffffff",   # Fill color
    "fill_opacity": 1,         # Fill opacity
}


# ═══════════════════════════════════════════════════════════════════════════
# 🔵 PROPOSED COVERAGE POLYGON STYLE
# ═══════════════════════════════════════════════════════════════════════════

PROPOSED_COVERAGE_STYLE: Dict[str, Any] = {
    "color": "#2980b9",        # Border color (blue)
    "weight": 3,               # Border width in pixels
    "opacity": 0.8,            # Border opacity
    "fill_color": "#3498db",   # Fill color (light blue)
    "fill_opacity": 0.25,      # Fill opacity
}


# ═══════════════════════════════════════════════════════════════════════════
# 🟢 EXISTING COVERAGE POLYGON STYLE
# ═══════════════════════════════════════════════════════════════════════════

EXISTING_COVERAGE_STYLE: Dict[str, Any] = {
    "color": "#27ae60",        # Border color (green)
    "weight": 2,               # Border width in pixels
    "opacity": 0.7,            # Border opacity
    "fill_color": "#2ecc71",   # Fill color (light green)
    "fill_opacity": 0.35,      # Fill opacity
}


# ═══════════════════════════════════════════════════════════════════════════
# 📊 COVERAGE STATS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

COVERAGE_STATS_CONFIG: Dict[str, Any] = {
    # Percentage thresholds for color coding
    "good_threshold": 90,      # >= 90% = green
    "medium_threshold": 50,    # >= 50% = yellow, < 50% = red
    
    # Colors for coverage stats
    "good_color": "#27ae60",   # Green
    "medium_color": "#f39c12", # Yellow/orange
    "poor_color": "#e74c3c",   # Red
}


# ═══════════════════════════════════════════════════════════════════════════
# 🖱️ UI SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

UI_CONFIG: Dict[str, Any] = {
    # Maximum undo history entries
    "max_undo_history": 50,
    
    # Button colors
    "undo_btn_color": "#95a5a6",
    "add_btn_color": "#27ae60",
    "add_btn_active_color": "#e74c3c",
    "delete_hint_color": "#e74c3c",
    
    # Add mode indicator background
    "add_mode_indicator_color": "#27ae60",
}


# ═══════════════════════════════════════════════════════════════════════════
# 🔧 GEOMETRY SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

GEOMETRY_CONFIG: Dict[str, Any] = {
    # Buffer resolution (number of segments in circle approximation)
    "buffer_resolution": 32,
    
    # Default max spacing if not specified in zone
    "default_max_spacing_m": 100,
}


# ═══════════════════════════════════════════════════════════════════════════
# 📦 EXPORT ALL CONFIG
# ═══════════════════════════════════════════════════════════════════════════

def get_frontend_config() -> Dict[str, Any]:
    """
    Get configuration for frontend JavaScript.
    
    Returns a dict suitable for JSON serialization and use in the frontend.
    """
    return {
        "map": MAP_CONFIG,
        "boreholeMarker": BOREHOLE_MARKER_CONFIG,
        "zoneColors": ZONE_COLORS,
        "defaultZoneColor": DEFAULT_ZONE_COLOR,
        "zonePolygonStyle": ZONE_POLYGON_STYLE,
        "proposedCoverageStyle": PROPOSED_COVERAGE_STYLE,
        "existingCoverageStyle": EXISTING_COVERAGE_STYLE,
        "coverageStats": COVERAGE_STATS_CONFIG,
        "ui": UI_CONFIG,
    }
