#!/usr/bin/env python3
"""
Zone Coverage Visualization - Configuration

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Configuration dictionary for zone coverage visualization.
This is the user-facing configuration file - edit values here.

Pattern: Same as Main/Gap_Analysis_EC7/config.py
- viz_config.py defines the VIZ_CONFIG_DATA dictionary (edit this)
- viz_config_types.py defines typed dataclasses and loads from VIZ_CONFIG_DATA

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

from typing import Dict, Any

# ═══════════════════════════════════════════════════════════════════════════
# 🎨 ZONE COVERAGE VISUALIZATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

VIZ_CONFIG_DATA: Dict[str, Any] = {
    # ═══════════════════════════════════════════════════════════════════════
    # 🗺️ MAP SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    "map": {
        "center": [51.5, -1.0],  # [lat, lon] - Default center
        "zoom": 14,
        "base_layer_opacity": 0.25,
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 📍 BOREHOLE MARKER SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    "borehole_marker": {
        # All markers use L.circle (meters) for smooth drag transitions
        "visible_radius_m": 8.0,  # Radius in meters (scales with zoom) - used for inside-zone
        "grab_radius_multiplier": 2.0,  # Grab area = visible_radius * multiplier
        "color": "#000000",  # Black (inside-zone boreholes)
        "fill_color": "#000000",
        "fill_opacity": 1.0,
        "weight": 0,
        "hover_scale": 2.0,  # Scale factor on hover (2.0 = double size)
        # Outside-zone markers (larger than inside-zone, different color)
        "outside_zone_color": "#FF8C00",  # Dark orange
        "outside_zone_radius_multiplier": 2.0,  # Outside radius = visible_radius * 2
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🎨 ZONE STYLING
    # ═══════════════════════════════════════════════════════════════════════
    "zone_colors": {
        "Embankment": "#e74c3c",  # Red
        "Highways": "#3498db",  # Blue
    },
    "default_zone_color": "#3498db",
    "zone_polygon_style": {
        "color": "#666666",
        "weight": 3,
        "opacity": 0.8,
        "fill_color": "#ffffff",
        "fill_opacity": 1.0,
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🔵 PROPOSED COVERAGE STYLING (blue circles)
    # ═══════════════════════════════════════════════════════════════════════
    "proposed_coverage_style": {
        "color": "#2980b9",
        "weight": 3,
        "opacity": 0.8,
        "fill_color": "#3498db",
        "fill_opacity": 0.25,
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🟢 EXISTING COVERAGE STYLING (green, from main.py output)
    # ═══════════════════════════════════════════════════════════════════════
    "existing_coverage_style": {
        "color": "#27ae60",
        "weight": 2,
        "opacity": 0.8,
        "fill_color": "#3ff88c",
        "fill_opacity": 0.8,
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 📊 COVERAGE STATS PANEL
    # ═══════════════════════════════════════════════════════════════════════
    "coverage_stats": {
        "good_threshold": 90.0,  # Green if coverage >= 90%
        "medium_threshold": 50.0,  # Yellow if coverage >= 50%
        "good_color": "#27ae60",
        "medium_color": "#f39c12",
        "poor_color": "#e74c3c",
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🖱️ UI SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    "ui": {
        "max_undo_history": 50,
        "undo_btn_color": "#95a5a6",
        "add_btn_color": "#27ae60",
        "add_btn_active_color": "#e74c3c",
        "delete_hint_color": "#e74c3c",
        "add_mode_indicator_color": "#27ae60",
        "coverage_panel_width_px": 320,
        "coverage_progress_bar_width_px": 100,
        "show_zone_tooltips": True,  # Show zone name tooltips on hover
        "show_zone_focus_outline": False,  # Show focus outline (black rectangle) when clicking zones
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🔧 GEOMETRY SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    "geometry": {
        "buffer_resolution": 128,
        "default_max_spacing_m": 100.0,
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 👁️ ZONE VISIBILITY SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    "zone_visibility": {
        # Mode options:
        # - "clip_coverage": Only hide coverage portion over hidden zone (default)
        # - "hide_zone_boreholes": Hide ALL boreholes inside hidden zone AND their entire coverage
        "mode": "hide_zone_boreholes",
    },
}
