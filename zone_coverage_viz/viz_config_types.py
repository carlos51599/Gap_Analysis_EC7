#!/usr/bin/env python3
"""
Zone Coverage Visualization - Configuration Types

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Centralized, typed configuration for zone coverage visualization
using frozen dataclasses for immutability and type safety.

This follows the Typed Configuration Architecture pattern:
- viz_config.py defines VIZ_CONFIG_DATA dictionary (user edits this)
- viz_config_types.py defines frozen dataclasses (this file)
- VIZ_CONFIG module-level instance for orchestrator access
- Business logic receives primitives only

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

from dataclasses import dataclass
from typing import Dict, Any

# ═══════════════════════════════════════════════════════════════════════════
# 🗺️ MAP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MapConfig:
    """Configuration for map display settings."""

    center_lat: float = 51.5
    center_lon: float = -1.0
    zoom: int = 14
    base_layer_opacity: float = 0.25

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MapConfig":
        """Create from dictionary."""
        center = d.get("center", [51.5, -1.0])
        return cls(
            center_lat=(
                center[0] if isinstance(center, list) else d.get("center_lat", 51.5)
            ),
            center_lon=(
                center[1] if isinstance(center, list) else d.get("center_lon", -1.0)
            ),
            zoom=d.get("zoom", 14),
            base_layer_opacity=d.get("base_layer_opacity", 0.25),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "center": [self.center_lat, self.center_lon],
            "zoom": self.zoom,
            "base_layer_opacity": self.base_layer_opacity,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 📍 BOREHOLE MARKER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BoreholeMarkerConfig:
    """Configuration for borehole marker appearance."""

    visible_radius_m: float = 8.0
    grab_radius_multiplier: float = 2.0  # Grab area = visible_radius * multiplier
    color: str = "#000000"
    fill_color: str = "#000000"
    fill_opacity: float = 1.0
    weight: int = 0
    hover_scale: float = 2.0  # Scale factor on hover (2.0 = double size)
    # Outside-zone marker styling (boreholes not within any zone polygon)
    outside_zone_color: str = "#FF8C00"  # Dark orange
    outside_zone_radius_multiplier: float = (
        2.0  # Outside radius = visible_radius * this
    )

    @property
    def grab_radius_m(self) -> float:
        """Computed grab radius based on visible radius and multiplier."""
        return self.visible_radius_m * self.grab_radius_multiplier

    @property
    def outside_zone_radius_m(self) -> float:
        """Computed outside-zone marker radius (2x visible radius by default)."""
        return self.visible_radius_m * self.outside_zone_radius_multiplier

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoreholeMarkerConfig":
        """Create from dictionary."""
        return cls(
            visible_radius_m=d.get("visible_radius_m", 8.0),
            grab_radius_multiplier=d.get("grab_radius_multiplier", 8.0),
            color=d.get("color", "#000000"),
            fill_color=d.get("fill_color", "#000000"),
            fill_opacity=d.get("fill_opacity", 1.0),
            weight=d.get("weight", 0),
            hover_scale=d.get("hover_scale", 2.0),
            outside_zone_color=d.get("outside_zone_color", "#FF8C00"),
            outside_zone_radius_multiplier=d.get("outside_zone_radius_multiplier", 2.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "visible_radius_m": self.visible_radius_m,
            "grab_radius_multiplier": self.grab_radius_multiplier,
            "grab_radius_m": self.grab_radius_m,  # Computed value for frontend
            "color": self.color,
            "fill_color": self.fill_color,
            "fill_opacity": self.fill_opacity,
            "weight": self.weight,
            "hover_scale": self.hover_scale,
            "outside_zone_color": self.outside_zone_color,
            "outside_zone_radius_multiplier": self.outside_zone_radius_multiplier,
            "outside_zone_radius_m": self.outside_zone_radius_m,  # Computed for frontend
        }


# ═══════════════════════════════════════════════════════════════════════════
# 🎨 POLYGON STYLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PolygonStyleConfig:
    """Configuration for polygon styling (zones, coverage)."""

    color: str = "#666666"
    weight: int = 2
    opacity: float = 0.8
    fill_color: str = "#ffffff"
    fill_opacity: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolygonStyleConfig":
        """Create from dictionary."""
        return cls(
            color=d.get("color", "#666666"),
            weight=d.get("weight", 2),
            opacity=d.get("opacity", 0.8),
            fill_color=d.get("fill_color", "#ffffff"),
            fill_opacity=d.get("fill_opacity", 1.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "color": self.color,
            "weight": self.weight,
            "opacity": self.opacity,
            "fill_color": self.fill_color,
            "fill_opacity": self.fill_opacity,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 📊 COVERAGE STATS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CoverageStatsConfig:
    """Configuration for coverage statistics display."""

    good_threshold: float = 90.0
    medium_threshold: float = 50.0
    good_color: str = "#27ae60"
    medium_color: str = "#f39c12"
    poor_color: str = "#e74c3c"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CoverageStatsConfig":
        """Create from dictionary."""
        return cls(
            good_threshold=d.get("good_threshold", 90.0),
            medium_threshold=d.get("medium_threshold", 50.0),
            good_color=d.get("good_color", "#27ae60"),
            medium_color=d.get("medium_color", "#f39c12"),
            poor_color=d.get("poor_color", "#e74c3c"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "good_threshold": self.good_threshold,
            "medium_threshold": self.medium_threshold,
            "good_color": self.good_color,
            "medium_color": self.medium_color,
            "poor_color": self.poor_color,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 🖱️ UI CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class UIConfig:
    """Configuration for UI elements."""

    max_undo_history: int = 50
    undo_btn_color: str = "#95a5a6"
    add_btn_color: str = "#27ae60"
    add_btn_active_color: str = "#e74c3c"
    delete_hint_color: str = "#e74c3c"
    add_mode_indicator_color: str = "#27ae60"
    coverage_panel_width_px: int = 320
    coverage_progress_bar_width_px: int = 100
    show_zone_tooltips: bool = True  # Show zone name tooltips on hover
    show_zone_focus_outline: bool = (
        True  # Show focus outline (black rectangle) when clicking zones
    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UIConfig":
        """Create from dictionary."""
        return cls(
            max_undo_history=d.get("max_undo_history", 50),
            undo_btn_color=d.get("undo_btn_color", "#95a5a6"),
            add_btn_color=d.get("add_btn_color", "#27ae60"),
            add_btn_active_color=d.get("add_btn_active_color", "#e74c3c"),
            delete_hint_color=d.get("delete_hint_color", "#e74c3c"),
            add_mode_indicator_color=d.get("add_mode_indicator_color", "#27ae60"),
            coverage_panel_width_px=d.get("coverage_panel_width_px", 320),
            coverage_progress_bar_width_px=d.get("coverage_progress_bar_width_px", 100),
            show_zone_tooltips=d.get("show_zone_tooltips", True),
            show_zone_focus_outline=d.get("show_zone_focus_outline", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "max_undo_history": self.max_undo_history,
            "undo_btn_color": self.undo_btn_color,
            "add_btn_color": self.add_btn_color,
            "add_btn_active_color": self.add_btn_active_color,
            "delete_hint_color": self.delete_hint_color,
            "add_mode_indicator_color": self.add_mode_indicator_color,
            "coverage_panel_width_px": self.coverage_panel_width_px,
            "coverage_progress_bar_width_px": self.coverage_progress_bar_width_px,
            "show_zone_tooltips": self.show_zone_tooltips,
            "show_zone_focus_outline": self.show_zone_focus_outline,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 🔧 GEOMETRY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GeometryConfig:
    """Configuration for geometry calculations."""

    buffer_resolution: int = 128
    default_max_spacing_m: float = 100.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometryConfig":
        """Create from dictionary."""
        return cls(
            buffer_resolution=d.get("buffer_resolution", 128),
            default_max_spacing_m=d.get("default_max_spacing_m", 100.0),
        )


# ═══════════════════════════════════════════════════════════════════════════
# �️ ZONE VISIBILITY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ZoneVisibilityConfig:
    """Configuration for zone visibility behavior.
    
    Mode options:
        - "clip_coverage": Only hide coverage portion over hidden zone (default)
        - "hide_zone_boreholes": Hide ALL boreholes inside hidden zone AND their 
          entire coverage (even if coverage extends to visible zones)
    """

    mode: str = "clip_coverage"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ZoneVisibilityConfig":
        """Create from dictionary."""
        return cls(
            mode=d.get("mode", "clip_coverage"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for frontend."""
        return {"mode": self.mode}


# ═══════════════════════════════════════════════════════════════════════════
# �📦 MAIN VIZ CONFIGURATION CLASS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class VizConfig:
    """
    Main configuration class for zone coverage visualization.

    Access via the module-level VIZ_CONFIG instance.
    Orchestrators extract primitives; business logic receives primitives only.
    """

    map: MapConfig
    borehole_marker: BoreholeMarkerConfig
    zone_colors: Dict[str, str]
    default_zone_color: str
    zone_polygon_style: PolygonStyleConfig
    proposed_coverage_style: PolygonStyleConfig
    existing_coverage_style: PolygonStyleConfig
    coverage_stats: CoverageStatsConfig
    ui: UIConfig
    geometry: GeometryConfig
    zone_visibility: ZoneVisibilityConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VizConfig":
        """Create from dictionary."""
        return cls(
            map=MapConfig.from_dict(d.get("map", {})),
            borehole_marker=BoreholeMarkerConfig.from_dict(
                d.get("borehole_marker", {})
            ),
            zone_colors=d.get(
                "zone_colors", {"Embankment": "#e74c3c", "Highways": "#3498db"}
            ),
            default_zone_color=d.get("default_zone_color", "#3498db"),
            zone_polygon_style=PolygonStyleConfig.from_dict(
                d.get("zone_polygon_style", {"weight": 3})
            ),
            proposed_coverage_style=PolygonStyleConfig.from_dict(
                d.get(
                    "proposed_coverage_style",
                    {
                        "color": "#2980b9",
                        "weight": 3,
                        "fill_color": "#3498db",
                        "fill_opacity": 0.25,
                    },
                )
            ),
            existing_coverage_style=PolygonStyleConfig.from_dict(
                d.get(
                    "existing_coverage_style",
                    {
                        "color": "#27ae60",
                        "weight": 2,
                        "fill_color": "#3ff88c",
                        "fill_opacity": 0.8,
                    },
                )
            ),
            coverage_stats=CoverageStatsConfig.from_dict(d.get("coverage_stats", {})),
            ui=UIConfig.from_dict(d.get("ui", {})),
            geometry=GeometryConfig.from_dict(d.get("geometry", {})),
            zone_visibility=ZoneVisibilityConfig.from_dict(d.get("zone_visibility", {})),
        )

    @classmethod
    def defaults(cls) -> "VizConfig":
        """Create with all default values."""
        return cls.from_dict({})

    def to_frontend_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for frontend JSON API."""
        return {
            "map": self.map.to_dict(),
            "boreholeMarker": self.borehole_marker.to_dict(),
            "zoneColors": self.zone_colors,
            "defaultZoneColor": self.default_zone_color,
            "zonePolygonStyle": self.zone_polygon_style.to_dict(),
            "proposedCoverageStyle": self.proposed_coverage_style.to_dict(),
            "existingCoverageStyle": self.existing_coverage_style.to_dict(),
            "coverageStats": self.coverage_stats.to_dict(),
            "ui": self.ui.to_dict(),
            "zoneVisibility": self.zone_visibility.to_dict(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 📌 MODULE-LEVEL CONFIG INSTANCE
# ═══════════════════════════════════════════════════════════════════════════

# Import configuration data from separate file (user-editable)
from zone_coverage_viz.viz_config import VIZ_CONFIG_DATA

# Create typed config from data dictionary
# Edit viz_config.py to change settings (restart server after changes)
VIZ_CONFIG: VizConfig = VizConfig.from_dict(VIZ_CONFIG_DATA)


def get_frontend_config() -> Dict[str, Any]:
    """
    Get configuration for frontend JavaScript.

    Returns a dict suitable for JSON serialization and use in the frontend.
    This is the coordination boundary function for frontend config access.
    """
    return VIZ_CONFIG.to_frontend_dict()
