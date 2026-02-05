#!/usr/bin/env python3
"""
Shapefile Configuration for Gap Analysis EC7

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Centralized configuration for all shapefile layers displayed in
the Gap Analysis EC7 visualization. Separates shapefile settings from main CONFIG.

Key Features:
1. Multiple shapefile support with independent settings
2. Layer ordering control for visualization stacking
3. Coverage layer designation (use_for_coverage=True) for analysis
4. Per-feature rendering overrides (colors, linewidths)
5. Per-zone max_spacing_m configuration
6. Dynamic zone naming (Name column or display_name fallback)
7. Multiple coverage layers support with zone ID prefixing

Navigation Guide:
- SHAPEFILE_CONFIG: Main configuration dictionary
- "layers": Individual shapefile configurations
- "defaults": Fallback settings for unconfigured properties

Zone Naming:
- If name_column is set (e.g., "Name"), zone names come from that column
- If name_column is None, each feature becomes a separate zone named "display_name_0", "display_name_1", etc.
  This ensures the ILP treats each feature as a separate optimization problem.

Per-Zone Spacing Resolution (priority order):
1. features[zone_name]["max_spacing_m"] - Feature override
2. layer["max_spacing_m"] - Layer default
3. defaults["max_spacing_m"] - Global default
4. CONFIG["max_spacing_m"] - Ultimate fallback

MODIFICATION POINT: Add new shapefiles to the "layers" dictionary.
"""

from typing import Dict, Any, List, Optional
import warnings

# ═══════════════════════════════════════════════════════════════════════════
# � CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# Tolerance for comparing max_spacing_m values during priority resolution.
# Values within this tolerance are considered equal, triggering order comparison.
SPACING_EQUALITY_TOLERANCE_M = 0.01  # 1 cm tolerance

# ═══════════════════════════════════════════════════════════════════════════
# �📁 SHAPEFILE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
# Each shapefile layer has:
# - enabled: Whether to load and display this layer
# - file_path: Relative path from workspace root
# - display_name: Human-readable name for legends/UI
# - layer_order: Drawing order (higher = drawn on top)
# - use_for_coverage: If True, this layer is used for coverage analysis
# - name_column: Column for zone names (None = each feature is a separate zone, named by index)
# - max_spacing_m: Default spacing for this layer (overridable per-feature)
# - rendering: Default rendering settings for this layer
# - features: Per-feature overrides keyed by Name attribute value
# ═══════════════════════════════════════════════════════════════════════════

SHAPEFILE_CONFIG: Dict[str, Any] = {
    # ═══════════════════════════════════════════════════════════════════════
    # 📍 SHAPEFILE LAYERS
    # ═══════════════════════════════════════════════════════════════════════
    "layers": {
        # ───────────────────────────────────────────────────────────────────
        # Coverage layer: Embankment (used for coverage analysis)
        # Single zone - no per-zone breakdown (features identified by feature ID)
        # ───────────────────────────────────────────────────────────────────
        "embankment_zones": {
            "enabled": True,
            "file_path": "Project Shapefiles/Embankment.shp",
            "display_name": "Embankment",
            "layer_order": 10,  # Drawn on top
            "use_for_coverage": True,  # Used for coverage analysis
            "name_column": None,  # No per-zone breakdown - single max_spacing for whole shapefile
            "max_spacing_m": 200.0,  # Single spacing for all features
            "order": 2,  # Priority for overlap resolution (lower = higher priority, used when spacing equal)
            "rendering": {
                "boundary_color": "#FF0000",  # Red
                "boundary_linewidth": 2.5,
            },
            "features": {},  # No per-feature overrides (single zone from display_name)
        },
        # ───────────────────────────────────────────────────────────────────
        # Coverage layer: Highways (used for coverage analysis)
        # Single zone - no per-zone breakdown (features identified by feature ID)
        # ───────────────────────────────────────────────────────────────────
        "highways": {
            "enabled": True,
            "file_path": "Project Shapefiles/Highways.shp",
            "display_name": "Highways",
            "layer_order": 9,  # Drawn just below embankment
            "use_for_coverage": True,  # Used for coverage analysis
            "name_column": None,  # No per-zone breakdown - single max_spacing for whole shapefile
            "max_spacing_m": 200.0,  # Single spacing for all features
            "order": 1,  # Priority for overlap resolution (lower = higher priority, used when spacing equal)
            "rendering": {
                "boundary_color": "#0066CC",  # Blue
                "boundary_linewidth": 2.5,
            },
            "features": {},  # No per-feature overrides (single zone from display_name)
        },
        # ───────────────────────────────────────────────────────────────────
        # Display layer: GIR Boundary (display only, not used in analysis)
        # ───────────────────────────────────────────────────────────────────
        "gir_boundary": {
            "enabled": True,
            "file_path": "Project Shapefiles/GIR_LocationGroup_Zone_250923.shp",
            "display_name": "GIR Boundary",
            "layer_order": 5,  # Drawn behind embankment_zones
            "use_for_coverage": False,  # NOT used for coverage analysis
            "name_column": "Name",
            "rendering": {
                "boundary_color": "#666666",  # Dark gray
                "boundary_linewidth": 1.5,
            },
            "features": {},  # No per-feature overrides
        },
        # ───────────────────────────────────────────────────────────────────
        # MODIFICATION POINT: Add new shapefiles below
        # ───────────────────────────────────────────────────────────────────
        # Example template for new shapefile:
        # "new_layer_name": {
        #     "enabled": True,
        #     "file_path": "path/to/shapefile.shp",
        #     "display_name": "Display Name",
        #     "layer_order": 3,
        #     "use_for_coverage": False,  # Set True only for the coverage analysis layer
        #     "rendering": {
        #         "boundary_color": "#RRGGBB",
        #         "boundary_linewidth": 2.0,
        #     },
        #     "features": {},
        # },
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🎨 DEFAULT SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    # Applied when a shapefile or feature doesn't specify a setting
    "defaults": {
        "enabled": True,
        "layer_order": 5,
        "use_for_coverage": False,
        "name_column": "Name",  # Default column for zone names
        "max_spacing_m": 100.0,  # Default spacing (fallback)
        "order": 999,  # Default priority for overlap resolution (lower = higher priority)
        "rendering": {
            "boundary_color": "#000000",  # Black
            "boundary_linewidth": 2.0,
        },
        "feature_defaults": {
            "enabled": True,
            "boundary_color": "#000000",
            "boundary_linewidth": 2.0,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 🔧 HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def get_enabled_layers() -> Dict[str, Any]:
    """
    Get all enabled shapefile layers sorted by layer_order.

    Returns:
        Dict of enabled layers sorted by layer_order (ascending).
    """
    layers = SHAPEFILE_CONFIG["layers"]
    defaults = SHAPEFILE_CONFIG["defaults"]

    enabled = {
        key: config
        for key, config in layers.items()
        if config.get("enabled", defaults.get("enabled", True))
    }

    # Sort by layer_order (lower drawn first)
    return dict(
        sorted(
            enabled.items(),
            key=lambda x: x[1].get("layer_order", defaults.get("layer_order", 5)),
        )
    )


def get_coverage_layer_key() -> Optional[str]:
    """
    DEPRECATED: Use get_coverage_layer_keys() for multi-layer support.

    Returns first layer with use_for_coverage=True for backward compatibility.
    Logs warning if multiple coverage layers exist.

    Returns:
        Layer key string, or None if no coverage layer defined.
    """
    coverage_keys = get_coverage_layer_keys()

    if len(coverage_keys) > 1:
        warnings.warn(
            f"Multiple coverage layers found: {coverage_keys}. "
            f"Use get_coverage_layer_keys() instead. Returning first: {coverage_keys[0]}",
            DeprecationWarning,
            stacklevel=2,
        )

    return coverage_keys[0] if coverage_keys else None


def get_coverage_layer_keys() -> List[str]:
    """
    Get ALL shapefile layers marked for coverage analysis.

    Returns:
        List of layer keys with use_for_coverage=True and enabled=True.

    Example:
        >>> get_coverage_layer_keys()
        ['embankment_zones', 'test_boundary']
    """
    return [
        key
        for key, config in SHAPEFILE_CONFIG["layers"].items()
        if config.get("use_for_coverage", False) and config.get("enabled", True)
    ]


def get_layer_name_column(layer_key: str) -> Optional[str]:
    """
    Get the column name used for zone identification.

    Args:
        layer_key: The layer key (e.g., "embankment_zones")

    Returns:
        Column name string (e.g., "Name"), or None if layer uses display_name.

    Example:
        >>> get_layer_name_column("embankment_zones")
        "Name"
        >>> get_layer_name_column("test_boundary")
        None  # Uses display_name
    """
    config = get_layer_config(layer_key)
    defaults = SHAPEFILE_CONFIG.get("defaults", {})

    # Explicitly check for None (means use display_name)
    if "name_column" in config:
        return config["name_column"]  # Could be None intentionally

    return defaults.get("name_column", "Name")


def get_zone_max_spacing(layer_key: str, zone_name: str) -> float:
    """
    Get max_spacing_m for a specific zone.

    Resolution order:
    1. features[zone_name]["max_spacing_m"] if defined
    2. layer["max_spacing_m"] if defined
    3. defaults["max_spacing_m"] if defined
    4. CONFIG["max_spacing_m"] (ultimate fallback)

    Args:
        layer_key: The layer key
        zone_name: The zone name within that layer

    Returns:
        max_spacing_m value in meters.

    Example:
        >>> get_zone_max_spacing("embankment_zones", "Zone 1")
        50.0  # Feature override
        >>> get_zone_max_spacing("embankment_zones", "Zone 3")
        100.0  # Layer default
    """
    from Gap_Analysis_EC7.config import CONFIG

    layer_config = get_layer_config(layer_key)
    features = layer_config.get("features", {})
    defaults = SHAPEFILE_CONFIG.get("defaults", {})

    # 1. Check feature-specific override
    if zone_name in features:
        feature_config = features[zone_name]
        if "max_spacing_m" in feature_config:
            return float(feature_config["max_spacing_m"])

    # 2. Check layer default
    if "max_spacing_m" in layer_config:
        return float(layer_config["max_spacing_m"])

    # 3. Check global shapefile defaults
    if "max_spacing_m" in defaults:
        return float(defaults["max_spacing_m"])

    # 4. Ultimate fallback to main CONFIG
    return float(CONFIG.get("max_spacing_m", 100.0))


def get_zone_order(layer_key: str, zone_name: str) -> int:
    """
    Get processing order for a zone (used for overlap resolution).

    The order parameter determines priority when two overlapping zones have
    EXACTLY EQUAL max_spacing_m values. Lower order = higher priority.

    NOTE: When max_spacing_m differs, the zone with LOWER max_spacing_m always
    wins, regardless of order. Order is ONLY consulted for equal spacing.

    Resolution order:
    1. features[zone_name]["order"] if defined
    2. layer["order"] if defined
    3. defaults["order"] (999)

    Args:
        layer_key: The layer key (e.g., "embankment_zones")
        zone_name: The zone name within that layer

    Returns:
        order value (int). Lower = higher priority.

    Example:
        >>> get_zone_order("embankment_zones", "Zone 1")
        1  # Layer default
        >>> get_zone_order("highways", "Highways_0")
        2  # Layer default
    """
    layer_config = get_layer_config(layer_key)
    features = layer_config.get("features", {})
    defaults = SHAPEFILE_CONFIG.get("defaults", {})

    # 1. Check feature-specific override
    if zone_name in features:
        feature_config = features[zone_name]
        if "order" in feature_config:
            return int(feature_config["order"])

    # 2. Check layer default
    if "order" in layer_config:
        return int(layer_config["order"])

    # 3. Global default
    return int(defaults.get("order", 999))


def make_zone_id(layer_key: str, zone_name: str) -> str:
    """
    Create unique zone identifier for cross-layer disambiguation.

    Args:
        layer_key: The layer key (e.g., "embankment_zones")
        zone_name: The zone name (e.g., "Zone 1")

    Returns:
        Unique zone ID string.

    Example:
        >>> make_zone_id("embankment_zones", "Zone 1")
        "embankment_zones:Zone 1"
    """
    return f"{layer_key}:{zone_name}"


# Backward compatibility alias
def get_primary_layer_key() -> Optional[str]:
    """Deprecated: Use get_coverage_layer_key() instead."""
    return get_coverage_layer_key()


def get_layer_config(layer_key: str) -> Dict[str, Any]:
    """
    Get configuration for a specific layer with defaults applied.

    Args:
        layer_key: The layer key (e.g., "embankment_zones")

    Returns:
        Layer config dict with defaults merged in.
    """
    layers = SHAPEFILE_CONFIG["layers"]
    defaults = SHAPEFILE_CONFIG["defaults"]

    if layer_key not in layers:
        return defaults.copy()

    config = layers[layer_key].copy()

    # Merge defaults for missing keys
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif key == "rendering" and isinstance(value, dict):
            # Merge rendering defaults
            merged_rendering = value.copy()
            merged_rendering.update(config.get("rendering", {}))
            config["rendering"] = merged_rendering

    return config


def get_feature_config(layer_key: str, feature_name: str) -> Dict[str, Any]:
    """
    Get rendering configuration for a specific feature within a layer.

    Args:
        layer_key: The layer key (e.g., "embankment_zones")
        feature_name: The feature Name attribute value (e.g., "Zone 1")

    Returns:
        Feature config dict with layer and global defaults applied.
    """
    layer_config = get_layer_config(layer_key)
    features = layer_config.get("features", {})
    defaults = SHAPEFILE_CONFIG["defaults"]["feature_defaults"]
    layer_rendering = layer_config.get("rendering", {})

    if feature_name not in features:
        # Use layer rendering defaults
        return {
            "enabled": True,
            "boundary_color": layer_rendering.get(
                "boundary_color", defaults["boundary_color"]
            ),
            "boundary_linewidth": layer_rendering.get(
                "boundary_linewidth", defaults["boundary_linewidth"]
            ),
        }

    feature_config = features[feature_name].copy()

    # Merge defaults
    for key, value in defaults.items():
        if key not in feature_config:
            # Try layer rendering, then global default
            feature_config[key] = layer_rendering.get(key, value)

    return feature_config


def build_zones_config_for_visualization() -> Dict[str, Dict[str, Any]]:
    """
    Build a zones_config dictionary compatible with visualization code.

    This replaces the legacy CONFIG["zones"] section by extracting zone
    rendering settings from shapefile_config layers.

    Returns:
        Dict mapping zone_name to visualization config with keys:
        - enabled: bool
        - boundary_color: str (hex color)
        - boundary_linewidth: float

    Example:
        >>> zones_config = build_zones_config_for_visualization()
        >>> zones_config["Zone 1"]["boundary_color"]
        "#FF0000"
    """
    zones_config: Dict[str, Dict[str, Any]] = {}
    defaults = SHAPEFILE_CONFIG["defaults"]["feature_defaults"]

    for layer_key, layer_config in SHAPEFILE_CONFIG["layers"].items():
        if not layer_config.get("enabled", True):
            continue

        layer_rendering = layer_config.get("rendering", {})
        features = layer_config.get("features", {})

        # Get zones from features if available
        for zone_name, feature_config in features.items():
            zones_config[zone_name] = {
                "enabled": feature_config.get("enabled", True),
                "boundary_color": feature_config.get(
                    "boundary_color",
                    layer_rendering.get("boundary_color", defaults["boundary_color"]),
                ),
                "boundary_linewidth": feature_config.get(
                    "boundary_linewidth",
                    layer_rendering.get(
                        "boundary_linewidth", defaults["boundary_linewidth"]
                    ),
                ),
            }

        # If no features, use display_name as zone name (for layers without Name column)
        if not features and layer_config.get("use_for_coverage", False):
            display_name = layer_config.get("display_name", layer_key)
            zones_config[display_name] = {
                "enabled": True,
                "boundary_color": layer_rendering.get(
                    "boundary_color", defaults["boundary_color"]
                ),
                "boundary_linewidth": layer_rendering.get(
                    "boundary_linewidth", defaults["boundary_linewidth"]
                ),
            }

    return zones_config
