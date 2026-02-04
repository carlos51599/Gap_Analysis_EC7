#!/usr/bin/env python3
"""
EC7 HTML Builder for Gap Analysis

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Generate interactive HTML visualization for EC7 coverage analysis
with coverage zones and gap identification.

Key Features:
1. EC7 coverage zone visualization (green = covered, red = gaps)
2. Suggested borehole markers for uncovered areas
3. Borehole markers with depth slider filtering
4. Coverage statistics panel
5. BGS Geology layers (toggleable)
6. Satellite imagery overlay (toggleable)
7. Test data filtering (SPT, Triaxial Total, Triaxial Effective)

Navigation Guide:
- Section markers: # ===== for major sections
- Use VS Code outline (Ctrl+Shift+O) to jump between functions
"""

import json
import logging
import ssl
import os
import time
from typing import Dict, Any, Optional, Tuple, List, Union

from shapely.geometry.base import BaseGeometry

import plotly.graph_objects as go
from geopandas import GeoDataFrame

from Gap_Analysis_EC7.config import CONFIG
from Gap_Analysis_EC7.config_types import VisualizationConfig, BoreholeMarkerConfig

# Module-level logger for debug output
_logger = logging.getLogger(__name__)

# ===========================================================================
# PHASE 4: Import from visualization package
# ===========================================================================
from Gap_Analysis_EC7.visualization import (
    # Satellite utilities
    _fetch_satellite_tiles_base64,
    # Layout builder
    _build_layout_without_dropdown,
    # Panel generators
    _generate_filters_panel_html,
    _generate_layers_panel_html,
    _generate_legend_panel_html,
    _generate_coverage_stats_panel_html,
    # Trace builders
    _add_grid_cells_trace,
    _add_boreholes_trace,
    _add_zone_boundary_traces,
    # Hexgrid builder
    build_hexagon_grid_trace,
    # Client scripts
    generate_coverage_data_script,
    generate_click_to_copy_script,
)

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
    kwargs["verify"] = False
    return _original_request(self, *args, **kwargs)


requests.Session.request = _patched_request

# Suppress SSL warnings
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ===========================================================================
# COLOR NORMALIZATION FOR PLOTLY SCATTERGL
# ===========================================================================


def _normalize_color_for_scattergl(color: str) -> str:
    """Convert 8-char hex colors to rgba format for Scattergl compatibility.

    Plotly's Scattergl doesn't support 8-character hex colors (#RRGGBBAA).
    This function converts them to rgba() format.

    Args:
        color: Color string (hex, rgba, named color, etc.)

    Returns:
        Color string in Scattergl-compatible format
    """
    if not isinstance(color, str):
        return color

    # Check for 8-char hex format (#RRGGBBAA)
    if len(color) == 9 and color.startswith("#"):
        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            a = int(color[7:9], 16) / 255.0
            return f"rgba({r}, {g}, {b}, {a:.2f})"
        except ValueError:
            pass  # Invalid hex, return as-is

    return color


# ===========================================================================
# SATELLITE IMAGERY UTILITIES - MOVED TO visualization/satellite_tiles.py
# ===========================================================================
# Functions _get_satellite_cache_path, _load_satellite_cache,
# _save_satellite_cache, _fetch_satellite_tiles_base64 are now imported
# from Gap_Analysis_EC7.visualization (see imports above).
# ===========================================================================


# ===========================================================================
# EC7 COVERAGE ZONE VISUALIZATION
# ===========================================================================

# Coverage zone trace builders - used via backward compatibility aliases
from Gap_Analysis_EC7.visualization import (
    build_coverage_polygon_trace,
    build_coverage_marker_trace,
    build_coverage_buffer_trace,
    build_borehole_circles_trace,
)

# Backward compatibility aliases (old names -> new names)
build_merged_polygon_trace = build_coverage_polygon_trace
build_merged_marker_trace = build_coverage_marker_trace
build_merged_buffer_trace = build_coverage_buffer_trace


# ===========================================================================
# STYLING CONFIGURATION
# ===========================================================================
# LIQUID_GLASS_STYLE, PANEL_STYLE_ITEM, PANEL_STYLE_HEADER are now imported
# from Gap_Analysis_EC7.visualization (see imports above).

# Panel layout constants using typed config
# Create a typed VisualizationConfig from CONFIG for module-level defaults
_VIS_CONFIG = VisualizationConfig.from_dict(CONFIG.get("visualization", {}))
_PANEL_LAYOUT = _VIS_CONFIG.panel_layout
LEFT_PANEL_WIDTH = _PANEL_LAYOUT.left_panel_width
RIGHT_PANEL_WIDTH = _PANEL_LAYOUT.right_panel_width
PANEL_TOP_OFFSET = _PANEL_LAYOUT.top_offset
PANEL_VERTICAL_GAP = _PANEL_LAYOUT.vertical_gap
SIDEBAR_SPACING = _PANEL_LAYOUT.sidebar_spacing


# ===========================================================================
# CONFIGURATION HELPERS
# ===========================================================================


def _normalize_visualization_config(
    config: "Dict[str, Any] | VisualizationConfig",
) -> VisualizationConfig:
    """
    Normalize visualization config to typed VisualizationConfig.

    Supports backwards compatibility: accepts either raw dict or typed config.

    Args:
        config: Either CONFIG["visualization"] dict or VisualizationConfig dataclass

    Returns:
        VisualizationConfig instance
    """
    if isinstance(config, VisualizationConfig):
        return config
    return VisualizationConfig.from_dict(config)


def _get_borehole_marker_from_config(
    config: "Dict[str, Any] | VisualizationConfig",
) -> BoreholeMarkerConfig:
    """
    Extract borehole marker config from visualization config.

    Handles both dict and typed config formats.

    Args:
        config: Either CONFIG["visualization"] dict or VisualizationConfig dataclass

    Returns:
        BoreholeMarkerConfig instance
    """
    if isinstance(config, VisualizationConfig):
        return config.borehole_marker
    return BoreholeMarkerConfig.from_dict(config.get("borehole_marker", {}))


# ===========================================================================
# GRID CELL TRACE FUNCTIONS - MOVED TO visualization/plotly_traces.py
# ===========================================================================
# Functions _add_grid_cells_trace, _add_boreholes_trace, _add_hexagon_grid_overlay,
# and _add_zone_boundary_traces are now imported from Gap_Analysis_EC7.visualization
# (see imports above).


# ===========================================================================
# ===========================================================================
# PRECOMPUTED COVERAGE TRACES (VISIBILITY TOGGLING)
# ===========================================================================


def _add_single_combo_traces(
    fig: go.Figure,
    combo_key: str,
    data: Dict[str, Any],
    colors: Dict[str, str],
    is_visible: bool,
    max_spacing: float,
    hexgrid_config: Optional[Dict[str, Any]] = None,
    grid_spacing: float = 50.0,
    zones_gdf: Optional["GeoDataFrame"] = None,
) -> Dict[str, Tuple[int, int]]:
    """Add 5 merged traces for a single filter combination (includes hexagon grid)."""
    from Gap_Analysis_EC7.parallel.coverage_orchestrator import deserialize_geometry

    ranges: Dict[str, Tuple[int, int]] = {}

    # Parse geometries first
    covered_geom = (
        deserialize_geometry(data.get("covered")) if data.get("covered") else None
    )
    gaps_geom = deserialize_geometry(data.get("gaps")) if data.get("gaps") else None

    # IMPORTANT: Add gaps FIRST (underneath), then coverage (on top).
    # This ensures that any WebGL rendering imprecision at polygon boundaries
    # results in green coverage appearing on top of red gaps, not vice versa.
    # The geometries are mathematically non-overlapping, but WebGL fill="toself"
    # can have sub-pixel rendering artifacts at complex polygon edges.

    # Gap zone trace (rendered first = underneath)
    start_idx = len(fig.data)
    fig.add_trace(
        build_merged_polygon_trace(
            geometry=gaps_geom,
            name=f"Gaps ({combo_key})",
            fill_color=colors["gap"],
            visible=is_visible,
            show_legend=False,
        )
    )
    ranges["gaps"] = (start_idx, len(fig.data))

    # Covered zone trace (rendered second = on top)
    start_idx = len(fig.data)
    fig.add_trace(
        build_merged_polygon_trace(
            geometry=covered_geom,
            name=f"Coverage ({combo_key})",
            fill_color=colors["covered"],
            visible=is_visible,
            show_legend=False,
        )
    )
    ranges["covered"] = (start_idx, len(fig.data))

    # Proposed buffer and marker traces
    proposed = data.get("proposed", [])
    ranges.update(
        _add_proposed_traces(
            fig, combo_key, proposed, colors, is_visible, max_spacing, zones_gdf
        )
    )

    # Removed boreholes (from consolidation) - red traces
    # NOTE: Always hidden initially since Second Pass checkbox is unchecked by default
    removed = data.get("removed", [])
    _logger.debug(
        f"html_builder: removed count = {len(removed)}, data.keys = {list(data.keys())}"
    )
    if removed:
        ranges.update(
            _add_removed_traces(fig, combo_key, removed, False, max_spacing, zones_gdf)
        )

    # Added boreholes (from consolidation) - green traces
    # NOTE: Always hidden initially since Second Pass checkbox is unchecked by default
    added = data.get("added", [])
    _logger.debug(f"html_builder: added count = {len(added)}")
    if added:
        ranges.update(
            _add_added_traces(fig, combo_key, added, False, max_spacing, zones_gdf)
        )

    # Hexagon grid trace (per-combo, based on gap geometry)
    # Passes zones_gdf for per-zone spacing if available
    ranges["hexagon_grid"] = _add_hexgrid_trace(
        fig, combo_key, gaps_geom, max_spacing, grid_spacing, hexgrid_config, zones_gdf
    )

    # Second pass grid trace (consolidation candidates) - always hidden initially
    second_pass_grid = data.get("second_pass_grid", [])
    if second_pass_grid:
        ranges["second_pass_grid"] = _add_second_pass_grid_trace(
            fig, combo_key, second_pass_grid, hexgrid_config
        )

        # Add first-pass border boreholes as black X markers
        # These are non-grid candidates that were also used in the ILP
        ranges["first_pass_candidates"] = _add_first_pass_candidates_trace(
            fig, combo_key, second_pass_grid
        )

        # DEBUG: Add buffer zone outline if enabled in config
        # Controlled by CONFIG["border_consolidation"]["show_buffer_zone_outline"]
        from Gap_Analysis_EC7.config import CONFIG

        show_outline = CONFIG.get("border_consolidation", {}).get(
            "show_buffer_zone_outline", False
        )
        if show_outline:
            ranges["buffer_zone_outline"] = _add_buffer_zone_outline_trace(
                fig, combo_key, second_pass_grid
            )

        # DEBUG: Add split region boundaries if enabled in config
        # Controlled by CONFIG["border_consolidation"]["show_split_regions"]
        show_split_regions = CONFIG.get("border_consolidation", {}).get(
            "show_split_regions", False
        )
        if show_split_regions:
            ranges["split_regions"] = _add_split_regions_trace(
                fig, combo_key, second_pass_grid
            )

    # ═══════════════════════════════════════════════════════════════════════
    # 🌐 CZRC (Cross-Zone Reachability Consolidation) VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    # Shows coverage clouds and pairwise intersections
    czrc_data = data.get("czrc_data")
    if czrc_data:
        from Gap_Analysis_EC7.config import CONFIG

        czrc_config = CONFIG.get("border_consolidation", {})

        # Add coverage clouds (very light fills)
        # Always add - visibility controlled by Zone Overlap checkbox
        ranges["czrc_clouds"] = _add_czrc_coverage_clouds_trace(
            fig, combo_key, czrc_data, czrc_config
        )

        # Add pairwise intersection regions
        # Always add - visibility controlled by Zone Overlap checkbox
        ranges["czrc_pairwise"] = _add_czrc_pairwise_trace(
            fig, combo_key, czrc_data, czrc_config
        )

        # Add candidate grid points within CZRC pairwise regions
        # Always add traces - visibility controlled by CZRC Grid checkbox in Layers panel
        # This includes: Tier 2 visibility boundary + hexagonal grid + first pass candidates
        grid_start_idx = len(fig.data)

        # Add cell boundaries from split clusters (if any)
        czrc_cell_wkts = data.get("czrc_cell_wkts", [])
        if czrc_cell_wkts:
            _add_czrc_cell_boundaries_trace(fig, combo_key, czrc_cell_wkts, czrc_config)

        # Add Tier 2 visibility boundary (controlled by CZRC Grid checkbox)
        if czrc_config.get("show_czrc_ilp_visibility", True):
            czrc_cluster_stats = data.get("czrc_cluster_stats", {})
            _add_czrc_ilp_visibility_trace(
                fig, combo_key, czrc_data, czrc_config, czrc_cluster_stats
            )

        # Add hexagonal candidate grid
        _add_czrc_candidate_grid_trace(fig, combo_key, czrc_data, czrc_config)

        # Add CZRC first pass candidates (black X markers)
        czrc_first_pass_candidates = data.get("czrc_first_pass_candidates", [])
        if czrc_first_pass_candidates:
            _add_first_pass_candidates_trace(
                fig, combo_key, czrc_first_pass_candidates, trace_prefix="czrc_"
            )

        # Set czrc_grid range to include all CZRC grid-related traces
        ranges["czrc_grid"] = (grid_start_idx, len(fig.data))

    # ═══════════════════════════════════════════════════════════════════════
    # 🔗 CZRC SECOND PASS VISUALIZATION (removed/added boreholes)
    # ═══════════════════════════════════════════════════════════════════════
    # Shows red removed and green added boreholes from CZRC optimization
    # Uses the same functions as border consolidation with "czrc_" prefix
    czrc_removed = data.get("czrc_removed", [])
    czrc_added = data.get("czrc_added", [])
    if czrc_removed or czrc_added:
        # CZRC removed boreholes (red) - initially hidden
        # Reuses _add_removed_traces with trace_prefix="czrc_"
        if czrc_removed:
            ranges.update(
                _add_removed_traces(
                    fig,
                    combo_key,
                    czrc_removed,
                    False,
                    max_spacing,
                    zones_gdf,
                    trace_prefix="czrc_",
                )
            )
        # CZRC added boreholes (green) - initially hidden
        # Reuses _add_added_traces with trace_prefix="czrc_"
        if czrc_added:
            ranges.update(
                _add_added_traces(
                    fig,
                    combo_key,
                    czrc_added,
                    False,
                    max_spacing,
                    zones_gdf,
                    trace_prefix="czrc_",
                )
            )

    # ═══════════════════════════════════════════════════════════════════════
    # 🔷 CZRC TEST POINTS VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    # Shows test points used in CZRC optimization (Tier 1 + Tier 2 ring)
    czrc_test_points = data.get("czrc_test_points", [])
    if czrc_test_points:
        ranges.update(
            _add_czrc_test_points_trace(
                fig,
                combo_key,
                czrc_test_points,
                is_visible=False,
            )
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 🔶 THIRD PASS (Cell-Cell CZRC) VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    # Shows cell clouds, intersections, grid, and removed/added boreholes
    # from the cell-cell CZRC third pass optimization
    third_pass_data = data.get("third_pass_data")
    if third_pass_data:
        from Gap_Analysis_EC7.config import CONFIG

        czrc_config = CONFIG.get("border_consolidation", {})

        # Add cell coverage clouds (per-cell coverage areas) - hidden by default
        ranges["third_pass_clouds"] = _add_third_pass_clouds_trace(
            fig, combo_key, third_pass_data, czrc_config
        )

        # Add cell-cell intersection regions - hidden by default
        ranges["third_pass_intersections"] = _add_third_pass_intersections_trace(
            fig, combo_key, third_pass_data, czrc_config
        )

        # Add cell-cell candidate grid
        ranges["third_pass_grid"] = _add_third_pass_grid_trace(
            fig, combo_key, third_pass_data, czrc_config
        )

        # Add third pass test points (Tier 1 filtered from First Pass + fresh Tier 2 ring)
        # Third Pass generates its own test points within cell-cell context
        third_pass_test_points = third_pass_data.get("third_pass_test_points", [])
        if third_pass_test_points:
            ranges["third_pass_test_points"] = _add_third_pass_test_points_trace(
                fig, combo_key, third_pass_test_points, is_visible=False
            )

    # Third Pass removed/added boreholes
    third_pass_removed = data.get("third_pass_removed", [])
    third_pass_added = data.get("third_pass_added", [])
    if third_pass_removed or third_pass_added:
        # Third Pass removed boreholes (red) - initially hidden
        if third_pass_removed:
            ranges.update(
                _add_removed_traces(
                    fig,
                    combo_key,
                    third_pass_removed,
                    False,
                    max_spacing,
                    zones_gdf,
                    trace_prefix="third_pass_",
                )
            )
        # Third Pass added boreholes (green) - initially hidden
        if third_pass_added:
            ranges.update(
                _add_added_traces(
                    fig,
                    combo_key,
                    third_pass_added,
                    False,
                    max_spacing,
                    zones_gdf,
                    trace_prefix="third_pass_",
                )
            )

    return ranges


def _add_proposed_traces(
    fig: go.Figure,
    combo_key: str,
    proposed: list,
    colors: Dict[str, str],
    is_visible: bool,
    max_spacing: float,
    zones_gdf: Optional["GeoDataFrame"] = None,
) -> Dict[str, Tuple[int, int]]:
    """
    Add proposed buffer and marker traces. Returns range dict.

    Args:
        fig: Plotly figure to add traces to
        combo_key: Filter combination key for trace naming
        proposed: List of {"x", "y"} borehole coordinates
        colors: Color configuration dict
        is_visible: Initial visibility state
        max_spacing: Default coverage radius (used as fallback)
        zones_gdf: Optional GeoDataFrame with per-zone max_spacing_m for variable coverage

    Returns:
        Dictionary mapping trace names to (start_idx, end_idx) tuples
    """
    ranges: Dict[str, Tuple[int, int]] = {}

    # Check if per-zone spacing is available
    has_zone_spacing = (
        zones_gdf is not None
        and not zones_gdf.empty
        and "max_spacing_m" in zones_gdf.columns
    )

    # Proposed buffer trace(s)
    start_idx = len(fig.data)

    if has_zone_spacing and proposed:
        # Per-zone coverage: import helper and build zone-specific traces
        from Gap_Analysis_EC7.visualization.plotly_traces import (
            _build_proposed_coverage_per_zone,
        )

        assert zones_gdf is not None  # Type narrowing
        per_zone_traces = _build_proposed_coverage_per_zone(
            proposed,
            zones_gdf,
            colors["buffer"],
            line_color=colors.get("line_color"),
            line_width=colors.get("line_width", 1),
        )
        for i, trace in enumerate(per_zone_traces):
            # Override name and visibility for combo-specific trace
            trace.name = f"Proposed Coverage ({combo_key})" if i == 0 else None
            trace.visible = is_visible
            trace.showlegend = False
            fig.add_trace(trace)
    else:
        # Fallback: single buffer radius
        fig.add_trace(
            build_merged_buffer_trace(
                coordinates=proposed,
                buffer_radius=max_spacing,
                name=f"Proposed Coverage ({combo_key})",
                buffer_color=colors["buffer"],
                visible=is_visible,
                show_legend=False,
                line_color=colors.get("line_color"),
                line_width=colors.get("line_width", 1),
            )
        )
    ranges["proposed_buffers"] = (start_idx, len(fig.data))

    # Proposed markers trace
    start_idx = len(fig.data)
    fig.add_trace(
        build_merged_marker_trace(
            coordinates=proposed,
            name=f"Proposed BHs ({combo_key})",
            marker_color=colors["marker"],
            marker_size=colors["marker_size"],
            marker_symbol=colors["marker_symbol"],
            visible=is_visible,
            show_legend=False,
        )
    )
    ranges["proposed_markers"] = (start_idx, len(fig.data))

    # Borehole circles trace (outline-only circles showing coverage radii)
    # Get styling from CONFIG - separate from proposed_marker to allow independent toggle
    circles_config = CONFIG.get("visualization", {}).get("borehole_circles", {})
    circle_line_color = circles_config.get("line_color", "rgba(0, 100, 255, 0.7)")
    circle_line_width = circles_config.get("line_width", 2)

    start_idx = len(fig.data)
    fig.add_trace(
        build_borehole_circles_trace(
            coordinates=proposed,
            buffer_radius=max_spacing,
            name=f"Borehole Circles ({combo_key})",
            line_color=circle_line_color,
            line_width=circle_line_width,
            visible=False,  # Hidden by default - user toggles via Layers panel
            show_legend=False,
        )
    )
    ranges["borehole_circles"] = (start_idx, len(fig.data))

    return ranges


def _add_removed_traces(
    fig: go.Figure,
    combo_key: str,
    removed: list,
    is_visible: bool,
    max_spacing: float,
    zones_gdf: Optional["GeoDataFrame"] = None,
    trace_prefix: str = "",
) -> Dict[str, Tuple[int, int]]:
    """
    Add removed borehole traces (red buffers and markers).

    These are boreholes that were proposed in first pass but removed
    during border consolidation or CZRC optimization.

    Args:
        fig: Plotly figure to add traces to
        combo_key: Filter combination key for trace naming
        removed: List of {"x", "y", "coverage_radius"} borehole coordinates
        is_visible: Initial visibility state
        max_spacing: Default coverage radius (used as fallback)
        zones_gdf: Optional GeoDataFrame with per-zone max_spacing_m
        trace_prefix: Optional prefix for trace names (e.g., "czrc_" for CZRC traces)

    Returns:
        Dictionary mapping trace names to (start_idx, end_idx) tuples
    """
    ranges: Dict[str, Tuple[int, int]] = {}

    if not removed:
        return ranges

    # Build display name with optional prefix
    display_prefix = trace_prefix.upper().replace("_", " ").strip()
    if display_prefix:
        display_prefix = f"{display_prefix} "

    print(
        f"   🔴 _add_removed_traces: Adding traces for {len(removed)} {display_prefix.lower().strip() or ''}removed boreholes"
    )

    # Get colors from config (with fallback)
    from Gap_Analysis_EC7.config import CONFIG

    removed_style = CONFIG.get("visualization", {}).get("removed_marker", {})
    removed_buffer_color = removed_style.get("buffer_color", "rgba(220, 53, 69, 0.25)")
    removed_line_color = removed_style.get("line_color", None)
    removed_line_width = removed_style.get("line_width", 1)
    removed_marker_color = removed_style.get("color", "rgba(220, 53, 69, 0.85)")
    removed_marker_size = removed_style.get("size", 8)
    removed_marker_symbol = removed_style.get("symbol", "x")

    # Check if per-zone spacing is available
    has_zone_spacing = (
        zones_gdf is not None
        and not zones_gdf.empty
        and "max_spacing_m" in zones_gdf.columns
    )

    # Build range key prefix (e.g., "czrc_" for CZRC traces)
    range_key_prefix = trace_prefix

    # Removed buffer trace(s)
    start_idx = len(fig.data)

    if has_zone_spacing and removed:
        # Per-zone coverage: import helper and build zone-specific traces
        from Gap_Analysis_EC7.visualization.plotly_traces import (
            _build_proposed_coverage_per_zone,
        )

        assert zones_gdf is not None  # Type narrowing
        per_zone_traces = _build_proposed_coverage_per_zone(
            removed,
            zones_gdf,
            removed_buffer_color,
            line_color=removed_line_color,
            line_width=removed_line_width,
        )
        for i, trace in enumerate(per_zone_traces):
            # Override name and visibility for combo-specific trace
            trace.name = (
                f"{display_prefix}Removed Coverage ({combo_key})" if i == 0 else None
            )
            trace.visible = is_visible
            trace.showlegend = False
            trace.legendgroup = f"{range_key_prefix}removed_coverage"
            fig.add_trace(trace)
    else:
        # Fallback: single buffer radius
        fig.add_trace(
            build_merged_buffer_trace(
                coordinates=removed,
                buffer_radius=max_spacing,
                name=f"{display_prefix}Removed Coverage ({combo_key})",
                buffer_color=removed_buffer_color,
                visible=is_visible,
                show_legend=False,
                line_color=removed_line_color,
                line_width=removed_line_width,
            )
        )
    ranges[f"{range_key_prefix}removed_buffers"] = (start_idx, len(fig.data))

    # Removed markers trace (red X)
    start_idx = len(fig.data)
    fig.add_trace(
        build_merged_marker_trace(
            coordinates=removed,
            name=f"{display_prefix}Removed BHs ({combo_key})",
            marker_color=removed_marker_color,
            marker_size=removed_marker_size,
            marker_symbol=removed_marker_symbol,
            visible=is_visible,
            show_legend=False,
        )
    )
    ranges[f"{range_key_prefix}removed_markers"] = (start_idx, len(fig.data))

    return ranges


def _add_added_traces(
    fig: go.Figure,
    combo_key: str,
    added: list,
    is_visible: bool,
    max_spacing: float,
    zones_gdf: Optional["GeoDataFrame"] = None,
    trace_prefix: str = "",
) -> Dict[str, Tuple[int, int]]:
    """
    Add added borehole traces (green buffers and markers).

    These are NEW boreholes added during consolidation or CZRC optimization
    at different positions from the original first-pass boreholes.

    Args:
        fig: Plotly figure to add traces to
        combo_key: Filter combination key for trace naming
        added: List of {"x", "y", "coverage_radius"} borehole coordinates
        is_visible: Initial visibility state
        max_spacing: Default coverage radius (used as fallback)
        zones_gdf: Optional GeoDataFrame with per-zone max_spacing_m
        trace_prefix: Optional prefix for trace names (e.g., "czrc_" for CZRC traces)

    Returns:
        Dictionary mapping trace names to (start_idx, end_idx) tuples
    """
    ranges: Dict[str, Tuple[int, int]] = {}

    if not added:
        return ranges

    # Build display name with optional prefix
    display_prefix = trace_prefix.upper().replace("_", " ").strip()
    if display_prefix:
        display_prefix = f"{display_prefix} "

    # Build range key prefix (e.g., "czrc_" for CZRC traces)
    range_key_prefix = trace_prefix

    print(
        f"   🟢 _add_added_traces: Adding traces for {len(added)} {display_prefix.lower().strip() or ''}added boreholes"
    )

    # Get colors from config (with fallback)
    from Gap_Analysis_EC7.config import CONFIG

    added_style = CONFIG.get("visualization", {}).get("added_marker", {})
    added_buffer_color = added_style.get("buffer_color", "rgba(40, 167, 69, 0.25)")
    added_line_color = added_style.get("line_color", None)
    added_line_width = added_style.get("line_width", 1)
    added_marker_color = added_style.get("color", "rgba(40, 167, 69, 0.85)")
    added_marker_size = added_style.get("size", 8)
    added_marker_symbol = added_style.get("symbol", "x")

    # Check if per-zone spacing is available
    has_zone_spacing = (
        zones_gdf is not None
        and not zones_gdf.empty
        and "max_spacing_m" in zones_gdf.columns
    )

    # Added buffer trace(s)
    start_idx = len(fig.data)

    if has_zone_spacing and added:
        # Per-zone coverage: import helper and build zone-specific traces
        from Gap_Analysis_EC7.visualization.plotly_traces import (
            _build_proposed_coverage_per_zone,
        )

        assert zones_gdf is not None  # Type narrowing
        per_zone_traces = _build_proposed_coverage_per_zone(
            added,
            zones_gdf,
            added_buffer_color,
            line_color=added_line_color,
            line_width=added_line_width,
        )
        for i, trace in enumerate(per_zone_traces):
            # Override name and visibility for combo-specific trace
            trace.name = (
                f"{display_prefix}Added Coverage ({combo_key})" if i == 0 else None
            )
            trace.visible = is_visible
            trace.showlegend = False
            trace.legendgroup = f"{range_key_prefix}added_coverage"
            fig.add_trace(trace)
    else:
        # Fallback: single buffer radius
        fig.add_trace(
            build_merged_buffer_trace(
                coordinates=added,
                buffer_radius=max_spacing,
                name=f"{display_prefix}Added Coverage ({combo_key})",
                buffer_color=added_buffer_color,
                visible=is_visible,
                show_legend=False,
                line_color=added_line_color,
                line_width=added_line_width,
            )
        )
    ranges[f"{range_key_prefix}added_buffers"] = (start_idx, len(fig.data))

    # Added markers trace (green X)
    start_idx = len(fig.data)
    fig.add_trace(
        build_merged_marker_trace(
            coordinates=added,
            name=f"{display_prefix}Added BHs ({combo_key})",
            marker_color=added_marker_color,
            marker_size=added_marker_size,
            marker_symbol=added_marker_symbol,
            visible=is_visible,
            show_legend=False,
        )
    )
    ranges[f"{range_key_prefix}added_markers"] = (start_idx, len(fig.data))

    return ranges


def _add_czrc_test_points_trace(
    fig: go.Figure,
    combo_key: str,
    czrc_test_points: List[Dict[str, Any]],
    is_visible: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """
    Add CZRC test points as simple markers (Tier 1 + Tier 2 ring).

    These test points show the locations used for CZRC optimization constraints.
    Tier 1 test points are within the active optimization region.
    Tier 2 ring test points are in the coverage context region.

    Args:
        fig: Plotly figure to add traces to
        combo_key: Filter combination key for trace naming
        czrc_test_points: List of {"x", "y", "zone"} test point coordinates
        is_visible: Initial visibility state (default False - hidden)

    Returns:
        Dictionary mapping trace names to (start_idx, end_idx) tuples
    """
    ranges: Dict[str, Tuple[int, int]] = {}

    if not czrc_test_points:
        return ranges

    # Separate Tier 1 and Tier 2 ring test points
    tier1_pts = [tp for tp in czrc_test_points if tp.get("zone") != "tier2_ring"]
    tier2_pts = [tp for tp in czrc_test_points if tp.get("zone") == "tier2_ring"]

    # Further split by coverage status (is_covered flag from czrc_solver)
    tier1_covered = [tp for tp in tier1_pts if tp.get("is_covered", False)]
    tier1_uncovered = [tp for tp in tier1_pts if not tp.get("is_covered", False)]
    tier2_covered = [tp for tp in tier2_pts if tp.get("is_covered", False)]
    tier2_uncovered = [tp for tp in tier2_pts if not tp.get("is_covered", False)]

    print(
        f"   🔷 _add_czrc_test_points_trace: Tier1 {len(tier1_uncovered)} uncovered + "
        f"{len(tier1_covered)} covered, Tier2 {len(tier2_uncovered)} uncovered + "
        f"{len(tier2_covered)} covered"
    )

    # Get colors from config
    from Gap_Analysis_EC7.config import CONFIG

    czrc_viz = CONFIG.get("visualization", {}).get("czrc_test_points", {})
    tier1_color = czrc_viz.get("tier1_color", "rgba(255, 0, 0, 1)")
    tier2_color = czrc_viz.get("tier2_color", "rgba(174, 0, 255, 1)")
    tier1_covered_color = czrc_viz.get("tier1_covered_color", "rgba(0, 200, 0, 1)")
    tier2_covered_color = czrc_viz.get("tier2_covered_color", "rgba(0, 180, 0, 1)")
    marker_size = czrc_viz.get("size", 4)

    start_idx = len(fig.data)

    # Tier 1 UNCOVERED test points (red - require ILP coverage)
    if tier1_uncovered:
        fig.add_trace(
            go.Scattergl(
                x=[tp["x"] for tp in tier1_uncovered],
                y=[tp["y"] for tp in tier1_uncovered],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=tier1_color,
                    symbol="circle",
                ),
                name=f"CZRC Tier1 Uncovered ({combo_key})",
                visible=is_visible,
                showlegend=False,
                legendgroup="czrc_test_points",
                hovertemplate=(
                    "<b>Tier 1 Test Point (Uncovered)</b><br>"
                    "Easting: %{x:,.0f}<br>"
                    "Northing: %{y:,.0f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Tier 1 COVERED test points (green - pre-covered by locked boreholes)
    if tier1_covered:
        fig.add_trace(
            go.Scattergl(
                x=[tp["x"] for tp in tier1_covered],
                y=[tp["y"] for tp in tier1_covered],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=tier1_covered_color,
                    symbol="circle",
                ),
                name=f"CZRC Tier1 Covered ({combo_key})",
                visible=is_visible,
                showlegend=False,
                legendgroup="czrc_test_points",
                hovertemplate=(
                    "<b>Tier 1 Test Point (Covered by Locked BH)</b><br>"
                    "Easting: %{x:,.0f}<br>"
                    "Northing: %{y:,.0f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Tier 2 ring UNCOVERED test points (purple)
    if tier2_uncovered:
        fig.add_trace(
            go.Scattergl(
                x=[tp["x"] for tp in tier2_uncovered],
                y=[tp["y"] for tp in tier2_uncovered],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=tier2_color,
                    symbol="circle",
                ),
                name=f"CZRC Tier2 Ring Uncovered ({combo_key})",
                visible=is_visible,
                showlegend=False,
                legendgroup="czrc_test_points",
                hovertemplate=(
                    "<b>Tier 2 Ring Test Point (Uncovered)</b><br>"
                    "Easting: %{x:,.0f}<br>"
                    "Northing: %{y:,.0f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Tier 2 ring COVERED test points (green)
    if tier2_covered:
        fig.add_trace(
            go.Scattergl(
                x=[tp["x"] for tp in tier2_covered],
                y=[tp["y"] for tp in tier2_covered],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=tier2_covered_color,
                    symbol="circle",
                ),
                name=f"CZRC Tier2 Ring Covered ({combo_key})",
                visible=is_visible,
                showlegend=False,
                legendgroup="czrc_test_points",
                hovertemplate=(
                    "<b>Tier 2 Ring Test Point (Covered by Locked BH)</b><br>"
                    "Easting: %{x:,.0f}<br>"
                    "Northing: %{y:,.0f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    ranges["czrc_test_points"] = (start_idx, len(fig.data))

    return ranges


# ═══════════════════════════════════════════════════════════════════════════
# 🔶 THIRD PASS (Cell-Cell CZRC) TRACE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def _add_third_pass_clouds_trace(
    fig: go.Figure,
    combo_key: str,  # noqa: ARG001 - kept for API consistency
    third_pass_data: Dict[str, Any],
    czrc_config: Dict[str, Any],
) -> Tuple[int, int]:
    """Add cell coverage cloud fills (light fill showing per-cell reachability)."""
    from shapely import wkt

    start_idx = len(fig.data)
    cell_clouds_wkt = third_pass_data.get("cell_clouds_wkt", {})
    if not cell_clouds_wkt:
        return (start_idx, len(fig.data))

    # Cell clouds use separate opacity setting for better visibility
    opacity = czrc_config.get("czrc_cell_cloud_opacity", 0.40)

    # Colors for different cells - same vibrant palette as zone clouds for visibility
    cell_colors = [
        "rgba(255, 100, 100, {opacity})",  # Red (matches zone palette)
        "rgba(100, 255, 100, {opacity})",  # Green
        "rgba(100, 100, 255, {opacity})",  # Blue
        "rgba(255, 255, 100, {opacity})",  # Yellow
        "rgba(255, 100, 255, {opacity})",  # Magenta
        "rgba(100, 255, 255, {opacity})",  # Cyan
    ]

    for cell_idx, (cell_name, cloud_wkt) in enumerate(cell_clouds_wkt.items()):
        try:
            cloud_geom = wkt.loads(cloud_wkt)
            if cloud_geom.is_empty:
                continue
            x_coords, y_coords = _extract_polygon_coords(cloud_geom)
            color_tmpl = cell_colors[cell_idx % len(cell_colors)]
            fill_color = color_tmpl.format(opacity=opacity)
            line_color = color_tmpl.format(opacity=0.5)
            fig.add_trace(
                go.Scattergl(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    fill="toself",
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=1),
                    hoverinfo="text",
                    hovertext=f"Cell Cloud: {cell_name}",
                    name=f"Cell Cloud: {cell_name}",
                    legendgroup="third_pass_clouds",
                    showlegend=False,
                    visible=False,  # Hidden by default
                )
            )
        except Exception:  # noqa: BLE001 - WKT parsing can fail
            pass

    return (start_idx, len(fig.data))


def _add_third_pass_intersections_trace(
    fig: go.Figure,
    combo_key: str,  # noqa: ARG001 - kept for API consistency
    third_pass_data: Dict[str, Any],
    czrc_config: Dict[str, Any],
) -> Tuple[int, int]:
    """Add cell-cell intersection regions (overlap between adjacent cells)."""
    from shapely import wkt

    start_idx = len(fig.data)
    cell_intersections_wkt = third_pass_data.get("cell_intersections_wkt", {})
    if not cell_intersections_wkt:
        return (start_idx, len(fig.data))

    # Use same cyan color as zone pairwise for consistency
    color = czrc_config.get("czrc_pairwise_color", "cyan")
    opacity = czrc_config.get("czrc_pairwise_opacity", 0.3)
    line_width = czrc_config.get("czrc_line_width", 2)

    for pair_key, region_wkt in cell_intersections_wkt.items():
        try:
            region_geom = wkt.loads(region_wkt)
            if region_geom.is_empty:
                continue
            x_coords, y_coords = _extract_polygon_coords(region_geom)
            cells = pair_key.replace("_", " + ")
            fig.add_trace(
                go.Scattergl(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    fill="toself",
                    fillcolor=f"rgba(0, 255, 255, {opacity})",  # Cyan - same as zone pairwise
                    line=dict(color=color, width=line_width, dash="dot"),
                    hoverinfo="text",
                    hovertext=f"Cell Intersection: {cells}",
                    name=f"Cell CZRC: {cells}",
                    legendgroup="third_pass_intersections",
                    showlegend=False,
                    visible=False,  # Hidden by default
                )
            )
        except Exception:  # noqa: BLE001 - WKT parsing can fail
            pass

    return (start_idx, len(fig.data))


def _add_third_pass_grid_trace(
    fig: go.Figure,
    combo_key: str,  # noqa: ARG001 - kept for API consistency
    third_pass_data: Dict[str, Any],
    czrc_config: Dict[str, Any],  # noqa: ARG001 - kept for API consistency
) -> Tuple[int, int]:
    """
    Add hexagonal candidate grid for cell-cell CZRC regions (third pass).

    Generates hexagon grid on-the-fly from cell_intersections_wkt data,
    similar to how _add_czrc_candidate_grid_trace works for second pass.

    The grid is generated within the unified cell-cell intersection regions
    (buffered by tier1 multiplier × spacing for consistency with solver).
    """
    from shapely import wkt as shapely_wkt
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from Gap_Analysis_EC7.config import CONFIG
    from Gap_Analysis_EC7.solvers.optimization_geometry import (
        generate_hexagon_grid_polygons,
    )
    from Gap_Analysis_EC7.visualization.plotly_traces import build_hexagon_grid_trace

    start_idx = len(fig.data)

    # Get cell intersection WKTs (these are the CZRC regions for third pass)
    cell_intersections = third_pass_data.get("cell_intersections_wkt", {})
    if not cell_intersections:
        return (start_idx, len(fig.data))

    # Parse all intersection geometries and unify them
    intersection_geoms = []
    for _pair_key, wkt_str in cell_intersections.items():
        try:
            geom = shapely_wkt.loads(wkt_str)
            if not geom.is_empty:
                intersection_geoms.append(geom)
        except Exception:  # noqa: BLE001
            pass

    if not intersection_geoms:
        return (start_idx, len(fig.data))

    # Unify all intersection regions
    unified_czrc = unary_union(intersection_geoms)
    if unified_czrc.is_empty:
        return (start_idx, len(fig.data))

    # Get spacing and tier1 multiplier from config
    czrc_cfg = CONFIG.get("czrc_solver", {})
    tier1_mult = czrc_cfg.get("tier1_rmax_multiplier", 1.0)

    # Estimate spacing from cell_intersections (use first available)
    # For cells, we use the cluster spacing which is typically the max zone spacing
    ilp_cfg = CONFIG.get("ilp_solver", {})
    spacing_mult = ilp_cfg.get("candidate_spacing_mult", 0.5)

    # Get default spacing from the third pass data or config
    default_spacing = czrc_cfg.get("default_spacing", 100.0)
    grid_spacing = default_spacing * spacing_mult

    # Buffer the unified CZRC region to Tier 1 (CZRC + tier1_mult × spacing)
    tier1_region = unified_czrc.buffer(tier1_mult * default_spacing)
    if tier1_region.is_empty:
        return (start_idx, len(fig.data))

    # Generate hexagon grid within Tier 1
    search_bounds = tier1_region.bounds
    grid_origin = (search_bounds[0], search_bounds[1])

    hexagons: List[Polygon] = generate_hexagon_grid_polygons(
        bounds=search_bounds,
        grid_spacing=grid_spacing,
        clip_geometry=tier1_region,
        origin=grid_origin,
        logger=None,
    )

    if not hexagons:
        return (start_idx, len(fig.data))

    # Get third pass grid styling (use distinct color from second pass)
    third_pass_style = CONFIG.get("visualization", {}).get("third_pass_grid", {})
    grid_color = third_pass_style.get("color", "rgba(128, 0, 128, 0.55)")  # Purple
    grid_line_width = third_pass_style.get("line_width", 0.5)

    # Build unified hexagon grid trace
    hexgrid_trace = build_hexagon_grid_trace(
        hexagon_polygons=hexagons,
        grid_color=grid_color,
        grid_line_width=grid_line_width,
        visible=False,  # Hidden by default, controlled by Third Pass Grid checkbox
        name=f"Third Pass Grid ({len(hexagons)} cells)",
    )

    if hexgrid_trace:
        fig.add_trace(hexgrid_trace)

    # Add Tier 2 visibility boundary line (same as Second Pass Grid has)
    tier2_mult = czrc_cfg.get("tier2_rmax_multiplier", 2.0)
    tier2_region = unified_czrc.buffer(tier2_mult * default_spacing)

    if tier2_region is not None and not tier2_region.is_empty:
        # Get Tier 2 styling from config (same as CZRC uses)
        vis_cfg = CONFIG.get("visualization", {})
        ilp_style = vis_cfg.get("czrc_ilp_visibility", {})
        tier2_color = ilp_style.get("tier2_color", "rgba(138, 43, 226, 0.8)")
        tier2_dash = ilp_style.get("tier2_dash", "longdash")
        tier2_line_width = ilp_style.get("line_width", 2)

        # Format multiplier for legend (remove trailing zeros)
        mult_str = f"{tier2_mult:g}"
        
        # Parse per-cluster Tier 2 boundaries for individual tooltips
        # cell_intersections keys are like "cluster_{N}_{pair_key}" or "Cell_{i}_Cell_{j}"
        cluster_intersections: Dict[str, List] = {}  # cluster_num -> list of geometries
        import re
        for pair_key, wkt_str in cell_intersections.items():
            try:
                geom = shapely_wkt.loads(wkt_str)
                if geom.is_empty:
                    continue
                # Extract cluster number from key (e.g., "cluster_1_Cell_0_Cell_1")
                cluster_match = re.match(r"cluster_(\d+)_", pair_key)
                if cluster_match:
                    cluster_num = int(cluster_match.group(1))
                else:
                    # Key is just "Cell_{i}_Cell_{j}" - assign to cluster 1
                    cluster_num = 1
                if cluster_num not in cluster_intersections:
                    cluster_intersections[cluster_num] = []
                cluster_intersections[cluster_num].append(geom)
            except Exception:  # noqa: BLE001
                pass
        
        # Create per-cluster Tier 2 boundaries with tooltips
        for cluster_num, geoms in sorted(cluster_intersections.items()):
            cluster_czrc = unary_union(geoms)
            cluster_tier2 = cluster_czrc.buffer(tier2_mult * default_spacing)
            if cluster_tier2.is_empty:
                continue
            # Create tooltip text matching log file naming (1-based display)
            tooltip = f"Cluster{cluster_num}_Tier2"
            _add_boundary_trace(
                fig=fig,
                geometry=cluster_tier2,
                color=tier2_color,
                dash=tier2_dash,
                line_width=tier2_line_width,
                name=f"Third Pass Grid Tier 2 ({mult_str}× R_max)",
                legendgroup="third_pass_tier2",
                visible=False,  # Hidden by default, controlled by Third Pass Grid checkbox
                hovertext=tooltip,
            )

    # Add existing boreholes (Second Pass output) as grey markers
    # These are the boreholes from Second Pass OUTPUT (survivors + added) that fall
    # within Third Pass Tier 1 area - the input to Third Pass re-optimization
    existing_boreholes = third_pass_data.get("third_pass_existing_boreholes", [])
    if existing_boreholes:
        x_coords = [bh.get("x", bh.get("easting", 0)) for bh in existing_boreholes]
        y_coords = [bh.get("y", bh.get("northing", 0)) for bh in existing_boreholes]

        # Use same marker styling as Second Pass first_pass_candidates (black X markers)
        marker_config = CONFIG.get("visualization", {}).get(
            "first_pass_candidate_marker",
            {"size": 10, "color": "black", "symbol": "x", "line_width": 2},
        )
        marker_color = _normalize_color_for_scattergl(
            marker_config.get("color", "black")
        )

        fig.add_trace(
            go.Scattergl(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    symbol=marker_config.get("symbol", "x"),
                    size=marker_config.get("size", 10),
                    color=marker_color,
                    line=dict(
                        width=marker_config.get("line_width", 2),
                        color=marker_color,
                    ),
                ),
                name=f"Second Pass Boreholes ({len(existing_boreholes)})",
                legendgroup="third_pass_grid",
                showlegend=False,
                visible=False,  # Same visibility as grid layer
                hovertemplate="Second Pass Borehole<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>",
            )
        )

    return (start_idx, len(fig.data))


def _add_third_pass_test_points_trace(
    fig: go.Figure,
    combo_key: str,  # noqa: ARG001 - kept for API consistency
    czrc_test_points: List[Dict[str, Any]],
    is_visible: bool = False,
) -> Tuple[int, int]:
    """
    Add Third Pass test points as markers (Tier 1 + Tier 2 ring).

    Third Pass uses First Pass test points filtered to the Tier 1 area
    (cell-cell coverage cloud intersection), plus fresh Tier 2 ring test
    points generated with sparse spacing (3x multiplier) for the ring area.

    Test points are split by coverage status:
    - Covered (green): Pre-covered by locked boreholes, removed from ILP
    - Uncovered (red/purple): Need to be satisfied by ILP

    Args:
        fig: Plotly figure to add traces to
        combo_key: Filter combination key for trace naming
        czrc_test_points: List of {"x", "y", "zone", "is_covered"} test point coordinates
        is_visible: Initial visibility state (default False - hidden)

    Returns:
        Tuple of (start_idx, end_idx) for the test points traces
    """
    from Gap_Analysis_EC7.config import CONFIG

    ranges: Dict[str, Tuple[int, int]] = {}
    start_idx = len(fig.data)

    if not czrc_test_points:
        return (start_idx, start_idx)

    # Separate Tier 1 and Tier 2 ring test points (same logic as Second Pass)
    tier1_pts = [tp for tp in czrc_test_points if tp.get("zone") != "tier2_ring"]
    tier2_pts = [tp for tp in czrc_test_points if tp.get("zone") == "tier2_ring"]

    # Further split by coverage status (is_covered flag from czrc_solver)
    tier1_covered = [tp for tp in tier1_pts if tp.get("is_covered", False)]
    tier1_uncovered = [tp for tp in tier1_pts if not tp.get("is_covered", False)]
    tier2_covered = [tp for tp in tier2_pts if tp.get("is_covered", False)]
    tier2_uncovered = [tp for tp in tier2_pts if not tp.get("is_covered", False)]

    print(
        f"   🔶 _add_third_pass_test_points_trace: Tier1 {len(tier1_uncovered)} uncovered + "
        f"{len(tier1_covered)} covered, Tier2 {len(tier2_uncovered)} uncovered + "
        f"{len(tier2_covered)} covered"
    )

    # Get colors from config
    czrc_viz = CONFIG.get("visualization", {}).get("czrc_test_points", {})
    tier1_color = czrc_viz.get("tier1_color", "rgba(255, 0, 0, 1)")
    tier2_color = czrc_viz.get("tier2_color", "rgba(174, 0, 255, 1)")
    tier1_covered_color = czrc_viz.get("tier1_covered_color", "rgba(0, 200, 0, 1)")
    tier2_covered_color = czrc_viz.get("tier2_covered_color", "rgba(0, 180, 0, 1)")
    marker_size = czrc_viz.get("size", 4)

    # Tier 1 UNCOVERED test points (red - require ILP coverage)
    if tier1_uncovered:
        fig.add_trace(
            go.Scattergl(
                x=[tp["x"] for tp in tier1_uncovered],
                y=[tp["y"] for tp in tier1_uncovered],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=tier1_color,
                    symbol="circle",
                ),
                name=f"Third Pass Tier1 Uncovered ({combo_key})",
                visible=is_visible,
                showlegend=False,
                legendgroup="third_pass_test_points",
                hovertemplate=(
                    "<b>Tier 1 Test Point (Uncovered)</b><br>"
                    "Easting: %{x:,.0f}<br>"
                    "Northing: %{y:,.0f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Tier 1 COVERED test points (green - pre-covered by locked boreholes)
    if tier1_covered:
        fig.add_trace(
            go.Scattergl(
                x=[tp["x"] for tp in tier1_covered],
                y=[tp["y"] for tp in tier1_covered],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=tier1_covered_color,
                    symbol="circle",
                ),
                name=f"Third Pass Tier1 Covered ({combo_key})",
                visible=is_visible,
                showlegend=False,
                legendgroup="third_pass_test_points",
                hovertemplate=(
                    "<b>Tier 1 Test Point (Covered by Locked BH)</b><br>"
                    "Easting: %{x:,.0f}<br>"
                    "Northing: %{y:,.0f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Tier 2 ring UNCOVERED test points (purple)
    if tier2_uncovered:
        fig.add_trace(
            go.Scattergl(
                x=[tp["x"] for tp in tier2_uncovered],
                y=[tp["y"] for tp in tier2_uncovered],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=tier2_color,
                    symbol="circle",
                ),
                name=f"Third Pass Tier2 Ring Uncovered ({combo_key})",
                visible=is_visible,
                showlegend=False,
                legendgroup="third_pass_test_points",
                hovertemplate=(
                    "<b>Tier 2 Ring Test Point (Uncovered)</b><br>"
                    "Easting: %{x:,.0f}<br>"
                    "Northing: %{y:,.0f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Tier 2 ring COVERED test points (green)
    if tier2_covered:
        fig.add_trace(
            go.Scattergl(
                x=[tp["x"] for tp in tier2_covered],
                y=[tp["y"] for tp in tier2_covered],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=tier2_covered_color,
                    symbol="circle",
                ),
                name=f"Third Pass Tier2 Ring Covered ({combo_key})",
                visible=is_visible,
                showlegend=False,
                legendgroup="third_pass_test_points",
                hovertemplate=(
                    "<b>Tier 2 Ring Test Point (Covered by Locked BH)</b><br>"
                    "Easting: %{x:,.0f}<br>"
                    "Northing: %{y:,.0f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    return (start_idx, len(fig.data))


def _add_hexgrid_trace(
    fig: go.Figure,
    combo_key: str,
    gaps_geom: Optional[BaseGeometry],
    max_spacing: float,
    grid_spacing: float,
    hexgrid_config: Optional[Dict[str, Any]] = None,
    zones_gdf: Optional["GeoDataFrame"] = None,
) -> Tuple[int, int]:
    """
    Add hexagon grid trace for a combo. Returns (start_idx, end_idx).

    If zones_gdf is provided with per-zone spacing (max_spacing_m column),
    generates separate grids for each zone with zone-specific spacing.
    Otherwise falls back to uniform grid_spacing.
    """
    from Gap_Analysis_EC7.solvers.optimization_geometry import (
        generate_hexagon_grid_polygons,
    )
    from Gap_Analysis_EC7.config import CONFIG

    start_idx = len(fig.data)
    hexagon_polygons = []

    if gaps_geom is not None and not gaps_geom.is_empty:
        # Check if we have per-zone spacing
        has_zone_spacing = (
            zones_gdf is not None
            and not zones_gdf.empty
            and "max_spacing_m" in zones_gdf.columns
        )

        if has_zone_spacing:
            # Per-zone hexagon grids
            # Note: zones_gdf is guaranteed non-None here due to has_zone_spacing check
            assert zones_gdf is not None  # Type narrowing for Pylance
            candidate_mult = CONFIG.get("ilp_solver", {}).get(
                "candidate_spacing_mult", 0.5
            )
            zone_name_col = None
            for col in ["zone_name", "Name"]:
                if col in zones_gdf.columns:
                    zone_name_col = col
                    break

            for _, zone_row in zones_gdf.iterrows():
                zone_geom = zone_row.geometry
                if zone_geom is None or zone_geom.is_empty:
                    continue

                # Get zone-specific spacing
                zone_max_spacing = float(zone_row["max_spacing_m"])
                zone_grid_spacing = zone_max_spacing * candidate_mult

                # Intersect gaps with zone
                try:
                    zone_gaps = gaps_geom.intersection(zone_geom)
                except Exception:
                    continue

                if zone_gaps.is_empty:
                    continue

                # Generate grid for this zone's gaps
                zone_buffer = zone_gaps.buffer(zone_max_spacing)
                zone_bounds = zone_buffer.bounds
                zone_hexagons = generate_hexagon_grid_polygons(
                    bounds=zone_bounds,
                    grid_spacing=zone_grid_spacing,
                    clip_geometry=zone_buffer,
                    logger=None,
                )
                hexagon_polygons.extend(zone_hexagons)
        else:
            # Fallback: uniform grid spacing
            gap_buffer = gaps_geom.buffer(max_spacing)
            gap_bounds = gap_buffer.bounds
            hexagon_polygons = generate_hexagon_grid_polygons(
                bounds=gap_bounds,
                grid_spacing=grid_spacing,
                clip_geometry=gap_buffer,
                logger=None,
            )

    hg_config = hexgrid_config or {}
    grid_color = hg_config.get("color", "rgba(100, 100, 255, 0.4)")
    grid_line_width = hg_config.get("line_width", 1.0)

    hexgrid_trace = build_hexagon_grid_trace(
        hexagon_polygons=hexagon_polygons,
        grid_color=grid_color,
        grid_line_width=grid_line_width,
        visible=False,
        name=f"Candidate Grid ({combo_key})",
    )

    if hexgrid_trace is not None:
        fig.add_trace(hexgrid_trace)
    else:
        fig.add_trace(
            go.Scattergl(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=grid_color, width=grid_line_width),
                hoverinfo="skip",
                name=f"Candidate Grid ({combo_key})",
                legendgroup="hexgrid",
                showlegend=False,
                visible=False,
            )
        )

    return (start_idx, len(fig.data))


def _add_second_pass_grid_trace(
    fig: go.Figure,
    combo_key: str,
    second_pass_grid: Union[Dict[str, Any], List[Dict[str, Any]]],
    hexgrid_config: Optional[Dict[str, Any]] = None,  # noqa: ARG001 - deprecated
) -> Tuple[int, int]:
    """
    Add second pass candidate grid trace for a combo. Returns (start_idx, end_idx).

    This shows the unified hexagonal candidate grid used during consolidation.
    The grid is generated from the buffer_polygon_wkt to match first-pass grid style.

    IMPORTANT - Grid vs ILP Candidates Distinction:
        The ILP optimization uses BOTH fresh grid points AND first-pass border
        boreholes as candidates. However, the visualization shows ONLY the
        unified fresh grid (ignoring first-pass borehole positions) to avoid
        overlapping hexagon artifacts.

        Why: First-pass boreholes may not align with the hexagonal lattice,
        and drawing hexagons at each candidate point (including non-grid-aligned
        positions) creates visual overlap. The unified grid accurately represents
        the search space for fresh candidates.

    Data Format (new):
        {
            "candidates": [{"x", "y", "coverage_radius"}, ...],  # All candidates
            "buffer_polygon_wkt": "POLYGON(...)",  # Geometry for unified grid
            "candidate_spacing": 50.0,  # Grid spacing used
            "min_spacing": 100.0  # Minimum zone spacing
        }

    Fallback (legacy): List of {"x", "y", "coverage_radius"} candidate coords

    Args:
        fig: Plotly figure to add traces to
        combo_key: Filter combination key for trace naming
        second_pass_grid: Either dict with geometry data or legacy list of candidates
        hexgrid_config: DEPRECATED - now uses CONFIG["visualization"]["second_pass_grid"]

    Returns:
        Tuple of (start_idx, end_idx) for trace range
    """
    from Gap_Analysis_EC7.config import CONFIG
    from Gap_Analysis_EC7.solvers.optimization_geometry import (
        generate_hexagon_grid_polygons,
    )
    from shapely import wkt

    # Get second pass grid styling from config
    second_pass_style = CONFIG.get("visualization", {}).get("second_pass_grid", {})
    grid_color = second_pass_style.get("color", "rgba(26, 26, 60, 1)")
    grid_line_width = second_pass_style.get("line_width", 0.5)

    start_idx = len(fig.data)

    # Handle empty input
    if not second_pass_grid:
        fig.add_trace(
            go.Scattergl(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=grid_color, width=grid_line_width),
                hoverinfo="skip",
                name=f"Second Pass Grid ({combo_key})",
                legendgroup="second_pass_grid",
                showlegend=False,
                visible=False,
            )
        )
        return (start_idx, len(fig.data))

    # Parse input - new format is dict with geometry, legacy is list
    if isinstance(second_pass_grid, dict):
        buffer_polygon_wkt = second_pass_grid.get("buffer_polygon_wkt")
        candidate_spacing = second_pass_grid.get("candidate_spacing", 50.0)
        candidates = second_pass_grid.get("candidates", [])
    else:
        # Legacy format: list of candidate coords
        buffer_polygon_wkt = None
        candidates = second_pass_grid
        candidate_spacing = candidates[0].get("coverage_radius", 100.0) * 0.5

    hexagon_polygons = []

    # Prefer unified grid from buffer polygon if available
    if buffer_polygon_wkt:
        try:
            buffer_polygon = wkt.loads(buffer_polygon_wkt)
            if buffer_polygon and not buffer_polygon.is_empty:
                buffer_bounds = buffer_polygon.bounds
                hexagon_polygons = generate_hexagon_grid_polygons(
                    bounds=buffer_bounds,
                    grid_spacing=candidate_spacing,
                    clip_geometry=buffer_polygon,
                    logger=None,
                )
        except Exception:
            pass  # Fall back to candidate-based approach

    # Fallback: create hexagons at each candidate point (legacy behavior)
    if not hexagon_polygons and candidates:
        coverage_radius = candidates[0].get("coverage_radius", 100.0)
        for candidate in candidates:
            x, y = candidate["x"], candidate["y"]
            radius = candidate.get("coverage_radius", coverage_radius) * 0.5
            hexagon = _create_hexagon(x, y, radius)
            hexagon_polygons.append(hexagon)

    hexgrid_trace = build_hexagon_grid_trace(
        hexagon_polygons=hexagon_polygons,
        grid_color=grid_color,
        grid_line_width=grid_line_width,
        visible=False,
        name=f"Second Pass Grid ({combo_key})",
    )

    if hexgrid_trace is not None:
        fig.add_trace(hexgrid_trace)
    else:
        fig.add_trace(
            go.Scattergl(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=grid_color, width=grid_line_width),
                hoverinfo="skip",
                name=f"Second Pass Grid ({combo_key})",
                legendgroup="second_pass_grid",
                showlegend=False,
                visible=False,
            )
        )

    return (start_idx, len(fig.data))


def _add_buffer_zone_outline_trace(
    fig: go.Figure,
    combo_key: str,
    second_pass_grid: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> Tuple[int, int]:
    """
    Add buffer zone outline trace for debugging purposes. Returns (start_idx, end_idx).

    Shows the buffer zone polygon as a magenta dashed outline on the map.
    This is controlled by CONFIG["border_consolidation"]["show_buffer_zone_outline"].

    Args:
        fig: Plotly figure to add traces to
        combo_key: Filter combination key for trace naming
        second_pass_grid: Dict containing buffer_polygon_wkt (or legacy list, skipped)

    Returns:
        Tuple of (start_idx, end_idx) for trace range
    """
    from shapely import wkt

    start_idx = len(fig.data)

    # Only works with new dict format containing buffer_polygon_wkt
    if not isinstance(second_pass_grid, dict):
        return (start_idx, len(fig.data))

    buffer_polygon_wkt = second_pass_grid.get("buffer_polygon_wkt")
    if not buffer_polygon_wkt:
        return (start_idx, len(fig.data))

    try:
        buffer_polygon = wkt.loads(buffer_polygon_wkt)
        if buffer_polygon.is_empty:
            return (start_idx, len(fig.data))

        # Extract polygon exterior coordinates
        x_coords = []
        y_coords = []

        if buffer_polygon.geom_type == "Polygon":
            coords = list(buffer_polygon.exterior.coords)
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
        elif buffer_polygon.geom_type == "MultiPolygon":
            for poly in buffer_polygon.geoms:
                coords = list(poly.exterior.coords)
                x_coords.extend([c[0] for c in coords])
                y_coords.extend([c[1] for c in coords])
                x_coords.append(None)  # Separator between polygons
                y_coords.append(None)

        fig.add_trace(
            go.Scattergl(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(color="magenta", width=2, dash="dash"),
                hoverinfo="skip",
                name=f"Buffer Zone Outline ({combo_key})",
                legendgroup="buffer_zone_outline",
                showlegend=False,
                visible=True,  # Visible by default for debugging
            )
        )

    except Exception:
        pass  # Silently fail if WKT parsing fails

    return (start_idx, len(fig.data))


def _add_split_regions_trace(
    fig: go.Figure,
    combo_key: str,
    second_pass_grid: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> Tuple[int, int]:
    """
    Add split consolidation region outlines for debugging. Returns (start_idx, end_idx).

    Shows each independent consolidation region as a colored dashed outline.
    Regions separated by >2× max spacing are solved with separate ILPs.
    Controlled by CONFIG["border_consolidation"]["show_split_regions"].

    Args:
        fig: Plotly figure to add traces to
        combo_key: Filter combination key for trace naming
        second_pass_grid: Dict containing split_region_wkts (list of WKT strings)

    Returns:
        Tuple of (start_idx, end_idx) for trace range
    """
    from shapely import wkt

    # Distinct colors for up to 8 regions
    region_colors = [
        "cyan",
        "lime",
        "yellow",
        "orange",
        "red",
        "violet",
        "blue",
        "green",
    ]

    start_idx = len(fig.data)

    # Only works with new dict format containing split_region_wkts
    if not isinstance(second_pass_grid, dict):
        return (start_idx, len(fig.data))

    split_region_wkts = second_pass_grid.get("split_region_wkts", [])
    if not split_region_wkts:
        return (start_idx, len(fig.data))

    for region_idx, region_wkt in enumerate(split_region_wkts):
        try:
            region_geom = wkt.loads(region_wkt)
            if region_geom.is_empty:
                continue

            # Extract polygon exterior coordinates
            x_coords = []
            y_coords = []

            if region_geom.geom_type == "Polygon":
                coords = list(region_geom.exterior.coords)
                x_coords = [c[0] for c in coords]
                y_coords = [c[1] for c in coords]
            elif region_geom.geom_type == "MultiPolygon":
                for poly in region_geom.geoms:
                    coords = list(poly.exterior.coords)
                    x_coords.extend([c[0] for c in coords])
                    y_coords.extend([c[1] for c in coords])
                    x_coords.append(None)  # Separator between polygons
                    y_coords.append(None)

            color = region_colors[region_idx % len(region_colors)]

            fig.add_trace(
                go.Scattergl(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    line=dict(color=color, width=3, dash="dot"),
                    hoverinfo="text",
                    hovertext=f"Region {region_idx + 1}",
                    name=f"Split Region {region_idx + 1} ({combo_key})",
                    legendgroup="split_regions",
                    showlegend=False,
                    visible=True,  # Visible by default for debugging
                )
            )

        except Exception:
            pass  # Silently fail if WKT parsing fails

    return (start_idx, len(fig.data))


# ═══════════════════════════════════════════════════════════════════════════
# 🌐 CZRC (CROSS-ZONE REACHABILITY CONSOLIDATION) VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def _extract_polygon_coords(geom: Any) -> Tuple[List[float], List[float]]:
    """Extract x,y coordinate lists from Polygon/MultiPolygon for Scattergl."""
    x_coords: List[float] = []
    y_coords: List[float] = []
    if geom.geom_type == "Polygon":
        coords = list(geom.exterior.coords)
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            coords = list(poly.exterior.coords)
            x_coords.extend([c[0] for c in coords])
            y_coords.extend([c[1] for c in coords])
            x_coords.append(None)
            y_coords.append(None)
    return x_coords, y_coords


# Zone cloud color palette (6 distinct colors)
_ZONE_CLOUD_COLORS = [
    "rgba(255, 100, 100, {opacity})",  # Red
    "rgba(100, 255, 100, {opacity})",  # Green
    "rgba(100, 100, 255, {opacity})",  # Blue
    "rgba(255, 255, 100, {opacity})",  # Yellow
    "rgba(255, 100, 255, {opacity})",  # Magenta
    "rgba(100, 255, 255, {opacity})",  # Cyan
]


def _add_czrc_coverage_clouds_trace(
    fig: go.Figure,
    combo_key: str,  # noqa: ARG001 - kept for API consistency
    czrc_data: Dict[str, Any],
    czrc_config: Dict[str, Any],
) -> Tuple[int, int]:
    """Add coverage cloud fills for each zone (light fill showing reachability)."""
    from shapely import wkt

    start_idx = len(fig.data)
    coverage_clouds_wkt = czrc_data.get("coverage_clouds_wkt", {})
    if not coverage_clouds_wkt:
        return (start_idx, len(fig.data))

    opacity = czrc_config.get("czrc_cloud_opacity", 0.15)

    for zone_idx, (zone_name, cloud_wkt) in enumerate(coverage_clouds_wkt.items()):
        try:
            cloud_geom = wkt.loads(cloud_wkt)
            if cloud_geom.is_empty:
                continue
            x_coords, y_coords = _extract_polygon_coords(cloud_geom)
            color_tmpl = _ZONE_CLOUD_COLORS[zone_idx % len(_ZONE_CLOUD_COLORS)]
            fill_color = color_tmpl.format(opacity=opacity)
            line_color = color_tmpl.format(opacity=0.5)
            fig.add_trace(
                go.Scattergl(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    fill="toself",
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=1),
                    hoverinfo="text",
                    hovertext=f"Coverage Cloud: {zone_name}",
                    name=f"CZRC Cloud: {zone_name}",
                    legendgroup="czrc_clouds",
                    showlegend=False,
                    visible=False,  # Hidden by default, controlled by Zone Overlap checkbox
                )
            )
        except Exception:  # noqa: BLE001 - WKT parsing can fail
            pass

    return (start_idx, len(fig.data))


def _add_czrc_pairwise_trace(
    fig: go.Figure,
    combo_key: str,  # noqa: ARG001 - kept for API consistency
    czrc_data: Dict[str, Any],
    czrc_config: Dict[str, Any],
) -> Tuple[int, int]:
    """Add pairwise intersection regions (2 zones' clouds overlap)."""
    from shapely import wkt

    start_idx = len(fig.data)
    pairwise_wkts = czrc_data.get("pairwise_wkts", {})
    if not pairwise_wkts:
        return (start_idx, len(fig.data))

    color = czrc_config.get("czrc_pairwise_color", "cyan")
    opacity = czrc_config.get("czrc_pairwise_opacity", 0.3)
    line_width = czrc_config.get("czrc_line_width", 2)

    for pair_key, region_wkt in pairwise_wkts.items():
        try:
            region_geom = wkt.loads(region_wkt)
            if region_geom.is_empty:
                continue
            x_coords, y_coords = _extract_polygon_coords(region_geom)
            zones = pair_key.replace("_", " + ")
            fig.add_trace(
                go.Scattergl(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    fill="toself",
                    fillcolor=f"rgba(0, 255, 255, {opacity})",
                    line=dict(color=color, width=line_width, dash="dash"),
                    hoverinfo="text",
                    hovertext=f"CZRC Pairwise: {zones}",
                    name=f"CZRC: {zones}",
                    legendgroup="czrc_pairwise",
                    showlegend=False,
                    visible=False,  # Hidden by default, controlled by Zone Overlap checkbox
                )
            )
        except Exception:  # noqa: BLE001 - WKT parsing can fail
            pass

    return (start_idx, len(fig.data))


def _add_czrc_cell_boundaries_trace(
    fig: go.Figure,
    combo_key: str,  # noqa: ARG001 - kept for API consistency
    cell_wkts: List[str],
    czrc_config: Dict[str, Any],
) -> Tuple[int, int]:
    """
    Add cell boundary traces from split CZRC clusters.

    When large CZRC regions are split into grid cells for ILP solving,
    this function visualizes the cell boundaries to show how the region
    was decomposed.

    Args:
        fig: Plotly figure to add traces to
        combo_key: Filter combination key (for trace naming)
        cell_wkts: List of WKT strings for cell geometries
        czrc_config: CZRC configuration dict

    Returns:
        (start_idx, end_idx) tuple for trace range
    """
    from shapely import wkt

    start_idx = len(fig.data)
    if not cell_wkts:
        return (start_idx, len(fig.data))

    # Use a distinctive color for cell boundaries - orange dashed
    line_width = czrc_config.get("czrc_line_width", 2)

    for i, cell_wkt in enumerate(cell_wkts):
        try:
            cell_geom = wkt.loads(cell_wkt)
            if cell_geom.is_empty:
                continue
            x_coords, y_coords = _extract_polygon_coords(cell_geom)
            fig.add_trace(
                go.Scattergl(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    fill="none",  # No fill, just boundary
                    line=dict(color="orange", width=line_width + 1, dash="dot"),
                    hoverinfo="text",
                    hovertext=f"CZRC Cell {i + 1} ({cell_geom.area / 1e6:.2f} km²)",
                    name=f"Cell {i + 1}",
                    legendgroup="czrc_cells",
                    showlegend=False,
                    visible=False,  # Hidden by default, controlled by CZRC Grid checkbox
                )
            )
        except Exception:  # noqa: BLE001 - WKT parsing can fail
            pass

    return (start_idx, len(fig.data))


def _add_czrc_candidate_grid_trace(
    fig: go.Figure,
    combo_key: str,  # noqa: ARG001 - kept for API consistency
    czrc_data: Dict[str, Any],
    czrc_config: Dict[str, Any],  # noqa: ARG001 - kept for API consistency
) -> Tuple[int, int]:
    """
    Add hexagonal candidate grid within CZRC Tier 1 regions (CZRC + 1×R_max).

    Per CZRC ILP architecture:
    - Test points and candidate grid are generated in Tier 1 (CZRC + 1×R_max)
    - NOT in the original CZRC region
    - NOT in Tier 2 (which only contains locked boreholes)

    Unified Clusters:
    -----------------
    Uses the same clustering algorithm as the CZRC solver to group overlapping
    pairwise regions. This ensures visualization hexagons exactly match the
    ILP candidate grid used during optimization.

    Algorithm:
    1. Import _group_overlapping_pairs from czrc_solver
    2. Generate ONE hexagonal grid per unified cluster
    3. Grid origin matches solver's _prepare_candidates_for_ilp()
    """
    from shapely.geometry import Polygon

    from Gap_Analysis_EC7.config import CONFIG
    from Gap_Analysis_EC7.solvers.czrc_solver import _group_overlapping_pairs
    from Gap_Analysis_EC7.solvers.optimization_geometry import (
        generate_hexagon_grid_polygons,
    )
    from Gap_Analysis_EC7.visualization.plotly_traces import build_hexagon_grid_trace

    start_idx = len(fig.data)
    pairwise_wkts = czrc_data.get("pairwise_wkts", {})
    if not pairwise_wkts:
        return (start_idx, len(fig.data))

    # Get zone spacings for clustering
    zone_spacings = czrc_data.get("zone_spacings", {})

    # Use same tier1 multiplier as solver (default 1.0)
    czrc_cfg = CONFIG.get("czrc_solver", {})
    tier1_mult = czrc_cfg.get("tier1_rmax_multiplier", 1.0)

    # Group overlapping pairs using SAME algorithm as solver
    clusters = _group_overlapping_pairs(pairwise_wkts, zone_spacings, tier1_mult)
    if not clusters:
        return (start_idx, len(fig.data))

    # Compute grid spacing: min_zone_spacing * candidate_spacing_mult
    ilp_cfg = CONFIG.get("ilp_solver", {})
    spacing_mult = ilp_cfg.get("candidate_spacing_mult", 0.5)
    stats = czrc_data.get("stats", {})
    min_spacing = stats.get("min_zone_spacing", 100.0)
    grid_spacing = min_spacing * spacing_mult

    # Get CZRC grid styling
    second_pass_style = CONFIG.get("visualization", {}).get("second_pass_grid", {})
    grid_color = second_pass_style.get("color", "rgba(26, 26, 60, 0.55)")
    grid_line_width = second_pass_style.get("line_width", 0.5)

    # === GENERATE ONE HEXAGONAL GRID PER CLUSTER ===
    all_hexagons: List[Polygon] = []

    for cluster in clusters:
        unified_tier1 = cluster["unified_tier1"]
        overall_r_max = cluster["overall_r_max"]  # noqa: F841 - kept for reference

        if unified_tier1.is_empty:
            continue

        # FIXED: Use unified_tier1 directly as the clip geometry
        # unified_tier1 is ALREADY Tier 1 = CZRC + 1×R_max
        # Previously we buffered by r_max which extended to Tier 2
        search_bounds = unified_tier1.bounds
        grid_origin = (search_bounds[0], search_bounds[1])

        # Generate hexagons within Tier 1 (not buffered to Tier 2)
        hexagons = generate_hexagon_grid_polygons(
            bounds=search_bounds,
            grid_spacing=grid_spacing,
            clip_geometry=unified_tier1,  # FIXED: Use Tier 1 directly, not buffered
            origin=grid_origin,
            logger=None,
        )
        all_hexagons.extend(hexagons)

    if not all_hexagons:
        return (start_idx, len(fig.data))

    # Build unified hexagon grid trace (same style as second pass grid)
    hexgrid_trace = build_hexagon_grid_trace(
        hexagon_polygons=all_hexagons,
        grid_color=grid_color,
        grid_line_width=grid_line_width,
        visible=False,  # Hidden by default, controlled by CZRC Grid checkbox
        name=f"CZRC Tier 1 Grid ({len(all_hexagons)} cells)",
    )

    if hexgrid_trace is not None:
        fig.add_trace(hexgrid_trace)
    else:
        # Fallback empty trace
        fig.add_trace(
            go.Scattergl(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=grid_color, width=grid_line_width),
                hoverinfo="skip",
                name="CZRC Tier 1 Grid",
                legendgroup="czrc_tier1_grid",
                showlegend=False,
                visible=False,
            )
        )

    return (start_idx, len(fig.data))


def _add_czrc_ilp_visibility_trace(
    fig: go.Figure,
    combo_key: str,  # noqa: ARG001 - kept for API consistency
    czrc_data: Dict[str, Any],
    czrc_config: Dict[str, Any],  # noqa: ARG001 - kept for API consistency
    cluster_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int]:
    """
    Add Tier 2 visibility boundary line for CZRC regions.

    Displays the Tier 2 (N× R_max) expansion boundary showing where locked
    boreholes provide coverage context for CZRC optimization.

    Where R_max = max(R_A, R_B, ...) for all zones involved in the region.

    The Tier 2 multiplier is configurable via CONFIG["visualization"]["czrc_tier2_rmax_multiplier"].

    This trace is controlled by the CZRC Grid checkbox in the Layers panel.

    Styling controlled by CONFIG["visualization"]["czrc_ilp_visibility"].

    When cluster_stats is provided, creates per-cluster Tier 2 boundaries with
    tooltips like "Cluster1_Tier2" for easier correlation with log files.
    """
    from shapely import wkt
    from shapely.ops import unary_union

    from Gap_Analysis_EC7.config import CONFIG

    start_idx = len(fig.data)
    pairwise_wkts = czrc_data.get("pairwise_wkts", {})
    zone_spacings = czrc_data.get("zone_spacings", {})

    if not pairwise_wkts:
        return (start_idx, len(fig.data))

    # Get styling from config
    vis_cfg = CONFIG.get("visualization", {})
    ilp_style = vis_cfg.get("czrc_ilp_visibility", {})
    tier2_color = ilp_style.get("tier2_color", "rgba(138, 43, 226, 0.8)")
    tier2_dash = ilp_style.get("tier2_dash", "longdash")
    line_width = ilp_style.get("line_width", 2)

    # Get Tier 2 multiplier from config (default: 2.0)
    tier2_mult = vis_cfg.get("czrc_tier2_rmax_multiplier", 2.0)

    # === BUILD PAIR_KEY TO CLUSTER_INDEX MAPPING ===
    # cluster_stats keys are cluster_keys (e.g., "ZoneA_ZoneB" or "ZoneA_ZoneB+ZoneC_ZoneD")
    # Each cluster_stats entry has pair_keys list and cluster_index
    pair_to_cluster: Dict[str, int] = {}
    if cluster_stats:
        for cluster_key, stats in cluster_stats.items():
            cluster_idx = stats.get("cluster_index", 0)
            pair_keys = stats.get("pair_keys", [])
            for pk in pair_keys:
                pair_to_cluster[pk] = cluster_idx

    # === COMPUTE EXPANDED BOUNDARIES FOR EACH PAIRWISE REGION ===
    # Group by cluster_index for per-cluster traces
    cluster_boundaries: Dict[int, List["BaseGeometry"]] = {}
    unclustered_boundaries: List["BaseGeometry"] = []

    for pair_key, region_wkt in pairwise_wkts.items():
        try:
            region_geom = wkt.loads(region_wkt)
            if region_geom.is_empty or not region_geom.is_valid:
                continue

            # Parse zone names from pair_key (format: "ZoneA_ZoneB")
            zone_names = pair_key.split("_")

            # Get R_max = max spacing of all involved zones
            spacings = []
            for zn in zone_names:
                if zn in zone_spacings:
                    spacings.append(zone_spacings[zn])
            if not spacings:
                # Fallback to max_zone_spacing from stats
                stats = czrc_data.get("stats", {})
                r_max = stats.get("max_zone_spacing", 150.0)
            else:
                r_max = max(spacings)

            # Tier 2: N× R_max expansion (locked boreholes boundary)
            tier2_geom = region_geom.buffer(tier2_mult * r_max)
            if not tier2_geom.is_empty:
                # Group by cluster index if available
                cluster_idx = pair_to_cluster.get(pair_key)
                if cluster_idx is not None:
                    if cluster_idx not in cluster_boundaries:
                        cluster_boundaries[cluster_idx] = []
                    cluster_boundaries[cluster_idx].append(tier2_geom)
                else:
                    unclustered_boundaries.append(tier2_geom)

        except Exception:  # noqa: BLE001 - WKT parsing can fail
            continue

    # Format multiplier for legend (remove trailing zeros)
    mult_str = f"{tier2_mult:g}"

    # === ADD PER-CLUSTER TIER 2 TRACES WITH TOOLTIPS ===
    if cluster_boundaries:
        for cluster_idx in sorted(cluster_boundaries.keys()):
            geoms = cluster_boundaries[cluster_idx]
            cluster_union = unary_union(geoms)
            if cluster_union is not None and not cluster_union.is_empty:
                tooltip = f"Cluster{cluster_idx}_Tier2"
                _add_boundary_trace(
                    fig=fig,
                    geometry=cluster_union,
                    color=tier2_color,
                    dash=tier2_dash,
                    line_width=line_width,
                    name=f"CZRC Tier 2 ({mult_str}× R_max) - Locked BHs",
                    legendgroup="czrc_tier2",
                    visible=False,  # Hidden by default, controlled by CZRC Grid checkbox
                    hovertext=tooltip,
                )

    # === ADD UNCLUSTERED BOUNDARIES (FALLBACK) ===
    if unclustered_boundaries:
        tier2_union = unary_union(unclustered_boundaries)
        if tier2_union is not None and not tier2_union.is_empty:
            _add_boundary_trace(
                fig=fig,
                geometry=tier2_union,
                color=tier2_color,
                dash=tier2_dash,
                line_width=line_width,
                name=f"CZRC Tier 2 ({mult_str}× R_max) - Locked BHs",
                legendgroup="czrc_tier2",
                visible=False,  # Hidden by default, controlled by CZRC Grid checkbox
            )

    return (start_idx, len(fig.data))


def _add_boundary_trace(
    fig: go.Figure,
    geometry: "BaseGeometry",
    color: str,
    dash: str,
    line_width: float,
    name: str,
    legendgroup: str,
    visible: bool = True,
    hovertext: Optional[str] = None,
) -> None:
    """
    Add a boundary line trace for a geometry (Polygon or MultiPolygon).

    Handles both single polygons and multi-polygons, extracting exterior
    rings for visualization as line traces.

    Args:
        fig: Plotly figure to add trace to
        geometry: Shapely geometry (Polygon or MultiPolygon)
        color: Line color (rgba string)
        dash: Dash pattern (e.g., "5 5" for dashed)
        line_width: Line width in pixels
        name: Trace name for legend
        legendgroup: Legend group for toggling
        visible: Initial visibility (default True)
        hovertext: Optional hover text to display on boundary lines
    """
    from shapely.geometry import MultiPolygon, Polygon

    # Collect all exterior coordinates
    x_coords = []
    y_coords = []

    if isinstance(geometry, Polygon):
        polygons = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)
    else:
        return  # Unsupported geometry type

    for poly in polygons:
        if poly.is_empty:
            continue
        # Get exterior ring coordinates
        exterior = poly.exterior
        coords = list(exterior.coords)

        # Add coordinates with None separator for discontinuous lines
        if x_coords:  # Add separator if not first polygon
            x_coords.append(None)
            y_coords.append(None)

        for x, y in coords:
            x_coords.append(x)
            y_coords.append(y)

    if not x_coords:
        return

    # Add trace with optional hovertext
    hover_info = "text" if hovertext else "name"
    fig.add_trace(
        go.Scattergl(
            x=x_coords,
            y=y_coords,
            mode="lines",
            line=dict(color=color, width=line_width, dash=dash),
            hoverinfo=hover_info,
            hovertext=hovertext if hovertext else None,
            name=name,
            legendgroup=legendgroup,
            showlegend=True,
            visible=visible,
        )
    )


def _add_first_pass_candidates_trace(
    fig: go.Figure,
    combo_key: str,
    first_pass_candidates: List[Dict[str, Any]],
    trace_prefix: str = "",
) -> Tuple[int, int]:
    """
    Add first-pass border boreholes that were used as ILP candidates.

    These are shown as black X markers to distinguish from fresh grid candidates.
    Visibility matches the grid layer.

    Styling controlled by CONFIG["visualization"]["first_pass_candidate_marker"].

    Args:
        fig: Plotly figure to add traces to
        combo_key: Filter combination key for trace naming
        first_pass_candidates: List of {"x", "y"} candidate coordinates
        trace_prefix: Prefix for legend group (e.g., "czrc_" for CZRC)

    Returns:
        Tuple of (start_idx, end_idx) for trace range
    """
    from Gap_Analysis_EC7.config import CONFIG

    start_idx = len(fig.data)

    if not first_pass_candidates:
        return (start_idx, len(fig.data))

    # Get marker styling from config
    marker_config = CONFIG.get("visualization", {}).get(
        "first_pass_candidate_marker",
        {"size": 10, "color": "black", "symbol": "x", "line_width": 2},
    )

    # Normalize color for Scattergl compatibility (8-char hex → rgba)
    marker_color = _normalize_color_for_scattergl(marker_config.get("color", "black"))

    # Extract coordinates
    x_coords = [c["x"] for c in first_pass_candidates]
    y_coords = [c["y"] for c in first_pass_candidates]

    # Determine legend group based on prefix
    legend_group = f"{trace_prefix}grid" if trace_prefix else "second_pass_grid"

    fig.add_trace(
        go.Scattergl(
            x=x_coords,
            y=y_coords,
            mode="markers",
            marker=dict(
                symbol=marker_config.get("symbol", "x"),
                size=marker_config.get("size", 10),
                color=marker_color,
                line=dict(
                    width=marker_config.get("line_width", 2),
                    color=marker_color,
                ),
            ),
            hovertemplate="First-Pass Candidate<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>",
            name=f"First-Pass Candidates ({combo_key})",
            legendgroup=legend_group,
            showlegend=False,
            visible=False,  # Same visibility as grid layer
        )
    )

    return (start_idx, len(fig.data))


def _create_hexagon(cx: float, cy: float, radius: float) -> "Polygon":
    """Create a hexagon polygon centered at (cx, cy) with given radius."""
    from shapely.geometry import Polygon
    import math

    points = []
    for i in range(6):
        angle = math.pi / 3 * i + math.pi / 6  # Start at 30 degrees for flat-top
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    return Polygon(points)


def _add_precomputed_coverage_traces(
    fig: go.Figure,
    precomputed_coverages: Dict[str, Dict[str, Any]],
    coverage_colors: Dict[str, str],
    proposed_marker_config: Dict[str, Any],
    max_spacing: float,
    default_combo_key: str,
    hexgrid_config: Optional[Dict[str, Any]] = None,
    grid_spacing: float = 50.0,
    zones_gdf: Optional["GeoDataFrame"] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """
    Add pre-computed coverage traces for all filter combinations.

    Creates 5 merged traces per combo (covered, gaps, buffers, markers, hexagon_grid).
    Only default_combo_key coverage traces are initially visible.
    Hexagon grid traces are always initially hidden (toggled via checkbox).

    If zones_gdf is provided with per-zone spacing, generates per-zone hexagon grids.

    Returns:
        Dict mapping combo_key -> {"covered": (start, end), "hexagon_grid": (start, end), ...}
    """
    colors = {
        "covered": coverage_colors.get("covered", "rgba(0, 200, 0, 0.3)"),
        "gap": coverage_colors.get("gap", "rgba(255, 0, 0, 0.5)"),
        "buffer": proposed_marker_config.get("buffer_color", "rgba(4, 0, 255, 0.3)"),
        "line_color": proposed_marker_config.get("line_color", None),
        "line_width": proposed_marker_config.get("line_width", 1),
        "marker": proposed_marker_config.get("color", "rgba(0, 100, 255, 0.9)"),
        "marker_size": proposed_marker_config.get("size", 14),
        "marker_symbol": proposed_marker_config.get("symbol", "x"),
    }

    trace_ranges: Dict[str, Dict[str, Tuple[int, int]]] = {}
    combo_count = 0

    for combo_key, data in precomputed_coverages.items():
        if not data.get("success", False):
            continue
        is_default = combo_key == default_combo_key
        trace_ranges[combo_key] = _add_single_combo_traces(
            fig=fig,
            combo_key=combo_key,
            data=data,
            colors=colors,
            is_visible=is_default,
            max_spacing=max_spacing,
            hexgrid_config=hexgrid_config,
            grid_spacing=grid_spacing,
            zones_gdf=zones_gdf,
        )
        combo_count += 1

    if logger:
        logger.info(
            f"   Added {combo_count * 5} precomputed coverage traces "
            f"({combo_count} combinations × 5 traces incl. hexgrid)"
        )

    return trace_ranges


# ===========================================================================
# HTML GENERATION HELPERS
# ===========================================================================


def _add_bgs_layers_to_figure(
    fig: go.Figure,
    bgs_layers_data: Optional[Dict[str, Tuple[GeoDataFrame, Dict[str, Any]]]],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """Add BGS geology layers to figure (background, initially hidden). Returns layer ranges and final index."""
    bgs_layers: Dict[str, Tuple[int, int]] = {}
    current_idx = 0

    if bgs_layers_data:
        from Gap_Analysis_EC7.visualization.bgs_geology_layer import (
            add_bgs_geology_to_figure,
        )

        for layer_name, (layer_gdf, layer_config) in bgs_layers_data.items():
            if layer_gdf is not None and not layer_gdf.empty:
                if logger:
                    logger.info(
                        f"   Adding {layer_name} layer ({len(layer_gdf)} polygons)..."
                    )
                bgs_start, bgs_end = add_bgs_geology_to_figure(
                    fig=fig,
                    bedrock_gdf=layer_gdf,
                    bgs_config=layer_config,
                    visible=False,
                    logger=logger,
                )
                bgs_layers[layer_name] = (bgs_start, bgs_end)
                current_idx = bgs_end

    return bgs_layers, current_idx


def _add_satellite_layer(
    fig: go.Figure,
    boreholes_gdf: GeoDataFrame,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, Optional[Tuple[float, float, float, float]]]:
    """Fetch and add satellite imagery layer to figure. Returns (has_satellite, bounds)."""
    SATELLITE_BUFFER_M = 500.0
    has_satellite = False
    satellite_bounds = None

    if boreholes_gdf is None or boreholes_gdf.empty:
        return has_satellite, satellite_bounds

    total_bounds = boreholes_gdf.total_bounds
    satellite_bounds = (
        total_bounds[0] - SATELLITE_BUFFER_M,
        total_bounds[1] - SATELLITE_BUFFER_M,
        total_bounds[2] + SATELLITE_BUFFER_M,
        total_bounds[3] + SATELLITE_BUFFER_M,
    )

    satellite_base64 = _fetch_satellite_tiles_base64(
        bounds=satellite_bounds,
        crs="EPSG:27700",
        alpha=1.0,
        logger=logger,
    )

    if satellite_base64:
        has_satellite = True
        fig.add_layout_image(
            dict(
                source=satellite_base64,
                xref="x",
                yref="y",
                x=satellite_bounds[0],
                y=satellite_bounds[3],
                sizex=satellite_bounds[2] - satellite_bounds[0],
                sizey=satellite_bounds[3] - satellite_bounds[1],
                sizing="stretch",
                opacity=0,
                layer="below",
            )
        )
        if logger:
            logger.info("   Added satellite imagery layer (initially hidden)")

    return has_satellite, satellite_bounds


def _generate_sidebar_panels(
    boreholes_gdf: GeoDataFrame,
    zones_gdf: Optional[GeoDataFrame],
    zones_config: Dict[str, Any],
    visualization_config: Union[Dict[str, Any], VisualizationConfig],
    test_data_locations: Optional[Dict[str, set]],
    precomputed_coverages: Optional[Dict[str, Dict[str, Any]]],
    bgs_layers: Dict[str, Tuple[int, int]],
    has_satellite: bool,
    proposed_trace_range: Optional[Tuple[int, int]],
    hexgrid_trace_range: Optional[Tuple[int, int]],
    coverage_trace_ranges: Dict[str, Dict[str, Tuple[int, int]]],
    boreholes_start: int,
    max_spacing: float,
    logger: Optional[logging.Logger] = None,
    default_filter: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Generate left and right sidebar panel HTML. Returns (left_html, right_html)."""
    # Normalize to typed config
    borehole_marker = _get_borehole_marker_from_config(visualization_config)

    # Coverage statistics panel from precomputed data
    stats_panels_html = _generate_stats_panel_from_precomputed(
        precomputed_coverages, max_spacing
    )

    # Layers panel
    has_bgs = len(bgs_layers) > 0
    has_proposed = proposed_trace_range is not None
    has_hexgrid = hexgrid_trace_range is not None or bool(
        precomputed_coverages and coverage_trace_ranges
    )

    # Check if any combo has second pass data (removed or added boreholes)
    has_second_pass = False
    if coverage_trace_ranges:
        for combo_ranges in coverage_trace_ranges.values():
            removed_range = combo_ranges.get("removed_buffers", (0, 0))
            added_range = combo_ranges.get("added_buffers", (0, 0))
            if removed_range[0] != removed_range[1] or added_range[0] != added_range[1]:
                has_second_pass = True
                break

    # Check if any combo has CZRC second pass data
    has_czrc_second_pass = False
    if coverage_trace_ranges:
        for combo_ranges in coverage_trace_ranges.values():
            czrc_removed = combo_ranges.get("czrc_removed_buffers", (0, 0))
            czrc_added = combo_ranges.get("czrc_added_buffers", (0, 0))
            if czrc_removed[0] != czrc_removed[1] or czrc_added[0] != czrc_added[1]:
                has_czrc_second_pass = True
                break

    # Check if any combo has CZRC test points
    has_czrc_test_points = False
    if coverage_trace_ranges:
        for combo_ranges in coverage_trace_ranges.values():
            czrc_tp_range = combo_ranges.get("czrc_test_points", (0, 0))
            if czrc_tp_range[0] != czrc_tp_range[1]:
                has_czrc_test_points = True
                break

    # Check if any combo has CZRC zone overlap data (clouds/pairwise)
    # This enables the Zone Overlap checkbox even when skip_ilp=True
    has_czrc_zone_overlap = False
    if coverage_trace_ranges:
        for combo_ranges in coverage_trace_ranges.values():
            czrc_clouds = combo_ranges.get("czrc_clouds", (0, 0))
            czrc_pairwise = combo_ranges.get("czrc_pairwise", (0, 0))
            if czrc_clouds[0] != czrc_clouds[1] or czrc_pairwise[0] != czrc_pairwise[1]:
                has_czrc_zone_overlap = True
                break

    # Check if any combo has CZRC grid data (visibility boundary, hex grid)
    # This is added whenever czrc_data exists, even with skip_ilp=True
    has_czrc_grid = False
    if coverage_trace_ranges:
        for combo_ranges in coverage_trace_ranges.values():
            czrc_grid = combo_ranges.get("czrc_grid", (0, 0))
            if czrc_grid[0] != czrc_grid[1]:
                has_czrc_grid = True
                break

    # Check if any combo has third pass data (cell-cell CZRC removed/added)
    has_third_pass = False
    if coverage_trace_ranges:
        for combo_ranges in coverage_trace_ranges.values():
            third_removed = combo_ranges.get("third_pass_removed_buffers", (0, 0))
            third_added = combo_ranges.get("third_pass_added_buffers", (0, 0))
            if third_removed[0] != third_removed[1] or third_added[0] != third_added[1]:
                has_third_pass = True
                break

    # Check if any combo has third pass overlap data (cell clouds/intersections)
    has_third_pass_overlap = False
    if coverage_trace_ranges:
        for combo_ranges in coverage_trace_ranges.values():
            third_clouds = combo_ranges.get("third_pass_clouds", (0, 0))
            third_intersections = combo_ranges.get("third_pass_intersections", (0, 0))
            if (
                third_clouds[0] != third_clouds[1]
                or third_intersections[0] != third_intersections[1]
            ):
                has_third_pass_overlap = True
                break

    # Check if any combo has third pass grid data (cell-cell candidate grid)
    has_third_pass_grid = False
    if coverage_trace_ranges:
        for combo_ranges in coverage_trace_ranges.values():
            third_grid = combo_ranges.get("third_pass_grid", (0, 0))
            if third_grid[0] != third_grid[1]:
                has_third_pass_grid = True
                break

    # Check if any combo has third pass test points data
    has_third_pass_test_points = False
    if coverage_trace_ranges:
        for combo_ranges in coverage_trace_ranges.values():
            third_test_points = combo_ranges.get("third_pass_test_points", (0, 0))
            if third_test_points[0] != third_test_points[1]:
                has_third_pass_test_points = True
                break

    layers_panel_html = ""
    if (
        has_bgs
        or has_satellite
        or has_proposed
        or has_hexgrid
        or has_second_pass
        or has_czrc_second_pass
        or has_czrc_test_points
        or has_czrc_zone_overlap
        or has_czrc_grid
        or has_third_pass
        or has_third_pass_overlap
        or has_third_pass_grid
        or has_third_pass_test_points
    ):
        layers_panel_html = _generate_layers_panel_html(
            bgs_layers=bgs_layers if has_bgs else None,
            has_satellite=has_satellite,
            proposed_trace_range=proposed_trace_range,
            hexgrid_trace_range=hexgrid_trace_range,
            uses_precomputed_coverages=bool(
                precomputed_coverages and coverage_trace_ranges
            ),
            has_second_pass=has_second_pass,
            has_czrc_second_pass=has_czrc_second_pass,
            has_czrc_test_points=has_czrc_test_points,
            has_czrc_zone_overlap=has_czrc_zone_overlap,
            has_czrc_grid=has_czrc_grid,
            has_third_pass=has_third_pass,
            has_third_pass_overlap=has_third_pass_overlap,
            has_third_pass_grid=has_third_pass_grid,
            has_third_pass_test_points=has_third_pass_test_points,
        )
        _log_layers_panel(
            logger, has_satellite, has_bgs, bgs_layers, has_proposed, has_hexgrid
        )

    # Filters and legend panels
    filters_panel_html = _generate_depth_filters_panel(
        boreholes_gdf, test_data_locations, boreholes_start, logger, default_filter
    )
    legend_panel_html = _generate_legend_panel_html(
        zones_gdf=zones_gdf,
        zones_config=zones_config,
        boreholes_count=len(boreholes_gdf) if boreholes_gdf is not None else 0,
        borehole_marker_config=borehole_marker,
    )
    if logger:
        logger.info("   Added external legend panel")

    return layers_panel_html + filters_panel_html, legend_panel_html + stats_panels_html


def _generate_stats_panel_from_precomputed(
    precomputed_coverages: Optional[Dict[str, Dict[str, Any]]],
    max_spacing: float,
) -> str:
    """Generate coverage stats panel HTML from precomputed data."""
    # Check if stats panel is disabled in config
    if not CONFIG.get("visualization", {}).get("show_coverage_stats_panel", True):
        return ""

    if not precomputed_coverages:
        return ""

    default_key = _get_default_combo_key(precomputed_coverages)
    default_combo = precomputed_coverages.get(default_key, {})
    combo_stats = default_combo.get("stats", {})

    if not combo_stats:
        return ""

    return _generate_coverage_stats_panel_html(
        gap_stats=[],
        covered_area_ha=combo_stats.get("covered_area_ha", 0),
        uncovered_area_ha=combo_stats.get("uncovered_area_ha", 0),
        borehole_count=default_combo.get("boreholes_count", 0),
        max_spacing=max_spacing,
    )


def _log_layers_panel(
    logger: Optional[logging.Logger],
    has_satellite: bool,
    has_bgs: bool,
    bgs_layers: Dict[str, Tuple[int, int]],
    has_proposed: bool,
    has_hexgrid: bool,
) -> None:
    """Log layers panel info."""
    if not logger:
        return
    layer_info = []
    if has_satellite:
        layer_info.append("satellite")
    if has_bgs:
        layer_info.append(f"{len(bgs_layers)} BGS layers")
    if has_proposed:
        layer_info.append("proposed boreholes")
    if has_hexgrid:
        layer_info.append("candidate grid")
    logger.info(f"   Added layers panel with {', '.join(layer_info)}")


def _generate_depth_filters_panel(
    boreholes_gdf: GeoDataFrame,
    test_data_locations: Optional[Dict[str, set]],
    boreholes_start: int,
    logger: Optional[logging.Logger] = None,
    default_filter: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate filters panel HTML with depth slider and test data checkboxes.

    Args:
        boreholes_gdf: GeoDataFrame with borehole data
        test_data_locations: Dict with sets of borehole IDs having each test type
        boreholes_start: Index of boreholes trace in figure
        logger: Optional logger for debug output
        default_filter: Optional dict with default filter values for testing mode.
                       Keys: min_depth, require_spt, require_triaxial_total, require_triaxial_effective
    """
    if boreholes_gdf is None or len(boreholes_gdf) == 0:
        return ""

    if "Final Depth" not in boreholes_gdf.columns:
        return ""

    depths = boreholes_gdf["Final Depth"].dropna()
    if len(depths) == 0:
        return ""

    min_depth = float(depths.min())
    max_depth = float(depths.max())

    test_data_counts = None
    if test_data_locations:
        test_data_counts = {
            "spt": len(test_data_locations.get("spt", set())),
            "triaxial_total": len(test_data_locations.get("triaxial_total", set())),
            "triaxial_effective": len(
                test_data_locations.get("triaxial_effective", set())
            ),
        }

    result = _generate_filters_panel_html(
        min_depth=min_depth,
        max_depth=max_depth,
        boreholes_trace_idx=boreholes_start,
        test_data_counts=test_data_counts,
        depth_step_m=CONFIG.get("depth_slider_step_m", 10.0),
        default_filter=default_filter,
    )

    if logger:
        logger.info(
            f"   Added filters panel (depth range: {min_depth:.1f}m - {max_depth:.1f}m)"
        )
        if test_data_counts:
            logger.info(
                f"   Test data filters: SPT={test_data_counts['spt']}, "
                f"TxT={test_data_counts['triaxial_total']}, "
                f"TxE={test_data_counts['triaxial_effective']}"
            )

    return result


def _get_default_combo_key(precomputed_coverages: Dict[str, Dict[str, Any]]) -> str:
    """Get the default combo key from precomputed coverages."""
    default_key = "d0_spt0_txt0_txe0"
    if default_key not in precomputed_coverages:
        available_keys = [
            k
            for k in precomputed_coverages.keys()
            if precomputed_coverages[k].get("success", False)
        ]
        if available_keys:
            default_key = available_keys[0]
    return default_key


def _build_flex_wrapper_html(
    left_sidebar_html: str,
    right_sidebar_html: str,
    left_sidebar_width: int,
    right_sidebar_width: int,
) -> Tuple[str, str]:
    """Build flex wrapper HTML for sidebars. Returns (wrapper_start, wrapper_end)."""
    flex_wrapper_start = f"""
<div id="plotPageWrapper" style="
    display: flex;
    flex-direction: row;
    min-height: 100vh;
    width: fit-content;
    padding: 0 20px 20px 0;
    box-sizing: border-box;
">
    <div id="leftSidebar" style="
        display: flex;
        flex-direction: column;
        width: {left_sidebar_width}px;
        min-width: {left_sidebar_width}px;
        flex-shrink: 0;
        padding-top: {PANEL_TOP_OFFSET}px;
    ">
{left_sidebar_html}
    </div>
    <div id="plotContainer" style="
        flex-shrink: 0;
        min-width: 0;
    ">
"""

    flex_wrapper_end = f"""
    </div>
    <div id="rightSidebar" style="
        display: flex;
        flex-direction: column;
        width: {right_sidebar_width}px;
        min-width: {right_sidebar_width}px;
        flex-shrink: 0;
        padding-top: {PANEL_TOP_OFFSET}px;
    ">
{right_sidebar_html}
    </div>
</div>
"""
    return flex_wrapper_start, flex_wrapper_end


def _embed_coverage_script(
    html: str,
    precomputed_coverages: Dict[str, Dict[str, Any]],
    coverage_trace_ranges: Dict[str, Dict[str, Tuple[int, int]]],
    max_spacing: float,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Embed precomputed coverage data script into HTML head."""
    if not precomputed_coverages or not coverage_trace_ranges:
        return html

    trace_ranges_json = json.dumps(coverage_trace_ranges, separators=(",", ":"))

    stats_only = {
        k: {
            "stats": v.get("stats", {}),
            "boreholes_count": v.get("boreholes_count", 0),
        }
        for k, v in precomputed_coverages.items()
        if v.get("success", False)
    }
    stats_json = json.dumps(stats_only, separators=(",", ":"))

    default_combo_key_js = _get_default_combo_key(precomputed_coverages)
    if default_combo_key_js not in coverage_trace_ranges:
        available_keys = list(coverage_trace_ranges.keys())
        if available_keys:
            default_combo_key_js = available_keys[0]

    coverage_script_js = generate_coverage_data_script(
        trace_ranges_json=trace_ranges_json,
        stats_json=stats_json,
        max_spacing=max_spacing,
        default_combo_key=default_combo_key_js,
    )
    precomputed_script = f"<script>\n{coverage_script_js}</script>"

    final_html = html.replace("</head>", f"{precomputed_script}</head>")

    if logger:
        json_size_kb = len(trace_ranges_json) / 1024
        stats_size_kb = len(stats_json) / 1024
        logger.info(
            f"   📦 Embedded trace ranges ({json_size_kb:.1f} KB) + stats ({stats_size_kb:.1f} KB)"
        )

    return final_html


def _add_coverage_traces_section(
    fig: go.Figure,
    precomputed_coverages: Optional[Dict[str, Dict[str, Any]]],
    max_spacing: float,
    zones_gdf: Optional["GeoDataFrame"] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """Add precomputed coverage traces to figure. Returns trace ranges dict."""
    if not precomputed_coverages:
        return {}

    if logger:
        logger.info(
            f"   Adding precomputed coverage traces for "
            f"{len(precomputed_coverages)} combinations..."
        )

    proposed_marker_config = CONFIG.get("visualization", {}).get("proposed_marker", {})
    coverage_colors = CONFIG.get("visualization", {}).get("coverage_colors", {})

    default_combo_key = _get_default_combo_key(precomputed_coverages)
    if default_combo_key != "d0_spt0_txt0_txe0" and logger:
        logger.info(f"   🧪 Testing mode: using '{default_combo_key}' as default combo")

    max_spacing_for_grid = CONFIG.get("max_spacing_m", 100.0)
    candidate_mult = CONFIG.get("ilp_solver", {}).get("candidate_spacing_mult", 0.5)
    computed_grid_spacing = max_spacing_for_grid * candidate_mult

    return _add_precomputed_coverage_traces(
        fig=fig,
        precomputed_coverages=precomputed_coverages,
        coverage_colors=coverage_colors,
        proposed_marker_config=proposed_marker_config,
        max_spacing=max_spacing,
        default_combo_key=default_combo_key,
        hexgrid_config=CONFIG.get("visualization", {}).get("hexgrid_overlay", {}),
        grid_spacing=computed_grid_spacing,
        zones_gdf=zones_gdf,
        logger=logger,
    )


def _add_secondary_shapefiles_section(
    fig: go.Figure,
    all_shapefiles: Optional[Dict[str, "GeoDataFrame"]],
    current_idx: int,
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Add secondary (non-primary) shapefile boundary traces.

    Secondary shapefiles are drawn before primary zones for proper layering.
    Uses SHAPEFILE_CONFIG for rendering settings.

    Args:
        fig: Plotly figure to add traces to
        all_shapefiles: Dict mapping layer key to GeoDataFrame
        current_idx: Current trace index
        logger: Optional logger

    Returns:
        Updated trace index after adding secondary shapefile traces.
    """
    if not all_shapefiles:
        return current_idx

    # Import shapefile config for rendering settings
    from Gap_Analysis_EC7.shapefile_config import (
        SHAPEFILE_CONFIG,
        get_coverage_layer_key,
        get_layer_config,
        get_feature_config,
    )

    coverage_key = get_coverage_layer_key()
    traces_added = 0

    for layer_key, gdf in all_shapefiles.items():
        # Skip coverage layer (handled separately by zone boundaries)
        if layer_key == coverage_key:
            continue

        if gdf is None or gdf.empty:
            continue

        layer_config = get_layer_config(layer_key)
        display_name = layer_config.get("display_name", layer_key)
        rendering = layer_config.get("rendering", {})
        default_color = rendering.get("boundary_color", "#666666")
        default_linewidth = rendering.get("boundary_linewidth", 1.5)

        if logger:
            logger.info(f"   Adding {display_name} boundary ({len(gdf)} features)...")

        # Build traces for this shapefile
        for idx, row in gdf.iterrows():
            geom = row.geometry

            # Get feature name from zone_name, Name column, or fallback to index
            feature_name = row.get("zone_name") or row.get("Name") or f"Feature {idx}"

            # Get feature-specific config
            feature_cfg = get_feature_config(layer_key, feature_name)
            color = feature_cfg.get("boundary_color", default_color)
            linewidth = feature_cfg.get("boundary_linewidth", default_linewidth)

            # Handle Polygon and MultiPolygon
            polygons = []
            if geom.geom_type == "Polygon":
                polygons = [geom]
            elif geom.geom_type == "MultiPolygon":
                polygons = list(geom.geoms)

            first_poly = True
            for poly in polygons:
                x_coords = list(poly.exterior.coords.xy[0])
                y_coords = list(poly.exterior.coords.xy[1])
                x_coords.append(x_coords[0])
                y_coords.append(y_coords[0])

                trace = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    line=dict(color=color, width=linewidth),
                    name=display_name,
                    legendgroup=f"shapefile_{layer_key}",
                    showlegend=first_poly and idx == 0,
                    hovertemplate=(
                        f"<b>{display_name}</b><br>"
                        f"{feature_name}<br>"
                        "Easting: %{x:,.0f}<br>"
                        "Northing: %{y:,.0f}<br>"
                        "<extra></extra>"
                    ),
                    visible=True,
                )
                fig.add_trace(trace)
                traces_added += 1
                first_poly = False

    if logger and traces_added > 0:
        logger.info(f"   ✅ Added {traces_added} secondary shapefile traces")

    return current_idx + traces_added


def _add_zone_boundaries_section(
    fig: go.Figure,
    zones_gdf: Optional[GeoDataFrame],
    zones_config: Optional[Dict[str, Any]],
    zone_defaults: Optional[Dict[str, Any]],
    current_idx: int,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Add zone boundary outlines to figure. Returns updated current_idx."""
    if zones_gdf is None or zones_gdf.empty:
        return current_idx

    if logger:
        logger.info(f"   Adding zone boundary outlines ({len(zones_gdf)} zones)...")

    zone_boundary_traces = _add_zone_boundary_traces(
        fig=fig,
        zones_gdf=zones_gdf,
        zones_config=zones_config or {},
        zone_defaults=zone_defaults or {"boundary_linewidth": 3.0},
        logger=logger,
    )
    return current_idx + zone_boundary_traces


def _add_boreholes_section(
    fig: go.Figure,
    boreholes_gdf: GeoDataFrame,
    visualization_config: Union[Dict[str, Any], VisualizationConfig],
    test_data_locations: Optional[Dict[str, set]],
    precomputed_coverages: Optional[Dict[str, Dict[str, Any]]],
    coverage_trace_ranges: Dict[str, Dict[str, Tuple[int, int]]],
    current_idx: int,  # noqa: ARG001 - kept for API compatibility
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, int, Optional[Tuple[int, int]]]:
    """Add boreholes trace and compute proposed range. Returns (start_idx, new_idx, proposed_range)."""
    # Get borehole marker config (handles both dict and typed config)
    borehole_marker = _get_borehole_marker_from_config(visualization_config)

    # CRITICAL: Use actual trace count from figure, not manual tracking
    # Manual current_idx tracking can be wrong due to variable trace counts per combo
    boreholes_start = len(fig.data)
    bh_count = _add_boreholes_trace(
        fig,
        boreholes_gdf,
        borehole_marker,
        test_data_locations=test_data_locations,
    )
    new_current_idx = len(fig.data)  # Get actual count after adding

    proposed_trace_range = None
    if precomputed_coverages and coverage_trace_ranges:
        default_combo_key = _get_default_combo_key(precomputed_coverages)
        if default_combo_key not in coverage_trace_ranges:
            available_keys = list(coverage_trace_ranges.keys())
            if available_keys:
                default_combo_key = available_keys[0]
        if default_combo_key in coverage_trace_ranges:
            default_ranges = coverage_trace_ranges[default_combo_key]
            buffer_range = default_ranges.get("proposed_buffers", (0, 0))
            marker_range = default_ranges.get("proposed_markers", (0, 0))
            proposed_trace_range = (buffer_range[0], marker_range[1])
            if logger:
                logger.info(
                    f"   Proposed boreholes trace range: {proposed_trace_range}"
                )

    return boreholes_start, new_current_idx, proposed_trace_range


def _assemble_final_html(
    plotly_html: str,
    left_sidebar_html: str,
    right_sidebar_html: str,
    precomputed_coverages: Optional[Dict[str, Dict[str, Any]]],
    coverage_trace_ranges: Dict[str, Dict[str, Tuple[int, int]]],
    max_spacing: float,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Assemble final HTML with flex layout and embedded scripts."""
    left_sidebar_width = LEFT_PANEL_WIDTH + SIDEBAR_SPACING
    right_sidebar_width = RIGHT_PANEL_WIDTH + SIDEBAR_SPACING

    flex_wrapper_start, flex_wrapper_end = _build_flex_wrapper_html(
        left_sidebar_html, right_sidebar_html, left_sidebar_width, right_sidebar_width
    )

    final_html = plotly_html.replace(
        "<div>",
        f"{flex_wrapper_start}<div>",
        1,
    )
    final_html = final_html.replace("</body>", f"{flex_wrapper_end}</body>")

    # Embed coverage switching script
    final_html = _embed_coverage_script(
        final_html, precomputed_coverages, coverage_trace_ranges, max_spacing, logger
    )

    # Embed click-to-copy tooltip script
    click_to_copy_js = generate_click_to_copy_script()
    click_to_copy_script = f"<script>\n{click_to_copy_js}</script>"
    final_html = final_html.replace("</body>", f"{click_to_copy_script}</body>")

    return final_html


def _log_substep_timing(
    substep_times: Dict[str, float],
    html_start: float,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log sub-step timing summary."""
    if not logger:
        return

    total_html_time = time.perf_counter() - html_start
    logger.info("   ⏱️ HTML Generation Sub-step Timing:")
    for substep_name, duration in substep_times.items():
        pct = (duration / total_html_time) * 100 if total_html_time > 0 else 0
        logger.info(f"      {substep_name}: {duration:.2f}s ({pct:.1f}%)")
    logger.info(f"      HTML Total: {total_html_time:.2f}s")


# ===========================================================================
# MAIN HTML GENERATION
# ===========================================================================


def generate_multi_layer_html(
    boreholes_gdf: GeoDataFrame,
    output_path: str,
    visualization_config: Union[Dict[str, Any], VisualizationConfig],
    generator_name: str = "EC7",
    bgs_layers_data: Optional[Dict[str, Tuple[GeoDataFrame, Dict[str, Any]]]] = None,
    zones_gdf: Optional[GeoDataFrame] = None,
    zones_config: Optional[Dict[str, Any]] = None,
    zone_defaults: Optional[Dict[str, Any]] = None,
    all_shapefiles: Optional[Dict[str, GeoDataFrame]] = None,
    test_data_locations: Optional[Dict[str, set]] = None,
    max_spacing: float = 200.0,
    logger: Optional[logging.Logger] = None,
    precomputed_coverages: Optional[Dict[str, Dict[str, Any]]] = None,
    default_filter: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """
    Generate interactive Plotly HTML for EC7 coverage visualization with toggleable layers.

    Args:
        visualization_config: Dict or VisualizationConfig with visualization settings.
        default_filter: Optional dict with default filter values for HTML controls.
                       Keys: min_depth, require_spt, require_triaxial_total, require_triaxial_effective
                       When provided, the HTML will load with these filter values pre-selected.
    """
    _ = generator_name  # Currently unused but kept for API compatibility
    if logger:
        logger.info("Generating EC7 HTML map...")

    substep_times, html_start, fig, current_idx = (
        {},
        time.perf_counter(),
        go.Figure(),
        0,
    )

    # Step 6a: Add BGS geology layers
    substep_start = time.perf_counter()
    bgs_layers, current_idx = _add_bgs_layers_to_figure(fig, bgs_layers_data, logger)
    substep_times["6a_bgs_layers"] = time.perf_counter() - substep_start

    # Step 6b: Add precomputed coverage traces
    substep_start = time.perf_counter()
    coverage_trace_ranges = _add_coverage_traces_section(
        fig, precomputed_coverages, max_spacing, zones_gdf, logger
    )
    if precomputed_coverages:
        current_idx += len(precomputed_coverages) * 5
    substep_times["6b_coverage_zones"] = time.perf_counter() - substep_start

    # Step 6c: Add secondary shapefile boundaries (non-primary, lower layer_order)
    substep_start = time.perf_counter()
    current_idx = _add_secondary_shapefiles_section(
        fig, all_shapefiles, current_idx, logger
    )
    substep_times["6c_secondary_shapefiles"] = time.perf_counter() - substep_start

    # Step 6c2: Add primary zone boundaries (zones_gdf)
    substep_start = time.perf_counter()
    current_idx = _add_zone_boundaries_section(
        fig, zones_gdf, zones_config, zone_defaults, current_idx, logger
    )
    substep_times["6c2_zone_boundaries"] = time.perf_counter() - substep_start

    # Step 6c3: Hexgrids (added per-combo in coverage traces)
    substep_start = time.perf_counter()
    hexgrid_trace_range = None
    substep_times["6c3_hexagon_grid"] = time.perf_counter() - substep_start

    # Step 6d: Add boreholes
    substep_start = time.perf_counter()
    boreholes_start, current_idx, proposed_trace_range = _add_boreholes_section(
        fig,
        boreholes_gdf,
        visualization_config,
        test_data_locations,
        precomputed_coverages,
        coverage_trace_ranges,
        current_idx,
        logger,
    )
    substep_times["6d_boreholes"] = time.perf_counter() - substep_start

    if logger:
        logger.info(
            f"   Total traces: {current_idx} (boreholes at idx {boreholes_start})"
        )

    # Step 6f: Add satellite imagery
    substep_start = time.perf_counter()
    has_satellite, _ = _add_satellite_layer(fig, boreholes_gdf, logger)
    substep_times["6f_satellite"] = time.perf_counter() - substep_start

    # Step 6g: Generate Plotly HTML
    substep_start = time.perf_counter()
    fig.update_layout(
        **_build_layout_without_dropdown(
            visualization_config, "Gap Analysis - EC7 Coverage"
        )
    )
    plotly_html = fig.to_html(
        include_plotlyjs=True,
        full_html=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    substep_times["6g_plotly_to_html"] = time.perf_counter() - substep_start

    # Step 6h: Generate sidebar panels
    substep_start = time.perf_counter()
    left_sidebar_html, right_sidebar_html = _generate_sidebar_panels(
        boreholes_gdf,
        zones_gdf,
        zones_config or {},
        visualization_config,
        test_data_locations,
        precomputed_coverages,
        bgs_layers,
        has_satellite,
        proposed_trace_range,
        hexgrid_trace_range,
        coverage_trace_ranges,
        boreholes_start,
        max_spacing,
        logger,
        default_filter,
    )
    substep_times["6h_panels"] = time.perf_counter() - substep_start

    # Assemble final HTML and write to file
    final_html = _assemble_final_html(
        plotly_html,
        left_sidebar_html,
        right_sidebar_html,
        precomputed_coverages,
        coverage_trace_ranges,
        max_spacing,
        logger,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)
    if logger:
        logger.info("Multi-layer HTML saved: %s", output_path)
    # Substep timing disabled - uncomment if detailed profiling needed
    # _log_substep_timing(substep_times, html_start, logger)
    return fig


def generate_single_layer_html(
    grid_gdf: GeoDataFrame,
    boreholes_gdf: GeoDataFrame,
    proposed_gdf: GeoDataFrame,
    output_path: str,
    visualization_config: Union[Dict[str, Any], VisualizationConfig],
    generator_name: str = "Grid",
    logger: Optional[logging.Logger] = None,
) -> go.Figure:
    """Generate interactive HTML for a single grid generator."""
    _ = proposed_gdf

    if logger:
        logger.info("Generating HTML for %s...", generator_name)

    # Normalize config for consistent access
    viz_config = _normalize_visualization_config(visualization_config)
    borehole_marker = viz_config.borehole_marker

    fig = go.Figure()

    _add_grid_cells_trace(fig, grid_gdf, visualization_config, generator_name, True)
    _add_boreholes_trace(fig, boreholes_gdf, borehole_marker)

    empty_count = grid_gdf["has_gap"].sum() if "has_gap" in grid_gdf.columns else 0
    total_count = len(grid_gdf)
    coverage_pct = 100 * (1 - empty_count / total_count) if total_count > 0 else 0

    fig.update_layout(
        title=dict(
            text=(
                f"{generator_name} Grid Analysis<br><sup>"
                f"{total_count} cells | {empty_count} gaps | "
                f"{coverage_pct:.0f}% coverage</sup>"
            ),
            x=0.5,
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Easting (m)",
            tickformat=",",
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
        ),
        yaxis=dict(
            title="Northing (m)",
            tickformat=",",
            showgrid=True,
        ),
        hovermode="closest",
        dragmode="pan",  # Default to pan mode instead of zoom
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
        ),
        width=viz_config.figure_width,
        height=viz_config.figure_height,
        plot_bgcolor="white",
    )

    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )

    if logger:
        logger.info("HTML saved: %s", output_path)

    return fig
