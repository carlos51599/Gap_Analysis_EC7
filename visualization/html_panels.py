#!/usr/bin/env python3
"""
HTML Panel Generators

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Generate HTML strings for UI panels.
Pure functions that return HTML strings - no Plotly dependency.

Panels:
- Filter panel (depth slider, checkboxes)
- Layers panel (visibility toggles)
- Coverage stats panel
- Legend panel (color legends)

Dependencies:
- json (for JavaScript data embedding)
- geopandas (for GeoDataFrame operations)
- client_scripts (for JavaScript code generation)

Navigation Guide:
- Use VS Code outline (Ctrl+Shift+O) to jump between functions
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from geopandas import GeoDataFrame

from Gap_Analysis_EC7.visualization.client_scripts import (
    generate_filter_panel_scripts,
    generate_layer_toggle_scripts,
)


# ===========================================================================
# STYLING CONSTANTS (matching plotly_utils.py liquid glass style)
# ===========================================================================

LIQUID_GLASS_STYLE = (
    "background: rgba(255, 255, 255, 0.75); "
    "backdrop-filter: blur(12px) saturate(180%); "
    "-webkit-backdrop-filter: blur(12px) saturate(180%); "
    "border: 1px solid rgba(255, 255, 255, 0.4); "
    "border-radius: 16px; "
    "box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), inset 0 1px 1px rgba(255, 255, 255, 0.6); "
    "padding: 12px 16px;"
)

PANEL_STYLE_ITEM = (
    "font-family: Arial, sans-serif; " "font-size: 11px; " "color: rgb(42, 63, 95);"
)

PANEL_STYLE_HEADER = (
    "font-family: Arial, sans-serif; "
    "font-size: 12px; "
    "font-weight: bold; "
    "color: rgb(42, 63, 95);"
)

# Default panel dimensions (can be overridden via CONFIG)
DEFAULT_LEFT_PANEL_WIDTH = 200
DEFAULT_RIGHT_PANEL_WIDTH = 220
DEFAULT_PANEL_VERTICAL_GAP = 12


# ===========================================================================
# LEGEND PANEL
# ===========================================================================


def generate_legend_panel_html(
    zones_gdf: Optional[GeoDataFrame],
    zones_config: Dict[str, Any],
    boreholes_count: int,
    borehole_marker_config: Dict[str, Any],
    panel_width: int = DEFAULT_RIGHT_PANEL_WIDTH,
) -> str:
    """
    Generate HTML for external legend panel (non-interactive, display only).

    Shows zone boundaries with their configured colors and borehole marker.
    This replaces the in-plot Plotly legend.

    Args:
        zones_gdf: GeoDataFrame with zone boundaries (must have 'Name' column)
        zones_config: Dict with per-zone config including 'boundary_color'
        boreholes_count: Number of boreholes to display in legend
        borehole_marker_config: Dict with borehole marker settings (size, color, symbol)
        panel_width: Panel width in pixels

    Returns:
        HTML string for legend panel
    """
    legend_items = []

    # Add borehole legend entry
    # Handle both typed BoreholeMarkerConfig and legacy dict access
    if hasattr(borehole_marker_config, "color"):
        bh_color = borehole_marker_config.color
    else:
        bh_color = borehole_marker_config.get("color", "black")
    legend_items.append(
        f'<div style="display: flex; align-items: center; margin: 6px 0;">'
        f'<span style="display: inline-block; width: 12px; height: 12px; '
        f"background-color: {bh_color}; border-radius: 50%; margin-right: 10px;"
        f'"></span>'
        f'<span style="font-size: 11px;">Existing Boreholes ({boreholes_count})</span>'
        f"</div>"
    )

    # Add zone boundary legend entries
    if zones_gdf is not None and not zones_gdf.empty:
        default_color = "#000000"

        # Determine which column to use for zone names
        zone_col = None
        for col in ["zone_name", "Name"]:
            if col in zones_gdf.columns:
                zone_col = col
                break

        if zone_col:
            for zone_name in sorted(zones_gdf[zone_col].unique()):
                zone_cfg = zones_config.get(zone_name, {})
                color = zone_cfg.get("boundary_color", default_color)

                # Include spacing in legend if available
                spacing_info = ""
                if "max_spacing_m" in zones_gdf.columns:
                    zone_row = zones_gdf[zones_gdf[zone_col] == zone_name].iloc[0]
                    spacing = zone_row["max_spacing_m"]
                    spacing_info = f" ({spacing:.0f}m)"

                legend_items.append(
                    f'<div style="display: flex; align-items: center; margin: 6px 0;">'
                    f'<span style="display: inline-block; width: 20px; height: 3px; '
                    f'background-color: {color}; margin-right: 8px;"></span>'
                    f'<span style="font-size: 11px;">{zone_name}{spacing_info}</span>'
                    f"</div>"
                )

    legend_html = f"""
<div id="legendPanel" style="
    {LIQUID_GLASS_STYLE}
    {PANEL_STYLE_ITEM}
    width: {panel_width}px;
">
    <div style="{PANEL_STYLE_HEADER} margin-bottom: 8px;">
        Legend
    </div>
{''.join(legend_items)}
</div>
"""
    return legend_html


# ===========================================================================
# COVERAGE STATS PANEL
# ===========================================================================


def generate_coverage_stats_panel_html(
    gap_stats: List[Dict[str, Any]],
    covered_area_ha: float,
    uncovered_area_ha: float,
    borehole_count: int,
    max_spacing: float,
    panel_width: int = DEFAULT_RIGHT_PANEL_WIDTH,
    vertical_gap: int = DEFAULT_PANEL_VERTICAL_GAP,
) -> str:
    """
    Generate HTML panel for EC7 coverage statistics.

    Args:
        gap_stats: List of dicts with gap area/centroid info
        covered_area_ha: Total covered area in hectares
        uncovered_area_ha: Total uncovered area in hectares
        borehole_count: Number of existing boreholes
        max_spacing: EC7 maximum spacing in meters
        panel_width: Panel width in pixels
        vertical_gap: Vertical gap from previous panel

    Returns:
        HTML string for coverage statistics panel (liquid glass style)
    """
    total_area_ha = covered_area_ha + uncovered_area_ha
    coverage_pct = 100 * covered_area_ha / total_area_ha if total_area_ha > 0 else 0
    gap_count = len(gap_stats)

    # Calculate gap statistics
    gap_areas = [g["area_ha"] for g in gap_stats] if gap_stats else [0]
    avg_gap_ha = sum(gap_areas) / len(gap_areas) if gap_areas else 0
    max_gap_ha = max(gap_areas) if gap_areas else 0

    # Build stat rows
    stats_rows = []
    stat_items = [
        ("Max Spacing", f"{max_spacing:.0f}m"),
        ("Coverage", f"{coverage_pct:.1f}%"),
        ("Covered Area", f"{covered_area_ha:.1f} ha"),
        ("Uncovered Area", f"{uncovered_area_ha:.1f} ha"),
        ("Gap Count", f"{gap_count}"),
        ("Avg Gap Size", f"{avg_gap_ha:.2f} ha"),
        ("Max Gap Size", f"{max_gap_ha:.2f} ha"),
        ("Boreholes", f"{borehole_count}"),
    ]

    for label, value in stat_items:
        stats_rows.append(
            f'<div style="display: flex; justify-content: space-between; margin: 4px 0;">'
            f'<span style="color: #666;">{label}:</span>'
            f'<span style="font-weight: bold;">{value}</span>'
            f"</div>"
        )

    return f"""
<div id="coverageStatsPanel" style="
    {LIQUID_GLASS_STYLE}
    {PANEL_STYLE_ITEM}
    width: {panel_width}px;
    margin-top: {vertical_gap}px;
">
    <div style="{PANEL_STYLE_HEADER} margin-bottom: 8px;">
        EC7 Coverage Statistics
    </div>
{''.join(stats_rows)}
</div>
"""


# ===========================================================================
# LAYERS PANEL
# ===========================================================================


def generate_layers_panel_html(
    bgs_layers: Optional[Dict[str, Tuple[int, int]]] = None,
    has_satellite: bool = False,
    proposed_trace_range: Optional[Tuple[int, int]] = None,
    hexgrid_trace_range: Optional[Tuple[int, int]] = None,
    uses_precomputed_coverages: bool = False,
    has_second_pass: bool = False,
    has_czrc_second_pass: bool = False,
    has_czrc_test_points: bool = False,
    has_czrc_zone_overlap: bool = False,
    has_czrc_grid: bool = False,
    has_third_pass: bool = False,
    has_third_pass_overlap: bool = False,
    has_third_pass_grid: bool = False,
    has_third_pass_test_points: bool = False,
    panel_width: int = DEFAULT_LEFT_PANEL_WIDTH,
    vertical_gap: int = DEFAULT_PANEL_VERTICAL_GAP,
) -> str:
    """
    Generate HTML for Layers panel with toggleable layer checkboxes.

    Supports:
    - Satellite imagery toggle (background layer)
    - Multiple BGS Geology layers (independent toggles)
    - Proposed boreholes (ILP-optimized locations with buffer zones)
    - Second Pass (removed/added boreholes from cross-zone redundancy check)
    - Second Pass Zone Overlap (coverage clouds and pairwise overlap regions)
    - Second Pass Grid (visibility boundary and hexagonal candidate grid)
    - Second Pass Test Points (test points used in optimization)
    - Third Pass (removed/added from cell-cell CZRC optimization)
    - Cell Overlap (cell clouds and intersection regions)
    - Cell Grid (hexagonal candidate grid for third pass)
    - Third Pass Test Points (test points used in cell-cell optimization)
    - Candidate grid (hexagonal grid overlay showing placement grid)

    Args:
        bgs_layers: Dict mapping layer_name -> (start_idx, end_idx) for BGS layers
        has_satellite: Whether satellite imagery is available
        proposed_trace_range: Tuple of (start_idx, end_idx) for proposed boreholes+buffers
        hexgrid_range_json: Tuple of (start_idx, end_idx) for candidate grid hexagons
        uses_precomputed_coverages: Whether using precomputed coverage traces
        has_second_pass: Whether second pass traces (removed/added) are available
        has_czrc_second_pass: Whether Second Pass traces (cross-zone optimization) are available
        has_czrc_test_points: Whether Second Pass test points trace is available
        has_czrc_zone_overlap: Whether Second Pass zone overlap traces (clouds/pairwise) exist
        has_czrc_grid: Whether Second Pass grid traces (visibility boundary, hex grid) exist
        has_third_pass: Whether third pass traces (cell-cell removed/added) are available
        has_third_pass_overlap: Whether third pass overlap traces (cell clouds/intersections) exist
        has_third_pass_grid: Whether third pass grid traces (cell-cell candidate grid) exist
        has_third_pass_test_points: Whether third pass test points trace is available
        panel_width: Panel width in pixels
        vertical_gap: Vertical gap from previous panel

    Returns:
        HTML string for layers panel
    """
    has_bgs = bgs_layers is not None and len(bgs_layers) > 0
    has_proposed = proposed_trace_range is not None
    has_hexgrid = hexgrid_trace_range is not None or uses_precomputed_coverages

    # Panel shown if we have any toggleable layers
    if (
        not has_bgs
        and not has_satellite
        and not has_proposed
        and not has_hexgrid
        and not has_second_pass
        and not has_czrc_second_pass
        and not has_czrc_zone_overlap
        and not has_czrc_grid
        and not has_third_pass
        and not has_third_pass_overlap
        and not has_third_pass_grid
        and not has_third_pass_test_points
    ):
        return ""

    # Convert trace indices to JSON
    bgs_layers_json = json.dumps(bgs_layers) if bgs_layers else "{}"
    proposed_range_json = (
        json.dumps(list(proposed_trace_range))
        if proposed_trace_range is not None
        else "null"
    )
    hexgrid_range_json = (
        json.dumps(list(hexgrid_trace_range))
        if hexgrid_trace_range is not None
        else "null"
    )

    # Build checkbox items
    checkbox_items = []

    # Add Satellite checkbox at the top (if available)
    if has_satellite:
        checkbox_items.append(
            """
    <label style="display: flex; align-items: center; cursor: pointer; margin: 5px 0;">
        <input type="checkbox" id="satelliteLayerCheckbox" style="margin-right: 8px;">
        <span style="font-size: 11px;">Satellite</span>
    </label>"""
        )

    # Add BGS layer checkboxes
    if has_bgs:
        for layer_name in bgs_layers.keys():
            layer_id = layer_name.lower().replace(" ", "_").replace("bgs_", "bgs")
            checkbox_items.append(
                f"""
    <label style="display: flex; align-items: center; cursor: pointer; margin: 5px 0;">
        <input type="checkbox" id="{layer_id}Checkbox" class="bgsLayerCheckbox" data-layer="{layer_name}" style="margin-right: 8px;">
        <span style="font-size: 11px;">{layer_name}</span>
    </label>"""
            )

    # Add proposed boreholes checkbox (checked by default since they're important)
    if has_proposed:
        checkbox_items.append(
            """
    <label style="display: flex; align-items: center; cursor: pointer; margin: 5px 0;">
        <input type="checkbox" id="proposedBoreholesCheckbox" checked style="margin-right: 8px;">
        <span style="font-size: 11px;">Proposed Boreholes</span>
    </label>"""
        )

    # Add candidate grid (hexagon overlay) checkbox - unchecked by default
    if has_hexgrid:
        checkbox_items.append(
            """
    <label style="display: flex; align-items: center; cursor: pointer; margin: 5px 0;">
        <input type="checkbox" id="candidateGridCheckbox" style="margin-right: 8px;">
        <span style="font-size: 11px;">Candidate Grid</span>
    </label>"""
        )

    # Add second pass checkbox (removed/added boreholes from consolidation) - unchecked by default
    if has_second_pass:
        checkbox_items.append(
            """
    <label style="display: flex; align-items: center; cursor: pointer; margin: 5px 0;">
        <input type="checkbox" id="secondPassCheckbox" style="margin-right: 8px;">
        <span style="font-size: 11px;">Second Pass</span>
    </label>"""
        )

    # Add Zone Overlap checkbox (coverage clouds and pairwise regions) - unchecked by default
    # Show when zone overlap data exists (even with skip_ilp=True)
    if has_czrc_zone_overlap:
        checkbox_items.append(
            """
    <label style="display: flex; align-items: center; cursor: pointer; margin: 5px 0;">
        <input type="checkbox" id="zoneOverlapCheckbox" style="margin-right: 8px;">
        <span style="font-size: 11px;">Zone Overlap</span>
    </label>"""
        )

    # Add Second Pass checkbox (removed/added from cross-zone optimization) - unchecked by default
    if has_czrc_second_pass:
        checkbox_items.append(
            """
    <label style="display: flex; align-items: center; cursor: pointer; margin: 5px 0;">
        <input type="checkbox" id="czrcSecondPassCheckbox" style="margin-right: 8px;">
        <span style="font-size: 11px;">Second Pass</span>
    </label>"""
        )

    # Add Second Pass grid checkbox (hexagon overlay for CZRC candidates) - unchecked by default
    # Show when grid data exists (even with skip_ilp=True)
    if has_czrc_grid:
        checkbox_items.append(
            """
    <label style="display: flex; align-items: center; cursor: pointer; margin: 5px 0;">
        <input type="checkbox" id="czrcGridCheckbox" style="margin-right: 8px;">
        <span style="font-size: 11px;">Second Pass Grid</span>
    </label>"""
        )

    # Add Second Pass test points checkbox (test points used in CZRC optimization) - unchecked by default
    if has_czrc_test_points:
        checkbox_items.append(
            """
    <label style="display: flex; align-items: center; cursor: pointer; margin: 5px 0;">
        <input type="checkbox" id="czrcTestPointsCheckbox" style="margin-right: 8px;">
        <span style="font-size: 11px;">Second Pass Test Points</span>
    </label>"""
        )

    # Add third pass overlap checkbox (cell clouds/intersections) - unchecked by default
    if has_third_pass_overlap:
        checkbox_items.append(
            """
    <label style="display: flex; align-items: center; cursor: pointer; margin: 5px 0;">
        <input type="checkbox" id="thirdPassOverlapCheckbox" style="margin-right: 8px;">
        <span style="font-size: 11px;">Cell Overlap</span>
    </label>"""
        )

    # Add third pass checkbox (cell-cell CZRC removed/added boreholes) - unchecked by default
    if has_third_pass:
        checkbox_items.append(
            """
    <label style="display: flex; align-items: center; cursor: pointer; margin: 5px 0;">
        <input type="checkbox" id="thirdPassCheckbox" style="margin-right: 8px;">
        <span style="font-size: 11px;">Third Pass</span>
    </label>"""
        )

    # Add third pass grid checkbox (cell-cell candidate grid) - unchecked by default
    if has_third_pass_grid:
        checkbox_items.append(
            """
    <label style="display: flex; align-items: center; cursor: pointer; margin: 5px 0;">
        <input type="checkbox" id="thirdPassGridCheckbox" style="margin-right: 8px;">
        <span style="font-size: 11px;">Cell Grid</span>
    </label>"""
        )

    # Add third pass test points checkbox - unchecked by default
    if has_third_pass_test_points:
        checkbox_items.append(
            """
    <label style="display: flex; align-items: center; cursor: pointer; margin: 5px 0;">
        <input type="checkbox" id="thirdPassTestPointsCheckbox" style="margin-right: 8px;">
        <span style="font-size: 11px;">Third Pass Test Points</span>
    </label>"""
        )

    # Generate JavaScript for layer toggles from client_scripts module
    layer_scripts = generate_layer_toggle_scripts(
        bgs_layers_json=bgs_layers_json,
        proposed_range_json=proposed_range_json,
        hexgrid_range_json=hexgrid_range_json,
        has_second_pass=has_second_pass,
        has_czrc_second_pass=has_czrc_second_pass,
        has_czrc_zone_overlap=has_czrc_zone_overlap,
        has_czrc_grid=has_czrc_grid,
        has_third_pass=has_third_pass,
        has_third_pass_overlap=has_third_pass_overlap,
        has_third_pass_grid=has_third_pass_grid,
        has_third_pass_test_points=has_third_pass_test_points,
    )

    checkbox_html = f"""
<div id="layersPanel" style="
    margin-top: {vertical_gap}px;
    {LIQUID_GLASS_STYLE}
    {PANEL_STYLE_ITEM}
    width: {panel_width}px;
">
    <div style="{PANEL_STYLE_HEADER} margin-bottom: 8px;">
        Layers
    </div>
{''.join(checkbox_items)}
</div>

<script>
{layer_scripts}
</script>
"""
    return checkbox_html


# ===========================================================================
# FILTERS PANEL
# ===========================================================================


def generate_filters_panel_html(
    min_depth: float,
    max_depth: float,
    boreholes_trace_idx: int,
    test_data_counts: Optional[Dict[str, int]] = None,
    depth_step_m: float = 10.0,
    panel_width: int = DEFAULT_LEFT_PANEL_WIDTH,
    vertical_gap: int = DEFAULT_PANEL_VERTICAL_GAP,
    default_filter: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate HTML for Filters panel with discrete depth slider and test data checkboxes.

    The depth slider filters to show only boreholes with Final Depth >= slider value.
    Slider uses discrete intervals starting at 0 (0, step, 2*step, ...).

    Args:
        min_depth: Minimum borehole depth in dataset (not used, slider starts at 0)
        max_depth: Maximum borehole depth in dataset
        boreholes_trace_idx: Index of boreholes trace in figure
        test_data_counts: Optional dict with counts for SPT, triaxial_total, triaxial_effective
        depth_step_m: Slider step size in meters (default: 10.0 from CONFIG)
        panel_width: Panel width in pixels
        vertical_gap: Vertical gap from previous panel
        default_filter: Optional dict with default filter values for testing mode.
                       Keys: min_depth, require_spt, require_triaxial_total, require_triaxial_effective

    Returns:
        HTML string for filters panel
    """
    # Slider always starts at 0, max rounded up to next step interval
    step = int(depth_step_m)
    min_val = 0
    max_val = int((max_depth // step + 1) * step)  # Round up to next step interval

    # Determine initial slider value (from testing mode default or 0)
    initial_depth = 0
    if default_filter:
        initial_depth = int(default_filter.get("min_depth", 0))
        # Ensure it's within valid range and snapped to step
        initial_depth = max(min_val, min(initial_depth, max_val))
        initial_depth = (initial_depth // step) * step

    # Determine initial checkbox states
    spt_checked = ""
    triaxial_total_checked = ""
    triaxial_effective_checked = ""
    if default_filter:
        if default_filter.get("require_spt", False):
            spt_checked = "checked"
        if default_filter.get("require_triaxial_total", False):
            triaxial_total_checked = "checked"
        if default_filter.get("require_triaxial_effective", False):
            triaxial_effective_checked = "checked"

    # Build test data checkbox HTML if counts provided
    test_checkboxes_html = ""
    if test_data_counts:
        spt_count = test_data_counts.get("spt", 0)
        triaxial_total_count = test_data_counts.get("triaxial_total", 0)
        triaxial_effective_count = test_data_counts.get("triaxial_effective", 0)

        test_checkboxes_html = f"""
    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(200, 200, 200, 0.5);">
        <div style="font-weight: 500; margin-bottom: 6px;">Show boreholes with:</div>
        <div style="margin: 4px 0; display: flex; align-items: center;">
            <input type="checkbox" id="filterSPT" class="test-data-filter" data-filter="spt"
                   style="margin-right: 6px; cursor: pointer; width: 14px; height: 14px;" {spt_checked}>
            <label for="filterSPT" style="cursor: pointer; user-select: none;">
                SPT Data ({spt_count} boreholes)
            </label>
        </div>
        <div style="margin: 4px 0; display: flex; align-items: center;">
            <input type="checkbox" id="filterTriaxialTotal" class="test-data-filter" data-filter="triaxial_total"
                   style="margin-right: 6px; cursor: pointer; width: 14px; height: 14px;" {triaxial_total_checked}>
            <label for="filterTriaxialTotal" style="cursor: pointer; user-select: none;">
                Triaxial Total ({triaxial_total_count} boreholes)
            </label>
        </div>
        <div style="margin: 4px 0; display: flex; align-items: center;">
            <input type="checkbox" id="filterTriaxialEffective" class="test-data-filter" data-filter="triaxial_effective"
                   style="margin-right: 6px; cursor: pointer; width: 14px; height: 14px;" {triaxial_effective_checked}>
            <label for="filterTriaxialEffective" style="cursor: pointer; user-select: none;">
                Triaxial Effective ({triaxial_effective_count} boreholes)
            </label>
        </div>
    </div>
"""

    filters_html = f"""
<div id="filtersPanel" style="
    margin-top: {vertical_gap}px;
    {LIQUID_GLASS_STYLE}
    {PANEL_STYLE_ITEM}
    width: {panel_width}px;
">
    <div style="{PANEL_STYLE_HEADER} margin-bottom: 8px;">
        Filters
    </div>
    <div style="font-size: 11px; margin-bottom: 5px;">
        <span style="font-weight: 500;">Minimum Final Depth:</span>
        <span id="depthFilterValue" style="margin-left: 5px;">≥ {initial_depth}m</span>
    </div>
    <div id="sliderContainer" style="position: relative; height: 30px; margin-bottom: 5px;">
        <div id="sliderTrack" style="
            position: absolute;
            top: 12px;
            left: 0;
            right: 0;
            height: 6px;
            background: linear-gradient(to right, #4a90d9, #ddd);
            border-radius: 3px;
        "></div>
        <input type="range" id="depthSlider" 
               min="{min_val}" max="{max_val}" value="{initial_depth}" step="{step}"
               style="position: absolute; width: 100%; top: 5px;
                      -webkit-appearance: none; background: transparent; z-index: 3; cursor: pointer;">
    </div>
    <div style="font-size: 9px; color: #666; display: flex; justify-content: space-between;">
        <span>{min_val}m</span>
        <span>{max_val}m</span>
    </div>
    <div id="boreholeCount" style="font-size: 9px; color: #666; margin-top: 5px;">
        Showing all boreholes
    </div>
    {test_checkboxes_html}
</div>

{generate_slider_styles_css()}

<script>
{generate_filter_panel_scripts(boreholes_trace_idx, min_val, max_val)}
</script>
"""
    return filters_html


# ===========================================================================
# SLIDER STYLES CSS
# ===========================================================================


def generate_slider_styles_css() -> str:
    """
    Generate CSS styles for range slider components.

    Returns:
        CSS style block as string (with <style> tags)
    """
    return """
<style>
    /* Range slider thumb styling */
    #depthSlider::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 16px;
        height: 16px;
        background: #4a90d9;
        border: 2px solid white;
        border-radius: 50%;
        cursor: pointer;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    #depthSlider::-moz-range-thumb {
        width: 16px;
        height: 16px;
        background: #4a90d9;
        border: 2px solid white;
        border-radius: 50%;
        cursor: pointer;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
</style>
"""


# ===========================================================================
# PANEL STYLES CSS
# ===========================================================================


def generate_panel_styles_css() -> str:
    """Generate CSS styles for all panels."""
    return """
    <style>
        .panel {
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(12px) saturate(180%);
            -webkit-backdrop-filter: blur(12px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.4);
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), inset 0 1px 1px rgba(255, 255, 255, 0.6);
            padding: 12px 16px;
            font-family: Arial, sans-serif;
            font-size: 11px;
            color: rgb(42, 63, 95);
        }
        
        .panel h3 {
            margin: 0 0 8px 0;
            font-size: 12px;
            font-weight: bold;
            color: rgb(42, 63, 95);
        }
        
        .filter-group {
            margin-bottom: 12px;
        }
        
        .filter-group label {
            display: block;
            margin-bottom: 4px;
            font-weight: 500;
        }
        
        .filter-checkbox {
            margin: 6px 0;
        }
        
        .layer-toggle {
            margin: 6px 0;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
        }
        
        .stat-label {
            color: #666;
        }
        
        .stat-value {
            font-weight: 600;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            margin: 6px 0;
        }
        
        .legend-swatch {
            width: 20px;
            height: 14px;
            margin-right: 8px;
            border-radius: 2px;
        }
        
        .legend-line {
            width: 20px;
            height: 3px;
            margin-right: 8px;
        }
        
        .legend-marker {
            width: 20px;
            text-align: center;
            margin-right: 8px;
            font-size: 12px;
        }
        
        .legend-label {
            flex: 1;
        }
    </style>
    """


# ===========================================================================
# BACKWARD COMPATIBILITY ALIASES
# ===========================================================================

# Aliases matching internal naming in html_builder_EC7.py
_generate_legend_panel_html = generate_legend_panel_html
_generate_coverage_stats_panel_html = generate_coverage_stats_panel_html
_generate_layers_panel_html = generate_layers_panel_html
_generate_filters_panel_html = generate_filters_panel_html
