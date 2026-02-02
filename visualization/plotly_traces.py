#!/usr/bin/env python3
"""
Plotly Trace Builder Functions

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Build Plotly trace objects for map visualization.
Each function creates a single trace or list of traces.

Key Patterns:
- Functions return go.Scatter or go.Scattergl trace objects
- All geometry conversion happens before trace building
- Visibility and styling controlled via parameters
- Merged polygon traces use None separators for multiple polygons

KNOWN LIMITATIONS:
==================
Plotly's Scattergl fill='toself' cannot render polygon holes as transparent.
When a polygon has interior rings (holes), Plotly fills them with the fill color
instead of leaving them transparent. This affects coverage/gaps visualization:

1. Gaps polygon geometry has coverage areas cut out as interior holes
2. But Plotly fills these holes with red instead of showing them as transparent
3. WORKAROUND: Render gaps without holes, then render coverage on top with
   HIGH OPACITY (85%+) so coverage completely covers the gaps underneath
4. This requires coverage_colors["covered"] opacity >= 0.85 in config.py

See GitHub issue: https://github.com/plotly/plotly.js/issues/2291

Dependencies:
- plotly.graph_objects
- numpy (for array operations)
- geopandas (for GeoDataFrame handling)

Navigation Guide:
- Use VS Code outline (Ctrl+Shift+O) to jump between functions
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from geopandas import GeoDataFrame
from shapely.geometry import Polygon

from Gap_Analysis_EC7.config_types import BoreholeMarkerConfig, ProposedMarkerConfig


# ===========================================================================
# CONFIGURATION HELPERS
# ===========================================================================


def _normalize_borehole_marker_config(
    config: Union[Dict[str, Any], BoreholeMarkerConfig],
) -> BoreholeMarkerConfig:
    """
    Normalize borehole marker config to typed BoreholeMarkerConfig.

    Supports backwards compatibility: accepts either raw dict or typed config.

    Args:
        config: Either CONFIG["visualization"]["borehole_marker"] dict or
                BoreholeMarkerConfig dataclass

    Returns:
        BoreholeMarkerConfig instance
    """
    if isinstance(config, BoreholeMarkerConfig):
        return config
    return BoreholeMarkerConfig.from_dict(config)


def _normalize_proposed_marker_config(
    config: Union[Dict[str, Any], ProposedMarkerConfig],
) -> ProposedMarkerConfig:
    """
    Normalize proposed marker config to typed ProposedMarkerConfig.

    Supports backwards compatibility: accepts either raw dict or typed config.

    Args:
        config: Either CONFIG["visualization"]["proposed_marker"] dict or
                ProposedMarkerConfig dataclass

    Returns:
        ProposedMarkerConfig instance
    """
    if isinstance(config, ProposedMarkerConfig):
        return config
    return ProposedMarkerConfig.from_dict(config)


# ===========================================================================
# HEXAGON GRID OVERLAY
# ===========================================================================


def build_hexagon_grid_trace(
    hexagon_polygons: List[Polygon],
    grid_color: str = "rgba(100, 100, 255, 0.4)",
    grid_line_width: float = 1.0,
    visible: bool = False,
    name: str = "Candidate Grid",
) -> go.Scattergl:
    """
    Build a single trace containing all hexagon outlines.

    Uses None separators to render multiple disconnected polygons efficiently.

    Args:
        hexagon_polygons: List of Shapely Polygon objects (hexagons)
        grid_color: Line color for hexagon outlines (RGBA string)
        grid_line_width: Width of hexagon outline lines
        visible: Initial visibility state (default False = hidden)
        name: Trace name

    Returns:
        go.Scattergl trace with all hexagons
    """
    if not hexagon_polygons:
        return None

    # Combine all hexagon outlines into a single trace for performance
    # Using None separators between polygons
    all_x = []
    all_y = []

    for poly in hexagon_polygons:
        coords = list(poly.exterior.coords)
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        # Close the polygon
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        # Add to combined arrays with None separator
        all_x.extend(x_coords)
        all_x.append(None)
        all_y.extend(y_coords)
        all_y.append(None)

    return go.Scattergl(
        x=all_x,
        y=all_y,
        mode="lines",
        line=dict(
            color=grid_color,
            width=grid_line_width,
        ),
        hoverinfo="skip",  # No hover for grid overlay
        name=name,
        legendgroup="hexgrid",
        showlegend=False,
        visible=visible,
    )


# ===========================================================================
# ZONE BOUNDARY TRACES
# ===========================================================================


def build_zone_boundary_traces(
    zones_gdf: GeoDataFrame,
    zones_config: Dict[str, Any],
    zone_defaults: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> List[go.Scatter]:
    """
    Build zone boundary outline traces.

    Each zone gets a colored outline based on its config.

    Args:
        zones_gdf: GeoDataFrame with zone polygon geometries with 'Name' column
        zones_config: Dict with per-zone config including 'boundary_color'
        zone_defaults: Dict with default settings including 'boundary_linewidth'
        logger: Optional logger instance

    Returns:
        List of go.Scatter traces for zone boundaries
    """
    if zones_gdf is None or zones_gdf.empty:
        return []

    linewidth = zone_defaults.get("boundary_linewidth", 3.0)
    default_color = "#000000"  # Black default
    traces = []
    zone_idx = 0

    for _, row in zones_gdf.iterrows():
        zone_name = row.get("Name", f"Zone {zone_idx + 1}")
        geom = row.geometry

        # Get color from config
        zone_cfg = zones_config.get(zone_name, {})
        color = zone_cfg.get("boundary_color", default_color)

        # Handle both Polygon and MultiPolygon
        polygons = []
        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)

        # Track if this is the first polygon for this zone (for legend)
        first_poly_for_zone = True

        for poly in polygons:
            # Get exterior ring coordinates
            x_coords = list(poly.exterior.coords.xy[0])
            y_coords = list(poly.exterior.coords.xy[1])

            # Close the polygon
            x_coords.append(x_coords[0])
            y_coords.append(y_coords[0])

            trace = go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(
                    color=color,
                    width=linewidth,
                ),
                hovertemplate=(
                    f"<b>{zone_name}</b><br>"
                    "Easting: %{x:,.0f}<br>"
                    "Northing: %{y:,.0f}<br>"
                    "<extra></extra>"
                ),
                name=zone_name,
                legendgroup=f"zone_{zone_name}",
                showlegend=first_poly_for_zone,
                visible=True,
            )
            traces.append(trace)
            first_poly_for_zone = False

        zone_idx += 1

    if logger and traces:
        logger.info(f"   Built {len(traces)} zone boundary traces")

    return traces


# ===========================================================================
# BOREHOLE MARKER TRACES
# ===========================================================================


def build_boreholes_trace(
    boreholes_gdf: GeoDataFrame,
    marker_config: Union[Dict[str, Any], BoreholeMarkerConfig],
    test_data_locations: Optional[Dict[str, set]] = None,
    visible: bool = True,
    name: str = "Existing Boreholes",
) -> Optional[go.Scattergl]:
    """
    Build existing borehole markers trace.

    Boreholes are always visible and shown in legend.
    LocationID, Final Depth, and test data flags stored in customdata for filtering.

    Args:
        boreholes_gdf: GeoDataFrame with borehole points
        marker_config: Dict or BoreholeMarkerConfig with marker styling
        test_data_locations: Optional dict mapping test type to set of LocationIDs
            {'spt': {'BH001', ...}, 'triaxial_total': {...}, 'triaxial_effective': {...}}
        visible: Initial visibility
        name: Trace name

    Returns:
        go.Scattergl trace or None if empty
    """
    if boreholes_gdf.empty:
        return None

    # Normalize to typed config
    config = _normalize_borehole_marker_config(marker_config)

    # Store LocationID as string for JavaScript filtering
    # Support both "LocationID" and "Location ID" column names
    if "LocationID" in boreholes_gdf.columns:
        location_ids = boreholes_gdf["LocationID"].astype(str).values
    elif "Location ID" in boreholes_gdf.columns:
        location_ids = boreholes_gdf["Location ID"].astype(str).values
    else:
        location_ids = [f"BH-{i}" for i in range(len(boreholes_gdf))]

    # Get Final Depth values (default to 0 if missing)
    if "Final Depth" in boreholes_gdf.columns:
        final_depths = boreholes_gdf["Final Depth"].fillna(0).astype(float).values
    else:
        final_depths = [0.0] * len(boreholes_gdf)

    # Build test data flags for each borehole
    spt_set = test_data_locations.get("spt", set()) if test_data_locations else set()
    triaxial_total_set = (
        test_data_locations.get("triaxial_total", set())
        if test_data_locations
        else set()
    )
    triaxial_effective_set = (
        test_data_locations.get("triaxial_effective", set())
        if test_data_locations
        else set()
    )

    # Combine all data into customdata
    # Format: [[id, depth, has_spt, has_triaxial_total, has_triaxial_effective], ...]
    customdata = []
    for lid, fd in zip(location_ids, final_depths):
        has_spt = 1 if lid in spt_set else 0
        has_triaxial_total = 1 if lid in triaxial_total_set else 0
        has_triaxial_effective = 1 if lid in triaxial_effective_set else 0
        customdata.append(
            [lid, fd, has_spt, has_triaxial_total, has_triaxial_effective]
        )

    return go.Scattergl(
        x=boreholes_gdf.geometry.x,
        y=boreholes_gdf.geometry.y,
        mode="markers",
        marker=dict(
            size=config.size,
            color=config.color,
            symbol=config.symbol,
            line=dict(
                color=config.line_color,
                width=config.line_width,
            ),
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Easting: %{x:,.0f}<br>"
            "Northing: %{y:,.0f}<br>"
            "Final Depth: %{customdata[1]:.1f}m<br>"
            "<extra></extra>"
        ),
        name=name,
        legendgroup="boreholes",
        showlegend=False,
        visible=visible,
    )


def build_proposed_boreholes_trace(
    proposed_locations: List[Dict[str, Any]],
    marker_config: Union[Dict[str, Any], ProposedMarkerConfig],
    visible: bool = True,
    name: str = "Proposed Boreholes",
) -> Optional[go.Scattergl]:
    """
    Build proposed borehole markers trace.

    Args:
        proposed_locations: List of dicts with 'x', 'y' coordinates
        marker_config: Dict or ProposedMarkerConfig with marker styling
        visible: Initial visibility
        name: Trace name

    Returns:
        go.Scattergl trace or None if empty
    """
    if not proposed_locations:
        return None

    # Normalize to typed config
    config = _normalize_proposed_marker_config(marker_config)

    x_coords = [loc["x"] for loc in proposed_locations]
    y_coords = [loc["y"] for loc in proposed_locations]

    return go.Scattergl(
        x=x_coords,
        y=y_coords,
        mode="markers",
        marker=dict(
            size=config.size,
            color=config.color,
            symbol=config.symbol,
            line=dict(
                color="rgba(0,0,0,0)",  # ProposedMarkerConfig doesn't have line_color
                width=0,
            ),
        ),
        hovertemplate=(
            "<b>Proposed Borehole</b><br>"
            "Easting: %{x:,.0f}<br>"
            "Northing: %{y:,.0f}<br>"
            "<extra></extra>"
        ),
        name=name,
        legendgroup="proposed",
        showlegend=False,
        visible=visible,
    )


# ===========================================================================
# SINGLE POLYGON TRACES
# ===========================================================================


def build_single_polygon_trace(
    polygon: Polygon,
    name: str,
    fill_color: str = "rgba(0, 100, 255, 0.3)",
    line_color: str = "rgba(0, 0, 0, 0.5)",
    line_width: float = 1.0,
    visible: bool = True,
    hover_text: Optional[str] = None,
    legend_group: Optional[str] = None,
    show_legend: bool = False,
) -> go.Scatter:
    """
    Build a Plotly trace for a single polygon.

    Args:
        polygon: Shapely Polygon object
        name: Trace name for legend
        fill_color: Fill color with alpha
        line_color: Outline color
        line_width: Outline width
        visible: Initial visibility
        hover_text: Text to show on hover
        legend_group: Group for legend toggling
        show_legend: Whether to show in legend

    Returns:
        Plotly Scatter trace
    """
    # Extract coordinates
    exterior = polygon.exterior.coords
    x_coords = [c[0] for c in exterior]
    y_coords = [c[1] for c in exterior]

    return go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="lines",
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=line_width),
        name=name,
        visible=visible,
        hovertext=hover_text,
        hoverinfo="text" if hover_text else "skip",
        legendgroup=legend_group,
        showlegend=show_legend,
    )


# ===========================================================================
# GRID CELL TRACES
# ===========================================================================


def add_grid_cells_trace(
    fig: go.Figure,
    grid_gdf: GeoDataFrame,
    visualization_config: Dict[str, Any],
    generator_name: str,
    visible: bool = True,
) -> int:
    """
    Add grid cell polygons to Plotly figure.

    Grid cells are NOT shown in legend (only boreholes appear in legend).

    CRITICAL: Grid must be sorted by cell_id to match the ordering used in
    compute_cell_borehole_presence_per_geology(), which produces boolean arrays
    indexed by cell_id order.

    Args:
        fig: Plotly figure to add traces to
        grid_gdf: GeoDataFrame with grid cell geometries
        visualization_config: Visualization settings
        generator_name: Name of the grid generator
        visible: Initial visibility

    Returns:
        Number of traces added
    """
    if grid_gdf.empty:
        return 0

    # CRITICAL: Sort by cell_id to match presence array ordering
    if "cell_id" in grid_gdf.columns:
        grid_gdf = grid_gdf.sort_values("cell_id").reset_index(drop=True)

    # Pre-extract columns
    if "borehole_count" in grid_gdf.columns:
        bh_counts = np.asarray(grid_gdf["borehole_count"].fillna(0).astype(int))
        has_borehole = bh_counts > 0
    elif "coverage_status" in grid_gdf.columns:
        has_borehole = np.asarray(grid_gdf["coverage_status"] == "covered")
        bh_counts = has_borehole.astype(int)
    else:
        has_borehole = np.zeros(len(grid_gdf), dtype=bool)
        bh_counts = np.zeros(len(grid_gdf), dtype=int)

    cell_ids = (
        np.asarray(grid_gdf["cell_id"])
        if "cell_id" in grid_gdf.columns
        else np.arange(len(grid_gdf))
    )
    zones = (
        np.asarray(grid_gdf["zone"])
        if "zone" in grid_gdf.columns
        else np.array(["Unknown"] * len(grid_gdf))
    )
    nearest_dist = (
        np.asarray(grid_gdf["nearest_bh_dist"])
        if "nearest_bh_dist" in grid_gdf.columns
        else np.zeros(len(grid_gdf))
    )
    geometries = grid_gdf.geometry.values

    # Pre-compute colors
    cell_colors = visualization_config.get("cell_colors", {})
    color_has_bh = cell_colors.get("covered", "rgba(18, 141, 38, 1)")
    color_empty = cell_colors.get("uncovered", "rgba(255, 100, 100, 1)")
    colors = np.where(has_borehole, color_has_bh, color_empty)

    line_style = dict(
        color=visualization_config.get(
            "grid_line_color",
            visualization_config.get("cell_line_color", "rgba(0, 0, 0, 0.3)"),
        ),
        width=visualization_config.get(
            "grid_line_width", visualization_config.get("cell_line_width", 1)
        ),
    )

    traces_added = 0

    for i, poly in enumerate(geometries):
        if poly.geom_type != "Polygon":
            continue

        coords = np.array(poly.exterior.coords)
        status = "Covered" if has_borehole[i] else "Uncovered"
        hover_text = (
            f"<b>{generator_name}</b><br>"
            f"Cell {cell_ids[i]}<br>"
            f"Zone: {zones[i]}<br>"
            f"Status: {status}<br>"
            f"Nearest BH: {nearest_dist[i]:.1f}m<br>"
            "<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="lines",
                fill="toself",
                fillcolor=colors[i],
                line=line_style,
                name=generator_name,
                legendgroup=generator_name,
                showlegend=False,
                visible=visible,
                hovertemplate=hover_text,
            )
        )
        traces_added += 1

    return traces_added


def add_hexagon_grid_overlay(
    fig: go.Figure,
    hexagon_polygons: List[Polygon],
    grid_color: str = "rgba(100, 100, 255, 0.4)",
    grid_line_width: float = 1.0,
    visible: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, int]:
    """
    Add hexagonal candidate grid overlay to the figure.

    Args:
        fig: Plotly figure to add traces to
        hexagon_polygons: List of Shapely Polygon objects (hexagons)
        grid_color: Line color for hexagon outlines
        grid_line_width: Width of hexagon outline lines
        visible: Initial visibility state
        logger: Optional logger

    Returns:
        Tuple of (start_trace_index, end_trace_index)
    """
    if not hexagon_polygons:
        return (0, 0)

    start_idx = len(fig.data)

    trace = build_hexagon_grid_trace(
        hexagon_polygons=hexagon_polygons,
        grid_color=grid_color,
        grid_line_width=grid_line_width,
        visible=visible,
    )

    if trace is not None:
        fig.add_trace(trace)

    end_idx = len(fig.data)

    if logger:
        logger.info(
            f"   🔷 Added hexagon grid overlay: {len(hexagon_polygons)} hexagons "
            f"(trace indices {start_idx}-{end_idx})"
        )

    return (start_idx, end_idx)


def add_zone_boundary_traces(
    fig: go.Figure,
    zones_gdf: GeoDataFrame,
    zones_config: Dict[str, Any],
    zone_defaults: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Add zone boundary outline traces to the figure.

    Args:
        fig: Plotly figure to add traces to
        zones_gdf: GeoDataFrame with zone polygon geometries
        zones_config: Dict with per-zone config
        zone_defaults: Dict with default settings
        logger: Optional logger

    Returns:
        Number of traces added
    """
    traces = build_zone_boundary_traces(zones_gdf, zones_config, zone_defaults, logger)
    for trace in traces:
        fig.add_trace(trace)
    return len(traces)


def add_boreholes_trace(
    fig: go.Figure,
    boreholes_gdf: GeoDataFrame,
    marker_config: Union[Dict[str, Any], BoreholeMarkerConfig],
    test_data_locations: Optional[Dict[str, set]] = None,
) -> int:
    """
    Add existing borehole markers to Plotly figure.

    Args:
        fig: Plotly figure to add trace to
        boreholes_gdf: GeoDataFrame with borehole points
        marker_config: Dict or BoreholeMarkerConfig with marker styling
        test_data_locations: Optional dict mapping test type to set of LocationIDs

    Returns:
        1 if trace added, 0 otherwise
    """
    trace = build_boreholes_trace(
        boreholes_gdf=boreholes_gdf,
        marker_config=marker_config,
        test_data_locations=test_data_locations,
    )
    if trace is not None:
        fig.add_trace(trace)
        return 1
    return 0


# ===========================================================================
# PLOTLY LAYOUT BUILDERS
# ===========================================================================


# ===========================================================================
# COVERAGE ZONE TRACE BUILDERS (moved from coverage_zones.py)
# ===========================================================================


def _extract_coverage_coords(
    geometry: Any,
    include_holes: bool = False,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Extract merged x,y coordinates from Polygon/MultiPolygon with None separators.

    IMPORTANT: By default, interior holes are NOT included because Plotly's
    fill='toself' does NOT properly render holes as transparent cut-outs.
    Instead, it fills each closed path independently, which causes holes to
    be filled with the same color as the exterior.

    When coverage and gaps polygons are rendered with the correct z-order
    (gaps underneath, coverage on top), the coverage polygon visually
    "covers" the gap areas, making explicit holes unnecessary.

    Args:
        geometry: Shapely Polygon or MultiPolygon
        include_holes: If True, include interior ring coordinates (rarely needed).
                       Default False to avoid Plotly hole-filling bug.

    Returns:
        Tuple of (x_coords, y_coords) lists with None separators between polygons
    """
    from shapely.geometry.polygon import orient

    x_coords: List[Optional[float]] = []
    y_coords: List[Optional[float]] = []

    if geometry is None or geometry.is_empty:
        return x_coords, y_coords

    polys = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)

    for i, poly in enumerate(polys):
        if poly.is_empty or poly.area < 1:
            continue

        # CRITICAL: Reorient polygon for Plotly's WebGL renderer
        # sign=-1.0 makes exterior CW and holes CCW (what Plotly expects)
        oriented_poly = orient(poly, sign=-1.0)

        # Exterior ring (now CW)
        x, y = oriented_poly.exterior.xy
        x_coords.extend(list(x))
        y_coords.extend(list(y))

        # Interior rings - only include if explicitly requested
        # NOTE: Plotly's fill='toself' fills holes with color, not transparent!
        if include_holes:
            for interior in oriented_poly.interiors:
                x_coords.append(None)
                y_coords.append(None)
                int_x, int_y = interior.xy
                x_coords.extend(list(int_x))
                y_coords.extend(list(int_y))

        # Separator between polygons
        if i < len(polys) - 1:
            x_coords.append(None)
            y_coords.append(None)

    return x_coords, y_coords


def build_coverage_polygon_trace(
    geometry: Any,
    name: str,
    fill_color: str,
    visible: bool = False,
    legend_group: Optional[str] = None,
    show_legend: bool = True,
    line_color: Optional[str] = None,
    line_width: int = 1,
) -> go.Scattergl:
    """
    Build single merged Scattergl trace from Shapely Polygon/MultiPolygon.

    Uses None separators for efficient visibility toggling via Plotly.restyle().
    Designed for coverage zone visualization (covered/uncovered areas).

    Note: This is different from build_merged_polygon_trace which works with
    GeoDataFrame grid cells + boolean mask.

    Args:
        geometry: Shapely Polygon or MultiPolygon
        name: Trace name for legend
        fill_color: RGBA fill color (e.g. "rgba(0, 200, 0, 0.3)")
        visible: Initial visibility state
        legend_group: Legend group name (defaults to name)
        show_legend: Whether to show in legend
        line_color: Optional RGBA line color for outline. If None, auto-derives
                    from fill_color by increasing alpha.
        line_width: Line width in pixels for outline (default: 1)

    Returns:
        go.Scattergl trace with merged polygon coordinates
    """
    x_coords, y_coords = _extract_coverage_coords(geometry)
    # Use provided line_color or auto-derive from fill_color
    if line_color is None:
        line_color = fill_color.replace("0.3", "0.6").replace("0.5", "0.8")

    return go.Scattergl(
        x=x_coords,
        y=y_coords,
        fill="toself",
        fillcolor=fill_color,
        mode="lines",
        line=dict(width=line_width, color=line_color),
        name=name,
        legendgroup=legend_group or name,
        showlegend=show_legend,
        visible=visible,
        hoverinfo="skip",
    )


def build_coverage_marker_trace(
    coordinates: List[Dict[str, float]],
    name: str,
    marker_color: str,
    marker_size: int = 14,
    marker_symbol: str = "x",
    visible: bool = False,
    show_legend: bool = True,
) -> go.Scattergl:
    """
    Build a single merged marker trace for proposed boreholes.

    Args:
        coordinates: List of {"x": float, "y": float, "source_pass"?: str} dicts
            source_pass is optional - if provided, shown in tooltip
        name: Trace name for legend
        marker_color: RGBA marker color
        marker_size: Marker size in pixels
        marker_symbol: Plotly marker symbol
        visible: Initial visibility state
        show_legend: Whether to show in legend

    Returns:
        go.Scattergl trace with all markers
    """
    if not coordinates:
        return go.Scattergl(
            x=[],
            y=[],
            mode="markers",
            marker=dict(size=marker_size, color=marker_color, symbol=marker_symbol),
            name=name,
            showlegend=show_legend,
            visible=visible,
            hoverinfo="skip",
        )

    x_coords = [c["x"] for c in coordinates]
    y_coords = [c["y"] for c in coordinates]
    # Include source_pass in customdata: [index, source_pass]
    customdata = [
        [i + 1, c.get("source_pass", "First Pass")] for i, c in enumerate(coordinates)
    ]

    return go.Scattergl(
        x=x_coords,
        y=y_coords,
        mode="markers",
        marker=dict(
            size=marker_size,
            color=marker_color,
            symbol=marker_symbol,
            line=dict(width=2, color=marker_color),
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>Proposed Borehole #%{customdata[0]}</b><br>"
            "Easting: %{x:,.0f}<br>"
            "Northing: %{y:,.0f}<br>"
            "Source: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
        name=name,
        legendgroup="proposed",
        showlegend=show_legend,
        visible=visible,
    )


def build_borehole_circles_trace(
    coordinates: List[Dict[str, float]],
    buffer_radius: float,
    name: str,
    line_color: str = "rgba(0, 100, 255, 0.7)",
    line_width: int = 2,
    visible: bool = False,
    show_legend: bool = True,
) -> go.Scattergl:
    """
    Build outline-only circles showing proposed borehole coverage radii.

    Unlike build_coverage_buffer_trace(), this creates individual circles
    without fill, showing only the outline of each borehole's coverage radius.
    The circles are NOT merged, preserving individual borehole identity.

    Each point includes customdata with:
    - borehole_id: Unique identifier for the borehole (index-based)
    - center_x: Circle center X coordinate
    - center_y: Circle center Y coordinate
    - radius: Circle radius
    This enables interactive removal of individual circles via JavaScript.

    Args:
        coordinates: List of {"x": float, "y": float, "coverage_radius"?: float}
                     borehole locations. coverage_radius is optional per point.
        buffer_radius: Default radius in meters (fallback if coverage_radius not set)
        name: Trace name for legend
        line_color: RGBA line color for circle outlines
        line_width: Line width in pixels
        visible: Initial visibility state
        show_legend: Whether to show in legend

    Returns:
        go.Scattergl trace with all circle outlines (no fill)
    """
    if not coordinates or buffer_radius <= 0:
        return go.Scattergl(
            x=[],
            y=[],
            mode="lines",
            line=dict(color=line_color, width=line_width),
            customdata=[],
            name=name,
            showlegend=show_legend,
            visible=visible,
            hoverinfo="skip",
        )

    # Build individual circle outlines using None separators
    # This approach draws each circle separately without merging
    all_x: List[Optional[float]] = []
    all_y: List[Optional[float]] = []
    all_customdata: List[Optional[List]] = []

    # Number of points per circle for smooth appearance
    num_points = 64

    for borehole_id, coord in enumerate(coordinates):
        cx = coord["x"]
        cy = coord["y"]
        radius = coord.get("coverage_radius", buffer_radius)

        # Generate circle points
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
        circle_x = cx + radius * np.cos(angles)
        circle_y = cy + radius * np.sin(angles)

        # Add circle coordinates
        all_x.extend(circle_x.tolist())
        all_y.extend(circle_y.tolist())

        # Add customdata for each point: [borehole_id, center_x, center_y, radius]
        # All points in the same circle share the same borehole_id
        for _ in range(len(circle_x)):
            all_customdata.append([borehole_id, cx, cy, radius])

        # Add None separator for next circle
        all_x.append(None)
        all_y.append(None)
        all_customdata.append(None)

    return go.Scattergl(
        x=all_x,
        y=all_y,
        mode="lines",
        line=dict(color=line_color, width=line_width),
        fill=None,  # No fill - outline only
        customdata=all_customdata,
        name=name,
        legendgroup="borehole_circles",
        showlegend=show_legend,
        visible=visible,
        hovertemplate=(
            "<b>Proposed BH #%{customdata[0]}</b><br>"
            "Center: (%{customdata[1]:,.0f}, %{customdata[2]:,.0f})<br>"
            "Radius: %{customdata[3]:.1f}m<br>"
            "<i>Shift+Click to remove</i><br>"
            "<extra></extra>"
        ),
    )


def build_coverage_buffer_trace(
    coordinates: List[Dict[str, float]],
    buffer_radius: float,
    name: str,
    buffer_color: str = "rgba(4, 0, 255, 0.3)",
    visible: bool = False,
    show_legend: bool = True,
    line_color: Optional[str] = None,
    line_width: int = 1,
) -> go.Scattergl:
    """
    Build a merged trace for proposed borehole coverage buffer circles.

    Supports per-borehole radii: if a coordinate dict has a "coverage_radius"
    key, that value is used for that borehole's buffer. Otherwise, the
    buffer_radius parameter is used as a fallback.

    Args:
        coordinates: List of {"x": float, "y": float, "coverage_radius"?: float}
                     borehole locations. coverage_radius is optional per point.
        buffer_radius: Default radius in meters for buffer circles (fallback)
        name: Trace name for legend
        buffer_color: RGBA buffer fill color
        visible: Initial visibility state
        show_legend: Whether to show in legend
        line_color: Optional RGBA line color for buffer outline. If None,
                    auto-derives from buffer_color.
        line_width: Line width in pixels for buffer outline (default: 1)

    Returns:
        go.Scattergl trace with all buffer circles merged
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union

    if not coordinates or buffer_radius <= 0:
        return go.Scattergl(
            x=[],
            y=[],
            fill="toself",
            fillcolor=buffer_color,
            mode="lines",
            name=name,
            showlegend=show_legend,
            visible=visible,
            hoverinfo="skip",
        )

    # Create buffer circles using per-borehole radius if available
    buffer_circles = [
        Point(c["x"], c["y"]).buffer(c.get("coverage_radius", buffer_radius))
        for c in coordinates
    ]
    buffer_union = unary_union(buffer_circles)

    # Use the coverage polygon builder
    return build_coverage_polygon_trace(
        geometry=buffer_union,
        name=name,
        fill_color=buffer_color,
        visible=visible,
        legend_group="proposed_coverage",
        show_legend=show_legend,
        line_color=line_color,
        line_width=line_width,
    )


# ===========================================================================
# COVERAGE ZONE FIGURE MODIFIERS (moved from coverage_zones.py)
# ===========================================================================


def add_coverage_zone_traces(
    fig: go.Figure,
    covered_union: Any,
    uncovered_gaps: Any,
    gap_stats: List[Dict[str, Any]],  # Kept for API compatibility, unused here
    covered_color: str = "rgba(0, 200, 0, 0.3)",
    gap_color: str = "rgba(255, 0, 0, 0.5)",
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Add coverage zone and gap traces to Plotly figure.

    Note: Proposed borehole markers are now added separately via
    add_proposed_borehole_traces() for better layering control.

    Args:
        fig: Plotly figure to add traces to
        covered_union: Shapely geometry of covered area
        uncovered_gaps: Shapely geometry of uncovered gaps
        gap_stats: List of gap statistics dicts (kept for API compatibility)
        covered_color: RGBA color for covered area
        gap_color: RGBA color for uncovered gaps
        logger: Optional logger

    Returns:
        Number of traces added
    """
    from shapely.geometry.polygon import orient

    traces_added = 0

    def polygon_to_coords(polygon):
        """Extract x, y coordinates from shapely polygon, including interior holes."""
        if polygon.is_empty:
            return [], []

        oriented_poly = orient(polygon, sign=-1.0)
        x_coords = []
        y_coords = []

        ext_x, ext_y = oriented_poly.exterior.xy
        x_coords.extend(list(ext_x))
        y_coords.extend(list(ext_y))

        for interior in oriented_poly.interiors:
            x_coords.append(None)
            y_coords.append(None)
            int_x, int_y = interior.xy
            x_coords.extend(list(int_x))
            y_coords.extend(list(int_y))

        return x_coords, y_coords

    def add_polygon_trace(geom, name, color, show_legend=True):
        nonlocal traces_added

        if geom.is_empty:
            return

        polys = []
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)

        for i, poly in enumerate(polys):
            if poly.is_empty or poly.area < 1:
                continue

            x, y = polygon_to_coords(poly)
            if not x:
                continue

            line_color = color.replace("0.3", "0.6").replace("0.5", "0.8")

            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    fill="toself",
                    fillcolor=color,
                    mode="lines",
                    line=dict(width=1, color=line_color),
                    name=name if (i == 0 and show_legend) else None,
                    legendgroup=name,
                    showlegend=(i == 0 and show_legend),
                    hoverinfo="skip",
                )
            )
            traces_added += 1

    if not covered_union.is_empty:
        add_polygon_trace(
            covered_union, "Coverage Zone", covered_color, show_legend=True
        )
        if logger:
            logger.info(f"   Added coverage zone trace")

    if not uncovered_gaps.is_empty:
        add_polygon_trace(uncovered_gaps, "Uncovered Gaps", gap_color, show_legend=True)
        if logger:
            logger.info(f"   Added gap traces")

    return traces_added


def _build_proposed_coverage_per_zone(
    proposed_boreholes: List[Dict[str, float]],
    zones_gdf: GeoDataFrame,
    buffer_color: str = "rgba(4, 0, 255, 0.3)",
    logger: Optional[logging.Logger] = None,
    line_color: Optional[str] = None,
    line_width: int = 1,
) -> List[go.Scattergl]:
    """
    Build per-zone coverage traces for proposed boreholes.

    For each zone, buffers proposed boreholes by that zone's max_spacing_m
    and clips to the zone boundary. This ensures proposed coverage reflects
    the coverage requirement of the TARGET zone, not the borehole's origin.

    Args:
        proposed_boreholes: List of {"x", "y", "coverage_radius"} dicts
        zones_gdf: GeoDataFrame with zone geometries and max_spacing_m column
        buffer_color: RGBA fill color for buffers
        logger: Optional logger
        line_color: Optional RGBA line color for outline. If None, auto-derives
                    from buffer_color.
        line_width: Line width in pixels for outline (default: 1)

    Returns:
        List of Scattergl traces (one merged trace per zone with coverage)
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union

    if not proposed_boreholes or zones_gdf is None or zones_gdf.empty:
        return []

    # Check if per-zone spacing is available
    has_per_zone = "max_spacing_m" in zones_gdf.columns

    traces = []
    # Use provided line_color or auto-derive from buffer_color
    if line_color is None:
        line_color = buffer_color.replace("0.3", "0.6").replace("0.25", "0.6")

    for idx, zone_row in zones_gdf.iterrows():
        zone_geom = zone_row.geometry
        if zone_geom is None or zone_geom.is_empty:
            continue

        # Get zone spacing (fallback to 100m if not available)
        zone_spacing = zone_row.get("max_spacing_m", 100.0) if has_per_zone else 100.0

        # Expand zone to include boreholes that could contribute coverage
        expanded_zone = zone_geom.buffer(zone_spacing)

        # Find proposed boreholes near this zone
        nearby_boreholes = [
            bh
            for bh in proposed_boreholes
            if Point(bh["x"], bh["y"]).within(expanded_zone)
        ]

        if not nearby_boreholes:
            continue

        # Buffer each borehole by ZONE's spacing (not borehole's origin zone)
        borehole_buffers = [
            Point(bh["x"], bh["y"]).buffer(zone_spacing) for bh in nearby_boreholes
        ]

        # Union and clip to zone
        coverage_union = unary_union(borehole_buffers).intersection(zone_geom)

        if coverage_union.is_empty:
            continue

        # Convert to trace
        polys = []
        if coverage_union.geom_type == "Polygon":
            polys = [coverage_union]
        elif coverage_union.geom_type == "MultiPolygon":
            polys = list(coverage_union.geoms)

        for i, poly in enumerate(polys):
            if poly.is_empty or poly.area < 1:
                continue

            x_coords, y_coords = list(poly.exterior.xy[0]), list(poly.exterior.xy[1])
            if not x_coords:
                continue

            # Only first trace in first zone shows legend
            show_legend = (len(traces) == 0) and (i == 0)

            traces.append(
                go.Scattergl(
                    x=x_coords,
                    y=y_coords,
                    fill="toself",
                    fillcolor=buffer_color,
                    mode="lines",
                    line=dict(width=line_width, color=line_color),
                    name="Proposed Coverage" if show_legend else None,
                    legendgroup="proposed_coverage",
                    showlegend=show_legend,
                    hoverinfo="skip",
                )
            )

    if logger and traces:
        logger.info(f"   Built {len(traces)} per-zone proposed coverage traces")

    return traces


def add_proposed_borehole_traces(
    fig: go.Figure,
    proposed_boreholes: List[Dict[str, float]],
    gap_stats: Optional[List[Dict[str, Any]]] = None,
    marker_color: str = "rgba(0, 100, 255, 0.9)",
    marker_size: int = 14,
    marker_symbol: str = "x",
    coverage_radius: Optional[float] = None,
    buffer_color: str = "rgba(4, 0, 255, 0.3)",
    zones_gdf: Optional[GeoDataFrame] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, Optional[int], Optional[int]]:
    """
    Add proposed/optimized borehole location markers and coverage buffers to Plotly figure.

    These markers indicate optimal locations for new boreholes to fill coverage gaps.

    Args:
        fig: Plotly figure to add trace to
        proposed_boreholes: List of {"x": float, "y": float} borehole locations
        gap_stats: Optional gap statistics for enhanced hover info
        marker_color: RGBA color for markers
        marker_size: Marker size in pixels
        marker_symbol: Plotly marker symbol (default "x" for cross)
        coverage_radius: Radius in meters for coverage buffer circles
        buffer_color: RGBA color for buffer fill
        logger: Optional logger

    Returns:
        Tuple of (traces_added, start_trace_idx, end_trace_idx)
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union

    if not proposed_boreholes:
        return 0, None, None

    traces_added = 0
    start_trace_idx = len(fig.data)

    # Add buffer circles first (so markers render on top)
    # Use per-zone coverage if zones_gdf provided, otherwise single-radius buffers
    if zones_gdf is not None and not zones_gdf.empty:
        # Per-zone coverage: buffer by each zone's max_spacing_m and clip to zone
        per_zone_traces = _build_proposed_coverage_per_zone(
            proposed_boreholes, zones_gdf, buffer_color, logger
        )
        for trace in per_zone_traces:
            fig.add_trace(trace)
            traces_added += 1
    elif coverage_radius is not None and coverage_radius > 0:
        # Fallback: single radius for all boreholes (legacy behavior)
        buffer_circles = [
            Point(bh["x"], bh["y"]).buffer(bh.get("coverage_radius", coverage_radius))
            for bh in proposed_boreholes
        ]
        buffer_union = unary_union(buffer_circles)
        line_color = buffer_color.replace("0.3", "0.6")

        polys = []
        if buffer_union.geom_type == "Polygon":
            polys = [buffer_union]
        elif buffer_union.geom_type == "MultiPolygon":
            polys = list(buffer_union.geoms)

        for i, poly in enumerate(polys):
            if poly.is_empty or poly.area < 1:
                continue

            x, y = list(poly.exterior.xy[0]), list(poly.exterior.xy[1])
            if not x:
                continue

            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    fill="toself",
                    fillcolor=buffer_color,
                    mode="lines",
                    line=dict(width=1, color=line_color),
                    name="Proposed Coverage" if i == 0 else None,
                    legendgroup="proposed_coverage",
                    showlegend=(i == 0),
                    hoverinfo="skip",
                )
            )
            traces_added += 1

        if logger:
            logger.info(
                f"   Added {len(buffer_circles)} proposed borehole coverage buffers"
            )

    # Add marker trace
    proposed_x = [bh["x"] for bh in proposed_boreholes]
    proposed_y = [bh["y"] for bh in proposed_boreholes]
    customdata = [[i + 1] for i in range(len(proposed_boreholes))]

    fig.add_trace(
        go.Scattergl(
            x=proposed_x,
            y=proposed_y,
            mode="markers",
            marker=dict(
                size=marker_size,
                color=marker_color,
                symbol=marker_symbol,
                line=dict(width=2, color=marker_color),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>Proposed Borehole #%{customdata[0]}</b><br>"
                "Easting: %{x:,.0f}<br>"
                "Northing: %{y:,.0f}<br>"
                "<extra></extra>"
            ),
            name="Proposed Boreholes",
            legendgroup="proposed",
            showlegend=True,
        )
    )
    traces_added += 1

    if logger:
        logger.info(f"   Added {len(proposed_boreholes)} proposed borehole markers (×)")

    end_trace_idx = len(fig.data)
    return traces_added, start_trace_idx, end_trace_idx


# ===========================================================================
# MAP LAYOUT
# ===========================================================================


def build_map_layout(
    visualization_config: Any,
    title: str,
    show_legend: bool = False,
) -> Dict[str, Any]:
    """
    Build Plotly layout for map visualization.

    Creates a layout optimized for geospatial display:
    - Equal aspect ratio (scaleanchor)
    - Comma-formatted axis tick labels
    - White background with light grid
    - Pan mode as default dragmode
    - Configurable dimensions from visualization_config

    Args:
        visualization_config: VisualizationConfig object or dict with figure dimensions
        title: Title text to display at top of figure
        show_legend: Whether to show in-plot legend (default False for external panels)

    Returns:
        Dict of Plotly layout properties
    """
    # Handle both typed VisualizationConfig and legacy dict access
    if hasattr(visualization_config, "figure_width"):
        width = visualization_config.figure_width
        height = visualization_config.figure_height
        bg_color = getattr(visualization_config, "figure_background", "white")
    else:
        width = visualization_config.get("figure_width", 1200)
        height = visualization_config.get("figure_height", 1000)
        bg_color = visualization_config.get("figure_background", "white")

    return dict(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Easting (m)",
            tickformat=",",
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)",
        ),
        yaxis=dict(
            title="Northing (m)",
            tickformat=",",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)",
        ),
        hovermode="closest",
        dragmode="pan",  # Default to pan mode instead of zoom
        showlegend=show_legend,
        width=width,
        height=height,
        plot_bgcolor=bg_color,
        margin=dict(t=80, l=60, r=60),  # Reduced margins (no internal legend needed)
    )


# ===========================================================================
# BACKWARD COMPATIBILITY ALIASES
# ===========================================================================

# Aliases matching internal naming in html_builder_EC7.py
_add_grid_cells_trace = add_grid_cells_trace

# Legacy coverage zone trace names (moved from coverage_zones.py)
# These aliases allow gradual migration to new names
build_merged_polygon_trace_coverage = build_coverage_polygon_trace
build_merged_marker_trace = build_coverage_marker_trace
build_merged_buffer_trace = build_coverage_buffer_trace
_extract_merged_coords = _extract_coverage_coords
_add_boreholes_trace = add_boreholes_trace
_add_hexagon_grid_overlay = add_hexagon_grid_overlay
_add_zone_boundary_traces = add_zone_boundary_traces
_build_layout_without_dropdown = build_map_layout  # Legacy name
