"""
BGS Geology Layer Module

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Load BGS bedrock geology shapefile, clip to extent, and generate
Plotly traces for interactive HTML visualization.

Key Algorithm:
1. Load BGS bedrock shapefile clipped to specified extent (bbox filtering)
2. Generate color mapping for unique geology formations
3. Create Plotly Scatter traces with polygon fills for each formation
4. Return traces for integration with HTML builder

Navigation Guide:
- Section markers: # ═════ for major sections
- Use VS Code outline (Ctrl+Shift+O) to jump between functions
"""

# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ IMPORTS SECTION
# ═══════════════════════════════════════════════════════════════════════════
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import box

import plotly.graph_objects as go
import plotly.express as px


# ═══════════════════════════════════════════════════════════════════════════
# 📂 DATA LOADING SECTION
# ═══════════════════════════════════════════════════════════════════════════


def load_bgs_bedrock_clipped(
    shp_path: str,
    extent: Tuple[float, float, float, float],
    target_crs: str = "EPSG:27700",
    buffer_m: float = 500.0,
    logger: Optional[logging.Logger] = None,
) -> GeoDataFrame:
    """
    Load BGS bedrock shapefile clipped to specified extent.

    Args:
        shp_path: Path to BGS bedrock shapefile
        extent: Bounding box (min_x, max_x, min_y, max_y) - NOTE: different order from shapely
        target_crs: Target coordinate reference system
        buffer_m: Buffer distance to add around extent (meters)
        logger: Optional logger instance

    Returns:
        GeoDataFrame with bedrock polygons clipped to extent
    """
    if logger:
        logger.info(f"📂 Loading BGS bedrock shapefile: {shp_path}")

    # Unpack extent (min_x, max_x, min_y, max_y) and add buffer
    min_x, max_x, min_y, max_y = extent
    min_x -= buffer_m
    max_x += buffer_m
    min_y -= buffer_m
    max_y += buffer_m

    if logger:
        logger.info(
            f"   Extent with {buffer_m}m buffer: E[{min_x:.0f},{max_x:.0f}] N[{min_y:.0f},{max_y:.0f}]"
        )

    # Read only features within bounding box for efficiency
    bbox = (min_x, min_y, max_x, max_y)

    try:
        gdf = gpd.read_file(shp_path, bbox=bbox)
    except FileNotFoundError:
        if logger:
            logger.error(f"❌ BGS shapefile not found: {shp_path}")
        raise
    except Exception as e:
        if logger:
            logger.error(f"❌ Failed to load BGS shapefile: {e}")
        raise

    if logger:
        logger.info(f"✅ Loaded {len(gdf)} bedrock polygons within extent")

    # Ensure correct CRS
    if gdf.crs is None:
        gdf.set_crs(target_crs, inplace=True)
    elif gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)

    # Clip to exact extent
    clip_box = box(min_x, min_y, max_x, max_y)
    gdf["geometry"] = gdf.geometry.intersection(clip_box)
    gdf = gdf[~gdf.geometry.is_empty]

    if logger:
        logger.info(f"   After clipping: {len(gdf)} polygons")
        if "LITHOSTRAT" in gdf.columns:
            unique_formations = gdf["LITHOSTRAT"].nunique()
            logger.info(f"   Unique formations (LITHOSTRAT): {unique_formations}")

    return gdf


# ═══════════════════════════════════════════════════════════════════════════
# 🎨 VISUALIZATION SECTION
# ═══════════════════════════════════════════════════════════════════════════


def _get_formation_colors(
    bedrock_gdf: GeoDataFrame,
    color_column: str = "LITHOSTRAT",
) -> Dict[str, str]:
    """
    Generate a color mapping for unique geology formations.

    Args:
        bedrock_gdf: GeoDataFrame with bedrock polygons
        color_column: Column name to use for unique values

    Returns:
        Dict mapping formation names to hex color strings
    """
    if color_column not in bedrock_gdf.columns:
        return {}

    unique_values = bedrock_gdf[color_column].unique()

    # Use a qualitative color palette (pastel colors work well as background)
    colors = (
        px.colors.qualitative.Set3
        + px.colors.qualitative.Pastel
        + px.colors.qualitative.Set2
    )

    color_map = {}
    for i, val in enumerate(unique_values):
        color_map[val] = colors[i % len(colors)]

    return color_map


def create_bgs_geology_traces(
    bedrock_gdf: GeoDataFrame,
    opacity: float = 0.4,
    line_color: str = "rgba(50, 50, 50, 0.3)",
    line_width: float = 0.5,
    color_column: str = "LITHOSTRAT",
    visible: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[go.Scatter]:
    """
    Create Plotly Scatter traces for BGS bedrock geology polygons.

    Args:
        bedrock_gdf: GeoDataFrame with bedrock polygons
        opacity: Fill opacity for polygons
        line_color: Color for polygon outlines
        line_width: Width of polygon outlines
        color_column: Column to use for coloring
        visible: Initial visibility state
        logger: Optional logger instance

    Returns:
        List of Plotly Scatter traces
    """
    if logger:
        logger.info(f"   Creating BGS geology traces ({len(bedrock_gdf)} polygons)...")

    traces = []
    color_map = _get_formation_colors(bedrock_gdf, color_column)
    added_to_legend = set()

    for _, row in bedrock_gdf.iterrows():
        geom = row.geometry
        if geom.is_empty:
            continue

        formation = (
            row.get(color_column, "Unknown")
            if color_column in bedrock_gdf.columns
            else "Unknown"
        )
        color = color_map.get(formation, "rgba(200, 200, 200, 0.5)")

        # Convert color to RGBA with specified opacity
        fill_color = _convert_color_to_rgba(color, opacity)

        # Handle MultiPolygon and Polygon
        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            continue

        for poly in polygons:
            if poly.is_empty or not hasattr(poly, "exterior"):
                continue

            coords = np.array(poly.exterior.coords)

            show_legend = formation not in added_to_legend
            if show_legend:
                added_to_legend.add(formation)

            # Truncate formation name for legend
            legend_name = (
                formation[:40] + "..." if len(str(formation)) > 40 else str(formation)
            )

            traces.append(
                go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode="lines",
                    fill="toself",
                    fillcolor=fill_color,
                    line=dict(
                        color=line_color,
                        width=line_width,
                    ),
                    name=legend_name,
                    legendgroup=f"bgs_{formation}",
                    showlegend=False,  # Don't show in main legend (too many items)
                    visible=visible,
                    hovertemplate=(
                        f"<b>BGS Bedrock</b><br>"
                        f"Formation: {formation}<br>"
                        f"Lithology: {row.get('LITHOLOGY', 'N/A')}<br>"
                        f"Age: {row.get('AGE', 'N/A')}<br>"
                        "<extra></extra>"
                    ),
                )
            )

    if logger:
        logger.info(f"   Created {len(traces)} BGS geology traces")

    return traces


def _convert_color_to_rgba(color: str, opacity: float) -> str:
    """
    Convert a color string to RGBA format with specified opacity.

    Args:
        color: Color string (hex, rgb, or rgba format)
        opacity: Opacity value (0-1)

    Returns:
        RGBA color string
    """
    if color.startswith("rgba"):
        # Already RGBA - replace opacity
        parts = color.replace("rgba(", "").replace(")", "").split(",")
        if len(parts) >= 3:
            return f"rgba({parts[0].strip()}, {parts[1].strip()}, {parts[2].strip()}, {opacity})"
        return color

    if color.startswith("#"):
        # Hex color
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r}, {g}, {b}, {opacity})"

    if color.startswith("rgb("):
        # RGB color
        return color.replace("rgb(", "rgba(").replace(")", f", {opacity})")

    # Unknown format - return as is
    return color


# ═══════════════════════════════════════════════════════════════════════════
# 📊 INTEGRATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def add_bgs_geology_to_figure(
    fig: go.Figure,
    bedrock_gdf: GeoDataFrame,
    bgs_config: Dict[str, Any],
    visible: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, int]:
    """
    Add BGS geology traces to a Plotly figure.

    Args:
        fig: Plotly Figure to add traces to
        bedrock_gdf: GeoDataFrame with bedrock polygons
        bgs_config: Configuration dict with opacity, line_color, etc.
        visible: Initial visibility state
        logger: Optional logger instance

    Returns:
        Tuple of (start_trace_index, end_trace_index) for the added traces
    """
    start_idx = len(fig.data)

    traces = create_bgs_geology_traces(
        bedrock_gdf=bedrock_gdf,
        opacity=bgs_config.get("opacity", 0.4),
        line_color=bgs_config.get("line_color", "rgba(50, 50, 50, 0.3)"),
        line_width=bgs_config.get("line_width", 0.5),
        color_column=bgs_config.get("color_column", "LITHOSTRAT"),
        visible=visible,
        logger=logger,
    )

    for trace in traces:
        fig.add_trace(trace)

    end_idx = len(fig.data)

    if logger:
        logger.info(
            f"   Added BGS geology traces: indices {start_idx} to {end_idx - 1}"
        )

    return start_idx, end_idx
