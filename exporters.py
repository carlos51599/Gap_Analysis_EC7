"""
EC7 Unified Export Module - PNG, CSV, and GeoJSON exports.

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Export coverage results and proposed boreholes in various formats.
Consolidates coverage_exporter.py and proposed_borehole_exporter.py.

Export Formats:
- PNG: Coverage visualization images for AI inspection
- CSV: Proposed borehole coordinates for import into other tools
- GeoJSON: Coverage polygons for GIS software

Key Entry Points:
- export_all_coverage_outputs(): Testing mode PNG + GeoJSON export
- export_proposed_boreholes_to_csv(): CSV export for proposed boreholes
- export_coverage_polygons_to_geojson(): GeoJSON export for all combinations

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd
from shapely.geometry import mapping, shape, Point, MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely import wkt

if TYPE_CHECKING:
    import matplotlib.axes
    import geopandas as gpd

logger = logging.getLogger("EC7.Exporters")


# ═══════════════════════════════════════════════════════════════════════════
# 🔧 HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def _generate_csv_filename(combo_key: str) -> str:
    """
    Generate CSV filename from combination key.

    Args:
        combo_key: Filter combination key (e.g., "d30_spt0_txt0_txe0")

    Returns:
        Filename string (e.g., "proposed_boreholes_d30_spt0_txt0_txe0.csv")
    """
    return f"proposed_boreholes_{combo_key}.csv"


def _format_combo_key_human_readable(combo_key: str) -> str:
    """
    Convert combo key to human-readable description.

    Args:
        combo_key: Filter combination key (e.g., "d30_spt1_txt0_txe1")

    Returns:
        Human-readable string (e.g., "Depth>=30m, SPT:Yes, TxT:No, TxE:Yes")
    """
    parts = combo_key.split("_")
    descriptions = []

    for part in parts:
        if part.startswith("d"):
            depth = part[1:]
            descriptions.append(f"Depth>={depth}m")
        elif part.startswith("spt"):
            val = "Yes" if part[-1] == "1" else "No"
            descriptions.append(f"SPT:{val}")
        elif part.startswith("txt"):
            val = "Yes" if part[-1] == "1" else "No"
            descriptions.append(f"TxT:{val}")
        elif part.startswith("txe"):
            val = "Yes" if part[-1] == "1" else "No"
            descriptions.append(f"TxE:{val}")

    return ", ".join(descriptions)


def _geometry_to_geojson_feature_collection(
    geometry: Any,
    properties: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert Shapely geometry to GeoJSON FeatureCollection.

    Args:
        geometry: Shapely geometry (Polygon, MultiPolygon, etc.)
        properties: Dictionary of properties to attach to feature

    Returns:
        GeoJSON FeatureCollection dict
    """
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": properties,
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 📤 CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════


def export_proposed_boreholes_to_csv(
    precomputed_coverages: Dict[str, Dict[str, Any]],
    output_dir: Path,
    log: logging.Logger = None,
) -> Dict[str, str]:
    """
    Export proposed borehole coordinates to CSV files.

    Creates one CSV per filter combination containing:
    - Location_ID: Generated ID (e.g., "PROP_001")
    - Easting: X coordinate (BNG)
    - Northing: Y coordinate (BNG)
    - Filter_Combination: Human-readable filter description

    Files are overwritten if they already exist.

    Args:
        precomputed_coverages: Dict mapping combo_key -> coverage result dict.
                              Each result contains "proposed" list of {"x", "y"} dicts.
        output_dir: Base output directory (will create "proposed_boreholes/" subfolder)
        log: Logger instance (optional)

    Returns:
        Dict mapping combo_key -> absolute path to created CSV file
        Only includes combinations that have proposed boreholes.
    """
    if log is None:
        log = logger

    # Create subfolder for proposed borehole CSVs
    csv_dir = output_dir / "proposed_boreholes"
    csv_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"📤 Exporting proposed boreholes to: {csv_dir}")

    exported_files: Dict[str, str] = {}
    total_boreholes = 0
    files_created = 0

    for combo_key, coverage_data in precomputed_coverages.items():
        # Skip if no proposed boreholes
        proposed = coverage_data.get("proposed", [])
        if not proposed:
            continue

        # Generate filename and path
        filename = _generate_csv_filename(combo_key)
        csv_path = csv_dir / filename

        # Create DataFrame with proposed borehole data
        filter_description = _format_combo_key_human_readable(combo_key)

        rows = []
        for i, bh in enumerate(proposed, start=1):
            rows.append(
                {
                    "Location_ID": f"PROP_{i:03d}",
                    "Easting": round(bh["x"], 2),
                    "Northing": round(bh["y"], 2),
                    "Filter_Combination": filter_description,
                }
            )

        df = pd.DataFrame(rows)

        # Export to CSV (overwrites existing)
        df.to_csv(csv_path, index=False)

        exported_files[combo_key] = str(csv_path)
        total_boreholes += len(proposed)
        files_created += 1

    log.info(
        f"   ✅ Created {files_created} CSV files with {total_boreholes} total proposed boreholes"
    )

    return exported_files


def export_per_pass_boreholes_to_csv(
    precomputed_coverages: Dict[str, Dict[str, Any]],
    output_dir: Path,
    is_testing_mode: bool,
    log: logging.Logger = None,
) -> Dict[str, str]:
    """
    Export per-pass borehole data to timestamped folder structure.

    Folder naming:
    - Testing mode: testing_d50spt0txt0txe0_MMDD_HHMM/
    - Production mode: production_MMDD_HHMM/ with subfolders for each combo

    Files per folder:
    - first_pass.csv: Boreholes after Zone Decomposition
    - second_pass.csv: Boreholes after CZRC per-cell optimization
    - third_pass.csv: Boreholes after Cell-Cell boundary consolidation
    - final_proposed.csv: Same data object used by HTML (result["proposed"])

    Args:
        precomputed_coverages: Dict mapping combo_key -> coverage result dict
        output_dir: Base output directory
        is_testing_mode: Whether running in testing mode (single combo)
        log: Logger instance (optional)

    Returns:
        Dict mapping combo_key -> folder path
    """
    from datetime import datetime

    if log is None:
        log = logger

    # Generate timestamp: MMDD_HHMM
    timestamp = datetime.now().strftime("%m%d_%H%M")

    exported_folders: Dict[str, str] = {}

    for combo_key, coverage_data in precomputed_coverages.items():
        # Skip if no proposed boreholes (nothing to export)
        proposed = coverage_data.get("proposed", [])
        if not proposed:
            continue

        # Create folder name based on mode
        if is_testing_mode:
            folder_name = f"testing_{combo_key}_{timestamp}"
            combo_folder = output_dir / folder_name
        else:
            # Production mode: production_MMDD_HHMM/combo_key/
            prod_folder = output_dir / f"production_{timestamp}"
            combo_folder = prod_folder / combo_key

        combo_folder.mkdir(parents=True, exist_ok=True)

        # Extract per-pass data
        first_pass_bhs = coverage_data.get("first_pass_boreholes", [])
        second_pass_bhs = coverage_data.get("second_pass_boreholes", [])
        # Third pass is final proposed (same as HTML)
        third_pass_bhs = proposed

        # Export first_pass.csv
        _write_borehole_csv(
            boreholes=first_pass_bhs,
            csv_path=combo_folder / "first_pass.csv",
            id_prefix="FP",
            log=log,
        )

        # Export second_pass.csv
        _write_borehole_csv(
            boreholes=second_pass_bhs,
            csv_path=combo_folder / "second_pass.csv",
            id_prefix="SP",
            log=log,
        )

        # Export third_pass.csv (Cell-Cell CZRC)
        _write_borehole_csv(
            boreholes=third_pass_bhs,
            csv_path=combo_folder / "third_pass.csv",
            id_prefix="TP",
            log=log,
        )

        # Export final_proposed.csv (same data as HTML uses)
        _write_borehole_csv(
            boreholes=proposed,
            csv_path=combo_folder / "final_proposed.csv",
            id_prefix="PROP",
            log=log,
        )

        exported_folders[combo_key] = str(combo_folder)

        log.info(
            f"   📁 {combo_folder.name}: "
            f"P1={len(first_pass_bhs)}, P2={len(second_pass_bhs)}, "
            f"P3={len(third_pass_bhs)}, Final={len(proposed)}"
        )

    return exported_folders


def _write_borehole_csv(
    boreholes: List[Dict[str, Any]],
    csv_path: Path,
    id_prefix: str,
    log: logging.Logger,
) -> None:
    """
    Write borehole list to CSV with standard format.

    Args:
        boreholes: List of {"x", "y", "coverage_radius"} dicts
        csv_path: Path to output CSV file
        id_prefix: Prefix for Location_ID (e.g., "FP", "SP", "TP", "PROP")
        log: Logger instance
    """
    if not boreholes:
        # Create empty CSV with headers only
        df = pd.DataFrame(
            columns=["Location_ID", "Easting", "Northing", "Coverage_Radius"]
        )
        df.to_csv(csv_path, index=False)
        return

    rows = []
    for i, bh in enumerate(boreholes, start=1):
        rows.append(
            {
                "Location_ID": f"{id_prefix}_{i:03d}",
                "Easting": round(bh.get("x", 0), 2),
                "Northing": round(bh.get("y", 0), 2),
                "Coverage_Radius": round(bh.get("coverage_radius", 100.0), 1),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


# ═══════════════════════════════════════════════════════════════════════════
# 📷 PNG EXPORT HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _plot_zone_boundaries(
    ax: "matplotlib.axes.Axes",
    zones_gdf: Optional["gpd.GeoDataFrame"],
) -> None:
    """
    Plot zone boundaries as dashed gray lines (primary zones).

    Args:
        ax: Matplotlib axes object
        zones_gdf: GeoDataFrame with zone boundaries
    """
    if zones_gdf is None or zones_gdf.empty:
        return
    for _, row in zones_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            ax.plot(x, y, color="gray", linewidth=1.5, linestyle="--", alpha=0.7)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="gray", linewidth=1.5, linestyle="--", alpha=0.7)


def _plot_secondary_shapefiles(
    ax: "matplotlib.axes.Axes",
    all_shapefiles: Optional[Dict[str, "gpd.GeoDataFrame"]],
) -> None:
    """
    Plot secondary (non-primary) shapefile boundaries.

    Secondary shapefiles are drawn with their configured colors from SHAPEFILE_CONFIG.

    Args:
        ax: Matplotlib axes object
        all_shapefiles: Dict mapping layer key to GeoDataFrame
    """
    if not all_shapefiles:
        return

    from Gap_Analysis_EC7.shapefile_config import (
        get_coverage_layer_key,
        get_layer_config,
    )

    coverage_key = get_coverage_layer_key()

    for layer_key, gdf in all_shapefiles.items():
        # Skip coverage layer (handled separately)
        if layer_key == coverage_key:
            continue

        if gdf is None or gdf.empty:
            continue

        layer_config = get_layer_config(layer_key)
        rendering = layer_config.get("rendering", {})
        color = rendering.get("boundary_color", "#666666")
        linewidth = rendering.get("boundary_linewidth", 1.0)

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "Polygon":
                x, y = geom.exterior.xy
                ax.plot(x, y, color=color, linewidth=linewidth, alpha=0.6)
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color=color, linewidth=linewidth, alpha=0.6)


def _plot_proposed_boreholes(
    ax: "matplotlib.axes.Axes",
    proposed_coords: List[Dict[str, float]],
) -> None:
    """
    Plot proposed borehole markers as blue triangles.

    Args:
        ax: Matplotlib axes object
        proposed_coords: List of {"x": float, "y": float} dicts
    """
    if not proposed_coords:
        return
    xs = [p["x"] for p in proposed_coords]
    ys = [p["y"] for p in proposed_coords]
    ax.scatter(
        xs,
        ys,
        c="blue",
        s=120,
        marker="^",
        edgecolors="darkblue",
        linewidths=1,
        zorder=10,
        label=f"Proposed BHs ({len(proposed_coords)})",
    )


def _create_coverage_legend(
    ax: "matplotlib.axes.Axes",
    proposed_count: int,
) -> None:
    """
    Create legend with coverage/gap/boundary/proposed entries.

    Args:
        ax: Matplotlib axes object
        proposed_count: Number of proposed boreholes (for label)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    legend_elements = [
        mpatches.Patch(
            facecolor="green", alpha=0.4, edgecolor="darkgreen", label="Covered"
        ),
        mpatches.Patch(facecolor="red", alpha=0.5, edgecolor="darkred", label="Gaps"),
        mpatches.Patch(
            facecolor="none",
            edgecolor="gray",
            linestyle="--",
            label="Zone Boundary",
        ),
    ]
    if proposed_count > 0:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="blue",
                markeredgecolor="darkblue",
                markersize=10,
                label=f"Proposed BHs ({proposed_count})",
            )
        )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)


def _add_stats_annotation(
    ax: "matplotlib.axes.Axes",
    covered_geom: Optional[BaseGeometry],
    gaps_geom: Optional[BaseGeometry],
    proposed_count: int,
) -> None:
    """
    Add coverage statistics text box to plot.

    Args:
        ax: Matplotlib axes object
        covered_geom: Shapely geometry of covered area
        gaps_geom: Shapely geometry of gap areas
        proposed_count: Number of proposed boreholes
    """
    import math

    from Gap_Analysis_EC7.config import CONFIG

    max_spacing = CONFIG.get("max_spacing_m", 100.0)
    candidate_mult = CONFIG.get("ilp_solver", {}).get("candidate_spacing_mult", 0.5)
    grid_spacing = max_spacing * candidate_mult
    circle_area = math.pi * max_spacing * max_spacing
    hex_efficiency = 0.907  # Hexagonal packing efficiency

    covered_ha = covered_geom.area / 10000 if covered_geom else 0
    gaps_ha = gaps_geom.area / 10000 if gaps_geom else 0
    total_ha = covered_ha + gaps_ha
    coverage_pct = (covered_ha / total_ha * 100) if total_ha > 0 else 0

    # Theoretical minimum boreholes (hex packing)
    if gaps_geom and not gaps_geom.is_empty:
        gap_area_m2 = gaps_geom.area
        theoretical_min = math.ceil(gap_area_m2 / (circle_area * hex_efficiency))
    else:
        theoretical_min = 0

    stats_text = (
        f"Covered: {covered_ha:.1f} ha ({coverage_pct:.1f}%)\n"
        f"Gaps: {gaps_ha:.1f} ha\n"
        f"Proposed: {proposed_count} boreholes\n"
        f"Theoretical Min: {theoretical_min} (hex grid, {int(grid_spacing)}m spacing)"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 📷 PNG EXPORT
# ═══════════════════════════════════════════════════════════════════════════


def export_coverage_png(
    covered_geom: Optional[BaseGeometry],
    gaps_geom: Optional[BaseGeometry],
    proposed_coords: List[Dict[str, float]],
    zones_gdf: Optional["gpd.GeoDataFrame"],
    output_path: Path,
    title: str = "EC7 Coverage Analysis",
    dpi: int = 300,
) -> bool:
    """
    Export coverage geometry as PNG image for AI visual inspection.

    Args:
        covered_geom: Shapely geometry of covered area
        gaps_geom: Shapely geometry of gap areas (can be MultiPolygon)
        proposed_coords: List of {"x": float, "y": float} for proposed boreholes
        zones_gdf: GeoDataFrame with zone boundaries
        output_path: Path to save PNG file
        title: Title for the plot
        dpi: Resolution (300 = high quality for AI inspection)

    Returns:
        True if export successful, False otherwise
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("❌ matplotlib not installed - cannot export PNG")
        return False

    try:
        fig, ax = plt.subplots(figsize=(18, 15))

        # Plot layers using helper functions
        _plot_zone_boundaries(ax, zones_gdf)

        if covered_geom is not None and not covered_geom.is_empty:
            _plot_geometry(
                ax,
                covered_geom,
                facecolor="green",
                alpha=0.4,
                edgecolor="darkgreen",
                linewidth=1,
            )

        if gaps_geom is not None and not gaps_geom.is_empty:
            _plot_geometry(
                ax,
                gaps_geom,
                facecolor="red",
                alpha=0.5,
                edgecolor="darkred",
                linewidth=1,
            )

        _plot_proposed_boreholes(ax, proposed_coords)

        # Formatting
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (m)", fontsize=12)
        ax.set_ylabel("Northing (m)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Annotations using helper functions
        _create_coverage_legend(ax, len(proposed_coords))
        _add_stats_annotation(ax, covered_geom, gaps_geom, len(proposed_coords))

        # Save
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"📷 PNG exported: {output_path}")
        return True

    except Exception as e:
        logger.error(f"❌ PNG export failed: {e}")
        return False


def _plot_geometry(
    ax: "matplotlib.axes.Axes",
    geom: BaseGeometry,
    facecolor: str,
    alpha: float,
    edgecolor: str,
    linewidth: float,
) -> None:
    """Helper to plot a single geometry or MultiPolygon with proper hole handling."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    def polygon_to_path(polygon):
        """Convert a Shapely Polygon (with holes) to a matplotlib Path."""
        # Get exterior ring coordinates (drop Z if present)
        exterior_coords = [(c[0], c[1]) for c in polygon.exterior.coords]

        # Start with exterior ring
        all_vertices = list(exterior_coords)
        codes = (
            [Path.MOVETO]
            + [Path.LINETO] * (len(exterior_coords) - 2)
            + [Path.CLOSEPOLY]
        )

        # Add interior rings (holes) - these create holes in the fill
        for interior in polygon.interiors:
            interior_coords = [(c[0], c[1]) for c in interior.coords]
            all_vertices.extend(interior_coords)
            codes.extend(
                [Path.MOVETO]
                + [Path.LINETO] * (len(interior_coords) - 2)
                + [Path.CLOSEPOLY]
            )

        return Path(all_vertices, codes)

    def plot_single_polygon(polygon):
        """Plot a single polygon with holes correctly rendered."""
        path = polygon_to_path(polygon)
        patch = PathPatch(
            path,
            facecolor=facecolor,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        ax.add_patch(patch)

    if geom.geom_type == "Polygon":
        plot_single_polygon(geom)
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            plot_single_polygon(poly)


def export_candidate_grid_png(
    gaps_geom: Optional[BaseGeometry],
    zones_gdf: Optional["gpd.GeoDataFrame"],
    output_path: Path,
    combo_key: str,
    max_spacing: float = 200.0,
    dpi: int = 300,
) -> bool:
    """
    Export candidate grid outline as separate PNG for AI inspection.

    Shows the buffered gap area where candidate boreholes are placed.

    Args:
        gaps_geom: Shapely geometry of gap areas
        zones_gdf: GeoDataFrame with zone boundaries
        output_path: Path to save PNG file
        combo_key: Filter combination key for title
        max_spacing: Borehole spacing (for buffer calculation)
        dpi: Resolution (300 = high quality)

    Returns:
        True if export successful, False otherwise
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.error("❌ matplotlib not installed - cannot export PNG")
        return False

    if gaps_geom is None or gaps_geom.is_empty:
        logger.warning("⚠️ No gaps geometry for candidate grid PNG")
        return False

    try:
        fig, ax = plt.subplots(figsize=(18, 15))

        # Plot zone boundaries (background)
        if zones_gdf is not None and not zones_gdf.empty:
            for _, row in zones_gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == "Polygon":
                    x, y = geom.exterior.xy
                    ax.plot(
                        x, y, color="gray", linewidth=1.5, linestyle="--", alpha=0.7
                    )
                elif geom.geom_type == "MultiPolygon":
                    for poly in geom.geoms:
                        x, y = poly.exterior.xy
                        ax.plot(
                            x, y, color="gray", linewidth=1.5, linestyle="--", alpha=0.7
                        )

        # Plot original gaps (red, semi-transparent)
        _plot_geometry(
            ax, gaps_geom, facecolor="red", alpha=0.3, edgecolor="darkred", linewidth=1
        )

        # Plot buffered candidate area (blue outline - exterior AND interior rings)
        buffer_dist = max_spacing  # Matches ILP candidate grid buffer
        candidate_area = gaps_geom.buffer(buffer_dist)

        def plot_polygon_boundary(poly, is_first=True):
            """Plot both exterior and interior rings of a polygon."""
            # Exterior ring
            x, y = poly.exterior.xy
            ax.plot(
                x,
                y,
                color="blue",
                linewidth=2,
                linestyle="-",
                label="Candidate Grid Boundary" if is_first else None,
            )
            # Interior rings (holes) - important for donut shapes!
            for interior in poly.interiors:
                x, y = interior.xy
                ax.plot(x, y, color="blue", linewidth=2, linestyle="-")

        if candidate_area.geom_type == "Polygon":
            plot_polygon_boundary(candidate_area, is_first=True)
        elif candidate_area.geom_type == "MultiPolygon":
            for i, poly in enumerate(candidate_area.geoms):
                plot_polygon_boundary(poly, is_first=(i == 0))

        # Formatting
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (m)", fontsize=12)
        ax.set_ylabel("Northing (m)", fontsize=12)
        ax.set_title(
            f"Candidate Grid Boundary: {combo_key}\nBuffer = {buffer_dist}m from gaps",
            fontsize=14,
            fontweight="bold",
        )

        # Legend
        legend_elements = [
            mpatches.Patch(
                facecolor="red", alpha=0.3, edgecolor="darkred", label="Gap Areas"
            ),
            mpatches.Patch(
                facecolor="none",
                edgecolor="blue",
                linewidth=2,
                label="Candidate Grid Boundary",
            ),
            mpatches.Patch(
                facecolor="none",
                edgecolor="gray",
                linestyle="--",
                label="Zone Boundary",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Stats annotation
        gap_area_ha = gaps_geom.area / 10000
        candidate_area_ha = candidate_area.area / 10000
        stats_text = (
            f"Gap Area: {gap_area_ha:.1f} ha\n"
            f"Candidate Area: {candidate_area_ha:.1f} ha\n"
            f"Buffer: {buffer_dist}m"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"📷 Candidate grid PNG exported: {output_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Candidate grid PNG export failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 📄 GEOJSON EXPORT
# ═══════════════════════════════════════════════════════════════════════════


def export_coverage_geojson(
    covered_geom: Optional[BaseGeometry],
    gaps_geom: Optional[BaseGeometry],
    proposed_coords: List[Dict[str, float]],
    output_dir: Path,
    combo_key: str,
) -> Dict[str, Path]:
    """
    Export coverage geometries as GeoJSON files (testing mode).

    Args:
        covered_geom: Shapely geometry of covered area
        gaps_geom: Shapely geometry of gap areas
        proposed_coords: List of {"x": float, "y": float}
        output_dir: Directory to save GeoJSON files (already combo-specific)
        combo_key: Filter combination key (e.g., "d25_spt0_txt0_txe0")

    Returns:
        Dict mapping layer name to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = {}

    # Export covered area
    if covered_geom is not None and not covered_geom.is_empty:
        path = output_dir / "covered.geojson"
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"layer": "covered", "combo_key": combo_key},
                    "geometry": mapping(covered_geom),
                }
            ],
        }
        with open(path, "w") as f:
            json.dump(geojson, f, indent=2)
        exported["covered"] = path
        logger.info(f"📄 GeoJSON exported: {path.name}")

    # Export gaps
    if gaps_geom is not None and not gaps_geom.is_empty:
        path = output_dir / "gaps.geojson"
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"layer": "gaps", "combo_key": combo_key},
                    "geometry": mapping(gaps_geom),
                }
            ],
        }
        with open(path, "w") as f:
            json.dump(geojson, f, indent=2)
        exported["gaps"] = path
        logger.info(f"📄 GeoJSON exported: {path.name}")

    # Export proposed boreholes as points
    if proposed_coords:
        path = output_dir / "proposed.geojson"
        features = [
            {
                "type": "Feature",
                "properties": {"id": i + 1, "x": p["x"], "y": p["y"]},
                "geometry": {"type": "Point", "coordinates": [p["x"], p["y"]]},
            }
            for i, p in enumerate(proposed_coords)
        ]
        geojson = {"type": "FeatureCollection", "features": features}
        with open(path, "w") as f:
            json.dump(geojson, f, indent=2)
        exported["proposed"] = path
        logger.info(f"📄 GeoJSON exported: {path.name}")

    return exported


# ═══════════════════════════════════════════════════════════════════════════
# 📦 GEOJSON EXPORT HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _export_geojson_feature(
    geometry: BaseGeometry,
    output_path: Path,
    combo_key: str,
    feature_type: str,
    description: str,
    log: logging.Logger,
) -> bool:
    """
    Export single geometry as GeoJSON feature collection.

    Args:
        geometry: Shapely geometry to export
        output_path: Path to output file
        combo_key: Filter combination key for properties
        feature_type: Type string ("covered" or "uncovered")
        description: Human-readable description
        log: Logger instance

    Returns:
        True if export successful, False otherwise
    """
    try:
        geojson_data = _geometry_to_geojson_feature_collection(
            geometry=geometry,
            properties={
                "filter_combination": combo_key,
                "type": feature_type,
                "description": description,
                "area_ha": round(geometry.area / 10000, 2),
            },
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)
        return True
    except Exception as e:
        log.warning("   ⚠️ Failed to export %s for %s: %s", feature_type, combo_key, e)
        return False


def _export_hexgrid_geojson(
    gaps_wkt: str,
    output_path: Path,
    combo_key: str,
    log: logging.Logger,
) -> bool:
    """
    Export candidate hexagon grid clipped to gap buffer.

    Args:
        gaps_wkt: WKT string of gap geometry
        output_path: Path to output file
        combo_key: Filter combination key
        log: Logger instance

    Returns:
        True if export successful, False otherwise
    """
    from Gap_Analysis_EC7.config import CONFIG
    from Gap_Analysis_EC7.solvers.optimization_geometry import (
        generate_hexagon_grid_polygons,
    )

    try:
        gaps_geom = wkt.loads(gaps_wkt)
        max_spacing = CONFIG.get("max_spacing_m", 100.0)
        candidate_mult = CONFIG.get("ilp_solver", {}).get("candidate_spacing_mult", 0.5)
        grid_spacing = max_spacing * candidate_mult
        gap_buffer = gaps_geom.buffer(max_spacing)

        hexagons = generate_hexagon_grid_polygons(
            bounds=gap_buffer.bounds,
            grid_spacing=grid_spacing,
            clip_geometry=gap_buffer,
            logger=None,
        )

        features = [
            {
                "type": "Feature",
                "geometry": mapping(hex_poly),
                "properties": {"id": i + 1, "type": "candidate_hexagon"},
            }
            for i, hex_poly in enumerate(hexagons)
        ]

        hexgrid_geojson = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "filter_combination": combo_key,
                "grid_spacing_m": grid_spacing,
                "hexagon_count": len(hexagons),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(hexgrid_geojson, f, indent=2)
        log.info("   🔷 %s: %d hexagons exported", combo_key, len(hexagons))
        return True
    except Exception as e:
        log.warning("   ⚠️ Failed to export hexgrid for %s: %s", combo_key, e)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 📦 GEOJSON EXPORT
# ═══════════════════════════════════════════════════════════════════════════


def export_coverage_polygons_to_geojson(
    precomputed_coverages: Dict[str, Dict[str, Any]],
    output_dir: Path,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Export coverage polygons (covered/uncovered) to GeoJSON files.

    Creates two GeoJSON files per filter combination:
    - covered_<combo_key>.geojson: Green areas (borehole coverage)
    - uncovered_<combo_key>.geojson: Red areas (gaps needing boreholes)

    Args:
        precomputed_coverages: Dict mapping combo_key -> coverage result dict.
        output_dir: Base output directory (will create "coverage_polygons/" subfolder)
        log: Logger instance (optional)

    Returns:
        Dict mapping combo_key -> dict with "covered", "uncovered", "hexgrid" paths
    """
    if log is None:
        log = logger

    geojson_dir = output_dir / "coverage_polygons"
    geojson_dir.mkdir(parents=True, exist_ok=True)
    log.info("📤 Exporting coverage polygons to: %s", geojson_dir)

    from Gap_Analysis_EC7.parallel.coverage_orchestrator import deserialize_geometry

    exported_files: Dict[str, Dict[str, str]] = {}
    files_created = 0

    for combo_key, coverage_data in precomputed_coverages.items():
        combo_dir = geojson_dir / combo_key
        combo_dir.mkdir(parents=True, exist_ok=True)
        exported_files[combo_key] = {}
        description = _format_combo_key_human_readable(combo_key)

        # Export covered (green) polygons
        covered_wkt = coverage_data.get("covered")
        if covered_wkt:
            covered_geom = deserialize_geometry(covered_wkt)
            if covered_geom:
                covered_path = combo_dir / "covered.geojson"
                if _export_geojson_feature(
                    covered_geom, covered_path, combo_key, "covered", description, log
                ):
                    exported_files[combo_key]["covered"] = str(covered_path)
                    files_created += 1

        # Export uncovered (red) polygons
        gaps_wkt = coverage_data.get("gaps")
        if gaps_wkt:
            gaps_geom = wkt.loads(gaps_wkt)
            gaps_path = combo_dir / "uncovered.geojson"
            if _export_geojson_feature(
                gaps_geom, gaps_path, combo_key, "uncovered", description, log
            ):
                exported_files[combo_key]["uncovered"] = str(gaps_path)
                files_created += 1

            # Export hexgrid
            hexgrid_path = combo_dir / "candidate_hexgrid.geojson"
            if _export_hexgrid_geojson(gaps_wkt, hexgrid_path, combo_key, log):
                exported_files[combo_key]["hexgrid"] = str(hexgrid_path)
                files_created += 1

    log.info("   ✅ Created %d GeoJSON files for coverage polygons", files_created)
    return exported_files


# ═══════════════════════════════════════════════════════════════════════════
# 🚀 MAIN EXPORT ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


def export_all_coverage_outputs(
    precomputed_coverages: Dict[str, Dict[str, Any]],
    zones_gdf: Optional["gpd.GeoDataFrame"],
    output_dir: Path,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Export PNG and GeoJSON files for AI inspection (testing mode only).

    Creates folder: output_dir/coverage_polygons/{combo_key}/
    with:
    - coverage_inspection.png (coverage + gaps + proposed)
    - candidate_grid.png (buffered gap area boundary)
    - covered.geojson, gaps.geojson, proposed.geojson

    Args:
        precomputed_coverages: Dict of combo_key -> coverage data
        zones_gdf: GeoDataFrame with zone boundaries
        output_dir: Base output directory (Gap_Analysis_EC7/Output)
        log: Logger instance (optional)

    Returns:
        Dict with export results: {"png": [...], "geojson": {...}}
    """
    if log is None:
        log = logger

    from Gap_Analysis_EC7.parallel.coverage_orchestrator import deserialize_geometry
    from Gap_Analysis_EC7.config import CONFIG

    if not precomputed_coverages:
        log.warning("⚠️ No precomputed coverages to export")
        return {}

    # Get the first (typically only) combo in testing mode
    combo_key = list(precomputed_coverages.keys())[0]
    data = precomputed_coverages[combo_key]

    # Create combo-specific output folder
    combo_dir = output_dir / "coverage_polygons" / combo_key
    combo_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"📦 Exporting to: {combo_dir}")

    # Deserialize geometries
    covered_geom = deserialize_geometry(data.get("covered"))
    gaps_geom = deserialize_geometry(data.get("gaps"))
    proposed_coords = data.get("proposed", [])

    results = {"png": [], "geojson": {}}

    # === PNG 1: Coverage Inspection ===
    png_path = combo_dir / "coverage_inspection.png"
    stats = data.get("stats", {})
    title = (
        f"EC7 Coverage Analysis: {combo_key}\n"
        f"Coverage: {stats.get('coverage_pct', 0):.1f}% | "
        f"Proposed: {len(proposed_coords)} BHs"
    )
    if export_coverage_png(
        covered_geom=covered_geom,
        gaps_geom=gaps_geom,
        proposed_coords=proposed_coords,
        zones_gdf=zones_gdf,
        output_path=png_path,
        title=title,
    ):
        results["png"].append(png_path)

    # === PNG 2: Candidate Grid Outline ===
    max_spacing = CONFIG.get("max_borehole_spacing_m", 200.0)
    grid_png_path = combo_dir / "candidate_grid.png"
    if export_candidate_grid_png(
        gaps_geom=gaps_geom,
        zones_gdf=zones_gdf,
        output_path=grid_png_path,
        combo_key=combo_key,
        max_spacing=max_spacing,
    ):
        results["png"].append(grid_png_path)

    # === GeoJSON Files ===
    geojson_paths = export_coverage_geojson(
        covered_geom=covered_geom,
        gaps_geom=gaps_geom,
        proposed_coords=proposed_coords,
        output_dir=combo_dir,
        combo_key=combo_key,
    )
    results["geojson"] = geojson_paths

    log.info(
        f"✅ Export complete: {len(results['png'])} PNGs, {len(results['geojson'])} GeoJSONs"
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 📦 MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # CSV export
    "export_proposed_boreholes_to_csv",
    # PNG export
    "export_coverage_png",
    "export_candidate_grid_png",
    # GeoJSON export
    "export_coverage_geojson",
    "export_coverage_polygons_to_geojson",
    # Main entry point
    "export_all_coverage_outputs",
]
