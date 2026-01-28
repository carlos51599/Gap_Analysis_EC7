#!/usr/bin/env python3
"""
EC7 Coverage Zone Computation Module

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Compute EC7 borehole coverage zones using buffer union approach.
Creates coverage polygons and identifies gaps based on max spacing requirement.

This is a PURE COMPUTATION module - no Plotly or visualization dependencies.
All trace building functions have been moved to visualization/plotly_traces.py

Key Features:
1. Buffer union around boreholes (200m radius by default)
2. Gap identification by differencing zone boundaries from coverage
3. Gap statistics: area, suggested borehole locations, max distance to nearest BH

Navigation Guide:
- compute_coverage_zones: Core computation logic
- split_gaps_by_zones: Split gap geometry by zone boundaries
- get_coverage_summary: Summary statistics
- _compute_gap_statistics: Detailed gap analysis
- _split_to_polygons: Geometry normalization helper

Visualization Functions (moved to visualization/plotly_traces.py):
- add_coverage_zone_traces
- add_proposed_borehole_traces
- build_coverage_polygon_trace
- build_coverage_marker_trace
- build_coverage_buffer_trace

CONFIGURATION ARCHITECTURE:
- No CONFIG access - all functions accept explicit parameters
- max_spacing controls buffer radius
- Pure data-in, data-out design
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, Polygon


# ===========================================================================
# COVERAGE ZONE COMPUTATION
# ===========================================================================


def compute_coverage_zones(
    boreholes_gdf: GeoDataFrame,
    zones_gdf: GeoDataFrame,
    max_spacing: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[BaseGeometry], Optional[BaseGeometry], List[Dict[str, Any]]]:
    """
    Compute coverage union and uncovered gaps based on EC7 spacing requirement.

    Supports per-zone spacing if zones_gdf has 'max_spacing_m' column.
    Otherwise uses global max_spacing parameter.

    Args:
        boreholes_gdf: GeoDataFrame with borehole points
        zones_gdf: GeoDataFrame with zone boundary polygons
        max_spacing: EC7 max spacing in meters (fallback if no per-zone spacing)
        logger: Optional logger instance

    Returns:
        Tuple of (covered_union, uncovered_gaps, gap_stats)
        - covered_union: Shapely geometry of all covered area
        - uncovered_gaps: Shapely geometry of uncovered areas
        - gap_stats: List of dicts with gap info (area, centroid, max_distance)
    """
    # Check if per-zone spacing is available
    if zones_gdf is not None and "max_spacing_m" in zones_gdf.columns:
        return _compute_coverage_per_zone_spacing(boreholes_gdf, zones_gdf, logger)

    # Fallback to global spacing
    if max_spacing is None:
        max_spacing = 100.0  # Default fallback

    return _compute_coverage_global_spacing(
        boreholes_gdf, zones_gdf, max_spacing, logger
    )


def _compute_coverage_global_spacing(
    boreholes_gdf: GeoDataFrame,
    zones_gdf: GeoDataFrame,
    max_spacing: float,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Any, Any, List[Dict[str, Any]]]:
    """
    Compute coverage with single global max_spacing value.

    Original implementation - all zones use the same spacing.
    """
    if logger:
        logger.info(f"   Computing EC7 coverage zones (radius={max_spacing}m)...")

    # Create buffers around all boreholes
    borehole_buffers = boreholes_gdf.geometry.buffer(max_spacing)

    # Union all buffers to get covered area
    covered_union = unary_union(borehole_buffers)

    # Get zone boundary union
    zone_boundary = unary_union(zones_gdf.geometry)

    # Clip covered area to zone boundary
    covered_union = covered_union.intersection(zone_boundary)

    # Calculate uncovered gaps
    uncovered_gaps = zone_boundary.difference(covered_union)

    # Calculate gap statistics
    gap_stats = _compute_gap_statistics(uncovered_gaps, boreholes_gdf, logger)

    if logger:
        covered_area = covered_union.area / 10000 if not covered_union.is_empty else 0
        uncovered_area = (
            uncovered_gaps.area / 10000 if not uncovered_gaps.is_empty else 0
        )
        logger.info(
            f"   ✅ Covered: {covered_area:.1f} ha, "
            f"Uncovered: {uncovered_area:.1f} ha ({len(gap_stats)} gaps)"
        )

    return covered_union, uncovered_gaps, gap_stats


def _compute_coverage_per_zone_spacing(
    boreholes_gdf: GeoDataFrame,
    zones_gdf: GeoDataFrame,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[BaseGeometry], Optional[BaseGeometry], List[Dict[str, Any]]]:
    """
    Compute coverage with per-zone max_spacing_m values.

    Each zone uses its own spacing requirement from the max_spacing_m column.
    """
    if logger:
        logger.info("   Computing EC7 coverage zones (per-zone spacing)...")

    all_covered = []
    all_gaps = []
    gap_stats = []

    # Use zone_name if available, otherwise fallback to Name or index
    zone_name_col = None
    for col in ["zone_name", "Name"]:
        if col in zones_gdf.columns:
            zone_name_col = col
            break

    for _, zone_row in zones_gdf.iterrows():
        zone_name = (
            zone_row.get(zone_name_col, str(zone_row.name))
            if zone_name_col
            else str(zone_row.name)
        )
        zone_spacing = zone_row["max_spacing_m"]
        zone_geom = zone_row.geometry

        if zone_geom is None or zone_geom.is_empty:
            continue

        # Ensure valid geometry
        if not zone_geom.is_valid:
            zone_geom = zone_geom.buffer(0)

        # Get boreholes that could provide coverage to this zone
        # Include boreholes within zone_spacing distance of zone boundary
        # This allows cross-boundary coverage where a borehole in an adjacent zone
        # can still contribute coverage (using this zone's spacing requirement)
        expanded_zone = zone_geom.buffer(zone_spacing)
        zone_boreholes = boreholes_gdf[boreholes_gdf.within(expanded_zone)]

        zone_area_ha = zone_geom.area / 10000

        if zone_boreholes.empty:
            # Entire zone is uncovered
            all_gaps.append(zone_geom)
            if logger:
                logger.info(
                    f"   ⚠️ {zone_name}: No boreholes within {zone_spacing}m "
                    f"(0% coverage)"
                )
            continue

        # Compute coverage for this zone with its specific spacing
        # All boreholes (including cross-boundary) are buffered by zone's spacing
        buffer_radius = zone_spacing
        covered = zone_boreholes.geometry.buffer(buffer_radius).unary_union
        covered_in_zone = covered.intersection(zone_geom)
        uncovered_in_zone = zone_geom.difference(covered_in_zone)

        all_covered.append(covered_in_zone)

        covered_area_ha = covered_in_zone.area / 10000
        gap_area_ha = (
            uncovered_in_zone.area / 10000 if not uncovered_in_zone.is_empty else 0.0
        )
        coverage_pct = (
            (covered_area_ha / zone_area_ha * 100) if zone_area_ha > 0 else 0.0
        )

        if not uncovered_in_zone.is_empty:
            all_gaps.append(uncovered_in_zone)

        if logger:
            logger.info(
                f"   📊 {zone_name}: {coverage_pct:.1f}% coverage "
                f"({len(zone_boreholes)} BHs, {zone_spacing}m spacing)"
            )

    # Merge results
    covered_union = unary_union(all_covered) if all_covered else None
    uncovered_gaps = unary_union(all_gaps) if all_gaps else None

    # Calculate gap statistics from merged gaps
    if uncovered_gaps and not uncovered_gaps.is_empty:
        gap_stats = _compute_gap_statistics(uncovered_gaps, boreholes_gdf, logger)

    if logger:
        covered_area = (
            covered_union.area / 10000
            if covered_union and not covered_union.is_empty
            else 0
        )
        uncovered_area = (
            uncovered_gaps.area / 10000
            if uncovered_gaps and not uncovered_gaps.is_empty
            else 0
        )
        logger.info(
            f"   ✅ Total Covered: {covered_area:.1f} ha, "
            f"Uncovered: {uncovered_area:.1f} ha ({len(gap_stats)} gaps)"
        )

    return covered_union, uncovered_gaps, gap_stats


def _compute_gap_statistics(
    uncovered_gaps: Union[Polygon, MultiPolygon, BaseGeometry],
    boreholes_gdf: GeoDataFrame,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Compute statistics for each uncovered gap.

    For each gap polygon, calculates:
    - Area in m² and hectares
    - Centroid coordinates
    - Suggested borehole location (representative point)
    - Maximum distance to nearest existing borehole

    Args:
        uncovered_gaps: Shapely geometry of uncovered areas
        boreholes_gdf: GeoDataFrame with borehole points
        logger: Optional logger

    Returns:
        List of gap statistics dicts
    """
    gap_stats: List[Dict[str, Any]] = []

    if uncovered_gaps.is_empty:
        return gap_stats

    # Handle both Polygon and MultiPolygon
    if isinstance(uncovered_gaps, Polygon):
        gap_polys = [uncovered_gaps]
    elif isinstance(uncovered_gaps, MultiPolygon):
        gap_polys = list(uncovered_gaps.geoms)
    else:
        gap_polys = []

    for i, gap in enumerate(gap_polys):
        if gap.is_empty or gap.area < 100:  # Skip tiny gaps (<100m²)
            continue

        try:
            # Get representative point (guaranteed inside polygon)
            rep_point = gap.representative_point()

            # Calculate centroid (may be outside for concave shapes)
            centroid = gap.centroid
            if not gap.contains(centroid):
                centroid = rep_point

            # Find distance to nearest borehole from suggested location
            min_dist = float("inf")
            for bh_geom in boreholes_gdf.geometry:
                dist = rep_point.distance(bh_geom)
                if dist < min_dist:
                    min_dist = dist

            gap_stats.append(
                {
                    "id": len(gap_stats) + 1,
                    "area_m2": gap.area,
                    "area_ha": gap.area / 10000,
                    "centroid_x": centroid.x,
                    "centroid_y": centroid.y,
                    "suggested_x": rep_point.x,
                    "suggested_y": rep_point.y,
                    "max_distance_m": min_dist,
                    "geometry": gap,
                }
            )
        except Exception as e:
            if logger:
                logger.warning(f"   ⚠️ Error processing gap {i}: {e}")

    # Sort by area (largest first)
    gap_stats.sort(key=lambda x: x["area_m2"], reverse=True)

    # Re-number after sorting
    for i, gap in enumerate(gap_stats):
        gap["id"] = i + 1

    return gap_stats


# ===========================================================================
# ZONE-BASED GAP SPLITTING
# ===========================================================================


def split_gaps_by_zones(
    uncovered_gaps: Optional[BaseGeometry],
    zones_gdf: GeoDataFrame,
    min_area_m2: float = 100.0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Polygon]]:
    """
    Split uncovered gaps by zone boundaries.

    Takes the unified gap geometry and intersects it with each zone boundary
    to create per-zone gap lists. This enables zone-parallel optimization.

    MATHEMATICAL NOTE:
    This introduces bounded suboptimality at zone boundaries (5-15% more
    boreholes) but provides dramatic complexity reduction for large problems.

    Args:
        uncovered_gaps: Shapely geometry (Polygon/MultiPolygon) of all gaps
        zones_gdf: GeoDataFrame with zone polygons (must have 'Name' column)
        min_area_m2: Minimum gap area to include (filters boundary slivers)
        logger: Optional logger instance

    Returns:
        Dict mapping zone_name -> list of gap Polygons within that zone.
        Empty zones are omitted from the dict.

    Example:
        >>> zone_gaps = split_gaps_by_zones(gaps, zones_gdf)
        >>> for zone_name, polys in zone_gaps.items():
        ...     print(f"{zone_name}: {len(polys)} gaps")
    """
    zone_gaps: Dict[str, List[Polygon]] = {}

    if uncovered_gaps is None or uncovered_gaps.is_empty:
        if logger:
            logger.info("   ℹ️ No gaps to split by zones")
        return zone_gaps

    # Ensure gap geometry is valid (defensive)
    if not uncovered_gaps.is_valid:
        uncovered_gaps = uncovered_gaps.buffer(0)

    if zones_gdf is None or zones_gdf.empty:
        if logger:
            logger.warning("   ⚠️ No zones provided for gap splitting")
        return zone_gaps

    if logger:
        logger.info(f"   🔀 Splitting gaps by {len(zones_gdf)} zones...")

    total_input_area = uncovered_gaps.area / 10000  # ha
    total_output_area = 0.0

    for _, zone_row in zones_gdf.iterrows():
        # Get zone name (try 'Name' column, fall back to index)
        zone_name = zone_row.get("Name", str(zone_row.name))
        zone_geom = zone_row.geometry

        # Ensure zone geometry is valid
        if zone_geom is None or zone_geom.is_empty:
            continue
        if not zone_geom.is_valid:
            zone_geom = zone_geom.buffer(0)

        # Intersect gaps with zone boundary
        try:
            intersected = uncovered_gaps.intersection(zone_geom)
        except Exception as e:
            if logger:
                logger.warning(f"   ⚠️ Intersection failed for {zone_name}: {e}")
            continue

        if intersected.is_empty:
            continue

        # Normalize to list of Polygons
        polys = _normalize_intersection_to_polygons(intersected)

        # Filter by minimum area (removes boundary slivers)
        polys = [p for p in polys if p.area >= min_area_m2]

        if polys:
            zone_gaps[zone_name] = polys
            zone_area = sum(p.area for p in polys) / 10000
            total_output_area += zone_area
            if logger:
                logger.info(f"      {zone_name}: {len(polys)} gaps, {zone_area:.1f} ha")

    if logger:
        area_diff = total_input_area - total_output_area
        logger.info(
            f"   ✅ Split into {len(zone_gaps)} zones "
            f"(input: {total_input_area:.1f} ha, output: {total_output_area:.1f} ha, "
            f"slivers filtered: {area_diff:.1f} ha)"
        )

    return zone_gaps


def split_gaps_by_zones_with_spacing(
    uncovered_gaps: Optional[BaseGeometry],
    zones_gdf: GeoDataFrame,
    min_area_m2: float = 100.0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Split uncovered gaps by zone boundaries with per-zone spacing info.

    Extended version of split_gaps_by_zones that also returns the max_spacing_m
    for each zone, enabling the solver to use zone-specific spacing.

    Args:
        uncovered_gaps: Shapely geometry (Polygon/MultiPolygon) of all gaps
        zones_gdf: GeoDataFrame with zone polygons (must have 'zone_name' or 'Name'
            column, and 'max_spacing_m' column)
        min_area_m2: Minimum gap area to include (filters boundary slivers)
        logger: Optional logger instance

    Returns:
        Dict mapping zone_name -> {"gaps": List[Polygon], "max_spacing_m": float}
        Empty zones are omitted from the dict.

    Example:
        >>> zone_data = split_gaps_by_zones_with_spacing(gaps, zones_gdf)
        >>> for zone_name, data in zone_data.items():
        ...     print(f"{zone_name}: {len(data['gaps'])} gaps, {data['max_spacing_m']}m spacing")
    """
    from Gap_Analysis_EC7.config import CONFIG

    zone_data: Dict[str, Dict[str, Any]] = {}

    if uncovered_gaps is None or uncovered_gaps.is_empty:
        if logger:
            logger.info("   ℹ️ No gaps to split by zones")
        return zone_data

    # Ensure gap geometry is valid
    if not uncovered_gaps.is_valid:
        uncovered_gaps = uncovered_gaps.buffer(0)

    if zones_gdf is None or zones_gdf.empty:
        if logger:
            logger.warning("   ⚠️ No zones provided for gap splitting")
        return zone_data

    # Determine which column has zone names
    zone_name_col = None
    for col in ["zone_name", "Name"]:
        if col in zones_gdf.columns:
            zone_name_col = col
            break

    has_spacing = "max_spacing_m" in zones_gdf.columns
    default_spacing = CONFIG.get("max_spacing_m", 100.0)

    if logger:
        logger.info(f"   🔀 Splitting gaps by {len(zones_gdf)} zones (with spacing)...")

    total_input_area = uncovered_gaps.area / 10000
    total_output_area = 0.0

    for _, zone_row in zones_gdf.iterrows():
        # Get zone name
        zone_name = (
            zone_row.get(zone_name_col, str(zone_row.name))
            if zone_name_col
            else str(zone_row.name)
        )
        zone_geom = zone_row.geometry

        # Get zone spacing
        zone_spacing = (
            float(zone_row["max_spacing_m"]) if has_spacing else default_spacing
        )

        # Ensure zone geometry is valid
        if zone_geom is None or zone_geom.is_empty:
            continue
        if not zone_geom.is_valid:
            zone_geom = zone_geom.buffer(0)

        # Intersect gaps with zone boundary
        try:
            intersected = uncovered_gaps.intersection(zone_geom)
        except Exception as e:
            if logger:
                logger.warning(f"   ⚠️ Intersection failed for {zone_name}: {e}")
            continue

        if intersected.is_empty:
            continue

        # Normalize to list of Polygons
        polys = _normalize_intersection_to_polygons(intersected)

        # Filter by minimum area
        polys = [p for p in polys if p.area >= min_area_m2]

        if polys:
            zone_data[zone_name] = {
                "gaps": polys,
                "max_spacing_m": zone_spacing,
            }
            zone_area = sum(p.area for p in polys) / 10000
            total_output_area += zone_area
            if logger:
                logger.info(
                    f"      {zone_name}: {len(polys)} gaps, {zone_area:.1f} ha, "
                    f"{zone_spacing}m spacing"
                )

    if logger:
        area_diff = total_input_area - total_output_area
        logger.info(
            f"   ✅ Split into {len(zone_data)} zones "
            f"(input: {total_input_area:.1f} ha, output: {total_output_area:.1f} ha, "
            f"slivers filtered: {area_diff:.1f} ha)"
        )

    return zone_data


def _normalize_intersection_to_polygons(
    geom: Union[Polygon, MultiPolygon, BaseGeometry],
) -> List[Polygon]:
    """
    Normalize intersection result to list of Polygon objects.

    Handles Polygon, MultiPolygon, and GeometryCollection results.

    Args:
        geom: Shapely geometry from intersection operation

    Returns:
        List of Polygon objects (empty polygons filtered out)
    """
    from shapely.geometry import GeometryCollection

    if geom.is_empty:
        return []

    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty else []

    elif isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if not p.is_empty]

    elif isinstance(geom, GeometryCollection):
        # Extract only Polygon types from collection
        polys: List[Polygon] = []
        for g in geom.geoms:
            if isinstance(g, Polygon) and not g.is_empty:
                polys.append(g)
            elif isinstance(g, MultiPolygon):
                polys.extend([p for p in g.geoms if not p.is_empty])
        return polys

    else:
        # Ignore lines, points, etc.
        return []


# ===========================================================================
# SUMMARY STATISTICS
# ===========================================================================


def get_coverage_summary(
    covered_union: BaseGeometry,
    uncovered_gaps: BaseGeometry,
    gap_stats: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Get summary statistics for coverage analysis.

    Args:
        covered_union: Shapely geometry of covered area
        uncovered_gaps: Shapely geometry of uncovered gaps
        gap_stats: List of gap statistics dicts

    Returns:
        Dict with summary statistics
    """
    covered_area = covered_union.area if not covered_union.is_empty else 0
    uncovered_area = uncovered_gaps.area if not uncovered_gaps.is_empty else 0
    total_area = covered_area + uncovered_area

    return {
        "covered_area_m2": covered_area,
        "covered_area_ha": covered_area / 10000,
        "uncovered_area_m2": uncovered_area,
        "uncovered_area_ha": uncovered_area / 10000,
        "total_area_m2": total_area,
        "total_area_ha": total_area / 10000,
        "coverage_pct": 100 * covered_area / total_area if total_area > 0 else 0,
        "num_gaps": len(gap_stats),
        "largest_gap_ha": gap_stats[0]["area_ha"] if gap_stats else 0,
        "max_gap_distance_m": (
            max(g["max_distance_m"] for g in gap_stats) if gap_stats else 0
        ),
    }


# ===========================================================================
# ZONE BOUNDARY EXTRACTION (for Buffer Zone Consolidation)
# ===========================================================================


def extract_internal_zone_boundaries(
    zones_gdf: GeoDataFrame,
    buffer_tolerance: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> Optional[BaseGeometry]:
    """
    Extract internal zone boundaries (edges shared between adjacent zones).

    For buffer zone consolidation, we need to identify the lines where zones
    meet. Boreholes near these lines are "border" candidates; those far from
    these lines are "interior" and can be locked.

    Algorithm:
    1. Extract boundary LineString from each zone polygon
    2. Buffer each boundary by a small tolerance to handle precision issues
    3. Find intersections between adjacent zone boundaries
    4. Merge all internal boundaries into a single MultiLineString

    Args:
        zones_gdf: GeoDataFrame with zone polygons
        buffer_tolerance: Small buffer to handle coordinate precision (meters)
        logger: Optional logger

    Returns:
        MultiLineString or LineString of internal zone boundaries
        Returns None if zones_gdf is None or has <2 zones

    Example:
        >>> boundaries = extract_internal_zone_boundaries(zones_gdf)
        >>> distance_to_boundary = point.distance(boundaries)
        >>> is_border = distance_to_boundary <= buffer_width
    """
    from shapely.geometry import MultiLineString, LineString
    from shapely.ops import linemerge

    if zones_gdf is None or len(zones_gdf) < 2:
        if logger:
            logger.info("   ℹ️ Fewer than 2 zones - no internal boundaries")
        return None

    if logger:
        logger.info(f"   🔍 Extracting internal boundaries from {len(zones_gdf)} zones")

    # === APPROACH: Pairwise intersection of zone boundaries ===
    # For each pair of zones, find where their boundaries touch/overlap

    internal_boundaries = []
    zone_geoms = list(zones_gdf.geometry)
    zone_names = (
        list(zones_gdf["Name"])
        if "Name" in zones_gdf.columns
        else [f"Zone_{i}" for i in range(len(zones_gdf))]
    )

    # Pairwise comparison
    for i in range(len(zone_geoms)):
        for j in range(i + 1, len(zone_geoms)):
            geom_i = zone_geoms[i]
            geom_j = zone_geoms[j]

            if geom_i.is_empty or geom_j.is_empty:
                continue

            # Check if zones are adjacent (touching or overlapping slightly)
            # Use a small buffer to handle coordinate precision
            geom_i_buffered = geom_i.buffer(buffer_tolerance)

            if not geom_i_buffered.intersects(geom_j):
                continue  # Zones don't touch

            # Get the shared boundary between zones
            # Method: intersection of the two boundaries
            boundary_i = geom_i.boundary
            boundary_j = geom_j.boundary

            # Buffer the boundaries slightly and intersect
            shared = boundary_i.buffer(buffer_tolerance).intersection(
                boundary_j.buffer(buffer_tolerance)
            )

            if shared.is_empty:
                continue

            # Extract linear components from the intersection
            if shared.geom_type == "LineString":
                internal_boundaries.append(shared)
            elif shared.geom_type == "MultiLineString":
                internal_boundaries.extend(list(shared.geoms))
            elif shared.geom_type == "Polygon":
                # Buffer intersection resulted in polygon; extract its boundary
                internal_boundaries.append(shared.boundary)
            elif shared.geom_type == "GeometryCollection":
                for geom in shared.geoms:
                    if geom.geom_type in ("LineString", "MultiLineString"):
                        if geom.geom_type == "LineString":
                            internal_boundaries.append(geom)
                        else:
                            internal_boundaries.extend(list(geom.geoms))
                    elif geom.geom_type == "Polygon":
                        internal_boundaries.append(geom.boundary)

            if logger:
                logger.debug(
                    f"      Found shared boundary between {zone_names[i]} and {zone_names[j]}"
                )

    if not internal_boundaries:
        if logger:
            logger.warning("   ⚠️ No internal boundaries found between zones")
        return None

    # Merge all internal boundaries
    try:
        merged = linemerge(internal_boundaries)
    except Exception:
        # If linemerge fails, fall back to MultiLineString
        merged = MultiLineString(internal_boundaries)

    if logger:
        if hasattr(merged, "geoms"):
            n_lines = len(merged.geoms)
        else:
            n_lines = 1
        total_length = merged.length
        logger.info(
            f"   ✅ Extracted {n_lines} internal boundary segment(s), "
            f"total length {total_length:.0f}m"
        )

    return merged
