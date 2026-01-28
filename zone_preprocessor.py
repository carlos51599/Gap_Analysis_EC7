#!/usr/bin/env python3
"""
Zone Preprocessor for Pre-Analysis Geometry Cutting

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Process zone geometries to eliminate overlaps before any
analysis (gap computation, ILP optimization, visualization). Higher-priority
zones claim overlapping regions from lower-priority zones.

This is a FULL ANALYSIS change - all downstream processing uses the
effective_geometry column instead of the original geometry.

Key Functions:
- preprocess_zones(): Main entry point
- detect_zone_overlaps(): Identify overlapping pairs
- apply_priority_cuts(): Apply geometry subtraction
- compare_zone_priority(): Determine which zone wins overlap

Priority Rules:
1. Lower max_spacing_m = higher priority (stricter requirement wins)
2. IF spacing equal (within tolerance): lower order = higher priority
3. By design, no two zones have equal spacing AND equal order

CONFIGURATION ARCHITECTURE:
- min_overlap_area_m2: Threshold for significant overlaps (default 100)
- min_component_area_m2: Filter tiny geometry fragments (default 1)
- spacing_tolerance: Float comparison tolerance for max_spacing_m (default 0.01)

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

from typing import Dict, List, Tuple, Optional, Any
import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from Gap_Analysis_EC7.shapefile_config import SPACING_EQUALITY_TOLERANCE_M

# Module-level logger
_logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 📋 CONFIGURATION DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_MIN_OVERLAP_AREA_M2 = 100.0  # Ignore overlaps smaller than 100 m²
DEFAULT_MIN_COMPONENT_AREA_M2 = 1.0  # Filter geometry slivers under 1 m²


# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


def preprocess_zones(
    zones_gdf: gpd.GeoDataFrame,
    min_overlap_area_m2: float = DEFAULT_MIN_OVERLAP_AREA_M2,
    min_component_area_m2: float = DEFAULT_MIN_COMPONENT_AREA_M2,
    spacing_tolerance: float = SPACING_EQUALITY_TOLERANCE_M,
    logger: Optional[logging.Logger] = None,
) -> gpd.GeoDataFrame:
    """
    Cut overlapping zone geometries based on priority.

    Priority rules:
    1. Lower max_spacing_m = higher priority (stricter wins)
    2. If spacing equal (within tolerance): lower order = higher priority

    This should be called immediately after loading zones, before any analysis.

    Args:
        zones_gdf: GeoDataFrame with zone boundaries.
            Required columns: geometry, zone_id, max_spacing_m
            Optional columns: order (defaults to 999)
        min_overlap_area_m2: Ignore overlaps smaller than this (sq meters)
        min_component_area_m2: Filter geometry fragments smaller than this
        spacing_tolerance: Consider spacings equal if within this tolerance
        logger: Optional logger for progress messages

    Returns:
        GeoDataFrame with additional columns:
            - original_geometry: Pre-cut geometry
            - effective_geometry: Post-cut geometry
            - was_cut: True if zone was cut by higher-priority zone
            - cut_by: List of zone_ids that cut this zone
            - cut_area_m2: Total area removed
            - fully_consumed: True if zone completely eliminated

    Note:
        After return, caller should do:
        zones_gdf["geometry"] = zones_gdf["effective_geometry"]
    """
    log = logger or _logger

    log.info("🔧 Pre-processing zones for overlap resolution...")

    # Make a copy to avoid modifying the original
    zones_gdf = zones_gdf.copy()

    # Ensure required columns exist
    if "order" not in zones_gdf.columns:
        zones_gdf["order"] = 999
        log.debug("   Added default 'order' column (999)")

    # Initialize preprocessing columns
    zones_gdf["original_geometry"] = zones_gdf["geometry"].copy()
    zones_gdf["effective_geometry"] = zones_gdf["geometry"].copy()
    zones_gdf["was_cut"] = False
    zones_gdf["cut_by"] = [[] for _ in range(len(zones_gdf))]
    zones_gdf["cut_area_m2"] = 0.0
    zones_gdf["fully_consumed"] = False

    # Detect overlaps
    overlaps = detect_zone_overlaps(
        zones_gdf, min_overlap_area_m2, spacing_tolerance, log
    )

    if not overlaps:
        log.info("   ✅ No significant overlaps detected")
        return zones_gdf

    log.info(f"   📐 Found {len(overlaps)} overlap(s) to resolve")

    # Apply priority-based cutting
    zones_gdf = apply_priority_cuts(zones_gdf, overlaps, min_component_area_m2, log)

    # Summary
    cut_count = zones_gdf["was_cut"].sum()
    consumed_count = zones_gdf["fully_consumed"].sum()
    total_cut_area = zones_gdf["cut_area_m2"].sum()

    log.info(
        f"   ✅ Preprocessing complete: {cut_count} zone(s) cut, "
        f"{total_cut_area:.0f} m² overlap resolved"
    )

    if consumed_count > 0:
        log.warning(
            f"   ⚠️ {consumed_count} zone(s) fully consumed by higher-priority zones"
        )

    return zones_gdf


# ═══════════════════════════════════════════════════════════════════════════
# 🔍 OVERLAP DETECTION
# ═══════════════════════════════════════════════════════════════════════════


def detect_zone_overlaps(
    zones_gdf: gpd.GeoDataFrame,
    min_overlap_area_m2: float,
    spacing_tolerance: float,
    logger: Optional[logging.Logger] = None,
) -> List[Tuple[str, str, BaseGeometry, float]]:
    """
    Detect significant overlaps between zone pairs.

    For each pair of zones that overlap with area > min_overlap_area_m2,
    determines which zone has higher priority and returns the overlap info.

    Args:
        zones_gdf: GeoDataFrame with zone_id, geometry, max_spacing_m, order
        min_overlap_area_m2: Minimum overlap area to consider
        spacing_tolerance: Tolerance for equal spacing comparison
        logger: Optional logger

    Returns:
        List of (winner_zone_id, loser_zone_id, overlap_geometry, overlap_area)
        The winner claims the overlap; the loser is cut.
    """
    log = logger or _logger
    overlaps: List[Tuple[str, str, BaseGeometry, float]] = []

    zone_ids = zones_gdf["zone_id"].tolist()
    n_zones = len(zone_ids)

    for i in range(n_zones):
        for j in range(i + 1, n_zones):
            zone_a_id = zone_ids[i]
            zone_b_id = zone_ids[j]

            # Get geometries
            geom_a = zones_gdf.loc[zones_gdf["zone_id"] == zone_a_id, "geometry"].iloc[
                0
            ]
            geom_b = zones_gdf.loc[zones_gdf["zone_id"] == zone_b_id, "geometry"].iloc[
                0
            ]

            # Check for intersection (fast bounding box check first)
            if not geom_a.intersects(geom_b):
                continue

            # Compute actual intersection
            overlap = geom_a.intersection(geom_b)
            overlap_area = overlap.area

            # Skip tiny overlaps (includes boundary-only touching where area=0)
            if overlap_area < min_overlap_area_m2:
                continue

            # Determine priority winner
            winner_id = compare_zone_priority(
                zone_a_id, zone_b_id, zones_gdf, spacing_tolerance
            )
            loser_id = zone_b_id if winner_id == zone_a_id else zone_a_id

            overlaps.append((winner_id, loser_id, overlap, overlap_area))

            log.info(f"   📐 {winner_id} claims {overlap_area:.0f} m² from {loser_id}")

    return overlaps


def compare_zone_priority(
    zone_a_id: str,
    zone_b_id: str,
    zones_gdf: gpd.GeoDataFrame,
    spacing_tolerance: float = SPACING_EQUALITY_TOLERANCE_M,
) -> str:
    """
    Determine which zone has higher priority.

    Priority rules:
    1. Lower max_spacing_m wins (stricter requirement)
    2. If spacing equal (within tolerance): lower order wins

    Args:
        zone_a_id: First zone ID
        zone_b_id: Second zone ID
        zones_gdf: GeoDataFrame with zone data
        spacing_tolerance: Tolerance for equal spacing comparison

    Returns:
        zone_id of the higher-priority zone (the winner)
    """
    # Get zone data
    row_a = zones_gdf.loc[zones_gdf["zone_id"] == zone_a_id].iloc[0]
    row_b = zones_gdf.loc[zones_gdf["zone_id"] == zone_b_id].iloc[0]

    spacing_a = float(row_a["max_spacing_m"])
    spacing_b = float(row_b["max_spacing_m"])
    order_a = int(row_a.get("order", 999))
    order_b = int(row_b.get("order", 999))

    # Rule 1: Lower spacing wins (stricter requirement)
    spacing_diff = abs(spacing_a - spacing_b)
    if spacing_diff > spacing_tolerance:
        # Spacings are different - lower wins
        return zone_a_id if spacing_a < spacing_b else zone_b_id

    # Rule 2: Spacing is equal (within tolerance) - lower order wins
    return zone_a_id if order_a < order_b else zone_b_id


# ═══════════════════════════════════════════════════════════════════════════
# ✂️ APPLY CUTS
# ═══════════════════════════════════════════════════════════════════════════


def apply_priority_cuts(
    zones_gdf: gpd.GeoDataFrame,
    overlaps: List[Tuple[str, str, BaseGeometry, float]],
    min_component_area_m2: float,
    logger: Optional[logging.Logger] = None,
) -> gpd.GeoDataFrame:
    """
    Apply geometry cuts based on detected overlaps.

    For each overlap (winner, loser, geometry, area):
    - winner keeps its geometry unchanged
    - loser's effective_geometry is reduced by the overlap

    A zone may be cut by multiple higher-priority zones. Cuts are accumulated.

    Args:
        zones_gdf: GeoDataFrame with effective_geometry column
        overlaps: List from detect_zone_overlaps()
        min_component_area_m2: Filter tiny geometry fragments
        logger: Optional logger

    Returns:
        Modified zones_gdf with cuts applied
    """
    log = logger or _logger

    for winner_id, loser_id, overlap_geom, overlap_area in overlaps:
        # Find loser row index
        loser_mask = zones_gdf["zone_id"] == loser_id
        if not loser_mask.any():
            log.warning(f"   ⚠️ Zone {loser_id} not found, skipping cut")
            continue

        loser_idx = zones_gdf.index[loser_mask].tolist()[0]

        # Get CURRENT effective geometry (may already be cut by other zones)
        current_geom = zones_gdf.at[loser_idx, "effective_geometry"]

        # Skip if already fully consumed
        if current_geom.is_empty:
            continue

        # Subtract overlap
        new_geom = current_geom.difference(overlap_geom)

        # Clean up geometry (handle GeometryCollection, filter tiny fragments)
        new_geom = _clean_geometry_to_polygon(new_geom, min_component_area_m2)

        # Update row
        zones_gdf.at[loser_idx, "effective_geometry"] = new_geom
        zones_gdf.at[loser_idx, "was_cut"] = True
        zones_gdf.at[loser_idx, "cut_by"] = zones_gdf.at[loser_idx, "cut_by"] + [
            winner_id
        ]
        zones_gdf.at[loser_idx, "cut_area_m2"] += overlap_area

        # Check if fully consumed
        if new_geom.is_empty or new_geom.area < min_component_area_m2:
            zones_gdf.at[loser_idx, "fully_consumed"] = True
            log.warning(f"   ⚠️ Zone {loser_id} fully consumed by higher-priority zones")
        else:
            log.debug(
                f"   ✂️ Cut {loser_id}: removed {overlap_area:.0f} m² "
                f"(claimed by {winner_id})"
            )

    return zones_gdf


# ═══════════════════════════════════════════════════════════════════════════
# 🧹 GEOMETRY CLEANING
# ═══════════════════════════════════════════════════════════════════════════


def _clean_geometry_to_polygon(
    geometry: BaseGeometry,
    min_area_m2: float = 1.0,
) -> BaseGeometry:
    """
    Clean up geometry after cutting operations.

    Handles:
    - None/empty → return empty Polygon
    - Invalid topology → buffer(0) fix
    - GeometryCollection → extract polygons only (discard lines/points)
    - MultiPolygon with tiny fragments → filter by area

    Args:
        geometry: Shapely geometry (may be messy after difference())
        min_area_m2: Minimum area threshold for components

    Returns:
        Clean Polygon, MultiPolygon, or empty Polygon
    """
    if geometry is None or geometry.is_empty:
        return Polygon()

    # Fix invalid topology
    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    # Extract only polygon types from GeometryCollection
    if isinstance(geometry, GeometryCollection):
        polygons = [g for g in geometry.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polygons:
            return Polygon()

        # Flatten any MultiPolygons and filter by area
        all_polys = []
        for p in polygons:
            if isinstance(p, MultiPolygon):
                all_polys.extend([g for g in p.geoms if g.area >= min_area_m2])
            elif p.area >= min_area_m2:
                all_polys.append(p)

        if not all_polys:
            return Polygon()
        elif len(all_polys) == 1:
            return all_polys[0]
        else:
            return MultiPolygon(all_polys)

    # Handle MultiPolygon - filter tiny components
    if isinstance(geometry, MultiPolygon):
        significant = [g for g in geometry.geoms if g.area >= min_area_m2]
        if not significant:
            return Polygon()
        elif len(significant) == 1:
            return significant[0]
        else:
            return MultiPolygon(significant)

    # Single Polygon - check area
    if isinstance(geometry, Polygon) and geometry.area < min_area_m2:
        return Polygon()

    return geometry
