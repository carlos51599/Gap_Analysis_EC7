#!/usr/bin/env python3
"""
Centreline Constraints Module for Gap Analysis EC7

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Compute centreline geometries for designated shapefile layers,
sample borehole positions at zone-spacing intervals along centrelines, and return
them as locked constants for ILP solver passes.

Key Interactions:
    Input:  Shapefile GeoDataFrames, shapefile_config centreline settings
    Output: List of locked borehole dicts compatible with solver pipeline
    Config: shapefile_config.py centreline sub-dicts per layer

Navigation Guide:
    1. CENTRELINE COMPUTATION — wraps Centreline_Tool/centreline.py
    2. BOREHOLE SAMPLING — places boreholes at spacing intervals along centreline
    3. PRE-COVERAGE — computes which test points are covered by centreline BHs
    4. ORCHESTRATION — main entry point for generating all centreline boreholes

For Navigation: Use VS Code outline (Ctrl+Shift+O) to jump between sections.

CONFIGURATION ARCHITECTURE:
- All config accessed through shapefile_config helper functions
- Business logic functions accept explicit parameters only (no CONFIG access)
- Orchestrator function extracts config values at boundary

MODIFICATION POINT: Extend sampling_mode options in sample_boreholes_along_centreline()
"""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    import geopandas as gpd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 CENTRELINE COMPUTATION SECTION
# ═══════════════════════════════════════════════════════════════════════════


def _import_compute_centreline():
    """
    Import compute_centreline from Centreline_Tool with path setup.

    Returns:
        The compute_centreline function, or None if import fails.
    """
    # Add workspace root to path if needed (Centreline_Tool is at workspace root)
    workspace_root = Path(__file__).resolve().parent.parent.parent
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    try:
        from Centreline_Tool.centreline import compute_centreline

        return compute_centreline
    except ImportError as e:
        logger.warning(f"⚠️ Could not import Centreline_Tool: {e}")
        return None


def compute_centreline_for_layer(
    layer_gdf: "gpd.GeoDataFrame",
    min_branch_length_m: float,
    spacing_m: float,
    simplify_tolerance_m: float,
    log: Optional[logging.Logger] = None,
) -> List[BaseGeometry]:
    """
    Compute centreline geometries for all polygon features in a layer.

    Uses the Voronoi medial-axis algorithm from Centreline_Tool/centreline.py.
    Returns one centreline per feature (LineString or MultiLineString).

    Args:
        layer_gdf: GeoDataFrame with polygon features
        min_branch_length_m: Minimum branch length to keep (prune shorter)
        spacing_m: Boundary point density for Voronoi computation
        simplify_tolerance_m: Simplification tolerance (0 = none)
        log: Optional logger

    Returns:
        List of centreline geometries (one per feature, None entries skipped).
    """
    compute_fn = _import_compute_centreline()
    if compute_fn is None:
        if log:
            log.warning("   ⚠️ Centreline computation unavailable (import failed)")
        return []

    centrelines: List[BaseGeometry] = []

    for idx, row in layer_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        try:
            centreline = compute_fn(
                polygon=geom,
                spacing=spacing_m,
                min_branch_length=min_branch_length_m,
                simplify_tolerance=simplify_tolerance_m,
            )
            if centreline is not None and not centreline.is_empty:
                centrelines.append(centreline)
                if log:
                    log.info(
                        f"      Feature {idx}: centreline {centreline.length:.1f}m"
                    )
            else:
                if log:
                    log.warning(f"      Feature {idx}: centreline computation failed")
        except Exception as e:
            if log:
                log.warning(f"      Feature {idx}: centreline error: {e}")

    return centrelines


# ═══════════════════════════════════════════════════════════════════════════
# 📍 BOREHOLE SAMPLING SECTION
# ═══════════════════════════════════════════════════════════════════════════


def _sample_points_along_line(
    line: LineString,
    interval_m: float,
) -> List[Tuple[float, float]]:
    """
    Sample (x, y) points at regular intervals along a single LineString.

    Places points at the start, at every interval, and at the end.
    Handles edge case where line is shorter than interval.

    Args:
        line: Shapely LineString geometry
        interval_m: Distance between sample points in metres

    Returns:
        List of (x, y) coordinate tuples along the line.
    """
    total_length = line.length
    if total_length < 1.0:
        return []

    # Number of intervals (at least 1, so we get start + end)
    n_intervals = max(1, int(total_length / interval_m))
    distances = np.linspace(0, total_length, n_intervals + 1)

    points: List[Tuple[float, float]] = []
    for d in distances:
        pt = line.interpolate(d)
        points.append((pt.x, pt.y))

    return points


def sample_boreholes_along_centreline(
    centreline: BaseGeometry,
    spacing_m: float,
    coverage_radius: float,
    layer_key: str,
    zone_name: str,
    dedup_tolerance_m: float = 1.0,
    log: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Sample borehole positions at regular intervals along a centreline.

    Handles both LineString and MultiLineString. Deduplicates points
    within tolerance (branch junctions produce nearby points).

    Args:
        centreline: LineString or MultiLineString geometry
        spacing_m: Interval between boreholes (zone's max_spacing_m)
        coverage_radius: Coverage radius for each borehole (= spacing_m)
        layer_key: Source layer key for provenance
        zone_name: Zone name for provenance (e.g., "Highways_0")
        dedup_tolerance_m: Merge points closer than this (default 1m)
        log: Optional logger

    Returns:
        List of borehole dicts with keys:
            x, y, coverage_radius, source_pass, status, zone_id,
            is_centreline, centreline_distance_m
    """
    # === STEP 1: Normalise to list of LineStrings ===
    lines: List[LineString] = []
    if isinstance(centreline, MultiLineString):
        lines = list(centreline.geoms)
    elif isinstance(centreline, LineString):
        lines = [centreline]
    else:
        if log:
            log.warning(f"   ⚠️ Unexpected centreline type: {type(centreline)}")
        return []

    # === STEP 2: Sample points along each line ===
    raw_points: List[Tuple[float, float]] = []
    for line in lines:
        pts = _sample_points_along_line(line, spacing_m)
        raw_points.extend(pts)

    if not raw_points:
        return []

    # === STEP 3: Deduplicate near-identical points (branch junctions) ===
    deduped = _deduplicate_points(raw_points, dedup_tolerance_m)

    # === STEP 3b: Enforce minimum spacing between all kept points ===
    # Branch junctions produce start-of-branch points that may be closer
    # than spacing_m to an existing borehole on the parent branch.
    # Drop any point too close to an already-accepted point.
    # Same-line points are always >= spacing_m apart (from linspace), so
    # this only removes cross-branch junction points that violate spacing.
    if spacing_m > dedup_tolerance_m:
        before_count = len(deduped)
        deduped = _deduplicate_points(deduped, spacing_m)
        if log and len(deduped) < before_count:
            log.info(
                f"   🛤️ Spacing enforcement removed {before_count - len(deduped)} "
                f"branch-junction points (min spacing {spacing_m:.0f}m)"
            )

    # === STEP 4: Create borehole dicts ===
    from Gap_Analysis_EC7.shapefile_config import make_zone_id

    zone_id = make_zone_id(layer_key, zone_name)
    boreholes: List[Dict[str, Any]] = []

    for x, y in deduped:
        boreholes.append(
            {
                "x": x,
                "y": y,
                "coverage_radius": coverage_radius,
                "source_pass": "First Pass",
                "status": "locked",
                "zone_id": zone_id,
                "is_centreline": True,
            }
        )

    if log:
        total_length = sum(l.length for l in lines)
        log.info(
            f"   🛤️ Sampled {len(boreholes)} centreline boreholes "
            f"({total_length:.0f}m, interval={spacing_m:.0f}m, "
            f"deduped from {len(raw_points)})"
        )

    return boreholes


def _deduplicate_points(
    points: List[Tuple[float, float]],
    tolerance: float,
) -> List[Tuple[float, float]]:
    """
    Remove near-duplicate points within tolerance distance.

    Uses a greedy approach: keep first point, skip any within tolerance
    of an already-kept point. O(n*k) where k is kept points.

    Args:
        points: List of (x, y) tuples
        tolerance: Merge distance in metres

    Returns:
        Deduplicated list of (x, y) tuples.
    """
    if not points:
        return []

    tol_sq = tolerance * tolerance
    kept: List[Tuple[float, float]] = [points[0]]

    for px, py in points[1:]:
        is_dup = False
        for kx, ky in kept:
            dx = px - kx
            dy = py - ky
            if dx * dx + dy * dy < tol_sq:
                is_dup = True
                break
        if not is_dup:
            kept.append((px, py))

    return kept


# ═══════════════════════════════════════════════════════════════════════════
# 🔒 PRE-COVERAGE COMPUTATION SECTION
# ═══════════════════════════════════════════════════════════════════════════


def compute_centreline_precoverage(
    test_points: List[Dict[str, Any]],
    centreline_boreholes: List[Dict[str, Any]],
    log: Optional[logging.Logger] = None,
) -> Set[int]:
    """
    Identify test points already covered by centreline-locked boreholes.

    This function is used by the first pass to remove pre-covered test points
    from the ILP constraint set. Coverage uses each test point's required_radius
    (not the borehole's coverage_radius).

    Args:
        test_points: Test points with x, y, required_radius keys
        centreline_boreholes: Locked centreline boreholes with x, y keys

    Returns:
        Set of test point indices that are pre-covered by centreline boreholes.
    """
    if not centreline_boreholes or not test_points:
        return set()

    # Vectorised distance computation for performance
    bh_coords = np.array(
        [(bh["x"], bh["y"]) for bh in centreline_boreholes], dtype=np.float64
    )
    pre_covered: Set[int] = set()

    for i, tp in enumerate(test_points):
        tp_x, tp_y = tp["x"], tp["y"]
        required_radius = tp.get("required_radius", tp.get("coverage_radius", 100.0))

        # Vectorised distance to all centreline boreholes
        dx = bh_coords[:, 0] - tp_x
        dy = bh_coords[:, 1] - tp_y
        dists_sq = dx * dx + dy * dy

        if np.any(dists_sq <= required_radius * required_radius):
            pre_covered.add(i)

    if log:
        pct = len(pre_covered) / len(test_points) * 100 if test_points else 0
        log.info(
            f"   🔒 Centreline pre-coverage: {len(pre_covered)} of "
            f"{len(test_points)} test points ({pct:.1f}%)"
        )

    return pre_covered


# ═══════════════════════════════════════════════════════════════════════════
# 🚀 ORCHESTRATION SECTION
# ═══════════════════════════════════════════════════════════════════════════


def generate_centreline_boreholes(
    all_shapefiles: Dict[str, "gpd.GeoDataFrame"],
    log: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate centreline-constrained boreholes for all enabled layers.

    Main entry point called once during startup, before the first solver
    pass. Returns locked boreholes that must be included in every
    optimisation pass as constants.

    Args:
        all_shapefiles: Dict of layer_key -> GeoDataFrame
        log: Optional logger

    Returns:
        Tuple of (centreline_boreholes, centreline_stats)
        - centreline_boreholes: List of locked borehole dicts
        - centreline_stats: Summary statistics per layer
    """
    from Gap_Analysis_EC7.shapefile_config import (
        get_centreline_config,
        get_centreline_enabled_layers,
        get_zone_max_spacing,
    )

    enabled_layers = get_centreline_enabled_layers()
    stats: Dict[str, Any] = {"layers": {}, "total_boreholes": 0, "geometries_wkt": []}

    if not enabled_layers:
        if log:
            log.info("   🛤️ Centreline constraints: no layers enabled")
        return [], stats

    all_boreholes: List[Dict[str, Any]] = []

    for layer_key in enabled_layers:
        if log:
            log.info(f"   🛤️ Computing centreline for '{layer_key}'...")

        # Get config values (orchestrator boundary)
        cl_config = get_centreline_config(layer_key)
        min_branch = cl_config.get("min_branch_length_m", 50.0)
        spacing = cl_config.get("spacing_m", 5.0)
        simplify_tol = cl_config.get("simplify_tolerance_m", 2.0)

        # Get layer GeoDataFrame
        layer_gdf = all_shapefiles.get(layer_key)
        if layer_gdf is None:
            if log:
                log.warning(f"   ⚠️ Layer '{layer_key}' not in loaded shapefiles")
            stats["layers"][layer_key] = {"status": "missing_gdf", "boreholes": 0}
            continue

        # Compute centrelines for all features in layer
        centrelines = compute_centreline_for_layer(
            layer_gdf=layer_gdf,
            min_branch_length_m=min_branch,
            spacing_m=spacing,
            simplify_tolerance_m=simplify_tol,
            log=log,
        )

        if not centrelines:
            stats["layers"][layer_key] = {"status": "no_centrelines", "boreholes": 0}
            continue

        # Store centreline geometries as WKT for visualization
        for cl in centrelines:
            stats["geometries_wkt"].append(cl.wkt)

        # Sample boreholes along each centreline at zone spacing
        layer_boreholes: List[Dict[str, Any]] = []
        for feat_idx, centreline in enumerate(centrelines):
            # Zone name follows the naming convention for unnamed layers
            from Gap_Analysis_EC7.shapefile_config import get_layer_config

            layer_cfg = get_layer_config(layer_key)
            display_name = layer_cfg.get("display_name", layer_key)
            zone_name = f"{display_name}_{feat_idx}"

            # Get zone-specific max_spacing_m
            zone_spacing = get_zone_max_spacing(layer_key, zone_name)

            bhs = sample_boreholes_along_centreline(
                centreline=centreline,
                spacing_m=zone_spacing,
                coverage_radius=zone_spacing,
                layer_key=layer_key,
                zone_name=zone_name,
                log=log,
            )
            layer_boreholes.extend(bhs)

        all_boreholes.extend(layer_boreholes)
        total_length = sum(cl.length for cl in centrelines)
        stats["layers"][layer_key] = {
            "status": "success",
            "features": len(centrelines),
            "total_length_m": round(total_length, 1),
            "boreholes": len(layer_boreholes),
        }

    stats["total_boreholes"] = len(all_boreholes)

    if log:
        log.info(
            f"   🛤️ Centreline total: {len(all_boreholes)} locked boreholes "
            f"across {len(enabled_layers)} layer(s)"
        )

    return all_boreholes, stats
