"""
Cross-Zone Reachability Consolidation (CZRC) Geometry Module.

Computes coverage clouds and consolidation regions for cross-zone
borehole optimization. This replaces the border buffer approach
with precise geometric computation based on test point reachability.

Architectural Overview:
    Responsibility: Compute CZRC regions from zone geometries and spacing requirements
    Key Interactions: Called from consolidation.py, returns geometries for viz/optimization
    Navigation Guide: See compute_czrc_consolidation_region() for main algorithm

Core Concept:
    Instead of buffering zone borders by an arbitrary distance, CZRC computes
    the exact regions where a borehole could satisfy test points from multiple zones.

    For each zone, a "coverage cloud" is computed as the union of circles
    (radius = max_spacing) around test points. The intersection of clouds
    from different zones gives the exact region where cross-zone boreholes
    are possible.

Key Formula:
    For two test points TP_A and TP_B with spacing requirements r_A and r_B:
    A single borehole can cover both IFF: r_A + r_B > distance(TP_A, TP_B)

    This is geometrically equivalent to: coverage_cloud_A INTERSECTS coverage_cloud_B
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from shapely import wkt
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# Type alias for GeoDataFrame (avoid import for type hints)
GeoDataFrame = Any


# ═══════════════════════════════════════════════════════════════════════════
# 🌐 COVERAGE CLOUD COMPUTATION SECTION
# ═══════════════════════════════════════════════════════════════════════════


def compute_zone_coverage_cloud(
    zone_geometry: BaseGeometry,
    max_spacing_m: float,
    grid_spacing: float,
    logger: Optional[logging.Logger] = None,
) -> BaseGeometry:
    """
    Compute the coverage cloud for a single zone.

    The coverage cloud is the union of circles (radius = max_spacing_m) centered
    at a grid of test points within the zone. This represents everywhere a
    borehole could satisfy at least one test point in this zone.

    Args:
        zone_geometry: The zone's polygon geometry
        max_spacing_m: Maximum spacing requirement for this zone (coverage radius)
        grid_spacing: Spacing between test point grid (typically test_spacing_mult × max_spacing)
        logger: Optional logger for progress info

    Returns:
        Polygon or MultiPolygon representing the coverage cloud
    """
    # Generate test points within the zone
    test_points = _generate_zone_test_points(zone_geometry, grid_spacing)

    if not test_points:
        if logger:
            logger.warning("   ⚠️ No test points generated for zone")
        return zone_geometry.buffer(0)  # Empty but valid geometry

    # Create coverage circles for each test point
    coverage_circles = [pt.buffer(max_spacing_m) for pt in test_points]

    # Union all circles into a single coverage cloud
    coverage_cloud = unary_union(coverage_circles)

    if logger:
        logger.debug(
            f"   📊 Coverage cloud: {len(test_points)} test points, "
            f"radius={max_spacing_m:.1f}m"
        )

    return coverage_cloud


def _generate_zone_test_points(
    zone_geometry: BaseGeometry,
    spacing: float,
) -> List[Point]:
    """
    Generate test points within a zone using hexagonal grid pattern.

    Uses the same hexagonal geometry as the main gap analysis for consistency.

    Args:
        zone_geometry: Zone polygon
        spacing: Center-to-center distance between points

    Returns:
        List of Point objects within the zone
    """
    test_points = []

    # Hexagonal grid parameters
    d = spacing
    dx = d  # Horizontal spacing within row
    dy = d * math.sqrt(3) / 2  # Vertical spacing between rows (~0.866 * d)

    bounds = zone_geometry.bounds  # (minx, miny, maxx, maxy)

    row_idx = 0
    y = bounds[1]
    while y <= bounds[3] + dy:
        # Offset odd rows by half dx (creates honeycomb pattern)
        x_offset = (row_idx % 2) * (dx / 2)
        x = bounds[0] + x_offset
        while x <= bounds[2] + dx:
            pt = Point(x, y)
            if zone_geometry.contains(pt):
                test_points.append(pt)
            x += dx
        y += dy
        row_idx += 1

    return test_points


# ═══════════════════════════════════════════════════════════════════════════
# 🔗 PAIRWISE INTERSECTION SECTION
# ═══════════════════════════════════════════════════════════════════════════


def compute_pairwise_intersection(
    cloud_a: BaseGeometry,
    cloud_b: BaseGeometry,
    zone_name_a: str,
    zone_name_b: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[BaseGeometry]:
    """
    Compute the intersection of two coverage clouds.

    The intersection represents the region where a single borehole could
    satisfy test points from BOTH zones.

    Args:
        cloud_a: Coverage cloud from zone A
        cloud_b: Coverage cloud from zone B
        zone_name_a: Name of zone A (for logging)
        zone_name_b: Name of zone B (for logging)
        logger: Optional logger

    Returns:
        Intersection geometry, or None if clouds don't intersect
    """
    # Early exit: check bounding box overlap first (fast)
    if not cloud_a.bounds or not cloud_b.bounds:
        return None

    # Check if bounding boxes overlap
    box_a = cloud_a.bounds
    box_b = cloud_b.bounds

    if (
        box_a[2] < box_b[0]
        or box_b[2] < box_a[0]
        or box_a[3] < box_b[1]
        or box_b[3] < box_a[1]
    ):
        # Bounding boxes don't overlap, no intersection possible
        if logger:
            logger.debug(f"   ⏭️ {zone_name_a} ↔ {zone_name_b}: No bbox overlap")
        return None

    # Compute actual intersection
    try:
        intersection = cloud_a.intersection(cloud_b)

        if intersection.is_empty:
            if logger:
                logger.debug(f"   ⏭️ {zone_name_a} ↔ {zone_name_b}: Empty intersection")
            return None

        if logger:
            area = intersection.area
            logger.debug(
                f"   ✅ {zone_name_a} ↔ {zone_name_b}: "
                f"Intersection area = {area:.0f} m²"
            )

        return intersection

    except Exception as e:
        if logger:
            logger.warning(
                f"   ⚠️ {zone_name_a} ↔ {zone_name_b}: " f"Intersection failed: {e}"
            )
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 🌍 FULL CZRC REGION COMPUTATION SECTION
# ═══════════════════════════════════════════════════════════════════════════


def compute_czrc_consolidation_region(
    zones_gdf: GeoDataFrame,
    test_spacing_mult: float = 0.2,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Compute the full CZRC consolidation region from all zones.

    Algorithm:
    1. For each zone: compute coverage_cloud (union of test point buffers)
    2. For each zone pair: compute intersection of their clouds
    3. Union all pairwise intersections → total consolidation region

    Args:
        zones_gdf: GeoDataFrame with columns: geometry, zone_name, max_spacing_m
        test_spacing_mult: Multiplier for test point spacing (e.g., 0.2 × max_spacing)
        logger: Optional logger

    Returns:
        Dict with:
        - "total_region": Union of all pairwise intersections
        - "total_region_wkt": WKT string of total region
        - "coverage_clouds": Dict of {zone_name: cloud_geometry}
        - "coverage_clouds_wkt": Dict of {zone_name: wkt_string}
        - "pairwise_regions": Dict of {"ZoneA_ZoneB": intersection_geometry}
        - "pairwise_wkts": Dict of {"ZoneA_ZoneB": wkt_string}
        - "triple_overlaps": Dict of {"ZoneA_ZoneB_ZoneC": geometry}
        - "triple_overlaps_wkt": Dict of {"ZoneA_ZoneB_ZoneC": wkt_string}
        - "stats": Statistics dict
    """
    if logger:
        logger.info("   🌐 Computing CZRC consolidation region...")

    # === STEP 1: COMPUTE COVERAGE CLOUDS PER ZONE ===
    coverage_clouds: Dict[str, BaseGeometry] = {}
    zone_names = list(zones_gdf["zone_name"])

    for _, row in zones_gdf.iterrows():
        zone_name = row["zone_name"]
        zone_geom = row["geometry"]
        max_spacing = row["max_spacing_m"]
        grid_spacing = max_spacing * test_spacing_mult

        cloud = compute_zone_coverage_cloud(
            zone_geometry=zone_geom,
            max_spacing_m=max_spacing,
            grid_spacing=grid_spacing,
            logger=logger,
        )
        coverage_clouds[zone_name] = cloud

        if logger:
            logger.info(
                f"      • {zone_name}: coverage cloud computed "
                f"(spacing={max_spacing:.0f}m)"
            )

    # === STEP 2: COMPUTE PAIRWISE INTERSECTIONS ===
    pairwise_regions: Dict[str, BaseGeometry] = {}

    n_zones = len(zone_names)
    for i in range(n_zones):
        for j in range(i + 1, n_zones):
            zone_a = zone_names[i]
            zone_b = zone_names[j]

            intersection = compute_pairwise_intersection(
                cloud_a=coverage_clouds[zone_a],
                cloud_b=coverage_clouds[zone_b],
                zone_name_a=zone_a,
                zone_name_b=zone_b,
                logger=logger,
            )

            if intersection is not None and not intersection.is_empty:
                pair_key = f"{zone_a}_{zone_b}"
                pairwise_regions[pair_key] = intersection

    if logger:
        logger.info(
            f"      📊 Found {len(pairwise_regions)} zone pairs with overlapping clouds"
        )

    # === STEP 3: IDENTIFY TRIPLE+ OVERLAPS ===
    triple_overlaps = _identify_multi_zone_overlaps(
        pairwise_regions=pairwise_regions,
        coverage_clouds=coverage_clouds,
        logger=logger,
    )

    # === STEP 4: UNION ALL PAIRWISE INTERSECTIONS ===
    if pairwise_regions:
        total_region = unary_union(list(pairwise_regions.values()))
    else:
        total_region = Polygon()  # Empty polygon

    # === STEP 5: COMPUTE STATISTICS ===
    stats = _compute_czrc_stats(
        zones_gdf=zones_gdf,
        coverage_clouds=coverage_clouds,
        pairwise_regions=pairwise_regions,
        triple_overlaps=triple_overlaps,
        total_region=total_region,
        logger=logger,
    )

    # === STEP 6: CONVERT TO WKT FOR SERIALIZATION ===
    coverage_clouds_wkt = {name: cloud.wkt for name, cloud in coverage_clouds.items()}
    pairwise_wkts = {key: geom.wkt for key, geom in pairwise_regions.items()}
    triple_overlaps_wkt = {key: geom.wkt for key, geom in triple_overlaps.items()}
    total_region_wkt = total_region.wkt if not total_region.is_empty else None

    # === STEP 7: BUILD ZONE SPACINGS DICT FOR ILP VISIBILITY ===
    zone_spacings = {
        row["zone_name"]: row["max_spacing_m"] for _, row in zones_gdf.iterrows()
    }

    # === STEP 8: BUILD RAW ZONE GEOMETRIES (for clipping Tier 2 test points) ===
    zone_geometries = {
        row["zone_name"]: row["geometry"] for _, row in zones_gdf.iterrows()
    }

    return {
        "total_region": total_region,
        "total_region_wkt": total_region_wkt,
        "coverage_clouds": coverage_clouds,
        "coverage_clouds_wkt": coverage_clouds_wkt,
        "pairwise_regions": pairwise_regions,
        "pairwise_wkts": pairwise_wkts,
        "triple_overlaps": triple_overlaps,
        "triple_overlaps_wkt": triple_overlaps_wkt,
        "zone_spacings": zone_spacings,
        "zone_geometries": zone_geometries,  # Raw zone geometries for clipping
        "stats": stats,
    }


def _identify_multi_zone_overlaps(
    pairwise_regions: Dict[str, BaseGeometry],
    coverage_clouds: Dict[str, BaseGeometry],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, BaseGeometry]:
    """
    Identify regions where 3+ zones overlap.

    These are the highest-value areas for optimization - a single borehole
    there could satisfy test points from 3+ zones.

    Args:
        pairwise_regions: Dict of pairwise intersections
        coverage_clouds: Dict of per-zone coverage clouds
        logger: Optional logger

    Returns:
        Dict mapping "ZoneA_ZoneB_ZoneC" -> triple-intersection geometry
    """
    triple_overlaps: Dict[str, BaseGeometry] = {}

    # Get all unique zones from pairwise keys
    zones_in_pairs = set()
    for pair_key in pairwise_regions.keys():
        zone_a, zone_b = pair_key.split("_", 1)
        zones_in_pairs.add(zone_a)
        zones_in_pairs.add(zone_b)

    zones_list = sorted(zones_in_pairs)
    n_zones = len(zones_list)

    if n_zones < 3:
        return triple_overlaps

    # Check all zone triplets
    for i in range(n_zones):
        for j in range(i + 1, n_zones):
            for k in range(j + 1, n_zones):
                zone_a = zones_list[i]
                zone_b = zones_list[j]
                zone_c = zones_list[k]

                # Check if all three pairwise intersections exist
                pair_ab = f"{zone_a}_{zone_b}"
                pair_ac = f"{zone_a}_{zone_c}"
                pair_bc = f"{zone_b}_{zone_c}"

                # Need all three pairs to have overlap
                if not all(p in pairwise_regions for p in [pair_ab, pair_ac, pair_bc]):
                    continue

                # Compute triple intersection
                try:
                    # Intersect all three clouds
                    triple = (
                        coverage_clouds[zone_a]
                        .intersection(coverage_clouds[zone_b])
                        .intersection(coverage_clouds[zone_c])
                    )

                    if not triple.is_empty and triple.area > 0:
                        triple_key = f"{zone_a}_{zone_b}_{zone_c}"
                        triple_overlaps[triple_key] = triple

                        if logger:
                            logger.info(
                                f"      🎯 Triple overlap: {triple_key} "
                                f"({triple.area:.0f} m²)"
                            )
                except Exception:
                    pass  # Skip on error

    return triple_overlaps


def _compute_czrc_stats(
    zones_gdf: GeoDataFrame,
    coverage_clouds: Dict[str, BaseGeometry],
    pairwise_regions: Dict[str, BaseGeometry],
    triple_overlaps: Dict[str, BaseGeometry],
    total_region: BaseGeometry,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Compute statistics about the CZRC consolidation region.

    Returns:
        Dict with various statistics for comparison with buffer strip approach
    """
    # Total zone area
    total_zone_area = sum(row["geometry"].area for _, row in zones_gdf.iterrows())

    # Coverage cloud areas
    total_cloud_area = sum(cloud.area for cloud in coverage_clouds.values())

    # CZRC region area
    czrc_area = total_region.area if not total_region.is_empty else 0

    # Pairwise intersection areas
    pairwise_total_area = sum(geom.area for geom in pairwise_regions.values())

    # Triple overlap area
    triple_total_area = sum(geom.area for geom in triple_overlaps.values())

    # Zone spacing statistics (needed for candidate grid generation)
    zone_spacings = [row["max_spacing_m"] for _, row in zones_gdf.iterrows()]
    min_zone_spacing = min(zone_spacings) if zone_spacings else 100.0
    max_zone_spacing = max(zone_spacings) if zone_spacings else 200.0

    stats = {
        "n_zones": len(zones_gdf),
        "total_zone_area_m2": total_zone_area,
        "total_cloud_area_m2": total_cloud_area,
        "n_pairwise_overlaps": len(pairwise_regions),
        "pairwise_overlap_area_m2": pairwise_total_area,
        "n_triple_overlaps": len(triple_overlaps),
        "triple_overlap_area_m2": triple_total_area,
        "czrc_total_area_m2": czrc_area,
        "min_zone_spacing": min_zone_spacing,
        "max_zone_spacing": max_zone_spacing,
    }

    if logger:
        logger.info("   📊 CZRC Statistics:")
        logger.info(f"      • Zones: {stats['n_zones']}")
        logger.info(f"      • Total zone area: {total_zone_area:,.0f} m²")
        logger.info(f"      • Coverage cloud area: {total_cloud_area:,.0f} m²")
        logger.info(
            f"      • Pairwise overlaps: {len(pairwise_regions)} pairs, "
            f"{pairwise_total_area:,.0f} m²"
        )
        logger.info(
            f"      • Triple overlaps: {len(triple_overlaps)} triplets, "
            f"{triple_total_area:,.0f} m²"
        )
        logger.info(f"      • CZRC total region: {czrc_area:,.0f} m²")

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# 🔧 UTILITY SECTION
# ═══════════════════════════════════════════════════════════════════════════


def load_czrc_region_from_wkt(wkt_string: str) -> Optional[BaseGeometry]:
    """
    Load a CZRC region from WKT string.

    Useful for deserializing cached or stored CZRC data.

    Args:
        wkt_string: WKT representation of geometry

    Returns:
        Geometry object, or None if parsing fails
    """
    if not wkt_string:
        return None
    try:
        return wkt.loads(wkt_string)
    except Exception:
        return None


def get_zone_pair_key(zone_a: str, zone_b: str) -> str:
    """
    Create a consistent key for a zone pair (alphabetically sorted).

    Ensures "ZoneA_ZoneB" and "ZoneB_ZoneA" produce the same key.
    """
    return "_".join(sorted([zone_a, zone_b]))
