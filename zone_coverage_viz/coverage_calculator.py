"""
Coverage geometry calculator using Shapely.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURAL OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

Responsibility: Pre-compute initial zone-aware coverage geometries using Shapely.
This enables fast initial rendering before the Web Worker takes over for dynamic
updates during drag operations.

Key Features:
- Zone-aware coverage computation (different radii per zone)
- Multi-zone "flower petal" coverage shapes at boundaries
- GeoJSON output for deck.gl polygon layers

For Navigation: Use VS Code outline (Ctrl+Shift+O)

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, mapping
from shapely.ops import unary_union
from shapely import buffer

from .zone_geometry import ZoneData, BoreholeData, _ensure_wgs84

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 📐 COORDINATE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def _meters_to_degrees(meters: float, latitude: float) -> float:
    """
    Convert meters to approximate degrees at given latitude.

    Uses simple spherical Earth approximation. For UK latitudes (~51-55°N),
    this is accurate to within ~1% for typical coverage radii (50-200m).

    Args:
        meters: Distance in meters
        latitude: Latitude in degrees

    Returns:
        Approximate distance in degrees
    """
    # Earth radius in meters
    earth_radius = 6371000.0

    # At equator, 1 degree ≈ 111km
    # At latitude θ, 1 degree longitude ≈ 111km × cos(θ)
    lat_rad = np.radians(latitude)

    # Average of lat and lon degree lengths
    deg_lat = meters / (earth_radius * np.pi / 180.0)
    deg_lon = meters / (earth_radius * np.cos(lat_rad) * np.pi / 180.0)

    # Use average for circular buffer (approximation)
    return (deg_lat + deg_lon) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# 🔵 SINGLE BOREHOLE COVERAGE
# ═══════════════════════════════════════════════════════════════════════════════


def compute_borehole_coverage(
    borehole_point: Point,
    zones_gdf: gpd.GeoDataFrame,
    max_spacing_field: str = "max_spacing_m",
    default_max_spacing: float = 100.0,
) -> Optional[Dict[str, Any]]:
    """
    Compute zone-aware coverage for a single borehole.

    The coverage geometry is the union of circle-zone intersections,
    where each circle has the zone's specific max_spacing radius.
    This produces "flower petal" shapes at zone boundaries.

    Args:
        borehole_point: Shapely Point in WGS84 coordinates
        zones_gdf: GeoDataFrame with zone polygons (WGS84)
        max_spacing_field: Column containing max spacing values
        default_max_spacing: Default if field not found

    Returns:
        GeoJSON geometry dict, or None if no coverage
    """
    from shapely.validation import make_valid

    # === FIND INTERSECTING ZONES ===
    coverage_parts = []

    for idx, zone_row in zones_gdf.iterrows():
        zone_geom = zone_row.geometry

        # Skip empty or invalid geometries
        if zone_geom is None or zone_geom.is_empty:
            continue

        # Fix invalid geometries
        if not zone_geom.is_valid:
            try:
                zone_geom = make_valid(zone_geom)
            except Exception:
                continue

        # Get zone's max spacing (in meters)
        if max_spacing_field in zone_row.index:
            max_spacing_m = float(zone_row[max_spacing_field])
        else:
            max_spacing_m = default_max_spacing

        # Convert meters to degrees at borehole latitude
        radius_deg = _meters_to_degrees(max_spacing_m, borehole_point.y)

        # Create coverage circle
        coverage_circle = borehole_point.buffer(radius_deg)

        # Intersect with zone (with error handling)
        try:
            intersection = coverage_circle.intersection(zone_geom)

            if not intersection.is_empty:
                coverage_parts.append(intersection)
        except Exception as e:
            # Skip zones that cause topology errors
            logger.debug(f"Intersection failed for zone {idx}: {e}")
            continue

    # === UNION ALL PARTS ===
    if not coverage_parts:
        return None

    try:
        coverage = unary_union(coverage_parts)
    except Exception as e:
        # If union fails, try to return first valid part
        logger.debug(f"Union failed: {e}")
        coverage = coverage_parts[0] if coverage_parts else None

    if coverage is None or coverage.is_empty:
        return None

    return mapping(coverage)


# ═══════════════════════════════════════════════════════════════════════════════
# 🔷 BATCH COVERAGE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════


def compute_all_coverage(
    boreholes_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    max_spacing_field: str = "max_spacing_m",
    default_max_spacing: float = 100.0,
    id_field: str = "LocationID",
) -> Dict[str, Dict[str, Any]]:
    """
    Pre-compute zone-aware coverage for all boreholes.

    Computes coverage geometries for each borehole, considering zone-specific
    max_spacing values. Used for initial rendering before Web Worker takes over.

    Args:
        boreholes_gdf: GeoDataFrame with borehole points
        zones_gdf: GeoDataFrame with zone polygons
        max_spacing_field: Column containing max spacing values
        default_max_spacing: Default if field not found
        id_field: Column containing borehole IDs

    Returns:
        Dict mapping borehole ID to GeoJSON geometry dict
    """
    logger.info(f"📐 Pre-computing coverage for {len(boreholes_gdf)} boreholes")

    # === ENSURE WGS84 ===
    boreholes_wgs84 = _ensure_wgs84(boreholes_gdf.copy())
    zones_wgs84 = _ensure_wgs84(zones_gdf.copy())

    # === COMPUTE COVERAGE FOR EACH BOREHOLE ===
    coverage_dict: Dict[str, Dict[str, Any]] = {}
    computed = 0
    skipped = 0

    for idx, bh_row in boreholes_wgs84.iterrows():
        # Get borehole ID
        if id_field in bh_row.index and bh_row[id_field] is not None:
            bh_id = str(bh_row[id_field])
        else:
            bh_id = f"BH_{idx}"

        # Compute coverage
        coverage_geom = compute_borehole_coverage(
            borehole_point=bh_row.geometry,
            zones_gdf=zones_wgs84,
            max_spacing_field=max_spacing_field,
            default_max_spacing=default_max_spacing,
        )

        if coverage_geom is not None:
            coverage_dict[bh_id] = coverage_geom
            computed += 1
        else:
            skipped += 1

    logger.info(f"   ✅ Computed {computed} coverage geometries ({skipped} skipped)")

    return coverage_dict


# ═══════════════════════════════════════════════════════════════════════════════
# 📤 GEOJSON CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════


def coverage_to_geojson_fc(
    coverage_dict: Dict[str, Dict[str, Any]],
    boreholes_gdf: gpd.GeoDataFrame,
    id_field: str = "LocationID",
) -> Dict[str, Any]:
    """
    Convert coverage dict to GeoJSON FeatureCollection.

    Creates a FeatureCollection with one feature per borehole's coverage
    geometry, including the borehole ID for linking.

    Args:
        coverage_dict: Dict mapping borehole ID to geometry
        boreholes_gdf: Original borehole GeoDataFrame (for properties)
        id_field: Column containing borehole IDs

    Returns:
        GeoJSON FeatureCollection dict
    """
    features = []

    for bh_id, geometry in coverage_dict.items():
        feature = {
            "type": "Feature",
            "id": bh_id,
            "properties": {
                "borehole_id": bh_id,
            },
            "geometry": geometry,
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def coverage_to_json_string(
    coverage_dict: Dict[str, Dict[str, Any]],
    boreholes_gdf: gpd.GeoDataFrame,
    id_field: str = "LocationID",
) -> str:
    """
    Convert coverage dict to compact JSON string.

    Args:
        coverage_dict: Dict mapping borehole ID to geometry
        boreholes_gdf: Original borehole GeoDataFrame
        id_field: Column containing borehole IDs

    Returns:
        Compact JSON string
    """
    fc = coverage_to_geojson_fc(coverage_dict, boreholes_gdf, id_field)
    return json.dumps(fc, separators=(",", ":"))


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 COVERAGE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════


def compute_coverage_stats(
    coverage_dict: Dict[str, Dict[str, Any]],
    zones_gdf: gpd.GeoDataFrame,
) -> Dict[str, Any]:
    """
    Compute coverage statistics for zones.

    Calculates total zone area vs covered area to provide
    coverage percentage statistics.

    Args:
        coverage_dict: Dict mapping borehole ID to geometry
        zones_gdf: GeoDataFrame with zone polygons (WGS84)

    Returns:
        Dict with coverage statistics
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    zones_wgs84 = _ensure_wgs84(zones_gdf.copy())

    # Union all zones
    total_zone_area = unary_union([row.geometry for _, row in zones_wgs84.iterrows()])

    # Union all coverage
    coverage_shapes = [
        shape(geom) for geom in coverage_dict.values() if geom is not None
    ]

    if not coverage_shapes:
        return {
            "total_zone_area_deg2": total_zone_area.area,
            "covered_area_deg2": 0.0,
            "coverage_pct": 0.0,
            "borehole_count": 0,
        }

    total_coverage = unary_union(coverage_shapes)

    # Intersect coverage with zones (coverage outside zones doesn't count)
    effective_coverage = total_coverage.intersection(total_zone_area)

    coverage_pct = (
        effective_coverage.area / total_zone_area.area * 100.0
        if total_zone_area.area > 0
        else 0.0
    )

    return {
        "total_zone_area_deg2": total_zone_area.area,
        "covered_area_deg2": effective_coverage.area,
        "coverage_pct": round(coverage_pct, 2),
        "borehole_count": len(coverage_dict),
    }
