#!/usr/bin/env python3
"""
Zone Coverage Visualization - Geometry Service

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Compute zone-clipped coverage polygons using Shapely.

Key Features:
1. Buffer creation for borehole positions
2. Zone intersection to create zone-clipped coverage
3. Multi-zone support with different max_spacing per zone
4. Coordinate transformation (BNG <-> WGS84)

Navigation Guide:
- CoverageService: Main service class
- compute_coverage: Core computation method

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, mapping
from shapely.ops import unary_union

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# Coordinate reference systems
CRS_WGS84 = "EPSG:4326"  # GPS coordinates (lon, lat)
CRS_BNG = "EPSG:27700"  # British National Grid (Easting, Northing in meters)

# Default buffer resolution (number of segments in circle)
BUFFER_RESOLUTION = 32

# Logging
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 📐 COVERAGE SERVICE
# ═══════════════════════════════════════════════════════════════════════════


class CoverageService:
    """
    Service for computing zone-clipped coverage polygons.

    Uses Shapely for geometry operations. All coordinates are handled in
    British National Grid (EPSG:27700) for accurate distance calculations,
    then converted to WGS84 for display on web maps.
    """

    def __init__(self, zones_gdf: gpd.GeoDataFrame) -> None:
        """
        Initialize with zone geometries.

        Args:
            zones_gdf: GeoDataFrame with zone polygons.
                       Must have 'max_spacing_m' column or property.
                       Expected CRS: EPSG:27700 or EPSG:4326.
        """
        # Ensure zones are in BNG for meter-based operations
        if zones_gdf.crs is None:
            logger.warning("Zones GDF has no CRS, assuming EPSG:27700")
            self.zones_gdf = zones_gdf.set_crs(CRS_BNG)
        elif zones_gdf.crs.to_string() != CRS_BNG:
            logger.info(f"Converting zones from {zones_gdf.crs} to {CRS_BNG}")
            self.zones_gdf = zones_gdf.to_crs(CRS_BNG)
        else:
            self.zones_gdf = zones_gdf

        # Pre-compute transformers for coordinate conversion
        self._wgs84_to_bng = Transformer.from_crs(CRS_WGS84, CRS_BNG, always_xy=True)
        self._bng_to_wgs84 = Transformer.from_crs(CRS_BNG, CRS_WGS84, always_xy=True)

        # Pre-compute zone bounds for fast filtering
        self._zone_bounds = self.zones_gdf.bounds

        logger.info(f"CoverageService initialized with {len(self.zones_gdf)} zones")

    def compute_coverage(
        self,
        lon: float,
        lat: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute zone-clipped coverage polygon for a borehole position.

        The coverage is the union of all zone-clipped buffers. Each zone
        clips the buffer to its boundary, using that zone's max_spacing_m
        as the buffer radius.

        Args:
            lon: Longitude in WGS84 (EPSG:4326)
            lat: Latitude in WGS84 (EPSG:4326)

        Returns:
            GeoJSON Feature with the coverage polygon in WGS84,
            or None if the point is outside all zones.
        """
        # Convert WGS84 to BNG
        x, y = self._wgs84_to_bng.transform(lon, lat)
        point = Point(x, y)

        # Collect coverage fragments from each zone
        fragments: List[Any] = []
        zone_names: List[str] = []

        for idx, zone in self.zones_gdf.iterrows():
            # Get zone geometry
            zone_geom = zone.geometry
            if zone_geom is None or zone_geom.is_empty:
                continue

            # Quick bounds check for performance
            if not zone_geom.envelope.contains(point) and not zone_geom.contains(point):
                # Check if point could be within max_spacing of zone
                max_spacing = self._get_zone_spacing(zone)
                if point.distance(zone_geom) > max_spacing:
                    continue

            # Get max_spacing for this zone
            max_spacing_m = self._get_zone_spacing(zone)

            # Create buffer around point
            buffer = point.buffer(max_spacing_m, resolution=BUFFER_RESOLUTION)

            # Intersect with zone boundary
            try:
                intersection = buffer.intersection(zone_geom)
            except Exception as e:
                logger.warning(f"Intersection failed for zone {idx}: {e}")
                continue

            if not intersection.is_empty:
                fragments.append(intersection)
                zone_name = self._get_zone_name(zone, idx)
                zone_names.append(zone_name)

        if not fragments:
            return None

        # Merge all fragments into single coverage polygon
        coverage = unary_union(fragments)

        # Convert back to WGS84 for display
        coverage_wgs84 = self._transform_to_wgs84(coverage)

        # Build GeoJSON Feature
        return {
            "type": "Feature",
            "geometry": mapping(coverage_wgs84),
            "properties": {
                "zones": zone_names,
                "num_zones": len(zone_names),
            },
        }

    def get_zone_info(
        self,
        lon: float,
        lat: float,
    ) -> Dict[str, float]:
        """
        Get zone information for a point.

        Args:
            lon: Longitude in WGS84
            lat: Latitude in WGS84

        Returns:
            Dict mapping zone_name -> max_spacing_m for zones containing the point.
        """
        # Convert WGS84 to BNG
        x, y = self._wgs84_to_bng.transform(lon, lat)
        point = Point(x, y)

        zone_info = {}

        for idx, zone in self.zones_gdf.iterrows():
            zone_geom = zone.geometry
            if zone_geom is None or zone_geom.is_empty:
                continue

            if zone_geom.contains(point):
                zone_name = self._get_zone_name(zone, idx)
                max_spacing = self._get_zone_spacing(zone)
                zone_info[zone_name] = max_spacing

        return zone_info

    def _get_zone_spacing(self, zone: gpd.GeoSeries) -> float:
        """
        Get max_spacing_m for a zone.

        Resolution order:
        1. 'max_spacing_m' column in GeoDataFrame
        2. 'max_spacing' column (legacy)
        3. Default: 100.0 meters

        Args:
            zone: Row from zones GeoDataFrame

        Returns:
            max_spacing_m value in meters.
        """
        # Try different possible column names
        for col in ["max_spacing_m", "max_spacing", "spacing"]:
            if col in zone.index and zone[col] is not None:
                try:
                    return float(zone[col])
                except (ValueError, TypeError):
                    continue

        return 100.0  # Default

    def _get_zone_name(self, zone: gpd.GeoSeries, idx: int) -> str:
        """
        Get display name for a zone.

        Args:
            zone: Row from zones GeoDataFrame
            idx: Index of the zone

        Returns:
            Zone name string.
        """
        for col in ["Name", "name", "zone_name", "display_name"]:
            if col in zone.index and zone[col] is not None:
                return str(zone[col])

        return f"Zone_{idx}"

    def _transform_to_wgs84(self, geom: Any) -> Any:
        """
        Transform a geometry from BNG to WGS84.

        Args:
            geom: Shapely geometry in EPSG:27700

        Returns:
            Shapely geometry in EPSG:4326
        """
        from shapely.ops import transform

        def transform_coords(x: float, y: float) -> Tuple[float, float]:
            return self._bng_to_wgs84.transform(x, y)

        return transform(transform_coords, geom)

    def compute_coverage_stats(
        self,
        borehole_positions: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """
        Compute coverage statistics for all zones given current borehole positions.

        Args:
            borehole_positions: List of (lon, lat) tuples in WGS84

        Returns:
            Dict with:
            - per_zone: List of {zone_name, total_area_m2, covered_area_m2, coverage_pct}
            - total: {total_area_m2, covered_area_m2, coverage_pct}
        """
        # Build all coverage circles in BNG
        all_coverages: List[Any] = []

        for lon, lat in borehole_positions:
            x, y = self._wgs84_to_bng.transform(lon, lat)
            point = Point(x, y)

            # Collect fragments from each zone this borehole could cover
            for idx, zone in self.zones_gdf.iterrows():
                zone_geom = zone.geometry
                if zone_geom is None or zone_geom.is_empty:
                    continue

                max_spacing_m = self._get_zone_spacing(zone)

                # Check if point could cover any part of this zone
                if point.distance(zone_geom) > max_spacing_m:
                    continue

                # Create buffer and clip to zone
                buffer = point.buffer(max_spacing_m, resolution=BUFFER_RESOLUTION)
                try:
                    intersection = buffer.intersection(zone_geom)
                    if not intersection.is_empty:
                        all_coverages.append(intersection)
                except Exception:
                    continue

        # Merge all coverage fragments
        total_coverage = unary_union(all_coverages) if all_coverages else None

        # Compute per-zone stats
        per_zone_stats = []
        total_zone_area = 0.0
        total_covered_area = 0.0

        for idx, zone in self.zones_gdf.iterrows():
            zone_geom = zone.geometry
            if zone_geom is None or zone_geom.is_empty:
                continue

            zone_name = self._get_zone_name(zone, idx)
            zone_area = zone_geom.area  # In square meters (BNG)

            if total_coverage is not None:
                try:
                    covered_in_zone = total_coverage.intersection(zone_geom)
                    covered_area = covered_in_zone.area if not covered_in_zone.is_empty else 0.0
                except Exception:
                    covered_area = 0.0
            else:
                covered_area = 0.0

            coverage_pct = (covered_area / zone_area * 100.0) if zone_area > 0 else 0.0

            per_zone_stats.append({
                "zone_name": zone_name,
                "total_area_m2": round(zone_area, 1),
                "covered_area_m2": round(covered_area, 1),
                "coverage_pct": round(coverage_pct, 1),
            })

            total_zone_area += zone_area
            total_covered_area += covered_area

        # Compute overall stats
        overall_pct = (total_covered_area / total_zone_area * 100.0) if total_zone_area > 0 else 0.0

        return {
            "per_zone": per_zone_stats,
            "total": {
                "total_area_m2": round(total_zone_area, 1),
                "covered_area_m2": round(total_covered_area, 1),
                "coverage_pct": round(overall_pct, 1),
            },
        }
