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

from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union


# ═══════════════════════════════════════════════════════════════════════════
# 📦 CACHE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CachedCoverage:
    """Cached coverage data for a single borehole."""

    borehole_id: str
    lon: float
    lat: float
    coverage_bng: Any  # Shapely geometry in BNG coordinates
    zone_names: List[str]


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

        # Warm up transformers (first call is slow due to proj4 grid loading)
        import time

        t1 = time.perf_counter()
        _ = self._wgs84_to_bng.transform(-1.22, 51.55)
        t2 = time.perf_counter()
        _ = self._bng_to_wgs84.transform(450000, 180000)
        t3 = time.perf_counter()
        logger.info(
            f"Transformer warmup: WGS84→BNG {(t2-t1)*1000:.1f}ms, BNG→WGS84 {(t3-t2)*1000:.1f}ms"
        )

        # Pre-compute zone bounds for fast filtering
        self._zone_bounds = self.zones_gdf.bounds

        # ═══════════════════════════════════════════════════════════════════
        # 🗄️ COVERAGE CACHE - Delta updates optimization
        # ═══════════════════════════════════════════════════════════════════

        # Cache: borehole_id -> CachedCoverage
        self._coverage_cache: Dict[str, CachedCoverage] = {}

        # Precompute zone areas (never change) for fast stats calculation
        self._zone_areas: Dict[str, float] = self._precompute_zone_areas()

        # Stats cache - invalidated when any coverage changes
        self._cached_stats: Optional[Dict[str, Any]] = None
        self._stats_dirty: bool = True

        logger.info(f"CoverageService initialized with {len(self.zones_gdf)} zones")

    def compute_coverage(
        self,
        lon: float,
        lat: float,
        exclude_zones: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute zone-clipped coverage polygon for a borehole position.

        The coverage is the union of all zone-clipped buffers. Each zone
        clips the buffer to its boundary, using that zone's max_spacing_m
        as the buffer radius.

        Args:
            lon: Longitude in WGS84 (EPSG:4326)
            lat: Latitude in WGS84 (EPSG:4326)
            exclude_zones: Optional list of zone names to exclude from coverage

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

        # Normalize exclude_zones to a set for O(1) lookup
        excluded_set = set(exclude_zones) if exclude_zones else set()

        for idx, zone in self.zones_gdf.iterrows():
            # Get zone geometry
            zone_geom = zone.geometry
            if zone_geom is None or zone_geom.is_empty:
                continue

            # Get zone name early for exclusion check
            zone_name = self._get_zone_name(zone, idx)

            # Skip excluded zones
            if zone_name in excluded_set:
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

    def _get_zone_layer_key(self, zone: gpd.GeoSeries) -> str:
        """
        Get layer key for a zone (e.g., 'embankment_zones', 'highways').

        Args:
            zone: Row from zones GeoDataFrame

        Returns:
            Layer key string.
        """
        if "layer_key" in zone.index and zone["layer_key"] is not None:
            return str(zone["layer_key"])
        return "unknown"

    def _get_zone_display_name(self, zone: gpd.GeoSeries, idx: int) -> str:
        """
        Get display name for a zone's layer (e.g., 'Embankment', 'Highways').

        Args:
            zone: Row from zones GeoDataFrame
            idx: Index of the zone

        Returns:
            Display name string for the layer.
        """
        # Priority order for display name column
        for col in ["display_name", "original_name"]:
            if col in zone.index and zone[col] is not None:
                return str(zone[col])

        # Fallback: derive from zone_name by removing trailing _N index
        zone_name = self._get_zone_name(zone, idx)
        import re

        match = re.match(r"^(.+?)_\d+$", zone_name)
        if match:
            return match.group(1)

        # Last resort: derive from layer_key
        layer_key = self._get_zone_layer_key(zone)
        return layer_key.replace("_zones", "").replace("_", " ").title()

    def transform_wgs84_to_bng(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        Transform WGS84 coordinates to BNG (accurate).

        Args:
            lon: Longitude in WGS84
            lat: Latitude in WGS84

        Returns:
            Tuple of (x, y) in BNG (EPSG:27700)
        """
        return self._wgs84_to_bng.transform(lon, lat)

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

    # ═══════════════════════════════════════════════════════════════════════
    # 🗄️ CACHE METHODS - Delta updates optimization
    # ═══════════════════════════════════════════════════════════════════════

    def _precompute_zone_areas(self) -> Dict[str, float]:
        """Precompute zone areas for fast stats calculation."""
        areas = {}
        for idx, zone in self.zones_gdf.iterrows():
            zone_geom = zone.geometry
            if zone_geom is None or zone_geom.is_empty:
                continue
            zone_name = self._get_zone_name(zone, idx)
            areas[zone_name] = zone_geom.area
        return areas

    def _compute_coverage_bng(
        self, lon: float, lat: float, bng_coords: Optional[Tuple[float, float]] = None
    ) -> Tuple[Any, List[str]]:
        """
        Compute zone-clipped coverage in BNG coordinates (internal).

        Args:
            lon: Longitude in WGS84 (used if bng_coords not provided)
            lat: Latitude in WGS84 (used if bng_coords not provided)
            bng_coords: Pre-computed (x, y) in BNG to avoid duplicate transform

        Returns:
            Tuple of (coverage_geometry_bng, zone_names_list)
        """
        import time

        t_start = time.perf_counter()

        # Use pre-computed BNG coords if provided, else transform
        if bng_coords:
            x, y = bng_coords
            print(f"    ⏱️ Using pre-computed BNG coords")
        else:
            x, y = self._wgs84_to_bng.transform(lon, lat)
            t1 = time.perf_counter()
            print(f"    ⏱️ CRS transform + Point: {(t1-t_start)*1000:.1f}ms")

        point = Point(x, y)

        fragments: List[Any] = []
        zone_names: List[str] = []

        t_loop_start = time.perf_counter()
        zones_checked = 0
        zones_intersected = 0

        for idx, zone in self.zones_gdf.iterrows():
            zone_geom = zone.geometry
            if zone_geom is None or zone_geom.is_empty:
                continue
            zones_checked += 1

            # Quick bounds check
            if not zone_geom.envelope.contains(point) and not zone_geom.contains(point):
                max_spacing = self._get_zone_spacing(zone)
                if point.distance(zone_geom) > max_spacing:
                    continue

            max_spacing_m = self._get_zone_spacing(zone)
            buffer = point.buffer(max_spacing_m, resolution=BUFFER_RESOLUTION)

            try:
                intersection = buffer.intersection(zone_geom)
            except Exception as e:
                logger.warning(f"Intersection failed for zone {idx}: {e}")
                continue

            if not intersection.is_empty:
                zones_intersected += 1
                fragments.append(intersection)
                zone_name = self._get_zone_name(zone, idx)
                zone_names.append(zone_name)

        t_loop_end = time.perf_counter()
        print(
            f"    ⏱️ Zone loop ({zones_checked} checked, {zones_intersected} hit): {(t_loop_end-t_loop_start)*1000:.1f}ms"
        )

        if not fragments:
            return None, []

        t_union_start = time.perf_counter()
        coverage_bng = unary_union(fragments)
        t_union_end = time.perf_counter()
        print(
            f"    ⏱️ unary_union ({len(fragments)} fragments): {(t_union_end-t_union_start)*1000:.1f}ms"
        )

        return coverage_bng, zone_names

    def compute_coverage_cached(
        self,
        borehole_id: str,
        lon: float,
        lat: float,
        bng_coords: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute coverage with caching - returns cached if position unchanged.

        Args:
            borehole_id: Unique identifier for the borehole
            lon: Longitude in WGS84
            lat: Latitude in WGS84
            bng_coords: Pre-computed (x, y) in BNG to avoid duplicate transform

        Returns:
            GeoJSON Feature with coverage polygon, or None if outside zones.
        """
        # Check cache
        if borehole_id in self._coverage_cache:
            cached = self._coverage_cache[borehole_id]
            if cached.lon == lon and cached.lat == lat:
                # Position unchanged - return from cache
                if cached.coverage_bng is None:
                    return None
                coverage_wgs84 = self._transform_to_wgs84(cached.coverage_bng)
                return {
                    "type": "Feature",
                    "geometry": mapping(coverage_wgs84),
                    "properties": {
                        "borehole_id": borehole_id,
                        "zones": cached.zone_names,
                        "num_zones": len(cached.zone_names),
                    },
                }

        # Compute new coverage (pass through BNG coords if provided)
        coverage_bng, zone_names = self._compute_coverage_bng(lon, lat, bng_coords)

        # Store in cache
        self._coverage_cache[borehole_id] = CachedCoverage(
            borehole_id=borehole_id,
            lon=lon,
            lat=lat,
            coverage_bng=coverage_bng,
            zone_names=zone_names,
        )

        # Mark stats as dirty (need recomputation)
        self._stats_dirty = True

        if coverage_bng is None:
            return None

        import time

        # Simplify the coverage polygon to reduce vertices (faster transform)
        # Tolerance of 1m is invisible at web map zoom levels
        t0 = time.perf_counter()
        simplified_bng = coverage_bng.simplify(tolerance=1.0, preserve_topology=True)
        t1 = time.perf_counter()
        print(
            f"    ⏱️ Simplify ({len(coverage_bng.exterior.coords) if hasattr(coverage_bng, 'exterior') else '?'} → {len(simplified_bng.exterior.coords) if hasattr(simplified_bng, 'exterior') else '?'} pts): {(t1-t0)*1000:.1f}ms"
        )

        coverage_wgs84 = self._transform_to_wgs84(simplified_bng)
        t2 = time.perf_counter()
        print(f"    ⏱️ BNG→WGS84 transform: {(t2-t1)*1000:.1f}ms")

        return {
            "type": "Feature",
            "geometry": mapping(coverage_wgs84),
            "properties": {
                "borehole_id": borehole_id,
                "zones": zone_names,
                "num_zones": len(zone_names),
            },
        }

    def invalidate_cache(self, borehole_id: str) -> None:
        """Remove a borehole from cache (on delete)."""
        self._coverage_cache.pop(borehole_id, None)
        self._stats_dirty = True
        logger.debug(f"Cache invalidated for borehole {borehole_id}")

    def clear_cache(self) -> None:
        """Clear all cached coverages."""
        self._coverage_cache.clear()
        self._stats_dirty = True
        self._cached_stats = None
        logger.info("Coverage cache cleared")

    def get_stats_from_cache(self) -> Dict[str, Any]:
        """
        Get coverage stats. Returns cached stats if available, otherwise computes.

        Returns:
            Stats dict with per_zone and total coverage percentages.
        """
        # Return cached stats if still valid
        if not self._stats_dirty and self._cached_stats is not None:
            logger.debug("Stats: Returning cached (0ms)")
            return self._cached_stats

        # Collect all cached BNG geometries
        cached_geoms = [
            c.coverage_bng
            for c in self._coverage_cache.values()
            if c.coverage_bng is not None
        ]

        total_coverage = unary_union(cached_geoms) if cached_geoms else None

        # Compute per-zone stats using precomputed areas
        per_zone_stats = []
        total_zone_area = 0.0
        total_covered_area = 0.0

        for idx, zone in self.zones_gdf.iterrows():
            zone_geom = zone.geometry
            if zone_geom is None or zone_geom.is_empty:
                continue

            zone_name = self._get_zone_name(zone, idx)
            zone_area = self._zone_areas.get(zone_name, zone_geom.area)

            if total_coverage is not None:
                try:
                    covered_in_zone = total_coverage.intersection(zone_geom)
                    covered_area = (
                        covered_in_zone.area if not covered_in_zone.is_empty else 0.0
                    )
                except Exception:
                    covered_area = 0.0
            else:
                covered_area = 0.0

            coverage_pct = (covered_area / zone_area * 100.0) if zone_area > 0 else 0.0

            # Get layer_key for grouping in UI
            layer_key = self._get_zone_layer_key(zone)
            layer_display_name = self._get_zone_display_name(zone, idx)

            per_zone_stats.append(
                {
                    "zone_name": zone_name,
                    "layer_key": layer_key,
                    "layer_display_name": layer_display_name,
                    "total_area_m2": round(zone_area, 1),
                    "covered_area_m2": round(covered_area, 1),
                    "coverage_pct": round(coverage_pct, 1),
                }
            )

            total_zone_area += zone_area
            total_covered_area += covered_area

        overall_pct = (
            (total_covered_area / total_zone_area * 100.0)
            if total_zone_area > 0
            else 0.0
        )

        logger.debug(
            f"Stats: Computed for {len(cached_geoms)} coverages, {len(per_zone_stats)} zones"
        )

        # Cache the computed stats and mark as clean
        self._cached_stats = {
            "per_zone": per_zone_stats,
            "total": {
                "total_area_m2": round(total_zone_area, 1),
                "covered_area_m2": round(total_covered_area, 1),
                "coverage_pct": round(overall_pct, 1),
            },
        }
        self._stats_dirty = False

        return self._cached_stats

    def compute_all_coverages_and_cache(
        self,
        boreholes_geojson: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute all coverages and populate cache. For initial load.

        Args:
            boreholes_geojson: GeoJSON FeatureCollection of boreholes

        Returns:
            GeoJSON FeatureCollection with all coverage polygons.
        """
        coverages = []

        for i, feature in enumerate(boreholes_geojson.get("features", [])):
            coords = feature.get("geometry", {}).get("coordinates", [])
            if len(coords) < 2:
                continue

            lon, lat = coords[0], coords[1]

            # Get borehole ID from properties, or generate from index
            props = feature.get("properties", {})
            borehole_id = props.get("id") or props.get("location_id") or f"BH_{i}"

            # Use cached computation
            coverage = self.compute_coverage_cached(borehole_id, lon, lat)

            if coverage:
                coverage["properties"]["borehole_index"] = i
                coverages.append(coverage)

        logger.info(f"Computed and cached {len(coverages)} coverages")
        return {"type": "FeatureCollection", "features": coverages}

    def compute_all_coverages_filtered(
        self,
        boreholes_geojson: Dict[str, Any],
        exclude_zones: List[str],
    ) -> Dict[str, Any]:
        """
        Compute all coverages excluding specified zones (for zone visibility filtering).

        This bypasses the cache since results depend on which zones are visible.
        Used when user hides zones via checkboxes.

        Args:
            boreholes_geojson: GeoJSON FeatureCollection of boreholes
            exclude_zones: List of zone names to exclude from coverage computation

        Returns:
            GeoJSON FeatureCollection with filtered coverage polygons.
        """
        coverages = []

        for i, feature in enumerate(boreholes_geojson.get("features", [])):
            coords = feature.get("geometry", {}).get("coordinates", [])
            if len(coords) < 2:
                continue

            lon, lat = coords[0], coords[1]

            # Compute coverage excluding specified zones (no caching)
            coverage = self.compute_coverage(lon, lat, exclude_zones=exclude_zones)

            if coverage:
                coverage["properties"]["borehole_index"] = i
                coverages.append(coverage)

        logger.info(
            f"Computed {len(coverages)} filtered coverages (excluding {exclude_zones})"
        )
        return {"type": "FeatureCollection", "features": coverages}

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
                    covered_area = (
                        covered_in_zone.area if not covered_in_zone.is_empty else 0.0
                    )
                except Exception:
                    covered_area = 0.0
            else:
                covered_area = 0.0

            coverage_pct = (covered_area / zone_area * 100.0) if zone_area > 0 else 0.0

            # Get layer_key for grouping in UI
            layer_key = self._get_zone_layer_key(zone)
            layer_display_name = self._get_zone_display_name(zone, idx)

            per_zone_stats.append(
                {
                    "zone_name": zone_name,
                    "layer_key": layer_key,
                    "layer_display_name": layer_display_name,
                    "total_area_m2": round(zone_area, 1),
                    "covered_area_m2": round(covered_area, 1),
                    "coverage_pct": round(coverage_pct, 1),
                }
            )

            total_zone_area += zone_area
            total_covered_area += covered_area

        # Compute overall stats
        overall_pct = (
            (total_covered_area / total_zone_area * 100.0)
            if total_zone_area > 0
            else 0.0
        )

        return {
            "per_zone": per_zone_stats,
            "total": {
                "total_area_m2": round(total_zone_area, 1),
                "covered_area_m2": round(total_covered_area, 1),
                "coverage_pct": round(overall_pct, 1),
            },
        }
