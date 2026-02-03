"""
Zone geometry preparation for deck.gl visualization.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURAL OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

Responsibility: Convert GeoDataFrame zones to GeoJSON format suitable for
deck.gl rendering.

Key Features:
- Ensures each zone has max_spacing property
- Transforms coordinates to WGS84 (EPSG:4326) for web mapping
- Computes bounding box for initial map view

For Navigation: Use VS Code outline (Ctrl+Shift+O)

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import mapping

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 📦 DATA CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ZoneData:
    """Container for processed zone data.

    Attributes:
        geojson: GeoJSON FeatureCollection dictionary
        zone_count: Number of zones processed
        bounds: (minx, miny, maxx, maxy) in WGS84
        center: (lon, lat) center point for map initialization
    """

    geojson: Dict[str, Any]
    zone_count: int
    bounds: Tuple[float, float, float, float]
    center: Tuple[float, float]


@dataclass
class BoreholeData:
    """Container for processed borehole data.

    Attributes:
        geojson: GeoJSON FeatureCollection dictionary
        borehole_count: Number of boreholes processed
        positions: List of [lon, lat] positions for deck.gl
    """

    geojson: Dict[str, Any]
    borehole_count: int
    positions: List[List[float]]


# ═══════════════════════════════════════════════════════════════════════════════
# 🗺️ COORDINATE TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════════


def _ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame is in WGS84 (EPSG:4326) for web mapping.

    Args:
        gdf: Input GeoDataFrame in any CRS

    Returns:
        GeoDataFrame reprojected to WGS84
    """
    if gdf.crs is None:
        logger.warning(
            "⚠️ GeoDataFrame has no CRS, assuming EPSG:27700 (British National Grid)"
        )
        gdf = gdf.set_crs("EPSG:27700")

    if gdf.crs.to_epsg() != 4326:
        logger.info(f"🔄 Reprojecting from {gdf.crs} to WGS84 (EPSG:4326)")
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def _compute_bounds(gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """Compute bounding box from GeoDataFrame.

    Args:
        gdf: GeoDataFrame (should be in WGS84)

    Returns:
        (minx, miny, maxx, maxy) bounds tuple
    """
    total_bounds = gdf.total_bounds
    return (
        float(total_bounds[0]),
        float(total_bounds[1]),
        float(total_bounds[2]),
        float(total_bounds[3]),
    )


def _compute_center(bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Compute center point from bounds.

    Args:
        bounds: (minx, miny, maxx, maxy) bounds tuple

    Returns:
        (lon, lat) center point
    """
    return (
        (bounds[0] + bounds[2]) / 2.0,
        (bounds[1] + bounds[3]) / 2.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 🔷 ZONE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_zone_geojson(
    zones_gdf: gpd.GeoDataFrame,
    max_spacing_field: str = "max_spacing_m",
    default_max_spacing: float = 100.0,
) -> ZoneData:
    """
    Prepare zone GeoJSON with max_spacing properties for deck.gl.

    Converts input GeoDataFrame to GeoJSON format, ensuring:
    - Coordinates are in WGS84 for web mapping
    - Each zone feature has max_spacing_m property
    - Zone names/IDs are preserved

    Args:
        zones_gdf: GeoDataFrame with zone polygons
        max_spacing_field: Column name containing max spacing values
        default_max_spacing: Default spacing if field missing

    Returns:
        ZoneData with processed GeoJSON and metadata
    """
    logger.info(f"📍 Preparing {len(zones_gdf)} zones for visualization")

    # === REPROJECT TO WGS84 ===
    zones_wgs84 = _ensure_wgs84(zones_gdf.copy())

    # === BUILD GEOJSON FEATURES ===
    features = []
    for idx, row in zones_wgs84.iterrows():
        # Get max spacing value
        if max_spacing_field in row.index and not np.isnan(row[max_spacing_field]):
            max_spacing = float(row[max_spacing_field])
        else:
            max_spacing = default_max_spacing

        # Get zone identifier
        zone_id = None
        for id_col in ["zone_id", "zone_name", "Name", "name", "ID", "id"]:
            if id_col in row.index and row[id_col] is not None:
                zone_id = str(row[id_col])
                break
        if zone_id is None:
            zone_id = f"zone_{idx}"

        # Build feature
        feature = {
            "type": "Feature",
            "id": zone_id,
            "properties": {
                "zone_id": zone_id,
                "max_spacing_m": max_spacing,
            },
            "geometry": mapping(row.geometry),
        }

        # Copy additional properties (excluding geometry)
        for col in row.index:
            if col != "geometry" and col not in feature["properties"]:
                val = row[col]
                # Convert numpy types to Python types for JSON
                if hasattr(val, "item"):
                    val = val.item()
                # Skip geometry-like objects and None/NaN values
                if val is None:
                    continue
                if isinstance(val, float) and np.isnan(val):
                    continue
                # Skip Shapely geometry objects (e.g., effective_geometry, original_geometry)
                if hasattr(val, "__geo_interface__"):
                    continue
                # Skip lists/sets that might contain non-serializable objects
                if isinstance(val, (list, set, frozenset)):
                    try:
                        json.dumps(val)
                    except (TypeError, ValueError):
                        continue
                feature["properties"][col] = val

        features.append(feature)

    # === BUILD GEOJSON COLLECTION ===
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    # === COMPUTE BOUNDS ===
    bounds = _compute_bounds(zones_wgs84)
    center = _compute_center(bounds)

    logger.info(f"   ✅ Prepared {len(features)} zone features")
    logger.info(
        f"   📐 Bounds: {bounds[0]:.4f}, {bounds[1]:.4f} to {bounds[2]:.4f}, {bounds[3]:.4f}"
    )

    return ZoneData(
        geojson=geojson,
        zone_count=len(features),
        bounds=bounds,
        center=center,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 📍 BOREHOLE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_borehole_geojson(
    boreholes_gdf: gpd.GeoDataFrame,
    id_field: str = "LocationID",
) -> BoreholeData:
    """
    Prepare borehole points as GeoJSON for deck.gl.

    Converts borehole GeoDataFrame to GeoJSON format, ensuring:
    - Coordinates are in WGS84 for web mapping
    - Each borehole has an ID property
    - Positions are extracted for deck.gl ScatterplotLayer

    Args:
        boreholes_gdf: GeoDataFrame with borehole points
        id_field: Column name for borehole IDs

    Returns:
        BoreholeData with processed GeoJSON and positions
    """
    logger.info(f"📍 Preparing {len(boreholes_gdf)} boreholes for visualization")

    # === REPROJECT TO WGS84 ===
    boreholes_wgs84 = _ensure_wgs84(boreholes_gdf.copy())

    # === BUILD GEOJSON FEATURES AND POSITIONS ===
    features = []
    positions = []

    for idx, row in boreholes_wgs84.iterrows():
        # Get borehole ID
        if id_field in row.index and row[id_field] is not None:
            bh_id = str(row[id_field])
        else:
            bh_id = f"BH_{idx}"

        # Get coordinates
        geom = row.geometry
        if geom is None:
            continue

        lon = float(geom.x)
        lat = float(geom.y)
        positions.append([lon, lat])

        # Build feature
        feature = {
            "type": "Feature",
            "id": bh_id,
            "properties": {
                "id": bh_id,
                "index": len(positions) - 1,  # For linking to positions array
            },
            "geometry": mapping(geom),
        }

        # Copy additional properties
        for col in row.index:
            if col != "geometry" and col not in feature["properties"]:
                val = row[col]
                if hasattr(val, "item"):
                    val = val.item()
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    feature["properties"][col] = val

        features.append(feature)

    # === BUILD GEOJSON COLLECTION ===
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    logger.info(f"   ✅ Prepared {len(features)} borehole features")

    return BoreholeData(
        geojson=geojson,
        borehole_count=len(features),
        positions=positions,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 📤 SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def zones_to_json_string(zone_data: ZoneData) -> str:
    """Serialize zone GeoJSON to compact JSON string.

    Args:
        zone_data: ZoneData container

    Returns:
        Compact JSON string (no pretty-printing)
    """
    return json.dumps(zone_data.geojson, separators=(",", ":"))


def boreholes_to_json_string(borehole_data: BoreholeData) -> str:
    """Serialize borehole GeoJSON to compact JSON string.

    Args:
        borehole_data: BoreholeData container

    Returns:
        Compact JSON string (no pretty-printing)
    """
    return json.dumps(borehole_data.geojson, separators=(",", ":"))


def positions_to_json_string(borehole_data: BoreholeData) -> str:
    """Serialize borehole positions to compact JSON array.

    Args:
        borehole_data: BoreholeData container

    Returns:
        JSON array string of [lon, lat] positions
    """
    return json.dumps(borehole_data.positions, separators=(",", ":"))
