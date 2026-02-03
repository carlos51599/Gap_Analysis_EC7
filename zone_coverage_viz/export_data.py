#!/usr/bin/env python3
"""
Zone Coverage Data Exporter

Exports zone and borehole data to a standalone JSON file for the zone_coverage_viz app.
This allows the visualization to work without importing from EC7 modules.

Usage:
    from zone_coverage_viz.export_data import export_zone_coverage_data

    export_zone_coverage_data(
        zones_gdf=zones_gdf,
        boreholes_gdf=proposed_boreholes_gdf,
        zones_config=zones_config,  # Dict with max_spacing_m per zone
        output_dir=Path("Output")
    )
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging

import geopandas as gpd
from shapely.geometry import mapping

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

CRS_WGS84 = "EPSG:4326"
CRS_BNG = "EPSG:27700"

OUTPUT_FILENAME = "zone_coverage_data.json"

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 📤 EXPORT FUNCTION
# ═══════════════════════════════════════════════════════════════════════════


def export_zone_coverage_data(
    zones_gdf: gpd.GeoDataFrame,
    boreholes_gdf: gpd.GeoDataFrame,
    zones_config: Dict[str, Any],
    output_dir: Path,
    log: Optional[logging.Logger] = None,
) -> Path:
    """
    Export zone and borehole data for zone_coverage_viz app.

    Args:
        zones_gdf: GeoDataFrame with zone polygons (in any CRS)
        boreholes_gdf: GeoDataFrame with proposed borehole points
        zones_config: Dict mapping zone_name -> {max_spacing_m, ...}
        output_dir: Directory to write the output file
        log: Optional logger

    Returns:
        Path to the created JSON file.

    Output Format:
        {
            "crs": "EPSG:4326",
            "zones": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {...},
                        "properties": {
                            "zone_name": "Embankment",
                            "max_spacing_m": 100.0,
                            "layer_key": "embankment_zones"
                        }
                    }
                ]
            },
            "boreholes": {
                "type": "FeatureCollection",
                "features": [...]
            }
        }
    """
    log = log or logger

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert zones to WGS84
    if zones_gdf.crs is None:
        log.warning("Zones GDF has no CRS, assuming EPSG:27700")
        zones_gdf = zones_gdf.set_crs(CRS_BNG)
    zones_wgs84 = zones_gdf.to_crs(CRS_WGS84)

    # Convert boreholes to WGS84
    if boreholes_gdf.crs is None:
        log.warning("Boreholes GDF has no CRS, assuming EPSG:27700")
        boreholes_gdf = boreholes_gdf.set_crs(CRS_BNG)
    boreholes_wgs84 = boreholes_gdf.to_crs(CRS_WGS84)

    # Build zones GeoJSON
    zone_features = []
    for idx, row in zones_wgs84.iterrows():
        zone_name = row.get("zone_name", row.get("Name", f"Zone_{idx}"))

        # Get max_spacing_m from zones_config or row
        max_spacing = 100.0  # Default
        if zone_name in zones_config:
            max_spacing = zones_config[zone_name].get("max_spacing_m", 100.0)
        elif "max_spacing_m" in row.index:
            max_spacing = float(row["max_spacing_m"])

        feature = {
            "type": "Feature",
            "geometry": mapping(row.geometry),
            "properties": {
                "zone_name": str(zone_name),
                "max_spacing_m": float(max_spacing),
                "layer_key": row.get("layer_key", "unknown"),
            },
        }
        zone_features.append(feature)

    # Build boreholes GeoJSON
    borehole_features = []
    for idx, row in boreholes_wgs84.iterrows():
        feature = {
            "type": "Feature",
            "geometry": mapping(row.geometry),
            "properties": {
                "index": int(idx),
                "location_id": row.get("Location_ID", f"PROP_{idx:03d}"),
            },
        }
        borehole_features.append(feature)

    # Combine into output data
    output_data = {
        "crs": CRS_WGS84,
        "zones": {"type": "FeatureCollection", "features": zone_features},
        "boreholes": {"type": "FeatureCollection", "features": borehole_features},
    }

    # Write JSON file
    output_path = output_dir / OUTPUT_FILENAME
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    log.info(f"📄 Exported zone coverage data: {output_path}")
    log.info(f"   {len(zone_features)} zones, {len(borehole_features)} boreholes")

    return output_path
