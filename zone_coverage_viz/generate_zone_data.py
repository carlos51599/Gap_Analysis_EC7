#!/usr/bin/env python3
"""
Generate zone_coverage_data.json from main.py output.

This script creates the standalone data file that zone_coverage_viz uses.
Run this after running main.py to generate the required data file.

Usage:
    python generate_zone_data.py

Output:
    Output/zone_coverage_data.json
"""

from pathlib import Path
import json
import logging
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, mapping

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Project paths
SCRIPT_DIR = Path(__file__).parent
GAP_ANALYSIS_DIR = (
    SCRIPT_DIR.parent if SCRIPT_DIR.name == "zone_coverage_viz" else SCRIPT_DIR
)
WORKSPACE_ROOT = None  # Auto-detected

OUTPUT_FILENAME = "zone_coverage_data.json"

CRS_WGS84 = "EPSG:4326"
CRS_BNG = "EPSG:27700"

# Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 🔍 PATH DETECTION
# ═══════════════════════════════════════════════════════════════════════════


def find_workspace_root() -> Path:
    """Find workspace root by looking for 'Project Shapefiles' folder."""
    current = GAP_ANALYSIS_DIR.resolve()

    for _ in range(5):
        if (current / "Project Shapefiles").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    raise FileNotFoundError("Could not find workspace root with 'Project Shapefiles'")


# ═══════════════════════════════════════════════════════════════════════════
# 📂 DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════


def load_zones(workspace_root: Path) -> gpd.GeoDataFrame:
    """Load zone shapefiles and add max_spacing_m from config."""
    # Add Gap_Analysis_EC7 to path for shapefile_config import
    sys.path.insert(0, str(GAP_ANALYSIS_DIR))

    from shapefile_config import get_coverage_layer_keys, get_layer_config

    coverage_keys = get_coverage_layer_keys()
    logger.info(f"📂 Coverage layers: {coverage_keys}")

    zones_list = []

    for layer_key in coverage_keys:
        layer_config = get_layer_config(layer_key)
        file_path = workspace_root / layer_config.get("file_path", "")

        if not file_path.exists():
            logger.warning(f"   ⚠️ Shapefile not found: {file_path}")
            continue

        # Load shapefile
        gdf = gpd.read_file(file_path)

        # Add properties from config
        max_spacing = layer_config.get("max_spacing_m", 100.0)
        display_name = layer_config.get("display_name", layer_key)

        gdf["max_spacing_m"] = max_spacing
        gdf["layer_key"] = layer_key
        gdf["zone_name"] = display_name

        zones_list.append(gdf)
        logger.info(
            f"   ✅ Loaded {len(gdf)} features from {layer_key} (max_spacing={max_spacing}m)"
        )

    if zones_list:
        combined = pd.concat(zones_list, ignore_index=True)
        if combined.crs is None:
            combined = combined.set_crs(CRS_BNG)
        return combined

    raise ValueError("No zones loaded!")


def load_boreholes(output_dir: Path) -> gpd.GeoDataFrame:
    """Load proposed boreholes from most recent CSV."""
    proposed_dir = output_dir / "proposed_boreholes"

    if not proposed_dir.exists():
        raise FileNotFoundError(
            f"Proposed boreholes directory not found: {proposed_dir}"
        )

    # Find CSV files
    csv_files = [f for f in proposed_dir.glob("*.csv") if f.is_file()]

    if not csv_files:
        raise FileNotFoundError("No proposed boreholes CSV files found")

    # Use most recent
    csv_file = sorted(csv_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
    logger.info(f"📂 Loading boreholes from: {csv_file.name}")

    # Load CSV
    df = pd.read_csv(csv_file)

    # Create geometry
    geometry = [Point(x, y) for x, y in zip(df["Easting"], df["Northing"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_BNG)

    logger.info(f"   ✅ Loaded {len(gdf)} proposed boreholes")

    return gdf


# ═══════════════════════════════════════════════════════════════════════════
# 📤 EXPORT
# ═══════════════════════════════════════════════════════════════════════════


def gdf_to_geojson(gdf: gpd.GeoDataFrame, properties_map: dict) -> dict:
    """Convert GeoDataFrame to GeoJSON FeatureCollection in WGS84."""
    # Convert to WGS84
    gdf_wgs84 = gdf.to_crs(CRS_WGS84)

    features = []
    for idx, row in gdf_wgs84.iterrows():
        props = {}
        for key, col in properties_map.items():
            if col in row.index:
                val = row[col]
                # Handle numpy types
                if hasattr(val, "item"):
                    val = val.item()
                props[key] = val
            else:
                props[key] = None
        props["index"] = int(idx)

        feature = {
            "type": "Feature",
            "geometry": mapping(row.geometry),
            "properties": props,
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def generate_zone_coverage_data() -> None:
    """Main function to generate zone_coverage_data.json."""
    logger.info("=" * 60)
    logger.info("🚀 Generating zone coverage data")
    logger.info("=" * 60)

    # Find paths
    workspace_root = find_workspace_root()
    output_dir = GAP_ANALYSIS_DIR / "Output"

    logger.info(f"\n📍 Workspace root: {workspace_root}")
    logger.info(f"📍 Output directory: {output_dir}")

    # Load data
    logger.info("\n📂 Loading zones...")
    zones_gdf = load_zones(workspace_root)

    logger.info("\n📂 Loading boreholes...")
    boreholes_gdf = load_boreholes(output_dir)

    # Convert to GeoJSON
    logger.info("\n📝 Converting to GeoJSON...")

    zones_geojson = gdf_to_geojson(
        zones_gdf,
        {
            "zone_name": "zone_name",
            "max_spacing_m": "max_spacing_m",
            "layer_key": "layer_key",
        },
    )

    boreholes_geojson = gdf_to_geojson(
        boreholes_gdf,
        {
            "location_id": "Location_ID",
        },
    )

    # Combine
    output_data = {
        "crs": CRS_WGS84,
        "zones": zones_geojson,
        "boreholes": boreholes_geojson,
    }

    # Write file
    output_path = output_dir / OUTPUT_FILENAME
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n✅ Generated: {output_path}")
    logger.info(f"   {len(zones_geojson['features'])} zones")
    logger.info(f"   {len(boreholes_geojson['features'])} boreholes")
    logger.info("=" * 60)


if __name__ == "__main__":
    generate_zone_coverage_data()
