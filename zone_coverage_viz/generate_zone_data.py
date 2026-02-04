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
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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


def get_file_timestamp(file_path: Path) -> Dict[str, str]:
    """Get file modification timestamp as ISO string and human-readable format."""
    if not file_path.exists():
        return {"iso": None, "display": "not found"}
    mtime = file_path.stat().st_mtime
    dt = datetime.fromtimestamp(mtime)
    return {
        "iso": dt.isoformat(),
        "display": dt.strftime("%d %b %Y, %H:%M"),
        "epoch": mtime,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 📂 DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════


def load_zones(workspace_root: Path) -> Tuple[gpd.GeoDataFrame, List[Dict]]:
    """Load zone shapefiles and add max_spacing_m from config.

    Returns:
        Tuple of (combined GeoDataFrame, list of source file info dicts)
    """
    # Add Gap_Analysis_EC7 to path for shapefile_config import
    sys.path.insert(0, str(GAP_ANALYSIS_DIR))

    from shapefile_config import get_coverage_layer_keys, get_layer_config

    coverage_keys = get_coverage_layer_keys()
    logger.info(f"📂 Coverage layers: {coverage_keys}")

    zones_list = []
    source_files = []

    for layer_key in coverage_keys:
        layer_config = get_layer_config(layer_key)
        file_path = workspace_root / layer_config.get("file_path", "")

        if not file_path.exists():
            logger.warning(f"   ⚠️ Shapefile not found: {file_path}")
            continue

        # Track source file
        source_files.append(
            {
                "name": layer_key,
                "path": str(file_path.relative_to(workspace_root)),
                "type": "zones",
                **get_file_timestamp(file_path),
            }
        )

        # Load shapefile
        gdf = gpd.read_file(file_path)

        # Add properties from config
        max_spacing = layer_config.get("max_spacing_m", 100.0)
        display_name = layer_config.get("display_name", layer_key)

        gdf["max_spacing_m"] = max_spacing
        gdf["layer_key"] = layer_key
        # Enumerate zones like main.py does: "Embankment_0", "Embankment_1", etc.
        gdf["zone_name"] = [f"{display_name}_{i}" for i in range(len(gdf))]

        zones_list.append(gdf)
        logger.info(
            f"   ✅ Loaded {len(gdf)} features from {layer_key} (max_spacing={max_spacing}m)"
        )

    if zones_list:
        combined = pd.concat(zones_list, ignore_index=True)
        if combined.crs is None:
            combined = combined.set_crs(CRS_BNG)
        return combined, source_files

    raise ValueError("No zones loaded!")


def load_boreholes(output_dir: Path) -> Tuple[gpd.GeoDataFrame, Dict]:
    """Load proposed boreholes from most recent CSV.

    Returns:
        Tuple of (GeoDataFrame, source file info dict)
    """
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

    # Track source file
    source_info = {
        "name": csv_file.stem,
        "path": str(csv_file.relative_to(output_dir.parent)),
        "type": "boreholes",
        **get_file_timestamp(csv_file),
    }

    # Load CSV
    df = pd.read_csv(csv_file)

    # Create geometry
    geometry = [Point(x, y) for x, y in zip(df["Easting"], df["Northing"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_BNG)

    logger.info(f"   ✅ Loaded {len(gdf)} proposed boreholes")

    return gdf, source_info


def load_existing_coverage(
    output_dir: Path,
) -> Tuple[Optional[gpd.GeoDataFrame], Optional[Dict]]:
    """Load existing borehole coverage from covered.geojson.

    Searches for the most recent coverage_polygons output folder.
    Returns (None, None) if not found.

    Returns:
        Tuple of (GeoDataFrame or None, source file info dict or None)
    """
    coverage_dir = output_dir / "coverage_polygons"

    if not coverage_dir.exists():
        logger.warning("   ⚠️ coverage_polygons directory not found")
        return None, None

    # Find subdirectories (each represents a combo_key like d45_spt0_txt0_txe0)
    combo_dirs = [d for d in coverage_dir.iterdir() if d.is_dir()]

    if not combo_dirs:
        logger.warning("   ⚠️ No combo directories found in coverage_polygons")
        return None, None

    # Find the combo with the most recently modified covered.geojson file
    # (Folder mtime doesn't update when files inside are modified on Windows)
    def get_covered_mtime(combo_dir: Path) -> float:
        covered_path = combo_dir / "covered.geojson"
        if covered_path.exists():
            return covered_path.stat().st_mtime
        return 0.0  # Non-existent files sort last

    combo_dirs_with_coverage = [
        d for d in combo_dirs if (d / "covered.geojson").exists()
    ]

    if not combo_dirs_with_coverage:
        logger.warning("   ⚠️ No combo directories contain covered.geojson")
        return None, None

    # Sort by covered.geojson file modification time
    combo_dir = sorted(combo_dirs_with_coverage, key=get_covered_mtime, reverse=True)[0]
    covered_path = combo_dir / "covered.geojson"

    logger.info(f"📂 Loading existing coverage from: {combo_dir.name}/covered.geojson")

    # Track source file
    source_info = {
        "name": f"{combo_dir.name}/covered.geojson",
        "path": str(covered_path.relative_to(output_dir.parent)),
        "type": "existing_coverage",
        **get_file_timestamp(covered_path),
    }

    gdf = gpd.read_file(covered_path)

    # The covered.geojson may have incorrect CRS metadata (says WGS84 but has BNG coords)
    # Force BNG CRS based on coordinate values (BNG has 6-digit eastings/northings)
    if gdf.crs is None or gdf.crs.to_epsg() == 4326:
        # Check if coordinates look like BNG (6-digit values)
        sample_coord = gdf.geometry.iloc[0].centroid.coords[0]
        if abs(sample_coord[0]) > 1000:  # BNG coordinates are typically 400000-700000
            logger.info(
                f"   ⚠️ Forcing CRS to BNG (coords look like BNG: {sample_coord[0]:.0f}, {sample_coord[1]:.0f})"
            )
            gdf = gdf.set_crs(CRS_BNG, allow_override=True)

    logger.info(f"   ✅ Loaded existing coverage ({len(gdf)} features, CRS: {gdf.crs})")

    return gdf, source_info


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

    # Track source files for timestamp consistency check
    source_files = []

    # Load data
    logger.info("\n📂 Loading zones...")
    zones_gdf, zones_sources = load_zones(workspace_root)
    source_files.extend(zones_sources)

    logger.info("\n📂 Loading boreholes...")
    boreholes_gdf, boreholes_source = load_boreholes(output_dir)
    source_files.append(boreholes_source)

    logger.info("\n📂 Loading existing coverage...")
    existing_coverage_gdf, existing_coverage_source = load_existing_coverage(output_dir)
    if existing_coverage_source:
        source_files.append(existing_coverage_source)

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

    # Check timestamp consistency for DYNAMIC files only (boreholes + coverage)
    # Shapefiles (zones) are static reference data and don't need timestamp checks
    dynamic_files = [
        f for f in source_files if f.get("type") in ("boreholes", "existing_coverage")
    ]
    timestamps = [f.get("epoch", 0) for f in dynamic_files if f.get("epoch")]
    if timestamps:
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        diff_seconds = max_ts - min_ts
        newest_file = max(dynamic_files, key=lambda f: f.get("epoch", 0))
        oldest_file = min(dynamic_files, key=lambda f: f.get("epoch", 0))

        if diff_seconds > 3600:  # More than 1 hour difference
            logger.warning(
                f"   ⚠️ Dynamic file timestamps differ by {diff_seconds/3600:.1f} hours!"
            )
            logger.warning(
                f"      Oldest: {oldest_file['name']} ({oldest_file['display']})"
            )
            logger.warning(
                f"      Newest: {newest_file['name']} ({newest_file['display']})"
            )

    # Combine
    output_data = {
        "crs": CRS_WGS84,
        "zones": zones_geojson,
        "boreholes": boreholes_geojson,
        "source_files": source_files,
        "generated_at": datetime.now().isoformat(),
    }

    # Add existing coverage if loaded
    if existing_coverage_gdf is not None:
        existing_coverage_geojson = gdf_to_geojson(
            existing_coverage_gdf,
            {
                "layer": "layer",
                "combo_key": "combo_key",
            },
        )
        output_data["existing_coverage"] = existing_coverage_geojson
        logger.info(
            f"   ✅ Added existing coverage ({len(existing_coverage_geojson['features'])} features)"
        )

    # Write file
    output_path = output_dir / OUTPUT_FILENAME
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n✅ Generated: {output_path}")
    logger.info(f"   {len(zones_geojson['features'])} zones")
    logger.info(f"   {len(boreholes_geojson['features'])} boreholes")
    if existing_coverage_gdf is not None:
        logger.info(
            f"   {len(existing_coverage_geojson['features'])} existing coverage polygons"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    generate_zone_coverage_data()
