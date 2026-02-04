#!/usr/bin/env python3
"""
Zone Coverage Visualization - Data Loader (Standalone)

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Load zone and borehole data from zone_coverage_data.json file.
This loader is fully standalone - no imports from EC7 modules.

Key Features:
1. Load from zone_coverage_data.json (exported by main.py)
2. Fallback to CSV + shapefile if JSON not found
3. Coordinate handling for BNG and WGS84
4. Provide GeoJSON output for frontend

Navigation Guide:
- DataLoader: Main loader class
- get_zones_geojson: Zone boundaries for map display
- get_boreholes_geojson: Borehole positions for draggable markers

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import logging
import os

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, shape, mapping

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

CRS_WGS84 = "EPSG:4326"
CRS_BNG = "EPSG:27700"

DATA_FILENAME = "zone_coverage_data.json"

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 📂 DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════


class DataLoader:
    """
    Load zone and borehole data from zone_coverage_data.json.

    This is a standalone loader that doesn't import from EC7 modules.
    All required data is in the JSON file exported by main.py.
    """

    def __init__(self, data_dir: Path) -> None:
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing zone_coverage_data.json
        """
        self.data_dir = Path(data_dir)

        # Loaded data (cached)
        self._zones_data: Optional[Dict[str, Any]] = None
        self._boreholes_data: Optional[Dict[str, Any]] = None
        self._existing_coverage_data: Optional[Dict[str, Any]] = None
        self._zones_gdf: Optional[gpd.GeoDataFrame] = None
        self._boreholes_gdf: Optional[gpd.GeoDataFrame] = None
        self._existing_coverage_gdf: Optional[gpd.GeoDataFrame] = None

        # Data file timestamp
        self._data_file_modified: Optional[datetime] = None
        self._data_loaded_at: Optional[datetime] = None

        # Coordinate transformer
        self._wgs84_to_bng = Transformer.from_crs(CRS_WGS84, CRS_BNG, always_xy=True)
        self._bng_to_wgs84 = Transformer.from_crs(CRS_BNG, CRS_WGS84, always_xy=True)

        # Load data on init
        self._load_data()

    def _load_data(self) -> None:
        """Load zones and boreholes from JSON file."""
        json_path = self.data_dir / DATA_FILENAME

        if json_path.exists():
            self._load_from_json(json_path)
        else:
            logger.warning(f"JSON file not found: {json_path}")
            logger.info("Attempting fallback to CSV + shapefile loading...")
            self._load_fallback()

    def _load_from_json(self, json_path: Path) -> None:
        """
        Load data from zone_coverage_data.json file.

        Args:
            json_path: Path to the JSON file
        """
        logger.info(f"Loading data from: {json_path.name}")

        # Capture file modification time
        file_mtime = os.path.getmtime(json_path)
        self._data_file_modified = datetime.fromtimestamp(file_mtime)
        self._data_loaded_at = datetime.now()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract zones and boreholes
        self._zones_data = data.get(
            "zones", {"type": "FeatureCollection", "features": []}
        )
        self._boreholes_data = data.get(
            "boreholes", {"type": "FeatureCollection", "features": []}
        )
        self._existing_coverage_data = data.get("existing_coverage", None)

        # Convert to GeoDataFrames (in BNG for geometry operations)
        self._zones_gdf = self._geojson_to_gdf(self._zones_data, CRS_BNG)
        self._boreholes_gdf = self._geojson_to_gdf(self._boreholes_data, CRS_BNG)

        if self._existing_coverage_data:
            self._existing_coverage_gdf = self._geojson_to_gdf(
                self._existing_coverage_data, CRS_BNG
            )
            existing_count = len(self._existing_coverage_gdf)
        else:
            existing_count = 0

        logger.info(
            f"Loaded {len(self._zones_gdf)} zones, {len(self._boreholes_gdf)} boreholes"
            + (
                f", {existing_count} existing coverage polygons"
                if existing_count
                else ""
            )
        )

    def _geojson_to_gdf(
        self,
        geojson: Dict[str, Any],
        target_crs: str,
    ) -> gpd.GeoDataFrame:
        """
        Convert GeoJSON FeatureCollection to GeoDataFrame.

        Args:
            geojson: GeoJSON FeatureCollection dict
            target_crs: Target CRS for the output GeoDataFrame

        Returns:
            GeoDataFrame in the target CRS
        """
        features = geojson.get("features", [])

        if not features:
            return gpd.GeoDataFrame()

        # Build lists for GeoDataFrame
        geometries = []
        properties_list = []

        for feature in features:
            geom = shape(feature.get("geometry", {}))
            geometries.append(geom)
            properties_list.append(feature.get("properties", {}))

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            properties_list, geometry=geometries, crs=CRS_WGS84  # JSON data is in WGS84
        )

        # Convert to target CRS
        if target_crs != CRS_WGS84:
            gdf = gdf.to_crs(target_crs)

        return gdf

    def _load_fallback(self) -> None:
        """
        Fallback loading from CSV (boreholes) and shapefiles (zones).

        This is used when zone_coverage_data.json is not available.
        It requires the EC7 module to be in the Python path.
        """
        # Try loading boreholes from CSV
        self._boreholes_gdf = self._load_boreholes_csv()

        # Try loading zones from shapefiles via EC7 config
        self._zones_gdf = self._load_zones_shapefile()

    def _load_boreholes_csv(self) -> gpd.GeoDataFrame:
        """Load proposed boreholes from CSV output."""
        try:
            proposed_dir = self.data_dir / "proposed_boreholes"

            if not proposed_dir.exists():
                logger.warning(
                    f"Proposed boreholes directory not found: {proposed_dir}"
                )
                return gpd.GeoDataFrame()

            # Find CSV files (not directories)
            csv_files = [f for f in proposed_dir.glob("*.csv") if f.is_file()]

            if not csv_files:
                logger.warning("No proposed boreholes CSV files found")
                return gpd.GeoDataFrame()

            # Use most recent CSV file
            csv_file = sorted(csv_files, key=lambda f: f.stat().st_mtime, reverse=True)[
                0
            ]
            logger.info(f"Loading boreholes from: {csv_file.name}")

            # Load CSV
            df = pd.read_csv(csv_file)

            if "Easting" not in df.columns or "Northing" not in df.columns:
                logger.error(
                    f"CSV missing Easting/Northing columns: {df.columns.tolist()}"
                )
                return gpd.GeoDataFrame()

            # Create geometry
            geometry = [Point(x, y) for x, y in zip(df["Easting"], df["Northing"])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_BNG)

            # Add index column for frontend
            gdf["index"] = range(len(gdf))

            logger.info(f"Loaded {len(gdf)} proposed boreholes")
            return gdf

        except Exception as e:
            logger.error(f"Failed to load boreholes CSV: {e}")
            return gpd.GeoDataFrame()

    def _load_zones_shapefile(self) -> gpd.GeoDataFrame:
        """Load zones from shapefiles using EC7 config."""
        try:
            # Find workspace root
            workspace_root = self._find_workspace_root(self.data_dir)

            # Import shapefile config from Gap_Analysis_EC7 folder
            import sys

            gap_analysis_dir = self.data_dir.parent
            if str(gap_analysis_dir) not in sys.path:
                sys.path.insert(0, str(gap_analysis_dir))

            from shapefile_config import (
                get_coverage_layer_keys,
                get_layer_config,
            )

            coverage_keys = get_coverage_layer_keys()
            logger.info(f"Coverage layers: {coverage_keys}")

            zones_list = []

            for layer_key in coverage_keys:
                layer_config = get_layer_config(layer_key)
                file_path = workspace_root / layer_config.get("file_path", "")

                if not file_path.exists():
                    logger.warning(f"Shapefile not found: {file_path}")
                    continue

                gdf = gpd.read_file(file_path)

                max_spacing = layer_config.get("max_spacing_m", 100.0)
                gdf["max_spacing_m"] = max_spacing
                gdf["layer_key"] = layer_key
                gdf["zone_name"] = layer_config.get("display_name", layer_key)

                zones_list.append(gdf)
                logger.info(f"Loaded {len(gdf)} features from {layer_key}")

            if zones_list:
                combined = pd.concat(zones_list, ignore_index=True)
                if combined.crs is None:
                    combined = combined.set_crs(CRS_BNG)
                return combined

            logger.error("No zones loaded from shapefiles!")
            return gpd.GeoDataFrame()

        except ImportError:
            logger.error("Cannot import shapefile_config - EC7 modules not available")
            return gpd.GeoDataFrame()
        except Exception as e:
            logger.error(f"Failed to load zones from shapefiles: {e}")
            import traceback

            traceback.print_exc()
            return gpd.GeoDataFrame()

    def _find_workspace_root(self, start_dir: Path) -> Path:
        """Find workspace root by looking for 'Project Shapefiles' folder."""
        current = start_dir.resolve()

        for _ in range(5):
            if (current / "Project Shapefiles").exists():
                return current
            if current.parent == current:
                break
            current = current.parent

        return start_dir.parent

    # ═══════════════════════════════════════════════════════════════════════
    # 📤 PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════

    def get_zones_gdf(self) -> gpd.GeoDataFrame:
        """Get zones as GeoDataFrame (in BNG)."""
        return self._zones_gdf if self._zones_gdf is not None else gpd.GeoDataFrame()

    def get_boreholes_gdf(self) -> gpd.GeoDataFrame:
        """Get boreholes as GeoDataFrame (in BNG)."""
        return (
            self._boreholes_gdf
            if self._boreholes_gdf is not None
            else gpd.GeoDataFrame()
        )

    def get_zones_geojson(self) -> Dict[str, Any]:
        """Get zones as GeoJSON FeatureCollection in WGS84."""
        # If we loaded from JSON, return the original
        if self._zones_data is not None:
            return self._zones_data

        # Otherwise convert from GeoDataFrame
        zones_gdf = self.get_zones_gdf()

        if zones_gdf.empty:
            return {"type": "FeatureCollection", "features": []}

        zones_wgs84 = zones_gdf.to_crs(CRS_WGS84)

        features = []
        for idx, row in zones_wgs84.iterrows():
            feature = {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": {
                    "index": idx,
                    "zone_name": row.get("zone_name", f"Zone_{idx}"),
                    "max_spacing_m": row.get("max_spacing_m", 100.0),
                    "layer_key": row.get("layer_key", "unknown"),
                    "display_name": row.get("zone_name", f"Zone_{idx}"),
                },
            }
            features.append(feature)

        return {"type": "FeatureCollection", "features": features}

    def get_boreholes_geojson(self) -> Dict[str, Any]:
        """Get boreholes as GeoJSON FeatureCollection in WGS84."""
        # If we loaded from JSON, return the original
        if self._boreholes_data is not None:
            return self._boreholes_data

        # Otherwise convert from GeoDataFrame
        boreholes_gdf = self.get_boreholes_gdf()

        if boreholes_gdf.empty:
            return {"type": "FeatureCollection", "features": []}

        boreholes_wgs84 = boreholes_gdf.to_crs(CRS_WGS84)

        features = []
        for idx, row in boreholes_wgs84.iterrows():
            feature = {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": {
                    "index": idx,
                    "location_id": row.get("Location_ID", f"PROP_{idx:03d}"),
                },
            }
            features.append(feature)

        return {"type": "FeatureCollection", "features": features}

    def get_existing_coverage_geojson(self) -> Optional[Dict[str, Any]]:
        """Get existing borehole coverage as GeoJSON FeatureCollection in WGS84.

        Returns:
            GeoJSON FeatureCollection or None if not available
        """
        # If we loaded from JSON, return the original
        if self._existing_coverage_data is not None:
            return self._existing_coverage_data

        # Otherwise no existing coverage available
        return None

    def update_borehole_position(
        self,
        index: int,
        lon: float,
        lat: float,
    ) -> None:
        """
        Update a borehole's position.

        Args:
            index: Borehole index (0-based)
            lon: New longitude in WGS84
            lat: New latitude in WGS84
        """
        if self._boreholes_gdf is None or index >= len(self._boreholes_gdf):
            logger.warning(f"Invalid borehole index: {index}")
            return

        # Convert WGS84 to BNG
        x, y = self._wgs84_to_bng.transform(lon, lat)

        # Update geometry in GeoDataFrame
        self._boreholes_gdf.at[index, "geometry"] = Point(x, y)

        # Also update the JSON data if present
        if self._boreholes_data is not None:
            features = self._boreholes_data.get("features", [])
            if index < len(features):
                features[index]["geometry"]["coordinates"] = [lon, lat]

        logger.debug(f"Updated borehole {index} to ({x:.2f}, {y:.2f})")

    def delete_borehole(self, index: int) -> Optional[str]:
        """
        Delete a borehole by index.

        Args:
            index: Borehole index (0-based)

        Returns:
            The deleted borehole's ID if successful, None otherwise.
        """
        if self._boreholes_gdf is None or index >= len(self._boreholes_gdf):
            logger.warning(f"Invalid borehole index for deletion: {index}")
            return None

        # Get the borehole ID before deleting
        deleted_id = self.get_borehole_id(index)

        # Delete from GeoDataFrame
        self._boreholes_gdf = self._boreholes_gdf.drop(index).reset_index(drop=True)

        # Also delete from JSON data if present
        if self._boreholes_data is not None:
            features = self._boreholes_data.get("features", [])
            if index < len(features):
                features.pop(index)
                # Re-index remaining features
                for i, feature in enumerate(features):
                    feature["properties"]["index"] = i

        logger.info(f"🗑️ Deleted borehole {deleted_id} at index {index}")
        return deleted_id

    def get_borehole_id(self, index: int) -> str:
        """
        Get the ID of a borehole by index.
        
        Args:
            index: Borehole index (0-based)
            
        Returns:
            The borehole's ID string.
        """
        # Try JSON data first
        if self._boreholes_data is not None:
            features = self._boreholes_data.get("features", [])
            if index < len(features):
                props = features[index].get("properties", {})
                return props.get("location_id") or props.get("id") or f"BH_{index}"
        
        # Try GeoDataFrame
        if self._boreholes_gdf is not None and index < len(self._boreholes_gdf):
            row = self._boreholes_gdf.iloc[index]
            return row.get("Location_ID") or row.get("location_id") or f"BH_{index}"
        
        return f"BH_{index}"

    def add_borehole(
        self,
        lon: float,
        lat: float,
        location_id: Optional[str] = None,
    ) -> int:
        """
        Add a new borehole at the specified location.

        Args:
            lon: Longitude in WGS84
            lat: Latitude in WGS84
            location_id: Optional location ID (auto-generated if not provided)

        Returns:
            Index of the newly added borehole.
        """
        # Convert WGS84 to BNG
        x, y = self._wgs84_to_bng.transform(lon, lat)

        # Generate location ID if not provided
        new_index = len(self._boreholes_gdf) if self._boreholes_gdf is not None else 0
        if location_id is None:
            location_id = f"NEW_{new_index:03d}"

        # Create new row
        new_row = gpd.GeoDataFrame(
            [{"Location_ID": location_id, "index": new_index}],
            geometry=[Point(x, y)],
            crs=CRS_BNG,
        )

        # Add to GeoDataFrame
        if self._boreholes_gdf is None or self._boreholes_gdf.empty:
            self._boreholes_gdf = new_row
        else:
            self._boreholes_gdf = pd.concat(
                [self._boreholes_gdf, new_row], ignore_index=True
            )

        # Also add to JSON data if present
        if self._boreholes_data is not None:
            new_feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"index": new_index, "location_id": location_id},
            }
            self._boreholes_data.setdefault("features", []).append(new_feature)

        logger.info(f"➕ Added borehole {location_id} at ({lon:.6f}, {lat:.6f})")
        return new_index

    def restore_state(self, boreholes_geojson: Dict[str, Any]) -> None:
        """
        Restore boreholes state from a GeoJSON snapshot (for undo).

        Args:
            boreholes_geojson: GeoJSON FeatureCollection to restore.
        """
        self._boreholes_data = boreholes_geojson
        self._boreholes_gdf = self._geojson_to_gdf(boreholes_geojson, CRS_BNG)
        logger.info(f"🔄 Restored state with {len(self._boreholes_gdf)} boreholes")

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data including timestamps.

        Returns:
            Dict with data_file_modified, data_loaded_at, zone_count, borehole_count.
        """
        return {
            "data_file_modified": (
                self._data_file_modified.isoformat()
                if self._data_file_modified
                else None
            ),
            "data_loaded_at": (
                self._data_loaded_at.isoformat() if self._data_loaded_at else None
            ),
            "zone_count": len(self._zones_gdf) if self._zones_gdf is not None else 0,
            "borehole_count": (
                len(self._boreholes_gdf) if self._boreholes_gdf is not None else 0
            ),
        }
