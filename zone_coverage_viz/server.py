#!/usr/bin/env python3
"""
Zone Coverage Visualization - Flask Server

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Lightweight Flask server for interactive zone coverage visualization.
Provides REST API endpoints for zone/borehole data and real-time coverage computation.

Key Interactions:
- Loads zone and borehole GeoJSON from main.py output
- Uses Shapely for zone-clipped coverage computation
- Serves Leaflet map interface for drag-and-drop interaction

Navigation Guide:
- ROUTES: API endpoints (/api/zones, /api/boreholes, /api/coverage)
- STARTUP: Server initialization and data loading

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import logging

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

from zone_coverage_viz.data_loader import DataLoader
from zone_coverage_viz.geometry_service import CoverageService

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Default data directory - can be overridden via environment variable
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "Output"

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5050

# ═══════════════════════════════════════════════════════════════════════════
# 🌐 FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global services - initialized on startup
data_loader: Optional[DataLoader] = None
coverage_service: Optional[CoverageService] = None

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 🛣️ API ROUTES
# ═══════════════════════════════════════════════════════════════════════════


@app.route("/")
def index() -> str:
    """Serve the main map interface."""
    return render_template("index.html")


@app.route("/api/zones")
def get_zones() -> Dict[str, Any]:
    """
    Get all zone boundaries as GeoJSON FeatureCollection.

    Returns:
        GeoJSON FeatureCollection with zone polygons and max_spacing_m property.
    """
    if data_loader is None:
        return jsonify({"error": "Server not initialized"}), 500

    zones_geojson = data_loader.get_zones_geojson()
    return jsonify(zones_geojson)


@app.route("/api/boreholes")
def get_boreholes() -> Dict[str, Any]:
    """
    Get all borehole positions as GeoJSON FeatureCollection.

    Returns:
        GeoJSON FeatureCollection with borehole points.
    """
    if data_loader is None:
        return jsonify({"error": "Server not initialized"}), 500

    boreholes_geojson = data_loader.get_boreholes_geojson()
    return jsonify(boreholes_geojson)


@app.route("/api/coverage", methods=["POST"])
def compute_coverage() -> Dict[str, Any]:
    """
    Compute zone-clipped coverage for a single borehole position.

    Request Body:
        {
            "lon": float,  # Longitude (WGS84)
            "lat": float   # Latitude (WGS84)
        }

    Returns:
        GeoJSON Feature with coverage polygon, or null if outside all zones.
    """
    if coverage_service is None:
        return jsonify({"error": "Server not initialized"}), 500

    data = request.get_json()
    if not data or "lon" not in data or "lat" not in data:
        return jsonify({"error": "Missing lon/lat in request body"}), 400

    lon = float(data["lon"])
    lat = float(data["lat"])

    coverage = coverage_service.compute_coverage(lon, lat)

    return jsonify(coverage)


@app.route("/api/coverage/update", methods=["POST"])
def update_borehole_coverage() -> Dict[str, Any]:
    """
    Update a borehole position and recompute its coverage.

    Request Body:
        {
            "index": int,   # Borehole index (0-based)
            "lon": float,   # New longitude (WGS84)
            "lat": float    # New latitude (WGS84)
        }

    Returns:
        {
            "coverage": GeoJSON Feature with coverage polygon,
            "zone_info": {zone_name: max_spacing_m} for intersected zones
        }
    """
    if coverage_service is None or data_loader is None:
        return jsonify({"error": "Server not initialized"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    required = ["index", "lon", "lat"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing {field} in request body"}), 400

    index = int(data["index"])
    lon = float(data["lon"])
    lat = float(data["lat"])

    # Update borehole position in data loader
    data_loader.update_borehole_position(index, lon, lat)

    # Compute coverage for new position
    coverage = coverage_service.compute_coverage(lon, lat)
    zone_info = coverage_service.get_zone_info(lon, lat)

    return jsonify({"coverage": coverage, "zone_info": zone_info})


@app.route("/api/coverage/all", methods=["GET"])
def compute_all_coverages() -> Dict[str, Any]:
    """
    Compute coverage polygons for all boreholes at their current positions.

    Returns:
        GeoJSON FeatureCollection with coverage polygons for all boreholes.
    """
    if coverage_service is None or data_loader is None:
        return jsonify({"error": "Server not initialized"}), 500

    boreholes = data_loader.get_boreholes_geojson()
    coverages = []

    for i, feature in enumerate(boreholes.get("features", [])):
        coords = feature.get("geometry", {}).get("coordinates", [])
        if len(coords) >= 2:
            lon, lat = coords[0], coords[1]
            coverage = coverage_service.compute_coverage(lon, lat)
            if coverage:
                coverage["properties"]["borehole_index"] = i
                coverages.append(coverage)

    return jsonify({"type": "FeatureCollection", "features": coverages})


# ═══════════════════════════════════════════════════════════════════════════
# 🚀 SERVER INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def initialize_services(data_dir: Path) -> bool:
    """
    Initialize data loader and coverage service.

    Args:
        data_dir: Directory containing zones.geojson and boreholes.geojson

    Returns:
        True if initialization successful, False otherwise.
    """
    global data_loader, coverage_service

    try:
        logger.info(f"🚀 Initializing services from: {data_dir}")

        # Initialize data loader
        data_loader = DataLoader(data_dir)

        # Initialize coverage service with zones
        zones_gdf = data_loader.get_zones_gdf()
        coverage_service = CoverageService(zones_gdf)

        logger.info(f"✅ Loaded {len(zones_gdf)} zones")
        logger.info(f"✅ Loaded {len(data_loader.get_boreholes_gdf())} boreholes")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        return False


def main() -> None:
    """Main entry point - initialize and start server."""
    import sys

    # Get data directory from command line or use default
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = DEFAULT_DATA_DIR

    # Initialize services
    if not initialize_services(data_dir):
        logger.error("Failed to initialize. Check data files exist.")
        sys.exit(1)

    # Start server
    logger.info(f"🌐 Starting server at http://{SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"   Open browser to: http://{SERVER_HOST}:{SERVER_PORT}")

    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=True)


if __name__ == "__main__":
    main()
