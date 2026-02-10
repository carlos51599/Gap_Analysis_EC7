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
import io
import csv
import sys

from flask import (
    Flask,
    jsonify,
    request,
    render_template,
    send_from_directory,
    Response,
)
from flask_cors import CORS
from pyproj import Transformer

from zone_coverage_viz.viz_config_types import get_frontend_config
from zone_coverage_viz.data_loader import DataLoader
from zone_coverage_viz.geometry_service import CoverageService

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


# Determine data directory - handle frozen exe vs regular script
def get_data_dir() -> Path:
    """Get the data directory, handling PyInstaller frozen exe case."""
    if getattr(sys, "frozen", False):
        # Running as frozen exe - look relative to exe location
        return Path(sys.executable).parent / "Output"
    else:
        # Running as script - look relative to this file's parent
        return Path(__file__).parent.parent / "Output"


DEFAULT_DATA_DIR = get_data_dir()

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5051

# ═══════════════════════════════════════════════════════════════════════════
# 🌐 FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global services - initialized on startup
data_loader: Optional[DataLoader] = None
coverage_service: Optional[CoverageService] = None

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# Debug logging helper - writes to stderr which is unbuffered
import sys


def debug_log(msg: str) -> None:
    """Write debug message to stderr (unbuffered) for guaranteed visibility."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# ═══════════════════════════════════════════════════════════════════════════
# 🛣️ API ROUTES
# ═══════════════════════════════════════════════════════════════════════════


@app.route("/")
def index() -> str:
    """Serve the main map interface."""
    return render_template("index.html")


@app.route("/api/config")
def get_config() -> Dict[str, Any]:
    """
    Get frontend configuration settings.

    Returns:
        JSON object with all configurable settings for the frontend.
    """
    return jsonify(get_frontend_config())


@app.route("/api/data/info")
def get_data_info() -> Dict[str, Any]:
    """
    Get information about loaded data including timestamps.

    Returns:
        JSON object with data_file_modified, data_loaded_at, zone_count, borehole_count.
    """
    if data_loader is None:
        return jsonify({"error": "Server not initialized"}), 500

    return jsonify(data_loader.get_data_info())


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
    Update a borehole position and recompute its coverage. Returns immediately (lazy stats).

    Request Body:
        {
            "index": int,   # Borehole index (0-based)
            "lon": float,   # New longitude (WGS84)
            "lat": float    # New latitude (WGS84)
        }

    Returns:
        {
            "coverage": GeoJSON Feature with coverage polygon,
            "zone_info": {zone_name: max_spacing_m} for intersected zones,
            "stats_pending": bool (true = frontend should request stats separately)
        }
    """
    import time

    # Force immediate output to terminal using stderr
    debug_log("=" * 60)
    debug_log("[MOVE] Request received")

    t_start = time.perf_counter()
    debug_log(f"    [0] Request started")

    if coverage_service is None or data_loader is None:
        return jsonify({"error": "Server not initialized"}), 500

    t1 = time.perf_counter()
    data = request.get_json()
    t2 = time.perf_counter()
    debug_log(f"    [1] get_json(): {(t2-t1)*1000:.3f}ms")

    if not data:
        return jsonify({"error": "Missing request body"}), 400

    required = ["index", "lon", "lat"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing {field} in request body"}), 400

    t1 = time.perf_counter()
    index = int(data["index"])
    lon = float(data["lon"])
    lat = float(data["lat"])
    exclude_zones = data.get(
        "excludeZones", []
    )  # R2: Hidden zones to exclude from zone_ids
    t2 = time.perf_counter()
    debug_log(
        f"    [2] Parse params: {(t2-t1)*1000:.3f}ms | index={index}, exclude_zones={exclude_zones}"
    )

    # Get borehole ID for cache
    t1 = time.perf_counter()
    borehole_id = data_loader.get_borehole_id(index)
    t2 = time.perf_counter()
    debug_log(f"    [3] get_borehole_id: {(t2-t1)*1000:.3f}ms | id={borehole_id}")

    # Transform ONCE and share with both functions
    t1 = time.perf_counter()
    bng_coords = coverage_service.transform_wgs84_to_bng(lon, lat)
    t2 = time.perf_counter()
    debug_log(f"    [4] CRS transform: {(t2-t1)*1000:.3f}ms -> BNG={bng_coords}")

    # Compute coverage using pre-computed BNG coords
    t1 = time.perf_counter()
    coverage = coverage_service.compute_coverage_cached(
        borehole_id, lon, lat, bng_coords, exclude_zones
    )
    t2 = time.perf_counter()
    debug_log(f"    [5] compute_coverage_cached: {(t2-t1)*1000:.1f}ms")

    t1 = time.perf_counter()
    zone_info = coverage_service.get_zone_info(lon, lat, exclude_zones)
    t2 = time.perf_counter()
    debug_log(f"    [6] get_zone_info: {(t2-t1)*1000:.1f}ms")

    # Update borehole position using pre-computed BNG coords
    # Returns updated zone_ids for the borehole
    t1 = time.perf_counter()
    old_zone_ids = data_loader.get_borehole_zone_ids(index)  # Get BEFORE update
    zone_ids = data_loader.update_borehole_position(
        index, lon, lat, bng_coords, exclude_zones
    )
    t2 = time.perf_counter()
    debug_log(f"    [7] update_borehole_position: {(t2-t1)*1000:.1f}ms")

    # Detect zone change: inside zone (black) <-> outside zone (orange)
    was_in_zone = len(old_zone_ids) > 0
    is_in_zone = len(zone_ids) > 0
    if was_in_zone != is_in_zone:
        color_change = "BLACK->ORANGE" if was_in_zone else "ORANGE->BLACK"
        debug_log(
            f"    [ZONE CHANGE] {color_change} | old_zones={old_zone_ids} -> new_zones={zone_ids}"
        )
    else:
        debug_log(f"    [NO ZONE CHANGE] zones={zone_ids}")

    # DON'T compute stats - let frontend fetch them lazily
    t1 = time.perf_counter()
    result = {
        "coverage": coverage,
        "zone_info": zone_info,
        "zone_ids": zone_ids,  # New: zone associations for this borehole
        "stats_pending": True,
    }
    t2 = time.perf_counter()
    debug_log(f"    [8] Build result dict: {(t2-t1)*1000:.3f}ms")

    # Log coverage result
    coverage_type = (
        "null"
        if coverage is None
        else ("polygon" if coverage.get("geometry") else "empty")
    )
    debug_log(f"    [COVERAGE] type={coverage_type}, zone_info={zone_info}")

    t1 = time.perf_counter()
    response = jsonify(result)
    t2 = time.perf_counter()
    debug_log(f"    [9] jsonify: {(t2-t1)*1000:.1f}ms")

    t_end = time.perf_counter()
    debug_log(f"    [TOTAL] MOVE: {(t_end-t_start)*1000:.1f}ms")
    debug_log("=" * 60)

    return response


@app.route("/api/borehole/delete", methods=["POST"])
def delete_borehole() -> Dict[str, Any]:
    """
    Delete a borehole by index. Returns immediately without stats (lazy update pattern).

    Request Body:
        {
            "index": int   # Borehole index (0-based)
        }

    Returns:
        {
            "success": bool,
            "boreholes": GeoJSON FeatureCollection (updated list),
            "deleted_id": str (ID of deleted borehole),
            "stats_pending": bool (true = frontend should request stats separately)
        }
    """
    import time

    debug_log("=" * 60)
    debug_log("[DELETE] Request received")

    t_start = time.perf_counter()

    if data_loader is None or coverage_service is None:
        return jsonify({"error": "Server not initialized"}), 500

    t1 = time.perf_counter()
    data = request.get_json()
    t2 = time.perf_counter()
    debug_log(f"    [1] get_json(): {(t2-t1)*1000:.3f}ms")

    if not data or "index" not in data:
        return jsonify({"error": "Missing index in request body"}), 400

    index = int(data["index"])
    debug_log(f"    [2] Deleting borehole at index {index}")

    # Delete and get the borehole ID
    t1 = time.perf_counter()
    deleted_id = data_loader.delete_borehole(index)
    t2 = time.perf_counter()
    debug_log(
        f"    [3] delete_borehole(): {(t2-t1)*1000:.3f}ms -> deleted_id={deleted_id}"
    )

    if deleted_id:
        # Invalidate cache for deleted borehole (marks stats as dirty)
        t1 = time.perf_counter()
        coverage_service.invalidate_cache(deleted_id)
        t2 = time.perf_counter()
        debug_log(f"    [4] invalidate_cache(): {(t2-t1)*1000:.3f}ms")

        # Get updated boreholes
        t1 = time.perf_counter()
        boreholes = data_loader.get_boreholes_geojson()
        t2 = time.perf_counter()
        debug_log(f"    [5] get_boreholes_geojson(): {(t2-t1)*1000:.3f}ms")

        # DON'T compute stats - let frontend fetch them lazily
        # This makes delete feel instant (~5ms vs ~230ms)

        t1 = time.perf_counter()
        result = jsonify(
            {
                "success": True,
                "boreholes": boreholes,
                "deleted_id": deleted_id,
                "stats_pending": True,  # Signal frontend to fetch stats separately
            }
        )
        t2 = time.perf_counter()
        debug_log(f"    [6] jsonify: {(t2-t1)*1000:.3f}ms")

        t_end = time.perf_counter()
        debug_log(f"    [TOTAL] DELETE: {(t_end-t_start)*1000:.1f}ms")
        debug_log("=" * 60)

        return result
    else:
        return jsonify({"success": False, "error": "Invalid index"}), 400


@app.route("/api/boreholes/delete-outside", methods=["POST"])
def delete_outside_boreholes() -> Dict[str, Any]:
    """
    Delete all boreholes outside every zone (SSOT: server identifies via zone_ids).

    Returns:
        {
            "success": bool,
            "deleted_count": int,
            "deleted_ids": list[str],
            "boreholes": GeoJSON FeatureCollection (updated list),
            "stats_pending": bool
        }
    """
    if data_loader is None or coverage_service is None:
        return jsonify({"error": "Server not initialized"}), 500

    debug_log("=" * 60)
    debug_log("[DELETE-OUTSIDE] Request received")

    # Server determines which boreholes are outside (SSOT via zone_ids)
    deleted_ids = data_loader.delete_outside_boreholes()

    # Invalidate cache for all deleted boreholes
    for bid in deleted_ids:
        coverage_service.invalidate_cache(bid)

    boreholes = data_loader.get_boreholes_geojson()

    debug_log(
        f"    [DELETE-OUTSIDE] Deleted {len(deleted_ids)} boreholes: {deleted_ids}"
    )
    debug_log("=" * 60)

    return jsonify(
        {
            "success": True,
            "deleted_count": len(deleted_ids),
            "deleted_ids": deleted_ids,
            "boreholes": boreholes,
            "stats_pending": True,
        }
    )


@app.route("/api/borehole/add", methods=["POST"])
def add_borehole() -> Dict[str, Any]:
    """
    Add a new borehole at the specified location. Returns stats (delta update pattern).

    Request Body:
        {
            "lon": float,        # Longitude (WGS84)
            "lat": float,        # Latitude (WGS84)
            "location_id": str   # Optional location ID
        }

    Returns:
        {
            "success": bool,
            "index": int (new borehole index),
            "borehole_id": str (ID of added borehole),
            "coverage": GeoJSON Feature with coverage polygon,
            "boreholes": GeoJSON FeatureCollection (updated list),
            "stats": coverage statistics (avoids separate call)
        }
    """
    if coverage_service is None or data_loader is None:
        return jsonify({"error": "Server not initialized"}), 500

    data = request.get_json()
    if not data or "lon" not in data or "lat" not in data:
        return jsonify({"error": "Missing lon/lat in request body"}), 400

    lon = float(data["lon"])
    lat = float(data["lat"])
    location_id = data.get("location_id")
    exclude_zones = data.get(
        "excludeZones", []
    )  # R2: Hidden zones to exclude from zone_ids

    # Add the borehole (with exclude_zones for R2)
    new_index = data_loader.add_borehole(lon, lat, location_id, exclude_zones)

    # Get the borehole ID for caching
    borehole_id = data_loader.get_borehole_id(new_index)

    # Compute coverage using cache (adds to cache)
    coverage = coverage_service.compute_coverage_cached(
        borehole_id, lon, lat, exclude_zones=exclude_zones
    )

    # Get updated boreholes list
    boreholes = data_loader.get_boreholes_geojson()

    # Get stats from cache (includes new coverage)
    stats = coverage_service.get_stats_from_cache()

    return jsonify(
        {
            "success": True,
            "index": new_index,
            "borehole_id": borehole_id,
            "coverage": coverage,
            "boreholes": boreholes,
            "stats": stats,
        }
    )


@app.route("/api/borehole/restore", methods=["POST"])
def restore_boreholes() -> Dict[str, Any]:
    """
    Restore boreholes state from a GeoJSON snapshot (for undo).

    Request Body:
        {
            "boreholes": GeoJSON FeatureCollection to restore
        }

    Returns:
        {
            "success": bool
        }
    """
    if data_loader is None:
        return jsonify({"error": "Server not initialized"}), 500

    data = request.get_json()
    if not data or "boreholes" not in data:
        return jsonify({"error": "Missing boreholes in request body"}), 400

    boreholes_geojson = data["boreholes"]
    data_loader.restore_state(boreholes_geojson)

    # Clear coverage cache so stats recompute from restored boreholes only
    if coverage_service is not None:
        coverage_service.clear_cache()

    return jsonify({"success": True})


@app.route("/api/boreholes/export", methods=["GET"])
def export_boreholes_csv() -> Response:
    """
    Export current borehole positions as CSV with Easting/Northing coordinates.

    Query Parameters:
        excludeIndices: Comma-separated list of borehole indices to exclude (optional)

    Returns:
        CSV file download with columns: ID, Easting, Northing, Zone
    """
    if data_loader is None:
        return jsonify({"error": "Server not initialized"}), 500

    # Parse excludeIndices query parameter (comma-separated integers)
    exclude_indices_param = request.args.get("excludeIndices", "")
    exclude_indices: set[int] = set()
    if exclude_indices_param:
        try:
            exclude_indices = {
                int(idx.strip())
                for idx in exclude_indices_param.split(",")
                if idx.strip()
            }
        except ValueError:
            pass  # Ignore invalid indices, export all

    boreholes_geojson = data_loader.get_boreholes_geojson()

    # Create transformer: WGS84 (lat/lng) -> British National Grid (easting/northing)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

    # Build CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Easting", "Northing", "Zone"])

    for idx, feature in enumerate(boreholes_geojson.get("features", [])):
        # Skip excluded indices
        if idx in exclude_indices:
            continue

        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [0, 0])

        # coords are [lon, lat] in WGS84
        lon, lat = coords[0], coords[1]
        easting, northing = transformer.transform(lon, lat)

        borehole_id = props.get("location_id", "") or props.get("id", "")
        # zone_ids is the SSOT list of zone names from the server
        zone_ids = props.get("zone_ids", [])
        zone = ", ".join(zone_ids) if zone_ids else ""

        writer.writerow([borehole_id, f"{easting:.2f}", f"{northing:.2f}", zone])

    # Return as downloadable CSV
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=boreholes_export.csv"},
    )


@app.route("/api/coverage/all", methods=["GET"])
def compute_all_coverages() -> Dict[str, Any]:
    """
    Compute coverage polygons for all boreholes at their current positions.
    Uses caching for performance - initializes cache on first call.

    Returns:
        {
            "type": "FeatureCollection",
            "features": [...coverage polygons...],
            "stats": coverage statistics (avoids separate /api/coverage/stats call)
        }
    """
    if coverage_service is None or data_loader is None:
        return jsonify({"error": "Server not initialized"}), 500

    boreholes = data_loader.get_boreholes_geojson()

    # Compute all coverages using cache (populates cache on first call)
    # Method extracts borehole IDs from the GeoJSON
    coverages_geojson = coverage_service.compute_all_coverages_and_cache(boreholes)

    # Get stats from cache (fast - already computed)
    stats = coverage_service.get_stats_from_cache()

    # Add stats to the response
    coverages_geojson["stats"] = stats

    return jsonify(coverages_geojson)


@app.route("/api/coverage/filtered", methods=["POST"])
def compute_filtered_coverages() -> Dict[str, Any]:
    """
    Compute coverage polygons with zone visibility filtering.

    Supports two modes (SSOT - server decides what to return):
    - "clip_coverage": Clips coverage polygons to visible zones only
    - "hide_zone_boreholes": Excludes entire coverage for boreholes inside hidden zones

    Request Body:
        {
            "excludeZones": ["ZoneName1", "ZoneName2", ...],
            "mode": "clip_coverage" | "hide_zone_boreholes"  # Optional, default: "clip_coverage"
        }

    Returns:
        {
            "type": "FeatureCollection",
            "features": [...filtered coverage polygons...]
        }
    """
    if coverage_service is None or data_loader is None:
        return jsonify({"error": "Server not initialized"}), 500

    data = request.get_json()
    exclude_zones = data.get("excludeZones", []) if data else []
    mode = data.get("mode", "clip_coverage") if data else "clip_coverage"

    boreholes = data_loader.get_boreholes_geojson()

    logger.info(f"[COVERAGE FILTER] mode={mode}, exclude_zones={exclude_zones}")
    logger.info(
        f"[COVERAGE FILTER] Total boreholes: {len(boreholes.get('features', []))}"
    )

    if mode == "hide_zone_boreholes":
        # Mode: Hide boreholes INSIDE hidden zones + clip coverage from remaining boreholes
        # 1. Filter boreholes to only those NOT inside any hidden zone
        # 2. For remaining boreholes, clip their coverage to exclude hidden zone areas
        excluded_set = set(exclude_zones)

        # Debug: Log first few boreholes to see their zone_ids
        for i, f in enumerate(boreholes.get("features", [])[:3]):
            zids = f.get("properties", {}).get("zone_ids", [])
            logger.info(f"[COVERAGE FILTER] Borehole {i} zone_ids: {zids}")

        filtered_boreholes = {
            "type": "FeatureCollection",
            "features": [
                f
                for f in boreholes.get("features", [])
                if not any(
                    zid in excluded_set
                    for zid in f.get("properties", {}).get("zone_ids", [])
                )
            ],
        }

        logger.info(
            f"[COVERAGE FILTER] After filtering: {len(filtered_boreholes.get('features', []))} boreholes"
        )

        # Compute CLIPPED coverages for visible boreholes (excludes hidden zone areas)
        # This ensures coverage from outside-zone boreholes doesn't show in hidden zones
        coverages_geojson = coverage_service.compute_all_coverages_filtered(
            filtered_boreholes, exclude_zones
        )
    else:
        # Mode: clip_coverage (default) - Clip coverage polygons to exclude hidden zones
        coverages_geojson = coverage_service.compute_all_coverages_filtered(
            boreholes, exclude_zones
        )

    return jsonify(coverages_geojson)


@app.route("/api/coverage/stats", methods=["GET"])
def get_coverage_stats() -> Dict[str, Any]:
    """
    Get coverage statistics (% area covered) for all zones.

    Returns:
        {
            "per_zone": [{zone_name, total_area_m2, covered_area_m2, coverage_pct}],
            "total": {total_area_m2, covered_area_m2, coverage_pct}
        }
    """
    if coverage_service is None or data_loader is None:
        return jsonify({"error": "Server not initialized"}), 500

    # Get current borehole positions
    boreholes = data_loader.get_boreholes_geojson()
    positions = []

    for feature in boreholes.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates", [])
        if len(coords) >= 2:
            positions.append((coords[0], coords[1]))  # (lon, lat)

    # Compute stats
    stats = coverage_service.compute_coverage_stats(positions)

    return jsonify(stats)


@app.route("/api/existing-coverage", methods=["GET"])
def get_existing_coverage() -> Dict[str, Any]:
    """
    Get existing borehole coverage polygons (from main.py output).

    Returns:
        GeoJSON FeatureCollection with existing coverage polygons,
        or empty FeatureCollection if not available.
    """
    if data_loader is None:
        return jsonify({"error": "Server not initialized"}), 500

    existing = data_loader.get_existing_coverage_geojson()
    logger.info(
        f"📊 Existing coverage: {type(existing)}, features: {len(existing.get('features', [])) if existing else 0}"
    )

    if existing is None:
        return jsonify({"type": "FeatureCollection", "features": []})

    return jsonify(existing)


@app.route("/api/existing-coverage/filtered", methods=["POST"])
def get_existing_coverage_filtered() -> Dict[str, Any]:
    """
    Get existing coverage clipped to visible zones (SSOT compliant).

    Server clips existing coverage polygon to exclude hidden zones.
    Same pattern as /api/coverage/filtered for proposed coverage.

    Request Body:
        {
            "excludeZones": ["ZoneName1", "ZoneName2", ...]
        }

    Returns:
        GeoJSON FeatureCollection with clipped existing coverage polygons.
    """
    if data_loader is None or coverage_service is None:
        return jsonify({"error": "Server not initialized"}), 500

    data = request.get_json()
    exclude_zones = data.get("excludeZones", []) if data else []

    existing = data_loader.get_existing_coverage_geojson()

    if existing is None or not existing.get("features"):
        return jsonify({"type": "FeatureCollection", "features": []})

    # If no zones excluded, return full existing coverage
    if not exclude_zones:
        return jsonify(existing)

    # Clip existing coverage to visible zones
    clipped = coverage_service.clip_coverage_to_visible_zones(existing, exclude_zones)
    logger.info(
        f"📊 Existing coverage filtered: excluded {exclude_zones}, "
        f"features: {len(clipped.get('features', []))}"
    )

    return jsonify(clipped)


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

    # Very visible startup message
    logger.info("=" * 70)
    logger.info("    ZONE COVERAGE VIZ SERVER STARTED - DEBUG LOGGING ACTIVE")
    logger.info("    All move/delete operations will be logged here")
    logger.info("=" * 70)

    # Disable debug mode for frozen exe (reloader doesn't work with PyInstaller)
    is_frozen = getattr(sys, "frozen", False)
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=not is_frozen)


if __name__ == "__main__":
    main()
