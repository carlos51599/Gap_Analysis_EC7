"""
deck.gl HTML builder - main orchestrator.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURAL OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

Responsibility: Orchestrate the zone coverage visualization HTML generation.
This is the main entry point for the module.

Key Features:
- Loads zone and borehole data
- Pre-computes initial coverage geometries
- Bundles JavaScript libraries and templates
- Generates standalone HTML output

Usage:
    from zone_coverage_viz import generate_zone_coverage_html

    html_path = generate_zone_coverage_html(
        zones_gdf=zones,
        boreholes_gdf=boreholes,
        output_path="output/zone_coverage.html"
    )

For Navigation: Use VS Code outline (Ctrl+Shift+O)

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd

from .config_types import ZoneCoverageConfig
from .zone_geometry import (
    prepare_zone_geojson,
    prepare_borehole_geojson,
    zones_to_json_string,
    boreholes_to_json_string,
    positions_to_json_string,
)
from .coverage_calculator import (
    compute_all_coverage,
    coverage_to_json_string,
)
from .html_template import generate_html, load_js_template

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 🏗️ MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════


def generate_zone_coverage_html(
    zones_gdf: gpd.GeoDataFrame,
    boreholes_gdf: gpd.GeoDataFrame,
    output_path: str = "zone_coverage.html",
    config: Optional[ZoneCoverageConfig] = None,
    id_field: str = "LocationID",
) -> str:
    """
    Generate standalone HTML visualization for zone-aware borehole coverage.

    Creates an interactive deck.gl-based visualization with:
    - Zone boundaries with configurable max_spacing per zone
    - Draggable borehole markers
    - Real-time zone-aware coverage updates
    - Export modified positions to CSV

    Args:
        zones_gdf: GeoDataFrame with zone polygons. Must have:
            - geometry: Polygon geometries
            - max_spacing_m (optional): Per-zone max spacing in meters
        boreholes_gdf: GeoDataFrame with borehole points. Must have:
            - geometry: Point geometries
            - id_field column (optional): Unique borehole identifiers
        output_path: Path for output HTML file
        config: Optional ZoneCoverageConfig for styling/behavior
        id_field: Column name for borehole IDs

    Returns:
        Absolute path to generated HTML file

    Raises:
        ValueError: If input GeoDataFrames are empty or invalid
        FileNotFoundError: If JS templates are missing
    """
    logger.info("=" * 60)
    logger.info("🚀 GENERATING ZONE COVERAGE VISUALIZATION")
    logger.info("=" * 60)

    # === VALIDATE INPUTS ===
    if zones_gdf is None or zones_gdf.empty:
        raise ValueError("zones_gdf cannot be None or empty")
    if boreholes_gdf is None or boreholes_gdf.empty:
        raise ValueError("boreholes_gdf cannot be None or empty")

    # === USE DEFAULT CONFIG IF NOT PROVIDED ===
    if config is None:
        config = ZoneCoverageConfig()

    # === PREPARE ZONE DATA ===
    logger.info("📍 Preparing zone data...")
    zone_data = prepare_zone_geojson(
        zones_gdf=zones_gdf,
        max_spacing_field=config.max_spacing_field,
        default_max_spacing=config.default_max_spacing,
    )
    zones_json = zones_to_json_string(zone_data)

    # === PREPARE BOREHOLE DATA ===
    logger.info("📍 Preparing borehole data...")
    borehole_data = prepare_borehole_geojson(
        boreholes_gdf=boreholes_gdf,
        id_field=id_field,
    )
    boreholes_json = boreholes_to_json_string(borehole_data)
    positions_json = positions_to_json_string(borehole_data)

    # === PRE-COMPUTE COVERAGE ===
    coverage_json = "{}"
    if config.pre_compute_coverage:
        logger.info("📐 Pre-computing coverage geometries...")
        coverage_dict = compute_all_coverage(
            boreholes_gdf=boreholes_gdf,
            zones_gdf=zones_gdf,
            max_spacing_field=config.max_spacing_field,
            default_max_spacing=config.default_max_spacing,
            id_field=id_field,
        )
        coverage_json = json.dumps(coverage_dict, separators=(",", ":"))

    # === LOAD JAVASCRIPT TEMPLATES ===
    logger.info("📜 Loading JavaScript templates...")
    try:
        main_app_code = load_js_template("main_app.js")
        worker_code = load_js_template("geometry_worker.js")
        ui_code = load_js_template("ui_controls.js")
    except FileNotFoundError as e:
        logger.error(f"❌ Failed to load JS template: {e}")
        raise

    # === GENERATE HTML ===
    logger.info("📄 Generating HTML...")
    html_content = generate_html(
        zones_json=zones_json,
        boreholes_json=boreholes_json,
        positions_json=positions_json,
        coverage_json=coverage_json,
        config=config,
        main_app_code=main_app_code,
        worker_code=worker_code,
        ui_code=ui_code,
        center=zone_data.center,
    )

    # === WRITE OUTPUT FILE ===
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(html_content, encoding="utf-8")

    file_size_kb = len(html_content) / 1024
    logger.info(f"   ✅ Generated HTML: {output_path}")
    logger.info(f"   📊 File size: {file_size_kb:.1f} KB")

    # === SUMMARY ===
    logger.info("=" * 60)
    logger.info("✅ ZONE COVERAGE VISUALIZATION COMPLETE")
    logger.info(f"   📍 Zones: {zone_data.zone_count}")
    logger.info(f"   📍 Boreholes: {borehole_data.borehole_count}")
    logger.info(f"   📄 Output: {output_path.absolute()}")
    logger.info("=" * 60)

    return str(output_path.absolute())


# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def generate_from_files(
    zones_shapefile: str,
    boreholes_shapefile: str,
    output_path: str = "zone_coverage.html",
    config: Optional[ZoneCoverageConfig] = None,
    id_field: str = "LocationID",
) -> str:
    """
    Generate visualization from shapefile paths.

    Convenience function that loads shapefiles and generates HTML.

    Args:
        zones_shapefile: Path to zones shapefile
        boreholes_shapefile: Path to boreholes shapefile
        output_path: Path for output HTML
        config: Optional configuration
        id_field: Borehole ID column name

    Returns:
        Absolute path to generated HTML
    """
    logger.info(f"📂 Loading zones from: {zones_shapefile}")
    zones_gdf = gpd.read_file(zones_shapefile)

    logger.info(f"📂 Loading boreholes from: {boreholes_shapefile}")
    boreholes_gdf = gpd.read_file(boreholes_shapefile)

    return generate_zone_coverage_html(
        zones_gdf=zones_gdf,
        boreholes_gdf=boreholes_gdf,
        output_path=output_path,
        config=config,
        id_field=id_field,
    )
