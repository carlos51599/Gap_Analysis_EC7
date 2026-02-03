#!/usr/bin/env python3
"""
Zone-Aware Coverage Visualization Entry Point

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURAL OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

Responsibility: Generate an interactive HTML visualization showing zone-aware
borehole coverage with variable radii per zone.

This is a standalone entry point that reuses data loading from the main
Gap_Analysis_EC7 module.

Usage:
    python main_zone_coverage.py

    Or from Embankment_Grid folder:
    python -m Gap_Analysis_EC7.main_zone_coverage

Output:
    Gap_Analysis_EC7/Output/zone_coverage.html

For Navigation: Use VS Code outline (Ctrl+Shift+O)

═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Workspace root for resolving config paths
WORKSPACE_ROOT = Path(__file__).parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# 📋 LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════


def setup_logging() -> logging.Logger:
    """Configure logging with console output.

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("ZoneCoverage")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# 📂 DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def resolve_path(relative_path: str) -> str:
    """Resolve a config path relative to workspace root."""
    return str(WORKSPACE_ROOT / relative_path)


def load_zones(logger: logging.Logger) -> gpd.GeoDataFrame:
    """Load zone polygons using the main module's data loading.

    This reuses the same shapefile configuration as the main Gap_Analysis_EC7
    module to ensure consistency.

    Args:
        logger: Logger instance

    Returns:
        GeoDataFrame with zone polygons and max_spacing_m
    """
    from Gap_Analysis_EC7.shapefile_config import (
        get_enabled_layers,
        get_coverage_layer_keys,
    )

    # Import WORKSPACE_ROOT from main to ensure consistent path resolution
    from Gap_Analysis_EC7.main import (
        load_all_shapefiles,
        get_zones_for_coverage_gdf,
        WORKSPACE_ROOT as MAIN_WORKSPACE_ROOT,
    )

    # Update module-level WORKSPACE_ROOT to match main.py
    global WORKSPACE_ROOT
    WORKSPACE_ROOT = MAIN_WORKSPACE_ROOT

    logger.info("📂 Loading zone shapefiles...")

    # Load all shapefiles using main module's function
    all_shapefiles = load_all_shapefiles(logger)

    # Get unified zones GeoDataFrame
    zones_gdf = get_zones_for_coverage_gdf(all_shapefiles, logger)

    if zones_gdf is None or zones_gdf.empty:
        raise ValueError("No zones loaded. Check shapefile configuration.")

    logger.info(f"   ✅ Loaded {len(zones_gdf)} zone(s)")

    return zones_gdf


def load_boreholes(logger: logging.Logger) -> gpd.GeoDataFrame:
    """Load borehole locations from CSV.

    Uses the same CSV file as the main Gap_Analysis_EC7 module.

    Args:
        logger: Logger instance

    Returns:
        GeoDataFrame with borehole points
    """
    from Gap_Analysis_EC7.config import CONFIG
    from Gap_Analysis_EC7.config_types import AppConfig
    from Gap_Analysis_EC7.main import WORKSPACE_ROOT as MAIN_WORKSPACE_ROOT

    app_config = AppConfig.from_dict(CONFIG)
    csv_path = app_config.file_paths.boreholes_csv

    # Use workspace root from main.py for consistent path resolution
    full_path = str(MAIN_WORKSPACE_ROOT / csv_path)
    logger.info(f"📂 Loading boreholes: {csv_path}")

    if not Path(full_path).exists():
        raise FileNotFoundError(f"CSV not found: {full_path}")

    df = pd.read_csv(full_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Check required columns
    required = ["Easting", "Northing", "Location ID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Drop rows with missing coordinates
    df = df.dropna(subset=["Easting", "Northing"])

    # Rename Location ID to LocationID for consistency
    if "Location ID" in df.columns:
        df = df.rename(columns={"Location ID": "LocationID"})

    # Create geometry
    geometry = [Point(x, y) for x, y in zip(df["Easting"], df["Northing"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:27700")

    logger.info(f"   ✅ Loaded {len(gdf)} boreholes")

    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point for zone coverage visualization."""
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("🚀 ZONE-AWARE COVERAGE VISUALIZATION")
    logger.info("=" * 60)
    logger.info(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    try:
        # === LOAD DATA ===
        zones_gdf = load_zones(logger)
        boreholes_gdf = load_boreholes(logger)

        # === GENERATE VISUALIZATION ===
        from zone_coverage_viz import generate_zone_coverage_html, ZoneCoverageConfig

        # Configure visualization
        config = ZoneCoverageConfig(
            title="Zone-Aware Borehole Coverage",
            default_zoom=13.0,
            enable_drag=True,
            show_zone_labels=True,
            pre_compute_coverage=True,
            use_cdn_libs=True,  # Use CDN for smaller file
        )

        # Output path
        output_dir = Path(__file__).parent / "Output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "zone_coverage.html"

        # Generate HTML
        logger.info("")
        result_path = generate_zone_coverage_html(
            zones_gdf=zones_gdf,
            boreholes_gdf=boreholes_gdf,
            output_path=str(output_path),
            config=config,
            id_field="LocationID",
        )

        # === SUCCESS ===
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ VISUALIZATION COMPLETE")
        logger.info(f"📄 Output: {result_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
