#!/usr/bin/env python3
"""
EC7 Simple Gap Analysis - Main Entry Point

Ultra-simplified EC7-compliant borehole spacing analysis.
Single site-wide spacing, single grid generator, minimal code.

Usage:
    python main.py

    Or from Embankment_Grid folder:
    python -m Gap_Analysis_EC7.main
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Set

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Add parent directory to path for imports
# Structure: Embankment_Grid / {Main|Worktree} / Gap_Analysis_EC7 / main.py
# Go up 2 levels to reach {Main|Worktree} for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Workspace root for resolving config paths (shared resources)
# Go up 3 levels to reach Embankment_Grid where shared resources live
WORKSPACE_ROOT = Path(__file__).parent.parent.parent

from Gap_Analysis_EC7.config import CONFIG
from Gap_Analysis_EC7.config_types import AppConfig, VisualizationConfig
from Gap_Analysis_EC7.shapefile_config import (
    get_enabled_layers,
    get_coverage_layer_keys,
    get_layer_config,
    get_layer_name_column,
    get_zone_max_spacing,
    get_zone_order,
    make_zone_id,
    build_zones_config_for_visualization,
)
from Gap_Analysis_EC7.zone_preprocessor import preprocess_zones
from Gap_Analysis_EC7.visualization.html_builder import generate_multi_layer_html
from Gap_Analysis_EC7.coverage_zones import compute_coverage_zones, get_coverage_summary

# ═══════════════════════════════════════════════════════════════════════════
# 🎯 MODULE-LEVEL CONFIG (Single Source of Truth)
# ═══════════════════════════════════════════════════════════════════════════
# Create typed AppConfig once at module load time.
# This provides typed access throughout the module.
APP_CONFIG = AppConfig.from_dict(CONFIG)


def resolve_path(relative_path: str) -> str:
    """Resolve a config path relative to workspace root."""
    return str(WORKSPACE_ROOT / relative_path)


# ═══════════════════════════════════════════════════════════════════════════
# 📋 LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════


def _get_testing_filter_combo_key() -> str:
    """Generate combo key from testing mode filter settings.

    Format: d{depth}stp{0|1}txt{0|1}txe{0|1} (no underscores for brevity)

    Returns:
        Compact combo key string (e.g., "d45spt0txt0txe0")
    """
    f = APP_CONFIG.testing_mode.filter
    return (
        f"d{f.min_depth}"
        f"spt{int(f.require_spt)}"
        f"txt{int(f.require_triaxial_total)}"
        f"txe{int(f.require_triaxial_effective)}"
    )


def setup_logging() -> Tuple[logging.Logger, Path]:
    """Configure logging with file and console handlers.

    Returns:
        Tuple of (logger, run_log_folder) where run_log_folder is the path
        to store all logs for this run (main log + HiGHS solver output).

    Folder naming convention:
        - Testing mode: testing_{combo_key}_{MMDD}_{HHMM}
          e.g., testing_d45spt0txt0txe0_0129_1028
        - Production mode: production_{MMDD}_{HHMM}
          e.g., production_0129_1028
    """
    log_dir = APP_CONFIG.file_paths.log_dir_path(WORKSPACE_ROOT)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Compact timestamp: MMDD_HHMM
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # Build folder name based on mode
    if APP_CONFIG.testing_mode.enabled:
        combo_key = _get_testing_filter_combo_key()
        folder_name = f"testing_{combo_key}_{timestamp}"
    else:
        folder_name = f"production_{timestamp}"

    # Create run-specific log folder for all logs (main + HiGHS)
    run_log_folder = log_dir / folder_name
    run_log_folder.mkdir(parents=True, exist_ok=True)

    # Main log saved inside the run folder
    log_path = run_log_folder / "main.log"

    logger = logging.getLogger("EC7Simple")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, run_log_folder


# ═══════════════════════════════════════════════════════════════════════════
# 📂 DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════


def load_shapefile(
    shapefile_path: str, logger: logging.Logger, name: str = "shapefile"
) -> gpd.GeoDataFrame:
    """Load shapefile and ensure correct CRS."""
    # Resolve path relative to workspace root
    full_path = resolve_path(shapefile_path)
    logger.info(f"📂 Loading {name}: {shapefile_path}")

    if not Path(full_path).exists():
        raise FileNotFoundError(f"Shapefile not found: {full_path}")

    # Note: Removed SHAPE_RESTORE_SHX=YES as it was corrupting valid .shx files
    gdf = gpd.read_file(full_path)

    # Ensure CRS is British National Grid
    target_crs = APP_CONFIG.target_crs
    if gdf.crs is None:
        gdf.set_crs(target_crs, inplace=True)
    elif gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)

    logger.info(f"   ✅ Loaded {len(gdf)} features")
    return gdf


def load_zones(shapefile_path: str, logger: logging.Logger) -> gpd.GeoDataFrame:
    """Load zone boundaries from shapefile."""
    return load_shapefile(shapefile_path, logger, name="zones")


def load_all_shapefiles(logger: logging.Logger) -> Dict[str, gpd.GeoDataFrame]:
    """
    Load all enabled shapefiles from SHAPEFILE_CONFIG.

    Returns:
        Dict mapping layer key to GeoDataFrame, sorted by layer_order.
    """
    loaded = {}
    enabled_layers = get_enabled_layers()

    for layer_key, layer_config in enabled_layers.items():
        file_path = layer_config.get("file_path")
        display_name = layer_config.get("display_name", layer_key)

        if not file_path:
            logger.warning(f"   ⚠️ No file_path for layer: {layer_key}")
            continue

        try:
            gdf = load_shapefile(file_path, logger, name=display_name)
            if gdf is not None and not gdf.empty:
                # Add source column for traceability
                gdf["_shapefile_source"] = layer_key
                loaded[layer_key] = gdf
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to load {layer_key}: {e}")

    return loaded


def get_zones_for_coverage_gdf(
    all_shapefiles: Dict[str, gpd.GeoDataFrame],
    logger: Optional[logging.Logger] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Merge all coverage shapefiles into unified zones GeoDataFrame.

    Handles:
    - Multiple use_for_coverage=True layers
    - Dynamic zone naming (Name column or display_name)
    - Per-zone max_spacing_m resolution
    - Zone ID prefixing for disambiguation

    Args:
        all_shapefiles: Dict of loaded shapefiles keyed by layer name
        logger: Optional logger instance

    Returns:
        GeoDataFrame with columns:
        - geometry: Zone boundary polygon
        - zone_id: Unique ID "layer_key:zone_name"
        - zone_name: Display name for UI
        - max_spacing_m: Zone-specific spacing requirement
        - source_layer: Layer key this zone came from
        - original_name: Original Name column value (or None)

    Returns None if no coverage layers defined or loaded.
    """
    coverage_keys = get_coverage_layer_keys()
    if not coverage_keys:
        if logger:
            logger.warning("⚠️ No coverage layers defined (use_for_coverage=True)")
        return None

    merged_zones = []
    reference_crs = None

    for layer_key in coverage_keys:
        if layer_key not in all_shapefiles:
            if logger:
                logger.warning(f"   ⚠️ Coverage layer '{layer_key}' not loaded")
            continue

        gdf = all_shapefiles[layer_key].copy()
        config = get_layer_config(layer_key)
        name_col = get_layer_name_column(layer_key)

        # Store reference CRS from first layer
        if reference_crs is None:
            reference_crs = gdf.crs

        # === ZONE NAME RESOLUTION ===
        if name_col and name_col in gdf.columns:
            # Use Name column values
            gdf["zone_name"] = gdf[name_col].astype(str)
            gdf["original_name"] = gdf[name_col].astype(str)
        else:
            # No Name column → generate unique zone names per feature
            # Each feature becomes a separate zone for ILP, identified by index
            display_name = config.get("display_name", layer_key)
            gdf["zone_name"] = [f"{display_name}_{i}" for i in range(len(gdf))]
            gdf["original_name"] = display_name  # Track the layer display name

        # === ZONE ID (Unique across layers) ===
        # Use default parameter binding to capture layer_key value
        current_layer = layer_key  # Capture current value
        gdf["zone_id"] = gdf["zone_name"].apply(
            lambda zn, lk=current_layer: make_zone_id(lk, zn)
        )

        # === MAX SPACING RESOLUTION ===
        gdf["max_spacing_m"] = gdf["zone_name"].apply(
            lambda zn, lk=current_layer: get_zone_max_spacing(lk, zn)
        )

        # === ORDER RESOLUTION (for overlap priority) ===
        gdf["order"] = gdf["zone_name"].apply(
            lambda zn, lk=current_layer: get_zone_order(lk, zn)
        )

        # === SOURCE TRACKING ===
        gdf["source_layer"] = layer_key

        # === FILTER ENABLED ZONES ===
        features = config.get("features", {})
        if features:
            # Determine which zones are enabled
            enabled_zones = set()
            disabled_zones = set()

            for feat_name, feat_cfg in features.items():
                if feat_cfg.get("enabled", True):
                    enabled_zones.add(feat_name)
                else:
                    disabled_zones.add(feat_name)

            # Zones not in features config are enabled by default
            all_zone_names = set(gdf["zone_name"].unique())
            unconfigured = all_zone_names - set(features.keys())
            enabled_zones = enabled_zones | unconfigured

            # Filter to enabled only
            gdf = gdf[gdf["zone_name"].isin(enabled_zones)]

            if logger and disabled_zones:
                logger.info(f"   🚫 {layer_key}: Disabled zones: {disabled_zones}")

        if not gdf.empty:
            # Select and reorder columns
            result_cols = [
                "geometry",
                "zone_id",
                "zone_name",
                "max_spacing_m",
                "order",
                "source_layer",
                "original_name",
            ]
            merged_zones.append(gdf[result_cols])

            if logger:
                logger.info(f"   📍 {layer_key}: {len(gdf)} zone(s) added")

    if not merged_zones:
        if logger:
            logger.warning("   ⚠️ No zones loaded from coverage layers")
        return None

    # === CONCATENATE ALL COVERAGE ZONES ===
    result = pd.concat(merged_zones, ignore_index=True)
    result = gpd.GeoDataFrame(result, crs=reference_crs)

    # === ZONE PREPROCESSING (Cut overlapping geometries) ===
    # Higher-priority zones claim overlapping regions from lower-priority zones.
    # Priority: 1. Lower max_spacing_m wins, 2. Lower order wins (if spacing equal)
    result = preprocess_zones(result, logger=logger)

    # Swap geometry with effective_geometry for downstream processing
    if "effective_geometry" in result.columns:
        result["geometry"] = result["effective_geometry"]
        result = gpd.GeoDataFrame(result, geometry="geometry", crs=reference_crs)

    if logger:
        logger.info(f"   ✅ Total coverage zones: {len(result)}")
        for _, row in result.iterrows():
            cut_info = ""
            if row.get("was_cut", False):
                cut_info = f" (cut by: {', '.join(row.get('cut_by', []))})"
            logger.info(
                f"      • {row['zone_name']}: max_spacing={row['max_spacing_m']}m "
                f"(from {row['source_layer']}){cut_info}"
            )

    return result


def load_boreholes(csv_path: str, logger: logging.Logger) -> gpd.GeoDataFrame:
    """Load borehole locations from CSV."""
    # Resolve path relative to workspace root
    full_path = resolve_path(csv_path)
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

    # Create geometry
    geometry = [Point(x, y) for x, y in zip(df["Easting"], df["Northing"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=APP_CONFIG.target_crs)

    logger.info(f"   ✅ Loaded {len(gdf)} boreholes")
    return gdf


def load_bgs_layer_clipped(
    shp_path: str,
    extent: tuple,
    buffer_m: float,
    logger: logging.Logger,
    layer_name: str = "BGS Layer",
) -> gpd.GeoDataFrame:
    """
    Load BGS layer shapefile clipped to borehole extent.

    Args:
        shp_path: Path to shapefile (relative to workspace root)
        extent: Bounding box (min_x, max_x, min_y, max_y)
        buffer_m: Buffer distance to add around extent (meters)
        logger: Logger instance
        layer_name: Name for logging

    Returns:
        GeoDataFrame with polygons clipped to buffered extent
    """
    from shapely.geometry import box as shapely_box

    full_path = resolve_path(shp_path)
    logger.info(f"📂 Loading {layer_name}: {shp_path}")

    if not Path(full_path).exists():
        raise FileNotFoundError(f"Shapefile not found: {full_path}")

    # Unpack extent and add buffer
    min_x, max_x, min_y, max_y = extent
    min_x -= buffer_m
    max_x += buffer_m
    min_y -= buffer_m
    max_y += buffer_m

    logger.info(
        f"   Extent with {buffer_m:.0f}m buffer: "
        f"E[{min_x:.0f},{max_x:.0f}] N[{min_y:.0f},{max_y:.0f}]"
    )

    # Read only features within bounding box for efficiency
    bbox = (min_x, min_y, max_x, max_y)
    gdf = gpd.read_file(full_path, bbox=bbox)

    logger.info(f"   Loaded {len(gdf)} polygons within extent")

    # Ensure correct CRS
    target_crs = APP_CONFIG.target_crs
    if gdf.crs is None:
        gdf.set_crs(target_crs, inplace=True)
    elif gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)

    # Clip to exact extent
    clip_box = shapely_box(min_x, min_y, max_x, max_y)
    gdf["geometry"] = gdf.geometry.intersection(clip_box)
    gdf = gdf[~gdf.geometry.is_empty]

    logger.info(f"   ✅ After clipping: {len(gdf)} polygons")
    return gdf


def get_borehole_extent(
    boreholes_gdf: gpd.GeoDataFrame,
) -> Tuple[float, float, float, float]:
    """
    Get bounding box of all boreholes.

    Returns:
        Tuple of (min_x, max_x, min_y, max_y)
    """
    bounds = boreholes_gdf.total_bounds  # (minx, miny, maxx, maxy)
    return (bounds[0], bounds[2], bounds[1], bounds[3])  # (min_x, max_x, min_y, max_y)


def load_bgs_layers(boreholes_gdf: gpd.GeoDataFrame, logger: logging.Logger) -> dict:
    """
    Load BGS geology layers clipped to borehole extent.

    Args:
        boreholes_gdf: GeoDataFrame with borehole locations (for extent)
        logger: Logger instance

    Returns:
        Dict mapping layer_name -> (GeoDataFrame, config_dict)
    """
    logger.info("📂 Loading BGS layers...")
    bgs_layers = {}

    # Get borehole extent for clipping
    extent = get_borehole_extent(boreholes_gdf)

    # Load BGS Bedrock
    bgs_bedrock_config = CONFIG.get("bgs_bedrock", {})
    if bgs_bedrock_config.get("enabled", False):
        try:
            bedrock_gdf = load_bgs_layer_clipped(
                shp_path=bgs_bedrock_config["shapefile_path"],
                extent=extent,
                buffer_m=bgs_bedrock_config.get("buffer_m", 500.0),
                logger=logger,
                layer_name=bgs_bedrock_config.get("layer_name", "BGS Bedrock"),
            )
            bgs_layers[bgs_bedrock_config.get("layer_name", "BGS Bedrock")] = (
                bedrock_gdf,
                bgs_bedrock_config,
            )
        except FileNotFoundError as e:
            logger.warning(f"   ⚠️ BGS Bedrock not found: {e}")
        except Exception as e:
            logger.warning(f"   ⚠️ BGS Bedrock loading failed: {e}")

    # Load BGS Superficial Deposits
    bgs_deposits_config = CONFIG.get("bgs_deposits", {})
    if bgs_deposits_config.get("enabled", False):
        try:
            deposits_gdf = load_bgs_layer_clipped(
                shp_path=bgs_deposits_config["shapefile_path"],
                extent=extent,
                buffer_m=bgs_deposits_config.get("buffer_m", 500.0),
                logger=logger,
                layer_name=bgs_deposits_config.get("layer_name", "BGS Deposits"),
            )
            bgs_layers[bgs_deposits_config.get("layer_name", "BGS Deposits")] = (
                deposits_gdf,
                bgs_deposits_config,
            )
        except FileNotFoundError as e:
            logger.warning(f"   ⚠️ BGS Deposits not found: {e}")
        except Exception as e:
            logger.warning(f"   ⚠️ BGS Deposits loading failed: {e}")

    logger.info(f"   ✅ Loaded {len(bgs_layers)} BGS layers")
    return bgs_layers


def load_test_data_locations(logger: logging.Logger) -> Dict[str, set]:
    """
    Load test data CSVs and extract Location IDs for each test type.

    Returns:
        Dict mapping test type to set of Location IDs:
        {
            'spt': {'BH001', 'BH002', ...},
            'triaxial_total': {'EA1', 'EA2', ...},
            'triaxial_effective': {'BH043', ...}
        }
    """
    logger.info("📂 Loading test data CSVs...")
    test_data = {
        "spt": set(),
        "triaxial_total": set(),
        "triaxial_effective": set(),
    }

    test_files = CONFIG.get("test_data_files", {})

    # Load SPT data
    spt_path = test_files.get(
        "spt_csv", "Openground CSVs/GIR Location Group/SPT by Geology.csv"
    )
    try:
        full_path = resolve_path(spt_path)
        if Path(full_path).exists():
            df = pd.read_csv(full_path)
            df.columns = df.columns.str.strip()
            if "Location ID" in df.columns:
                test_data["spt"] = set(df["Location ID"].dropna().unique())
                logger.info(f"   ✅ SPT: {len(test_data['spt'])} locations")
        else:
            logger.warning(f"   ⚠️ SPT CSV not found: {spt_path}")
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to load SPT data: {e}")

    # Load Triaxial Total Stress data
    triaxial_total_path = test_files.get(
        "triaxial_total_csv",
        "Openground CSVs/GIR Location Group/Triaxial Total Stress by Geology.csv",
    )
    try:
        full_path = resolve_path(triaxial_total_path)
        if Path(full_path).exists():
            df = pd.read_csv(full_path)
            df.columns = df.columns.str.strip()
            if "Location ID" in df.columns:
                test_data["triaxial_total"] = set(df["Location ID"].dropna().unique())
                logger.info(
                    f"   ✅ Triaxial Total: {len(test_data['triaxial_total'])} locations"
                )
        else:
            logger.warning(f"   ⚠️ Triaxial Total CSV not found: {triaxial_total_path}")
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to load Triaxial Total data: {e}")

    # Load Triaxial Effective Stress data
    triaxial_effective_path = test_files.get(
        "triaxial_effective_csv",
        "Openground CSVs/GIR Location Group/Triaxial Effective Stress by Geology.csv",
    )
    try:
        full_path = resolve_path(triaxial_effective_path)
        if Path(full_path).exists():
            df = pd.read_csv(full_path)
            df.columns = df.columns.str.strip()
            if "Location ID" in df.columns:
                test_data["triaxial_effective"] = set(
                    df["Location ID"].dropna().unique()
                )
                logger.info(
                    f"   ✅ Triaxial Effective: {len(test_data['triaxial_effective'])} locations"
                )
        else:
            logger.warning(
                f"   ⚠️ Triaxial Effective CSV not found: {triaxial_effective_path}"
            )
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to load Triaxial Effective data: {e}")

    return test_data


# ═══════════════════════════════════════════════════════════════════════════
# 🎯 MAIN WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════


def _load_analysis_data(
    file_paths: Dict[str, str],
    logger: logging.Logger,
) -> Tuple[
    Dict[str, gpd.GeoDataFrame],
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    Dict[str, Any],
    Dict[str, Set[str]],
    Dict[str, float],
]:
    """Load all input data: shapefiles, boreholes, BGS layers, test data."""
    timings = {}

    # Step 1: Load all shapefiles (zones and boundary layers)
    logger.info("\nSTEP 1: Loading shapefile layers")
    step_start = time.perf_counter()
    all_shapefiles = load_all_shapefiles(logger)
    zones_gdf = get_zones_for_coverage_gdf(all_shapefiles, logger)
    if zones_gdf is None:
        raise ValueError("No coverage layer shapefile loaded - check SHAPEFILE_CONFIG")
    timings["1_load_shapefiles"] = time.perf_counter() - step_start
    logger.info(f"   📊 Loaded {len(all_shapefiles)} shapefile layers")
    coverage_keys = get_coverage_layer_keys()
    logger.info(f"   🎯 Coverage layers: {coverage_keys}")
    logger.info(f"   ⏱️ Step 1 completed in {timings['1_load_shapefiles']:.2f}s")

    # Step 2: Load boreholes
    logger.info("\nSTEP 2: Loading boreholes")
    step_start = time.perf_counter()
    boreholes_gdf = load_boreholes(file_paths["boreholes_csv"], logger)
    timings["2_load_boreholes"] = time.perf_counter() - step_start
    logger.info(f"   ⏱️ Step 2 completed in {timings['2_load_boreholes']:.2f}s")

    # Step 3: Load BGS layers
    logger.info("\nSTEP 3: Loading BGS geology layers")
    step_start = time.perf_counter()
    bgs_layers_data = load_bgs_layers(boreholes_gdf, logger)
    timings["3_load_bgs_layers"] = time.perf_counter() - step_start
    logger.info(f"   ⏱️ Step 3 completed in {timings['3_load_bgs_layers']:.2f}s")

    # Step 4: Load test data
    logger.info("\nSTEP 4: Loading test data (SPT, Triaxial)")
    step_start = time.perf_counter()
    test_data_locations = load_test_data_locations(logger)
    timings["4_load_test_data"] = time.perf_counter() - step_start
    logger.info(f"   ⏱️ Step 4 completed in {timings['4_load_test_data']:.2f}s")

    return (
        all_shapefiles,
        zones_gdf,
        boreholes_gdf,
        bgs_layers_data,
        test_data_locations,
        timings,
    )


def _log_solver_configuration(config: Dict, logger: logging.Logger) -> None:
    """Log solver and parallel processing configuration."""
    testing_config = config.get("testing_mode", {})
    ilp_config = config.get("ilp_solver", {})
    parallel_config = config.get("parallel", {})

    # Effective solver mode
    if testing_config.get("enabled", False):
        solver_mode = testing_config.get("solver_overrides", {}).get(
            "solver_mode", config.get("optimization", {}).get("solver_mode", "ilp")
        )
    else:
        solver_mode = config.get("optimization", {}).get("solver_mode", "ilp")
    logger.info(f"   🧮 Solver method: {solver_mode.upper()}")

    # Constraint mode
    constraint_mode = ilp_config.get("conflict_constraint_mode", "clique")
    exclusion_factor = ilp_config.get("exclusion_factor", 0.8)
    logger.info(
        f"   🔷 Constraint mode: {constraint_mode.upper()} (exclusion={exclusion_factor}x)"
    )

    # Worker count
    if testing_config.get("enabled", False) and testing_config.get(
        "force_single_worker", False
    ):
        worker_count = 1
    elif not parallel_config.get("enabled", True):
        worker_count = 1
    else:
        max_workers = parallel_config.get("max_workers", -1)
        if max_workers == -1:
            worker_count = parallel_config.get(
                "optimal_workers_default", min(os.cpu_count() or 4, 14)
            )
        else:
            worker_count = max_workers
    logger.info("   👷 Parallel workers: %d", worker_count)

    # Zone decomposition and connected components
    logger.info(
        "   🔀 Zone decomposition: %s",
        "ENABLED" if ilp_config.get("use_zone_decomposition", False) else "DISABLED",
    )
    logger.info(
        "   🧩 Connected components: %s",
        "ENABLED" if ilp_config.get("use_connected_components", False) else "DISABLED",
    )

    # Border consolidation (second pass after zone decomposition)
    border_config = CONFIG.get("border_consolidation", {})
    border_mode = border_config.get("mode", "disabled")
    logger.info(
        "   🔗 Border consolidation: %s",
        border_mode.upper() if border_mode != "disabled" else "DISABLED",
    )

    # Mode indicator
    if testing_config.get("enabled", False):
        f = testing_config.get("filter", {})
        logger.info("   🧪 TESTING MODE: Single filter combination")
        logger.info(
            "      Filter: min_depth=%sm, SPT=%s, TxT=%s, TxE=%s",
            f.get("min_depth", 0),
            "Yes" if f.get("require_spt") else "No",
            "Yes" if f.get("require_triaxial_total") else "No",
            "Yes" if f.get("require_triaxial_effective") else "No",
        )
    else:
        logger.info("   🏭 PRODUCTION MODE: All filter combinations")


def _log_solver_summary(
    precomputed_coverages: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """
    Log a summary of HiGHS solver results for first pass (zone decomposition)
    and second pass (border consolidation).

    Args:
        precomputed_coverages: Dict of filter_key -> coverage results
        logger: Logger instance
    """
    logger.info("\n" + "=" * 60)
    logger.info("SOLVER SUMMARY")
    logger.info("=" * 60)

    # Iterate through each filter combination
    for combo_key, combo_data in precomputed_coverages.items():
        if not isinstance(combo_data, dict):
            continue

        opt_stats = combo_data.get("optimization_stats", {})
        if not opt_stats:
            continue

        # Extract first pass (zone decomposition) stats
        zone_stats = opt_stats.get("zone_decomposition_stats", {})
        total_first_pass = opt_stats.get("selected_count", 0)

        # Extract second pass stats - check both consolidation and CZRC
        consol_stats = opt_stats.get("consolidation", {})
        czrc_stats = opt_stats.get("czrc_optimization", {})

        # Use CZRC stats if available, otherwise consolidation
        if czrc_stats:
            original_count = czrc_stats.get("original_count", total_first_pass)
            final_count = czrc_stats.get("final_count", total_first_pass)
            solve_time = czrc_stats.get("solve_time", 0)
            second_pass_type = "CZRC"
        elif consol_stats:
            original_count = consol_stats.get("original_count", total_first_pass)
            final_count = consol_stats.get("final_count", total_first_pass)
            solve_time = consol_stats.get("solve_time", 0)
            second_pass_type = "Consolidation"
        else:
            original_count = total_first_pass
            final_count = total_first_pass
            solve_time = 0
            second_pass_type = None

        # Net change: negative means fewer boreholes after second pass
        net_change = final_count - original_count

        # Log summary for this filter combination
        logger.info(f"\nFilter: {combo_key}")
        logger.info(f"   First Pass (Zone Decomposition): {original_count} proposed")

        if second_pass_type:
            if net_change != 0:
                logger.info(
                    f"   Second Pass ({second_pass_type}): {original_count} -> {final_count} "
                    f"({net_change:+d} boreholes, {solve_time:.2f}s)"
                )
            else:
                logger.info(
                    f"   Second Pass ({second_pass_type}): No change ({original_count} boreholes)"
                )
        else:
            logger.info("   Second Pass: Disabled")

        logger.info(f"   Final Proposed: {final_count} boreholes")

    logger.info("=" * 60)


def _compute_coverages(
    boreholes_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    test_data_locations: Dict[str, Set[str]],
    max_spacing: float,
    config: Dict[str, Any],
    workspace_root: Path,
    logger: logging.Logger,
    highs_log_folder: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Dict[str, Any]]], Dict[str, float]]:
    """Compute coverage statistics and precompute all filter combinations."""
    timings = {}

    # Step 5: Compute coverage statistics
    logger.info("\nSTEP 5: Computing EC7 coverage statistics")
    step_start = time.perf_counter()
    covered_union, uncovered_gaps, gap_stats = compute_coverage_zones(
        boreholes_gdf=boreholes_gdf,
        zones_gdf=zones_gdf,
        max_spacing=max_spacing,
        logger=logger,
    )
    coverage_summary = get_coverage_summary(covered_union, uncovered_gaps, gap_stats)
    timings["5_coverage_zones"] = time.perf_counter() - step_start
    logger.info(f"   ⏱️ Step 5 completed in {timings['5_coverage_zones']:.2f}s")

    # Step 5.5: Precompute all filter combinations
    logger.info("\nSTEP 5.5: Pre-computing coverage for all filter combinations")
    _log_solver_configuration(config, logger)

    step_start = time.perf_counter()
    from Gap_Analysis_EC7.parallel.cached_coverage_orchestrator import (
        precompute_all_coverages_cached,
    )

    max_depth = boreholes_gdf["Final Depth"].dropna().max() or 0.0
    depth_step = config.get("depth_slider_step_m", 10.0)

    precomputed_coverages = precompute_all_coverages_cached(
        boreholes_gdf=boreholes_gdf,
        zones_gdf=zones_gdf,
        test_data_locations=test_data_locations,
        max_spacing=max_spacing,
        max_depth=max_depth,
        depth_step=depth_step,
        config=config,
        workspace_root=workspace_root,
        highs_log_folder=str(highs_log_folder) if highs_log_folder else None,
    )
    timings["5.5_precompute_coverages"] = time.perf_counter() - step_start
    logger.info(
        f"   ⏱️ Step 5.5 completed in {timings['5.5_precompute_coverages']:.2f}s"
    )
    logger.info(f"   📊 Pre-computed {len(precomputed_coverages)} filter combinations")

    return coverage_summary, precomputed_coverages, timings


def _export_analysis_outputs(
    precomputed_coverages: Dict,
    zones_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    config: Dict,
    logger: logging.Logger,
    total_time: float = 0.0,
) -> Dict[str, float]:
    """Export CSV, GeoJSON (testing mode), PNG outputs, and run stats summary."""
    from Gap_Analysis_EC7.exporters import (
        export_proposed_boreholes_to_csv,
        export_per_pass_boreholes_to_csv,
        export_coverage_polygons_to_geojson,
        export_run_stats_summary,
    )

    timings = {}

    # Log solver summary (first pass + consolidation results)
    _log_solver_summary(precomputed_coverages, logger)

    # Step 5.5c: Export run stats summary (overwrites each run)
    logger.info("\nSTEP 5.5c: Exporting run statistics summary")
    step_start = time.perf_counter()
    _ = export_run_stats_summary(
        precomputed_coverages=precomputed_coverages,
        output_dir=output_dir,
        total_time=total_time,
        log=logger,
    )
    timings["5.5c_export_stats_summary"] = time.perf_counter() - step_start
    logger.info(
        f"   ⏱️ Step 5.5c completed in {timings['5.5c_export_stats_summary']:.2f}s"
    )

    # Step 5.6: Export CSV (legacy single file per combo)
    logger.info("\nSTEP 5.6: Exporting proposed borehole coordinates to CSV")
    step_start = time.perf_counter()
    _ = export_proposed_boreholes_to_csv(
        precomputed_coverages=precomputed_coverages, output_dir=output_dir, log=logger
    )
    timings["5.6_export_csv"] = time.perf_counter() - step_start
    logger.info(f"   ⏱️ Step 5.6 completed in {timings['5.6_export_csv']:.2f}s")

    # Step 5.6b: Export per-pass CSV (timestamped folders)
    testing_enabled = config.get("testing_mode", {}).get("enabled", False)
    logger.info("\nSTEP 5.6b: Exporting per-pass borehole CSVs")
    step_start = time.perf_counter()
    _ = export_per_pass_boreholes_to_csv(
        precomputed_coverages=precomputed_coverages,
        output_dir=output_dir,
        is_testing_mode=testing_enabled,
        zones_gdf=zones_gdf,
        log=logger,
    )
    timings["5.6b_export_per_pass_csv"] = time.perf_counter() - step_start
    logger.info(
        f"   ⏱️ Step 5.6b completed in {timings['5.6b_export_per_pass_csv']:.2f}s"
    )

    # Step 5.7: Export GeoJSON (testing mode only)
    if testing_enabled:
        logger.info("\nSTEP 5.7: Exporting coverage polygons to GeoJSON (testing mode)")
        step_start = time.perf_counter()
        _ = export_coverage_polygons_to_geojson(
            precomputed_coverages=precomputed_coverages,
            output_dir=output_dir,
            log=logger,
        )
        timings["5.7_export_geojson"] = time.perf_counter() - step_start
        logger.info(f"   ⏱️ Step 5.7 completed in {timings['5.7_export_geojson']:.2f}s")
    else:
        timings["5.7_export_geojson"] = 0.0

    # Step 5.8: Export PNG (testing mode only)
    if testing_enabled:
        logger.info("\nSTEP 5.8: Exporting PNG for AI inspection (testing mode)")
        step_start = time.perf_counter()
        from Gap_Analysis_EC7.exporters import export_all_coverage_outputs

        _ = export_all_coverage_outputs(
            precomputed_coverages=precomputed_coverages,
            zones_gdf=zones_gdf,
            output_dir=output_dir,
            log=logger,
        )
        timings["5.8_export_png"] = time.perf_counter() - step_start
        logger.info(f"   ⏱️ Step 5.8 completed in {timings['5.8_export_png']:.2f}s")

    return timings


def _generate_html_and_summary(
    boreholes_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    all_shapefiles: Dict[str, gpd.GeoDataFrame],
    bgs_layers_data: Dict,
    test_data_locations: Dict[str, set],
    precomputed_coverages: Dict,
    coverage_summary: Dict,
    output_dir: Path,
    zones_config: Dict,
    vis_config: "Dict[str, Any] | VisualizationConfig",
    max_spacing: float,
    timings: Dict[str, float],
    logger: logging.Logger,
    default_filter: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Generate HTML output and log summary statistics."""
    # Step 6: Generate HTML
    logger.info("\nSTEP 6: Generating HTML output")
    step_start = time.perf_counter()
    output_html = str(output_dir / "ec7_coverage.html")
    generate_multi_layer_html(
        boreholes_gdf=boreholes_gdf,
        output_path=output_html,
        visualization_config=vis_config,
        generator_name="EC7",
        zones_gdf=zones_gdf,
        all_shapefiles=all_shapefiles,
        zones_config=zones_config,
        bgs_layers_data=bgs_layers_data,
        test_data_locations=test_data_locations,
        max_spacing=max_spacing,
        logger=logger,
        precomputed_coverages=precomputed_coverages,
        default_filter=default_filter,
    )
    timings["6_generate_html"] = time.perf_counter() - step_start
    logger.info(f"   ⏱️ Step 6 completed in {timings['6_generate_html']:.2f}s")

    # Summary
    covered_ha, uncovered_ha = (
        coverage_summary["covered_area_ha"],
        coverage_summary["uncovered_area_ha"],
    )
    coverage_pct, gap_count = (
        coverage_summary["coverage_pct"],
        coverage_summary["num_gaps"],
    )

    logger.info("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    logger.info(f"   Coverage Area: {covered_ha:.1f} ha ({coverage_pct:.1f}%)")
    logger.info(f"   Uncovered Area: {uncovered_ha:.1f} ha")
    logger.info(f"   Gap Count: {gap_count}")
    logger.info(f"   Existing boreholes: {len(boreholes_gdf)}")

    # Extract proposed borehole stats and gap percentages from precomputed_coverages
    for combo_key, combo_data in precomputed_coverages.items():
        if not isinstance(combo_data, dict):
            continue
        opt_stats = combo_data.get("optimization_stats", {})

        # Display combination name
        logger.info(f"\n   Combination: {combo_key}")

        # Calculate average first pass gap from zone stats
        zones_data = opt_stats.get("zones", {})
        first_pass_gaps = []
        for zone_name, zone_info in zones_data.items():
            zone_stats = zone_info.get("stats", {})
            stall_det = zone_stats.get("stall_detection", {})
            if stall_det and stall_det.get("final_gap_pct") is not None:
                first_pass_gaps.append(stall_det["final_gap_pct"])

        # Get second pass gap from consolidation/CZRC
        consol_stats = opt_stats.get("consolidation", {})
        czrc_stats = opt_stats.get("czrc_optimization", {})
        second_pass_gap = None

        if czrc_stats:
            czrc_stall = czrc_stats.get("stall_detection") or {}
            second_pass_gap = czrc_stall.get("final_gap_pct")
        elif consol_stats:
            consol_stall = consol_stats.get("stall_detection") or {}
            second_pass_gap = consol_stall.get("final_gap_pct")

        # Log gap statistics
        if first_pass_gaps:
            avg_first_gap = sum(first_pass_gaps) / len(first_pass_gaps)
            logger.info(f"   Avg First Pass Gap: {avg_first_gap:.1f}%")
        if second_pass_gap is not None:
            logger.info(f"   Second Pass Gap: {second_pass_gap:.1f}%")

        # Proposed borehole counts
        if consol_stats:
            original = consol_stats.get("original_count", 0)
            final = consol_stats.get("final_count", 0)
            solve_time = consol_stats.get("solve_time", 0.0)
            net_change = final - original
            if net_change != 0:
                logger.info(
                    f"   Proposed boreholes: {original} -> {final} "
                    f"({net_change:+d} boreholes, {solve_time:.2f}s)"
                )
            else:
                logger.info(f"   Proposed boreholes: {final}")
        else:
            # No consolidation - use proposed count directly
            proposed = combo_data.get("proposed", [])
            logger.info(f"   Proposed boreholes: {len(proposed)}")
        break  # Only report first filter (testing mode has one)

    logger.info(f"\n   📄 Output: {output_html}")

    # Only show TOTAL timing (simplified from detailed breakdown)
    total_time = timings["total"]
    logger.info(f"\nTOTAL: {total_time:.2f}s")
    logger.info("=" * 60 + "\n✅ Analysis complete!")

    return {
        "coverage_summary": coverage_summary,
        "boreholes_gdf": boreholes_gdf,
        "zones_gdf": zones_gdf,
        "bgs_layers_data": bgs_layers_data,
        "output_html": output_html,
        "stats": {
            "covered_area_ha": covered_ha,
            "uncovered_area_ha": uncovered_ha,
            "coverage_pct": coverage_pct,
            "gap_count": gap_count,
        },
    }


def run_ec7_analysis() -> dict:
    """Run EC7 gap analysis workflow with phase-based organization."""
    logger, run_log_folder = setup_logging()
    logger.info("=" * 60)
    logger.info("🎯 EC7 Simple Gap Analysis")
    logger.info("=" * 60)
    logger.info(f"   Log folder: {run_log_folder}")

    # =========================================================================
    # TYPED CONFIG ACCESS (using module-level APP_CONFIG)
    # =========================================================================
    # APP_CONFIG was created at module load time. Use it for typed access.
    app_config = APP_CONFIG

    # Extract commonly-used values using typed properties
    max_spacing = app_config.max_spacing_m
    # Build zones_config from shapefile_config (replaces legacy CONFIG["zones"])
    zones_config = build_zones_config_for_visualization()
    vis_config = app_config.visualization  # Typed VisualizationConfig

    # Create file_paths dict for backwards compatibility with _load_analysis_data
    # TODO: Update _load_analysis_data to accept FilePathsConfig directly
    file_paths = {
        "boreholes_csv": app_config.file_paths.boreholes_csv,
        "output_dir": app_config.file_paths.output_dir,
        "log_dir": app_config.file_paths.log_dir,
    }

    logger.info(f"   Max Spacing: {max_spacing}m")
    cell_size = max_spacing / (2**0.5)
    logger.info(f"   Cell Size: {cell_size:.1f}m × {cell_size:.1f}m")

    # Create output directory using typed file_paths
    output_dir = app_config.file_paths.output_dir_path(WORKSPACE_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    timings = {}
    total_start = time.perf_counter()

    try:
        # Phase 1: Load all input data (shapefiles, boreholes, BGS, test data)
        (
            all_shapefiles,
            zones_gdf,
            boreholes_gdf,
            bgs_layers_data,
            test_data_locations,
            load_timings,
        ) = _load_analysis_data(file_paths, logger)
        timings.update(load_timings)

        # Phase 2: Compute coverage statistics and precompute combinations
        coverage_summary, precomputed_coverages, coverage_timings = _compute_coverages(
            boreholes_gdf,
            zones_gdf,
            test_data_locations,
            max_spacing,
            CONFIG,
            WORKSPACE_ROOT,
            logger,
            run_log_folder,  # HiGHS logs also saved in this folder
        )
        timings.update(coverage_timings)

        # Phase 3: Export analysis outputs
        # Calculate elapsed time so far for stats summary
        elapsed_time = time.perf_counter() - total_start
        export_timings = _export_analysis_outputs(
            precomputed_coverages,
            zones_gdf,
            output_dir,
            CONFIG,
            logger,
            total_time=elapsed_time,
        )
        timings.update(export_timings)

        # Phase 4: Generate HTML and log summary
        timings["total"] = time.perf_counter() - total_start

        # Get testing mode filter for HTML default values (only if testing mode enabled)
        # Using typed config for testing_mode access
        default_filter = None
        if app_config.testing_mode.enabled:
            # Convert TestingFilterConfig to dict for HTML generation
            filter_config = app_config.testing_mode.filter
            default_filter = {
                "min_depth": filter_config.min_depth,
                "require_spt": filter_config.require_spt,
                "require_triaxial_total": filter_config.require_triaxial_total,
                "require_triaxial_effective": filter_config.require_triaxial_effective,
            }

        result = _generate_html_and_summary(
            boreholes_gdf,
            zones_gdf,
            all_shapefiles,
            bgs_layers_data,
            test_data_locations,
            precomputed_coverages,
            coverage_summary,
            output_dir,
            zones_config,
            vis_config,
            max_spacing,
            timings,
            logger,
            default_filter,
        )

        # Phase 5: Generate zone_coverage_data.json for zone_coverage_viz
        try:
            from zone_coverage_viz.generate_zone_data import generate_zone_coverage_data

            logger.info("\n📊 Generating zone_coverage_viz data...")
            generate_zone_coverage_data()
        except ImportError:
            logger.warning("   ⚠️ zone_coverage_viz not available, skipping data export")
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to generate zone_coverage_viz data: {e}")

        return result

    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise


# ═══════════════════════════════════════════════════════════════════════════
# 🚀 ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    run_ec7_analysis()
