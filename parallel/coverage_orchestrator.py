"""
Orchestrator for parallel coverage pre-computation.

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Dispatch all filter combinations to parallel workers
and collect results into a unified data structure for HTML embedding.

This module consolidates:
- coverage_config.py: Filter combination generation
- coverage_serialization.py: GeoDataFrame serialization for pickling

Follows Gap_Analysis parallel_orchestrator.py patterns:
- should_use_parallel() check for environment validation
- Serialize inputs ONCE before dispatch (avoid per-worker overhead)
- joblib Parallel with delayed for process-based parallelism
- Result collection with error aggregation
- Always uses parallel infrastructure (n_jobs=1 for sequential execution)

Key Functions:
- precompute_all_coverages(): Main entry point for pre-computation
- _dispatch_parallel_coverages(): Parallel job dispatch (also handles n_jobs=1)
- generate_filter_combinations(): Build all (depth, spt, txt, txe) tuples
- serialize_geodataframe(): GeoDataFrame → (records, crs_string)

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry.base import BaseGeometry

# Phase 3: Import AppConfig for typed configuration support
from Gap_Analysis_EC7.config_types import AppConfig

logger = logging.getLogger("EC7.Parallel.Orchestrator")


# ═══════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIG NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def _normalize_config(config: Union[Dict[str, Any], AppConfig]) -> AppConfig:
    """
    Normalize config to AppConfig for internal use.

    Accepts either a raw CONFIG dictionary or an AppConfig object.
    This enables backwards compatibility while supporting typed config.

    Args:
        config: Either raw CONFIG dict or AppConfig object.

    Returns:
        AppConfig instance for typed access.
    """
    if isinstance(config, AppConfig):
        return config
    return AppConfig.from_dict(config)


# ═══════════════════════════════════════════════════════════════════════════
# 📋 DEFAULT PARALLEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_PARALLEL_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "max_workers": -1,  # Auto-detect from CPU
    "optimal_workers_default": 14,
    "min_combinations_for_parallel": 10,
    "timeout_per_combo_seconds": 60,
    "fallback_on_error": True,
    "backend": "loky",
    "verbose": 10,
}


# ═══════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIG ACCESS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def get_parallel_config(
    config: Union[Dict[str, Any], AppConfig],
) -> Dict[str, Any]:
    """
    Extract parallel config from main CONFIG with defaults fallback.

    Args:
        config: Main CONFIG dictionary or AppConfig object.

    Returns:
        Merged parallel config dict with defaults applied.
    """
    app_config = _normalize_config(config)

    # Build dict from typed config
    user_config = {
        "enabled": app_config.parallel.enabled,
        "max_workers": app_config.parallel.max_workers,
        "optimal_workers_default": app_config.parallel.optimal_workers_default,
        "min_combinations_for_parallel": app_config.parallel.min_combinations_for_parallel,
        "backend": app_config.parallel.backend,
        "verbose": app_config.parallel.verbose,
    }

    # Merge with defaults (for any missing fields)
    merged = {**DEFAULT_PARALLEL_CONFIG, **user_config}
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# 🧮 FILTER COMBINATION GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def generate_filter_combinations(
    max_depth: float,
    depth_step: float = 10.0,
    config: Union[Dict[str, Any], AppConfig, None] = None,
) -> List[Dict[str, Any]]:
    """
    Generate all (depth, spt, txt, txe) filter combinations.

    Creates the full matrix of filter combinations that need to be
    pre-computed for the HTML interactive viewer.

    When testing_mode is enabled in config, returns only a single
    combination for rapid algorithm testing.

    Args:
        max_depth: Maximum borehole depth in dataset.
        depth_step: Depth slider step (from CONFIG["depth_slider_step_m"]).
        config: Optional CONFIG dict or AppConfig (for testing_mode settings).

    Returns:
        List of filter combination dicts with keys:
        - min_depth: int (depth threshold value)
        - require_spt: bool (SPT checkbox state)
        - require_triaxial_total: bool (TxT checkbox state)
        - require_triaxial_effective: bool (TxE checkbox state)

    Example:
        >>> combos = generate_filter_combinations(95.0, 10.0)
        >>> len(combos)
        80  # 10 depths × 8 checkbox states
    """
    # Check for testing mode using typed config
    if config is not None:
        app_config = _normalize_config(config)
        testing = app_config.testing_mode
        if testing.enabled:
            test_combo = {
                "min_depth": testing.filter.min_depth,
                "require_spt": testing.filter.require_spt,
                "require_triaxial_total": testing.filter.require_triaxial_total,
                "require_triaxial_effective": testing.filter.require_triaxial_effective,
            }
            logger.info("🧪 TESTING MODE: Single filter combination only")
            logger.info(f"   Filter: depth≥{test_combo['min_depth']}m")
            logger.info(
                f"   SPT={test_combo['require_spt']}, "
                f"TxT={test_combo['require_triaxial_total']}, "
                f"TxE={test_combo['require_triaxial_effective']}"
            )
            return [test_combo]

    # Normal mode: generate full matrix of combinations
    step = int(depth_step)
    max_val = int((max_depth // step + 1) * step)
    depths = list(range(0, max_val + 1, step))

    combinations = []
    for depth in depths:
        for spt in [False, True]:
            for txt in [False, True]:
                for txe in [False, True]:
                    combinations.append(
                        {
                            "min_depth": depth,
                            "require_spt": spt,
                            "require_triaxial_total": txt,
                            "require_triaxial_effective": txe,
                        }
                    )

    logger.info(f"📊 Generated {len(combinations)} filter combinations")
    logger.info(f"   Depths: {len(depths)} values (0 to {max_val}m, step {step}m)")
    logger.info(f"   Checkboxes: 8 combinations (2³)")

    return combinations


def get_combination_key(combo: Dict[str, Any]) -> str:
    """
    Generate unique key for filter combination.

    Format: d{depth}_spt{0|1}_txt{0|1}_txe{0|1}

    Args:
        combo: Filter combination dict

    Returns:
        Unique string key for this combination

    Example:
        >>> combo = {"min_depth": 25, "require_spt": True,
        ...          "require_triaxial_total": False,
        ...          "require_triaxial_effective": False}
        >>> get_combination_key(combo)
        'd25_spt1_txt0_txe0'
    """
    return (
        f"d{combo['min_depth']}_"
        f"spt{int(combo['require_spt'])}_"
        f"txt{int(combo['require_triaxial_total'])}_"
        f"txe{int(combo['require_triaxial_effective'])}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 📦 GEODATAFRAME SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def serialize_geodataframe(gdf: gpd.GeoDataFrame) -> Tuple[List[Dict], str]:
    """
    Serialize GeoDataFrame to list of records with WKT geometry.

    Converts a GeoDataFrame to a picklable format for transport
    between worker processes.

    Args:
        gdf: GeoDataFrame to serialize

    Returns:
        Tuple of (records_list, crs_string)
        - records_list: List of dicts, one per row, geometry as WKT string
        - crs_string: CRS as string (e.g., "EPSG:27700") or empty string
    """
    if gdf is None or gdf.empty:
        return [], ""

    # Convert to records
    records = []
    for _, row in gdf.iterrows():
        record = row.to_dict()
        # Convert geometry to WKT string
        if "geometry" in record and record["geometry"] is not None:
            record["geometry"] = record["geometry"].wkt
        records.append(record)

    crs_str = str(gdf.crs) if gdf.crs else ""
    return records, crs_str


def deserialize_geodataframe(
    records: List[Dict],
    crs_str: str,
) -> gpd.GeoDataFrame:
    """
    Reconstruct GeoDataFrame from serialized records.

    Args:
        records: List of dicts with WKT geometry strings
        crs_str: CRS string (e.g., "EPSG:27700")

    Returns:
        Reconstructed GeoDataFrame with proper geometry and CRS
    """
    if not records:
        return gpd.GeoDataFrame()

    # Convert WKT back to geometries
    for record in records:
        if "geometry" in record and isinstance(record["geometry"], str):
            try:
                record["geometry"] = wkt.loads(record["geometry"])
            except Exception as e:
                logger.warning(f"Failed to parse WKT geometry: {e}")
                record["geometry"] = None

    df = pd.DataFrame(records)

    # Handle case where no geometry column exists
    if "geometry" not in df.columns:
        return gpd.GeoDataFrame(df)

    gdf = gpd.GeoDataFrame(df, geometry="geometry")

    if crs_str:
        gdf = gdf.set_crs(crs_str)

    return gdf


# ═══════════════════════════════════════════════════════════════════════════
# 📐 SINGLE GEOMETRY SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def serialize_geometry(geom: Optional[BaseGeometry]) -> Optional[str]:
    """
    Serialize a single Shapely geometry to WKT string.

    Used for coverage polygons (covered area, gaps) which are
    single geometries rather than GeoDataFrames.

    Args:
        geom: Shapely geometry object (Polygon, MultiPolygon, etc.)

    Returns:
        WKT string representation, or None if geometry is empty/None
    """
    if geom is None or geom.is_empty:
        return None
    return geom.wkt


def deserialize_geometry(wkt_str: Optional[str]) -> Optional[BaseGeometry]:
    """
    Deserialize WKT string back to Shapely geometry.

    Args:
        wkt_str: WKT string representation

    Returns:
        Shapely geometry object, or None if input is None/empty
    """
    if wkt_str is None or wkt_str == "":
        return None
    try:
        return wkt.loads(wkt_str)
    except Exception as e:
        logger.warning(f"Failed to parse WKT: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 📊 RESOURCE MONITORING
# ═══════════════════════════════════════════════════════════════════════════


def _log_resource_usage(stage: str) -> None:
    """
    Log memory and CPU usage at key stages.

    Helps debug OOM issues and understand resource utilization patterns.
    Gracefully degrades if psutil is not installed.

    Args:
        stage: Description of current execution stage (e.g., "before_dispatch")
    """
    try:
        import psutil

        process = psutil.Process()
        mem_mb = process.memory_info().rss / (1024 * 1024)
        cpu_pct = process.cpu_percent()
        logger.debug(f"[{stage}] Memory: {mem_mb:.0f}MB, CPU: {cpu_pct:.1f}%")
    except ImportError:
        pass  # psutil not installed, skip monitoring
    except Exception as e:
        logger.debug(f"[{stage}] Resource monitoring failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 🔍 PARALLEL DECISION LOGIC
# ═══════════════════════════════════════════════════════════════════════════


def should_use_parallel(
    n_combinations: int,
    config: Union[Dict[str, Any], AppConfig],
) -> Tuple[bool, str]:
    """
    Determine if parallel processing should be used.

    Checks configuration, environment, and job count to decide
    whether parallel dispatch is appropriate.

    Args:
        n_combinations: Number of filter combinations to process.
        config: Main CONFIG dictionary or AppConfig object.

    Returns:
        Tuple of (should_use: bool, reason: str).
    """
    parallel_config = get_parallel_config(config)

    # Check master toggle
    if not parallel_config.get("enabled", True):
        return False, "Parallel disabled in config"

    # Check minimum job threshold
    min_combos = parallel_config.get("min_combinations_for_parallel", 10)
    if n_combinations < min_combos:
        return False, f"Only {n_combinations} combinations (< {min_combos} threshold)"

    # Check joblib availability
    try:
        from joblib import Parallel, delayed  # noqa: F401
    except ImportError:
        return False, "joblib not installed"

    return True, f"OK ({n_combinations} combinations)"


def get_effective_worker_count(
    n_combinations: int,
    config: Union[Dict[str, Any], AppConfig],
) -> int:
    """
    Calculate optimal worker count based on combinations and config.

    Args:
        n_combinations: Number of jobs to process.
        config: Main CONFIG dictionary or AppConfig object.

    Returns:
        Number of workers to use.
    """
    import os

    parallel_config = get_parallel_config(config)
    max_workers = parallel_config.get("max_workers", -1)

    if max_workers == -1:
        # Auto-detect based on CPU cores
        cpu_count = os.cpu_count() or 4
        optimal = parallel_config.get("optimal_workers_default", 4)
        max_workers = min(cpu_count, optimal)

    # Don't use more workers than combinations
    return min(max_workers, n_combinations)


# ═══════════════════════════════════════════════════════════════════════════
# 🚀 MAIN ORCHESTRATOR FUNCTION
# ═══════════════════════════════════════════════════════════════════════════


def precompute_all_coverages(
    boreholes_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    test_data_locations: Dict[str, Set[str]],
    max_spacing: float,
    max_depth: float,
    depth_step: float,
    config: Union[Dict[str, Any], AppConfig],
    highs_log_folder: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Pre-compute coverage zones for all filter combinations.

    Main entry point for parallel coverage computation. Generates all
    filter combinations, dispatches to parallel workers (with n_jobs=1
    for sequential mode), and returns unified results for HTML embedding.

    When testing_mode is enabled in config:
    - Generates only a single filter combination (for rapid testing)
    - Uses n_jobs=1 for sequential execution (same code path as production)
    - Applies testing solver overrides (extended timeout, tighter tolerance)

    Args:
        boreholes_gdf: All boreholes GeoDataFrame (will be filtered per combo).
        zones_gdf: Zone boundaries GeoDataFrame.
        test_data_locations: Dict mapping test type -> set of Location IDs.
            Keys: "spt", "triaxial_total", "triaxial_effective"
        max_spacing: EC7 maximum spacing in meters (typically 200.0).
        max_depth: Maximum borehole depth (determines slider range).
        depth_step: Depth slider step from CONFIG["depth_slider_step_m"].
        config: Full CONFIG dict or AppConfig object.

    Returns:
        Dict mapping combo_key -> coverage data dict.
        Each coverage data dict contains:
        - key: str - combination key
        - success: bool
        - boreholes_count: int
        - covered: Optional[str] - WKT polygon
        - gaps: Optional[str] - WKT polygon
        - proposed: List[Dict] - [{"x": float, "y": float}, ...]
        - stats: Dict
        - duration_seconds: float
        - error: Optional[str]
    """
    start_time = time.time()

    # Normalize to AppConfig for typed access
    app_config = _normalize_config(config)
    is_testing = app_config.testing_mode.enabled

    # Generate filter combinations (respects testing_mode)
    combinations = generate_filter_combinations(max_depth, depth_step, app_config)
    n_combos = len(combinations)

    logger.info("=" * 60)
    if is_testing:
        logger.info("🧪 TESTING MODE: Single filter combination")
    else:
        logger.info("🧮 PRE-COMPUTING COVERAGE FOR ALL FILTER COMBINATIONS")
    logger.info("=" * 60)
    logger.info(f"   Total combinations: {n_combos}")
    logger.info(f"   Depth step: {depth_step}m")
    logger.info(f"   Max depth: {max_depth:.1f}m")

    # For dispatch, we need raw config dict (worker expects ilp_config dict)
    # We keep the raw dict interface for workers since they run in separate processes
    raw_config = app_config._raw_config.copy()

    # Apply testing mode config overrides if needed
    testing_overrides = app_config.testing_mode.solver_overrides
    if is_testing and testing_overrides:
        raw_config["parallel_solver_overrides"] = {
            **raw_config.get("parallel_solver_overrides", {}),
            **testing_overrides,  # solver_overrides is a dict
        }
        logger.info(
            f"   🧪 Testing solver: mode={testing_overrides.get('solver_mode')}, "
            f"time={testing_overrides.get('time_limit_s')}s"
        )

    # Determine execution mode - always use parallel infrastructure
    force_single_worker = is_testing and app_config.testing_mode.force_single_worker
    use_parallel, reason = should_use_parallel(n_combos, app_config)

    if force_single_worker or not use_parallel:
        # Use parallel infrastructure with n_jobs=1 for sequential execution
        raw_config = raw_config.copy()
        if "parallel" not in raw_config:
            raw_config["parallel"] = {}
        raw_config["parallel"]["max_workers"] = 1
        raw_config["parallel"]["min_combinations_for_parallel"] = 0
        if force_single_worker:
            reason = "Testing mode (single worker)"
        else:
            reason = f"{reason} -> using n_jobs=1"

    logger.info(f"   ⚡ Using parallel processing: {reason}")
    results = _dispatch_parallel_coverages(
        boreholes_gdf,
        zones_gdf,
        test_data_locations,
        max_spacing,
        combinations,
        raw_config,  # Workers need raw config dict for serialization
        is_testing=is_testing,  # Zone cache only in production mode
        highs_log_folder=highs_log_folder,
    )

    elapsed = time.time() - start_time
    success_count = sum(1 for r in results.values() if r.get("success"))
    error_count = len(results) - success_count

    logger.info("=" * 60)
    logger.info("✅ PRE-COMPUTATION COMPLETE")
    logger.info(f"   Success: {success_count}/{n_combos}")
    logger.info(f"   Errors: {error_count}")
    logger.info(f"   Total time: {elapsed:.1f}s ({elapsed/n_combos:.2f}s/combo)")
    logger.info("=" * 60)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# ⚡ PARALLEL DISPATCH
# ═══════════════════════════════════════════════════════════════════════════


def _dispatch_parallel_coverages(
    boreholes_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    test_data_locations: Dict[str, Set[str]],
    max_spacing: float,
    combinations: List[Dict[str, Any]],
    config: Dict[str, Any],
    is_testing: bool = False,
    highs_log_folder: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Dispatch parallel jobs for all filter combinations.

    Serializes inputs once before dispatch to avoid per-worker overhead.
    Uses joblib Parallel with loky backend for process-based parallelism.

    In PRODUCTION mode (is_testing=False), creates a temporary zone cache
    directory shared across workers to avoid redundant ILP solves for
    identical zone-gap geometries within the same run.

    Zone cache is DISABLED in testing mode to ensure test reproducibility
    and avoid cache overhead for single-combination tests.

    Args:
        boreholes_gdf: All boreholes GeoDataFrame
        zones_gdf: Zone boundaries GeoDataFrame
        test_data_locations: Dict of test type -> set of Location IDs
        max_spacing: EC7 maximum spacing
        combinations: List of filter combination dicts
        config: Full CONFIG dict
        is_testing: If True, disable zone cache (testing mode)

    Returns:
        Dict mapping combo_key -> result dict
    """
    import tempfile
    import shutil
    from joblib import Parallel, delayed
    from Gap_Analysis_EC7.parallel.coverage_worker import (
        worker_process_filter_combination,
    )

    parallel_config = get_parallel_config(config)
    n_workers = get_effective_worker_count(len(combinations), config)

    logger.info(
        f"🚀 Dispatching {len(combinations)} combinations to {n_workers} workers..."
    )

    # Create zone cache directory ONLY when:
    # 1. NOT in testing mode (testing_mode.enabled = False)
    # 2. Zone cache is enabled in config (parallel.zone_cache_enabled = True)
    #
    # Testing mode always disables cache for reproducibility.
    # Config toggle allows disabling cache in production for debugging.
    zone_cache_enabled = parallel_config.get("zone_cache_enabled", True)
    zone_cache_dir = None

    if is_testing:
        logger.info("   🧪 Zone cache disabled (testing mode)")
    elif not zone_cache_enabled:
        logger.info("   ⚠️ Zone cache disabled (zone_cache_enabled=False in config)")
    else:
        zone_cache_dir = tempfile.mkdtemp(prefix="zone_cache_")
        logger.info(f"   🗂️ Zone cache directory: {zone_cache_dir}")

    # Create CZRC cache directory (SHARED across all workers)
    # This enables cross-worker caching for CZRC second-pass optimization.
    # Without shared directory, each worker would have its own isolated cache.
    czrc_config = config.get("czrc_optimization", {})
    czrc_cache_enabled = czrc_config.get("enabled", False) and czrc_config.get(
        "cache_enabled", True
    )
    czrc_cache_dir = None

    if is_testing:
        logger.debug("   🧪 CZRC cache disabled (testing mode)")
    elif not czrc_cache_enabled:
        logger.debug("   ⚠️ CZRC cache disabled (not enabled in config)")
    else:
        czrc_cache_dir = tempfile.mkdtemp(prefix="czrc_cache_")
        logger.info(f"   🗂️ CZRC cache directory: {czrc_cache_dir}")

    # Serialize inputs ONCE (expensive operation done before parallel)
    logger.info("   📦 Serializing inputs...")
    bh_records, bh_crs = serialize_geodataframe(boreholes_gdf)
    zones_records, zones_crs = serialize_geodataframe(zones_gdf)

    # Convert sets to lists for pickling
    spt_list = list(test_data_locations.get("spt", set()))
    txt_list = list(test_data_locations.get("triaxial_total", set()))
    txe_list = list(test_data_locations.get("triaxial_effective", set()))

    logger.info(f"   Boreholes: {len(bh_records)} records")
    logger.info(f"   Zones: {len(zones_records)} records")
    logger.info(
        f"   Test data: SPT={len(spt_list)}, TxT={len(txt_list)}, TxE={len(txe_list)}"
    )

    # Build solver config using shared helper
    ilp_config = _build_solver_config(config)

    # Add zone cache directory to solver config (for intra-run caching)
    ilp_config["zone_cache_dir"] = zone_cache_dir

    # Add CZRC cache directory to solver config (for cross-worker CZRC caching)
    ilp_config["czrc_cache_dir"] = czrc_cache_dir

    logger.info(
        f"   Solver config: mode={ilp_config.get('solver_mode')}, "
        f"time_limit={ilp_config.get('time_limit_s')}s, mip_gap={ilp_config.get('mip_gap')}"
    )

    try:
        dispatch_start = time.time()
        results_list = list(
            Parallel(
                n_jobs=n_workers,
                backend=parallel_config.get("backend", "loky"),
                verbose=parallel_config.get("verbose", 10),
            )(
                delayed(worker_process_filter_combination)(
                    combo_key=get_combination_key(combo),
                    combo=combo,
                    boreholes_records=bh_records,
                    boreholes_crs=bh_crs,
                    zones_records=zones_records,
                    zones_crs=zones_crs,
                    max_spacing=max_spacing,
                    ilp_config=ilp_config,
                    spt_locations=spt_list,
                    triaxial_total_locations=txt_list,
                    triaxial_effective_locations=txe_list,
                    highs_log_folder=highs_log_folder,
                )
                for combo in combinations
            )
        )
        dispatch_time = time.time() - dispatch_start
        logger.info(f"   ⏱️ Parallel dispatch completed in {dispatch_time:.1f}s")

    except (ImportError, RuntimeError, OSError) as e:
        logger.warning(f"⚠️ Parallel dispatch failed: {e}")
        if parallel_config.get("fallback_on_error", True):
            logger.info("📋 Falling back to inline sequential processing...")
            # Inline sequential fallback - no separate function needed
            results_list = []
            for i, combo in enumerate(combinations):
                combo_key = get_combination_key(combo)
                logger.info(f"📋 Processing {i+1}/{len(combinations)}: {combo_key}")
                result = worker_process_filter_combination(
                    combo_key=combo_key,
                    combo=combo,
                    boreholes_records=bh_records,
                    boreholes_crs=bh_crs,
                    zones_records=zones_records,
                    zones_crs=zones_crs,
                    max_spacing=max_spacing,
                    ilp_config=ilp_config,
                    spt_locations=spt_list,
                    triaxial_total_locations=txt_list,
                    triaxial_effective_locations=txe_list,
                    highs_log_folder=highs_log_folder,
                )
                results_list.append(result)
        else:
            raise
    finally:
        # Log zone cache statistics before cleanup (only if enabled)
        if zone_cache_dir is not None:
            try:
                cache_files = list(Path(zone_cache_dir).glob("*.pkl"))
                lock_files = list(Path(zone_cache_dir).glob("*.lock"))
                if cache_files:
                    logger.info(
                        f"   📊 Zone cache summary: {len(cache_files)} unique zone-gap "
                        f"geometries cached (duplicate filter combos reused cached results)"
                    )
            except Exception:
                pass  # Non-critical

            # Clean up zone cache directory
            try:
                shutil.rmtree(zone_cache_dir)
                logger.debug(f"   🧹 Cleaned up zone cache: {zone_cache_dir}")
            except Exception as cleanup_err:
                logger.warning(f"⚠️ Failed to cleanup zone cache: {cleanup_err}")

        # Log CZRC cache statistics and cleanup (only if enabled)
        if czrc_cache_dir is not None:
            try:
                cache_files = list(Path(czrc_cache_dir).glob("*.pkl"))
                lock_files = list(Path(czrc_cache_dir).glob("*.lock"))
                if cache_files:
                    logger.info(
                        f"   📊 CZRC cache summary: {len(cache_files)} unique CZRC "
                        f"problems cached (cross-worker sharing enabled)"
                    )
            except Exception:
                pass  # Non-critical

            # Clean up CZRC cache directory
            try:
                shutil.rmtree(czrc_cache_dir)
                logger.debug(f"   🧹 Cleaned up CZRC cache: {czrc_cache_dir}")
            except Exception as cleanup_err:
                logger.warning(f"⚠️ Failed to cleanup CZRC cache: {cleanup_err}")

    return _collect_results(results_list)


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _build_solver_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build solver configuration dict from CONFIG.

    Centralizes solver config building for parallel dispatch.

    Args:
        config: Full CONFIG dict

    Returns:
        Dict with solver settings (ilp_config)
    """
    ilp_config = config.get("ilp_solver", {}).copy()
    # Add greedy solver config
    ilp_config["greedy_solver"] = config.get("greedy_solver", {})
    # Add candidate grid config (hexagonal vs rectangular)
    ilp_config["candidate_grid"] = config.get("candidate_grid", {})
    # Add stall detection config for early termination
    ilp_config["stall_detection"] = ilp_config.get("stall_detection", {})

    # Compute spacing from multipliers (derived values)
    max_spacing = config.get("max_spacing_m", 100.0)
    candidate_mult = ilp_config.get("candidate_spacing_mult", 0.5)
    test_mult = ilp_config.get("test_spacing_mult", 0.2)
    ilp_config["candidate_spacing_m"] = max_spacing * candidate_mult
    ilp_config["test_spacing_m"] = max_spacing * test_mult

    # Get base solver mode from optimization section
    optimization_config = config.get("optimization", {})
    base_solver_mode = optimization_config.get("solver_mode", "ilp")

    # Apply parallel overrides (new format takes precedence)
    parallel_overrides = config.get("parallel_solver_overrides", {})
    if not parallel_overrides:
        # Fallback to legacy format
        parallel_overrides = config.get("parallel_ilp_overrides", {})

    # Apply numeric overrides
    for key in ["time_limit_s", "mip_gap"]:
        if key in parallel_overrides:
            ilp_config[key] = parallel_overrides[key]

    # Apply solver_mode override if specified
    parallel_solver_mode = parallel_overrides.get("solver_mode")
    if parallel_solver_mode is not None:
        ilp_config["solver_mode"] = parallel_solver_mode
    else:
        # Use base solver mode
        ilp_config["solver_mode"] = base_solver_mode

    # Extract conflict constraint settings (new feature for sparse layouts)
    ilp_config["use_conflict_constraints"] = ilp_config.get(
        "use_conflict_constraints", True
    )
    ilp_config["exclusion_factor"] = ilp_config.get("exclusion_factor", 0.8)
    ilp_config["max_conflict_pairs"] = ilp_config.get("max_conflict_pairs", 200000)

    # Apply verbose setting based on testing_mode
    # In production (testing_mode disabled), silence HiGHS output
    testing_mode = config.get("testing_mode", {})
    if not testing_mode.get("enabled", False):
        # Production mode: force verbose=0 (silent HiGHS)
        ilp_config["verbose"] = 0
    # Otherwise keep the configured verbose value (default 1 in config)

    # Add consolidation mode for second-pass optimization
    border_config = config.get("border_consolidation", {})
    ilp_config["consolidation_mode"] = border_config.get("mode", "disabled")

    # Pass second-pass specific settings (separate from first-pass ilp_solver)
    ilp_config["consolidation_config"] = {
        "time_limit": border_config.get("time_limit", 60),
        "mip_gap": border_config.get("mip_gap", 0.03),
        "coverage_target_pct": border_config.get("coverage_target_pct", 97.0),
        "use_conflict_constraints": border_config.get("use_conflict_constraints", True),
        "exclusion_factor": border_config.get("exclusion_factor", 0.8),
        "verbose": border_config.get("verbose", 1),
    }

    return ilp_config


# ═══════════════════════════════════════════════════════════════════════════
# 📦 RESULT COLLECTION
# ═══════════════════════════════════════════════════════════════════════════


def _collect_results(
    results_list: List[Optional[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate worker results into final dict keyed by combo_key.

    Args:
        results_list: List of result dicts from workers

    Returns:
        Dict mapping combo_key -> result dict
    """
    output = {}
    success_count = 0
    error_count = 0

    for result in results_list:
        if result is None:
            continue
        key = result.get("key", "unknown")
        output[key] = result

        if result.get("success"):
            success_count += 1
        else:
            error_count += 1
            logger.warning(f"⚠️ {key}: {result.get('error', 'Unknown error')}")

    logger.info(f"📦 Collected {success_count} successes, {error_count} errors")
    return output


# ═══════════════════════════════════════════════════════════════════════════
# 📦 MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Main entry point
    "precompute_all_coverages",
    # Parallel decision functions
    "should_use_parallel",
    "get_effective_worker_count",
    # Config functions (merged from coverage_config.py)
    "DEFAULT_PARALLEL_CONFIG",
    "get_parallel_config",
    "generate_filter_combinations",
    "get_combination_key",
    # Serialization functions (merged from coverage_serialization.py)
    "serialize_geodataframe",
    "deserialize_geodataframe",
    "serialize_geometry",
    "deserialize_geometry",
]
