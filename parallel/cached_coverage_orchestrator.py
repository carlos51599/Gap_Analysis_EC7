"""
Cached wrapper for coverage pre-computation.

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Add caching layer around precompute_all_coverages().
This module provides a drop-in replacement that checks cache first.

Why a Wrapper (Not Direct Modification):
- Zero risk to existing working code
- Easy to disable caching by switching import
- Testable in isolation
- Clear separation of concerns

Usage:
    # Instead of:
    from Gap_Analysis_EC7.parallel.coverage_orchestrator import precompute_all_coverages

    # Use:
    from Gap_Analysis_EC7.parallel.cached_coverage_orchestrator import (
        precompute_all_coverages_cached,
    )

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Set, Union, Optional

import geopandas as gpd

# Phase 3: Import AppConfig for typed configuration support
from Gap_Analysis_EC7.config_types import AppConfig
from Gap_Analysis_EC7.parallel.coverage_orchestrator import precompute_all_coverages
from Gap_Analysis_EC7.parallel.coverage_cache import (
    DEFAULT_CACHE_CONFIG,
    compute_cache_fingerprint,
    load_from_cache,
    save_to_cache,
    prune_old_cache,
    get_cache_stats,
    build_solver_config_for_cache,
)

logger = logging.getLogger("EC7.Parallel.CachedOrchestrator")


# ═══════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIG NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def _normalize_config(config: Union[Dict[str, Any], AppConfig]) -> AppConfig:
    """
    Normalize config to AppConfig for internal use.

    Args:
        config: Either raw CONFIG dict or AppConfig object.

    Returns:
        AppConfig instance for typed access.
    """
    if isinstance(config, AppConfig):
        return config
    return AppConfig.from_dict(config)


# ═══════════════════════════════════════════════════════════════════════════
# 🔑 CACHE HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _get_cache_settings(
    config: Union[Dict[str, Any], AppConfig],
) -> tuple[bool, bool, Dict[str, Any]]:
    """
    Extract cache settings from config.

    Args:
        config: Full CONFIG dict or AppConfig object.

    Returns:
        Tuple of (cache_enabled, force_overwrite, cache_config).
    """
    app_config = _normalize_config(config)

    # Use typed cache config
    cache_enabled = app_config.cache.enabled
    force_overwrite = app_config.cache.force_overwrite

    # Build cache config dict for compatibility with existing code
    cache_config = {
        "enabled": app_config.cache.enabled,
        "cache_dir": app_config.cache.cache_dir,
        "max_cache_entries": app_config.cache.max_cache_entries,
        "max_cache_age_days": app_config.cache.max_cache_age_days,
    }

    # Check for testing mode force_cache_overwrite
    if app_config.testing_mode.enabled:
        if app_config.testing_mode.force_cache_overwrite:
            force_overwrite = True
            logger.info("🧪 Testing mode: force_cache_overwrite enabled")

    return cache_enabled, force_overwrite, cache_config


def _build_and_log_fingerprint(
    boreholes_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    test_data_locations: Dict[str, Set[str]],
    max_spacing: float,
    depth_step: float,
    config: Union[Dict[str, Any], AppConfig],
    force_overwrite: bool,
) -> str:
    """
    Build cache fingerprint and log cache check header.

    Args:
        boreholes_gdf: Boreholes GeoDataFrame.
        zones_gdf: Zone boundaries GeoDataFrame.
        test_data_locations: Test data location mapping.
        max_spacing: EC7 maximum spacing.
        depth_step: Depth slider step.
        config: Full CONFIG dict or AppConfig object.
        force_overwrite: Whether to force cache overwrite.

    Returns:
        Cache fingerprint string.
    """
    solver_config = build_solver_config_for_cache(config)

    fingerprint = compute_cache_fingerprint(
        boreholes_gdf=boreholes_gdf,
        zones_gdf=zones_gdf,
        test_data_locations=test_data_locations,
        max_spacing=max_spacing,
        depth_step=depth_step,
        solver_config=solver_config,
    )

    logger.info("=" * 60)
    logger.info("🔍 CHECKING COVERAGE CACHE")
    logger.info("=" * 60)
    logger.info("   Fingerprint: %s", fingerprint)
    if force_overwrite:
        logger.info("   ⚠️ FORCE OVERWRITE enabled - will recompute and overwrite cache")

    return fingerprint


def _log_cache_stats_and_try_load(
    cache_dir: Path,
    fingerprint: str,
    force_overwrite: bool,
) -> Dict[str, Dict[str, Any]] | None:
    """
    Report cache stats and attempt to load cached results.

    Args:
        cache_dir: Path to cache directory
        fingerprint: Cache fingerprint
        force_overwrite: Skip loading if True

    Returns:
        Cached results dict or None if cache miss/force overwrite
    """
    stats = get_cache_stats(cache_dir)
    if stats["exists"]:
        logger.info(
            "   Cache entries: %d (%s MB)", stats["entries"], stats["total_size_mb"]
        )

    # Skip load if force_overwrite
    if force_overwrite:
        return None

    return load_from_cache(cache_dir, fingerprint, logger)


def _save_and_prune_cache(
    cache_dir: Path,
    fingerprint: str,
    results: Dict[str, Dict[str, Any]],
    computation_time: float,
    max_spacing: float,
    depth_step: float,
    max_depth: float,
    config: Union[Dict[str, Any], AppConfig],
    boreholes_count: int,
    zones_count: int,
    cache_config: Dict[str, Any],
) -> None:
    """
    Save results to cache and prune old entries.

    Args:
        cache_dir: Path to cache directory.
        fingerprint: Cache fingerprint.
        results: Computed coverage results.
        computation_time: Seconds taken to compute.
        max_spacing: EC7 maximum spacing.
        depth_step: Depth slider step.
        max_depth: Maximum borehole depth.
        config: Full CONFIG dict or AppConfig object.
        boreholes_count: Number of boreholes.
        zones_count: Number of zones.
        cache_config: Cache configuration dict.
    """
    solver_config = build_solver_config_for_cache(config)
    metadata = {
        "max_spacing": max_spacing,
        "depth_step": depth_step,
        "max_depth": max_depth,
        "solver_mode": solver_config.get("solver_mode"),
        "computation_time_s": round(computation_time, 2),
        "combination_count": len(results),
        "boreholes_count": boreholes_count,
        "zones_count": zones_count,
    }
    save_to_cache(cache_dir, fingerprint, results, metadata, logger)

    prune_old_cache(
        cache_dir,
        max_entries=cache_config.get("max_cache_entries", 10),
        max_age_days=cache_config.get("max_cache_age_days", 30),
        log=logger,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 🚀 CACHED PRECOMPUTATION ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


def precompute_all_coverages_cached(
    boreholes_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    test_data_locations: Dict[str, Set[str]],
    max_spacing: float,
    max_depth: float,
    depth_step: float,
    config: Union[Dict[str, Any], AppConfig],
    workspace_root: Path,
    highs_log_folder: Optional[str] = None,
    centreline_boreholes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Pre-compute coverage zones for all filter combinations WITH CACHING.

    This is a wrapper around precompute_all_coverages() that:
    1. Computes a fingerprint of all cache-affecting inputs
    2. Checks if valid cached results exist
    3. Returns cached results on hit (FAST PATH - ~2 seconds)
    4. Computes and caches results on miss (SLOW PATH - ~5 minutes)

    Args:
        boreholes_gdf: All boreholes GeoDataFrame.
        zones_gdf: Zone boundaries GeoDataFrame.
        test_data_locations: Dict mapping test type -> set of Location IDs.
        max_spacing: EC7 maximum spacing in meters.
        max_depth: Maximum borehole depth.
        depth_step: Depth slider step from config.
        config: Full CONFIG dict or AppConfig object.
        workspace_root: Path to workspace root (for resolving cache dir).

    Returns:
        Dict mapping combo_key -> coverage data dict.
    """
    start_time = time.time()

    # Normalize to AppConfig for typed access
    app_config = _normalize_config(config)

    # Extract cache settings
    cache_enabled, force_overwrite, cache_config = _get_cache_settings(app_config)

    # Fast path: caching disabled
    if not cache_enabled:
        logger.info("📦 Caching disabled - running full computation")
        return precompute_all_coverages(
            boreholes_gdf=boreholes_gdf,
            zones_gdf=zones_gdf,
            test_data_locations=test_data_locations,
            max_spacing=max_spacing,
            max_depth=max_depth,
            depth_step=depth_step,
            config=app_config,
            highs_log_folder=highs_log_folder,
            centreline_boreholes=centreline_boreholes,
        )

    # Build fingerprint and log cache check header
    fingerprint = _build_and_log_fingerprint(
        boreholes_gdf,
        zones_gdf,
        test_data_locations,
        max_spacing,
        depth_step,
        app_config,
        force_overwrite,
    )

    # Resolve cache directory
    cache_dir_rel = cache_config.get("cache_dir", "Gap_Analysis_EC7/cache")
    cache_dir = workspace_root / cache_dir_rel

    # Try loading from cache
    cached_coverages = _log_cache_stats_and_try_load(
        cache_dir, fingerprint, force_overwrite
    )

    if cached_coverages is not None:
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("⚡ CACHE HIT - Skipping computation (%.2fs)", elapsed)
        logger.info("=" * 60)
        return cached_coverages

    # CACHE MISS - Run full computation (SLOW PATH)
    logger.info("   💨 Cache miss - computing coverage for all combinations...")

    results = precompute_all_coverages(
        boreholes_gdf=boreholes_gdf,
        zones_gdf=zones_gdf,
        test_data_locations=test_data_locations,
        max_spacing=max_spacing,
        max_depth=max_depth,
        depth_step=depth_step,
        centreline_boreholes=centreline_boreholes,
        config=app_config,
        highs_log_folder=highs_log_folder,
    )

    # Save to cache and prune old entries
    _save_and_prune_cache(
        cache_dir,
        fingerprint,
        results,
        time.time() - start_time,
        max_spacing,
        depth_step,
        max_depth,
        app_config,
        len(boreholes_gdf),
        len(zones_gdf),
        cache_config,
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 📦 MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "precompute_all_coverages_cached",
]
