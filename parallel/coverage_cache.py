"""
Coverage Cache Management Module

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Persist precomputed coverage results to disk.
Enables skipping expensive ILP/greedy computation when inputs unchanged.

Key Functions:
- compute_cache_fingerprint(): Generate deterministic hash from inputs
- load_from_cache(): Load cached results if valid
- save_to_cache(): Persist results to cache directory
- get_cache_stats(): Report cache status for logging
- invalidate_cache(): Clear specific or all cached results

Cache Format:
- Pickle files keyed by SHA256 fingerprint
- JSON manifest for human-readable cache registry
- Automatic pruning of old cache entries

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Set, List, Union

import geopandas as gpd

# Phase 3: Import AppConfig for typed configuration support
from Gap_Analysis_EC7.config_types import AppConfig

logger = logging.getLogger("EC7.Parallel.Cache")


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
# 📋 CACHE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CACHE_CONFIG: Dict[str, Any] = {
    "enabled": True,  # Master toggle for caching
    "cache_dir": "Gap_Analysis_EC7/cache",  # Relative to workspace root
    "max_cache_entries": 10,  # Max cached runs to keep
    "max_cache_age_days": 30,  # Auto-expire old cache entries
    "log_cache_hits": True,  # Log when cache is used
}


# ═══════════════════════════════════════════════════════════════════════════
# 🔑 FINGERPRINT COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════


def compute_cache_fingerprint(
    boreholes_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    test_data_locations: Dict[str, Set[str]],
    max_spacing: float,
    depth_step: float,
    solver_config: Dict[str, Any],
) -> str:
    """
    Compute deterministic fingerprint from all cache-affecting inputs.

    The fingerprint is a SHA256 hash of all inputs that affect the coverage
    computation results. If ANY of these inputs change, the fingerprint
    changes and the cache is invalidated.

    Args:
        boreholes_gdf: Boreholes GeoDataFrame with Location ID, Easting,
                       Northing, Final Depth columns
        zones_gdf: Zone boundaries GeoDataFrame with geometry column
        test_data_locations: Dict mapping test type -> set of Location IDs
                            Keys: "spt", "triaxial_total", "triaxial_effective"
        max_spacing: EC7 max spacing in meters (e.g., 200.0)
        depth_step: Depth slider step from config (e.g., 50.0)
        solver_config: Dict with solver parameters that affect results:
                      solver_mode, time_limit_s, mip_gap, candidate_spacing_m,
                      test_spacing_m, coverage_target_pct

    Returns:
        SHA256 hex string (first 16 chars for readability)

    Example:
        >>> fingerprint = compute_cache_fingerprint(
        ...     boreholes_gdf, zones_gdf, test_data_locations,
        ...     max_spacing=200.0, depth_step=50.0,
        ...     solver_config={"solver_mode": "ilp", "mip_gap": 0.03}
        ... )
        >>> len(fingerprint)
        16
    """
    hasher = hashlib.sha256()

    # === BOREHOLE DATA ===
    # Sort by Location ID for deterministic ordering
    bh_key_cols = ["Location ID", "Easting", "Northing", "Final Depth"]
    available_cols = [c for c in bh_key_cols if c in boreholes_gdf.columns]
    bh_subset = boreholes_gdf[available_cols].copy()
    if "Location ID" in bh_subset.columns:
        bh_subset = bh_subset.sort_values("Location ID").reset_index(drop=True)
    hasher.update(bh_subset.to_csv(index=False).encode("utf-8"))

    # === ZONE BOUNDARIES ===
    # Use WKT of zone geometries (sorted for determinism)
    zone_wkts = sorted(zones_gdf.geometry.apply(lambda g: g.wkt).tolist())
    hasher.update(json.dumps(zone_wkts).encode("utf-8"))

    # === TEST DATA LOCATIONS ===
    # Sort each set and convert to sorted list
    td_normalized: Dict[str, List[str]] = {}
    for k, v in test_data_locations.items():
        if isinstance(v, set):
            td_normalized[k] = sorted(list(v))
        elif isinstance(v, list):
            td_normalized[k] = sorted(v)
        else:
            td_normalized[k] = []
    hasher.update(json.dumps(td_normalized, sort_keys=True).encode("utf-8"))

    # === SPACING PARAMETERS ===
    hasher.update(f"max_spacing:{max_spacing}".encode("utf-8"))
    hasher.update(f"depth_step:{depth_step}".encode("utf-8"))

    # === SOLVER CONFIG ===
    # Only include cache-affecting solver parameters
    solver_keys = [
        "solver_mode",
        "time_limit_s",
        "mip_gap",
        "candidate_spacing_m",
        "test_spacing_m",
        "coverage_target_pct",
        # Conflict constraint settings (affect solution)
        "use_conflict_constraints",
        "conflict_constraint_mode",  # "clique" or "pairwise"
        "exclusion_factor",
        "min_clique_size",  # For clique mode
        "max_cliques",  # For clique mode
        "max_conflict_pairs",  # For pairwise mode
    ]
    solver_fingerprint = {
        k: solver_config.get(k) for k in solver_keys if k in solver_config
    }
    hasher.update(json.dumps(solver_fingerprint, sort_keys=True).encode("utf-8"))

    return hasher.hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# 💾 CACHE I/O
# ═══════════════════════════════════════════════════════════════════════════


def get_cache_path(
    cache_dir: Path,
    fingerprint: str,
) -> Path:
    """
    Get cache file path for given fingerprint.

    Args:
        cache_dir: Cache directory path
        fingerprint: Cache key fingerprint (16-char hex string)

    Returns:
        Path to cache file (may not exist yet)
    """
    return cache_dir / f"ec7_coverage_{fingerprint}.pkl"


def load_from_cache(
    cache_dir: Path,
    fingerprint: str,
    log: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Load cached coverage results if valid.

    Attempts to load and validate a cached result file. Returns None
    on any error (missing file, corruption, invalid format).

    Args:
        cache_dir: Cache directory path
        fingerprint: Cache key fingerprint
        log: Optional logger for status messages

    Returns:
        Dict of precomputed coverages (combo_key -> coverage_data),
        or None if cache miss or error
    """
    cache_path = get_cache_path(cache_dir, fingerprint)

    if not cache_path.exists():
        if log:
            log.info(f"   ❌ Cache miss: {fingerprint}")
        return None

    try:
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)

        # Validate structure
        if not isinstance(cached_data, dict):
            raise ValueError("Invalid cache format: not a dict")

        if "coverages" not in cached_data:
            raise ValueError("Invalid cache format: missing 'coverages' key")

        if log:
            timestamp = cached_data.get("timestamp", "unknown")
            combo_count = len(cached_data["coverages"])
            log.info(f"   ✅ Cache HIT: {fingerprint}")
            log.info(f"      Created: {timestamp}")
            log.info(f"      Combinations: {combo_count}")

        return cached_data["coverages"]

    except Exception as e:
        if log:
            log.warning(f"   ⚠️ Cache corrupted ({fingerprint}): {e}")
        # Delete corrupted cache file
        try:
            cache_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def save_to_cache(
    cache_dir: Path,
    fingerprint: str,
    coverages: Dict[str, Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    log: Optional[logging.Logger] = None,
) -> bool:
    """
    Save coverage results to cache.

    Creates the cache directory if needed and saves the results with
    metadata for debugging and cache management.

    Args:
        cache_dir: Cache directory path
        fingerprint: Cache key fingerprint
        coverages: Dict of precomputed coverage results
        metadata: Optional metadata (timing info, config snapshot)
        log: Optional logger

    Returns:
        True if save succeeded, False otherwise
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = get_cache_path(cache_dir, fingerprint)

        cached_data = {
            "fingerprint": fingerprint,
            "timestamp": datetime.now().isoformat(),
            "coverages": coverages,
            "metadata": metadata or {},
            "version": "1.0",
        }

        with open(cache_path, "wb") as f:
            pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if log:
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            log.info(f"   💾 Cache saved: {fingerprint} ({size_mb:.2f} MB)")

        # Update manifest (non-critical, ignore errors)
        _update_manifest(cache_dir, fingerprint, len(coverages))

        return True

    except Exception as e:
        if log:
            log.warning(f"   ⚠️ Failed to save cache: {e}")
        return False


def _update_manifest(cache_dir: Path, fingerprint: str, combo_count: int) -> None:
    """
    Update JSON manifest with new cache entry.

    The manifest is a human-readable registry of cache entries.
    It's optional and non-critical - errors are silently ignored.
    """
    manifest_path = cache_dir / "cache_manifest.json"

    try:
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        else:
            manifest = {"entries": [], "version": "1.0"}

        # Add/update entry
        entry = {
            "fingerprint": fingerprint,
            "created": datetime.now().isoformat(),
            "combinations": combo_count,
        }

        # Remove old entry if exists (update in place)
        manifest["entries"] = [
            e for e in manifest["entries"] if e.get("fingerprint") != fingerprint
        ]
        manifest["entries"].append(entry)

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    except Exception:
        pass  # Manifest is optional, don't fail on errors


# ═══════════════════════════════════════════════════════════════════════════
# 🧹 CACHE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════


def prune_old_cache(
    cache_dir: Path,
    max_entries: int = 10,
    max_age_days: int = 30,
    log: Optional[logging.Logger] = None,
) -> int:
    """
    Remove old cache entries to prevent unbounded growth.

    Keeps the most recent max_entries files and removes older ones.

    Args:
        cache_dir: Cache directory path
        max_entries: Maximum number of cache entries to keep
        max_age_days: Maximum age in days (currently unused, for future)
        log: Optional logger

    Returns:
        Number of entries pruned
    """
    if not cache_dir.exists():
        return 0

    cache_files = list(cache_dir.glob("ec7_coverage_*.pkl"))

    if len(cache_files) <= max_entries:
        return 0

    # Sort by modification time (oldest first)
    cache_files.sort(key=lambda p: p.stat().st_mtime)

    # Remove oldest entries beyond max_entries
    pruned = 0
    for cache_file in cache_files[:-max_entries]:
        try:
            cache_file.unlink()
            pruned += 1
        except Exception:
            pass

    if log and pruned > 0:
        log.info(f"   🧹 Pruned {pruned} old cache entries")

    return pruned


def invalidate_cache(
    cache_dir: Path,
    fingerprint: Optional[str] = None,
    log: Optional[logging.Logger] = None,
) -> int:
    """
    Invalidate (delete) cache entries.

    Can invalidate a specific entry by fingerprint, or all entries
    if fingerprint is None.

    Args:
        cache_dir: Cache directory path
        fingerprint: Specific fingerprint to invalidate, or None for all
        log: Optional logger

    Returns:
        Number of entries invalidated
    """
    if not cache_dir.exists():
        return 0

    if fingerprint:
        # Invalidate specific entry
        cache_path = get_cache_path(cache_dir, fingerprint)
        if cache_path.exists():
            try:
                cache_path.unlink()
                if log:
                    log.info(f"   🗑️ Invalidated cache: {fingerprint}")
                return 1
            except Exception:
                pass
        return 0

    # Invalidate all entries
    cache_files = list(cache_dir.glob("ec7_coverage_*.pkl"))
    invalidated = 0
    for cache_file in cache_files:
        try:
            cache_file.unlink()
            invalidated += 1
        except Exception:
            pass

    if log and invalidated > 0:
        log.info(f"   🗑️ Invalidated {invalidated} cache entries")

    return invalidated


def get_cache_stats(cache_dir: Path) -> Dict[str, Any]:
    """
    Get cache statistics for reporting.

    Args:
        cache_dir: Cache directory path

    Returns:
        Dict with cache statistics:
        - exists: bool - whether cache directory exists
        - entries: int - number of cached results
        - total_size_mb: float - total size in megabytes
        - cache_dir: str - cache directory path
    """
    if not cache_dir.exists():
        return {
            "exists": False,
            "entries": 0,
            "total_size_mb": 0.0,
            "cache_dir": str(cache_dir),
        }

    cache_files = list(cache_dir.glob("ec7_coverage_*.pkl"))
    total_size = sum(f.stat().st_size for f in cache_files)

    return {
        "exists": True,
        "entries": len(cache_files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "cache_dir": str(cache_dir),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 🔧 HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def build_solver_config_for_cache(
    config: Union[Dict[str, Any], AppConfig],
) -> Dict[str, Any]:
    """
    Extract solver configuration relevant to cache fingerprint.

    Combines settings from optimization, ilp_solver, parallel_solver_overrides,
    and testing_mode.solver_overrides config sections into a single dict for
    fingerprinting.

    Priority (highest to lowest):
    1. testing_mode.solver_overrides (if testing_mode.enabled)
    2. parallel_solver_overrides
    3. ilp_solver / optimization base settings

    Args:
        config: Full CONFIG dictionary or AppConfig object.

    Returns:
        Dict with cache-relevant solver settings.
    """
    # Normalize to AppConfig but use raw dict for extraction
    # (since we need access to nested ilp_solver settings not yet in typed config)
    app_config = _normalize_config(config)
    raw_config = app_config._raw_config

    solver_config: Dict[str, Any] = {}

    # Get solver mode from optimization section
    optimization = raw_config.get("optimization", {})
    solver_config["solver_mode"] = optimization.get("solver_mode", "ilp")

    # Get ILP solver settings
    ilp_config = raw_config.get("ilp_solver", {})
    solver_config["time_limit_s"] = ilp_config.get("time_limit_s", 120)
    solver_config["mip_gap"] = ilp_config.get("mip_gap", 0.03)

    # Compute spacing from multipliers (derived values)
    max_spacing = app_config.max_spacing_m
    candidate_mult = ilp_config.get("candidate_spacing_mult", 0.5)
    test_mult = ilp_config.get("test_spacing_mult", 0.2)
    solver_config["candidate_spacing_m"] = max_spacing * candidate_mult
    solver_config["test_spacing_m"] = max_spacing * test_mult
    solver_config["coverage_target_pct"] = ilp_config.get("coverage_target_pct", 100.0)

    # Apply parallel overrides if they'll be used (these are the effective
    # parallel context settings - always apply if solver_mode is set)
    parallel_overrides = app_config.parallel_solver_overrides
    if parallel_overrides.solver_mode:
        solver_config["solver_mode"] = parallel_overrides.solver_mode
        # When parallel mode is active, also apply its time/gap settings
        solver_config["time_limit_s"] = parallel_overrides.time_limit_s
        solver_config["mip_gap"] = parallel_overrides.mip_gap

    # Apply testing_mode overrides (highest priority when enabled)
    if app_config.testing_mode.enabled:
        testing_overrides = app_config.testing_mode.solver_overrides
        if testing_overrides.get("solver_mode"):
            solver_config["solver_mode"] = testing_overrides["solver_mode"]
        if testing_overrides.get("time_limit_s"):
            solver_config["time_limit_s"] = testing_overrides["time_limit_s"]
        if testing_overrides.get("mip_gap"):
            solver_config["mip_gap"] = testing_overrides["mip_gap"]

    return solver_config


# ═══════════════════════════════════════════════════════════════════════════
# 📦 MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "DEFAULT_CACHE_CONFIG",
    "compute_cache_fingerprint",
    "load_from_cache",
    "save_to_cache",
    "prune_old_cache",
    "invalidate_cache",
    "get_cache_stats",
    "get_cache_path",
    "build_solver_config_for_cache",
]
