"""
EC7 Coverage Parallel Processing Module

Provides parallel pre-computation of coverage zones for all filter combinations.
Follows Gap_Analysis parallel architecture patterns:
- Thin workers calling existing functions
- Always uses parallel infrastructure (n_jobs=1 for sequential)
- Serialization layer for GeoDataFrame transport

Module Structure (Consolidated Phase 4):
- coverage_orchestrator.py: Main orchestrator + config + serialization (merged)
- coverage_worker.py: Thin worker for single filter combination
- cached_coverage_orchestrator.py: Caching wrapper (decorator pattern)
- coverage_cache.py: Cache storage and fingerprinting
- zone_cache.py: Intra-run zone result caching (first-pass)
- czrc_cache.py: Intra-run CZRC result caching (second-pass)
"""

from Gap_Analysis_EC7.parallel.coverage_orchestrator import (
    # Orchestrator functions
    precompute_all_coverages,
    should_use_parallel,
    get_effective_worker_count,
    # Config functions (merged from coverage_config.py)
    generate_filter_combinations,
    get_combination_key,
    get_parallel_config,
    DEFAULT_PARALLEL_CONFIG,
    # Serialization functions (merged from coverage_serialization.py)
    serialize_geodataframe,
    deserialize_geodataframe,
    serialize_geometry,
    deserialize_geometry,
)

# Zone cache exports (first-pass ILP caching)
from Gap_Analysis_EC7.parallel.zone_cache import (
    ZoneCacheManager,
    ZoneCacheStats,
    create_zone_cache,
    get_zone_cache_from_path,
    generate_cache_key as generate_zone_cache_key,
)

# CZRC cache exports (second-pass ILP caching)
from Gap_Analysis_EC7.parallel.czrc_cache import (
    CZRCCacheManager,
    CZRCCacheStats,
    create_czrc_cache,
    get_czrc_cache_from_path,
    generate_czrc_cache_key,
)

__all__ = [
    # Orchestrator
    "precompute_all_coverages",
    "should_use_parallel",
    "get_effective_worker_count",
    # Config
    "generate_filter_combinations",
    "get_combination_key",
    "get_parallel_config",
    "DEFAULT_PARALLEL_CONFIG",
    # Serialization
    "serialize_geodataframe",
    "deserialize_geodataframe",
    "serialize_geometry",
    "deserialize_geometry",
    # Zone cache (first-pass)
    "ZoneCacheManager",
    "ZoneCacheStats",
    "create_zone_cache",
    "get_zone_cache_from_path",
    "generate_zone_cache_key",
    # CZRC cache (second-pass)
    "CZRCCacheManager",
    "CZRCCacheStats",
    "create_czrc_cache",
    "get_czrc_cache_from_path",
    "generate_czrc_cache_key",
]
