"""
Intra-Run Zone Cache Module

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Cache ILP solver results at zone-gap granularity within a
single production run. Eliminates redundant computation when multiple filter
combinations produce identical gap geometries for the same zone.

Key Insight: Within a single run, all CONFIG parameters are CONSTANT. Only
filtered boreholes (and thus coverage/gaps) vary between filter combinations.
This means cache key = zone_name + gap_geometry_hash is sufficient.

Key Functions:
- ZoneCacheManager: Thread-safe cache manager with per-key file locking
- get_or_compute(): Atomic check→compute→save pattern to prevent race conditions
- create_zone_cache(): Factory function to create cache manager

Concurrency Model:
- 14 joblib workers (loky backend) access shared cache directory
- Per-key filelock prevents duplicate computation for same zone-gap
- Workers on different zone-gaps run in parallel (no global lock)
- 10-minute lock timeout with graceful fallback

Cache Format:
- Pickle files: {cache_key}.pkl containing solver result dict
- Lock files: {cache_key}.lock for cross-process coordination
- Automatic cleanup when run completes (temp directory)

Logging Note:
- This module uses print() instead of logger.info() for cache hit/miss messages
- REASON: joblib loky backend runs workers in separate processes where logging
  handlers are not inherited. logger.info() output is lost, but print() to
  stdout IS captured by joblib and displayed alongside HiGHS solver output.
- This is a documented limitation of Python multiprocessing + logging.
- See: https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

import hashlib
import logging
import os
import pickle
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import filelock

logger = logging.getLogger("EC7.Parallel.ZoneCache")


# ═══════════════════════════════════════════════════════════════════════════
# 📊 CACHE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ZoneCacheStats:
    """Statistics for zone cache performance tracking."""

    hits: int = 0
    misses: int = 0
    lock_timeouts: int = 0
    corrupted_entries: int = 0
    total_compute_time_s: float = 0.0
    total_wait_time_s: float = 0.0

    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for logging."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "lock_timeouts": self.lock_timeouts,
            "corrupted_entries": self.corrupted_entries,
            "hit_rate_pct": round(self.hit_rate(), 1),
            "total_compute_time_s": round(self.total_compute_time_s, 2),
            "total_wait_time_s": round(self.total_wait_time_s, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 🔑 CACHE KEY GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def normalize_wkt_coordinates(wkt: str, precision: int = 6) -> str:
    """
    Normalize WKT coordinates to prevent floating-point hash misses.

    Rounds all numeric values in WKT string to specified decimal precision
    to ensure geometrically identical polygons produce identical hashes.

    Args:
        wkt: WKT string (e.g., "POLYGON((0.123456789 1.23456789, ...))")
        precision: Decimal places to round to (default: 6 = ~0.1m accuracy)

    Returns:
        Normalized WKT string with rounded coordinates.
    """
    import re

    def round_match(match: re.Match) -> str:
        value = float(match.group(0))
        return f"{value:.{precision}f}"

    # Match floating point numbers (including negative and scientific notation)
    pattern = r"-?\d+\.?\d*(?:[eE][+-]?\d+)?"
    return re.sub(pattern, round_match, wkt)


def generate_cache_key(zone_name: str, gap_wkt: str) -> str:
    """
    Generate deterministic cache key from zone name and gap geometry.

    The key is an 8-character prefix of SHA256 hash of normalized inputs.
    This is sufficient for uniqueness within a single run while keeping
    filenames manageable.

    Args:
        zone_name: Zone identifier (e.g., "Zone_1_Central")
        gap_wkt: WKT string of gap polygon (union of all gaps in zone)

    Returns:
        Cache key string: "{zone_name}_{hash8}"
    """
    # Normalize zone name (remove special characters, lowercase)
    safe_zone = "".join(c if c.isalnum() else "_" for c in zone_name).lower()

    # Normalize WKT coordinates for consistent hashing
    normalized_wkt = normalize_wkt_coordinates(gap_wkt)

    # Combine and hash
    combined = f"{safe_zone}|{normalized_wkt}"
    hash_full = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    hash_prefix = hash_full[:8]

    return f"{safe_zone}_{hash_prefix}"


# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ ZONE CACHE MANAGER
# ═══════════════════════════════════════════════════════════════════════════


class ZoneCacheManager:
    """
    Manages intra-run zone result caching with per-key file locking.

    This class provides atomic cache access that eliminates race conditions
    when multiple joblib workers request the same zone-gap result. The
    get_or_compute() method holds a file lock during the entire check→compute→save
    cycle, ensuring only one worker computes while others wait and reuse.

    Attributes:
        cache_dir: Path to cache directory (typically tempfile.mkdtemp())
        lock_timeout_s: Seconds to wait for lock before computing without cache
        stats: ZoneCacheStats tracking hits, misses, timeouts

    Example:
        cache_dir = tempfile.mkdtemp(prefix="zone_cache_")
        cache = ZoneCacheManager(cache_dir)

        result = cache.get_or_compute(
            zone_name="Zone_A",
            gap_wkt="POLYGON((...)",
            compute_fn=lambda: expensive_ilp_solve(...)
        )
    """

    def __init__(
        self,
        cache_dir: str,
        lock_timeout_s: float = 600.0,  # 10 minutes
    ):
        """
        Initialize zone cache manager.

        Args:
            cache_dir: Absolute path to cache directory (must exist)
            lock_timeout_s: Seconds to wait for lock (default: 600 = 10 min)
        """
        self.cache_dir = Path(cache_dir)
        self.lock_timeout_s = lock_timeout_s
        self.stats = ZoneCacheStats()

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"ZoneCacheManager initialized: {self.cache_dir}")

    def get_or_compute(
        self,
        zone_name: str,
        gap_wkt: str,
        compute_fn: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Atomic cache access with computation fallback.

        This method ELIMINATES race conditions by holding the lock during
        the entire check→compute→save cycle. When multiple workers request
        the same zone-gap result:
        1. First worker acquires lock, checks cache (miss), computes, saves
        2. Other workers block on lock acquisition
        3. When lock released, other workers check cache (hit), return result

        Args:
            zone_name: Zone identifier (e.g., "Zone_1_Central")
            gap_wkt: WKT string of gap polygon (will be normalized & hashed)
            compute_fn: Zero-argument callable that computes result if cache miss

        Returns:
            Cached or freshly computed result dict containing:
            - "boreholes": List[Dict] of borehole placements
            - "stats": Dict of solver statistics
        """
        cache_key = generate_cache_key(zone_name, gap_wkt)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        lock_file = self.cache_dir / f"{cache_key}.lock"

        lock = filelock.FileLock(str(lock_file), timeout=self.lock_timeout_s)
        lock_wait_start = time.perf_counter()

        try:
            with lock:
                lock_wait_time = time.perf_counter() - lock_wait_start
                self.stats.total_wait_time_s += lock_wait_time

                # === STEP 1: Check cache (inside lock) ===
                if cache_file.exists():
                    try:
                        with open(cache_file, "rb") as f:
                            result = pickle.load(f)
                        self.stats.hits += 1
                        # Use print() for visibility in joblib worker processes
                        # (logger.info may not propagate from loky workers)
                        print(
                            f"   ♻️ REUSING cached result for {zone_name} "
                            f"(skipping ILP solve)"
                        )
                        logger.debug(
                            f"   Cache details: key={cache_key}, waited={lock_wait_time:.2f}s"
                        )
                        return result
                    except Exception as e:
                        # Corrupted cache entry - delete and recompute
                        logger.warning(f"⚠️ Corrupted cache entry for {zone_name}: {e}")
                        self.stats.corrupted_entries += 1
                        cache_file.unlink(missing_ok=True)

                # === STEP 2: Compute (inside lock - other workers wait) ===
                self.stats.misses += 1
                # Use print() for visibility in joblib worker processes
                print(f"   📊 Computing ILP for {zone_name} (not cached yet)...")

                compute_start = time.perf_counter()
                result = compute_fn()
                compute_time = time.perf_counter() - compute_start
                self.stats.total_compute_time_s += compute_time

                # === STEP 3: Save to cache (inside lock) ===
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump(result, f)
                    print(f"   💾 Cached {zone_name} for reuse ({compute_time:.1f}s)")
                except Exception as e:
                    # Cache write failure is non-fatal - log and continue
                    logger.warning(f"⚠️ Failed to cache {zone_name}: {e}")

                return result

        except filelock.Timeout:
            # Lock acquisition timed out - compute without caching
            logger.warning(
                f"⏱️ Lock timeout for {zone_name} after {self.lock_timeout_s}s "
                f"- computing without cache"
            )
            self.stats.lock_timeouts += 1
            return compute_fn()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics as dictionary."""
        return self.stats.to_dict()

    def log_summary(self, level: int = logging.INFO) -> None:
        """Log cache statistics summary."""
        stats = self.stats
        logger.log(
            level,
            f"Zone Cache Summary: {stats.hits} hits, {stats.misses} misses "
            f"({stats.hit_rate():.1f}% hit rate), "
            f"{stats.lock_timeouts} timeouts, "
            f"compute={stats.total_compute_time_s:.1f}s, "
            f"wait={stats.total_wait_time_s:.1f}s",
        )

    def cleanup(self) -> None:
        """
        Clean up cache directory.

        Note: If using tempfile.mkdtemp(), cleanup is automatic. This method
        is provided for explicit cleanup if needed.
        """
        import shutil

        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                logger.debug(f"Cleaned up cache directory: {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup cache directory: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 🏭 FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def create_zone_cache(
    prefix: str = "zone_cache_",
    lock_timeout_s: float = 600.0,
) -> ZoneCacheManager:
    """
    Create a new zone cache manager with temporary directory.

    This factory function creates a fresh temp directory for the cache,
    which is automatically cleaned up when the process exits or when
    cleanup() is called explicitly.

    Args:
        prefix: Prefix for temp directory name (default: "zone_cache_")
        lock_timeout_s: Lock timeout in seconds (default: 600)

    Returns:
        Configured ZoneCacheManager instance.

    Example:
        cache = create_zone_cache()
        try:
            # Use cache in workers...
            result = cache.get_or_compute(zone, gap_wkt, solver_fn)
        finally:
            cache.cleanup()  # Or rely on OS temp cleanup
    """
    cache_dir = tempfile.mkdtemp(prefix=prefix)
    logger.info(f"🗂️ Created zone cache directory: {cache_dir}")
    return ZoneCacheManager(cache_dir, lock_timeout_s)


def get_zone_cache_from_path(
    cache_dir: str,
    lock_timeout_s: float = 600.0,
) -> ZoneCacheManager:
    """
    Get zone cache manager for existing cache directory.

    This function is used by workers to connect to an existing cache
    directory created by the orchestrator and passed via config.

    Args:
        cache_dir: Absolute path to existing cache directory
        lock_timeout_s: Lock timeout in seconds (default: 600)

    Returns:
        ZoneCacheManager instance for the existing directory.
    """
    return ZoneCacheManager(cache_dir, lock_timeout_s)


# ═══════════════════════════════════════════════════════════════════════════
# 🧪 SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    # Basic functionality test
    import concurrent.futures

    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("Zone Cache Self-Test")
    print("=" * 60)

    # Test 1: Basic get_or_compute
    print("\n1. Testing basic get_or_compute...")
    cache = create_zone_cache(prefix="test_zone_cache_")

    compute_count = {"value": 0}

    def expensive_compute() -> Dict[str, Any]:
        compute_count["value"] += 1
        time.sleep(0.5)  # Simulate ILP solve
        return {"boreholes": [{"x": 100, "y": 200}], "stats": {"method": "ilp"}}

    # First call - should miss and compute
    result1 = cache.get_or_compute(
        zone_name="Zone_A",
        gap_wkt="POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))",
        compute_fn=expensive_compute,
    )
    assert compute_count["value"] == 1, "Should compute on first call"
    print(f"   First call: computed (count={compute_count['value']})")

    # Second call - should hit
    result2 = cache.get_or_compute(
        zone_name="Zone_A",
        gap_wkt="POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))",
        compute_fn=expensive_compute,
    )
    assert compute_count["value"] == 1, "Should NOT compute on cache hit"
    assert result1 == result2, "Results should be identical"
    print(f"   Second call: cache hit (count={compute_count['value']})")

    # Test 2: Concurrent access
    print("\n2. Testing concurrent access (10 threads, same key)...")
    compute_count["value"] = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                cache.get_or_compute,
                "Zone_B",
                "POLYGON((0 0, 50 0, 50 50, 0 50, 0 0))",
                expensive_compute,
            )
            for _ in range(10)
        ]
        results = [f.result() for f in futures]

    assert (
        compute_count["value"] == 1
    ), f"Should compute exactly once, got {compute_count['value']}"
    assert all(r == results[0] for r in results), "All results should be identical"
    print(f"   10 concurrent requests, computed {compute_count['value']} time(s)")

    # Test 3: Different keys run in parallel
    print("\n3. Testing parallel computation on different keys...")
    compute_count["value"] = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                cache.get_or_compute,
                f"Zone_{i}",
                f"POLYGON((0 0, {i*10} 0, {i*10} {i*10}, 0 {i*10}, 0 0))",
                expensive_compute,
            )
            for i in range(3)
        ]
        results = [f.result() for f in futures]

    assert (
        compute_count["value"] == 3
    ), f"Should compute 3 times for 3 different keys, got {compute_count['value']}"
    print(f"   3 different keys, computed {compute_count['value']} time(s)")

    # Stats
    print("\n4. Cache statistics:")
    cache.log_summary()
    stats = cache.get_stats()
    print(f"   {stats}")

    # Cleanup
    cache.cleanup()
    print("\n✅ All tests passed!")
