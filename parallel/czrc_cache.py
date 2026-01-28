"""
CZRC Intra-Run Cache Module

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Cache CZRC (Cross-Zone Redundancy Check) ILP solver results
at cluster-problem granularity within a single production run. Eliminates
redundant computation when multiple filter combinations produce equivalent
ILP problems (same unsatisfied test points, same candidates).

Key Insight: The CZRC ILP problem is fully determined by:
1. Tier 1 geometry (determines candidate grid)
2. Zone spacings (determines spacing constraints)
3. Unsatisfied test points (the actual coverage requirements)

Different first-pass boreholes can produce the SAME unsatisfied test points
(after pre-coverage from locked boreholes). Cache key must be the ACTUAL
problem definition, not the input path.

Key Functions:
- CZRCCacheManager: Thread-safe cache manager with per-key file locking
- get_or_compute(): Atomic check→compute→save pattern
- generate_czrc_cache_key(): Create deterministic key from problem definition
- create_czrc_cache(): Factory function to create cache manager

Concurrency Model:
- Each joblib worker gets its OWN cache instance (separate temp directory)
- Per-key filelock prevents duplicate computation for same problem
- Workers on different problems run in parallel (no global lock)
- 10-minute lock timeout with graceful fallback

Cache Format:
- Pickle files: {cache_key}.pkl containing (selected_indices, ilp_stats)
- Lock files: {cache_key}.lock for cross-process coordination
- Automatic cleanup when run completes (temp directory)

IMPORTANT DIFFERENCE FROM ZONE CACHE:
- Zone cache key = zone_name + gap_WKT (input-based)
- CZRC cache key = tier1_WKT + spacings + unsatisfied_points (problem-based)

This is because in CZRC, different locked borehole sets can produce the
same unsatisfied test points, and we want to cache based on the actual
ILP problem, not the path to derive it.

Logging Note:
- This module uses print() instead of logger.info() for cache hit/miss messages
- REASON: joblib loky backend runs workers in separate processes where logging
  handlers are not inherited. logger.info() output is lost, but print() to
  stdout IS captured by joblib and displayed alongside HiGHS solver output.

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

import hashlib
import json
import logging
import pickle
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import filelock

logger = logging.getLogger("EC7.Parallel.CZRCCache")


# ═══════════════════════════════════════════════════════════════════════════
# 📊 CACHE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CZRCCacheStats:
    """Statistics for CZRC cache performance tracking."""

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


def normalize_wkt_coordinates(wkt_str: str, precision: int = 6) -> str:
    """
    Normalize WKT coordinates to prevent floating-point hash misses.

    Rounds all numeric values in WKT string to specified decimal precision
    to ensure geometrically identical polygons produce identical hashes.

    Args:
        wkt_str: WKT string (e.g., "POLYGON((0.123456789 1.23456789, ...))")
        precision: Decimal places to round to (default: 6 = ~0.1m accuracy in BNG)

    Returns:
        Normalized WKT string with rounded coordinates.
    """

    def round_match(match: re.Match) -> str:
        value = float(match.group(0))
        return f"{value:.{precision}f}"

    # Match floating point numbers (including negative and scientific notation)
    pattern = r"-?\d+\.?\d*(?:[eE][+-]?\d+)?"
    return re.sub(pattern, round_match, wkt_str)


def generate_czrc_cache_key(
    cluster_key: str,
    tier1_wkt: str,
    zone_spacings: Dict[str, float],
    unsatisfied_test_points: List[Dict[str, Any]],
    precision: int = 6,
) -> str:
    """
    Generate deterministic cache key from CZRC problem definition.

    The key captures the actual ILP problem: tier1 region (determines
    candidate grid), zone spacings, and unsatisfied test points.

    IMPORTANT: The key is based on the ACTUAL problem (unsatisfied test
    points after pre-coverage), NOT the input data (first-pass boreholes).
    This means different first-pass borehole sets that produce the same
    unsatisfied test points will share the same cache key.

    Args:
        cluster_key: Cluster identifier (e.g., "Zone1_Zone2" or "Zone1+Zone2")
        tier1_wkt: WKT string of unified Tier 1 region
        zone_spacings: Dict mapping zone_name -> max_spacing_m
        unsatisfied_test_points: Test points not covered by locked boreholes,
            each with keys: x, y, required_radius
        precision: Decimal places for coordinate rounding (default: 6)

    Returns:
        Cache key string: "{safe_cluster_key}_{hash16}"

    Example:
        >>> key = generate_czrc_cache_key(
        ...     "Zone1_Zone2",
        ...     "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))",
        ...     {"Zone1": 100.0, "Zone2": 150.0},
        ...     [{"x": 50.0, "y": 50.0, "required_radius": 100.0}]
        ... )
        >>> print(key)  # "zone1_zone2_a1b2c3d4e5f6g7h8"
    """
    # 1. Safe cluster key (alphanumeric and underscores only)
    safe_cluster = "".join(c if c.isalnum() else "_" for c in cluster_key).lower()

    # 2. Normalize tier1 WKT coordinates to fixed precision
    tier1_normalized = normalize_wkt_coordinates(tier1_wkt, precision)

    # 3. Serialize zone spacings deterministically (sorted by key)
    spacings_sorted = sorted(zone_spacings.items())
    spacings_str = json.dumps(spacings_sorted)

    # 4. Serialize test points: sorted by (x, y), include required_radius
    # CRITICAL: Must include required_radius - different radii = different ILP problems
    point_strings = []
    for tp in unsatisfied_test_points:
        x = round(tp["x"], precision)
        y = round(tp["y"], precision)
        r = round(tp.get("required_radius", 100.0), 1)  # 1 decimal for radius
        point_strings.append(f"{x:.{precision}f},{y:.{precision}f},{r:.1f}")

    point_strings.sort()  # Alphabetical sort for determinism
    points_str = ";".join(point_strings)

    # 5. Combine and hash using SHA256 (stable across processes/runs)
    combined = f"{tier1_normalized}|{spacings_str}|{points_str}"
    hash_full = hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # DEBUG: Log cache key components for investigation
    # (Use print() for joblib worker visibility)
    n_test_pts = len(unsatisfied_test_points)
    tier1_wkt_preview = tier1_normalized[:80] + "..." if len(tier1_normalized) > 80 else tier1_normalized
    print(f"   🔑 CZRC key debug: cluster={cluster_key}, n_test_points={n_test_pts}, spacings={spacings_str}")
    print(f"   🔑 CZRC key debug: tier1_wkt={tier1_wkt_preview}")
    print(f"   🔑 CZRC key debug: hash={hash_full[:16]}")

    # 16-char prefix is sufficient for intra-run uniqueness
    return f"{safe_cluster}_{hash_full[:16]}"


# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ CZRC CACHE MANAGER
# ═══════════════════════════════════════════════════════════════════════════


class CZRCCacheManager:
    """
    Manages intra-run CZRC result caching with per-key file locking.

    This class provides atomic cache access that eliminates race conditions
    when multiple joblib workers request the same CZRC problem result. The
    get_or_compute() method holds a file lock during the entire check→compute→save
    cycle, ensuring only one worker computes while others wait and reuse.

    Attributes:
        cache_dir: Path to cache directory (typically tempfile.mkdtemp())
        lock_timeout_s: Seconds to wait for lock before computing without cache
        stats: CZRCCacheStats tracking hits, misses, timeouts

    Example:
        cache_dir = tempfile.mkdtemp(prefix="czrc_cache_")
        cache = CZRCCacheManager(cache_dir)

        def compute_ilp():
            # expensive ILP solve
            return [0, 2, 5], {"status": "optimal"}

        indices, stats = cache.get_or_compute(
            cluster_key="Zone1_Zone2",
            tier1_wkt="POLYGON(...)",
            zone_spacings={"Zone1": 100.0},
            unsatisfied_test_points=[{"x": 50, "y": 50, "required_radius": 100}],
            n_candidates=10,
            compute_fn=compute_ilp,
        )
    """

    def __init__(
        self,
        cache_dir: str,
        lock_timeout_s: float = 600.0,  # 10 minutes
    ):
        """
        Initialize CZRC cache manager.

        Args:
            cache_dir: Absolute path to cache directory (will be created if needed)
            lock_timeout_s: Seconds to wait for lock (default: 600 = 10 min)
        """
        self.cache_dir = Path(cache_dir)
        self.lock_timeout_s = lock_timeout_s
        self.stats = CZRCCacheStats()

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"CZRCCacheManager initialized: {self.cache_dir}")

    def get_or_compute(
        self,
        cluster_key: str,
        tier1_wkt: str,
        zone_spacings: Dict[str, float],
        unsatisfied_test_points: List[Dict[str, Any]],
        n_candidates: int,
        compute_fn: Callable[[], Tuple[List[int], Dict[str, Any]]],
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Atomic cache access with computation fallback.

        This method ELIMINATES race conditions by holding the lock during
        the entire check→compute→save cycle. When multiple workers request
        the same CZRC problem result:
        1. First worker acquires lock, checks cache (miss), computes, saves
        2. Other workers block on lock acquisition
        3. When lock released, other workers check cache (hit), return result

        Args:
            cluster_key: Cluster identifier for logging
            tier1_wkt: WKT of Tier 1 region (determines candidate grid)
            zone_spacings: Dict mapping zone_name -> max_spacing_m
            unsatisfied_test_points: Test points needing coverage
            n_candidates: Expected candidate count (for validation)
            compute_fn: Zero-arg callable returning (selected_indices, ilp_stats)

        Returns:
            Tuple of (selected_indices, ilp_stats)

        Raises:
            No exceptions — lock timeout falls back to uncached computation.
        """
        cache_key = generate_czrc_cache_key(
            cluster_key, tier1_wkt, zone_spacings, unsatisfied_test_points
        )
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
                            cached = pickle.load(f)

                        # Validate cached entry matches current problem dimensions
                        if self._validate_cache_entry(
                            cached, n_candidates, len(unsatisfied_test_points)
                        ):
                            self.stats.hits += 1
                            # Use print() for visibility in joblib worker processes
                            print(f"   ♻️ REUSING cached CZRC result for {cluster_key}")
                            return cached["selected_indices"], cached["ilp_stats"]
                        else:
                            # Validation failed — treat as cache miss
                            self.stats.corrupted_entries += 1
                            cache_file.unlink(missing_ok=True)

                    except Exception as e:
                        logger.warning(
                            f"⚠️ Corrupted CZRC cache entry for {cluster_key}: {e}"
                        )
                        self.stats.corrupted_entries += 1
                        cache_file.unlink(missing_ok=True)

                # === STEP 2: Compute (inside lock — other workers wait) ===
                self.stats.misses += 1
                # Use print() for visibility in joblib worker processes
                print(f"   📊 Computing CZRC ILP for {cluster_key} (not cached)...")

                compute_start = time.perf_counter()
                selected_indices, ilp_stats = compute_fn()
                compute_time = time.perf_counter() - compute_start
                self.stats.total_compute_time_s += compute_time

                # === STEP 3: Save to cache (inside lock) ===
                try:
                    entry = {
                        "selected_indices": selected_indices,
                        "ilp_stats": ilp_stats,
                        "n_candidates": n_candidates,
                        "n_test_points": len(unsatisfied_test_points),
                        "solve_time": compute_time,
                    }
                    with open(cache_file, "wb") as f:
                        pickle.dump(entry, f)
                    print(
                        f"   💾 Cached CZRC {cluster_key} for reuse ({compute_time:.1f}s)"
                    )
                except Exception as e:
                    # Cache write failure is non-fatal — log and continue
                    logger.warning(f"⚠️ Failed to cache CZRC {cluster_key}: {e}")

                return selected_indices, ilp_stats

        except filelock.Timeout:
            # Lock acquisition timed out — compute without caching
            logger.warning(
                f"⏱️ CZRC lock timeout for {cluster_key} after {self.lock_timeout_s}s "
                f"— computing without cache"
            )
            self.stats.lock_timeouts += 1
            return compute_fn()

    def _validate_cache_entry(
        self,
        cached: Dict[str, Any],
        expected_candidates: int,
        expected_test_points: int,
    ) -> bool:
        """
        Validate cached entry matches current problem dimensions.

        Returns False if indices could be invalid for current candidate count.
        This can happen if the candidate grid generation changed between caches.

        Args:
            cached: Cached entry dict
            expected_candidates: Current candidate count
            expected_test_points: Current unsatisfied test point count

        Returns:
            True if cache entry is valid, False otherwise.
        """
        if "selected_indices" not in cached or "ilp_stats" not in cached:
            return False

        # Check candidate count matches (grid is deterministic from tier1)
        cached_n_candidates = cached.get("n_candidates", -1)
        if cached_n_candidates != expected_candidates:
            return False

        # Check test point count (sanity check)
        cached_n_test_points = cached.get("n_test_points", -1)
        if cached_n_test_points != expected_test_points:
            return False

        # Check indices are in valid range
        indices = cached["selected_indices"]
        if indices and max(indices) >= expected_candidates:
            return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics as dictionary."""
        return self.stats.to_dict()

    def log_summary(self, level: int = logging.INFO) -> None:
        """Log cache statistics summary."""
        stats = self.stats
        logger.log(
            level,
            f"CZRC Cache Summary: {stats.hits} hits, {stats.misses} misses "
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
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                logger.debug(f"Cleaned up CZRC cache directory: {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup CZRC cache directory: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 🏭 FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def create_czrc_cache(
    prefix: str = "czrc_cache_",
    lock_timeout_s: float = 600.0,
) -> CZRCCacheManager:
    """
    Create a new CZRC cache manager with temporary directory.

    This factory function creates a fresh temp directory for the cache,
    which is automatically cleaned up when the process exits or when
    cleanup() is called explicitly.

    Args:
        prefix: Prefix for temp directory name (default: "czrc_cache_")
        lock_timeout_s: Lock timeout in seconds (default: 600)

    Returns:
        Configured CZRCCacheManager instance.

    Example:
        cache = create_czrc_cache()
        try:
            # Use cache in workers...
            result = cache.get_or_compute(...)
        finally:
            cache.cleanup()  # Or rely on OS temp cleanup
    """
    cache_dir = tempfile.mkdtemp(prefix=prefix)
    logger.info(f"🗂️ Created CZRC cache directory: {cache_dir}")
    return CZRCCacheManager(cache_dir, lock_timeout_s)


def get_czrc_cache_from_path(
    cache_dir: str,
    lock_timeout_s: float = 600.0,
) -> CZRCCacheManager:
    """
    Get CZRC cache manager for existing cache directory.

    This function is used by workers to connect to an existing cache
    directory created by the orchestrator and passed via config.

    Args:
        cache_dir: Absolute path to existing cache directory
        lock_timeout_s: Lock timeout in seconds (default: 600)

    Returns:
        CZRCCacheManager instance for the existing directory.
    """
    return CZRCCacheManager(cache_dir, lock_timeout_s)


# ═══════════════════════════════════════════════════════════════════════════
# 🧪 SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    # Basic functionality test
    import concurrent.futures

    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("CZRC Cache Self-Test")
    print("=" * 60)

    # Test 1: Cache key determinism
    print("\n1. Testing cache key determinism...")
    key1 = generate_czrc_cache_key(
        "Zone1_Zone2",
        "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))",
        {"Zone1": 100.0, "Zone2": 150.0},
        [{"x": 50.0, "y": 50.0, "required_radius": 100.0}],
    )
    key2 = generate_czrc_cache_key(
        "Zone1_Zone2",
        "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))",
        {"Zone1": 100.0, "Zone2": 150.0},
        [{"x": 50.0, "y": 50.0, "required_radius": 100.0}],
    )
    assert key1 == key2, f"Keys should match: {key1} vs {key2}"
    print(f"   Same inputs produce same key: {key1}")

    # Test 2: Cache key order independence (test points)
    print("\n2. Testing test point order independence...")
    tp1 = [
        {"x": 50.0, "y": 50.0, "required_radius": 100.0},
        {"x": 75.0, "y": 75.0, "required_radius": 100.0},
    ]
    tp2 = [
        {"x": 75.0, "y": 75.0, "required_radius": 100.0},
        {"x": 50.0, "y": 50.0, "required_radius": 100.0},
    ]
    wkt = "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))"
    spacings = {"Zone1": 100.0}

    key1 = generate_czrc_cache_key("Zone1_Zone2", wkt, spacings, tp1)
    key2 = generate_czrc_cache_key("Zone1_Zone2", wkt, spacings, tp2)
    assert key1 == key2, f"Keys should match regardless of test point order"
    print(f"   Test point order doesn't affect key: ✅")

    # Test 3: Different radius produces different key
    print("\n3. Testing different required_radius produces different key...")
    tp_r100 = [{"x": 50.0, "y": 50.0, "required_radius": 100.0}]
    tp_r150 = [{"x": 50.0, "y": 50.0, "required_radius": 150.0}]

    key1 = generate_czrc_cache_key("Zone1_Zone2", wkt, spacings, tp_r100)
    key2 = generate_czrc_cache_key("Zone1_Zone2", wkt, spacings, tp_r150)
    assert key1 != key2, f"Keys should differ for different radii"
    print(f"   Different radii produce different keys: ✅")

    # Test 4: Basic get_or_compute
    print("\n4. Testing basic get_or_compute...")
    cache = create_czrc_cache(prefix="test_czrc_cache_")

    compute_count = {"value": 0}

    def expensive_compute() -> Tuple[List[int], Dict[str, Any]]:
        compute_count["value"] += 1
        time.sleep(0.5)  # Simulate ILP solve
        return [0, 2, 5], {"status": "optimal", "solve_time": 0.5}

    # First call - should miss and compute
    indices1, stats1 = cache.get_or_compute(
        cluster_key="Zone_A+Zone_B",
        tier1_wkt="POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))",
        zone_spacings={"Zone_A": 100.0, "Zone_B": 150.0},
        unsatisfied_test_points=[{"x": 50, "y": 50, "required_radius": 100}],
        n_candidates=10,
        compute_fn=expensive_compute,
    )
    assert compute_count["value"] == 1, "Should compute on first call"
    assert indices1 == [0, 2, 5], f"Wrong indices: {indices1}"
    print(f"   First call: computed (count={compute_count['value']})")

    # Second call - should hit
    indices2, stats2 = cache.get_or_compute(
        cluster_key="Zone_A+Zone_B",
        tier1_wkt="POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))",
        zone_spacings={"Zone_A": 100.0, "Zone_B": 150.0},
        unsatisfied_test_points=[{"x": 50, "y": 50, "required_radius": 100}],
        n_candidates=10,
        compute_fn=expensive_compute,
    )
    assert compute_count["value"] == 1, "Should NOT compute on cache hit"
    assert indices1 == indices2, "Results should be identical"
    print(f"   Second call: cache hit (count={compute_count['value']})")

    # Test 5: Concurrent access
    print("\n5. Testing concurrent access (10 threads, same key)...")
    compute_count["value"] = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                cache.get_or_compute,
                "Zone_C+Zone_D",
                "POLYGON((0 0, 50 0, 50 50, 0 50, 0 0))",
                {"Zone_C": 100.0},
                [{"x": 25, "y": 25, "required_radius": 100}],
                5,
                expensive_compute,
            )
            for _ in range(10)
        ]
        results = [f.result() for f in futures]

    assert (
        compute_count["value"] == 1
    ), f"Should compute exactly once, got {compute_count['value']}"
    assert all(
        r[0] == results[0][0] for r in results
    ), "All results should be identical"
    print(f"   10 concurrent requests, computed {compute_count['value']} time(s)")

    # Test 6: Different keys run in parallel
    print("\n6. Testing parallel computation on different keys...")
    compute_count["value"] = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                cache.get_or_compute,
                f"Zone_{i}",
                f"POLYGON((0 0, {i*10} 0, {i*10} {i*10}, 0 {i*10}, 0 0))",
                {f"Zone_{i}": 100.0},
                [{"x": i * 5, "y": i * 5, "required_radius": 100}],
                i + 1,
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
    print("\n7. Cache statistics:")
    cache.log_summary()
    stats = cache.get_stats()
    print(f"   {stats}")

    # Cleanup
    cache.cleanup()
    print("\n✅ All tests passed!")
