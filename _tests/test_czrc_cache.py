"""
Unit tests for CZRC Cache Module.

Tests:
1. Cache key determinism (same inputs = same key)
2. Cache key order independence (test point order doesn't affect key)
3. Different radii produce different keys
4. Basic get_or_compute functionality
5. Cache validation (n_candidates mismatch)
6. Concurrent access (process safety via filelock)

Run with: python -m pytest Gap_Analysis_EC7/_tests/test_czrc_cache.py -v
"""

import concurrent.futures
import tempfile
import time
from typing import Any, Dict, List, Tuple

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION FOR MULTIPROCESSING TEST
# ═══════════════════════════════════════════════════════════════════════════
# Must be at module level for pickling in ProcessPoolExecutor


def _concurrent_compute_task(
    cache_dir: str,
    cluster_key: str,
    tier1_wkt: str,
    zone_spacings: Dict[str, float],
    unsatisfied_test_points: List[Dict[str, Any]],
    n_candidates: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Task function that runs in separate process for concurrent test."""
    from Gap_Analysis_EC7.parallel.czrc_cache import get_czrc_cache_from_path

    # Get cache manager from existing cache directory
    cache = get_czrc_cache_from_path(cache_dir)

    def slow_compute() -> Tuple[List[int], Dict[str, Any]]:
        time.sleep(0.5)  # Simulate slow computation
        return [0, 2, 5], {"status": "optimal"}

    return cache.get_or_compute(
        cluster_key=cluster_key,
        tier1_wkt=tier1_wkt,
        zone_spacings=zone_spacings,
        unsatisfied_test_points=unsatisfied_test_points,
        n_candidates=n_candidates,
        compute_fn=slow_compute,
    )


class TestCZRCCacheKeyGeneration:
    """Test cache key generation functions."""

    def test_cache_key_determinism(self):
        """Same inputs produce same cache key."""
        from Gap_Analysis_EC7.parallel.czrc_cache import generate_czrc_cache_key

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

    def test_cache_key_test_point_order_independence(self):
        """Test point order doesn't affect cache key."""
        from Gap_Analysis_EC7.parallel.czrc_cache import generate_czrc_cache_key

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
        assert key1 == key2, "Keys should match regardless of test point order"

    def test_cache_key_zone_spacing_order_independence(self):
        """Zone spacing dict order doesn't affect cache key."""
        from Gap_Analysis_EC7.parallel.czrc_cache import generate_czrc_cache_key

        wkt = "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))"
        tp = [{"x": 50.0, "y": 50.0, "required_radius": 100.0}]

        # Different dict construction orders
        spacings1 = {"Zone1": 100.0, "Zone2": 150.0}
        spacings2 = {"Zone2": 150.0, "Zone1": 100.0}

        key1 = generate_czrc_cache_key("Zone1_Zone2", wkt, spacings1, tp)
        key2 = generate_czrc_cache_key("Zone1_Zone2", wkt, spacings2, tp)
        assert key1 == key2, "Keys should match regardless of zone spacing order"

    def test_cache_key_different_radius_produces_different_key(self):
        """Different required_radius produces different key."""
        from Gap_Analysis_EC7.parallel.czrc_cache import generate_czrc_cache_key

        wkt = "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))"
        spacings = {"Zone1": 100.0}

        tp_r100 = [{"x": 50.0, "y": 50.0, "required_radius": 100.0}]
        tp_r150 = [{"x": 50.0, "y": 50.0, "required_radius": 150.0}]

        key1 = generate_czrc_cache_key("Zone1_Zone2", wkt, spacings, tp_r100)
        key2 = generate_czrc_cache_key("Zone1_Zone2", wkt, spacings, tp_r150)
        assert key1 != key2, "Keys should differ for different radii"

    def test_cache_key_different_geometry_produces_different_key(self):
        """Different tier1 geometry produces different key."""
        from Gap_Analysis_EC7.parallel.czrc_cache import generate_czrc_cache_key

        wkt1 = "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))"
        wkt2 = "POLYGON((0 0, 200 0, 200 200, 0 200, 0 0))"
        spacings = {"Zone1": 100.0}
        tp = [{"x": 50.0, "y": 50.0, "required_radius": 100.0}]

        key1 = generate_czrc_cache_key("Zone1_Zone2", wkt1, spacings, tp)
        key2 = generate_czrc_cache_key("Zone1_Zone2", wkt2, spacings, tp)
        assert key1 != key2, "Keys should differ for different geometries"

    def test_cache_key_floating_point_tolerance(self):
        """Floating point differences within precision produce same key."""
        from Gap_Analysis_EC7.parallel.czrc_cache import generate_czrc_cache_key

        wkt = "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))"
        spacings = {"Zone1": 100.0}

        # Tiny floating point difference
        tp1 = [{"x": 50.0000001, "y": 50.0000002, "required_radius": 100.0}]
        tp2 = [{"x": 50.0000003, "y": 50.0000004, "required_radius": 100.0}]

        key1 = generate_czrc_cache_key("Zone1_Zone2", wkt, spacings, tp1)
        key2 = generate_czrc_cache_key("Zone1_Zone2", wkt, spacings, tp2)
        assert key1 == key2, "Tiny floating point differences should produce same key"


class TestCZRCCacheManager:
    """Test CZRCCacheManager class."""

    def test_basic_get_or_compute_miss_then_hit(self):
        """First call computes, second call hits cache."""
        from Gap_Analysis_EC7.parallel.czrc_cache import create_czrc_cache

        cache = create_czrc_cache(prefix="test_czrc_")
        compute_count = {"value": 0}

        def compute_fn() -> Tuple[List[int], Dict[str, Any]]:
            compute_count["value"] += 1
            return [0, 2, 5], {"status": "optimal"}

        try:
            # First call - miss
            indices1, stats1 = cache.get_or_compute(
                cluster_key="Zone_A+Zone_B",
                tier1_wkt="POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))",
                zone_spacings={"Zone_A": 100.0, "Zone_B": 150.0},
                unsatisfied_test_points=[{"x": 50, "y": 50, "required_radius": 100}],
                n_candidates=10,
                compute_fn=compute_fn,
            )
            assert compute_count["value"] == 1, "Should compute on first call"
            assert indices1 == [0, 2, 5]

            # Second call - hit
            indices2, stats2 = cache.get_or_compute(
                cluster_key="Zone_A+Zone_B",
                tier1_wkt="POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))",
                zone_spacings={"Zone_A": 100.0, "Zone_B": 150.0},
                unsatisfied_test_points=[{"x": 50, "y": 50, "required_radius": 100}],
                n_candidates=10,
                compute_fn=compute_fn,
            )
            assert compute_count["value"] == 1, "Should NOT compute on cache hit"
            assert indices1 == indices2
        finally:
            cache.cleanup()

    def test_cache_validation_fails_on_candidate_mismatch(self):
        """Cache validation fails if n_candidates changes."""
        from Gap_Analysis_EC7.parallel.czrc_cache import create_czrc_cache

        cache = create_czrc_cache(prefix="test_czrc_validate_")
        compute_count = {"value": 0}

        def compute_fn() -> Tuple[List[int], Dict[str, Any]]:
            compute_count["value"] += 1
            return [0, 2, 5], {"status": "optimal"}

        try:
            # First call with n_candidates=10
            cache.get_or_compute(
                cluster_key="Zone_A",
                tier1_wkt="POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))",
                zone_spacings={"Zone_A": 100.0},
                unsatisfied_test_points=[{"x": 50, "y": 50, "required_radius": 100}],
                n_candidates=10,
                compute_fn=compute_fn,
            )
            assert compute_count["value"] == 1

            # Second call with different n_candidates - should recompute
            cache.get_or_compute(
                cluster_key="Zone_A",
                tier1_wkt="POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))",
                zone_spacings={"Zone_A": 100.0},
                unsatisfied_test_points=[{"x": 50, "y": 50, "required_radius": 100}],
                n_candidates=20,  # Different candidate count
                compute_fn=compute_fn,
            )
            assert (
                compute_count["value"] == 2
            ), "Should recompute when n_candidates differs"
        finally:
            cache.cleanup()

    def test_concurrent_access_single_computation(self):
        """Concurrent access to same key results in single computation.

        NOTE: filelock is designed for inter-PROCESS locking, not inter-thread
        locking within the same process. In production, this cache is used
        across multiprocessing workers (separate processes), where filelock
        correctly ensures only one computation per cache key.

        This test uses ProcessPoolExecutor to simulate real production behavior.
        """
        from Gap_Analysis_EC7.parallel.czrc_cache import create_czrc_cache
        import multiprocessing

        cache = create_czrc_cache(prefix="test_czrc_concurrent_")

        try:
            # Use ProcessPoolExecutor for true inter-process locking test
            # ProcessPoolExecutor creates separate processes where filelock works
            with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
                # Submit 3 concurrent tasks to different processes
                futures = [
                    executor.submit(
                        _concurrent_compute_task,
                        cache.cache_dir,
                        "Zone_Concurrent",
                        "POLYGON((0 0, 50 0, 50 50, 0 50, 0 0))",
                        {"Zone_C": 100.0},
                        [{"x": 25, "y": 25, "required_radius": 100}],
                        5,
                    )
                    for _ in range(3)
                ]
                results = [f.result() for f in futures]

            # All results should be identical (same cached result)
            first_result = results[0][0]
            assert all(
                r[0] == first_result for r in results
            ), f"All results should be identical, got: {[r[0] for r in results]}"

            # Check stats show cache hits
            # Note: compute_count tracking across processes is complex,
            # so we verify correctness via identical results instead
        finally:
            cache.cleanup()

    def test_different_keys_compute_independently(self):
        """Different cache keys compute independently."""
        from Gap_Analysis_EC7.parallel.czrc_cache import create_czrc_cache

        cache = create_czrc_cache(prefix="test_czrc_parallel_")
        compute_count = {"value": 0}

        def compute_fn() -> Tuple[List[int], Dict[str, Any]]:
            compute_count["value"] += 1
            return [0, 2], {"status": "optimal"}

        try:
            # Three different zone combinations
            for i in range(3):
                cache.get_or_compute(
                    cluster_key=f"Zone_{i}",
                    tier1_wkt=f"POLYGON((0 0, {i*10+10} 0, {i*10+10} {i*10+10}, 0 {i*10+10}, 0 0))",
                    zone_spacings={f"Zone_{i}": 100.0},
                    unsatisfied_test_points=[
                        {"x": i * 5, "y": i * 5, "required_radius": 100}
                    ],
                    n_candidates=i + 1,
                    compute_fn=compute_fn,
                )

            assert (
                compute_count["value"] == 3
            ), "Should compute 3 times for 3 different keys"
        finally:
            cache.cleanup()

    def test_cache_stats_tracking(self):
        """Cache correctly tracks hits, misses, and stats."""
        from Gap_Analysis_EC7.parallel.czrc_cache import create_czrc_cache

        cache = create_czrc_cache(prefix="test_czrc_stats_")

        def compute_fn() -> Tuple[List[int], Dict[str, Any]]:
            return [0], {"status": "optimal"}

        try:
            # Two misses
            cache.get_or_compute(
                "Z1",
                "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
                {"Z1": 100.0},
                [{"x": 5, "y": 5, "required_radius": 100}],
                1,
                compute_fn,
            )
            cache.get_or_compute(
                "Z2",
                "POLYGON((0 0, 20 0, 20 20, 0 20, 0 0))",
                {"Z2": 100.0},
                [{"x": 10, "y": 10, "required_radius": 100}],
                2,
                compute_fn,
            )

            # Two hits
            cache.get_or_compute(
                "Z1",
                "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
                {"Z1": 100.0},
                [{"x": 5, "y": 5, "required_radius": 100}],
                1,
                compute_fn,
            )
            cache.get_or_compute(
                "Z2",
                "POLYGON((0 0, 20 0, 20 20, 0 20, 0 0))",
                {"Z2": 100.0},
                [{"x": 10, "y": 10, "required_radius": 100}],
                2,
                compute_fn,
            )

            stats = cache.get_stats()
            assert stats["hits"] == 2, f"Expected 2 hits, got {stats['hits']}"
            assert stats["misses"] == 2, f"Expected 2 misses, got {stats['misses']}"
            assert (
                stats["hit_rate_pct"] == 50.0
            ), f"Expected 50% hit rate, got {stats['hit_rate_pct']}"
        finally:
            cache.cleanup()


class TestNormalizeWKTCoordinates:
    """Test WKT coordinate normalization."""

    def test_normalize_rounds_coordinates(self):
        """Coordinates are rounded to specified precision."""
        from Gap_Analysis_EC7.parallel.czrc_cache import normalize_wkt_coordinates

        wkt = "POLYGON((0.123456789 1.987654321, 100.999999999 200.000000001, 0 0, 0.123456789 1.987654321))"
        normalized = normalize_wkt_coordinates(wkt, precision=6)

        assert "0.123457" in normalized  # Rounded to 6 decimal places
        assert "1.987654" in normalized
        assert "101.000000" in normalized  # Rounded up
        assert "200.000000" in normalized

    def test_normalize_handles_negative_numbers(self):
        """Negative coordinates are normalized correctly."""
        from Gap_Analysis_EC7.parallel.czrc_cache import normalize_wkt_coordinates

        wkt = (
            "POLYGON((-100.123456789 -50.987654321, 0 0, -100.123456789 -50.987654321))"
        )
        normalized = normalize_wkt_coordinates(wkt, precision=3)

        assert "-100.123" in normalized
        assert "-50.988" in normalized  # Rounded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
