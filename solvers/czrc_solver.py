"""
CZRC (Cross-Zone Redundancy Check) Second-Pass and Third-Pass Optimization.

Architectural Overview:
=======================
Responsibility: Second-pass and Third-pass optimization using precise
coverage-cloud-based regions for multi-zone and cell-cell borehole
consolidation. Uses mathematically rigorous tier-based classification.

Key Interactions:
    - Uses czrc_geometry.py for CZRC region computation
    - Reuses ILP infrastructure from solver_algorithms.py
    - Reuses helper functions from consolidation.py
    - Optional: czrc_cache.py for intra-run result caching

Third Pass Visualization Definitions:
    - GREY MARKERS: Second Pass OUTPUT (First Pass survivors + Second Pass
      additions) that fall within Third Pass Tier 1. These are the candidates
      that may be removed by Third Pass re-optimization.
    - LOCKED BOREHOLES: Second Pass OUTPUT that fall within Third Pass Tier 2
      (but outside Tier 1). These provide coverage context but are NOT
      re-optimized - they're treated as fixed constants during ILP.

Navigation Guide:
    1. run_czrc_optimization() - Main entry point (Second Pass)
    2. check_and_split_large_cluster() - Handles large clusters via cell splitting
    3. run_cell_czrc_pass() - Third Pass cell-cell CZRC
    4. solve_czrc_ilp_for_pair() - Per-pair ILP orchestration
    5. compute_czrc_tiers() - Tier boundary computation
    6. classify_first_pass_boreholes() - Borehole classification (Tier 1 vs Tier 2)

REUSED FUNCTIONS (from existing modules):
    - _solve_ilp() from solver_algorithms.py
    - _generate_candidate_grid() from optimization_geometry.py
    - _build_coverage_dict_variable_test_radii() from consolidation.py
    - _compute_locked_coverage() from consolidation.py
    - _dedupe_candidates() from consolidation.py

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import logging
import math
import time

import numpy as np
from shapely import wkt
from shapely.geometry import Point, Polygon, MultiPolygon, MultiPoint
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union, voronoi_diagram
from sklearn.cluster import KMeans

# REUSED IMPORTS - existing infrastructure
from Gap_Analysis_EC7.solvers.solver_algorithms import _solve_ilp
from Gap_Analysis_EC7.solvers.optimization_geometry import _generate_candidate_grid
from Gap_Analysis_EC7.solvers.consolidation import (
    _build_coverage_dict_variable_test_radii,
    _compute_locked_coverage,
    _dedupe_candidates,
    _filter_coincident_pairs,
)

# Type-only import to avoid circular imports
if TYPE_CHECKING:
    from Gap_Analysis_EC7.parallel.czrc_cache import CZRCCacheManager

# Module-level logger
_logger = logging.getLogger(__name__)


def _get_czrc_stall_detection_config(czrc_ilp_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve stall detection config for CZRC, respecting apply_to_czrc flag.

    When ilp_solver.stall_detection.apply_to_czrc is True (default), uses the
    first-pass stall detection settings. Otherwise uses CZRC-specific settings.

    Args:
        czrc_ilp_config: The czrc_optimization.ilp config section

    Returns:
        Resolved stall detection config dict
    """
    # Import CONFIG here to avoid circular imports
    from Gap_Analysis_EC7.config import CONFIG

    first_pass_stall = CONFIG.get("ilp_solver", {}).get("stall_detection", {})
    apply_to_czrc = first_pass_stall.get("apply_to_czrc", True)

    if apply_to_czrc:
        # Use first-pass settings (without the apply_to_czrc flag itself)
        return {
            "enabled": first_pass_stall.get("enabled", True),
            "gap_threshold_pct": first_pass_stall.get("gap_threshold_pct", 15.0),
            "warmup_seconds": first_pass_stall.get("warmup_seconds", 15.0),
            "comparison_seconds": first_pass_stall.get("comparison_seconds", 10.0),
            "min_improvement_pct": first_pass_stall.get("min_improvement_pct", 5.0),
        }
    else:
        # Use CZRC-specific settings
        czrc_stall = czrc_ilp_config.get("stall_detection", {})
        return {
            "enabled": czrc_stall.get("enabled", True),
            "gap_threshold_pct": czrc_stall.get("gap_threshold_pct", 15.0),
            "warmup_seconds": czrc_stall.get("warmup_seconds", 15.0),
            "comparison_seconds": czrc_stall.get("comparison_seconds", 10.0),
            "min_improvement_pct": czrc_stall.get("min_improvement_pct", 5.0),
        }


def _get_cross_zone_exclusion_method(config: Dict[str, Any]) -> str:
    """
    Resolve cross-zone exclusion method from config.

    CZRC can override the global ilp_solver setting. If CZRC config specifies
    None, falls back to the global ilp_solver.cross_zone_exclusion_method.

    Args:
        config: CZRC config section (czrc_optimization.ilp)

    Returns:
        Method string: "min", "max", or "average"
    """
    # Import CONFIG here to avoid circular imports
    from Gap_Analysis_EC7.config import CONFIG

    # Check for CZRC-specific override (None = inherit from ilp_solver)
    czrc_method = config.get("cross_zone_exclusion_method", None)
    if czrc_method is not None:
        return czrc_method

    # Fall back to global ilp_solver setting
    ilp_solver_config = CONFIG.get("ilp_solver", {})
    return ilp_solver_config.get("cross_zone_exclusion_method", "average")


def _aggregate_zone_spacings(
    zone_spacings: Dict[str, float],
    zones: List[str],
    method: str,
) -> float:
    """
    Aggregate zone spacings using the specified method.

    Used for cross-zone conflict constraints where boreholes from different
    zones may have different spacing requirements.

    Args:
        zone_spacings: Dict mapping zone_name -> max_spacing_m
        zones: List of zone names to aggregate
        method: Aggregation method ("min", "max", or "average")

    Returns:
        Aggregated spacing value

    Raises:
        ValueError: If method is not recognized
    """
    spacings = [zone_spacings.get(z, 100.0) for z in zones]
    if not spacings:
        return 100.0  # Default fallback

    if method == "min":
        return min(spacings)
    elif method == "max":
        return max(spacings)
    elif method == "average":
        return sum(spacings) / len(spacings)
    else:
        raise ValueError(
            f"Invalid cross_zone_exclusion_method: '{method}'. "
            f"Must be 'min', 'max', or 'average'."
        )


# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ TIER COMPUTATION SECTION
# ═══════════════════════════════════════════════════════════════════════════


def compute_czrc_tiers(
    czrc_region: BaseGeometry,
    zone_spacings: Dict[str, float],
    pair_key: str,
    tier1_mult: float = 1.0,
    tier2_mult: float = 2.0,
) -> Tuple[BaseGeometry, BaseGeometry, float]:
    """
    Compute Tier 1 and Tier 2 regions for a CZRC pairwise region.

    Tier 1 = CZRC + (tier1_mult × R_max): Active optimization region
    Tier 2 = CZRC + (tier2_mult × R_max): Coverage context region

    Args:
        czrc_region: Original CZRC region (coverage cloud intersection)
        zone_spacings: Dict mapping zone_name -> max_spacing_m
        pair_key: Underscore-separated zone names (e.g., "Zone 1_Zone 2")
        tier1_mult: Tier 1 expansion multiplier (default 1.0)
        tier2_mult: Tier 2 expansion multiplier (default 2.0)

    Returns:
        Tuple of (tier1_region, tier2_region, r_max)
    """
    # Parse zone names from pair_key (underscore-separated)
    zones = pair_key.split("_")

    # Get maximum spacing from zones in this pair (default 100m if not found)
    r_values = [zone_spacings.get(z, 100.0) for z in zones]
    r_max = max(r_values) if r_values else 100.0

    # Compute tier regions via buffer expansion
    tier1_region = czrc_region.buffer(tier1_mult * r_max)
    tier2_region = czrc_region.buffer(tier2_mult * r_max)

    return tier1_region, tier2_region, r_max


def filter_test_points_to_tier1(
    all_test_points: List[Dict[str, Any]],
    tier1_region: BaseGeometry,
) -> List[Dict[str, Any]]:
    """
    Filter first-pass test points to those within Tier 1 region.

    IMPORTANT: We reuse test points from first-pass, NOT generate new ones.
    Each test point already has 'required_radius' from its origin zone.

    Args:
        all_test_points: From all_stats["test_points"] (has x, y, required_radius, zone)
        tier1_region: Tier 1 boundary geometry

    Returns:
        List of test point dicts within Tier 1
    """
    return [
        tp for tp in all_test_points if tier1_region.contains(Point(tp["x"], tp["y"]))
    ]


def generate_tier2_ring_test_points(
    tier1_region: BaseGeometry,
    tier2_region: BaseGeometry,
    r_max: float,
    tier2_spacing_multiplier: float = 3.0,
    base_test_spacing_mult: float = 0.2,
    clip_geometry: Optional[BaseGeometry] = None,
) -> List[Dict[str, Any]]:
    """
    Generate sparse test points in the Tier 2 ring (Tier2 - Tier1).

    These test points protect against removing boreholes that are the sole
    coverage for the Tier 2 region. Uses hexagonal grid pattern at 3× sparser
    spacing than Tier 1 test points.

    Args:
        tier1_region: Tier 1 boundary geometry
        tier2_region: Tier 2 boundary geometry (contains Tier 1)
        r_max: Maximum coverage radius for this CZRC region
        tier2_spacing_multiplier: Multiplier relative to Tier 1 spacing (default 3.0)
        base_test_spacing_mult: Base test spacing multiplier (default 0.2)
        clip_geometry: Optional geometry to clip test points to (e.g., zones union)

    Returns:
        List of test point dicts with x, y, required_radius
    """
    # Compute Tier 2 ring = Tier 2 - Tier 1
    tier2_ring = tier2_region.difference(tier1_region)
    if tier2_ring.is_empty:
        return []

    # Clip to provided geometry if specified (e.g., zones coverage area)
    if clip_geometry is not None and not clip_geometry.is_empty:
        tier2_ring = tier2_ring.intersection(clip_geometry)
        if tier2_ring.is_empty:
            return []

    # Tier 2 ring test point spacing = base × multiplier × r_max
    # e.g., 0.2 × 3.0 × 100m = 60m grid spacing
    spacing = base_test_spacing_mult * tier2_spacing_multiplier * r_max

    # Hexagonal grid parameters
    dx = spacing
    dy = spacing * math.sqrt(3) / 2

    bounds = tier2_ring.bounds  # (minx, miny, maxx, maxy)
    test_points = []

    row_idx = 0
    y = bounds[1]
    while y <= bounds[3] + dy:
        x_offset = (row_idx % 2) * (dx / 2)
        x = bounds[0] + x_offset
        while x <= bounds[2] + dx:
            pt = Point(x, y)
            if tier2_ring.contains(pt):
                test_points.append(
                    {
                        "x": x,
                        "y": y,
                        "required_radius": r_max,
                        "zone": "tier2_ring",
                    }
                )
            x += dx
        y += dy
        row_idx += 1

    return test_points


def classify_first_pass_boreholes(
    first_pass_boreholes: List[Dict[str, Any]],
    tier1_region: BaseGeometry,
    tier2_region: BaseGeometry,
    all_boreholes: Optional[List[Dict[str, Any]]] = None,
    r_max: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Classify boreholes into Tier 1 candidates, Tier 2 locked, and external coverage.

    This function is named for historical reasons but is used in both Second Pass
    (classifying First Pass boreholes) and Third Pass (classifying Second Pass output).

    Classification Rules:
        - Tier 1 (inside): CANDIDATES for ILP re-optimization (grey markers in viz)
        - Tier 2 only (outside Tier 1): LOCKED CONSTANTS providing pre-coverage
        - External (outside Tier 2 but within R_max buffer): Additional pre-coverage
        - Outside buffer zone: Not relevant to this CZRC pair

    For Third Pass specifically:
        - Input: Second Pass OUTPUT (First Pass survivors + Second Pass additions)
        - Tier 1 candidates: Shown as grey markers, eligible for removal
        - Tier 2 locked: Provide coverage context but are NOT re-optimized
        - External: Boreholes from other cells that can cover Tier 2 test points

    Args:
        first_pass_boreholes: Boreholes to classify (First Pass for 2nd pass,
            Second Pass OUTPUT for 3rd pass)
        tier1_region: Tier 1 boundary (active optimization region)
        tier2_region: Tier 2 boundary (always contains Tier 1, coverage context)
        all_boreholes: Optional full set of boreholes for external coverage detection.
            For Second Pass: all First Pass output from all zones.
            For Third Pass: all Second Pass output (second_pass_candidates).
        r_max: Maximum coverage radius, used to compute external buffer zone.

    Returns:
        Tuple of (candidates, locked, external) borehole lists.
        - candidates: Tier 1 boreholes (ILP candidates)
        - locked: Tier 2 ring boreholes (locked coverage)
        - external: Boreholes outside Tier 2 but within R_max buffer (additional coverage)
    """
    candidates = []
    locked = []

    for bh in first_pass_boreholes:
        pt = Point(bh["x"], bh["y"])
        # IMPORTANT: Check Tier 1 FIRST (since Tier 2 contains Tier 1)
        if tier1_region.contains(pt):
            candidates.append(bh)
        elif tier2_region.contains(pt):
            locked.append(bh)

    # Detect external boreholes that can cover Tier 2 test points
    external: List[Dict[str, Any]] = []
    if all_boreholes is not None and r_max is not None and r_max > 0:
        # Buffer Tier 2 outward by R_max to find external coverage zone
        tier2_buffer = tier2_region.buffer(r_max)
        # External zone = buffer ring outside Tier 2
        external_zone = tier2_buffer.difference(tier2_region)

        # Track positions already in candidates/locked to avoid duplicates
        known_positions = {(bh["x"], bh["y"]) for bh in candidates}
        known_positions.update((bh["x"], bh["y"]) for bh in locked)

        for bh in all_boreholes:
            pos = (bh["x"], bh["y"])
            if pos in known_positions:
                continue  # Already classified
            pt = Point(bh["x"], bh["y"])
            if external_zone.contains(pt):
                external.append(bh)
                known_positions.add(pos)  # Prevent re-adding

    return candidates, locked, external


# ═══════════════════════════════════════════════════════════════════════════
# � OVERLAPPING PAIR CLUSTERING SECTION
# ═══════════════════════════════════════════════════════════════════════════


def _compute_tier1_for_pair(
    czrc_wkt: str,
    pair_key: str,
    zone_spacings: Dict[str, float],
    tier1_mult: float = 1.0,
) -> Tuple[Optional[BaseGeometry], float]:
    """
    Compute Tier 1 region for a single pairwise CZRC region.

    Args:
        czrc_wkt: WKT string of CZRC region
        pair_key: Zone pair key (e.g., "ZoneA_ZoneB")
        zone_spacings: Dict mapping zone_name -> max_spacing_m
        tier1_mult: Tier 1 expansion multiplier

    Returns:
        Tuple of (tier1_geometry, r_max) or (None, 0) on failure
    """
    try:
        czrc_region = wkt.loads(czrc_wkt)
        if czrc_region.is_empty or not czrc_region.is_valid:
            return None, 0.0
    except (ValueError, TypeError):
        return None, 0.0

    zones = pair_key.split("_")
    r_values = [zone_spacings.get(z, 100.0) for z in zones]
    r_max = max(r_values) if r_values else 100.0
    tier1_region = czrc_region.buffer(tier1_mult * r_max)

    return tier1_region, r_max


def _group_overlapping_pairs(
    pairwise_wkts: Dict[str, str],
    zone_spacings: Dict[str, float],
    tier1_mult: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Group pairwise CZRC regions into clusters where Tier 1 regions overlap.

    Uses union-find algorithm to efficiently cluster overlapping regions.

    Algorithm:
    1. Compute Tier 1 for each pair
    2. Test all pair combinations for Tier 1 intersection
    3. Group intersecting pairs into clusters using union-find

    Args:
        pairwise_wkts: Dict of {pair_key: czrc_wkt}
        zone_spacings: Dict mapping zone_name -> max_spacing_m
        tier1_mult: Tier 1 expansion multiplier

    Returns:
        List of cluster dicts, each containing:
        - "pair_keys": List of pair keys in this cluster
        - "tier1_regions": Dict of {pair_key: tier1_geometry}
        - "r_max_values": Dict of {pair_key: r_max}
        - "unified_tier1": Union of all Tier 1 regions in cluster
        - "overall_r_max": Max R_max across all pairs in cluster
    """
    # Step 1: Compute Tier 1 for each pair
    tier1_data: Dict[str, Tuple[BaseGeometry, float]] = {}
    for pair_key, czrc_wkt in pairwise_wkts.items():
        tier1, r_max = _compute_tier1_for_pair(
            czrc_wkt, pair_key, zone_spacings, tier1_mult
        )
        if tier1 is not None:
            tier1_data[pair_key] = (tier1, r_max)

    pair_keys = list(tier1_data.keys())
    n = len(pair_keys)

    if n == 0:
        return []

    # Step 2: Union-find for clustering
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Step 3: Test all pairs for Tier 1 intersection
    for i in range(n):
        for j in range(i + 1, n):
            tier1_i = tier1_data[pair_keys[i]][0]
            tier1_j = tier1_data[pair_keys[j]][0]
            if tier1_i.intersects(tier1_j):
                union(i, j)

    # Step 4: Group by cluster root
    clusters_dict: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(i)

    # Step 5: Build cluster results
    clusters: List[Dict[str, Any]] = []
    for indices in clusters_dict.values():
        cluster_pair_keys = [pair_keys[i] for i in indices]
        cluster_tier1_regions = {pk: tier1_data[pk][0] for pk in cluster_pair_keys}
        cluster_r_max_values = {pk: tier1_data[pk][1] for pk in cluster_pair_keys}
        unified_tier1 = unary_union(list(cluster_tier1_regions.values()))
        overall_r_max = max(cluster_r_max_values.values())

        clusters.append(
            {
                "pair_keys": cluster_pair_keys,
                "tier1_regions": cluster_tier1_regions,
                "r_max_values": cluster_r_max_values,
                "unified_tier1": unified_tier1,
                "overall_r_max": overall_r_max,
            }
        )

    return clusters


# ═══════════════════════════════════════════════════════════════════════════
# �🔧 ILP PREPARATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _prepare_candidates_for_ilp(
    tier1_region: BaseGeometry,
    bh_candidates: List[Dict[str, Any]],
    r_max: float,
    min_zone_spacing: float,
    config: Dict[str, Any],
) -> Tuple[List[Point], float]:
    """
    Generate candidate grid and merge with first-pass boreholes.

    Args:
        tier1_region: Tier 1 boundary for grid generation
        bh_candidates: First-pass boreholes in Tier 1
        r_max: Maximum zone spacing for grid boundary
        min_zone_spacing: Minimum zone spacing for grid density
        config: CZRC config section

    Returns:
        Tuple of (deduplicated_candidates, candidate_spacing)
    """
    candidate_mult = config.get("candidate_grid_spacing_mult", 0.5)
    candidate_spacing = min_zone_spacing * candidate_mult

    # Convert Tier 1 to polygon list for _generate_candidate_grid
    tier1_polys: List[Polygon] = []
    if tier1_region.geom_type == "Polygon":
        tier1_polys = [tier1_region]  # type: ignore
    elif tier1_region.geom_type == "MultiPolygon":
        tier1_polys = list(tier1_region.geoms)  # type: ignore

    # REUSED: _generate_candidate_grid() from optimization_geometry.py
    # NOTE: Pass buffer_distance=0 because tier1_region is ALREADY buffered
    # (Tier 1 = CZRC + 1×R_max). Without this, candidates would extend to Tier 2.
    grid_candidates = _generate_candidate_grid(
        gap_polys=tier1_polys,
        max_spacing=r_max,
        grid_spacing=candidate_spacing,
        grid_type="hexagonal",
        hexagonal_density=1.5,
        logger=None,
        buffer_distance=0,  # Tier 1 is already the target region - no extra buffer
    )

    # Add first-pass Tier 1 boreholes as candidates
    for bh in bh_candidates:
        grid_candidates.append(Point(bh["x"], bh["y"]))

    # REUSED: _dedupe_candidates() - remove duplicates keeping largest radius
    candidate_radii = [min_zone_spacing] * len(grid_candidates)
    grid_candidates, _ = _dedupe_candidates(grid_candidates, candidate_radii, tol=1.0)

    return grid_candidates, candidate_spacing


def _solve_czrc_ilp(
    unsatisfied_test_points: List[Dict[str, Any]],
    candidates: List[Point],
    min_zone_spacing: float,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    highs_log_file: Optional[str] = None,
) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    """
    Build coverage dict and solve ILP for CZRC region with infeasibility retry.

    If the ILP is infeasible (coverage requirements conflict with spacing),
    automatically retries with progressively relaxed EXCLUSION FACTOR ONLY:
    1. First try with configured settings (exclusion_factor × 1.0)
    2. If infeasible: reduce exclusion_factor to 80% of configured
    3. If still infeasible: reduce exclusion_factor to 60% of configured
    4. If still infeasible: disable conflict constraints entirely (exclusion = 0)

    IMPORTANT: Coverage target is NEVER reduced (always 100%). Only exclusion
    factor is relaxed to resolve infeasibility. This ensures complete coverage
    while accepting potentially closer borehole spacing if needed.

    Args:
        unsatisfied_test_points: Test points not pre-covered by Tier 2
        candidates: All candidate locations (grid + first-pass BHs)
        min_zone_spacing: Minimum zone spacing for conflict constraints
        config: CZRC config section with ILP params
        logger: Optional logger
        highs_log_file: Optional path to write HiGHS solver log

    Returns:
        Tuple of (selected_indices, ilp_stats)
    """
    # REUSED: _build_coverage_dict_variable_test_radii() from consolidation.py
    coverage = _build_coverage_dict_variable_test_radii(
        test_points=unsatisfied_test_points,
        candidates=candidates,
        logger=logger,
    )

    # Convert test points to Point objects for _solve_ilp
    test_points_for_ilp = [Point(tp["x"], tp["y"]) for tp in unsatisfied_test_points]

    # REUSED: _solve_ilp() from solver_algorithms.py
    ilp_config = config.get("ilp", {})
    # Resolve stall detection config (respects apply_to_czrc flag)
    stall_detection_config = _get_czrc_stall_detection_config(ilp_config)

    # Get base settings from config
    base_exclusion_factor = ilp_config.get("exclusion_factor", 0.9)
    base_coverage_target = ilp_config.get("coverage_target_pct", 100.0)

    # Define retry strategies with progressively relaxed EXCLUSION ONLY
    # CRITICAL: Coverage target NEVER changes (must remain at 100%)
    # Only exclusion_factor is reduced to resolve infeasibility
    # Each tuple: (exclusion_factor, coverage_target_pct, use_conflict_constraints, description)
    retry_strategies = [
        (base_exclusion_factor, base_coverage_target, True, "default"),
        (
            base_exclusion_factor * 0.8,
            base_coverage_target,
            True,
            "reduced_exclusion_20pct",
        ),
        (
            base_exclusion_factor * 0.6,
            base_coverage_target,
            True,
            "reduced_exclusion_40pct",
        ),
        (0.0, base_coverage_target, False, "no_conflicts"),
    ]

    for excl_factor, cov_target, use_conflicts, strategy_name in retry_strategies:
        result, stats = _solve_ilp(
            test_points=test_points_for_ilp,
            candidates=candidates,
            coverage=coverage,
            time_limit=ilp_config.get("time_limit", 90),
            mip_gap=ilp_config.get("mip_gap", 0.03),
            threads=1,
            coverage_target_pct=cov_target,
            use_conflict_constraints=use_conflicts,
            conflict_constraint_mode="pairwise",
            exclusion_factor=excl_factor,
            max_spacing=min_zone_spacing,
            verbose=ilp_config.get("verbose", 1),
            logger=logger,
            highs_log_file=highs_log_file,
            stall_detection_config=stall_detection_config,
        )

        # Check if solve succeeded
        if result is not None:
            # Add retry info to stats
            stats["retry_strategy"] = strategy_name
            if strategy_name != "default":
                stats["constraint_relaxation"] = {
                    "exclusion_factor": excl_factor,
                    "coverage_target_pct": cov_target,
                    "use_conflict_constraints": use_conflicts,
                }
                if logger:
                    logger.warning(
                        f"⚠️ CZRC ILP solved with relaxed constraints ({strategy_name})"
                    )
            return result, stats

        # Check if infeasible - retry with next strategy
        reason = stats.get("reason", "")
        if reason == "Infeasible" and strategy_name != "no_conflicts":
            if logger:
                logger.info(
                    f"   ⚠️ ILP infeasible with {strategy_name}, trying relaxed constraints..."
                )
            continue

        # Non-infeasibility failure or last strategy - give up
        break

    # All strategies failed
    return None, stats


# ═══════════════════════════════════════════════════════════════════════════
# 🎯 PER-PAIR ILP ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════


def _parse_czrc_wkt(czrc_wkt: str) -> Optional[BaseGeometry]:
    """Parse WKT string into geometry, returning None on failure."""
    try:
        return wkt.loads(czrc_wkt)
    except (ValueError, TypeError):
        return None


def _compute_unsatisfied_test_points(
    tier1_test_points: List[Dict[str, Any]],
    locked_boreholes: List[Dict[str, Any]],
    logger: Optional[logging.Logger],
    external_boreholes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Compute test points not pre-covered by locked (Tier 2) or external boreholes.

    Args:
        tier1_test_points: Test points to check coverage for
        locked_boreholes: Boreholes inside Tier 2 (locked coverage)
        logger: Optional logger
        external_boreholes: Boreholes outside Tier 2 but within R_max buffer

    Returns:
        (unsatisfied_test_points, precovered_count)
    """
    # Combine locked and external boreholes for coverage check
    all_coverage_bhs = list(locked_boreholes)
    if external_boreholes:
        all_coverage_bhs.extend(external_boreholes)

    pre_covered = _compute_locked_coverage(tier1_test_points, all_coverage_bhs, logger)
    unsatisfied = [tp for i, tp in enumerate(tier1_test_points) if i not in pre_covered]
    return unsatisfied, len(pre_covered)


def _annotate_test_points_with_coverage(
    test_points: List[Dict[str, Any]],
    locked_boreholes: List[Dict[str, Any]],
    external_boreholes: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Annotate test points with is_covered flag based on locked and external coverage.

    This function marks each test point with is_covered=True if it falls within
    the coverage radius of any locked (Tier 2) or external borehole, or
    is_covered=False otherwise. Used for visualization to show which test points
    are pre-covered vs unsatisfied.

    Args:
        test_points: List of test point dicts with x, y, required_radius, zone
        locked_boreholes: Locked boreholes from Tier 2 region
        external_boreholes: Boreholes outside Tier 2 but within R_max buffer

    Returns:
        List of test point dicts with added is_covered boolean field
    """
    if not test_points:
        return []

    # Combine locked and external boreholes for coverage check
    all_coverage_bhs = list(locked_boreholes)
    if external_boreholes:
        all_coverage_bhs.extend(external_boreholes)

    # Compute pre-covered indices using existing function
    pre_covered_indices = _compute_locked_coverage(test_points, all_coverage_bhs, None)

    # Annotate each test point with coverage status
    annotated = []
    for i, tp in enumerate(test_points):
        annotated_tp = dict(tp)  # Copy original fields
        annotated_tp["is_covered"] = i in pre_covered_indices
        annotated.append(annotated_tp)

    return annotated


def _build_pair_stats(
    pair_key: str,
    r_max: float,
    tier1_count: int,
    precovered: int,
    unsatisfied: int,
    candidates: int,
    selected: int,
    elapsed: float,
    ilp_stats: Dict[str, Any],
    bh_candidates: List[Dict[str, Any]],
    tier2_ring_total: int = 0,
    tier2_ring_precovered: int = 0,
    tier2_ring_unsatisfied: int = 0,
) -> Dict[str, Any]:
    """Build statistics dict for a solved pair."""
    return {
        "status": "success",
        "pair_key": pair_key,
        "r_max": r_max,
        "tier1_test_points": tier1_count,
        "precovered_count": precovered,
        "unsatisfied_count": unsatisfied,
        "candidates_count": candidates,
        "selected_count": selected,
        "solve_time": elapsed,
        "ilp_stats": ilp_stats,
        "first_pass_candidates": bh_candidates,
        # Tier 2 ring protection stats
        "tier2_ring_total": tier2_ring_total,
        "tier2_ring_precovered": tier2_ring_precovered,
        "tier2_ring_unsatisfied": tier2_ring_unsatisfied,
    }


def solve_czrc_ilp_for_pair(
    pair_key: str,
    czrc_wkt: str,
    zone_spacings: Dict[str, float],
    all_test_points: List[Dict[str, Any]],
    first_pass_boreholes: List[Dict[str, Any]],
    config: Dict[str, Any],
    zones_clip_geometry: Optional[BaseGeometry] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]
]:
    """Solve CZRC ILP for a single zone pair. Returns (selected, removed, added, stats)."""
    start_time = time.perf_counter()

    # Step 1: Parse CZRC region
    czrc_region = _parse_czrc_wkt(czrc_wkt)
    if czrc_region is None:
        return [], [], [], {"status": "failed", "reason": "invalid_wkt"}

    # Step 2: Compute tiers
    tier1_mult = config.get("tier1_rmax_multiplier", 1.0)
    tier2_mult = config.get("tier2_rmax_multiplier", 2.0)
    tier1_region, tier2_region, r_max = compute_czrc_tiers(
        czrc_region, zone_spacings, pair_key, tier1_mult, tier2_mult
    )

    # Step 3: Filter test points
    tier1_test_points = filter_test_points_to_tier1(all_test_points, tier1_region)
    if not tier1_test_points:
        return [], [], [], {"status": "skipped", "reason": "no_tier1_test_points"}

    # Step 4: Classify boreholes
    # Note: This simpler function doesn't have access to all boreholes for external detection
    # External coverage detection is handled in solve_czrc_ilp_for_cluster
    bh_candidates, locked, _external = classify_first_pass_boreholes(
        first_pass_boreholes, tier1_region, tier2_region
    )

    # Step 5: Compute unsatisfied test points (Tier 1)
    tier1_unsatisfied, precovered_ct = _compute_unsatisfied_test_points(
        tier1_test_points, locked, logger
    )

    # Step 5.5: Generate Tier 2 ring test points if enabled
    t2_protection = config.get("tier2_test_point_protection", {})
    t2_enabled = t2_protection.get("enabled", True)
    t2_multiplier = t2_protection.get("tier2_test_spacing_multiplier", 3.0)
    base_test_mult = config.get("test_spacing_mult", 0.2)

    tier2_ring_unsatisfied = []
    tier2_ring_total = 0
    tier2_ring_precovered = 0

    if t2_enabled:
        # Generate sparse test points in Tier 2 ring
        tier2_ring_test_points = generate_tier2_ring_test_points(
            tier1_region,
            tier2_region,
            r_max,
            tier2_spacing_multiplier=t2_multiplier,
            base_test_spacing_mult=base_test_mult,
            clip_geometry=zones_clip_geometry,
        )
        tier2_ring_total = len(tier2_ring_test_points)

        if tier2_ring_test_points:
            # Filter by locked borehole coverage (same logic as Tier 1)
            tier2_ring_unsatisfied, t2_precov = _compute_unsatisfied_test_points(
                tier2_ring_test_points, locked, logger
            )
            tier2_ring_precovered = t2_precov
            if logger:
                logger.debug(
                    f"🔷 Tier 2 ring: {tier2_ring_total} test pts, "
                    f"{tier2_ring_precovered} pre-covered, "
                    f"{len(tier2_ring_unsatisfied)} unsatisfied"
                )

    # Combine Tier 1 and Tier 2 ring unsatisfied test points
    unsatisfied = tier1_unsatisfied + tier2_ring_unsatisfied

    if not unsatisfied:
        return (
            [],
            [],
            [],
            {
                "status": "success",
                "reason": "all_precovered",
                "precovered_count": precovered_ct + tier2_ring_precovered,
                "tier1_precovered": precovered_ct,
                "tier2_ring_precovered": tier2_ring_precovered,
            },
        )

    # Step 6: Prepare candidates
    zones = pair_key.split("_")
    # Use cross_zone_exclusion_method from config to aggregate spacings
    exclusion_method = _get_cross_zone_exclusion_method(config)
    min_spacing = _aggregate_zone_spacings(zone_spacings, zones, exclusion_method)
    candidates, _ = _prepare_candidates_for_ilp(
        tier1_region, bh_candidates, r_max, min_spacing, config
    )
    if not candidates:
        return [], [], [], {"status": "failed", "reason": "no_candidates"}

    # Step 7: Solve ILP
    indices, ilp_stats = _solve_czrc_ilp(
        unsatisfied, candidates, min_spacing, config, logger
    )
    if indices is None:
        return (
            [],
            [],
            [],
            {"status": "failed", "reason": "ilp_failed", "ilp_stats": ilp_stats},
        )

    # Step 8: Assemble results
    selected, removed, added = _assemble_czrc_results(
        indices, candidates, bh_candidates, min_spacing
    )
    elapsed = time.perf_counter() - start_time
    stats = _build_pair_stats(
        pair_key,
        r_max,
        len(tier1_test_points),
        precovered_ct,
        len(unsatisfied),
        len(candidates),
        len(selected),
        elapsed,
        ilp_stats,
        bh_candidates,
        tier2_ring_total=tier2_ring_total,
        tier2_ring_precovered=tier2_ring_precovered,
        tier2_ring_unsatisfied=len(tier2_ring_unsatisfied),
    )
    return selected, removed, added, stats


def _assemble_czrc_results(
    indices: List[int],
    candidates: List[Point],
    bh_candidates: List[Dict[str, Any]],
    min_spacing: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Assemble selected/removed/added boreholes from ILP results."""
    selected_set = {
        (candidates[i].x, candidates[i].y) for i in indices if i < len(candidates)
    }
    tier1_bh_set = {(bh["x"], bh["y"]) for bh in bh_candidates}

    # Selected = all ILP selections
    selected = [
        {"x": candidates[i].x, "y": candidates[i].y, "coverage_radius": min_spacing}
        for i in indices
        if i < len(candidates)
    ]
    # Removed = Tier 1 first-pass NOT selected
    removed = [
        {
            "x": bh["x"],
            "y": bh["y"],
            "coverage_radius": bh.get("coverage_radius", min_spacing),
        }
        for bh in bh_candidates
        if (bh["x"], bh["y"]) not in selected_set
    ]
    # Added = Selected NOT in first-pass Tier 1
    added = [
        {"x": candidates[i].x, "y": candidates[i].y, "coverage_radius": min_spacing}
        for i in indices
        if i < len(candidates)
        and (candidates[i].x, candidates[i].y) not in tier1_bh_set
    ]
    # Filter coincident pairs
    removed, added = _filter_coincident_pairs(removed, added)
    return selected, removed, added


# ═══════════════════════════════════════════════════════════════════════════
# � CELL SPLITTING (Large Region Decomposition)
# ═══════════════════════════════════════════════════════════════════════════


def _split_into_grid_cells(
    geometry: BaseGeometry,
    cell_size_m: float,
    min_cell_area_m2: float = 100.0,
) -> List[BaseGeometry]:
    """
    Split a geometry into fixed-size grid cells.

    Creates a grid of square cells aligned to the geometry's bounding box,
    clips each cell to the geometry, and returns cells that exceed the
    minimum area threshold.

    Args:
        geometry: Shapely geometry to split
        cell_size_m: Cell size in meters (e.g., 1000 for 1km cells)
        min_cell_area_m2: Minimum cell area to include (skip tiny slivers)

    Returns:
        List of cell geometries (clipped to input geometry)
    """
    from shapely.geometry import box

    if geometry.is_empty:
        return []

    minx, miny, maxx, maxy = geometry.bounds

    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            # Create cell
            cell_box = box(x, y, x + cell_size_m, y + cell_size_m)
            # Clip to geometry
            clipped = geometry.intersection(cell_box)
            # Keep if area exceeds threshold
            if not clipped.is_empty and clipped.area >= min_cell_area_m2:
                cells.append(clipped)
            y += cell_size_m
        x += cell_size_m

    return cells


def _determine_cell_count(
    region_area_m2: float,
    config: Dict[str, Any],
) -> int:
    """
    Calculate number of cells based on target average cell area.

    Uses rule: K = ceil(region_area / target_cell_area), with min/max bounds.

    Args:
        region_area_m2: Total area of region to split (m²)
        config: cell_splitting config section

    Returns:
        Number of cells (K for K-means)
    """
    kmeans_config = config.get("kmeans_voronoi", {})
    target_cell_area = kmeans_config.get(
        "target_cell_area_m2", 1_000_000
    )  # Default 1 km²
    min_cells = kmeans_config.get("min_cells", 2)
    max_cells = kmeans_config.get("max_cells", 50)

    # Calculate based on target area
    computed_k = math.ceil(region_area_m2 / target_cell_area)

    # Clamp to range
    return max(min_cells, min(computed_k, max_cells))


def _get_clipped_voronoi_cells(
    seeds: np.ndarray,
    region: BaseGeometry,
    buffer_margin: float = 1000.0,
    min_cell_area_m2: float = 100.0,
) -> List[BaseGeometry]:
    """
    Generate Voronoi cells from seeds, clipped to a region.

    Uses shapely.ops.voronoi_diagram for clean handling of boundaries.

    Args:
        seeds: K-means cluster centroids (n, 2) array
        region: Region polygon to clip cells to
        buffer_margin: Extra margin for envelope (meters)
        min_cell_area_m2: Minimum cell area to keep

    Returns:
        List of clipped cell polygons
    """
    if len(seeds) < 2:
        return [region]  # Cannot tessellate with <2 seeds

    # Create seed points
    seed_multipoint = MultiPoint([Point(s) for s in seeds])

    # Envelope larger than region to ensure bounded cells
    envelope = region.envelope.buffer(buffer_margin)

    # Generate Voronoi diagram
    voronoi_result = voronoi_diagram(seed_multipoint, envelope=envelope)

    # Clip each cell to region
    cells = []
    for voronoi_cell in voronoi_result.geoms:
        clipped = voronoi_cell.intersection(region)

        if clipped.is_empty:
            continue

        # Handle MultiPolygon (can occur with complex regions)
        if isinstance(clipped, MultiPolygon):
            for part in clipped.geoms:
                if part.area >= min_cell_area_m2:
                    cells.append(part)
        else:
            if clipped.area >= min_cell_area_m2:
                cells.append(clipped)

    return cells


def _split_into_voronoi_cells(
    geometry: BaseGeometry,
    candidate_positions: np.ndarray,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> List[BaseGeometry]:
    """
    Split a geometry into cells using K-means + Voronoi.

    Uses target average cell area to determine K, then K-means on
    candidate positions to find balanced seeds.

    Args:
        geometry: Region to split
        candidate_positions: (n, 2) array of candidate (x, y) positions
        config: cell_splitting config section
        logger: Optional logger

    Returns:
        List of cell geometries
    """
    log = logger or _logger

    if geometry.is_empty:
        return []

    region_area = geometry.area
    n_candidates = len(candidate_positions)

    # Determine cell count from target area
    n_cells = _determine_cell_count(region_area, config)

    if n_candidates < n_cells:
        log.debug(f"   Too few candidates ({n_candidates}) for {n_cells} cells")
        # Fall back to fewer cells
        n_cells = max(2, n_candidates)

    if n_cells < 2:
        return [geometry]

    # K-means clustering
    kmeans_config = config.get("kmeans_voronoi", {})
    random_state = kmeans_config.get("random_state", 42)

    kmeans = KMeans(n_clusters=n_cells, random_state=random_state, n_init="auto")
    kmeans.fit(candidate_positions)

    seeds = kmeans.cluster_centers_

    # Generate Voronoi cells clipped to region
    min_cell_area = config.get("min_cell_area_m2", 100.0)
    cells = _get_clipped_voronoi_cells(seeds, geometry, min_cell_area_m2=min_cell_area)

    target_area = kmeans_config.get("target_cell_area_m2", 1_000_000)
    log.info(
        f"   🔷 K-means + Voronoi: {region_area/1e6:.2f} km² → {len(cells)} cells "
        f"(target: {target_area/1e6:.1f} km² avg)"
    )

    return cells


def _create_cell_cluster(
    cell_geometry: BaseGeometry,
    original_cluster: Dict[str, Any],
    cell_index: int,
) -> Dict[str, Any]:
    """
    Create a cluster dict for a single cell.

    Copies the original cluster's metadata but replaces the geometry
    with the cell geometry. The pair_keys are suffixed with "_cell_N"
    to track origin.

    Args:
        cell_geometry: The clipped cell geometry
        original_cluster: The original cluster dict
        cell_index: Index of this cell (for naming)

    Returns:
        New cluster dict with cell geometry
    """
    return {
        "unified_tier1": cell_geometry,
        "overall_r_max": original_cluster["overall_r_max"],
        "pair_keys": [
            f"{pk}_cell_{cell_index}" for pk in original_cluster["pair_keys"]
        ],
    }


def check_and_split_large_cluster(
    cluster: Dict[str, Any],
    zone_spacings: Dict[str, float],
    all_test_points: List[Dict[str, Any]],
    first_pass_boreholes: List[Dict[str, Any]],
    config: Dict[str, Any],
    zones_clip_geometry: Optional[BaseGeometry] = None,
    logger: Optional[logging.Logger] = None,
    czrc_cache: Optional["CZRCCacheManager"] = None,
    highs_log_folder: Optional[str] = None,
    cluster_idx: int = 0,
) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]
]:
    """
    Wrapper that checks cluster size and splits if too large for ILP.

    Decision logic:
        1. Get cell_splitting config
        2. If disabled OR area <= max_area_for_direct_ilp: call solve_czrc_ilp_for_cluster()
        3. Otherwise split into grid cells and solve each independently

    Args:
        cluster: Cluster dict from _group_overlapping_pairs()
        zone_spacings: Dict mapping zone_name -> max_spacing_m
        all_test_points: From all_stats["test_points"]
        first_pass_boreholes: From all_stats["selected_boreholes"]
        config: CZRC config section
        zones_clip_geometry: Optional geometry to clip Tier 2 test points to
        logger: Optional logger
        czrc_cache: Optional CZRCCacheManager for intra-run result caching
        highs_log_folder: Optional folder for HiGHS log files
        cluster_idx: Index of this cluster (for unique log file naming)

    Returns:
        (selected, removed, added, stats) tuple - same as solve_czrc_ilp_for_cluster
    """
    cell_config = config.get("cell_splitting", {})
    enabled = cell_config.get("enabled", True)
    max_area = cell_config.get("max_area_for_direct_ilp_m2", 1_000_000)
    cell_size = cell_config.get("cell_size_m", 1000)
    min_cell_area = cell_config.get("min_cell_area_m2", 100)

    unified_tier1 = cluster["unified_tier1"]
    cluster_area = unified_tier1.area

    # Check if splitting needed
    if not enabled or cluster_area <= max_area:
        # Small enough for direct ILP
        # Use cluster_idx + 1000 to signal "direct solve" naming in log file
        return solve_czrc_ilp_for_cluster(
            cluster,
            zone_spacings,
            all_test_points,
            first_pass_boreholes,
            config,
            zones_clip_geometry,
            logger,
            czrc_cache,
            highs_log_folder,
            cluster_idx + 1000,  # Signal direct solve (not cell split)
        )

    # Determine splitting method
    log = logger or _logger
    method = cell_config.get("method", "kmeans_voronoi")

    if method == "kmeans_voronoi":
        # Need candidate positions for K-means
        # Generate sample grid at candidate spacing to get density distribution
        min_zone_spacing = min(zone_spacings.values()) if zone_spacings else 100.0
        candidate_mult = config.get("candidate_grid_spacing_mult", 0.5)
        sample_spacing = min_zone_spacing * candidate_mult

        # Convert geometry to polygon list for _generate_candidate_grid
        if unified_tier1.geom_type == "Polygon":
            tier1_polys = [unified_tier1]
        elif unified_tier1.geom_type == "MultiPolygon":
            tier1_polys = list(unified_tier1.geoms)
        else:
            tier1_polys = [unified_tier1]

        # Generate sample positions within the region
        sample_grid = _generate_candidate_grid(
            gap_polys=tier1_polys,
            max_spacing=min_zone_spacing,
            grid_spacing=sample_spacing,
            grid_type="hexagonal",
            hexagonal_density=1.5,
            logger=None,
            buffer_distance=0,  # Region is already Tier 1 - no buffer needed
        )
        candidate_positions = np.array([[p.x, p.y] for p in sample_grid])

        cells = _split_into_voronoi_cells(
            unified_tier1, candidate_positions, cell_config, logger
        )
    else:
        # Fallback to grid method
        grid_config = cell_config.get("grid", {})
        cell_size = grid_config.get("cell_size_m", cell_config.get("cell_size_m", 2000))
        cells = _split_into_grid_cells(unified_tier1, cell_size, min_cell_area)

    if not cells:
        return [], [], [], {"status": "skipped", "reason": "no_cells_after_split"}

    log.info(
        f"📊 Splitting large cluster ({cluster_area/1e6:.2f} km²) "
        f"into {len(cells)} cells (method: {method})"
    )

    # Store cell geometries as WKT for visualization
    cell_wkts = [cell.wkt for cell in cells]
    log.info(f"   🔪 Splitting large cluster into {len(cells)} cells")

    # Process each cell
    all_selected: List[Dict[str, Any]] = []
    all_removed: List[Dict[str, Any]] = []
    all_added: List[Dict[str, Any]] = []
    cell_stats: List[Dict[str, Any]] = []
    seen_positions: set = set()

    for i, cell_geom in enumerate(cells):
        cell_cluster = _create_cell_cluster(cell_geom, cluster, i)
        # Use cluster_idx + cell_idx for unique log file naming
        cell_log_idx = cluster_idx * 100 + i  # e.g., cluster 2, cell 3 -> 203
        selected, removed, added, stats = solve_czrc_ilp_for_cluster(
            cell_cluster,
            zone_spacings,
            all_test_points,
            first_pass_boreholes,
            config,
            zones_clip_geometry,
            logger,
            czrc_cache,
            highs_log_folder,
            cell_log_idx,
        )

        # Deduplicate results across cells
        for bh in selected:
            pos = (bh["x"], bh["y"])
            if pos not in seen_positions:
                all_selected.append(bh)
                seen_positions.add(pos)

        all_removed.extend(removed)
        all_added.extend(added)
        cell_stats.append(stats)

    # Deduplicate removed (same BH might be marked removed by multiple cells)
    removed_positions = set()
    deduped_removed = []
    for bh in all_removed:
        pos = (bh["x"], bh["y"])
        if pos not in removed_positions:
            deduped_removed.append(bh)
            removed_positions.add(pos)

    # Deduplicate added
    added_positions = set()
    deduped_added = []
    for bh in all_added:
        pos = (bh["x"], bh["y"])
        if pos not in added_positions:
            deduped_added.append(bh)
            added_positions.add(pos)

    # Aggregate first_pass_candidates from all cell stats for visualization
    # These are the Tier 1 boreholes that were used as ILP candidates in each cell
    all_first_pass_candidates: List[Dict[str, Any]] = []
    seen_candidates: set = set()
    for cs in cell_stats:
        for bh in cs.get("first_pass_candidates", []):
            pos = (bh["x"], bh["y"])
            if pos not in seen_candidates:
                all_first_pass_candidates.append(bh)
                seen_candidates.add(pos)

    # Aggregate czrc_test_points from all cell stats for visualization
    all_czrc_test_points: List[Dict[str, Any]] = []
    seen_test_points_for_viz: set = set()
    for cs in cell_stats:
        for tp in cs.get("czrc_test_points", []):
            pos = (tp["x"], tp["y"])
            if pos not in seen_test_points_for_viz:
                all_czrc_test_points.append(tp)
                seen_test_points_for_viz.add(pos)

    # Compute TRUE Second Pass output: first_pass - removed + added
    # CRITICAL: all_selected only contains boreholes that were in at least one cell's
    # Tier 1 and selected by its ILP. Boreholes that were always in Tier 2 (locked)
    # across all cells never appear in all_selected, but they DID survive Second Pass!
    # The correct Second Pass output is: first_pass_boreholes - removed + added
    removed_positions_for_output = {(bh["x"], bh["y"]) for bh in deduped_removed}

    # Start with all First Pass boreholes that weren't removed
    true_second_pass_output: List[Dict[str, Any]] = []
    seen_output_positions: set = set()

    # Keep First Pass survivors (not removed)
    for bh in first_pass_boreholes:
        pos = (bh["x"], bh["y"])
        if pos not in removed_positions_for_output and pos not in seen_output_positions:
            true_second_pass_output.append(
                {
                    "x": bh["x"],
                    "y": bh["y"],
                    "coverage_radius": bh.get("coverage_radius", 100.0),
                }
            )
            seen_output_positions.add(pos)

    # Add Second Pass additions
    for bh in deduped_added:
        pos = (bh["x"], bh["y"])
        if pos not in seen_output_positions:
            true_second_pass_output.append(
                {
                    "x": bh["x"],
                    "y": bh["y"],
                    "coverage_radius": bh.get("coverage_radius", 100.0),
                }
            )
            seen_output_positions.add(pos)

    log.info(
        f"   📊 Second Pass output: {len(true_second_pass_output)} boreholes "
        f"(first_pass={len(first_pass_boreholes)}, removed={len(deduped_removed)}, "
        f"added={len(deduped_added)})"
    )

    # Store Second Pass boreholes before Third Pass for per-pass CSV export
    second_pass_boreholes = true_second_pass_output

    # === THIRD PASS: Cell-Cell CZRC Boundary Consolidation ===
    cell_czrc_config = config.get("cell_boundary_consolidation", {})
    # Inherit ILP config from parent if not explicitly set in cell_boundary_consolidation
    # This ensures Third Pass respects exclusion_factor and other ILP settings from config
    if "ilp" not in cell_czrc_config:
        cell_czrc_config = dict(cell_czrc_config)  # Don't mutate original
        cell_czrc_config["ilp"] = config.get("ilp", {})
    cell_czrc_enabled = cell_czrc_config.get("enabled", True)
    cell_czrc_stats: Dict[str, Any] = {"status": "disabled"}

    # Debug: Check third pass conditions
    log.info(f"   🔗 Third pass: enabled={cell_czrc_enabled}, {len(cells)} cells")

    if cell_czrc_enabled and len(cells) > 1:
        # Use First Pass test points (all_test_points), NOT Second Pass test points
        # Third Pass should filter First Pass test points to its Tier 1 area
        # and generate fresh Tier 2 ring test points for its Tier 2 area

        # Get spacing from cluster (use overall_r_max as uniform spacing for cells)
        cell_spacing = cluster.get("overall_r_max", 100.0)

        # Run cell-cell CZRC pass
        # CRITICAL: Use true_second_pass_output (First Pass survivors + Second Pass additions)
        # This includes boreholes that were always in Tier 2 (locked) during Second Pass
        # Grey markers show: All Second Pass survivors filtered to Third Pass Tier 1
        (
            consolidated_selected,
            cell_czrc_removed,
            cell_czrc_added,
            cell_czrc_stats,
        ) = run_cell_czrc_pass(
            cell_wkts=cell_wkts,
            cell_boreholes=true_second_pass_output,  # All Second Pass survivors as ILP input
            cell_test_points=all_test_points,  # CRITICAL: Use First Pass test points
            spacing=cell_spacing,
            config=cell_czrc_config,
            zones_clip_geometry=zones_clip_geometry,
            logger=logger,
            highs_log_folder=highs_log_folder,
            cluster_idx=cluster_idx,
            second_pass_candidates=true_second_pass_output,  # For grey marker visualization
        )

        if cell_czrc_stats.get("status") == "success":
            all_selected = consolidated_selected
            deduped_removed.extend(cell_czrc_removed)
            deduped_added.extend(cell_czrc_added)
            # Re-deduplicate removed/added after adding cell CZRC results
            final_removed_positions: set = set()
            final_deduped_removed: List[Dict[str, Any]] = []
            for bh in deduped_removed:
                pos = (bh["x"], bh["y"])
                if pos not in final_removed_positions:
                    final_deduped_removed.append(bh)
                    final_removed_positions.add(pos)
            deduped_removed = final_deduped_removed
            # Store third pass results separately for visualization
            cell_czrc_stats["third_pass_removed"] = cell_czrc_removed
            cell_czrc_stats["third_pass_added"] = cell_czrc_added
    # === END THIRD PASS ===

    cluster_key = "+".join(sorted(cluster["pair_keys"]))
    return (
        all_selected,
        deduped_removed,
        deduped_added,
        {
            "status": "success",
            "cluster_key": cluster_key,
            "was_split": True,
            "original_area_m2": cluster_area,
            "cell_count": len(cells),
            "cell_wkts": cell_wkts,  # For visualization
            "cell_stats": cell_stats,
            "selected_count": len(all_selected),
            "cell_czrc_stats": cell_czrc_stats,  # Third pass stats
            # Aggregated from all cells for visualization (grey markers in Second Pass Grid)
            "first_pass_candidates": all_first_pass_candidates,
            "czrc_test_points": all_czrc_test_points,
            # Second Pass boreholes (before Third Pass) for per-pass CSV export
            "second_pass_boreholes": second_pass_boreholes,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# 🔗 THIRD PASS: CELL-CELL BOUNDARY CONSOLIDATION
# ═══════════════════════════════════════════════════════════════════════════


def detect_cell_adjacencies(
    cell_geometries: List[BaseGeometry],
    spacing: float,
    test_spacing_mult: float = 0.2,
    clip_geometry: Optional[BaseGeometry] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Tuple[int, int, BaseGeometry]], List[BaseGeometry]]:
    """
    Detect adjacent cell pairs via coverage cloud intersection.

    This is the cell-level equivalent of compute_czrc_consolidation_region()
    but simplified for cells within the same cluster (uniform spacing).

    Args:
        cell_geometries: List of cell polygons (from cell_wkts)
        spacing: max_spacing_m inherited from parent zone (uniform for all cells)
        test_spacing_mult: Test point grid density multiplier
        clip_geometry: Optional geometry to clip cells to (actual zone shapefile)
        logger: Optional logger

    Returns:
        Tuple of:
        - List of (cell_i_idx, cell_j_idx, intersection_geometry) tuples
          where intersection is the cell-cell CZRC region.
        - List of coverage clouds for each cell (for visualization)
    """
    from Gap_Analysis_EC7.solvers.czrc_geometry import (
        compute_zone_coverage_cloud,
        compute_pairwise_intersection,
    )

    grid_spacing = spacing * test_spacing_mult

    log = logger or _logger  # Use module logger as fallback
    log.info(
        f"   🔍 Cell adjacency detection: {len(cell_geometries)} cells, spacing={spacing}m"
    )

    # Step 1: Compute coverage cloud for each cell
    # If clip_geometry provided, intersect cell with it before generating test points
    # This ensures test points are only within actual zone shapefile, not full Tier 1 buffer
    clouds: List[BaseGeometry] = []
    for idx, cell_geom in enumerate(cell_geometries):
        # Clip cell to actual zone geometry if provided
        if clip_geometry is not None:
            clipped_cell = cell_geom.intersection(clip_geometry)
            if clipped_cell.is_empty:
                # Cell doesn't intersect zone - use empty cloud
                clouds.append(cell_geom.buffer(0))  # Empty but valid geometry
                continue
            cell_for_cloud = clipped_cell
        else:
            cell_for_cloud = cell_geom

        cloud = compute_zone_coverage_cloud(
            zone_geometry=cell_for_cloud,
            max_spacing_m=spacing,
            grid_spacing=grid_spacing,
            logger=None,  # Suppress per-cell logging
        )
        clouds.append(cloud)

    # Step 2: Find pairwise intersections
    adjacencies: List[Tuple[int, int, BaseGeometry]] = []
    n = len(clouds)

    for i in range(n):
        for j in range(i + 1, n):
            intersection = compute_pairwise_intersection(
                cloud_a=clouds[i],
                cloud_b=clouds[j],
                zone_name_a=f"Cell_{i}",
                zone_name_b=f"Cell_{j}",
                logger=None,  # Suppress per-pair logging
            )
            if intersection is not None and not intersection.is_empty:
                log.info(f"      ✅ Cell_{i} ↔ Cell_{j}: {intersection.area:.0f}m²")
                adjacencies.append((i, j, intersection))

    log.info(f"   🔗 Cell adjacency: {len(adjacencies)} pairs from {n} cells")

    return adjacencies, clouds


def solve_cell_cell_czrc(
    cell_i: int,
    cell_j: int,
    czrc_region: BaseGeometry,
    cell_boreholes: List[Dict[str, Any]],
    cell_test_points: List[Dict[str, Any]],
    spacing: float,
    config: Dict[str, Any],
    zones_clip_geometry: Optional[BaseGeometry] = None,
    logger: Optional[logging.Logger] = None,
    highs_log_file: Optional[str] = None,
    second_pass_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """
    Solve ILP for a cell-cell CZRC region.

    This is a thin wrapper that reuses all existing CZRC infrastructure.

    Args:
        cell_i: Index of first cell
        cell_j: Index of second cell
        czrc_region: Cell-cell coverage cloud intersection
        cell_boreholes: Boreholes output from cell processing (input to third pass)
        cell_test_points: Test points from cell processing
        spacing: max_spacing_m (uniform for cells in same cluster)
        config: Cell CZRC config section
        zones_clip_geometry: Optional geometry to clip test points to (actual zone shapefile)
        logger: Optional logger
        highs_log_file: Optional path to write HiGHS solver log
        second_pass_candidates: Second Pass OUTPUT boreholes (survivors + added) for
            grey marker visualization. Filtered to Third Pass Tier 1 before display.
            If None, uses bh_candidates (cell_boreholes in Tier 1) instead.

    Returns:
        (selected, removed, added, stats) tuple
    """
    start_time = time.perf_counter()
    pair_key = f"Cell_{cell_i}_Cell_{cell_j}"

    # Step 1: Compute tiers (REUSE existing function)
    tier1_mult = config.get("tier1_rmax_multiplier", 1.0)
    tier2_mult = config.get("tier2_rmax_multiplier", 2.0)
    cell_spacings = {f"Cell_{cell_i}": spacing, f"Cell_{cell_j}": spacing}

    tier1_region, tier2_region, r_max = compute_czrc_tiers(
        czrc_region, cell_spacings, pair_key, tier1_mult, tier2_mult
    )

    # Step 2: Filter test points to Tier 1 (REUSE)
    tier1_test_points = filter_test_points_to_tier1(cell_test_points, tier1_region)

    # Prepare grey marker visualization: Second Pass OUTPUT filtered to Third Pass Tier 1
    # These show which boreholes entered Third Pass re-optimization as candidates
    if second_pass_candidates is not None:
        viz_candidates, _, _ = classify_first_pass_boreholes(
            second_pass_candidates, tier1_region, tier2_region
        )
    else:
        viz_candidates = None  # Will fall back to bh_candidates below

    if not tier1_test_points:
        # Still classify boreholes for visualization even when skipping
        bh_candidates_viz, _, _ = classify_first_pass_boreholes(
            cell_boreholes, tier1_region, tier2_region
        )
        # Use viz_candidates for grey markers if provided, else fall back to bh_candidates_viz
        actual_viz = viz_candidates if viz_candidates is not None else bh_candidates_viz
        return (
            cell_boreholes,
            [],
            [],
            {
                "status": "skipped",
                "reason": "no_tier1_test_points",
                # Include tier geometry WKTs for diagnostic export
                "tier1_wkt": tier1_region.wkt,
                "tier2_wkt": tier2_region.wkt,
                "r_max": r_max,
                # Include existing boreholes for visualization even when skipped
                "viz_existing_boreholes": actual_viz,
            },
        )

    # Step 2b: Generate Tier 2 ring test points (sparse) for this cell-cell pair
    t2_protection = config.get("tier2_test_point_protection", {})
    t2_enabled = t2_protection.get("enabled", True)
    t2_multiplier = t2_protection.get("tier2_test_spacing_multiplier", 3.0)
    base_test_mult = config.get("test_spacing_mult", 0.2)

    tier2_ring_test_points: List[Dict[str, Any]] = []
    if t2_enabled:
        tier2_ring_test_points = generate_tier2_ring_test_points(
            tier1_region,
            tier2_region,
            r_max,
            tier2_spacing_multiplier=t2_multiplier,
            base_test_spacing_mult=base_test_mult,
            clip_geometry=zones_clip_geometry,
        )

    # Step 3: Classify boreholes with external coverage detection
    # Use second_pass_candidates (all Second Pass output) for external coverage detection
    # This captures boreholes from OTHER cells that can cover this cell-pair's test points
    bh_candidates, locked, external = classify_first_pass_boreholes(
        cell_boreholes,
        tier1_region,
        tier2_region,
        all_boreholes=second_pass_candidates,  # Full Second Pass output for external detection
        r_max=r_max,
    )

    if not bh_candidates:
        # No candidates in Tier 1 - nothing to optimize
        # Still return tier geometries and locked boreholes for diagnostics
        # Use viz_candidates for grey markers if available (shows ALL Second Pass candidates)
        actual_viz = viz_candidates if viz_candidates is not None else []
        return (
            cell_boreholes,
            [],
            [],
            {
                "status": "skipped",
                "reason": "no_candidates",
                # Include tier geometry WKTs for diagnostic export
                "tier1_wkt": tier1_region.wkt,
                "tier2_wkt": tier2_region.wkt,
                "r_max": r_max,
                # Use viz_candidates for grey markers if available
                "viz_existing_boreholes": actual_viz,
            },
        )

    # Step 4: Compute unsatisfied test points including external coverage
    tier1_unsatisfied, precovered_ct = _compute_unsatisfied_test_points(
        tier1_test_points, locked, logger, external_boreholes=external
    )

    # Step 5: Prepare candidates (REUSE)
    candidates, _ = _prepare_candidates_for_ilp(
        tier1_region, bh_candidates, r_max, spacing, config
    )

    # Step 6: Solve ILP (REUSE)
    # Use tier1_test_points if all are precovered (allows removing redundant candidates)
    solve_test_points = tier1_unsatisfied if tier1_unsatisfied else tier1_test_points
    selected_indices, ilp_stats = _solve_czrc_ilp(
        solve_test_points,
        candidates,
        spacing,
        config,
        logger,
        highs_log_file,
    )

    # Step 7: Classify results
    if selected_indices is None:
        return cell_boreholes, [], [], {"status": "failed", "ilp_stats": ilp_stats}

    selected_set = set(selected_indices)
    selected_positions = {(candidates[i].x, candidates[i].y) for i in selected_set}

    # Candidate positions from first-pass
    candidate_positions = {(bh["x"], bh["y"]) for bh in bh_candidates}

    # Removed = first-pass candidates NOT in selected
    removed = [
        bh for bh in bh_candidates if (bh["x"], bh["y"]) not in selected_positions
    ]

    # Added = selected positions NOT from first-pass
    added = []
    for i in selected_set:
        pos = (candidates[i].x, candidates[i].y)
        if pos not in candidate_positions:
            added.append({"x": pos[0], "y": pos[1], "coverage_radius": spacing})

    # Build final selected list: locked (Tier 2) + selected from ILP
    selected = list(locked)
    for i in selected_set:
        selected.append(
            {"x": candidates[i].x, "y": candidates[i].y, "coverage_radius": spacing}
        )

    elapsed = time.perf_counter() - start_time

    # Grey markers: Second Pass OUTPUT filtered to Third Pass Tier 1
    # Falls back to bh_candidates (cell_boreholes in Tier 1) if not provided
    actual_viz_boreholes = (
        viz_candidates if viz_candidates is not None else bh_candidates
    )

    return (
        selected,
        removed,
        added,
        {
            "status": "success",
            "pair_key": pair_key,
            "tier1_test_points": len(tier1_test_points),
            "tier1_unsatisfied": len(tier1_unsatisfied) if tier1_unsatisfied else 0,
            "tier2_ring_test_points": len(tier2_ring_test_points),
            "precovered": precovered_ct,
            "external_coverage_bhs": len(external),  # NEW: Track external coverage sources
            "candidates": len(candidates),
            "bh_candidates": len(bh_candidates),
            "locked": len(locked),
            "selected": len(selected_set),
            "removed": len(removed),
            "added": len(added),
            "solve_time": elapsed,
            "ilp_stats": ilp_stats,
            # Tier geometry WKTs for diagnostic export
            "tier1_wkt": tier1_region.wkt,
            "tier2_wkt": tier2_region.wkt,
            "r_max": r_max,
            # Visualization data: test points with is_covered flag including external coverage
            "viz_tier1_test_points": _annotate_test_points_with_coverage(
                tier1_test_points, locked, external_boreholes=external
            ),
            "viz_tier2_ring_test_points": _annotate_test_points_with_coverage(
                tier2_ring_test_points, locked, external_boreholes=external
            ),
            # Grey markers: Second Pass OUTPUT boreholes filtered to Third Pass Tier 1
            # These are the candidates that entered Third Pass re-optimization
            "viz_existing_boreholes": actual_viz_boreholes,
        },
    )


def run_cell_czrc_pass(
    cell_wkts: List[str],
    cell_boreholes: List[Dict[str, Any]],
    cell_test_points: List[Dict[str, Any]],
    spacing: float,
    config: Dict[str, Any],
    zones_clip_geometry: Optional[BaseGeometry] = None,
    logger: Optional[logging.Logger] = None,
    highs_log_folder: Optional[str] = None,
    cluster_idx: int = 0,
    second_pass_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]
]:
    """
    Run third pass cell CZRC optimization for a split cluster.

    Args:
        cell_wkts: WKT strings of cell geometries (from cluster_stats["cell_wkts"])
        cell_boreholes: Boreholes selected by cell processing
        cell_test_points: Test points from cell processing
        spacing: max_spacing_m (uniform for cells in cluster)
        config: Cell CZRC config section
        zones_clip_geometry: Optional geometry to clip cells to (actual zone shapefile)
        logger: Optional logger
        highs_log_folder: Optional folder path for HiGHS solver logs
        cluster_idx: Cluster index for log file naming (0-based)
        second_pass_candidates: Second Pass OUTPUT boreholes (survivors + added) for
            grey marker visualization. These are filtered to Third Pass Tier 1 before
            display. If None, uses cell_boreholes for visualization.

    Returns:
        (final_boreholes, all_removed, all_added, stats) tuple
    """
    import os

    start_time = time.perf_counter()
    log = logger or _logger

    # Step 1: Parse cell geometries
    cell_geometries = [wkt.loads(w) for w in cell_wkts]

    if len(cell_geometries) < 2:
        return cell_boreholes, [], [], {"status": "skipped", "reason": "single_cell"}

    # Step 2: Detect cell adjacencies
    test_spacing_mult = config.get("test_spacing_mult", 0.2)

    # Debug: log cell geometry info
    if log:
        for idx, geom in enumerate(cell_geometries):
            log.info(
                f"      📐 Cell {idx}: area={geom.area:.0f}m², bounds={geom.bounds}"
            )

    adjacencies, cell_coverage_clouds = detect_cell_adjacencies(
        cell_geometries, spacing, test_spacing_mult, zones_clip_geometry, logger
    )

    if log:
        log.info(
            f"   🔗 detect_cell_adjacencies found {len(adjacencies)} pairs from {len(cell_geometries)} cells with spacing={spacing}m"
        )

    if not adjacencies:
        return cell_boreholes, [], [], {"status": "skipped", "reason": "no_adjacencies"}

    log.info(
        f"   🔗 Cell CZRC (Third Pass): Processing {len(adjacencies)} cell-cell pairs"
    )

    # Step 2b: Use same clouds from adjacency detection for visualization consistency
    # These are the clouds whose intersections create the blue overlap regions
    cell_clouds_wkt: Dict[int, str] = {}
    for cell_idx, cloud in enumerate(cell_coverage_clouds):
        if not cloud.is_empty:
            cell_clouds_wkt[cell_idx] = cloud.wkt

    # Step 2c: Store cell-cell intersection WKTs for visualization
    cell_intersections_wkt: Dict[str, str] = {}
    for cell_i, cell_j, czrc_region in adjacencies:
        pair_key = f"Cell_{cell_i}_Cell_{cell_j}"
        if not czrc_region.is_empty:
            cell_intersections_wkt[pair_key] = czrc_region.wkt

    # Step 3: Process each adjacent pair
    all_removed: List[Dict[str, Any]] = []
    all_added: List[Dict[str, Any]] = []
    pair_stats: List[Dict[str, Any]] = []
    current_boreholes = list(cell_boreholes)

    for pair_idx, (cell_i, cell_j, czrc_region) in enumerate(adjacencies):
        # Generate HiGHS log file path for this cell-cell pair
        highs_log_file = None
        if highs_log_folder:
            os.makedirs(highs_log_folder, exist_ok=True)
            highs_log_file = os.path.join(
                highs_log_folder,
                f"third_c{cluster_idx + 1:02d}_p{pair_idx + 1:02d}.log",
            )

        log.info(
            f"      🔄 Cell_{cell_i} ↔ Cell_{cell_j} (pair {pair_idx+1}/{len(adjacencies)})"
        )

        selected, removed, added, stats = solve_cell_cell_czrc(
            cell_i,
            cell_j,
            czrc_region,
            current_boreholes,
            cell_test_points,
            spacing,
            config,
            zones_clip_geometry=zones_clip_geometry,
            logger=logger,
            highs_log_file=highs_log_file,
            second_pass_candidates=second_pass_candidates,
        )

        if stats.get("status") == "success":
            # FIX: Instead of replacing current_boreholes with selected (which only contains
            # boreholes from this pair's Tier1+Tier2), we need to:
            # 1. Remove the 'removed' boreholes from current_boreholes
            # 2. Add the 'added' boreholes to current_boreholes
            removed_positions = {(bh["x"], bh["y"]) for bh in removed}
            current_boreholes = [
                bh
                for bh in current_boreholes
                if (bh["x"], bh["y"]) not in removed_positions
            ]
            current_boreholes.extend(added)
            all_removed.extend(removed)
            all_added.extend(added)
        else:
            log.info(
                f"      ⏭️ Pair {pair_idx+1}: {stats.get('status')}, reason={stats.get('reason')}"
            )

        pair_stats.append(stats)

    # Deduplicate removed/added
    removed_positions: set = set()
    deduped_removed: List[Dict[str, Any]] = []
    for bh in all_removed:
        pos = (bh["x"], bh["y"])
        if pos not in removed_positions:
            deduped_removed.append(bh)
            removed_positions.add(pos)

    # Collect test points from all pairs for visualization (deduplicated)
    all_third_pass_test_points: List[Dict[str, Any]] = []
    seen_test_point_positions: set = set()
    for ps in pair_stats:
        # Tier 1 test points (from first pass, filtered to Third Pass Tier 1 area)
        for tp in ps.get("viz_tier1_test_points", []):
            pos = (tp["x"], tp["y"])
            if pos not in seen_test_point_positions:
                all_third_pass_test_points.append(tp)
                seen_test_point_positions.add(pos)
        # Tier 2 ring test points (freshly generated for Third Pass)
        for tp in ps.get("viz_tier2_ring_test_points", []):
            pos = (tp["x"], tp["y"])
            if pos not in seen_test_point_positions:
                all_third_pass_test_points.append(tp)
                seen_test_point_positions.add(pos)

    # Collect existing boreholes (from Second Pass output) for visualization
    all_existing_boreholes: List[Dict[str, Any]] = []
    seen_existing_positions: set = set()
    for ps in pair_stats:
        for bh in ps.get("viz_existing_boreholes", []):
            pos = (bh["x"], bh["y"])
            if pos not in seen_existing_positions:
                all_existing_boreholes.append(bh)
                seen_existing_positions.add(pos)

    # Collect tier geometries for diagnostic export
    tier_geometries: Dict[str, Dict[str, Any]] = {}
    for ps in pair_stats:
        pair_key = ps.get("pair_key", "")
        tier1_wkt = ps.get("tier1_wkt")
        tier2_wkt = ps.get("tier2_wkt")
        if tier1_wkt or tier2_wkt:
            tier_geometries[pair_key] = {
                "tier1_wkt": tier1_wkt,
                "tier2_wkt": tier2_wkt,
                "r_max": ps.get("r_max", 100.0),
            }

    elapsed = time.perf_counter() - start_time

    log.info(
        f"   ✅ Cell CZRC complete: {len(deduped_removed)} removed, "
        f"{len(all_added)} added ({elapsed:.2f}s)"
    )

    return (
        current_boreholes,
        deduped_removed,
        all_added,
        {
            "status": "success",
            "cell_count": len(cell_geometries),
            "pairs_processed": len(adjacencies),
            "total_removed": len(deduped_removed),
            "total_added": len(all_added),
            "pair_stats": pair_stats,
            "solve_time": elapsed,
            # Visualization data for Third Pass layers
            "cell_clouds_wkt": cell_clouds_wkt,
            "cell_intersections_wkt": cell_intersections_wkt,
            # Third Pass test points (Tier 1 filtered + Tier 2 ring generated)
            "third_pass_test_points": all_third_pass_test_points,
            # Existing boreholes (from Second Pass output) for grey marker display
            "third_pass_existing_boreholes": all_existing_boreholes,
            # Tier geometries for diagnostic export (GeoJSON)
            "tier_geometries": tier_geometries,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# 🔗 UNIFIED CLUSTER SOLVER
# ═══════════════════════════════════════════════════════════════════════════


def solve_czrc_ilp_for_cluster(
    cluster: Dict[str, Any],
    zone_spacings: Dict[str, float],
    all_test_points: List[Dict[str, Any]],
    first_pass_boreholes: List[Dict[str, Any]],
    config: Dict[str, Any],
    zones_clip_geometry: Optional[BaseGeometry] = None,
    logger: Optional[logging.Logger] = None,
    czrc_cache: Optional["CZRCCacheManager"] = None,
    highs_log_folder: Optional[str] = None,
    cluster_idx: int = 0,
) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]
]:
    """
    Solve CZRC ILP for a unified cluster of overlapping pairs.

    Uses a single hexagonal grid for the unified tier1 region, ensuring
    consistent grid alignment across all constituent pairs.

    Args:
        cluster: Cluster dict from _group_overlapping_pairs()
        zone_spacings: Dict mapping zone_name -> max_spacing_m
        all_test_points: From all_stats["test_points"]
        first_pass_boreholes: From all_stats["selected_boreholes"]
        config: CZRC config section
        zones_clip_geometry: Optional geometry to clip Tier 2 test points to
        logger: Optional logger
        czrc_cache: Optional CZRCCacheManager for intra-run result caching.
            When provided, caches ILP results based on the actual problem
            definition (tier1 geometry, spacings, unsatisfied test points).
        highs_log_folder: Optional folder for HiGHS log files
        cluster_idx: Index for unique log file naming (e.g., "czrc_cluster_003.log")

    Returns:
        (selected, removed, added, stats) tuple
    """
    start_time = time.perf_counter()

    unified_tier1 = cluster["unified_tier1"]
    overall_r_max = cluster["overall_r_max"]
    pair_keys = cluster["pair_keys"]

    # Compute unified Tier 2 = unified_tier1 expanded
    tier2_mult = config.get("tier2_rmax_multiplier", 2.0)
    tier1_mult = config.get("tier1_rmax_multiplier", 1.0)
    # Tier2 = Tier1 + (tier2_mult - tier1_mult) * R_max
    tier2_buffer_extra = (tier2_mult - tier1_mult) * overall_r_max
    unified_tier2 = unified_tier1.buffer(tier2_buffer_extra)

    # Step 1: Filter test points to unified Tier 1
    tier1_test_points = filter_test_points_to_tier1(all_test_points, unified_tier1)
    if not tier1_test_points:
        return [], [], [], {"status": "skipped", "reason": "no_tier1_test_points"}

    # Step 2: Classify boreholes with external coverage detection
    # first_pass_boreholes contains ALL First Pass output from all zones,
    # so we use it for both classification AND external detection
    bh_candidates, locked, external = classify_first_pass_boreholes(
        first_pass_boreholes,
        unified_tier1,
        unified_tier2,
        all_boreholes=first_pass_boreholes,  # Use full First Pass output for external detection
        r_max=overall_r_max,
    )

    # Step 3: Compute unsatisfied test points including external coverage
    tier1_unsatisfied, precovered_ct = _compute_unsatisfied_test_points(
        tier1_test_points, locked, logger, external_boreholes=external
    )

    # Step 3.5: Generate Tier 2 ring test points if enabled
    t2_protection = config.get("tier2_test_point_protection", {})
    t2_enabled = t2_protection.get("enabled", True)
    t2_multiplier = t2_protection.get("tier2_test_spacing_multiplier", 3.0)
    base_test_mult = config.get("test_spacing_mult", 0.2)

    tier2_ring_test_points = []
    tier2_ring_unsatisfied = []
    if t2_enabled:
        tier2_ring_test_points = generate_tier2_ring_test_points(
            unified_tier1,
            unified_tier2,
            overall_r_max,
            tier2_spacing_multiplier=t2_multiplier,
            base_test_spacing_mult=base_test_mult,
            clip_geometry=zones_clip_geometry,
        )
        if tier2_ring_test_points:
            tier2_ring_unsatisfied, _ = _compute_unsatisfied_test_points(
                tier2_ring_test_points, locked, logger, external_boreholes=external
            )

    # Combine Tier 1 and Tier 2 ring unsatisfied test points
    unsatisfied = tier1_unsatisfied + tier2_ring_unsatisfied

    if not unsatisfied:
        return (
            [],
            [],
            [],
            {
                "status": "success",
                "reason": "all_precovered",
                "precovered_count": len(tier1_test_points),
            },
        )

    # Step 4: Prepare candidates using UNIFIED tier1 region (single grid)
    # Get aggregated spacing from all zones using config method
    all_zones = set()
    for pk in pair_keys:
        all_zones.update(pk.split("_"))
    # Use cross_zone_exclusion_method from config to aggregate spacings
    exclusion_method = _get_cross_zone_exclusion_method(config)
    min_spacing = _aggregate_zone_spacings(
        zone_spacings, list(all_zones), exclusion_method
    )

    candidates, _ = _prepare_candidates_for_ilp(
        unified_tier1, bh_candidates, overall_r_max, min_spacing, config
    )
    if not candidates:
        return [], [], [], {"status": "failed", "reason": "no_candidates"}

    # Step 5: Solve ILP (with optional caching)
    # IMPORTANT: Candidates must be generated BEFORE cache check because
    # n_candidates is needed for cache validation
    cluster_key = "+".join(sorted(pair_keys))

    # Generate HiGHS log file path if folder provided
    highs_log_file: Optional[str] = None
    if highs_log_folder:
        import os

        os.makedirs(highs_log_folder, exist_ok=True)
        # Use 1-based naming to match HTML tooltip display:
        # - HTML shows "CZRC Cell {i + 1}" for cells (Cell 1, Cell 2, etc.)
        # - Cluster numbering also uses 1-based for consistency
        # cluster_idx < 1000: cell solve (e.g., cluster 0 cell 0 → c01_cell01)
        # cluster_idx >= 1000: direct solve (e.g., cluster 1 → c02_direct)
        if cluster_idx >= 1000:
            # Direct solve (from check_and_split_large_cluster non-split path)
            actual_cluster_idx = cluster_idx - 1000
            # Use 1-based cluster number
            highs_log_file = os.path.join(
                highs_log_folder, f"czrc_c{actual_cluster_idx + 1:02d}_direct.log"
            )
        else:
            # Cell solve (from split cluster)
            cell_cluster_idx = cluster_idx // 100
            cell_idx = cluster_idx % 100
            # Use 1-based for both cluster and cell to match HTML display
            highs_log_file = os.path.join(
                highs_log_folder,
                f"czrc_c{cell_cluster_idx + 1:02d}_cell{cell_idx + 1:02d}.log",
            )

    if czrc_cache is not None:
        # Use cache: key is based on actual problem (unsatisfied test points)
        tier1_wkt = unified_tier1.wkt

        def compute_ilp() -> Tuple[List[int], Dict[str, Any]]:
            """Compute function for cache miss."""
            idx, stats = _solve_czrc_ilp(
                unsatisfied, candidates, min_spacing, config, logger, highs_log_file
            )
            return idx if idx is not None else [], stats

        indices, ilp_stats = czrc_cache.get_or_compute(
            cluster_key=cluster_key,
            tier1_wkt=tier1_wkt,
            zone_spacings=zone_spacings,
            unsatisfied_test_points=unsatisfied,
            n_candidates=len(candidates),
            compute_fn=compute_ilp,
        )
        # Handle empty indices from cache (indicates no solution needed or ILP failure)
        if not indices:
            indices = None if ilp_stats.get("status") == "failed" else []
    else:
        # Direct solve (no caching)
        indices, ilp_stats = _solve_czrc_ilp(
            unsatisfied, candidates, min_spacing, config, logger, highs_log_file
        )

    if indices is None:
        return (
            [],
            [],
            [],
            {"status": "failed", "reason": "ilp_failed", "ilp_stats": ilp_stats},
        )

    # Step 6: Assemble results
    selected, removed, added = _assemble_czrc_results(
        indices, candidates, bh_candidates, min_spacing
    )
    elapsed = time.perf_counter() - start_time

    cluster_key = "+".join(sorted(pair_keys))
    stats = {
        "status": "success",
        "cluster_key": cluster_key,
        "pair_keys": pair_keys,
        "is_unified_cluster": len(pair_keys) > 1,
        "overall_r_max": overall_r_max,
        "r_max": overall_r_max,  # For tier geometry export
        "tier1_wkt": unified_tier1.wkt,  # For tier geometry export
        "tier2_wkt": unified_tier2.wkt,  # For tier geometry export
        "tier1_test_points": len(tier1_test_points),
        "tier2_ring_test_points": len(tier2_ring_test_points),
        "precovered_count": precovered_ct,
        "external_coverage_bhs": len(external),  # Track external coverage sources
        "unsatisfied_count": len(unsatisfied),
        "candidates_count": len(candidates),
        "selected_count": len(selected),
        "solve_time": elapsed,
        "ilp_stats": ilp_stats,
        "first_pass_candidates": bh_candidates,
        # CZRC test points for visualization with is_covered flag including external coverage
        "czrc_test_points": (
            _annotate_test_points_with_coverage(
                tier1_test_points, locked, external_boreholes=external
            )
            + _annotate_test_points_with_coverage(
                tier2_ring_test_points, locked, external_boreholes=external
            )
        ),
        # Second Pass output = selected (for direct ILP, this is final; for split, Third Pass comes after)
        "second_pass_boreholes": [
            {
                "x": bh["x"],
                "y": bh["y"],
                "coverage_radius": bh.get("coverage_radius", min_spacing),
            }
            for bh in selected
        ],
    }
    return selected, removed, added, stats


# ═══════════════════════════════════════════════════════════════════════════
# 🚀 MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


def run_czrc_optimization(
    first_pass_boreholes: List[Dict[str, Any]],
    all_test_points: List[Dict[str, Any]],
    czrc_data: Dict[str, Any],
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    czrc_cache: Optional["CZRCCacheManager"] = None,
    highs_log_folder: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main entry point for CZRC second-pass optimization.

    Args:
        first_pass_boreholes: Boreholes from first-pass zone decomposition
        all_test_points: Test points from first-pass (has x, y, required_radius)
        czrc_data: CZRC geometry data (pairwise_wkts, zone_spacings, etc.)
        config: CZRC config section
        logger: Optional logger
        czrc_cache: Optional CZRCCacheManager for intra-run result caching.
            When provided, caches ILP results to skip redundant computations
            when different filter combinations produce equivalent problems.
        highs_log_folder: Optional folder path for HiGHS solver log files.
            When provided, writes czrc_*.log files for each cluster/cell solve.

    Returns:
        Tuple of (optimized_boreholes, czrc_stats) where czrc_stats includes
        removed_boreholes and added_boreholes for visualization.
    """
    start_time = time.perf_counter()

    if not config.get("enabled", False):
        return first_pass_boreholes, {"status": "disabled"}

    pairwise_wkts = czrc_data.get("pairwise_wkts", {})
    zone_spacings = czrc_data.get("zone_spacings", {})

    if not pairwise_wkts:
        return first_pass_boreholes, {"status": "skipped", "reason": "no_pairwise"}

    # Build zones clip geometry from raw zone geometries for Tier 2 test point clipping
    # Use raw zone geometries (not buffered coverage clouds) to ensure test points
    # stay within actual zone boundaries
    zone_geometries = czrc_data.get("zone_geometries", {})
    zones_clip_geometry: Optional[BaseGeometry] = None
    if zone_geometries:
        # Use raw zone geometry objects (preferred - actual zone boundaries)
        zones_clip_geometry = unary_union(list(zone_geometries.values()))
    else:
        # Fallback to coverage clouds only if zone_geometries not available
        # (legacy compatibility - coverage clouds extend beyond zone boundaries)
        coverage_clouds = czrc_data.get("coverage_clouds", {})
        coverage_clouds_wkt = czrc_data.get("coverage_clouds_wkt", {})
        if coverage_clouds:
            zones_clip_geometry = unary_union(list(coverage_clouds.values()))
        elif coverage_clouds_wkt:
            cloud_geoms = [wkt.loads(w) for w in coverage_clouds_wkt.values()]
            zones_clip_geometry = unary_union(cloud_geoms)

    # Group overlapping pairs into clusters for unified grid generation
    tier1_mult = config.get("tier1_rmax_multiplier", 1.0)
    clusters = _group_overlapping_pairs(pairwise_wkts, zone_spacings, tier1_mult)

    if not clusters:
        return first_pass_boreholes, {"status": "skipped", "reason": "no_clusters"}

    # Process each cluster (single-pair or unified multi-pair)
    all_cluster_stats: Dict[str, Any] = {}
    all_removed: List[Dict[str, Any]] = []
    all_added: List[Dict[str, Any]] = []
    all_first_pass_candidates: List[Dict[str, Any]] = []
    all_czrc_test_points: List[Dict[str, Any]] = []
    boreholes_to_add: List[Dict[str, Any]] = []
    seen_candidates: set = set()  # Track unique first-pass candidates by (x, y)
    seen_test_points: set = set()  # Track unique test points by (x, y)
    all_cell_wkts: List[str] = []  # Collect cell geometries from split clusters
    all_third_pass_removed: List[Dict[str, Any]] = []  # Third pass removed boreholes
    all_third_pass_added: List[Dict[str, Any]] = []  # Third pass added boreholes
    # Third pass visualization data
    all_cell_clouds_wkt: Dict[str, str] = {}  # Cell coverage clouds for viz
    all_cell_intersections_wkt: Dict[str, str] = {}  # Cell-cell intersections for viz
    all_third_pass_test_points: List[Dict[str, Any]] = []  # Third pass test points
    seen_third_pass_test_points: set = set()  # Track unique third pass test points
    all_third_pass_existing: List[Dict[str, Any]] = []  # Third pass existing boreholes
    seen_third_pass_existing: set = set()  # Track unique existing boreholes
    all_tier_geometries: Dict[str, Dict[str, Any]] = {}  # Third pass tier geometries
    all_second_pass_boreholes: List[Dict[str, Any]] = (
        []
    )  # Second pass output (before third pass)
    seen_second_pass: set = set()  # Track unique second pass boreholes by (x, y)
    cluster_idx = 0  # For unique log file naming

    for cluster in clusters:
        # Use wrapper that checks size and splits if needed
        selected, removed, added, cluster_stats = check_and_split_large_cluster(
            cluster,
            zone_spacings,
            all_test_points,
            first_pass_boreholes,
            config,
            zones_clip_geometry,
            logger,
            czrc_cache,  # Pass cache to cluster solver
            highs_log_folder,
            cluster_idx,
        )
        cluster_key = cluster_stats.get("cluster_key", "+".join(cluster["pair_keys"]))
        all_cluster_stats[cluster_key] = cluster_stats
        boreholes_to_add.extend(selected)
        all_removed.extend(removed)
        all_added.extend(added)
        cluster_idx += 1

        # Collect cell geometries from split clusters (for visualization)
        if cluster_stats.get("was_split", False):
            all_cell_wkts.extend(cluster_stats.get("cell_wkts", []))
            # Collect third pass data from split clusters
            cell_czrc_stats = cluster_stats.get("cell_czrc_stats", {})
            if cell_czrc_stats.get("status") == "success":
                all_third_pass_removed.extend(
                    cell_czrc_stats.get("third_pass_removed", [])
                )
                all_third_pass_added.extend(cell_czrc_stats.get("third_pass_added", []))
                # Collect visualization data with cluster prefix to avoid key collisions
                cell_clouds = cell_czrc_stats.get("cell_clouds_wkt", {})
                for cell_key, wkt_str in cell_clouds.items():
                    prefixed_key = f"cluster_{cluster_idx}_{cell_key}"
                    all_cell_clouds_wkt[prefixed_key] = wkt_str
                cell_ints = cell_czrc_stats.get("cell_intersections_wkt", {})
                for pair_key, wkt_str in cell_ints.items():
                    prefixed_key = f"cluster_{cluster_idx}_{pair_key}"
                    all_cell_intersections_wkt[prefixed_key] = wkt_str
                # Collect Third Pass test points (Tier 1 filtered + Tier 2 ring)
                for tp in cell_czrc_stats.get("third_pass_test_points", []):
                    pos = (tp["x"], tp["y"])
                    if pos not in seen_third_pass_test_points:
                        all_third_pass_test_points.append(tp)
                        seen_third_pass_test_points.add(pos)
                # Collect Third Pass existing boreholes (from Second Pass output)
                for bh in cell_czrc_stats.get("third_pass_existing_boreholes", []):
                    pos = (bh["x"], bh["y"])
                    if pos not in seen_third_pass_existing:
                        all_third_pass_existing.append(bh)
                        seen_third_pass_existing.add(pos)
                # Collect Third Pass tier geometries for diagnostic export
                tier_geoms = cell_czrc_stats.get("tier_geometries", {})
                for pair_key, geom_data in tier_geoms.items():
                    prefixed_key = f"cluster_{cluster_idx}_{pair_key}"
                    all_tier_geometries[prefixed_key] = geom_data

        # Collect unique first-pass candidates across all clusters
        for bh in cluster_stats.get("first_pass_candidates", []):
            pos = (bh["x"], bh["y"])
            if pos not in seen_candidates:
                all_first_pass_candidates.append(bh)
                seen_candidates.add(pos)

        # Collect unique CZRC test points across all clusters
        for tp in cluster_stats.get("czrc_test_points", []):
            pos = (tp["x"], tp["y"])
            if pos not in seen_test_points:
                all_czrc_test_points.append(tp)
                seen_test_points.add(pos)

        # Collect Second Pass boreholes (before Third Pass) for per-pass CSV export
        sp_bhs = cluster_stats.get("second_pass_boreholes", [])
        for bh in sp_bhs:
            pos = (bh["x"], bh["y"])
            if pos not in seen_second_pass:
                all_second_pass_boreholes.append(bh)
                seen_second_pass.add(pos)

    # Compute true Second Pass output:
    # = First Pass boreholes not removed by Second Pass
    # + boreholes added by Second Pass (before Third Pass)
    #
    # For split clusters: second_pass_boreholes already captures this (all_selected before Third Pass)
    # For unsplit clusters: second_pass_boreholes = CZRC output
    # But we also need First Pass boreholes that weren't in any overlap region!
    #
    # Strategy: Build Second Pass from (First Pass - Second Pass removed) + Second Pass added
    # Note: For split clusters, Third Pass removed/added are separate from Second Pass removed/added

    # Build sets of removed/added from Second Pass only (exclude Third Pass changes)
    second_pass_removed_positions: set = set()
    for bh in all_removed:
        pos = (bh["x"], bh["y"])
        # Check if this removal came from Third Pass
        is_third_pass = any(
            (tp_bh["x"], tp_bh["y"]) == pos for tp_bh in all_third_pass_removed
        )
        if not is_third_pass:
            second_pass_removed_positions.add(pos)

    # Build Second Pass output: First Pass survivors + Second Pass additions
    # Start with First Pass boreholes not removed by Second Pass
    computed_second_pass: List[Dict[str, Any]] = []
    for bh in first_pass_boreholes:
        pos = (bh["x"], bh["y"])
        if pos not in second_pass_removed_positions:
            computed_second_pass.append(
                {
                    "x": bh["x"],
                    "y": bh["y"],
                    "coverage_radius": bh.get("coverage_radius", 100.0),
                }
            )

    # Add Second Pass additions (from all_second_pass_boreholes that aren't in First Pass)
    existing_second_pass_pos = {(bh["x"], bh["y"]) for bh in computed_second_pass}
    for bh in all_second_pass_boreholes:
        pos = (bh["x"], bh["y"])
        if pos not in existing_second_pass_pos:
            computed_second_pass.append(bh)
            existing_second_pass_pos.add(pos)

    # Replace the aggregated second pass with the computed one
    all_second_pass_boreholes = computed_second_pass

    # Assemble result: first-pass + new boreholes (deduplicated), minus removed
    # Build set of removed positions for efficient filtering
    removed_positions = {(bh["x"], bh["y"]) for bh in all_removed}

    # Start with first-pass boreholes, excluding those marked for removal
    optimized = [
        bh for bh in first_pass_boreholes if (bh["x"], bh["y"]) not in removed_positions
    ]
    existing_pos = {(bh["x"], bh["y"]) for bh in optimized}

    # Add new boreholes (deduplicated)
    for bh in boreholes_to_add:
        if (bh["x"], bh["y"]) not in existing_pos:
            optimized.append(bh)
            existing_pos.add((bh["x"], bh["y"]))

    elapsed = time.perf_counter() - start_time

    # Count unified clusters (clusters with more than one pair)
    unified_cluster_count = sum(1 for c in clusters if len(c["pair_keys"]) > 1)

    # Aggregate gap percentages from all cluster ILP stats
    cluster_gaps = []
    for cluster_key, cluster_stat in all_cluster_stats.items():
        ilp_stats = cluster_stat.get("ilp_stats", {})
        stall_det = ilp_stats.get("stall_detection", {})
        if stall_det and stall_det.get("final_gap_pct") is not None:
            cluster_gaps.append(stall_det["final_gap_pct"])

    stall_detection = None
    if cluster_gaps:
        avg_gap = sum(cluster_gaps) / len(cluster_gaps)
        stall_detection = {
            "final_gap_pct": avg_gap,
            "cluster_gaps": cluster_gaps,
        }

    return optimized, {
        "status": "success",
        "clusters_processed": len(clusters),
        "unified_clusters": unified_cluster_count,
        "pairs_in_clusters": sum(len(c["pair_keys"]) for c in clusters),
        "cluster_stats": all_cluster_stats,
        "original_count": len(first_pass_boreholes),
        "final_count": len(optimized),
        "boreholes_removed": len(all_removed),
        "boreholes_added": len(all_added),
        "net_change": len(all_added) - len(all_removed),
        "removed_boreholes": all_removed,
        "added_boreholes": all_added,
        "first_pass_candidates": all_first_pass_candidates,
        "czrc_test_points": all_czrc_test_points,  # For visualization
        "cell_wkts": all_cell_wkts,  # Cell boundaries from split clusters
        "third_pass_removed": all_third_pass_removed,  # Third pass removed for viz
        "third_pass_added": all_third_pass_added,  # Third pass added for viz
        # Second Pass boreholes (before Third Pass) for per-pass CSV export
        "second_pass_boreholes": all_second_pass_boreholes,
        # Third pass visualization data (cell clouds, intersections, test points, existing boreholes)
        "third_pass_data": (
            {
                "cell_clouds_wkt": all_cell_clouds_wkt,
                "cell_intersections_wkt": all_cell_intersections_wkt,
                "third_pass_test_points": all_third_pass_test_points,
                "third_pass_existing_boreholes": all_third_pass_existing,
                "tier_geometries": all_tier_geometries,  # For diagnostic GeoJSON export
            }
            if all_cell_clouds_wkt
            or all_cell_intersections_wkt
            or all_third_pass_test_points
            or all_third_pass_existing
            or all_tier_geometries
            else None
        ),
        "solve_time": elapsed,
        "stall_detection": stall_detection,  # Aggregated gap stats
    }
