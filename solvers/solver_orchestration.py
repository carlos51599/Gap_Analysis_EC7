#!/usr/bin/env python3
"""
EC7 Borehole Placement Solver Orchestration Module

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Orchestrate borehole placement optimization including decomposition
strategies (zone-based, connected components) and solution aggregation.

Key Functions:
- compute_optimal_boreholes: Main entry point - routes to appropriate strategy
- _solve_component: Solve optimization for a single connected component
- _ensure_complete_coverage: Post-processing for remaining fragments
- verify_coverage: Validate solution completeness
- compare_approaches: Benchmark utility (optional)

CONFIGURATION ARCHITECTURE:
- No CONFIG access - all functions accept explicit parameters
- Decomposition strategies (zones, connected components) configurable
- Solver selection delegated to solvers module

DECOMPOSITION STRATEGIES:
- Zone decomposition: Split by external zone boundaries (e.g., embankment zones)
- Connected components: Split by spatial proximity (independent subproblems)
- Single solve: Direct optimization when decomposition not needed

For Navigation: Use VS Code outline (Ctrl+Shift+O) to jump between sections.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union

import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# Import from sibling modules within solvers package
from Gap_Analysis_EC7.solvers.optimization_geometry import (
    _normalize_gaps,
    _decompose_into_components,
    _generate_candidate_grid,
    _generate_test_points,
    _build_coverage_dict,
)
from Gap_Analysis_EC7.solvers.solver_algorithms import (
    resolve_solver_mode,
    _solve_ilp,
    _solve_greedy,
)
from Gap_Analysis_EC7.solvers.solver_config import (
    SolverConfig,
    config_from_legacy_params,
    create_default_config,
    create_fast_config,
    create_parallel_config,
    create_precision_config,
)
from Gap_Analysis_EC7.models.data_models import (
    Borehole,
    BoreholePass,
    BoreholeStatus,
    get_bh_coords,
    get_bh_position,
)


# ===========================================================================
# 📂 FILENAME UTILITIES
# ===========================================================================


def _abbreviate_zone_name(zone: str) -> str:
    """
    Abbreviate a zone name to first 2 letters for shorter log filenames.

    Joins zone abbreviation with its number (no underscore): "Embankment_0" → "em0".

    Args:
        zone: Full zone name (e.g., "Embankment_0", "Highways_1")

    Returns:
        Abbreviated name (e.g., "em0", "hi1")
    """
    # Split on underscore, abbreviate text parts and join with number
    parts = zone.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        # Zone with number: "Embankment_0" → "em0"
        text_part = "_".join(parts[:-1])
        return f"{text_part[:2].lower()}{parts[-1]}"
    else:
        # Single part: just abbreviate
        return zone[:2].lower()


def _sanitize_log_name(name: str, max_len: int = 50, abbreviate: bool = True) -> str:
    """
    Sanitize a name for use in log filenames.

    Replaces problematic characters with underscores, abbreviates zone names,
    and truncates if needed. Zone names are joined with their numbers (em2),
    underscores only between different zones/cells (em2_hi1).

    Args:
        name: Raw name (e.g., "Embankment_0+Embankment_1")
        max_len: Maximum length before truncation (default 50)
        abbreviate: Whether to abbreviate zone names to 2 letters (default True)

    Returns:
        Filesystem-safe abbreviated name (e.g., "em0_em1")
    """
    import re

    # Replace problematic characters (including + from cluster_key) with underscores
    safe = re.sub(r'[<>:"/\\|?*\[\]\s+]+', "_", name)
    # Remove leading/trailing underscores and collapse multiple underscores
    safe = re.sub(r"_+", "_", safe.strip("_"))

    # Abbreviate zone names if enabled
    if abbreviate:
        # Split by underscore and abbreviate each zone portion
        # Zone names are like "Embankment_0", join as "em0" (no underscore within zone)
        parts = safe.split("_")
        zone_tokens = []  # Each token is a complete zone like "em0" or cell like "c1"
        i = 0
        while i < len(parts):
            part = parts[i]
            # Check if this is a text part followed by a number (zone pattern)
            if not part.isdigit() and i + 1 < len(parts) and parts[i + 1].isdigit():
                # Abbreviate zone name: "Embankment_0" → "em0" (no underscore)
                zone_tokens.append(f"{part[:2].lower()}{parts[i + 1]}")
                i += 2
            elif part.isdigit():
                # Standalone number - append to previous token or as is
                zone_tokens.append(part)
                i += 1
            elif part.lower().startswith("cell"):
                # Convert Cell to c: "Cell0" → "c0"
                cell_num = part[4:] if len(part) > 4 else ""
                zone_tokens.append(f"c{cell_num}")
                i += 1
            else:
                # Abbreviate standalone text
                zone_tokens.append(part[:2].lower())
                i += 1
        # Join zone tokens with underscores (underscore only between different zones)
        safe = "_".join(zone_tokens)

    # Truncate if too long (preserving end for uniqueness)
    if len(safe) > max_len:
        safe = safe[: max_len - 3] + "..."
    return safe if safe else "unnamed"


# ===========================================================================
# 🏗️ MAIN ORCHESTRATION SECTION
# ===========================================================================


def optimize_boreholes(
    gaps: Union[Polygon, MultiPolygon, BaseGeometry],
    config: Optional[SolverConfig] = None,
    centreline_boreholes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    """
    Compute optimal borehole locations using SolverConfig.

    **This is the recommended public API** for borehole optimization.
    Uses a typed SolverConfig object instead of 31 individual parameters,
    providing better IDE support, validation, and maintainability.

    Args:
        gaps: Shapely geometry (Polygon/MultiPolygon) of uncovered areas
        config: SolverConfig object (use factory functions to create):

            - ``create_default_config()``: EC7-compliant ILP optimization
            - ``create_fast_config()``: Quick greedy optimization
            - ``create_parallel_config()``: Safe for multiprocessing
            - ``create_precision_config()``: High-quality results
            - ``config_from_project_config(CONFIG)``: From project CONFIG dict
            - ``config_from_ilp_dict(ilp_config)``: From worker ilp_config dict

    Returns:
        Tuple of (borehole_locations, optimization_stats)

    Example:
        >>> from Gap_Analysis_EC7.solvers.solver_config import create_default_config
        >>> config = create_default_config(max_spacing=200.0, verbose=1)
        >>> boreholes, stats = optimize_boreholes(gaps, config)

        >>> # Or from project CONFIG:
        >>> from Gap_Analysis_EC7.config import CONFIG
        >>> from Gap_Analysis_EC7.solvers.solver_config import config_from_project_config
        >>> config = config_from_project_config(CONFIG, zones_gdf=zones)
        >>> boreholes, stats = optimize_boreholes(gaps, config)
    """
    if config is None:
        config = create_default_config()

    # Extract stall detection config as dict for solver
    stall_detection_dict = None
    if (
        hasattr(config.ilp, "stall_detection")
        and config.ilp.stall_detection is not None
    ):
        stall_detection_dict = config.ilp.stall_detection.to_dict()

    # Delegate to implementation function with extracted parameters
    return compute_optimal_boreholes(
        uncovered_gaps=gaps,
        max_spacing=config.max_spacing,
        candidate_spacing=config.grid.candidate_spacing,
        test_spacing=config.grid.test_spacing,
        time_limit=config.ilp.time_limit,
        mip_gap=config.ilp.mip_gap,
        coverage_target_pct=config.coverage_target_pct,
        threads=config.ilp.threads,
        use_ilp=(config.solver_mode == "ilp"),
        solver_mode=config.solver_mode,
        is_parallel_context=config.is_parallel_context,
        fill_remaining_fragments=config.greedy.fill_remaining,
        greedy_max_iterations=config.greedy.max_iterations,
        greedy_min_gain=config.greedy.min_gain,
        greedy_min_efficiency_pct=config.greedy.min_efficiency_pct,
        candidate_grid_type=config.grid.grid_type,
        hexagonal_density=config.grid.hexagonal_density,
        use_conflict_constraints=config.conflict.enabled,
        conflict_constraint_mode=config.conflict.mode,
        exclusion_factor=config.conflict.exclusion_factor,
        max_conflict_pairs=config.conflict.max_conflict_pairs,
        min_clique_size=config.conflict.min_clique_size,
        max_cliques=config.conflict.max_cliques,
        use_connected_components=config.decomposition.use_connected_components,
        min_component_gaps_for_ilp=config.decomposition.min_component_gaps_for_ilp,
        use_zone_decomposition=config.decomposition.use_zone_decomposition,
        zones_gdf=config.decomposition.zones_gdf,
        min_zone_gap_area_m2=config.decomposition.min_zone_gap_area_m2,
        zone_cache_dir=config.decomposition.zone_cache_dir,  # Intra-run caching
        verbose=config.ilp.verbose,
        mip_heuristic_effort=config.ilp.mip_heuristic_effort,
        highs_log_folder=config.highs_log_folder,
        logger=config.logger,
        stall_detection_config=stall_detection_dict,
        centreline_boreholes=centreline_boreholes,
    )


def compute_optimal_boreholes(
    uncovered_gaps: Optional[BaseGeometry],
    max_spacing: float = 200.0,
    candidate_spacing: float = 50.0,
    test_spacing: float = 15.0,
    time_limit: int = 120,
    mip_gap: float = 0.03,
    coverage_target_pct: float = 100.0,
    threads: int = 1,
    use_ilp: bool = True,
    solver_mode: Optional[str] = None,
    is_parallel_context: bool = False,
    fill_remaining_fragments: bool = False,
    greedy_max_iterations: int = 1000,
    greedy_min_gain: float = 1.0,
    greedy_min_efficiency_pct: float = 8.0,
    candidate_grid_type: str = "hexagonal",
    hexagonal_density: float = 1.5,
    # Conflict constraint parameters
    use_conflict_constraints: bool = True,
    conflict_constraint_mode: str = "clique",
    exclusion_factor: float = 0.8,
    max_conflict_pairs: int = 200000,
    min_clique_size: int = 3,
    max_cliques: int = 50000,
    # Connected components decomposition parameters
    use_connected_components: bool = True,
    min_component_gaps_for_ilp: int = 1,
    # Zone-based decomposition parameters
    use_zone_decomposition: bool = False,
    zones_gdf: Optional[gpd.GeoDataFrame] = None,
    min_zone_gap_area_m2: float = 100.0,
    zone_cache_dir: Optional[str] = None,  # Intra-run caching directory
    # Verbose mode for ILP solver progress
    verbose: int = 0,
    # MIP heuristic effort (HiGHS parameter)
    mip_heuristic_effort: float = 0.05,
    # HiGHS log capture
    highs_log_folder: Optional[str] = None,
    log_name_prefix: Optional[str] = None,  # Meaningful name for log files
    logger: Optional[logging.Logger] = None,
    # Stall detection for early termination
    stall_detection_config: Optional[Dict[str, Any]] = None,
    # Centreline-locked boreholes (pre-computed, constant across all passes)
    centreline_boreholes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    """
    Compute optimal borehole locations to cover uncovered gaps.

    .. note::
        **Recommended API:** Use ``optimize_boreholes(gaps, config)`` with a
        ``SolverConfig`` object for cleaner, type-safe configuration.
        This 31-parameter function is maintained for backwards compatibility
        and internal use.

    Main entry point for borehole optimization with automatic solver fallback.
    Supports zone decomposition, connected components, and single solve strategies.

    Args:
        uncovered_gaps: Shapely geometry (Polygon/MultiPolygon) of uncovered areas
        max_spacing: EC7 maximum borehole spacing in meters (default 200)
        ... (31 parameters total)

    Returns:
        Tuple of (borehole_locations, optimization_stats)

    See Also:
        optimize_boreholes: Recommended API using SolverConfig
        create_default_config: Factory for creating SolverConfig
    """
    # Resolve solver mode
    if solver_mode is not None:
        effective_use_ilp, solver_reason = resolve_solver_mode(
            solver_mode=solver_mode, is_parallel_context=is_parallel_context
        )
    else:
        effective_use_ilp = use_ilp
        solver_reason = f"{'ILP' if use_ilp else 'Greedy'} (legacy use_ilp={use_ilp})"

    if logger:
        logger.info("🚀 Starting borehole placement optimization...")
        logger.info(f"   Solver: {solver_reason}")
        logger.info(
            f"   Config: spacing={candidate_spacing}m, gap={mip_gap*100:.0f}%, target={coverage_target_pct}%"
        )

    opt_start = time.perf_counter()

    # Handle empty gaps
    if uncovered_gaps is None or uncovered_gaps.is_empty:
        if logger:
            logger.info("✅ No uncovered gaps - no boreholes needed")
        return [], {"method": "none", "boreholes": 0, "message": "No gaps to cover"}

    gap_polys = _normalize_gaps(uncovered_gaps)
    naive_count = len(gap_polys)

    if logger:
        total_area = sum(g.area for g in gap_polys)
        logger.info(
            f"   📊 {len(gap_polys)} gaps, total area: {total_area/10000:.2f} ha"
        )

    # Bundle shared parameters for helper functions
    shared_params = {
        "max_spacing": max_spacing,
        "candidate_spacing": candidate_spacing,
        "test_spacing": test_spacing,
        "time_limit": time_limit,
        "mip_gap": mip_gap,
        "coverage_target_pct": coverage_target_pct,
        "threads": threads,
        "candidate_grid_type": candidate_grid_type,
        "hexagonal_density": hexagonal_density,
        "use_conflict_constraints": use_conflict_constraints,
        "conflict_constraint_mode": conflict_constraint_mode,
        "exclusion_factor": exclusion_factor,
        "max_conflict_pairs": max_conflict_pairs,
        "min_clique_size": min_clique_size,
        "max_cliques": max_cliques,
        "greedy_max_iterations": greedy_max_iterations,
        "greedy_min_gain": greedy_min_gain,
        "greedy_min_efficiency_pct": greedy_min_efficiency_pct,
        "verbose": verbose,
        "mip_heuristic_effort": mip_heuristic_effort,
        "highs_log_folder": highs_log_folder,
        "log_name_prefix": log_name_prefix,
        "logger": logger,
        "stall_detection_config": stall_detection_config,
        "centreline_boreholes": centreline_boreholes,
    }

    # ZONE-BASED DECOMPOSITION
    if use_zone_decomposition and zones_gdf is not None and effective_use_ilp:
        result = _run_zone_decomposition(
            uncovered_gaps=uncovered_gaps,
            gap_polys=gap_polys,
            zones_gdf=zones_gdf,
            naive_count=naive_count,
            solver_reason=solver_reason,
            use_ilp=use_ilp,
            solver_mode=solver_mode,
            is_parallel_context=is_parallel_context,
            fill_remaining_fragments=fill_remaining_fragments,
            use_connected_components=use_connected_components,
            min_component_gaps_for_ilp=min_component_gaps_for_ilp,
            min_zone_gap_area_m2=min_zone_gap_area_m2,
            zone_cache_dir=zone_cache_dir,  # Intra-run caching
            **shared_params,
        )
        if result is not None:
            return result

    # CONNECTED COMPONENTS DECOMPOSITION
    if use_connected_components and effective_use_ilp and len(gap_polys) > 1:
        components = _decompose_into_components(gap_polys, max_spacing, logger)
        if len(components) > 1:
            return _run_connected_components(
                components=components,
                naive_count=naive_count,
                solver_reason=solver_reason,
                **shared_params,
            )

    # SINGLE SOLVE PATH
    return _run_single_solve(
        gap_polys=gap_polys,
        uncovered_gaps=uncovered_gaps,
        naive_count=naive_count,
        effective_use_ilp=effective_use_ilp,
        solver_reason=solver_reason,
        opt_start=opt_start,
        fill_remaining_fragments=fill_remaining_fragments,
        **shared_params,
    )


# ===========================================================================
# 🏗️ ZONE-BASED DECOMPOSITION
# ===========================================================================


def _filter_centreline_for_zone(
    centreline_boreholes: Optional[List[Dict[str, Any]]],
    zone_gap_union: BaseGeometry,
    zone_spacing: float,
) -> List[Dict[str, Any]]:
    """
    Filter centreline boreholes relevant to a specific zone.

    Keeps boreholes within zone_spacing distance of the zone's gap areas
    so they can pre-cover test points in this zone.

    Args:
        centreline_boreholes: All centreline boreholes (may be None)
        zone_gap_union: Union of gap polygons for this zone
        zone_spacing: Zone max_spacing_m (coverage radius)

    Returns:
        Filtered list of centreline boreholes for this zone.
    """
    if not centreline_boreholes:
        return []

    buffer = zone_gap_union.buffer(zone_spacing)
    return [
        bh for bh in centreline_boreholes if buffer.contains(Point(bh["x"], bh["y"]))
    ]


def _deduplicate_boreholes(
    boreholes: List[Dict[str, Any]],
    tolerance: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Remove duplicate boreholes by position within tolerance.

    Used after zone decomposition to remove centreline BHs that were
    assigned to multiple zones due to proximity to zone boundaries.

    Args:
        boreholes: List of borehole dicts with x, y keys
        tolerance: Distance in metres to consider as duplicate

    Returns:
        Deduplicated list preserving order and first occurrence.
    """
    tol_sq = tolerance * tolerance
    seen: List[Tuple[float, float]] = []
    result: List[Dict[str, Any]] = []

    for bh in boreholes:
        bx, by = bh["x"], bh["y"]
        is_dup = False
        for sx, sy in seen:
            dx = bx - sx
            dy = by - sy
            if dx * dx + dy * dy < tol_sq:
                is_dup = True
                break
        if not is_dup:
            seen.append((bx, by))
            result.append(bh)

    return result


def _run_zone_decomposition(
    uncovered_gaps: Optional[BaseGeometry],
    gap_polys: List[Polygon],
    zones_gdf: Optional[gpd.GeoDataFrame],
    naive_count: int,
    solver_reason: str,
    min_zone_gap_area_m2: float,
    zone_cache_dir: Optional[str],  # Intra-run caching directory
    logger: Optional[logging.Logger],
    **kwargs,
) -> Optional[Tuple[List[Dict[str, float]], Dict[str, Any]]]:
    """
    Run zone-based decomposition strategy with per-zone spacing.

    Uses zone-specific max_spacing_m when available in zones_gdf, applying
    global multipliers (candidate_spacing_mult, test_spacing_mult) to each
    zone's spacing value.

    When zone_cache_dir is provided, uses intra-run caching to avoid redundant
    ILP solves for identical zone-gap geometries across filter combinations.

    Returns None if zones reduce to single zone (fall through to next strategy).
    """
    from Gap_Analysis_EC7.coverage_zones import split_gaps_by_zones_with_spacing
    from Gap_Analysis_EC7.config import CONFIG

    # Initialize zone cache if directory provided
    zone_cache = None
    if zone_cache_dir:
        try:
            from Gap_Analysis_EC7.parallel.zone_cache import get_zone_cache_from_path

            zone_cache = get_zone_cache_from_path(zone_cache_dir)
        except ImportError:
            if logger:
                logger.warning("Zone cache module not available, caching disabled")

    # Use the extended function that returns per-zone spacing
    zone_data = split_gaps_by_zones_with_spacing(
        uncovered_gaps=uncovered_gaps,
        zones_gdf=zones_gdf,
        min_area_m2=min_zone_gap_area_m2,
        logger=logger,
    )

    if len(zone_data) > 1:
        # Multiple zones - solve each independently with zone-specific spacing
        if logger:
            logger.info(f"   🏗️ Solving {len(zone_data)} zones independently...")

        all_boreholes: List[Dict[str, float]] = []
        all_stats: Dict[str, Any] = {
            "method": "zone_decomposition",
            "zones": {},
            "solver_reason": solver_reason,
        }
        zone_start = time.perf_counter()

        # Get multipliers from config
        ilp_config = CONFIG.get("ilp_solver", {})
        candidate_mult = ilp_config.get("candidate_spacing_mult", 0.5)
        test_mult = ilp_config.get("test_spacing_mult", 0.2)

        for zone_name, data in zone_data.items():
            zone_polys = data["gaps"]
            zone_spacing = data["max_spacing_m"]
            zone_area = sum(g.area for g in zone_polys) / 10000

            # Compute zone-specific candidate and test spacing
            zone_candidate_spacing = zone_spacing * candidate_mult
            zone_test_spacing = zone_spacing * test_mult

            if logger:
                logger.info(
                    f"   📦 Zone '{zone_name}': {len(zone_polys)} gaps, "
                    f"{zone_area:.1f} ha, max={zone_spacing}m, "
                    f"cand={zone_candidate_spacing:.1f}m, test={zone_test_spacing:.1f}m"
                )

            zone_gap_union = unary_union(zone_polys)

            # Override spacing parameters for this zone
            zone_kwargs = kwargs.copy()
            zone_kwargs["max_spacing"] = zone_spacing
            zone_kwargs["candidate_spacing"] = zone_candidate_spacing
            zone_kwargs["test_spacing"] = zone_test_spacing
            # Use meaningful zone name for log files (without "pass" in prefix)
            zone_kwargs["log_name_prefix"] = f"first_{zone_name}"

            # Filter centreline boreholes for this zone (spatial containment)
            all_cl_bhs = zone_kwargs.pop("centreline_boreholes", None)
            zone_cl_bhs = _filter_centreline_for_zone(
                all_cl_bhs, zone_gap_union, zone_spacing
            )
            zone_kwargs["centreline_boreholes"] = zone_cl_bhs if zone_cl_bhs else None

            # Solve with or without caching
            if zone_cache is not None:
                # Cache-aware solve: use get_or_compute pattern
                gap_wkt = zone_gap_union.wkt

                def solve_zone():
                    """Compute function for cache miss."""
                    bh, stats = compute_optimal_boreholes(
                        uncovered_gaps=zone_gap_union,
                        use_zone_decomposition=False,
                        zones_gdf=None,
                        zone_cache_dir=None,  # Don't recurse caching
                        logger=None,  # Quiet per-zone
                        **zone_kwargs,
                    )
                    return {"boreholes": bh, "stats": stats}

                cached_result = zone_cache.get_or_compute(
                    zone_name=zone_name,
                    gap_wkt=gap_wkt,
                    compute_fn=solve_zone,
                )
                zone_boreholes = cached_result["boreholes"]
                zone_stats = cached_result["stats"]
            else:
                # Direct solve (no caching)
                zone_boreholes, zone_stats = compute_optimal_boreholes(
                    uncovered_gaps=zone_gap_union,
                    use_zone_decomposition=False,
                    zones_gdf=None,
                    zone_cache_dir=None,  # Don't recurse caching
                    logger=None,  # Quiet per-zone
                    **zone_kwargs,
                )

            all_boreholes.extend(zone_boreholes)
            all_stats["zones"][zone_name] = {
                "boreholes": len(zone_boreholes),
                "gaps": len(zone_polys),
                "area_ha": zone_area,
                "max_spacing_m": zone_spacing,
                "stats": zone_stats,
            }

            if logger:
                logger.info(f"      ✅ {len(zone_boreholes)} boreholes placed")

        # NOTE: Border consolidation now runs inside parallel workers
        # (see Gap_Analysis_EC7/parallel/coverage_worker.py)

        # Deduplicate centreline boreholes that may appear in multiple zones
        # (zone buffering can include BHs near zone boundaries in both zones)
        pre_dedup_count = len(all_boreholes)
        all_boreholes = _deduplicate_boreholes(all_boreholes)
        if logger and pre_dedup_count != len(all_boreholes):
            logger.info(
                f"   🔄 Deduped {pre_dedup_count - len(all_boreholes)} "
                f"cross-zone centreline boreholes"
            )

        # === ENSURE ALL CENTRELINE BOREHOLES ARE INCLUDED ===
        # _filter_centreline_for_zone() drops boreholes not near any zone's
        # gaps.  These are locked constants and must always appear in output.
        all_cl_bhs = kwargs.get("centreline_boreholes")
        if all_cl_bhs:
            existing_positions = {
                (round(bh["x"], 1), round(bh["y"], 1)) for bh in all_boreholes
            }
            missing_cl = [
                bh
                for bh in all_cl_bhs
                if (round(bh["x"], 1), round(bh["y"], 1)) not in existing_positions
            ]
            if missing_cl:
                all_boreholes.extend(missing_cl)
                if logger:
                    logger.info(
                        f"   🛤️ Restored {len(missing_cl)} centreline boreholes "
                        f"not near any zone gap (total: {len(all_boreholes)})"
                    )

        # Aggregate test points from all zones for consolidation pass
        # Each test point already has required_radius; add zone name for debugging
        all_test_points = []
        for zone_name, zone_info in all_stats["zones"].items():
            zone_test_pts = zone_info.get("stats", {}).get("test_points", [])
            for tp in zone_test_pts:
                tp["zone"] = zone_name  # Tag with origin zone for debugging
            all_test_points.extend(zone_test_pts)
        all_stats["test_points"] = all_test_points

        # Aggregate stats
        zone_time = time.perf_counter() - zone_start
        all_stats["total_boreholes"] = len(all_boreholes)
        all_stats["n_zones"] = len(zone_data)
        all_stats["naive_count"] = naive_count
        all_stats["optimal_count"] = len(all_boreholes)
        all_stats["improvement_percent"] = (
            ((naive_count - len(all_boreholes)) / naive_count * 100)
            if naive_count > 0
            else 0
        )
        all_stats["total_time"] = zone_time
        all_stats["timing"] = {"zone_decomposition_total": zone_time}

        if logger:
            improvement = all_stats["improvement_percent"]
            logger.info(
                f"✅ Zone decomposition complete: {len(all_boreholes)} boreholes "
                f"across {len(zone_data)} zones "
                f"(vs {naive_count} naive, {improvement:.1f}% improvement) "
                f"in {zone_time:.1f}s"
            )

        return all_boreholes, all_stats

    elif len(zone_data) == 1:
        # Single zone - fall through
        if logger:
            zone_name = list(zone_data.keys())[0]
            logger.info(f"   ℹ️ Single zone '{zone_name}' - using standard path")
        return None

    else:
        # No valid zone gaps
        if logger:
            logger.info("   ℹ️ No valid zone gaps after splitting")
        return [], {"method": "zone_decomposition", "message": "No valid gaps"}


# ===========================================================================
# 🧩 CONNECTED COMPONENTS DECOMPOSITION
# ===========================================================================


def _run_connected_components(
    components: List[List[Polygon]],
    naive_count: int,
    solver_reason: str,
    logger: Optional[logging.Logger],
    **kwargs,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    """Run connected components decomposition strategy."""
    if logger:
        logger.info(f"   🧩 Solving {len(components)} independent components...")

    all_boreholes: List[Dict[str, float]] = []
    all_stats: Dict[str, Any] = {
        "method": "connected_components",
        "components": [],
        "solver_reason": solver_reason,
    }
    decomp_start = time.perf_counter()

    for comp_idx, component_gaps in enumerate(components):
        comp_area = sum(g.area for g in component_gaps) / 10000
        if logger:
            logger.info(
                f"   📦 Component {comp_idx+1}/{len(components)}: "
                f"{len(component_gaps)} gaps, {comp_area:.1f} ha"
            )

        comp_boreholes, comp_stats = _solve_component(
            component_gaps=component_gaps,
            component_idx=comp_idx,
            logger=None,  # Quiet per-component
            **kwargs,
        )

        all_boreholes.extend(comp_boreholes)
        all_stats["components"].append(comp_stats)

        if logger:
            logger.info(f"      ✅ {len(comp_boreholes)} boreholes placed")

    # Aggregate stats
    decomp_time = time.perf_counter() - decomp_start
    all_stats["total_boreholes"] = len(all_boreholes)
    all_stats["n_components"] = len(components)
    all_stats["naive_count"] = naive_count
    all_stats["optimal_count"] = len(all_boreholes)
    all_stats["improvement_percent"] = (
        ((naive_count - len(all_boreholes)) / naive_count * 100)
        if naive_count > 0
        else 0
    )
    all_stats["total_time"] = decomp_time
    all_stats["timing"] = {"connected_components_total": decomp_time}

    if logger:
        improvement = all_stats["improvement_percent"]
        logger.info(
            f"✅ Connected components complete: {len(all_boreholes)} boreholes "
            f"(vs {naive_count} naive, {improvement:.1f}% improvement) "
            f"in {decomp_time:.1f}s"
        )

    return all_boreholes, all_stats


# ===========================================================================
# 🎯 SINGLE SOLVE PATH
# ===========================================================================


def _run_single_solve(
    gap_polys: List[Polygon],
    uncovered_gaps: Any,
    naive_count: int,
    effective_use_ilp: bool,
    solver_reason: str,
    opt_start: float,
    max_spacing: float,
    candidate_spacing: float,
    test_spacing: float,
    time_limit: int,
    mip_gap: float,
    coverage_target_pct: float,
    threads: int,
    candidate_grid_type: str,
    hexagonal_density: float,
    use_conflict_constraints: bool,
    conflict_constraint_mode: str,
    exclusion_factor: float,
    max_conflict_pairs: int,
    min_clique_size: int,
    max_cliques: int,
    greedy_max_iterations: int,
    greedy_min_gain: float,
    greedy_min_efficiency_pct: float,
    fill_remaining_fragments: bool,
    verbose: int,
    mip_heuristic_effort: float,
    highs_log_folder: Optional[str],
    log_name_prefix: Optional[str],
    logger: Optional[logging.Logger],
    stall_detection_config: Optional[Dict[str, Any]] = None,
    centreline_boreholes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    """Run single solve without decomposition."""
    opt_times = {}

    # Generate candidates and test points
    step_start = time.perf_counter()
    candidates = _generate_candidate_grid(
        gap_polys,
        max_spacing,
        candidate_spacing,
        grid_type=candidate_grid_type,
        hexagonal_density=hexagonal_density,
        logger=logger,
    )
    opt_times["candidate_grid"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    test_points = _generate_test_points(gap_polys, test_spacing, logger)
    all_test_points_count = len(test_points)  # Track total before pre-coverage
    opt_times["test_points"] = time.perf_counter() - step_start

    # === CENTRELINE PRE-COVERAGE ===
    # Remove test points already covered by locked centreline boreholes
    # BEFORE building coverage dict (keeps ILP indices consistent)
    if centreline_boreholes:
        step_start = time.perf_counter()
        from Gap_Analysis_EC7.centreline_constraints import (
            compute_centreline_precoverage,
        )

        tp_dicts = [
            {"x": tp.x, "y": tp.y, "required_radius": max_spacing} for tp in test_points
        ]
        pre_covered_indices = compute_centreline_precoverage(
            tp_dicts, centreline_boreholes, log=logger
        )
        if pre_covered_indices:
            test_points = [
                tp for i, tp in enumerate(test_points) if i not in pre_covered_indices
            ]
        opt_times["centreline_precoverage"] = time.perf_counter() - step_start

    if len(test_points) == 0:
        # All test points covered by centreline boreholes → return them directly
        if centreline_boreholes:
            if logger:
                logger.info(
                    f"   🛤️ All {all_test_points_count} test points pre-covered by "
                    f"{len(centreline_boreholes)} centreline boreholes"
                )
            return list(centreline_boreholes), {
                "method": "centreline_precoverage",
                "boreholes": len(centreline_boreholes),
                "centreline_count": len(centreline_boreholes),
                "centreline_precovered": all_test_points_count,
                "test_points": [],
            }
        if logger:
            logger.warning("⚠️ No test points generated - gaps may be too small")
        return [], {"method": "error", "message": "Gaps too small for test points"}

    # Build coverage mapping
    step_start = time.perf_counter()
    coverage_dict = _build_coverage_dict(test_points, candidates, max_spacing, logger)
    opt_times["coverage_dict"] = time.perf_counter() - step_start

    # Solve optimization problem
    step_start = time.perf_counter()
    if effective_use_ilp:
        # Generate HiGHS log file path if folder provided
        firstpass_log_file = None
        if highs_log_folder:
            import os
            import uuid

            # Use meaningful prefix if provided, otherwise fall back to UUID
            if log_name_prefix:
                log_name = _sanitize_log_name(log_name_prefix)
            else:
                # Use short uuid as fallback (handles multiple solves)
                log_name = f"first_{str(uuid.uuid4())[:8]}"
            firstpass_log_file = os.path.join(highs_log_folder, f"{log_name}.log")

        selected_indices, stats = _solve_ilp(
            test_points,
            candidates,
            coverage_dict,
            time_limit,
            mip_gap=mip_gap,
            threads=threads,
            coverage_target_pct=coverage_target_pct,
            use_conflict_constraints=use_conflict_constraints,
            conflict_constraint_mode=conflict_constraint_mode,
            exclusion_factor=exclusion_factor,
            max_spacing=max_spacing,
            max_conflict_pairs=max_conflict_pairs,
            min_clique_size=min_clique_size,
            max_cliques=max_cliques,
            verbose=verbose,
            mip_heuristic_effort=mip_heuristic_effort,
            logger=logger,
            highs_log_file=firstpass_log_file,
            stall_detection_config=stall_detection_config,
        )
        if selected_indices is None:
            if logger:
                logger.info("   ⚠️ ILP failed, falling back to greedy...")
            selected_indices, stats = _solve_greedy(
                gap_polys,
                candidates,
                max_spacing,
                max_iterations=greedy_max_iterations,
                min_coverage_gain=greedy_min_gain,
                min_efficiency_pct=greedy_min_efficiency_pct,
                logger=logger,
            )
            stats["method"] = "ilp_fallback_greedy"
    else:
        selected_indices, stats = _solve_greedy(
            gap_polys,
            candidates,
            max_spacing,
            max_iterations=greedy_max_iterations,
            min_coverage_gain=greedy_min_gain,
            min_efficiency_pct=greedy_min_efficiency_pct,
            logger=logger,
        )
    opt_times["solver"] = time.perf_counter() - step_start

    stats["solver_reason"] = solver_reason

    # Convert to output format via Borehole dataclass for type-safe provenance
    boreholes = [
        Borehole(
            x=candidates[i].x,
            y=candidates[i].y,
            coverage_radius=max_spacing,
            source_pass=BoreholePass.FIRST,
            status=BoreholeStatus.PROPOSED,
        ).as_dict()
        for i in selected_indices
    ]

    # === APPEND CENTRELINE BOREHOLES ===
    # Centreline boreholes are locked constants - always included in output
    if centreline_boreholes:
        boreholes.extend(centreline_boreholes)
        if logger:
            logger.info(
                f"   🛤️ Added {len(centreline_boreholes)} centreline boreholes "
                f"(total: {len(boreholes)})"
            )

    # Optional: ensure complete coverage
    step_start = time.perf_counter()
    if fill_remaining_fragments:
        boreholes = _ensure_complete_coverage(
            uncovered_gaps, boreholes, max_spacing, logger
        )
    else:
        if logger:
            logger.info(f"   ℹ️ Fragment fill disabled - using solver solution only")
    opt_times["ensure_coverage"] = time.perf_counter() - step_start

    # Calculate improvement
    optimal_count = len(boreholes)
    improvement = (
        ((naive_count - optimal_count) / naive_count * 100) if naive_count > 0 else 0
    )

    stats["naive_count"] = naive_count
    stats["optimal_count"] = optimal_count
    stats["improvement_percent"] = improvement

    # Track centreline stats for downstream visibility
    if centreline_boreholes:
        stats["centreline_count"] = len(centreline_boreholes)
        stats["centreline_precovered"] = all_test_points_count - len(test_points)

    # Store test points for consolidation pass (convert Point to dict for serialization)
    # Include required_radius for buffer zone consolidation with variable zone spacing
    stats["test_points"] = [
        {"x": tp.x, "y": tp.y, "required_radius": max_spacing} for tp in test_points
    ]

    # Store candidates for consolidation pass (unified candidate pool)
    stats["candidates"] = [{"x": c.x, "y": c.y} for c in candidates]

    # Add timing to stats
    total_opt_time = time.perf_counter() - opt_start
    stats["timing"] = opt_times
    stats["total_time"] = total_opt_time

    if logger:
        logger.info(
            f"✅ Optimization complete: {optimal_count} boreholes "
            f"(vs {naive_count} naive, {improvement:.1f}% improvement)"
        )
        logger.info(f"   ⏱️ Optimizer Sub-step Timing:")
        for step_name, duration in opt_times.items():
            pct = (duration / total_opt_time) * 100 if total_opt_time > 0 else 0
            logger.info(f"      {step_name}: {duration:.2f}s ({pct:.1f}%)")
        logger.info(f"      Optimizer Total: {total_opt_time:.2f}s")

    return boreholes, stats


# ===========================================================================
# 🧩 COMPONENT SOLVER
# ===========================================================================


def _solve_component(
    component_gaps: List[Polygon],
    max_spacing: float,
    candidate_spacing: float,
    test_spacing: float,
    time_limit: int,
    mip_gap: float,
    coverage_target_pct: float,
    threads: int,
    candidate_grid_type: str,
    hexagonal_density: float,
    use_conflict_constraints: bool,
    conflict_constraint_mode: str,
    exclusion_factor: float,
    max_conflict_pairs: int,
    min_clique_size: int,
    max_cliques: int,
    greedy_max_iterations: int,
    greedy_min_gain: float,
    greedy_min_efficiency_pct: float,
    verbose: int,
    mip_heuristic_effort: float = 0.05,
    highs_log_folder: Optional[str] = None,
    component_idx: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    # Extra params forwarded from shared_params — not used in component solve
    log_name_prefix: Optional[str] = None,
    stall_detection_config: Optional[Dict[str, Any]] = None,
    centreline_boreholes: Optional[List[Dict[str, Any]]] = None,
    **_extra,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    """
    Solve ILP for a single connected component.

    This is a thin wrapper that handles a subset of gaps as an
    independent optimization subproblem.

    Args:
        component_gaps: List of gap Polygons in this component
        [other params same as compute_optimal_boreholes]

    Returns:
        Tuple of (borehole_locations, component_stats)
    """
    # Generate candidates for this component only
    candidates = _generate_candidate_grid(
        component_gaps,
        max_spacing,
        candidate_spacing,
        grid_type=candidate_grid_type,
        hexagonal_density=hexagonal_density,
        logger=None,
    )

    if not candidates:
        return [], {"method": "empty", "message": "No candidates for component"}

    # Generate test points
    test_points = _generate_test_points(component_gaps, test_spacing, logger=None)

    if not test_points:
        return [], {"method": "empty", "message": "No test points for component"}

    # Build coverage dict
    coverage_dict = _build_coverage_dict(
        test_points, candidates, max_spacing, logger=None
    )

    # Scale time limit by component size
    component_area = sum(g.area for g in component_gaps)
    area_factor = min(2.0, max(0.2, component_area / 500000))
    component_time_limit = max(10, int(time_limit * area_factor))

    # Scale max conflict pairs proportionally
    scaled_max_conflict_pairs = max(10000, max_conflict_pairs // 4)

    # Generate HiGHS log file path if folder provided
    component_log_file = None
    if highs_log_folder:
        import os
        import uuid

        # Use meaningful component index if available, otherwise fall back to UUID
        if component_idx is not None:
            log_name = f"component_{component_idx + 1:02d}"
        else:
            log_name = f"component_{str(uuid.uuid4())[:8]}"
        component_log_file = os.path.join(highs_log_folder, f"{log_name}.log")

    # Solve ILP for this component
    selected_indices, stats = _solve_ilp(
        test_points,
        candidates,
        coverage_dict,
        component_time_limit,
        mip_gap=mip_gap,
        threads=threads,
        coverage_target_pct=coverage_target_pct,
        use_conflict_constraints=use_conflict_constraints,
        conflict_constraint_mode=conflict_constraint_mode,
        exclusion_factor=exclusion_factor,
        max_spacing=max_spacing,
        max_conflict_pairs=scaled_max_conflict_pairs,
        min_clique_size=min_clique_size,
        max_cliques=max_cliques,
        verbose=0,
        mip_heuristic_effort=mip_heuristic_effort,
        logger=None,
        highs_log_file=component_log_file,
    )

    if selected_indices is None:
        # ILP failed, fall back to greedy
        selected_indices, stats = _solve_greedy(
            component_gaps,
            candidates,
            max_spacing,
            max_iterations=greedy_max_iterations,
            min_coverage_gain=greedy_min_gain,
            min_efficiency_pct=greedy_min_efficiency_pct,
            logger=None,
        )
        stats["method"] = "component_greedy_fallback"

    # Convert to output format via Borehole dataclass for type-safe provenance
    boreholes = [
        Borehole(
            x=candidates[i].x,
            y=candidates[i].y,
            coverage_radius=max_spacing,
            source_pass=BoreholePass.FIRST,
            status=BoreholeStatus.PROPOSED,
        ).as_dict()
        for i in selected_indices
    ]

    # Add component-specific stats
    stats["component_gaps"] = len(component_gaps)
    stats["component_candidates"] = len(candidates)
    stats["component_test_points"] = len(test_points)
    stats["component_time_limit"] = component_time_limit
    stats["component_area_ha"] = component_area / 10000

    # Store test points for consolidation pass
    # Include required_radius for buffer zone consolidation with variable zone spacing
    stats["test_points"] = [
        {"x": tp.x, "y": tp.y, "required_radius": max_spacing} for tp in test_points
    ]

    # Store candidates for consolidation pass (unified candidate pool)
    stats["candidates"] = [{"x": c.x, "y": c.y} for c in candidates]

    return boreholes, stats


# ===========================================================================
# 🔧 COVERAGE REFINEMENT SECTION
# ===========================================================================


def _ensure_complete_coverage(
    uncovered_gaps: Any,
    boreholes: List[Dict[str, float]],
    radius: float,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, float]]:
    """
    Ensure complete coverage by adding boreholes for any remaining gaps.

    After ILP/greedy optimization, there may be small remaining uncovered areas
    due to discretization. This function adds boreholes at representative points
    of any remaining gap fragments.

    Args:
        uncovered_gaps: Original Shapely geometry of gaps
        boreholes: Current list of borehole locations
        radius: Coverage radius per borehole
        logger: Optional logger

    Returns:
        Updated list of borehole locations with complete coverage
    """
    if not boreholes:
        gap_polys = _normalize_gaps(uncovered_gaps)
        return [
            Borehole(
                x=g.representative_point().x,
                y=g.representative_point().y,
                coverage_radius=radius,
                source_pass=BoreholePass.FIRST,
                status=BoreholeStatus.PROPOSED,
            ).as_dict()
            for g in gap_polys
        ]

    # Calculate remaining uncovered area
    coverage_union = unary_union(
        [Point(*get_bh_coords(bh)).buffer(radius) for bh in boreholes]
    )
    remaining = uncovered_gaps.difference(coverage_union)

    if remaining.is_empty or remaining.area < 1.0:
        return boreholes

    # Add boreholes for remaining fragments
    result = list(boreholes)
    remaining_polys = _normalize_gaps(remaining)

    for frag in remaining_polys:
        rep_pt = frag.representative_point()
        result.append(
            Borehole(
                x=rep_pt.x,
                y=rep_pt.y,
                coverage_radius=radius,
                source_pass=BoreholePass.FIRST,
                status=BoreholeStatus.PROPOSED,
            ).as_dict()
        )

    if logger and len(remaining_polys) > 0:
        logger.info(
            f"   🔧 Added {len(remaining_polys)} boreholes for remaining fragments"
        )

    return result


# ===========================================================================
# ✅ VERIFICATION SECTION
# ===========================================================================


def verify_coverage(
    uncovered_gaps: Any,
    boreholes: List[Dict[str, float]],
    radius: float = 200.0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Verify that selected boreholes achieve complete coverage of gaps.

    Args:
        uncovered_gaps: Shapely geometry of original gaps
        boreholes: List of {"x": float, "y": float} borehole locations
        radius: Coverage radius per borehole
        logger: Optional logger

    Returns:
        Dict with verification results
    """
    if uncovered_gaps is None or uncovered_gaps.is_empty:
        return {
            "original_gap_area": 0,
            "remaining_uncovered": 0,
            "coverage_percentage": 100.0,
            "is_complete": True,
        }

    original_area = uncovered_gaps.area

    if not boreholes:
        remaining_area = original_area
    else:
        coverage_union = unary_union(
            [Point(*get_bh_coords(bh)).buffer(radius) for bh in boreholes]
        )
        remaining = uncovered_gaps.difference(coverage_union)
        remaining_area = remaining.area if not remaining.is_empty else 0

    covered_area = original_area - remaining_area
    coverage_pct = (covered_area / original_area * 100) if original_area > 0 else 100.0

    is_complete = coverage_pct >= 99.0 or remaining_area < 100.0

    result = {
        "original_gap_area": original_area,
        "remaining_uncovered": remaining_area,
        "coverage_percentage": coverage_pct,
        "is_complete": is_complete,
    }

    if logger:
        status = "✅ Complete" if result["is_complete"] else "⚠️ Incomplete"
        logger.info(
            f"   {status}: {coverage_pct:.1f}% coverage "
            f"({remaining_area:.0f} m² remaining)"
        )

    return result


# ===========================================================================
# 🔧 UTILITY FUNCTIONS
# ===========================================================================


def compare_approaches(
    uncovered_gaps: Any,
    max_spacing: float = 200.0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Compare naive vs optimal borehole placement approaches.

    Useful for demonstrating optimization improvement.

    Args:
        uncovered_gaps: Shapely geometry of uncovered gaps
        max_spacing: Coverage radius per borehole
        logger: Optional logger

    Returns:
        Dict with comparison results
    """
    gap_polys = _normalize_gaps(uncovered_gaps)
    naive_count = len(gap_polys)

    optimal_boreholes, stats = compute_optimal_boreholes(
        uncovered_gaps, max_spacing, logger=logger
    )
    optimal_count = len(optimal_boreholes)

    improvement = (
        ((naive_count - optimal_count) / naive_count * 100) if naive_count > 0 else 0
    )

    return {
        "naive_count": naive_count,
        "optimal_count": optimal_count,
        "improvement_percent": improvement,
        "boreholes_saved": naive_count - optimal_count,
        "optimal_locations": optimal_boreholes,
        "optimization_stats": stats,
    }
