"""
Worker function for processing single filter combination.

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Process one (depth, spt, txt, txe) filter combination.
THIN WRAPPER pattern - calls existing functions from:
- coverage_zones.py: compute_coverage_zones()
- borehole_optimizer.py: compute_optimal_boreholes()

Returns serializable dict with coverage data for embedding in HTML.

Follows Gap_Analysis parallel_worker.py patterns:
- Accept only primitive/serializable parameters
- Return dict with success/error status
- No business logic duplication
- Silent logging (no progress output to avoid interleaving)

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

import logging
import time
from typing import Dict, Any, List, Set, Optional, Tuple, Union

import math

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from Gap_Analysis_EC7.parallel.coverage_orchestrator import (
    deserialize_geodataframe,
    serialize_geometry,
)


def _safe_depth(value: Any) -> float:
    """
    Convert depth value to float, treating None/NaN as 0 but preserving actual 0.

    This matches JavaScript's `parseFloat(val) || 0` behavior for consistency
    between Python coverage computation and JavaScript marker visibility.

    Args:
        value: Depth value (float, int, None, or NaN)

    Returns:
        float: Cleaned depth value (0 if None/NaN, otherwise the actual value)

    Examples:
        >>> _safe_depth(67.3)
        67.3
        >>> _safe_depth(0)
        0.0
        >>> _safe_depth(None)
        0.0
        >>> _safe_depth(float('nan'))
        0.0
    """
    if value is None:
        return 0.0
    try:
        f = float(value)
        # NaN check: nan != nan is True
        if math.isnan(f):
            return 0.0
        return f
    except (ValueError, TypeError):
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 🔧 WORKER LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════


def _setup_worker_logging(combo_key: str) -> logging.Logger:
    """
    Configure logging for this worker process.

    Creates a named logger for this specific combination to allow
    identification of log messages from parallel workers.

    Args:
        combo_key: Unique key for this combination (used in logger name)

    Returns:
        Logger instance for this worker
    """
    logger = logging.getLogger(f"EC7.Worker.{combo_key}")
    logger.setLevel(logging.INFO)
    return logger


# ═══════════════════════════════════════════════════════════════════════════
# 🔍 BOREHOLE FILTERING
# ═══════════════════════════════════════════════════════════════════════════


def filter_boreholes_by_criteria(
    boreholes_records: List[Dict],
    min_depth: float,
    require_spt: bool,
    require_triaxial_total: bool,
    require_triaxial_effective: bool,
    spt_locations: Set[str],
    triaxial_total_locations: Set[str],
    triaxial_effective_locations: Set[str],
) -> List[Dict]:
    """
    Filter borehole records based on filter criteria.

    Applies the same filtering logic as the JavaScript client-side filter,
    but on the serialized borehole records.

    Args:
        boreholes_records: Serialized borehole records (list of dicts)
        min_depth: Minimum depth threshold (>= this value passes)
        require_spt: If True, only include boreholes with SPT data
        require_triaxial_total: If True, only include boreholes with TxT data
        require_triaxial_effective: If True, only include boreholes with TxE data
        spt_locations: Set of Location IDs that have SPT data
        triaxial_total_locations: Set of Location IDs that have TxT data
        triaxial_effective_locations: Set of Location IDs that have TxE data

    Returns:
        Filtered list of borehole records that pass all criteria
    """
    filtered = []
    for record in boreholes_records:
        # Check depth threshold using safe conversion (handles NaN correctly)
        # NaN depths are treated as 0 to match JavaScript marker visibility behavior
        depth = _safe_depth(record.get("Final Depth"))
        if depth < min_depth:
            continue

        # Check test data requirements
        location_id = record.get("Location ID", "")
        if require_spt and location_id not in spt_locations:
            continue
        if require_triaxial_total and location_id not in triaxial_total_locations:
            continue
        if (
            require_triaxial_effective
            and location_id not in triaxial_effective_locations
        ):
            continue

        filtered.append(record)

    return filtered


# ═══════════════════════════════════════════════════════════════════════════
# 📊 WORKER RESULT HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _create_empty_result(combo_key: str) -> Dict[str, Any]:
    """
    Create initial result structure with default values.

    Args:
        combo_key: Unique key for this combination

    Returns:
        Dict with initialized result structure
    """
    return {
        "key": combo_key,
        "success": False,
        "boreholes_count": 0,
        "covered": None,
        "gaps": None,
        "proposed": [],
        "stats": {},
        "duration_seconds": 0,
        "error": None,
    }


def _create_empty_filter_result(start_time: float) -> Dict[str, Any]:
    """
    Create stats for empty filter result (no boreholes matched).

    Args:
        start_time: Processing start time for duration calculation

    Returns:
        Dict with empty stats and success=True
    """
    return {
        "success": True,
        "stats": {
            "covered_area_ha": 0,
            "gap_area_ha": 0,
            "gap_count": 0,
            "proposed_count": 0,
        },
        "duration_seconds": time.time() - start_time,
    }


def _compute_proposed_boreholes(
    uncovered_gaps: Optional[BaseGeometry],
    zones_gdf: Optional[gpd.GeoDataFrame],
    max_spacing: float,
    ilp_config: Dict[str, Any],
    highs_log_folder: Optional[str] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    """
    Compute proposed boreholes for gaps using solver.

    Uses the clean optimize_boreholes API with SolverConfig instead of
    manually extracting 25+ parameters from ilp_config.

    Args:
        uncovered_gaps: MultiPolygon of uncovered areas (may be None/empty)
        zones_gdf: Zones GeoDataFrame for zone decomposition
        max_spacing: EC7 maximum spacing in meters
        ilp_config: Full ILP solver configuration dict (from _build_solver_config)
        highs_log_folder: Optional folder path for HiGHS solver log files

    Returns:
        Tuple of (proposed_boreholes_list, optimization_stats_dict)
    """
    if uncovered_gaps is None or uncovered_gaps.is_empty:
        return [], {}

    from Gap_Analysis_EC7.solvers.solver_config import config_from_ilp_dict
    from Gap_Analysis_EC7.solvers.solver_orchestration import optimize_boreholes

    # Create typed config from the CONFIG dict
    config = config_from_ilp_dict(
        ilp_config=ilp_config,
        max_spacing=max_spacing,
        zones_gdf=zones_gdf,
        logger=None,  # Silent in worker
        highs_log_folder=highs_log_folder,
    )

    return optimize_boreholes(uncovered_gaps, config)


def _build_result_stats(
    covered_union: Optional[BaseGeometry],
    uncovered_gaps: Optional[BaseGeometry],
    gap_stats: List[Dict[str, Any]],
    proposed: List[Dict[str, float]],
    optimization_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build statistics dict for worker result.

    Args:
        covered_union: Covered area polygon
        uncovered_gaps: Gap MultiPolygon
        gap_stats: List of gap statistics
        proposed: List of proposed boreholes
        optimization_stats: Stats from solver

    Returns:
        Dict with coverage statistics
    """
    covered_area = covered_union.area / 10000 if covered_union else 0
    gap_area = uncovered_gaps.area / 10000 if uncovered_gaps else 0

    return {
        "covered_area_ha": round(covered_area, 1),
        "gap_area_ha": round(gap_area, 1),
        "gap_count": len(gap_stats),
        "proposed_count": len(proposed),
        "solver_method": optimization_stats.get("method", "none"),
        "solver_reason": optimization_stats.get("solver_reason", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 🏭 MAIN WORKER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════


def worker_process_filter_combination(
    combo_key: str,
    combo: Dict[str, Any],
    boreholes_records: List[Dict],
    boreholes_crs: str,
    zones_records: List[Dict],
    zones_crs: str,
    max_spacing: float,
    ilp_config: Dict[str, Any],
    spt_locations: List[str],
    triaxial_total_locations: List[str],
    triaxial_effective_locations: List[str],
    highs_log_folder: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Worker function to process a single filter combination.

    THIN WRAPPER - calls existing coverage_zones and borehole_optimizer
    functions. Does not duplicate any business logic.

    Args:
        combo_key: Unique key for this combination (e.g., "d25_spt1_txt0_txe0")
        combo: Filter combination dict with keys:
            - min_depth: int
            - require_spt: bool
            - require_triaxial_total: bool
            - require_triaxial_effective: bool
        boreholes_records: Serialized boreholes GeoDataFrame
        boreholes_crs: CRS string for boreholes (e.g., "EPSG:27700")
        zones_records: Serialized zones GeoDataFrame
        zones_crs: CRS string for zones
        max_spacing: EC7 maximum spacing in meters (typically 200m)
        ilp_config: ILP solver configuration dict
        spt_locations: List of Location IDs with SPT data
        triaxial_total_locations: List of Location IDs with TxT data
        triaxial_effective_locations: List of Location IDs with TxE data
        highs_log_folder: Optional folder path to write HiGHS solver logs.

    Returns:
        Dict with coverage data (key, success, boreholes_count, covered, gaps,
        proposed, stats, duration_seconds, error)
    """
    start_time = time.time()
    logger = _setup_worker_logging(combo_key)
    result = _create_empty_result(combo_key)

    try:
        # Convert location lists to sets for O(1) lookup
        spt_set = set(spt_locations)
        txt_set = set(triaxial_total_locations)
        txe_set = set(triaxial_effective_locations)

        # Filter boreholes by criteria
        filtered_records = filter_boreholes_by_criteria(
            boreholes_records,
            combo["min_depth"],
            combo["require_spt"],
            combo["require_triaxial_total"],
            combo["require_triaxial_effective"],
            spt_set,
            txt_set,
            txe_set,
        )
        result["boreholes_count"] = len(filtered_records)

        # Handle empty filter result
        if not filtered_records:
            result.update(_create_empty_filter_result(start_time))
            logger.info("✅ %s: 0 boreholes (empty filter result)", combo_key)
            return result

        # Reconstruct GeoDataFrames from serialized records
        boreholes_gdf = deserialize_geodataframe(filtered_records, boreholes_crs)
        zones_gdf = deserialize_geodataframe(zones_records, zones_crs)

        # Compute coverage zones
        from Gap_Analysis_EC7.coverage_zones import compute_coverage_zones

        covered_union, uncovered_gaps, gap_stats = compute_coverage_zones(
            boreholes_gdf=boreholes_gdf,
            zones_gdf=zones_gdf,
            max_spacing=max_spacing,
            logger=None,
        )

        result["covered"] = serialize_geometry(covered_union)
        result["gaps"] = serialize_geometry(uncovered_gaps)

        # Compute proposed boreholes if gaps exist
        proposed, optimization_stats = _compute_proposed_boreholes(
            uncovered_gaps,
            zones_gdf,
            max_spacing,
            ilp_config,
            highs_log_folder=highs_log_folder,
        )

        # Consolidation pass: remove redundant boreholes (second pass)
        consolidation_mode = ilp_config.get("consolidation_mode", "disabled")
        if consolidation_mode != "disabled" and proposed:
            # Progress message (replaces verbose HiGHS output)
            print(f"🔗 Consolidating {combo_key}...")

            # Silence HiGHS during consolidation (override config)
            consol_ilp_config = {**ilp_config, "verbose": 0}

            # Generate HiGHS log file path if folder provided
            consol_log_file = None
            if highs_log_folder:
                import os

                consol_log_file = os.path.join(
                    highs_log_folder, f"consol_{combo_key}.log"
                )

            if consolidation_mode == "buffer_zone":
                # Buffer zone approach: lock interior boreholes, re-solve buffer only
                from Gap_Analysis_EC7.solvers.consolidation import (
                    consolidate_boreholes_buffer_zone,
                )
                from Gap_Analysis_EC7.coverage_zones import (
                    extract_internal_zone_boundaries,
                )

                # Extract internal zone boundaries (where zones meet)
                zone_boundaries = extract_internal_zone_boundaries(
                    zones_gdf=zones_gdf,
                    buffer_tolerance=0.1,  # 0.1m tolerance for edge detection
                    logger=None,
                )

                proposed, consolidation_stats = consolidate_boreholes_buffer_zone(
                    boreholes=proposed,
                    zones_gdf=zones_gdf,
                    zone_boundaries=zone_boundaries,
                    max_spacing=max_spacing,
                    ilp_config=consol_ilp_config,
                    optimization_stats=optimization_stats,
                    logger=None,  # Silent in worker
                    highs_log_file=consol_log_file,
                )
            else:
                # Traditional ILP consolidation
                from Gap_Analysis_EC7.solvers.consolidation import consolidate_boreholes

                proposed, consolidation_stats = consolidate_boreholes(
                    boreholes=proposed,
                    gap_polygon=uncovered_gaps,
                    max_spacing=max_spacing,
                    mode=consolidation_mode,
                    ilp_config=consol_ilp_config,
                    optimization_stats=optimization_stats,  # Pass for test_points access
                    logger=None,  # Silent in worker
                    highs_log_file=consol_log_file,
                )
            optimization_stats["consolidation"] = consolidation_stats

        result["proposed"] = [
            {
                "x": bh["x"],
                "y": bh["y"],
                "coverage_radius": bh.get("coverage_radius", max_spacing),
                "source_pass": "First Pass",  # Default before CZRC optimization
            }
            for bh in proposed
        ]

        # Include removed boreholes from consolidation (for visualization)
        consol_stats = optimization_stats.get("consolidation", {})
        removed_boreholes = consol_stats.get("removed_boreholes", [])
        if removed_boreholes:
            result["removed"] = [
                {
                    "x": bh["x"],
                    "y": bh["y"],
                    "coverage_radius": bh.get("coverage_radius", max_spacing),
                }
                for bh in removed_boreholes
            ]

        # Include added boreholes from consolidation (new positions from buffer zone solve)
        added_boreholes = consol_stats.get("added_boreholes", [])
        logger.debug(f"added_boreholes: added_count={len(added_boreholes)}")
        if added_boreholes:
            result["added"] = [
                {
                    "x": bh["x"],
                    "y": bh["y"],
                    "coverage_radius": bh.get("coverage_radius", max_spacing),
                }
                for bh in added_boreholes
            ]

        # Include second pass grid data for visualization
        # NOTE: Visualization shows unified fresh grid only (not first-pass BH positions)
        # See _add_second_pass_grid_trace() docstring for rationale.
        buffer_candidates_coords = consol_stats.get("buffer_candidates_coords", [])
        first_pass_candidates = consol_stats.get("first_pass_candidates", [])
        if buffer_candidates_coords:
            result["second_pass_grid"] = {
                "candidates": buffer_candidates_coords,  # All ILP candidates
                "first_pass_candidates": first_pass_candidates,  # First-pass border BHs
                "buffer_polygon_wkt": consol_stats.get(
                    "buffer_polygon_wkt"
                ),  # For unified grid
                "candidate_spacing": consol_stats.get(
                    "candidate_spacing"
                ),  # Grid spacing
                "min_spacing": consol_stats.get("min_spacing"),  # Min zone spacing
                "split_region_wkts": consol_stats.get(
                    "split_region_wkts", []
                ),  # Split regions for debug viz
            }
            print(
                f"🔍 DEBUG second_pass_grid: {len(buffer_candidates_coords)} candidates, "
                f"{len(first_pass_candidates)} first-pass"
            )

        # ═══════════════════════════════════════════════════════════════════
        # 🌐 CZRC (Cross-Zone Reachability Consolidation) VISUALIZATION
        # ═══════════════════════════════════════════════════════════════════
        # Compute CZRC regions for visualization if CZRC optimization is enabled
        from Gap_Analysis_EC7.config import CONFIG

        czrc_opt_enabled = CONFIG.get("czrc_optimization", {}).get("enabled", False)
        if czrc_opt_enabled:
            try:
                from Gap_Analysis_EC7.solvers.czrc_geometry import (
                    compute_czrc_consolidation_region,
                )

                test_spacing_mult = ilp_config.get("test_spacing_mult", 0.2)
                czrc_data = compute_czrc_consolidation_region(
                    zones_gdf=zones_gdf,
                    test_spacing_mult=test_spacing_mult,
                    logger=None,  # Silent in worker
                )

                # Add existing borehole coverage for Tier 2 test point filtering
                # This ensures Tier 2 test points don't spawn in already-covered areas
                if covered_union is not None and not covered_union.is_empty:
                    czrc_data["covered_union_wkt"] = covered_union.wkt
                    print(
                        f"   🔷 coverage_worker: added covered_union_wkt (area={covered_union.area/10000:.1f} ha)"
                    )
                else:
                    print(
                        f"   🔷 coverage_worker: NO covered_union (None={covered_union is None}, empty={covered_union.is_empty if covered_union else 'N/A'})"
                    )

                # Add CZRC data to result for visualization
                result["czrc_data"] = {
                    "total_region_wkt": czrc_data.get("total_region_wkt"),
                    "coverage_clouds_wkt": czrc_data.get("coverage_clouds_wkt", {}),
                    "pairwise_wkts": czrc_data.get("pairwise_wkts", {}),
                    "zone_spacings": czrc_data.get("zone_spacings", {}),
                    "stats": czrc_data.get("stats", {}),
                    # Include existing coverage for Tier 2 test point filtering
                    "covered_union_wkt": czrc_data.get("covered_union_wkt"),
                }

                # Log CZRC stats to console
                czrc_stats = czrc_data.get("stats", {})
                print(
                    f"   🌐 CZRC: {czrc_stats.get('n_pairwise_overlaps', 0)} pairwise overlaps, "
                    f"{czrc_stats.get('czrc_total_area_m2', 0):,.0f} m² total region"
                )

                # ═══════════════════════════════════════════════════════════════════
                # 🔗 CZRC SECOND-PASS OPTIMIZATION (New approach replacing border_consolidation)
                # ═══════════════════════════════════════════════════════════════════
                czrc_opt_config = CONFIG.get("czrc_optimization", {})
                
                # Apply testing mode mip_gap override to CZRC config
                testing_mode = CONFIG.get("testing_mode", {})
                if testing_mode.get("enabled", False):
                    solver_overrides = testing_mode.get("solver_overrides", {})
                    if "mip_gap" in solver_overrides:
                        # Deep copy to avoid modifying global CONFIG
                        czrc_opt_config = dict(czrc_opt_config)
                        czrc_ilp = dict(czrc_opt_config.get("ilp", {}))
                        czrc_ilp["mip_gap"] = solver_overrides["mip_gap"]
                        czrc_opt_config["ilp"] = czrc_ilp
                        print(f"   🧪 CZRC: testing mode mip_gap override applied: {solver_overrides['mip_gap']}")
                
                if czrc_opt_config.get("enabled", False):
                    czrc_cache = None  # Initialize for cleanup handling
                    try:
                        from Gap_Analysis_EC7.solvers.czrc_solver import (
                            run_czrc_optimization,
                        )

                        # Use SHARED CZRC cache directory from orchestrator (via ilp_config)
                        # This enables cross-worker cache sharing for duplicate ILP problems.
                        # The orchestrator creates the directory and passes the path.
                        czrc_cache_dir = ilp_config.get("czrc_cache_dir")
                        if czrc_cache_dir is not None:
                            try:
                                from Gap_Analysis_EC7.parallel.czrc_cache import (
                                    get_czrc_cache_from_path,
                                )

                                czrc_cache = get_czrc_cache_from_path(czrc_cache_dir)
                            except Exception as cache_err:
                                print(f"   ⚠️ CZRC cache init failed: {cache_err}")
                                czrc_cache = None

                        # Get test points from optimization stats (aggregated in zone decomposition)
                        all_test_points = optimization_stats.get("test_points", [])

                        # DEBUG: Log test point summary for cache investigation
                        print(
                            f"🔗 CZRC [{combo_key}]: {len(proposed)} boreholes, {len(all_test_points)} test points"
                        )
                        if all_test_points:
                            # Sample first test point to verify determinism
                            tp_sample = all_test_points[0]
                            print(
                                f"   🔍 test_point[0]: x={tp_sample.get('x')}, y={tp_sample.get('y')}, r={tp_sample.get('required_radius')}"
                            )

                        # Store First Pass boreholes before CZRC optimization
                        first_pass_bhs = [
                            {
                                "x": bh["x"],
                                "y": bh["y"],
                                "coverage_radius": bh.get(
                                    "coverage_radius", max_spacing
                                ),
                            }
                            for bh in proposed
                        ]

                        optimized_bhs, czrc_opt_stats = run_czrc_optimization(
                            first_pass_boreholes=proposed,
                            all_test_points=all_test_points,
                            czrc_data=czrc_data,
                            config=czrc_opt_config,
                            logger=None,  # Silent in worker
                            czrc_cache=czrc_cache,  # Pass cache for CZRC result caching
                            highs_log_folder=highs_log_folder,  # Pass log folder for CZRC HiGHS logs
                        )

                        # Log cache stats if enabled
                        if czrc_cache is not None:
                            czrc_cache.log_summary()
                            czrc_opt_stats["czrc_cache_stats"] = czrc_cache.get_stats()

                        # Update proposed with optimized result
                        proposed = optimized_bhs
                        optimization_stats["czrc_optimization"] = czrc_opt_stats

                        # Extract removed/added for CZRC second-pass visualization
                        czrc_removed = czrc_opt_stats.get("removed_boreholes", [])
                        czrc_added = czrc_opt_stats.get("added_boreholes", [])
                        czrc_first_pass_candidates = czrc_opt_stats.get(
                            "first_pass_candidates", []
                        )
                        czrc_test_points = czrc_opt_stats.get("czrc_test_points", [])
                        czrc_cell_wkts = czrc_opt_stats.get("cell_wkts", [])
                        # Extract third pass data for Cell-Cell CZRC visualization
                        third_pass_removed = czrc_opt_stats.get(
                            "third_pass_removed", []
                        )
                        third_pass_added = czrc_opt_stats.get("third_pass_added", [])
                        third_pass_viz_data = czrc_opt_stats.get("third_pass_data")
                        if czrc_removed:
                            result["czrc_removed"] = czrc_removed
                        if czrc_added:
                            result["czrc_added"] = czrc_added
                        if czrc_first_pass_candidates:
                            result["czrc_first_pass_candidates"] = (
                                czrc_first_pass_candidates
                            )
                        if czrc_test_points:
                            result["czrc_test_points"] = czrc_test_points
                        if czrc_cell_wkts:
                            result["czrc_cell_wkts"] = czrc_cell_wkts
                        # Add third pass data to result for visualization
                        if third_pass_removed:
                            result["third_pass_removed"] = third_pass_removed
                        if third_pass_added:
                            result["third_pass_added"] = third_pass_added
                        # Add third pass visualization data (cell clouds/intersections)
                        if third_pass_viz_data:
                            result["third_pass_data"] = third_pass_viz_data
                        # Extract cluster stats for Second Pass Tier 2 tooltips
                        czrc_cluster_stats = czrc_opt_stats.get("cluster_stats", {})
                        if czrc_cluster_stats:
                            result["czrc_cluster_stats"] = czrc_cluster_stats
                        # Store first pass boreholes for per-pass CSV export
                        if first_pass_bhs:
                            result["first_pass_boreholes"] = first_pass_bhs
                        # Store second pass boreholes for per-pass CSV export
                        second_pass_bhs = czrc_opt_stats.get(
                            "second_pass_boreholes", []
                        )
                        if second_pass_bhs:
                            result["second_pass_boreholes"] = second_pass_bhs

                        print(
                            f"   ✅ CZRC Optimization: {czrc_opt_stats.get('original_count', 0)} → "
                            f"{czrc_opt_stats.get('final_count', 0)} boreholes "
                            f"({len(czrc_removed)} removed, {len(czrc_added)} added)"
                        )

                        # Build source_pass for each proposed borehole
                        # Uses the same per-pass data already collected for CSV export
                        first_pass_positions = {
                            (bh["x"], bh["y"]) for bh in first_pass_bhs
                        }
                        # czrc_added = boreholes added in Second Pass (Zone-Zone CZRC)
                        second_pass_added_positions = {
                            (bh["x"], bh["y"]) for bh in czrc_added
                        }
                        # third_pass_added = boreholes added in Third Pass (Cell-Cell CZRC)
                        third_pass_added_positions = {
                            (bh["x"], bh["y"]) for bh in third_pass_added
                        }

                        def _get_source_pass(bh: Dict[str, Any]) -> str:
                            """Determine which pass a borehole came from."""
                            pos = (bh["x"], bh["y"])
                            if pos in third_pass_added_positions:
                                return "Third Pass"
                            if pos in second_pass_added_positions:
                                return "Second Pass"
                            if pos in first_pass_positions:
                                return "First Pass"
                            # Fallback (shouldn't happen if data is consistent)
                            return "First Pass"

                        # Update result with optimized boreholes including source_pass
                        result["proposed"] = [
                            {
                                "x": bh["x"],
                                "y": bh["y"],
                                "coverage_radius": bh.get(
                                    "coverage_radius", max_spacing
                                ),
                                "source_pass": _get_source_pass(bh),
                            }
                            for bh in proposed
                        ]
                    except Exception as e:
                        print(f"   ⚠️ CZRC optimization failed: {e}")
                        import traceback

                        traceback.print_exc()
                    # NOTE: No cleanup needed here - orchestrator owns the shared cache directory

            except Exception as e:
                print(f"   ⚠️ CZRC computation failed: {e}")

        # Build statistics
        result["stats"] = _build_result_stats(
            covered_union,
            uncovered_gaps,
            gap_stats,
            proposed,
            optimization_stats,
        )
        result["optimization_stats"] = optimization_stats

        # Mark success and return
        result["success"] = True
        result["duration_seconds"] = time.time() - start_time

        # Log with consolidation info if applicable
        consolidation_stats = optimization_stats.get("consolidation", {})
        if consolidation_stats.get("boreholes_removed", 0) > 0:
            logger.info(
                "✅ %s: %d BHs, %d proposed (%d→%d after consolidation), %.1fs",
                combo_key,
                len(filtered_records),
                len(proposed),
                consolidation_stats["original_count"],
                consolidation_stats["final_count"],
                result["duration_seconds"],
            )
        else:
            logger.info(
                "✅ %s: %d BHs, %d proposed, %.1fs",
                combo_key,
                len(filtered_records),
                len(proposed),
                result["duration_seconds"],
            )
        return result

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["duration_seconds"] = time.time() - start_time
        logger.error("❌ %s: %s", combo_key, result["error"])
        return result


# ═══════════════════════════════════════════════════════════════════════════
# 📦 MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "worker_process_filter_combination",
    "filter_boreholes_by_criteria",
]
