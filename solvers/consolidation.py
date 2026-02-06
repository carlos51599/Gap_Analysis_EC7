"""
Border Consolidation Module - Second Pass ILP Optimization

Architectural Overview:
    Responsibility: Remove redundant boreholes from first-pass zone-by-zone
    optimization by re-solving ILP in the buffer zone around zone boundaries.
    Runs in parallel (inside each worker per filter combination).

    Key Interactions (buffer_zone mode - recommended):
        - Classifies boreholes as interior (locked) vs border (candidates)
        - Generates FRESH hexagonal grid candidates in buffer zone
        - Also includes first-pass border boreholes as candidates
        - Solves reduced ILP for unsatisfied buffer zone test points only
        - Returns consolidated boreholes with original radii preserved

    Key Interactions (legacy "ilp" mode):
        - Receives proposed boreholes with per-borehole coverage_radius
        - Re-solves ILP using only first-pass boreholes as candidates
        - Not recommended (slower, less optimal)

    Navigation Guide:
        1. consolidate_boreholes_buffer_zone() - Recommended entry point
        2. consolidate_boreholes() - Legacy full-area re-solve
        3. _build_coverage_dict_variable_radii() - Per-borehole radius coverage
        4. _build_exclusion_pairs_variable_radii() - Per-pair conflict constraints

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import logging
import time

import numpy as np
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from Gap_Analysis_EC7.solvers.solver_algorithms import _solve_ilp
from Gap_Analysis_EC7.models.data_models import (
    Borehole,
    BoreholePass,
    BoreholeStatus,
    get_bh_coords,
    get_bh_position,
    get_bh_radius,
    get_bh_source_pass,
)

# Module-level logger for debug output (used when no logger passed to functions)
_logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 🔍 BUFFER ZONE CLASSIFICATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _filter_coincident_pairs(
    removed: List[Dict[str, Any]],
    added: List[Dict[str, Any]],
    tolerance: float = 1.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter out coincident removed/added borehole pairs.

    When a borehole is removed and another is added at the same position,
    neither should be shown in the visualization (no net change at that spot).

    Args:
        removed: List of removed boreholes {"x", "y", "coverage_radius"}
        added: List of added boreholes {"x", "y", "coverage_radius"}
        tolerance: Distance tolerance in meters for coincidence check

    Returns:
        Tuple of (filtered_removed, filtered_added) with coincident pairs removed
    """
    if not removed or not added:
        return removed, added

    # Build set of added positions for fast lookup
    added_positions = [get_bh_position(bh) for bh in added]

    # Find which removed boreholes have a coincident added borehole
    coincident_removed_indices = set()
    coincident_added_indices = set()

    for r_idx, removed_bh in enumerate(removed):
        r_x, r_y = get_bh_coords(removed_bh)
        for a_idx, (a_x, a_y) in enumerate(added_positions):
            dist = ((r_x - a_x) ** 2 + (r_y - a_y) ** 2) ** 0.5
            if dist <= tolerance:
                coincident_removed_indices.add(r_idx)
                coincident_added_indices.add(a_idx)
                break  # Each removed can match at most one added

    # Filter out coincident pairs
    filtered_removed = [
        bh for i, bh in enumerate(removed) if i not in coincident_removed_indices
    ]
    filtered_added = [
        bh for i, bh in enumerate(added) if i not in coincident_added_indices
    ]

    return filtered_removed, filtered_added


def _classify_by_boundary_distance(
    points: List[Point],
    zone_boundaries: BaseGeometry,
    buffer_width: float,
) -> Tuple[List[int], List[int]]:
    """
    Classify points as interior or border based on distance to zone boundaries.

    For buffer zone consolidation:
    - Interior points: Far from zone boundaries (> buffer_width), will be "locked"
    - Border points: Near zone boundaries (<= buffer_width), will be re-optimized

    Args:
        points: List of shapely Point objects to classify
        zone_boundaries: MultiLineString/LineString of internal zone boundaries
        buffer_width: Distance threshold for border classification (meters)

    Returns:
        Tuple of (interior_indices, border_indices)
        - interior_indices: Points > buffer_width from any boundary
        - border_indices: Points <= buffer_width from any boundary

    Example:
        >>> interior, border = _classify_by_boundary_distance(points, boundaries, 150.0)
        >>> print(f"{len(interior)} interior, {len(border)} border points")
    """
    if zone_boundaries is None:
        # No internal boundaries = all points are interior
        return list(range(len(points))), []

    interior_indices = []
    border_indices = []

    for i, pt in enumerate(points):
        dist = pt.distance(zone_boundaries)
        if dist > buffer_width:
            interior_indices.append(i)
        else:
            border_indices.append(i)

    return interior_indices, border_indices


def _classify_boreholes_and_test_points(
    boreholes: List[Dict[str, Any]],
    test_points: List[Dict[str, Any]],
    zone_boundaries: BaseGeometry,
    buffer_width: float,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Classify both boreholes and test points for buffer zone consolidation.

    This is the main classification entry point that coordinates the
    classification of all elements for the buffer zone approach.

    Args:
        boreholes: List of borehole dicts with x, y, coverage_radius
        test_points: List of test point dicts with x, y, required_radius
        zone_boundaries: Internal zone boundaries geometry
        buffer_width: Distance threshold for border classification (meters)
        logger: Optional logger

    Returns:
        Dict containing:
        - interior_boreholes: List of borehole dicts in interior zone
        - border_boreholes: List of borehole dicts in buffer zone
        - interior_borehole_indices: Indices into original boreholes list
        - border_borehole_indices: Indices into original boreholes list
        - interior_test_points: List of test point dicts in interior zone
        - buffer_test_points: List of test point dicts in buffer zone
        - interior_test_indices: Indices into original test_points list
        - buffer_test_indices: Indices into original test_points list
    """
    if zone_boundaries is None:
        # No internal boundaries - all are interior (skip consolidation)
        if logger:
            logger.info("   ℹ️ No internal zone boundaries - all points are interior")
        return {
            "interior_boreholes": boreholes,
            "border_boreholes": [],
            "interior_borehole_indices": list(range(len(boreholes))),
            "border_borehole_indices": [],
            "interior_test_points": test_points,
            "buffer_test_points": [],
            "interior_test_indices": list(range(len(test_points))),
            "buffer_test_indices": [],
        }

    # Convert to Point objects for classification
    bh_points = [Point(*get_bh_coords(bh)) for bh in boreholes]
    tp_points = [Point(*get_bh_coords(tp)) for tp in test_points]

    # Classify boreholes
    interior_bh_idx, border_bh_idx = _classify_by_boundary_distance(
        bh_points, zone_boundaries, buffer_width
    )

    # Classify test points
    interior_tp_idx, buffer_tp_idx = _classify_by_boundary_distance(
        tp_points, zone_boundaries, buffer_width
    )

    if logger:
        logger.info(
            f"   📊 Classification: {len(interior_bh_idx)} interior + "
            f"{len(border_bh_idx)} border boreholes; "
            f"{len(interior_tp_idx)} interior + {len(buffer_tp_idx)} buffer test points"
        )

    return {
        "interior_boreholes": [boreholes[i] for i in interior_bh_idx],
        "border_boreholes": [boreholes[i] for i in border_bh_idx],
        "interior_borehole_indices": interior_bh_idx,
        "border_borehole_indices": border_bh_idx,
        "interior_test_points": [test_points[i] for i in interior_tp_idx],
        "buffer_test_points": [test_points[i] for i in buffer_tp_idx],
        "interior_test_indices": interior_tp_idx,
        "buffer_test_indices": buffer_tp_idx,
    }


# ═══════════════════════════════════════════════════════════════════════════
# � PRE-COVERAGE COMPUTATION (Locked Interior Coverage)
# ═══════════════════════════════════════════════════════════════════════════


def _compute_locked_coverage(
    buffer_test_points: List[Dict[str, Any]],
    interior_boreholes: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> Set[int]:
    """
    Identify buffer test points already covered by locked interior boreholes.

    This is the key function for the buffer zone approach: interior boreholes
    are "locked" (always selected), so any test point they cover is already
    satisfied and doesn't need a constraint in the buffer ILP.

    IMPORTANT: Coverage uses each test point's required_radius, NOT the
    borehole's original coverage_radius. A test point from Zone A (50m radius)
    requires a borehole within 50m, regardless of whether that borehole
    was originally placed in Zone B (100m radius).

    Args:
        buffer_test_points: Test points in buffer zone, each with:
            - x, y: coordinates
            - required_radius: max distance for coverage (from origin zone)
        interior_boreholes: Locked boreholes from interior zones, each with:
            - x, y: coordinates

    Returns:
        Set of buffer test point indices that are pre-covered by interior boreholes

    Example:
        >>> pre_covered = _compute_locked_coverage(buffer_tps, interior_bhs)
        >>> # Test points in pre_covered don't need ILP constraints
        >>> unsatisfied = [tp for i, tp in enumerate(buffer_tps) if i not in pre_covered]
    """
    if not interior_boreholes:
        return set()

    if not buffer_test_points:
        return set()

    pre_covered: Set[int] = set()

    # Convert interior boreholes to Point objects once
    interior_points = [Point(*get_bh_coords(bh)) for bh in interior_boreholes]

    for i, tp in enumerate(buffer_test_points):
        tp_point = Point(*get_bh_coords(tp))
        required_radius = tp.get("required_radius", 100.0)  # Fallback to 100m

        # Check if any interior borehole covers this test point
        for bh_point in interior_points:
            if tp_point.distance(bh_point) <= required_radius:
                pre_covered.add(i)
                break  # Found coverage, no need to check more boreholes

    if logger:
        pct = (
            len(pre_covered) / len(buffer_test_points) * 100
            if buffer_test_points
            else 0
        )
        logger.info(
            f"   🔒 Interior coverage: {len(pre_covered)} of {len(buffer_test_points)} "
            f"buffer test points ({pct:.1f}%) pre-covered by locked boreholes"
        )

    return pre_covered


def _build_coverage_dict_variable_test_radii(
    test_points: List[Dict[str, Any]],
    candidates: List[Point],
    logger: Optional[logging.Logger] = None,
) -> Dict[int, List[int]]:
    """
    Build coverage dict using per-test-point required radius.

    Uses NumPy vectorization for ~10-50× speedup over Shapely distance calls.

    Unlike _build_coverage_dict_variable_radii (which uses per-candidate radius),
    this function uses each test point's required_radius to determine coverage.

    This is the correct approach for multi-zone consolidation where:
    - Zone A test points need coverage within 50m
    - Zone B test points need coverage within 100m
    - A borehole can satisfy Zone A points (if within 50m) AND Zone B points (if within 100m)

    Args:
        test_points: List of test point dicts with x, y, required_radius
        candidates: List of candidate Point objects

    Returns:
        Dict mapping test_point_index -> list of candidate indices that cover it
    """
    # Handle empty inputs
    if not test_points or not candidates:
        return {i: [] for i in range(len(test_points))}

    # Pre-compute coordinate arrays for vectorized distance calculation
    tp_coords = np.array([[tp["x"], tp["y"]] for tp in test_points])
    tp_radii = np.array([tp.get("required_radius", 100.0) for tp in test_points])
    cand_coords = np.array([[c.x, c.y] for c in candidates])

    coverage: Dict[int, List[int]] = {}

    for i in range(len(test_points)):
        # Vectorized distance from test point i to all candidates
        distances = np.sqrt(np.sum((cand_coords - tp_coords[i]) ** 2, axis=1))
        covering = np.where(distances <= tp_radii[i])[0].tolist()
        coverage[i] = covering

    if logger:
        total = sum(len(v) for v in coverage.values())
        uncovered = sum(1 for v in coverage.values() if not v)
        logger.info(
            f"   📊 Coverage matrix: {len(candidates)} candidates × "
            f"{len(test_points)} test points, {total} coverage entries, "
            f"{uncovered} uncoverable"
        )

    return coverage


# ═══════════════════════════════════════════════════════════════════════════
# �🔧 CANDIDATE MANAGEMENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _dedupe_candidates(
    candidates: List[Point], radii: List[float], tol: float = 0.1
) -> Tuple[List[Point], List[float]]:
    """
    Remove duplicate candidates within tolerance, keeping largest radius.

    Args:
        candidates: List of candidate points (may contain near-duplicates)
        radii: Corresponding coverage radius for each candidate
        tol: Distance tolerance for considering points identical (meters)

    Returns:
        Deduplicated (candidates, radii) tuple
    """
    if not candidates:
        return [], []

    # Simple O(n²) dedupe - sufficient for typical consolidation sizes
    # TODO: Use KDTree for O(n log n) if >5000 candidates
    seen = []  # List of (point, radius) tuples

    for cand, rad in zip(candidates, radii):
        is_dupe = False
        for i, (existing, existing_rad) in enumerate(seen):
            if cand.distance(existing) < tol:
                # Keep larger radius
                if rad > existing_rad:
                    seen[i] = (cand, rad)
                is_dupe = True
                break

        if not is_dupe:
            seen.append((cand, rad))

    deduped_candidates = [p for p, _ in seen]
    deduped_radii = [r for _, r in seen]

    return deduped_candidates, deduped_radii


def _build_warm_start_indices(
    boreholes: List[Dict[str, Any]], candidates: List[Point], tol: float = 0.1
) -> List[int]:
    """
    Map first-pass boreholes to candidate indices for warm start.

    Args:
        boreholes: First-pass borehole locations with x, y coordinates
        candidates: Unified candidate list (deduped)
        tol: Distance tolerance for matching (meters)

    Returns:
        List of candidate indices corresponding to first-pass boreholes
    """
    warm_indices = []

    for bh in boreholes:
        bh_point = Point(*get_bh_coords(bh))
        # Find closest candidate
        min_dist = float("inf")
        closest_idx = -1
        for j, cand in enumerate(candidates):
            dist = bh_point.distance(cand)
            if dist < min_dist:
                min_dist = dist
                closest_idx = j

        # Only include if within tolerance
        if closest_idx >= 0 and min_dist < tol:
            warm_indices.append(closest_idx)

    return warm_indices


# ═══════════════════════════════════════════════════════════════════════════
# ✂️ REGION SPLITTING HELPERS (OPTIONAL OPTIMIZATION)
# ═══════════════════════════════════════════════════════════════════════════


def _split_buffer_zone_regions(
    buffer_polygon: BaseGeometry,
    safe_distance: float,
    logger: Optional[logging.Logger] = None,
) -> List[BaseGeometry]:
    """
    Split a buffer zone polygon into independent regions.

    Regions separated by more than safe_distance can be solved independently
    as their boreholes cannot influence each other's coverage.

    Args:
        buffer_polygon: Buffer zone geometry (Polygon or MultiPolygon)
        safe_distance: Minimum distance for regions to be independent (2× max spacing)
        logger: Optional logger for debug output

    Returns:
        List of independent region geometries
    """
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union

    # Handle single polygon
    if buffer_polygon.geom_type == "Polygon":
        return [buffer_polygon]

    if buffer_polygon.geom_type != "MultiPolygon":
        if logger:
            logger.warning(f"   ⚠️ Unexpected geometry type: {buffer_polygon.geom_type}")
        return [buffer_polygon]

    # Get individual polygons from MultiPolygon (type narrowed by geom_type check)
    multi_poly: MultiPolygon = buffer_polygon  # type: ignore[assignment]
    polygons = list(multi_poly.geoms)

    if len(polygons) <= 1:
        return list(polygons)  # type: ignore[return-value]

    # Group polygons that are closer than safe_distance
    # These must be solved together (could influence each other)
    groups = []  # List of lists of polygon indices
    assigned = set()

    for i, poly_i in enumerate(polygons):
        if i in assigned:
            continue

        # Start a new group with this polygon
        current_group = [i]
        assigned.add(i)

        # Find all polygons close enough to influence this group
        changed = True
        while changed:
            changed = False
            for j, poly_j in enumerate(polygons):
                if j in assigned:
                    continue
                # Check if poly_j is close to any polygon in current group
                for idx in current_group:
                    dist = polygons[idx].distance(poly_j)
                    if dist < safe_distance:
                        current_group.append(j)
                        assigned.add(j)
                        changed = True
                        break

        groups.append(current_group)

    # Merge polygons in each group
    merged_regions = []
    for group in groups:
        if len(group) == 1:
            merged_regions.append(polygons[group[0]])
        else:
            # Union all polygons in this group (they're close enough to interact)
            group_polys = [polygons[idx] for idx in group]
            merged = unary_union(group_polys)
            merged_regions.append(merged)

    if logger and len(merged_regions) > 1:
        logger.info(
            f"   ✂️ Split buffer zone into {len(merged_regions)} independent regions "
            f"(safe distance: {safe_distance:.0f}m)"
        )

    return merged_regions


def _assign_points_to_regions(
    points: List[Dict[str, Any]],
    regions: List[BaseGeometry],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Assign points to their containing or nearest region.

    Args:
        points: List of point dicts with 'x', 'y' keys
        regions: List of region geometries

    Returns:
        Dict mapping region index to list of points in that region
    """
    assignments = {i: [] for i in range(len(regions))}

    for point in points:
        pt = Point(point["x"], point["y"])

        # Find which region contains or is closest to this point
        best_region = 0
        best_distance = float("inf")

        for i, region in enumerate(regions):
            if region.contains(pt) or region.touches(pt):
                best_region = i
                break
            dist = region.distance(pt)
            if dist < best_distance:
                best_distance = dist
                best_region = i

        assignments[best_region].append(point)

    return assignments


def _solve_split_regions(
    independent_regions: List[BaseGeometry],
    unsatisfied_test_points: List[Dict[str, Any]],
    buffer_candidates: List[Point],
    interior_boreholes: List[Dict[str, Any]],
    border_boreholes: List[Dict[str, Any]],
    min_spacing: float,
    max_spacing: float,
    actual_max_spacing: float,
    ilp_config: Dict[str, Any],
    consol_config: Dict[str, Any],
    boreholes: List[Dict[str, Any]],
    logger: Optional[logging.Logger],
    highs_log_file: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Solve consolidation for multiple independent regions sequentially.

    Each region gets its own subset of test points and candidates,
    and is solved as an independent ILP. Results are merged at the end.

    Args:
        independent_regions: List of disconnected buffer zone geometries
        unsatisfied_test_points: All unsatisfied test points
        buffer_candidates: All candidate boreholes
        interior_boreholes: Locked interior boreholes (added to all solutions)
        border_boreholes: First-pass border boreholes (for tracking removed)
        min_spacing: Minimum zone spacing
        max_spacing: Maximum spacing parameter
        actual_max_spacing: Actual max spacing from zones
        ilp_config: ILP configuration
        consol_config: Consolidation configuration
        boreholes: Original boreholes (for failure fallback)
        logger: Optional logger
        highs_log_file: Optional HiGHS log file path

    Returns:
        Tuple of (consolidated_boreholes, consolidation_stats)
    """
    import time as time_module

    start_time = time_module.perf_counter()

    if logger:
        logger.info(
            f"   📊 Solving {len(independent_regions)} independent regions sequentially..."
        )

    # Convert candidates to dict format for assignment
    candidates_dict = [{"x": c.x, "y": c.y} for c in buffer_candidates]

    # Assign test points and candidates to regions
    tp_assignments = _assign_points_to_regions(
        unsatisfied_test_points, independent_regions
    )
    cand_assignments = _assign_points_to_regions(candidates_dict, independent_regions)

    # Solve each region
    all_selected = []
    all_stats = []
    region_times = []

    for region_idx, region in enumerate(independent_regions):
        region_start = time_module.perf_counter()

        region_test_points = tp_assignments[region_idx]
        # Build list of candidates for this region (by matching indices)
        region_cand_indices = [
            i
            for i, cd in enumerate(candidates_dict)
            if cd in cand_assignments[region_idx]
        ]
        region_candidates = [buffer_candidates[i] for i in region_cand_indices]

        if not region_test_points or not region_candidates:
            # Empty region - skip
            region_times.append(0.0)
            continue

        if logger:
            logger.info(
                f"      Region {region_idx + 1}/{len(independent_regions)}: "
                f"{len(region_test_points)} test points, {len(region_candidates)} candidates"
            )

        # Build coverage for this region
        coverage = _build_coverage_dict_variable_test_radii(
            region_test_points, region_candidates, logger
        )

        # Determine verbosity based on config (production mode = quiet, testing = verbose)
        verbose = consol_config.get("verbose", ilp_config.get("verbose", 1))

        # Solve ILP for this region
        coverage_target_pct = consol_config.get(
            "coverage_target_pct", ilp_config.get("coverage_target_pct", 97.0)
        )
        test_points_for_ilp = [Point(tp["x"], tp["y"]) for tp in region_test_points]

        # Generate unique log file per region (avoid overwriting)
        region_log_file = None
        if highs_log_file:
            import os

            base, ext = os.path.splitext(highs_log_file)
            region_log_file = f"{base}_r{region_idx + 1}{ext}"

        selected_indices, ilp_stats = _solve_ilp(
            test_points=test_points_for_ilp,
            candidates=region_candidates,
            coverage=coverage,
            time_limit=consol_config.get(
                "time_limit", ilp_config.get("time_limit", 60)
            ),
            mip_gap=consol_config.get("mip_gap", ilp_config.get("mip_gap", 0.03)),
            threads=ilp_config.get("threads", 1),
            coverage_target_pct=coverage_target_pct,
            use_conflict_constraints=consol_config.get(
                "use_conflict_constraints",
                ilp_config.get("use_conflict_constraints", True),
            ),
            conflict_constraint_mode=ilp_config.get(
                "conflict_constraint_mode", "clique"
            ),
            exclusion_factor=consol_config.get(
                "exclusion_factor", ilp_config.get("exclusion_factor", 0.8)
            ),
            max_spacing=min_spacing,
            max_conflict_pairs=ilp_config.get("max_conflict_pairs", 200000),
            min_clique_size=ilp_config.get("min_clique_size", 3),
            max_cliques=ilp_config.get("max_cliques", 50000),
            verbose=verbose,
            mip_heuristic_effort=ilp_config.get("mip_heuristic_effort", 0.05),
            warm_start_indices=None,
            logger=logger,
            highs_log_file=region_log_file,
            stall_detection_config=ilp_config.get("stall_detection", {}),
        )

        region_elapsed = time_module.perf_counter() - region_start
        region_times.append(region_elapsed)

        if selected_indices is not None:
            for idx in selected_indices:
                if idx < len(region_candidates):
                    all_selected.append(region_candidates[idx])
            all_stats.append(ilp_stats)

    # Assemble final result
    consolidated = []

    # Add all interior boreholes (locked)
    for bh in interior_boreholes:
        x, y = get_bh_coords(bh)
        consolidated.append(
            Borehole(
                x=x,
                y=y,
                coverage_radius=get_bh_radius(bh, default=max_spacing),
                source_pass=get_bh_source_pass(bh, default=BoreholePass.FIRST),
            ).as_dict()
        )

    # Add selected boreholes from all regions
    for cand in all_selected:
        consolidated.append(
            Borehole(
                x=cand.x,
                y=cand.y,
                coverage_radius=min_spacing,
                source_pass=BoreholePass.FIRST,
            ).as_dict()
        )

    # Track removed boreholes
    selected_set = set((c.x, c.y) for c in all_selected)
    removed_boreholes = []
    for bh in border_boreholes:
        x, y = get_bh_coords(bh)
        if (x, y) not in selected_set:
            removed_boreholes.append(
                Borehole(
                    x=x,
                    y=y,
                    coverage_radius=get_bh_radius(bh, default=max_spacing),
                    source_pass=get_bh_source_pass(bh, default=BoreholePass.FIRST),
                    status=BoreholeStatus.REMOVED,
                ).as_dict()
            )

    # Track added boreholes (new locations not in original border boreholes)
    border_set = set(get_bh_position(bh) for bh in border_boreholes)
    added_boreholes = []
    for cand in all_selected:
        if (cand.x, cand.y) not in border_set:
            added_boreholes.append(
                Borehole(
                    x=cand.x,
                    y=cand.y,
                    coverage_radius=min_spacing,
                    source_pass=BoreholePass.FIRST,
                    status=BoreholeStatus.ADDED,
                ).as_dict()
            )

    # Filter coincident pairs
    removed_boreholes, added_boreholes = _filter_coincident_pairs(
        removed_boreholes, added_boreholes
    )

    total_elapsed = time_module.perf_counter() - start_time

    # Compute region extents for debugging/visualization
    region_extents = []
    region_wkts = []  # WKT strings for visualization
    for region in independent_regions:
        bounds = region.bounds  # (minx, miny, maxx, maxy)
        region_extents.append(
            {
                "min_x": bounds[0],
                "min_y": bounds[1],
                "max_x": bounds[2],
                "max_y": bounds[3],
                "width": bounds[2] - bounds[0],
                "height": bounds[3] - bounds[1],
            }
        )
        region_wkts.append(region.wkt)

    # Build buffer_candidates_coords for second pass grid visualization
    buffer_candidates_coords = [
        Borehole(
            x=cand.x,
            y=cand.y,
            coverage_radius=min_spacing,
            source_pass=BoreholePass.FIRST,
        ).as_dict()
        for cand in buffer_candidates
    ]

    # Build first_pass_candidates for visualization
    first_pass_candidates = [
        Borehole(
            x=get_bh_coords(bh)[0],
            y=get_bh_coords(bh)[1],
            coverage_radius=get_bh_radius(bh, default=min_spacing),
            source_pass=get_bh_source_pass(bh, default=BoreholePass.FIRST),
        ).as_dict()
        for bh in border_boreholes
    ]

    # Compute union of all regions for buffer polygon WKT
    from shapely.ops import unary_union

    buffer_polygon_union = unary_union(independent_regions)
    buffer_polygon_wkt = buffer_polygon_union.wkt if buffer_polygon_union else None

    # Get candidate spacing from config
    candidate_spacing = consol_config.get("candidate_spacing", min_spacing / 2)

    # Build combined stats
    combined_stats = {
        "status": "optimal" if all_stats else "failed",
        "method": "buffer_zone_split",
        "original_count": len(boreholes),
        "final_count": len(consolidated),
        "interior_locked": len(interior_boreholes),
        "border_candidates": len(border_boreholes),
        "buffer_selected": len(all_selected),
        "boreholes_removed": len(removed_boreholes),
        "boreholes_added": len(added_boreholes),
        "net_change": len(added_boreholes) - len(removed_boreholes),
        "removed_boreholes": removed_boreholes,
        "added_boreholes": added_boreholes,
        "split_regions": len(independent_regions),
        "split_region_wkts": region_wkts,  # WKT strings for visualization
        "region_extents": region_extents,
        "region_times": region_times,
        "solve_time": total_elapsed,  # Use same key as non-split path
        # Fields needed for second_pass_grid visualization:
        "buffer_candidates_coords": buffer_candidates_coords,
        "first_pass_candidates": first_pass_candidates,
        "buffer_polygon_wkt": buffer_polygon_wkt,
        "candidate_spacing": candidate_spacing,
        "min_spacing": min_spacing,
        # Aggregate stall detection stats for gap reporting
        "ilp_region_stats": all_stats,  # Detailed per-region ILP stats
    }

    # Extract and aggregate gap percentages from all region ILP stats
    region_gaps = []
    for region_stat in all_stats:
        stall_det = region_stat.get("stall_detection", {})
        if stall_det and stall_det.get("final_gap_pct") is not None:
            region_gaps.append(stall_det["final_gap_pct"])
    if region_gaps:
        combined_stats["avg_gap_pct"] = sum(region_gaps) / len(region_gaps)
        combined_stats["stall_detection"] = {
            "final_gap_pct": combined_stats["avg_gap_pct"],
            "region_gaps": region_gaps,
        }

    if logger:
        logger.info(
            f"   ✅ Split solving complete: {len(consolidated)} boreholes "
            f"({len(removed_boreholes)} removed, {len(added_boreholes)} added) "
            f"in {total_elapsed:.1f}s across {len(independent_regions)} regions"
        )
        # Log region extents for debugging
        for i, ext in enumerate(region_extents):
            logger.info(
                f"      Region {i+1}: {ext['width']:.0f}m × {ext['height']:.0f}m "
                f"(X: {ext['min_x']:.0f}-{ext['max_x']:.0f}, Y: {ext['min_y']:.0f}-{ext['max_y']:.0f})"
            )

    return consolidated, combined_stats


# ═══════════════════════════════════════════════════════════════════════════
# 🎯 BUFFER ZONE CONSOLIDATION (RECOMMENDED APPROACH)
# ═══════════════════════════════════════════════════════════════════════════


def consolidate_boreholes_buffer_zone(
    boreholes: List[Dict[str, Any]],
    zones_gdf: Any,  # GeoDataFrame with zones (for logging only)
    zone_boundaries: BaseGeometry,
    max_spacing: float,
    ilp_config: Optional[Dict[str, Any]] = None,
    optimization_stats: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    highs_log_file: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Remove redundant boreholes using the buffer zone approach.

    This is the recommended consolidation method for multi-zone optimization.
    It locks interior boreholes and only re-optimizes the buffer zone near
    zone boundaries, resulting in faster solve times and guaranteed feasibility.

    Algorithm:
    1. Classify boreholes/test points as interior vs border
    2. Lock interior boreholes (always selected)
    3. Compute which buffer test points are pre-covered by interior boreholes
    4. Generate fresh candidates in buffer zone at minimum zone spacing
    5. Solve reduced ILP for unsatisfied buffer test points only
    6. Combine interior boreholes + selected buffer boreholes

    ILP Candidates (for optimization):
        - Fresh hexagonal grid points in buffer_polygon (at candidate_spacing)
        - First-pass border boreholes (may still be optimal)

    Visualization Data (for UI):
        - buffer_polygon_wkt: Geometry of buffer zone (for unified grid generation)
        - candidate_spacing: Grid spacing used (for matching tessellation)
        - NOTE: Visualization shows ONLY the unified fresh grid, not first-pass
          borehole positions, to avoid overlapping hexagon artifacts.

    Args:
        boreholes: First-pass proposed boreholes with x, y, coverage_radius
        zones_gdf: GeoDataFrame with zones (used for logging context)
        zone_boundaries: MultiLineString of internal zone boundaries
        max_spacing: Fallback max spacing if zone-specific not available
        ilp_config: ILP solver parameters (includes consolidation_config)
        optimization_stats: First-pass stats containing test_points and zones
        logger: Optional logger
        highs_log_file: Optional path to write HiGHS solver output to a file.

    Returns:
        Tuple of (consolidated_boreholes, consolidation_stats)
    """
    from Gap_Analysis_EC7.solvers.optimization_geometry import _generate_candidate_grid

    start_time = time.perf_counter()
    ilp_config = ilp_config or {}
    optimization_stats = optimization_stats or {}

    # Extract test points and zones data from optimization_stats
    test_point_dicts = optimization_stats.get("test_points", [])
    zones_data = optimization_stats.get("zones", {})

    # Get buffer zone configuration
    consol_config = ilp_config.get("consolidation_config", {})
    buffer_width_factor = consol_config.get("buffer_width_factor", 1.5)

    # BUG FIX: Use actual max spacing from zones_data, not the function parameter
    # The function parameter is CONFIG["max_spacing_m"] (fallback, typically 100m)
    # But we need the actual max from zones that share borders (e.g., 150m for Zone 2)
    zone_spacings_for_buffer = []
    if zones_data:
        for zone_info in zones_data.values():
            zone_spacings_for_buffer.append(zone_info.get("max_spacing_m", 100.0))
    actual_max_spacing = (
        max(zone_spacings_for_buffer) if zone_spacings_for_buffer else max_spacing
    )
    buffer_width = actual_max_spacing * buffer_width_factor

    if logger:
        logger.info(
            f"🔗 Buffer Zone Consolidation: Starting with {len(boreholes)} boreholes"
        )
        logger.info(
            f"   Buffer width: {buffer_width:.1f}m ({buffer_width_factor}× actual_max_spacing={actual_max_spacing:.0f}m)"
        )

    # === VALIDATION ===
    if not boreholes:
        return [], {"status": "skipped", "reason": "no_boreholes"}

    if len(boreholes) < 2:
        return boreholes, {"status": "skipped", "reason": "single_borehole"}

    if not test_point_dicts:
        return boreholes, {"status": "skipped", "reason": "no_test_points"}

    if zone_boundaries is None:
        if logger:
            logger.info("   ℹ️ No zone boundaries - skipping buffer zone consolidation")
        return boreholes, {"status": "skipped", "reason": "no_zone_boundaries"}

    # === STEP 1: CLASSIFY BOREHOLES AND TEST POINTS ===
    classification = _classify_boreholes_and_test_points(
        boreholes, test_point_dicts, zone_boundaries, buffer_width, logger
    )

    interior_boreholes = classification["interior_boreholes"]
    border_boreholes = classification["border_boreholes"]
    buffer_test_points = classification["buffer_test_points"]

    # If no border boreholes or buffer test points, nothing to consolidate
    if not border_boreholes:
        if logger:
            logger.info("   ℹ️ No border boreholes - returning all as interior")
        return boreholes, {
            "status": "skipped",
            "reason": "no_border_boreholes",
            "interior_count": len(interior_boreholes),
        }

    if not buffer_test_points:
        if logger:
            logger.info("   ℹ️ No buffer test points - returning original")
        return boreholes, {
            "status": "skipped",
            "reason": "no_buffer_test_points",
        }

    # === STEP 2: COMPUTE PRE-COVERED TEST POINTS ===
    pre_covered_indices = _compute_locked_coverage(
        buffer_test_points, interior_boreholes, logger
    )

    # Get unsatisfied buffer test points
    unsatisfied_test_points = [
        tp for i, tp in enumerate(buffer_test_points) if i not in pre_covered_indices
    ]

    if not unsatisfied_test_points:
        if logger:
            logger.info(
                "   ✅ All buffer test points covered by interior - no ILP needed"
            )
        # All buffer test points are covered by interior, no consolidation needed
        return boreholes, {
            "status": "success",
            "method": "all_precovered",
            "original_count": len(boreholes),
            "final_count": len(boreholes),
            "boreholes_removed": 0,
            "interior_count": len(interior_boreholes),
            "border_count": len(border_boreholes),
            "precovered_pct": 100.0,
        }

    # === STEP 3: GENERATE FRESH CANDIDATES IN BUFFER ZONE ===
    # Find minimum spacing across all zones (for grid generation)
    # Note: We already computed zone_spacings_for_buffer earlier for buffer_width
    # Here we reuse that data but compute min_spacing for the candidate grid
    zone_spacings = zone_spacings_for_buffer if zone_spacings_for_buffer else [100.0]

    min_spacing = min(zone_spacings)
    # max_spacing is already computed as actual_max_spacing for buffer_width

    if logger:
        logger.info(
            f"   📊 Zone spacings: min={min_spacing:.0f}m, max={actual_max_spacing:.0f}m, "
            f"using {min_spacing:.0f}m for candidate grid"
        )

    # Create buffer zone polygon
    buffer_polygon = zone_boundaries.buffer(buffer_width)

    # Generate fresh candidates at minimum spacing
    from shapely.geometry import Polygon, MultiPolygon

    buffer_polys: List[Polygon] = []
    if buffer_polygon.geom_type == "Polygon":
        buffer_polys = [buffer_polygon]  # type: ignore[list-item]
    elif buffer_polygon.geom_type == "MultiPolygon":
        multi_poly: MultiPolygon = buffer_polygon  # type: ignore[assignment]
        buffer_polys = list(multi_poly.geoms)  # type: ignore[arg-type]

    # Get candidate spacing multiplier from config
    candidate_mult = ilp_config.get("candidate_spacing_mult", 0.5)
    candidate_spacing = min_spacing * candidate_mult

    buffer_candidates = _generate_candidate_grid(
        buffer_polys,
        max_spacing=min_spacing,
        grid_spacing=candidate_spacing,
        grid_type="hexagonal",
        hexagonal_density=1.5,
        logger=logger,
    )

    # Also include border boreholes as candidates (they may still be optimal)
    for bh in border_boreholes:
        buffer_candidates.append(Point(*get_bh_coords(bh)))

    if logger:
        logger.info(
            f"   🎯 Buffer zone: {len(buffer_candidates)} candidates "
            f"({len(border_boreholes)} from first-pass + fresh grid)"
        )

    # === STEP 3.5: CHECK FOR REGION SPLITTING (OPTIONAL OPTIMIZATION) ===
    consol_config = ilp_config.get("consolidation_config", {})
    splitting_enabled = consol_config.get("splitting_enabled", True)
    safe_distance = (
        2 * actual_max_spacing
    )  # 2× max zone spacing for guaranteed independence

    if splitting_enabled:
        independent_regions = _split_buffer_zone_regions(
            buffer_polygon, safe_distance, logger
        )

        # If multiple independent regions, use split solving
        if len(independent_regions) > 1:
            return _solve_split_regions(
                independent_regions=independent_regions,
                unsatisfied_test_points=unsatisfied_test_points,
                buffer_candidates=buffer_candidates,
                interior_boreholes=interior_boreholes,
                border_boreholes=border_boreholes,
                min_spacing=min_spacing,
                max_spacing=max_spacing,
                actual_max_spacing=actual_max_spacing,
                ilp_config=ilp_config,
                consol_config=consol_config,
                boreholes=boreholes,
                logger=logger,
                highs_log_file=highs_log_file,
            )

    # === STEP 4: BUILD COVERAGE DICT WITH VARIABLE TEST RADII ===
    coverage = _build_coverage_dict_variable_test_radii(
        unsatisfied_test_points, buffer_candidates, logger
    )

    # Check coverability
    uncoverable = sum(1 for v in coverage.values() if not v)
    if uncoverable > 0 and logger:
        logger.warning(
            f"   ⚠️ {uncoverable} test points cannot be covered by any candidate"
        )

    # === STEP 5: SOLVE REDUCED ILP ===
    # consol_config already extracted in Step 3.5
    coverage_target_pct = consol_config.get(
        "coverage_target_pct", ilp_config.get("coverage_target_pct", 97.0)
    )

    # Note: No warm start for buffer zone ILP - interior boreholes are locked as
    # constants (not decision variables), and the reduced ILP solves fast enough
    # without warm start hints.

    # Convert test points to Point objects for _solve_ilp
    test_points_for_ilp = [Point(tp["x"], tp["y"]) for tp in unsatisfied_test_points]

    selected_indices, ilp_stats = _solve_ilp(
        test_points=test_points_for_ilp,
        candidates=buffer_candidates,
        coverage=coverage,
        time_limit=consol_config.get("time_limit", ilp_config.get("time_limit", 60)),
        mip_gap=consol_config.get("mip_gap", ilp_config.get("mip_gap", 0.03)),
        threads=ilp_config.get("threads", 1),
        coverage_target_pct=coverage_target_pct,
        use_conflict_constraints=consol_config.get(
            "use_conflict_constraints", ilp_config.get("use_conflict_constraints", True)
        ),
        conflict_constraint_mode=ilp_config.get("conflict_constraint_mode", "clique"),
        exclusion_factor=consol_config.get(
            "exclusion_factor", ilp_config.get("exclusion_factor", 0.8)
        ),
        max_spacing=min_spacing,  # Use minimum spacing for conflict constraints
        max_conflict_pairs=ilp_config.get("max_conflict_pairs", 200000),
        min_clique_size=ilp_config.get("min_clique_size", 3),
        max_cliques=ilp_config.get("max_cliques", 50000),
        verbose=consol_config.get("verbose", ilp_config.get("verbose", 1)),
        mip_heuristic_effort=ilp_config.get("mip_heuristic_effort", 0.05),
        warm_start_indices=None,  # No warm start - interior boreholes are constants
        logger=logger,
        highs_log_file=highs_log_file,
        stall_detection_config=ilp_config.get("stall_detection", {}),
    )

    # === STEP 6: HANDLE ILP FAILURE ===
    if selected_indices is None:
        if logger:
            logger.warning("⚠️ Buffer zone ILP failed, returning original boreholes")
        return boreholes, {
            "status": "failed",
            "reason": ilp_stats.get("reason", "unknown"),
            "original_count": len(boreholes),
            "final_count": len(boreholes),
            "boreholes_removed": 0,
        }

    # === STEP 7: ASSEMBLE FINAL RESULT ===
    # Combine locked interior boreholes + selected buffer boreholes
    consolidated = []

    # Add all interior boreholes (locked)
    for bh in interior_boreholes:
        x, y = get_bh_coords(bh)
        consolidated.append(
            Borehole(
                x=x,
                y=y,
                coverage_radius=get_bh_radius(bh, default=max_spacing),
                source_pass=get_bh_source_pass(bh, default=BoreholePass.FIRST),
            ).as_dict()
        )

    # Add selected buffer boreholes
    for idx in selected_indices:
        if idx < len(buffer_candidates):
            cand = buffer_candidates[idx]
            # Determine radius - use minimum spacing for new candidates
            consolidated.append(
                Borehole(
                    x=cand.x,
                    y=cand.y,
                    coverage_radius=min_spacing,
                    source_pass=BoreholePass.FIRST,
                ).as_dict()
            )

    # Track removed boreholes (original border boreholes that are replaced)
    # These are the first-pass border boreholes that were replaced by new buffer candidates
    removed_boreholes = [
        Borehole(
            x=get_bh_coords(bh)[0],
            y=get_bh_coords(bh)[1],
            coverage_radius=get_bh_radius(bh, default=max_spacing),
            source_pass=get_bh_source_pass(bh, default=BoreholePass.FIRST),
            status=BoreholeStatus.REMOVED,
        ).as_dict()
        for bh in border_boreholes
    ]

    # Track added boreholes (new buffer candidates at new positions)
    # These are fresh candidates that were selected in the buffer zone re-solve
    added_boreholes = []
    if selected_indices:
        for idx in selected_indices:
            if idx < len(buffer_candidates):
                cand = buffer_candidates[idx]
                added_boreholes.append(
                    Borehole(
                        x=cand.x,
                        y=cand.y,
                        coverage_radius=min_spacing,
                        source_pass=BoreholePass.FIRST,
                        status=BoreholeStatus.ADDED,
                    ).as_dict()
                )

    # Filter out coincident removed/added pairs (same position = no net change)
    # When a borehole is removed and another is added at the same spot,
    # show neither red nor green marker - just the blue proposed marker
    removed_boreholes, added_boreholes = _filter_coincident_pairs(
        removed_boreholes, added_boreholes, tolerance=1.0
    )

    elapsed = time.perf_counter() - start_time

    # Convert buffer_candidates (Points) to serializable format for visualization
    buffer_candidates_coords = [
        Borehole(
            x=cand.x,
            y=cand.y,
            coverage_radius=min_spacing,
            source_pass=BoreholePass.FIRST,
        ).as_dict()
        for cand in buffer_candidates
    ]

    # Convert buffer_polygon to WKT for serialization (used for unified grid visualization)
    buffer_polygon_wkt = buffer_polygon.wkt if buffer_polygon else None

    # Extract first-pass border boreholes (used as non-grid candidates)
    # These are the original border boreholes that were also candidates in the ILP
    first_pass_candidates = [
        Borehole(
            x=get_bh_coords(bh)[0],
            y=get_bh_coords(bh)[1],
            coverage_radius=get_bh_radius(bh, default=min_spacing),
            source_pass=get_bh_source_pass(bh, default=BoreholePass.FIRST),
        ).as_dict()
        for bh in border_boreholes
    ]

    stats = {
        "status": "success",
        "method": "buffer_zone",
        "original_count": len(boreholes),
        "final_count": len(consolidated),
        "boreholes_removed": len(boreholes) - len(consolidated),
        "removed_boreholes": removed_boreholes,  # Original border boreholes that were replaced
        "added_boreholes": added_boreholes,  # New buffer candidates at new positions
        "buffer_candidates_coords": buffer_candidates_coords,  # For second pass grid visualization
        "first_pass_candidates": first_pass_candidates,  # First-pass border BHs used as ILP candidates
        "buffer_polygon_wkt": buffer_polygon_wkt,  # Geometry for unified grid visualization
        "candidate_spacing": candidate_spacing,  # Grid spacing used for candidate generation
        "min_spacing": min_spacing,  # Minimum zone spacing
        "improvement_pct": (
            (len(boreholes) - len(consolidated)) / len(boreholes) * 100
            if len(boreholes) > 0
            else 0
        ),
        "interior_count": len(interior_boreholes),
        "border_count": len(border_boreholes),
        "buffer_candidates": len(buffer_candidates),
        "buffer_test_points": len(buffer_test_points),
        "unsatisfied_test_points": len(unsatisfied_test_points),
        "precovered_count": len(pre_covered_indices),
        "selected_buffer_count": len(selected_indices) if selected_indices else 0,
        "solve_time": elapsed,
        "ilp_stats": ilp_stats,
    }

    if logger:
        logger.info(
            f"✅ Buffer zone consolidation: {stats['original_count']} → {stats['final_count']} "
            f"({stats['boreholes_removed']} removed, {stats['improvement_pct']:.1f}%) "
            f"in {elapsed:.2f}s"
        )
        logger.info(
            f"   Breakdown: {len(interior_boreholes)} interior (locked) + "
            f"{len(selected_indices) if selected_indices else 0} buffer (selected)"
        )

    return consolidated, stats


# ═══════════════════════════════════════════════════════════════════════════
# 🎯 MAIN CONSOLIDATION ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


def consolidate_boreholes(
    boreholes: List[Dict[str, Any]],
    gap_polygon: Any,  # Not used - we have test_points
    max_spacing: float,  # Fallback only
    mode: str = "ilp",
    ilp_config: Optional[Dict[str, Any]] = None,
    optimization_stats: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    highs_log_file: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Remove redundant boreholes via second-pass ILP optimization.

    Uses proposed boreholes from first pass as candidates (not full grid).
    Handles variable coverage radii from different zones via per-pair
    exclusion constraints.

    Args:
        boreholes: First-pass proposed boreholes with coverage_radius
        gap_polygon: Not used (kept for interface compatibility)
        max_spacing: Fallback if borehole lacks coverage_radius
        mode: "ilp" (only supported mode)
        ilp_config: ILP solver parameters
        optimization_stats: First-pass stats containing test_points
        logger: Optional logger
        highs_log_file: Optional path to write HiGHS solver output to a file.

    Returns:
        Tuple of (consolidated_boreholes, consolidation_stats)
    """
    start_time = time.perf_counter()
    ilp_config = ilp_config or {}
    _log = logger or _logger
    _log.debug(f"CONSOLIDATION: Starting with {len(boreholes)} boreholes")

    # === VALIDATION ===
    if not boreholes:
        _log.debug("CONSOLIDATION: Skipping - no boreholes")
        return [], {"status": "skipped", "reason": "no_boreholes"}

    if len(boreholes) < 2:
        _log.debug("CONSOLIDATION: Skipping - single borehole")
        return boreholes, {"status": "skipped", "reason": "single_borehole"}

    # Get test points from first-pass stats
    test_point_dicts = (optimization_stats or {}).get("test_points", [])
    _log.debug(f"CONSOLIDATION: Found {len(test_point_dicts)} test points")
    if not test_point_dicts:
        if logger:
            logger.warning("⚠️ Consolidation: No test points in stats, skipping")
        _log.debug("CONSOLIDATION: Skipping - no test points")
        return boreholes, {"status": "skipped", "reason": "no_test_points"}

    # === EXTRACT UNIFIED CANDIDATES FROM ALL ZONES ===
    # Instead of using only first-pass boreholes, use ALL zone candidates
    # This ensures feasibility when zones have different spacing
    zones_data = (optimization_stats or {}).get("zones", {})

    all_candidates = []
    all_radii = []

    if zones_data:
        # Unified candidates from all zones
        for zone_name, zone_info in zones_data.items():
            zone_spacing = zone_info.get("max_spacing_m", max_spacing)
            zone_stats = zone_info.get("stats", {})
            zone_candidates = zone_stats.get("candidates", [])

            for cand in zone_candidates:
                all_candidates.append(Point(cand["x"], cand["y"]))
                all_radii.append(zone_spacing)

        _log.debug(
            f"CONSOLIDATION: Unified {len(all_candidates)} candidates from {len(zones_data)} zones"
        )

    # Fallback: if no zone data, use first-pass boreholes (original behavior)
    if not all_candidates:
        _log.debug(
            "CONSOLIDATION: No zone candidates found, using first-pass boreholes"
        )
        all_candidates = [Point(*get_bh_coords(bh)) for bh in boreholes]
        all_radii = [get_bh_radius(bh, default=max_spacing) for bh in boreholes]

    # Deduplicate candidates (may overlap at zone boundaries)
    candidates, radii = _dedupe_candidates(all_candidates, all_radii)

    _log.debug(
        f"CONSOLIDATION: After dedupe: {len(candidates)} candidates, radii range [{min(radii):.0f}, {max(radii):.0f}]m"
    )

    # === BUILD WARM START FROM FIRST-PASS BOREHOLES ===
    # Map first-pass borehole locations to unified candidate indices
    warm_start_indices = _build_warm_start_indices(boreholes, candidates)
    _log.debug(
        f"CONSOLIDATION: Warm start: {len(warm_start_indices)} of {len(boreholes)} first-pass boreholes mapped to candidates"
    )

    test_points = [Point(tp["x"], tp["y"]) for tp in test_point_dicts]

    _log.debug(
        f"CONSOLIDATION: {len(candidates)} candidates, {len(test_points)} test points"
    )
    if logger:
        logger.info(
            f"🔗 Consolidation: {len(candidates)} candidates, "
            f"{len(test_points)} test points, radii range "
            f"[{min(radii):.0f}, {max(radii):.0f}]m"
        )

    # === BUILD VARIABLE-RADIUS COVERAGE DICT ===
    coverage = _build_coverage_dict_variable_radii(
        test_points, candidates, radii, logger
    )

    # Calculate coverability statistics
    uncoverable = sum(1 for v in coverage.values() if not v)
    coverable_count = len(test_points) - uncoverable
    coverable_pct = coverable_count / len(test_points) * 100 if test_points else 0
    _log.debug(
        f"CONSOLIDATION: Coverage dict built - {uncoverable} uncoverable test points ({coverable_pct:.1f}% coverable)"
    )

    # If too many test points are uncoverable, skip consolidation
    if coverable_pct < 50:
        _log.debug(
            f"CONSOLIDATION: Skipping - only {coverable_pct:.1f}% of test points are coverable (need >= 50%)"
        )
        return boreholes, {
            "status": "skipped",
            "reason": "too_few_coverable",
            "coverable_pct": coverable_pct,
        }

    # === GET CONSOLIDATION-SPECIFIC SETTINGS ===
    # Use consolidation_config if provided, otherwise fall back to ilp_config
    consol_config = ilp_config.get("consolidation_config", {})
    effective_max_spacing = max(radii)

    # Adjust coverage target based on coverability
    # If 70% of test points are coverable and we want 97% of those, that's 0.7 * 0.97 = 67.9% of all test points
    base_coverage_target = consol_config.get(
        "coverage_target_pct", ilp_config.get("coverage_target_pct", 97.0)
    )
    adjusted_coverage_target = coverable_pct * (base_coverage_target / 100.0)
    _log.debug(
        f"CONSOLIDATION: Adjusted coverage target: {base_coverage_target:.1f}% of {coverable_pct:.1f}% coverable = {adjusted_coverage_target:.1f}% overall"
    )
    _log.debug(
        f"CONSOLIDATION: Calling _solve_ilp with verbose={consol_config.get('verbose', ilp_config.get('verbose', 1))}"
    )

    selected_indices, ilp_stats = _solve_ilp(
        test_points=test_points,
        candidates=candidates,
        coverage=coverage,
        time_limit=consol_config.get("time_limit", ilp_config.get("time_limit", 60)),
        mip_gap=consol_config.get("mip_gap", ilp_config.get("mip_gap", 0.03)),
        threads=ilp_config.get("threads", 1),
        coverage_target_pct=adjusted_coverage_target,  # Use adjusted target
        use_conflict_constraints=consol_config.get(
            "use_conflict_constraints", ilp_config.get("use_conflict_constraints", True)
        ),
        conflict_constraint_mode=ilp_config.get("conflict_constraint_mode", "clique"),
        exclusion_factor=consol_config.get(
            "exclusion_factor", ilp_config.get("exclusion_factor", 0.8)
        ),
        max_spacing=effective_max_spacing,  # Use max radius for exclusion
        max_conflict_pairs=ilp_config.get("max_conflict_pairs", 200000),
        min_clique_size=ilp_config.get("min_clique_size", 3),
        max_cliques=ilp_config.get("max_cliques", 50000),
        verbose=consol_config.get("verbose", ilp_config.get("verbose", 1)),
        mip_heuristic_effort=ilp_config.get("mip_heuristic_effort", 0.05),
        warm_start_indices=warm_start_indices,
        logger=logger,
        highs_log_file=highs_log_file,
        stall_detection_config=ilp_config.get("stall_detection", {}),
    )
    _log.debug(
        f"CONSOLIDATION: _solve_ilp returned, selected_indices={selected_indices is not None}, ilp_stats={ilp_stats.get('status', 'unknown')}"
    )

    # === HANDLE ILP FAILURE ===
    if selected_indices is None:
        _log.debug(
            f"CONSOLIDATION: ILP failed with reason: {ilp_stats.get('reason', 'unknown')}"
        )
        if logger:
            logger.warning("⚠️ Consolidation ILP failed, returning original")
        return boreholes, {
            "status": "failed",
            "reason": ilp_stats.get("reason", "unknown"),
            "original_count": len(boreholes),
            "final_count": len(boreholes),
            "boreholes_removed": 0,
        }

    # === BUILD CONSOLIDATED RESULT ===
    # selected_indices are indices into the unified 'candidates' list
    # We need to convert back to borehole records with coordinates and radii
    selected_set = set(selected_indices) if selected_indices else set()
    consolidated = []
    removed = []

    # Build source_pass lookup from input boreholes for provenance tracking
    source_pass_lookup = {
        get_bh_position(bh): get_bh_source_pass(bh, default=BoreholePass.FIRST)
        for bh in boreholes
    }

    for idx in range(len(candidates)):
        cand = candidates[idx]
        rad = radii[idx] if idx < len(radii) else max_spacing
        pos_key = (round(cand.x, 6), round(cand.y, 6))
        pass_enum = source_pass_lookup.get(pos_key, BoreholePass.FIRST)
        is_selected = idx in selected_set
        bh_record = Borehole(
            x=cand.x,
            y=cand.y,
            coverage_radius=rad,
            source_pass=pass_enum,
            status=(BoreholeStatus.PROPOSED if is_selected else BoreholeStatus.REMOVED),
        ).as_dict()
        if is_selected:
            consolidated.append(bh_record)
        else:
            removed.append(bh_record)

    _log.debug(
        f"CONSOLIDATION: SUCCESS! {len(boreholes)} → {len(consolidated)} boreholes ({len(removed)} removed)"
    )

    elapsed = time.perf_counter() - start_time
    stats = {
        "status": "success",
        "original_count": len(boreholes),
        "final_count": len(consolidated),
        "boreholes_removed": len(removed),
        "removed_boreholes": removed,  # Include actual removed borehole records
        "improvement_pct": (len(removed)) / len(boreholes) * 100 if boreholes else 0,
        "solve_time": elapsed,
        "ilp_stats": ilp_stats,
    }

    if logger:
        logger.info(
            f"✅ Consolidation: {stats['original_count']} → {stats['final_count']} "
            f"({stats['boreholes_removed']} removed, {stats['improvement_pct']:.1f}%) "
            f"in {elapsed:.2f}s"
        )

    return consolidated, stats


# ═══════════════════════════════════════════════════════════════════════════
# 🔧 VARIABLE-RADIUS HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _build_coverage_dict_variable_radii(
    test_points: List[Point],
    candidates: List[Point],
    radii: List[float],
    logger: Optional[logging.Logger] = None,
) -> Dict[int, List[int]]:
    """
    Build coverage dict where each candidate has its own radius.

    Uses NumPy vectorization for ~10-50× speedup over Shapely distance calls.

    Args:
        test_points: Points that must be covered
        candidates: Candidate borehole positions
        radii: Per-candidate coverage radius

    Returns:
        Dict mapping test_point_index -> list of candidate indices that cover it
        (this is the format expected by _solve_ilp)
    """
    # Handle empty inputs
    if not test_points or not candidates:
        return {j: [] for j in range(len(test_points))}

    # Pre-compute coordinate arrays for vectorized distance calculation
    tp_coords = np.array([[tp.x, tp.y] for tp in test_points])
    cand_coords = np.array([[c.x, c.y] for c in candidates])
    radii_arr = np.array(radii)

    # Initialize coverage for each test point
    coverage: Dict[int, List[int]] = {j: [] for j in range(len(test_points))}

    # For each candidate, find which test points it covers (vectorized)
    for i in range(len(candidates)):
        # Vectorized distance from candidate i to all test points
        distances = np.sqrt(np.sum((tp_coords - cand_coords[i]) ** 2, axis=1))
        covered_tp_indices = np.where(distances <= radii_arr[i])[0]
        for j in covered_tp_indices:
            coverage[j].append(i)

    total_coverage = sum(len(v) for v in coverage.values())
    uncovered = sum(1 for v in coverage.values() if not v)
    coverable_count = len(test_points) - uncovered
    coverable_pct = coverable_count / len(test_points) * 100 if test_points else 0
    _log = logger or _logger
    _log.debug(
        f"CONSOLIDATION COVERAGE: {len(candidates)} candidates × {len(test_points)} test points, {total_coverage} coverage entries, {uncovered} uncoverable test points ({coverable_pct:.1f}% coverable)"
    )

    if logger:
        logger.info(
            f"   📊 Coverage matrix: {len(candidates)} candidates × "
            f"{len(test_points)} test points, {total_coverage} coverage entries, "
            f"{uncovered} uncoverable test points"
        )

    return coverage
