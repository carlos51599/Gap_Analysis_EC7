#!/usr/bin/env python3
"""
EC7 Optimization Geometry Module

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Geometric preprocessing for borehole placement optimization.
Generates candidate grids, test points, coverage mappings, and constraints.

Key Functions:
- _normalize_gaps: Convert gap geometries to list of Polygons
- _decompose_into_components: Split gaps into independent spatial clusters
- _generate_candidate_grid: Create hexagonal/rectangular candidate locations
- generate_hexagon_grid_polygons: Create hex polygons for visualization (PUBLIC)
- _generate_test_points: Create test points within gaps
- _build_coverage_dict: Map candidates to coverable test points
- _generate_conflict_pairs: Create pairwise exclusion constraints
- _generate_clique_constraints: Create clique-based exclusion constraints

CONFIGURATION ARCHITECTURE:
- No CONFIG access - all functions accept explicit parameters
- Geometry operations use Shapely for spatial processing
- Constraint generation uses scipy/networkx for graph algorithms

For Navigation: Use VS Code outline (Ctrl+Shift+O) to jump between sections.
"""

import logging
import math
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


# ===========================================================================
# 🏗️ GAP NORMALIZATION SECTION
# ===========================================================================


def _normalize_gaps(uncovered_gaps: BaseGeometry) -> List[Polygon]:
    """
    Normalize uncovered gaps to list of Polygon objects.

    Args:
        uncovered_gaps: Shapely geometry (Polygon, MultiPolygon, or GeometryCollection)

    Returns:
        List of Polygon objects, filtering out tiny gaps (<100m²)
    """
    if uncovered_gaps.geom_type == "Polygon":
        polys = [uncovered_gaps]
    elif uncovered_gaps.geom_type == "MultiPolygon":
        polys = list(uncovered_gaps.geoms)
    elif uncovered_gaps.geom_type == "GeometryCollection":
        polys = [g for g in uncovered_gaps.geoms if g.geom_type == "Polygon"]
    else:
        polys = []

    # Filter out tiny gaps
    return [p for p in polys if not p.is_empty and p.area >= 100]


# ===========================================================================
# 🧩 CONNECTED COMPONENTS DECOMPOSITION SECTION
# ===========================================================================


def _decompose_into_components(
    gap_polys: List[Polygon],
    max_spacing: float,
    logger: Optional[logging.Logger] = None,
) -> List[List[Polygon]]:
    """
    Decompose gaps into independent connected components.

    Two gaps are "connected" if a single borehole could potentially
    cover points from both (i.e., distance < 2 × max_spacing).

    MATHEMATICAL FOUNDATION:
    If dist(gap_A, gap_B) >= 2 × radius, no borehole can cover points
    from both gaps simultaneously, making them independent subproblems.

    COMPLEXITY:
    - O(n²) pairwise distance checks (acceptable for typical n < 100 gaps)
    - O(n) connected components using scipy.sparse.csgraph

    Args:
        gap_polys: List of gap Polygon objects
        max_spacing: Coverage radius (200m for EC7)
        logger: Optional logger

    Returns:
        List of components, where each component is a list of Polygons
        that must be solved together. Components are sorted by total
        area (largest first) for better progress feedback.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n_gaps = len(gap_polys)

    # Handle trivial cases
    if n_gaps <= 1:
        return [gap_polys] if gap_polys else []

    # Two gaps can share a borehole if distance < 2 × max_spacing
    # At distance = 2 × max_spacing, circles can just touch but not overlap
    connectivity_threshold = 2 * max_spacing  # 400m for 200m radius

    # Build adjacency list (sparse representation for efficiency)
    rows, cols = [], []
    for i in range(n_gaps):
        for j in range(i + 1, n_gaps):
            dist = gap_polys[i].distance(gap_polys[j])
            if dist < connectivity_threshold:
                # Bidirectional edge
                rows.extend([i, j])
                cols.extend([j, i])

    # Create sparse adjacency matrix
    if rows:
        data = [1] * len(rows)
        adjacency = csr_matrix((data, (rows, cols)), shape=(n_gaps, n_gaps))
    else:
        # No connections - all gaps are independent
        adjacency = csr_matrix((n_gaps, n_gaps))

    # Find connected components
    n_components, labels = connected_components(
        adjacency, directed=False, return_labels=True
    )

    # Group gaps by component
    components: List[List[Polygon]] = [[] for _ in range(n_components)]
    for gap_idx, comp_label in enumerate(labels):
        components[comp_label].append(gap_polys[gap_idx])

    # Sort components by total area (largest first for better progress feedback)
    components.sort(key=lambda comp: sum(g.area for g in comp), reverse=True)

    # Filter out empty components (shouldn't happen but defensive)
    components = [comp for comp in components if comp]

    if logger and n_components > 1:
        logger.info(
            f"   🧩 Decomposed {n_gaps} gaps into {len(components)} "
            f"independent components"
        )
        for i, comp in enumerate(components[:5]):  # Show first 5 only
            comp_area = sum(g.area for g in comp) / 10000  # ha
            logger.info(f"      Component {i+1}: {len(comp)} gaps, {comp_area:.1f} ha")
        if len(components) > 5:
            logger.info(f"      ... and {len(components) - 5} more components")

    return components


# ===========================================================================
# 📍 CANDIDATE GRID GENERATION SECTION
# ===========================================================================


def _generate_candidate_grid(
    gap_polys: List[Polygon],
    max_spacing: float,
    grid_spacing: float,
    grid_type: str = "hexagonal",
    hexagonal_density: float = 1.5,
    logger: Optional[logging.Logger] = None,
    buffer_distance: Optional[float] = None,
) -> List[Point]:
    """
    Generate candidate borehole locations on a grid near gaps.

    Creates grid points within buffer_distance of any gap polygon.
    This ensures candidates can cover any point inside the gaps.

    Supports two grid types:
    - "hexagonal": Honeycomb pattern - optimal for disk packing (15-25% fewer boreholes)
    - "rectangular": Traditional square grid (legacy behavior)

    Hexagonal Grid Geometry:
    ------------------------
    For circles of radius r to tile with overlap, hexagonal packing places centers:
    - Horizontal spacing: r × √3 × (2/density) = ~230m for r=200m, density=1.5
    - Vertical spacing: r × 1.5 × (2/density) = ~200m for r=200m, density=1.5
    - Odd rows are offset by half the horizontal spacing (honeycomb pattern)

    The density factor controls overlap:
    - density=1.0: Minimal overlap (circles just touch)
    - density=1.5: Moderate overlap (recommended for coverage guarantee)
    - density=2.0: Dense overlap (similar to rectangular grid)

    Args:
        gap_polys: List of gap Polygon objects
        max_spacing: Coverage radius (determines search area, e.g., 200m for EC7)
        grid_spacing: Distance between grid points (used for rectangular grid)
        grid_type: "hexagonal" (optimal) or "rectangular" (legacy)
        hexagonal_density: Density multiplier for hexagonal grid (1.0-2.0, default 1.5)
        logger: Optional logger
        buffer_distance: Distance to buffer gap_polys for search area.
                         Defaults to max_spacing if None.
                         Set to 0 for CZRC where gap_polys is already Tier 1.

    Returns:
        List of Point objects as candidate locations
    """
    if not gap_polys:
        return []

    # Union all gaps and create search buffer
    # buffer_distance defaults to max_spacing for backward compatibility
    # Set buffer_distance=0 for CZRC where tier1_region is passed directly
    actual_buffer = buffer_distance if buffer_distance is not None else max_spacing
    gaps_union = unary_union(gap_polys)
    if actual_buffer > 0:
        search_buffer = gaps_union.buffer(actual_buffer)
    else:
        search_buffer = gaps_union
    bounds = search_buffer.bounds  # (minx, miny, maxx, maxy)

    candidates = []

    if grid_type == "hexagonal":
        # === HEXAGONAL GRID (OPTIMAL FOR DISK PACKING) ===
        # Hexagonal packing achieves ~90.7% coverage efficiency vs ~78.5% for square
        # This means ~15% fewer boreholes needed for the same coverage
        #
        # GEOMETRY: For hexagonal grid with center-to-center distance = d:
        #   - Horizontal spacing (dx) = d (same row)
        #   - Vertical spacing (dy) = d * √3/2 ≈ 0.866d (between rows)
        #   - Odd row offset = d/2 (creates honeycomb pattern)
        #
        # This ensures EVERY adjacent hexagon center is exactly 'd' apart:
        #   - Same row neighbors: distance = dx = d ✓
        #   - Diagonal neighbors: distance = √((d/2)² + (d*√3/2)²) = √(d²/4 + 3d²/4) = d ✓
        #
        d = grid_spacing  # Center-to-center distance = candidate_spacing_m
        dx = d  # Horizontal spacing within row
        dy = d * math.sqrt(3) / 2  # Vertical spacing between rows (~0.866 * d)

        if logger:
            logger.info(
                f"   🔷 Hexagonal grid: spacing={d:.1f}m (center-to-center), "
                f"dx={dx:.1f}m, dy={dy:.1f}m"
            )

        # Generate hexagonal lattice pattern
        row_idx = 0
        y = bounds[1]
        while y <= bounds[3] + dy:
            # Offset odd rows by half dx (creates honeycomb pattern)
            x_offset = (row_idx % 2) * (dx / 2)
            x = bounds[0] + x_offset
            while x <= bounds[2] + dx:
                pt = Point(x, y)
                if search_buffer.contains(pt):
                    candidates.append(pt)
                x += dx
            y += dy
            row_idx += 1
    else:
        # === RECTANGULAR GRID (LEGACY BEHAVIOR) ===
        # Square grid with uniform spacing in both directions
        x_range = np.arange(bounds[0], bounds[2] + grid_spacing, grid_spacing)
        y_range = np.arange(bounds[1], bounds[3] + grid_spacing, grid_spacing)

        for x in x_range:
            for y in y_range:
                pt = Point(x, y)
                if search_buffer.contains(pt):
                    candidates.append(pt)

    if logger:
        logger.info(f"   📍 Generated {len(candidates)} candidate locations")

        # Log efficiency metrics
        total_gap_area = sum(g.area for g in gap_polys)
        circle_area = math.pi * max_spacing * max_spacing

        # Theoretical minimum boreholes (perfect packing)
        # Hexagonal packing: ~90.7% efficiency, Square: ~78.5% efficiency
        hex_efficiency = 0.907  # Hexagonal packing efficiency
        square_efficiency = 0.785  # Square packing efficiency (π/4)

        min_boreholes_hex = math.ceil(total_gap_area / (circle_area * hex_efficiency))
        min_boreholes_square = math.ceil(
            total_gap_area / (circle_area * square_efficiency)
        )

        logger.info(f"   📐 Gap geometry: {total_gap_area/10000:.2f} ha total area")
        logger.info(
            f"   🔵 Borehole coverage: {circle_area/10000:.2f} ha per borehole (r={max_spacing}m)"
        )
        logger.info(
            f"   📊 Theoretical minimums: {min_boreholes_hex} (hex packing) vs {min_boreholes_square} (square packing)"
        )
        logger.info(
            f"   🔷 Grid type: {grid_type} with {len(candidates)} candidates for ILP to choose from"
        )

    return candidates


def generate_hexagon_grid_polygons(
    bounds: Tuple[float, float, float, float],
    grid_spacing: float = 50.0,
    clip_geometry: Optional[BaseGeometry] = None,
    origin: Optional[Tuple[float, float]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Polygon]:
    """
    Generate hexagon polygon geometries for visualization overlay.

    Creates a honeycomb pattern of hexagon polygons covering the extent.
    This is used for visual representation of the candidate grid, NOT for
    borehole placement optimization (which uses point candidates).

    When clip_geometry is provided, only hexagons whose center point is
    within the clip geometry are included (matches ILP candidate filtering).

    Hexagonal Grid Geometry:
    ------------------------
    For hexagons with center-to-center distance = d:
    - Hexagon flat-to-flat width = d (matches grid spacing)
    - Hexagon vertex-to-vertex height = d / cos(30°) = 2d / √3 ≈ 1.155d
    - Horizontal offset for odd rows = d / 2

    Grid Alignment:
    ---------------
    When origin is provided, the grid is aligned so that a hexagon center
    would exist at the origin point (even if outside bounds). This ensures
    grids with the same spacing and origin are perfectly aligned.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy) in meters
        grid_spacing: Center-to-center distance between hexagon centers
        clip_geometry: Optional Shapely geometry - hexagons outside are excluded
        origin: Optional (x, y) tuple for grid alignment. If provided, hexagon
                centers are placed on a grid aligned to this origin point.
        logger: Optional logger instance

    Returns:
        List of Shapely Polygon objects representing hexagons
    """
    d = grid_spacing  # Center-to-center distance
    dx = d  # Horizontal spacing within row
    dy = d * math.sqrt(3) / 2  # Vertical spacing between rows

    # Hexagon dimensions: flat-to-flat = d, vertex-to-vertex = d / cos(30°)
    # For a hexagon centered at (cx, cy) with flat sides top/bottom:
    # The 6 vertices at angle offsets: 0°, 60°, 120°, 180°, 240°, 300°
    # Outer radius (center to vertex) = d / √3 for flat-top hexagon
    outer_radius = d / math.sqrt(3)

    # Compute grid start position aligned to origin (if provided)
    if origin is not None:
        # Align grid to origin: find first row/col that covers bounds[0], bounds[1]
        origin_x, origin_y = origin
        # Number of dy steps from origin to reach below bounds[1]
        n_rows_back = math.ceil((origin_y - bounds[1]) / dy)
        start_y = origin_y - n_rows_back * dy
        # Determine row_idx offset so odd/even row pattern aligns with origin
        row_idx_start = -n_rows_back
    else:
        start_y = bounds[1]
        row_idx_start = 0

    hexagons = []
    row_idx = row_idx_start
    y = start_y

    while y <= bounds[3] + dy:
        # Offset odd rows by half dx (honeycomb pattern)
        x_offset = (row_idx % 2) * (dx / 2)

        # Compute x start position aligned to origin (if provided)
        if origin is not None:
            origin_x, _ = origin
            # For this row, compute the effective x origin accounting for row offset
            # The origin row (row_idx=0) has x_offset based on row_idx_start % 2
            effective_origin_x = origin_x + x_offset
            # Number of dx steps from effective origin to reach below bounds[0]
            n_cols_back = math.ceil((effective_origin_x - bounds[0]) / dx)
            start_x = effective_origin_x - n_cols_back * dx
        else:
            start_x = bounds[0] + x_offset

        x = start_x
        while x <= bounds[2] + dx:
            # Check if center point is within clip geometry (if provided)
            center_pt = Point(x, y)
            if clip_geometry is None or clip_geometry.contains(center_pt):
                # Generate hexagon vertices (flat-top orientation)
                # Vertices at angles: 30°, 90°, 150°, 210°, 270°, 330°
                vertices = []
                for i in range(6):
                    angle = math.radians(30 + 60 * i)
                    vx = x + outer_radius * math.cos(angle)
                    vy = y + outer_radius * math.sin(angle)
                    vertices.append((vx, vy))

                hexagons.append(Polygon(vertices))
            x += dx

        y += dy
        row_idx += 1

    if logger:
        logger.info(
            f"   🔷 Generated {len(hexagons)} hexagon polygons "
            f"(spacing={d:.1f}m, outer_r={outer_radius:.1f}m)"
        )

    return hexagons


# ===========================================================================
# 📊 TEST POINT GENERATION SECTION
# ===========================================================================


def _generate_test_points(
    gap_polys: List[Polygon],
    spacing: float,
    logger: Optional[logging.Logger] = None,
) -> List[Point]:
    """
    Generate test points within gap polygons for coverage verification.

    Uses hexagonal lattice pattern matching the candidate grid geometry.
    All test points must be covered by selected boreholes for complete coverage.

    Hexagonal Grid Geometry:
    ------------------------
    For a hexagonal grid with center-to-center distance = d:
    - Horizontal spacing (dx) = d (within same row)
    - Vertical spacing (dy) = d × √3/2 ≈ 0.866d (between rows)
    - Odd rows are offset by d/2 (creates honeycomb pattern)

    This ensures uniform point density matching the hexagonal candidate grid,
    providing consistent coverage verification across the gap area.

    Args:
        gap_polys: List of gap Polygon objects
        spacing: Center-to-center distance between test points
        logger: Optional logger

    Returns:
        List of Point objects within gaps
    """
    test_points = []

    # Hexagonal grid parameters (same geometry as candidate grid)
    d = spacing  # Center-to-center distance
    dx = d  # Horizontal spacing within row
    dy = d * math.sqrt(3) / 2  # Vertical spacing between rows (~0.866 * d)

    for gap in gap_polys:
        bounds = gap.bounds  # (minx, miny, maxx, maxy)

        # Generate hexagonal lattice pattern within this gap
        row_idx = 0
        y = bounds[1]
        while y <= bounds[3] + dy:
            # Offset odd rows by half dx (creates honeycomb pattern)
            x_offset = (row_idx % 2) * (dx / 2)
            x = bounds[0] + x_offset
            while x <= bounds[2] + dx:
                pt = Point(x, y)
                if gap.contains(pt):
                    test_points.append(pt)
                x += dx
            y += dy
            row_idx += 1

    if logger:
        logger.info(
            f"   📊 Generated {len(test_points)} test points (hexagonal grid, "
            f"spacing={spacing:.1f}m)"
        )

    return test_points


# ===========================================================================
# 🔨 COVERAGE MATRIX SECTION
# ===========================================================================


def _build_coverage_dict(
    test_points: List[Point],
    candidates: List[Point],
    radius: float,
    logger: Optional[logging.Logger] = None,
) -> Dict[int, List[int]]:
    """
    Build mapping of which candidates cover which test points.

    Uses sparse representation (dict) instead of dense matrix for memory efficiency.

    Args:
        test_points: List of test points to cover
        candidates: List of candidate borehole locations
        radius: Coverage radius per borehole
        logger: Optional logger

    Returns:
        Dict mapping test_point_index -> list of candidate_indices that cover it
    """
    if logger:
        logger.info("   🔨 Building coverage mapping...")

    # Pre-compute coordinate arrays for vectorized distance calculation
    test_coords = np.array([[p.x, p.y] for p in test_points])
    cand_coords = np.array([[c.x, c.y] for c in candidates])

    coverage = {}
    uncoverable = []

    for i in range(len(test_points)):
        # Vectorized distance from test point i to all candidates
        distances = np.sqrt(np.sum((cand_coords - test_coords[i]) ** 2, axis=1))
        covering = np.where(distances <= radius)[0].tolist()
        coverage[i] = covering

        if len(covering) == 0:
            uncoverable.append(i)

    if uncoverable and logger:
        logger.warning(
            f"   ⚠️ {len(uncoverable)} test points cannot be covered by any candidate"
        )

    return coverage


# ===========================================================================
# 🔷 CONFLICT CONSTRAINT GENERATION SECTION
# ===========================================================================


def _generate_conflict_pairs(
    candidates: List[Point],
    exclusion_dist: float,
    max_pairs: int = 200000,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Tuple[int, int]], int, bool]:
    """
    Generate pairs of candidates that should not both be selected.

    Uses KDTree spatial index for O(n log n) query instead of O(n²) brute force.

    For hexagonal packing with coverage radius r:
    - Optimal spacing is r√3 ≈ 1.73r (tangent circles)
    - For guaranteed coverage with overlap, use exclusion ≈ 0.8 × r to 1.0 × r

    Args:
        candidates: List of candidate borehole locations
        exclusion_dist: Minimum distance between selected boreholes
        max_pairs: Maximum conflict pairs to generate (memory safety limit)
        logger: Optional logger

    Returns:
        Tuple of:
        - List of (i, j) index pairs representing mutual exclusion conflicts
        - Total pairs generated (before truncation)
        - Whether truncation occurred
    """
    from scipy.spatial import KDTree

    if len(candidates) < 2:
        return [], 0, False

    # Build spatial index for efficient nearest-neighbor queries
    coords = np.array([[c.x, c.y] for c in candidates])
    tree = KDTree(coords)

    # Find all candidate pairs within exclusion distance
    # query_pairs returns frozenset of (i, j) tuples where i < j
    pairs = list(tree.query_pairs(exclusion_dist))

    if logger:
        logger.info(
            f"   🔷 Generated {len(pairs)} conflict pairs (exclusion_dist={exclusion_dist:.1f}m)"
        )

    # Safety limit to prevent memory/performance issues
    # Track truncation
    total_generated = len(pairs)
    was_truncated = len(pairs) > max_pairs

    if was_truncated:
        if logger:
            logger.warning(
                f"   ⚠️ Too many conflict pairs ({len(pairs)} > {max_pairs}), truncating"
            )
        pairs = pairs[:max_pairs]

    return pairs, total_generated, was_truncated


def _generate_clique_constraints(
    candidates: List[Point],
    exclusion_dist: float,
    min_clique_size: int = 3,
    max_cliques: int = 50000,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Tuple[int, ...]], Dict[str, Any]]:
    """
    Generate maximal clique constraints from conflict graph.

    Uses NetworkX for efficient maximal clique enumeration (Bron-Kerbosch algorithm).
    Clique constraints are mathematically stronger than pairwise constraints,
    providing tighter LP bounds and faster solve times.

    Mathematical Benefit:
    ---------------------
    For a clique Q of size k:
    - Replaces C(k,2) = k(k-1)/2 pairwise constraints with 1 clique constraint
    - LP bound improvement: (k-2)/k × 100%
      - k=4: 50% improvement
      - k=7: 71% improvement
      - k=9: 78% improvement
      - k=13: 85% improvement

    With pairwise constraints, LP relaxation allows x_j = 0.5 for all members,
    giving LP bound = k/2. With clique constraint sum(x_j) <= 1, LP bound = 1.

    Args:
        candidates: List of candidate borehole locations
        exclusion_dist: Minimum separation distance (candidates closer conflict)
        min_clique_size: Minimum clique size to include (default 3, triangles)
        max_cliques: Maximum cliques to enumerate (memory/time safety)
        logger: Optional logger instance

    Returns:
        Tuple of:
        - List of cliques, each as tuple of candidate indices
        - Stats dict with clique_count, max_size, avg_size, size_distribution
    """
    import networkx as nx
    from scipy.spatial import KDTree

    stats: Dict[str, Any] = {
        "total_candidates": len(candidates),
        "exclusion_dist": exclusion_dist,
        "min_clique_size_param": min_clique_size,
    }

    if len(candidates) < min_clique_size:
        stats["clique_count"] = 0
        stats["max_clique_size"] = 0
        stats["edge_count"] = 0
        return [], stats

    # Build conflict graph using KDTree (same edge detection as pairwise)
    coords = np.array([[c.x, c.y] for c in candidates])
    tree = KDTree(coords)
    pairs = tree.query_pairs(exclusion_dist)

    stats["edge_count"] = len(pairs)

    if len(pairs) == 0:
        stats["clique_count"] = 0
        stats["max_clique_size"] = 0
        return [], stats

    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(len(candidates)))
    G.add_edges_from(pairs)

    # Enumerate maximal cliques using Bron-Kerbosch algorithm with pivoting
    # NetworkX's find_cliques() is optimized for sparse graphs
    cliques: List[Tuple[int, ...]] = []
    clique_sizes: List[int] = []
    was_truncated = False

    for clique in nx.find_cliques(G):
        if len(clique) >= min_clique_size:
            cliques.append(tuple(sorted(clique)))
            clique_sizes.append(len(clique))
            if len(cliques) >= max_cliques:
                was_truncated = True
                if logger:
                    logger.warning(
                        f"   ⚠️ Reached max_cliques limit ({max_cliques}), "
                        f"stopping enumeration"
                    )
                break

    # Compute statistics
    if cliques:
        stats["clique_count"] = len(cliques)
        stats["max_clique_size"] = max(clique_sizes)
        stats["min_clique_size_found"] = min(clique_sizes)
        stats["avg_clique_size"] = sum(clique_sizes) / len(clique_sizes)
        stats["was_truncated"] = was_truncated

        # Size distribution (how many cliques of each size)
        size_distribution: Dict[int, int] = {}
        for size in clique_sizes:
            size_distribution[size] = size_distribution.get(size, 0) + 1
        stats["size_distribution"] = size_distribution

        # Calculate theoretical pairwise equivalent (for comparison)
        pairwise_equivalent = sum(s * (s - 1) // 2 for s in clique_sizes)
        stats["pairwise_equivalent"] = pairwise_equivalent
        stats["reduction_ratio"] = pairwise_equivalent / len(cliques) if cliques else 0
    else:
        stats["clique_count"] = 0
        stats["max_clique_size"] = 0
        stats["was_truncated"] = False

    if logger:
        if cliques:
            logger.info(
                f"   🔷 Generated {len(cliques)} clique constraints "
                f"(max size {stats['max_clique_size']}, "
                f"avg size {stats['avg_clique_size']:.1f}, "
                f"~{stats['reduction_ratio']:.1f}× vs pairwise)"
            )
        else:
            logger.info(
                f"   🔷 No cliques of size >= {min_clique_size} found "
                f"(graph has {len(pairs)} edges)"
            )

    return cliques, stats
