"""
Shared Voronoi Cell Splitting Utilities.

Architectural Overview:
=======================
Responsibility: Decompose large geometries into smaller, balanced cells using
K-means clustering and Voronoi tessellation. Used by both first-pass zone
auto-splitting (solver_orchestration.py) and second-pass CZRC cell splitting
(czrc_solver.py) to ensure ILP-solvable region sizes.

Key Interactions:
    - Called by czrc_solver.check_and_split_large_cluster() for CZRC cell splitting
    - Called by zone expansion logic for first-pass auto-splitting
    - Uses sklearn.cluster.KMeans for balanced seed placement
    - Uses shapely.ops.voronoi_diagram for Voronoi tessellation

Algorithm Summary:
    1. determine_cell_count(): K = ceil(area / target_area), clamped to [min, max]
    2. K-means on candidate positions → K seed points
    3. Voronoi diagram from seeds, clipped to the region boundary
    4. Filter out tiny sliver cells below min_cell_area_m2

Configuration Keys (from cell_splitting config dict):
    - kmeans_voronoi.target_cell_area_m2: Target average cell area (default 1 km²)
    - kmeans_voronoi.min_cells: Minimum K (default 2)
    - kmeans_voronoi.max_cells: Maximum K (default 50)
    - kmeans_voronoi.random_state: KMeans seed for determinism (default 42)
    - min_cell_area_m2: Discard cells smaller than this (default 100 m²)

For Navigation: Use VS Code outline (Ctrl+Shift+O)
"""

# ═══════════════════════════════════════════════════════════════════════════
# 📦 IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

from typing import Any, Dict, List, Optional
import logging
import math

import numpy as np
from shapely.geometry import MultiPoint, MultiPolygon, Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import voronoi_diagram
from sklearn.cluster import KMeans

# Module-level logger
_logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 🔧 CELL COUNT DETERMINATION
# ═══════════════════════════════════════════════════════════════════════════


def determine_cell_count(
    region_area_m2: float,
    config: Dict[str, Any],
) -> int:
    """
    Calculate number of cells based on target average cell area.

    Uses rule: K = ceil(region_area / target_cell_area), with min/max bounds.

    Args:
        region_area_m2: Total area of region to split (m²)
        config: cell_splitting config section with kmeans_voronoi sub-dict

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


# ═══════════════════════════════════════════════════════════════════════════
# 🔷 VORONOI CELL GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def get_clipped_voronoi_cells(
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


def split_into_voronoi_cells(
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
    n_cells = determine_cell_count(region_area, config)

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
    cells = get_clipped_voronoi_cells(seeds, geometry, min_cell_area_m2=min_cell_area)

    target_area = kmeans_config.get("target_cell_area_m2", 1_000_000)
    log.info(
        f"   🔷 K-means + Voronoi: {region_area/1e6:.2f} km² → {len(cells)} cells "
        f"(target: {target_area/1e6:.1f} km² avg)"
    )

    return cells


# ═══════════════════════════════════════════════════════════════════════════
# 🔲 GRID CELL SPLITTING (Fallback Method)
# ═══════════════════════════════════════════════════════════════════════════


def split_into_grid_cells(
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


# ═══════════════════════════════════════════════════════════════════════════
# 📦 MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "determine_cell_count",
    "get_clipped_voronoi_cells",
    "split_into_voronoi_cells",
    "split_into_grid_cells",
]
