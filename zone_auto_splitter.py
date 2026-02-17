"""
═══════════════════════════════════════════════════════════════════════════════
🔀 ZONE AUTO-SPLITTER
═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURAL OVERVIEW
----------------------
Responsibility: Automatically split large zones into smaller Voronoi cells
before the first-pass ILP, eliminating the need for manual QGIS splitting.
Each sub-zone becomes a first-class zone in zones_gdf, so the CZRC second
pass naturally consolidates sibling cell boundaries.

Key Interactions:
    - Called by main.get_zones_for_coverage_gdf() after preprocess_zones()
    - Reuses voronoi_splitter.split_into_voronoi_cells() for cell generation
    - Reuses optimization_geometry._generate_candidate_grid() for K-means seeding
    - Output feeds directly into solver_orchestration._run_zone_decomposition()
    - CZRC second pass (czrc_geometry.py) sees expanded zones and handles
      sibling cell boundaries automatically

Data Flow:
    zones_gdf (post-overlap resolution)
    → expand_zones_with_auto_splitting()
    → for each zone > threshold:
        → _generate_candidate_grid() → split_into_voronoi_cells()
        → N sub-zone rows replace parent row
    → expanded zones_gdf (with parent_zone, is_auto_split columns)

NAVIGATION GUIDE
----------------
# ═════ IMPORTS
# ═════ CANDIDATE GRID GENERATION
# ═════ SINGLE ZONE SPLITTING
# ═════ ZONE EXPANSION (MAIN ENTRY POINT)
# ═════ MODULE EXPORTS

For Navigation: Use VS Code outline (Ctrl+Shift+O)
═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
# 📦 IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

from typing import List, Optional
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from Gap_Analysis_EC7.config_types import ZoneAutoSplittingConfig
from Gap_Analysis_EC7.solvers.optimization_geometry import _generate_candidate_grid
from Gap_Analysis_EC7.solvers.voronoi_splitter import split_into_voronoi_cells

# Module-level logger
_logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 🔲 CANDIDATE GRID GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def _build_candidate_positions(
    zone_geom: BaseGeometry,
    max_spacing_m: float,
    candidate_spacing_mult: float = 0.5,
) -> np.ndarray:
    """
    Generate candidate positions within a zone for K-means seeding.

    Creates a hexagonal grid of points within the zone geometry,
    spaced at max_spacing_m * candidate_spacing_mult.

    Args:
        zone_geom: Zone geometry (Polygon or MultiPolygon).
        max_spacing_m: Zone's max borehole spacing in meters.
        candidate_spacing_mult: Grid spacing as fraction of max_spacing.

    Returns:
        (N, 2) numpy array of candidate (x, y) positions.
    """
    # Convert geometry to polygon list for _generate_candidate_grid
    if isinstance(zone_geom, Polygon):
        polys = [zone_geom]
    elif isinstance(zone_geom, MultiPolygon):
        polys = list(zone_geom.geoms)
    else:
        polys = [zone_geom]

    grid_spacing = max_spacing_m * candidate_spacing_mult

    sample_grid = _generate_candidate_grid(
        gap_polys=polys,
        max_spacing=max_spacing_m,
        grid_spacing=grid_spacing,
        grid_type="hexagonal",
        hexagonal_density=1.5,
        logger=None,
        buffer_distance=0,  # Zone geometry is already the region — no buffer needed
    )

    if not sample_grid:
        return np.empty((0, 2))

    return np.array([[p.x, p.y] for p in sample_grid])


# ═══════════════════════════════════════════════════════════════════════════
# 🔷 SINGLE ZONE SPLITTING
# ═══════════════════════════════════════════════════════════════════════════


def _split_single_zone(
    zone_geom: BaseGeometry,
    max_spacing_m: float,
    config: ZoneAutoSplittingConfig,
    logger: logging.Logger,
) -> List[BaseGeometry]:
    """
    Split a single zone geometry into Voronoi cells.

    Generates a candidate grid within the zone, then delegates to
    split_into_voronoi_cells() from voronoi_splitter.

    Args:
        zone_geom: Zone geometry to split.
        max_spacing_m: Zone's max borehole spacing (for grid density).
        config: Auto-splitting configuration.
        logger: Logger instance.

    Returns:
        List of cell geometries (may be empty if splitting fails).
    """
    candidate_positions = _build_candidate_positions(zone_geom, max_spacing_m)

    if len(candidate_positions) < 2:
        logger.warning("   ⚠️ Too few candidates for splitting — keeping zone intact")
        return [zone_geom]

    # Build config dict in the format split_into_voronoi_cells expects
    splitter_config = {
        "kmeans_voronoi": {
            "target_cell_area_m2": config.target_cell_area_m2,
            "min_cells": config.min_cells,
            "max_cells": config.max_cells,
            "random_state": config.random_state,
        },
        "min_cell_area_m2": config.min_cell_area_m2,
    }

    cells = split_into_voronoi_cells(
        zone_geom, candidate_positions, splitter_config, logger
    )

    if not cells:
        logger.warning("   ⚠️ Voronoi splitting produced no cells — keeping zone intact")
        return [zone_geom]

    return cells


# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ ZONE EXPANSION (MAIN ENTRY POINT)
# ═══════════════════════════════════════════════════════════════════════════


def expand_zones_with_auto_splitting(
    zones_gdf: gpd.GeoDataFrame,
    config: ZoneAutoSplittingConfig,
    logger: Optional[logging.Logger] = None,
) -> gpd.GeoDataFrame:
    """
    Expand zones_gdf by splitting large zones into Voronoi cells.

    Each zone exceeding the area threshold is replaced by N sub-zone rows.
    Sub-zones inherit the parent's max_spacing_m, order, and other properties.
    Zones below the threshold pass through unchanged.

    After expansion, the CZRC second pass sees sub-zones as first-class zones
    and consolidates their boundaries automatically.

    Args:
        zones_gdf: GeoDataFrame with zone boundaries (post-overlap resolution).
            Required columns: geometry, zone_id, zone_name, max_spacing_m
        config: Auto-splitting configuration.
        logger: Optional logger instance.

    Returns:
        Expanded GeoDataFrame with additional columns:
        - parent_zone: Original zone_name (None for unsplit zones)
        - is_auto_split: True for auto-generated sub-zones
    """
    log = logger or _logger

    if not config.enabled:
        log.info("   ℹ️ Zone auto-splitting is disabled")
        zones_gdf["parent_zone"] = None
        zones_gdf["is_auto_split"] = False
        return zones_gdf

    threshold = config.max_zone_area_m2
    expanded_rows = []
    n_split = 0
    n_cells_total = 0

    log.info(
        f"🔀 Zone auto-splitting: threshold={threshold/1e6:.1f} km², "
        f"method={config.method}"
    )

    for idx, row in zones_gdf.iterrows():
        zone_geom = row["geometry"]
        zone_area = zone_geom.area
        zone_name = row["zone_name"]

        if zone_area <= threshold:
            # Zone is small enough — keep as-is
            new_row = row.to_dict()
            new_row["parent_zone"] = None
            new_row["is_auto_split"] = False
            expanded_rows.append(new_row)
            continue

        # Zone exceeds threshold — split into cells
        log.info(
            f"   🔷 Splitting '{zone_name}' ({zone_area/1e6:.2f} km²) "
            f"into sub-zones..."
        )

        cells = _split_single_zone(zone_geom, row["max_spacing_m"], config, log)

        if len(cells) <= 1:
            # Splitting didn't produce multiple cells — keep original
            new_row = row.to_dict()
            new_row["parent_zone"] = None
            new_row["is_auto_split"] = False
            expanded_rows.append(new_row)
            continue

        # Create a new row for each cell
        n_split += 1
        for cell_idx, cell_geom in enumerate(cells):
            cell_name = f"{zone_name}_cell_{cell_idx}"
            cell_id = f"{row['zone_id']}_cell_{cell_idx}"

            new_row = row.to_dict()
            new_row["geometry"] = cell_geom
            new_row["zone_name"] = cell_name
            new_row["zone_id"] = cell_id
            new_row["parent_zone"] = zone_name
            new_row["is_auto_split"] = True
            expanded_rows.append(new_row)

        n_cells_total += len(cells)
        log.info(
            f"      ✅ '{zone_name}' → {len(cells)} cells "
            f"(avg {zone_area/len(cells)/1e6:.2f} km² each)"
        )

    # Build expanded GeoDataFrame
    result = gpd.GeoDataFrame(expanded_rows, crs=zones_gdf.crs)

    if n_split > 0:
        log.info(
            f"   🔀 Auto-split summary: {n_split} zone(s) → "
            f"{n_cells_total} sub-zones "
            f"(total zones: {len(zones_gdf)} → {len(result)})"
        )
    else:
        log.info("   ✅ No zones exceeded auto-split threshold")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 📦 MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "expand_zones_with_auto_splitting",
]
