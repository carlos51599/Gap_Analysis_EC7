"""
Spacing-relative cell sizing utilities.

Architectural Overview:
    Responsibility: Compute effective cell area thresholds that scale with
        candidate grid spacing. Prevents cell-size/tier-width mismatch that
        causes third-pass borehole explosion for large-spacing zones.
    Key Interactions:
        - Called by zone_auto_splitter.expand_zones_with_auto_splitting()
        - Called by czrc_solver.check_and_split_large_cluster()
        - Config from config_types.SpacingRelativeSizingConfig
    Navigation Guide:
        Single function module — no sections needed.
    For Navigation: Use Ctrl+Shift+O → compute_effective_cell_thresholds
"""

from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════
# 📐 EFFECTIVE THRESHOLD COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════


def compute_effective_cell_thresholds(
    candidate_grid_spacing_m: float,
    base_threshold_m2: float,
    base_target_area_m2: float,
    spacing_relative_sizing: bool,
    cell_area_multiplier: float,
    threshold_multiplier: float,
) -> Tuple[float, float]:
    """
    Compute effective split threshold and target cell area.

    When spacing_relative_sizing is enabled, thresholds scale with
    candidate_grid_spacing² so that cell diameter >> tier 1 width.

    Design constraint (K=400, candidate_mult=0.5):
        cell_diameter ≈ √(K) × s_c ≈ 20 × s_c
        tier1_width   ≈ 8 × s_c
        ratio ≈ 40% (tier1 is thin strip, not full cell)

    Args:
        candidate_grid_spacing_m: Candidate grid spacing for this zone/cluster.
            First pass: max_spacing_m × candidate_spacing_mult
            Second pass: min(zone_spacings) × candidate_grid_spacing_mult
        base_threshold_m2: Absolute floor for split trigger (m²).
        base_target_area_m2: Absolute floor for target cell area (m²).
        spacing_relative_sizing: Enable scaling with candidate_spacing².
        cell_area_multiplier: K — target = max(base, K × spacing²).
        threshold_multiplier: M — threshold = max(base, M × spacing²).

    Returns:
        (effective_threshold_m2, effective_target_area_m2)
    """
    if not spacing_relative_sizing:
        return base_threshold_m2, base_target_area_m2

    spacing_sq = candidate_grid_spacing_m**2

    effective_target = max(
        base_target_area_m2,
        cell_area_multiplier * spacing_sq,
    )
    effective_threshold = max(
        base_threshold_m2,
        threshold_multiplier * spacing_sq,
    )

    return effective_threshold, effective_target


# ═══════════════════════════════════════════════════════════════════════════
# 📦 MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "compute_effective_cell_thresholds",
]
