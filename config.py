#!/usr/bin/env python3
"""
EC7 Simple Gap Analysis - Configuration

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Centralized configuration for EC7 gap analysis.
Single source of truth for file paths, visualization, zones, and BGS layers.

Configuration Sections (ordered by importance for algorithm tuning):
1. max_spacing_m: EC7 spacing parameter
2. testing_mode: Quick toggle for dev testing (highest visibility)
3. candidate_grid: Grid type configuration
4. ilp_solver: ILP solver parameters
5. greedy_solver: Greedy heuristic parameters
6. border_consolidation: Second-pass zone boundary deduplication
7. filter_ui: Depth slider settings
8. optimization: Solver mode selection
9. parallel: Parallel processing settings
10. cache: Cache settings
11. file_paths: Input/output file locations (bottom - rarely changed)
12. zones: Per-zone enable/disable and boundary colors (bottom)
13. visualization: Plotly styling settings (bottom)
14. bgs_bedrock/bgs_deposits: BGS geology layer settings (bottom)
15. quality_control: CRS and validation settings

Navigation Guide:
- Use VS Code outline (Ctrl+Shift+O) to jump between sections
"""

import os
from typing import Dict, Any, TypeVar, Callable, Optional

T = TypeVar("T")


def _env_or_default(
    key: str, default: T, type_fn: Optional[Callable[[str], T]] = None
) -> T:
    """
    Get value from environment variable or use default.

    Args:
        key: Environment variable name (e.g., "EC7_CONSTRAINT_MODE")
        default: Default value if env var not set
        type_fn: Optional type conversion function (e.g., float, int)

    Returns:
        Value from environment (converted) or default

    Example:
        >>> _env_or_default("EC7_EXCLUSION_FACTOR", 0.8, float)
        0.8  # If env var not set
    """
    val = os.getenv(key)
    if val is not None:
        if type_fn is not None:
            return type_fn(val)
        return val  # type: ignore
    return default


def _env_bool(key: str, default: bool) -> bool:
    """
    Get boolean value from environment variable.

    Treats "true", "1", "yes" as True (case-insensitive).
    Any other value or unset returns default.
    """
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


# ═══════════════════════════════════════════════════════════════════════════
# 🔧 ENVIRONMENT VARIABLE OVERRIDES
# ═══════════════════════════════════════════════════════════════════════════
# These settings can be overridden via environment variables for testing:
#
# EC7_CONSTRAINT_MODE        - "clique" or "pairwise" (default: "pairwise")
# EC7_EXCLUSION_FACTOR       - float, e.g., "0.8" (default: 0.9) - first pass
# EC7_CZRC_EXCLUSION_FACTOR  - float, e.g., "0.8" (default: 0.9) - second pass (CZRC)
# EC7_HEURISTIC_EFFORT       - float 0.0-1.0, e.g., "0.15" (default: 0.05)
# EC7_CONNECTED_COMPS        - "true" or "false" (default: "false")
# EC7_CANDIDATE_SPACING_MULT - float, multiplier of max_spacing_m (default: 0.5)
# EC7_TEST_SPACING_MULT      - float, multiplier of max_spacing_m (default: 0.2)
# EC7_EARLY_TERM_GAP         - float, early termination gap % (default: 15.0) - first pass
# EC7_CZRC_EARLY_TERM_GAP    - float, early termination gap % (default: 15.0) - second pass
# EC7_TIER1_RMAX_MULT        - float, Tier 1 = CZRC + mult×R_max (default: 1.0)
# EC7_TIER2_RMAX_MULT        - float, Tier 2 = CZRC + mult×R_max (default: 2.0)
#
# Example usage:
#   $env:EC7_CONSTRAINT_MODE = "pairwise"
#   $env:EC7_EXCLUSION_FACTOR = "0.8"
#   $env:EC7_HEURISTIC_EFFORT = "0.15"
#   $env:EC7_CONNECTED_COMPS = "true"
#   $env:EC7_CANDIDATE_SPACING_MULT = "0.5"
#   $env:EC7_TEST_SPACING_MULT = "0.2"
#   $env:EC7_TIER1_RMAX_MULT = "0.5"
#   $env:EC7_TIER2_RMAX_MULT = "1.5"
#   python -m Gap_Analysis_EC7.main
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# ⚙️ MASTER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CONFIG: Dict[str, Any] = {
    # ═══════════════════════════════════════════════════════════════════════
    # 📐 EC7 SPACING (Single Site-Wide Value)
    # ═══════════════════════════════════════════════════════════════════════
    # EN 1997-2 Annex B.1 recommends 20-50m for embankments
    # Conservative value for SESRO (GC3 project with variable formations)
    "max_spacing_m": 100.0,
    # ═══════════════════════════════════════════════════════════════════════
    # 🧪 TESTING MODE CONFIGURATION (High visibility - frequently toggled)
    # ═══════════════════════════════════════════════════════════════════════
    "testing_mode": {
        # Master toggle - set True to enable single-combination testing
        "enabled": True,  # Disabled for CZRC cross-worker cache testing
        # Fixed filter settings for testing (creates maximum gap scenario)
        "filter": {
            "min_depth": 50,  # Depth >= x (creates good test gaps)
            "require_spt": False,  # No SPT requirement
            "require_triaxial_total": False,  # No TxT requirement
            "require_triaxial_effective": False,  # No TxE requirement
        },
        # Cache overwrite - always recompute (recommended for algorithm testing)
        "force_cache_overwrite": True,
        # Force single worker - runs parallel with n_jobs=1 (same code path, sequential execution)
        "force_single_worker": True,
        # Use full ILP solver with extended timeout (for quality testing)
        "solver_overrides": {
            "solver_mode": "ilp",  # Force ILP for testing
            "time_limit_s": 90,  # Standard timeout
            "mip_gap": 0.1,  # Standard tolerance
        },
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🔷 CANDIDATE GRID CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    # Controls the geometry of candidate borehole placement grid.
    #
    # Hexagonal (honeycomb) packing is mathematically optimal for disk coverage:
    # - Achieves ~90.7% coverage efficiency vs ~78.5% for square packing
    # - Results in ~15-25% fewer boreholes for large rectangular gaps
    # - Positions candidates at optimal spacing for 200m radius circles
    #
    # grid_type options:
    #   - "hexagonal": Honeycomb pattern (RECOMMENDED - optimal disk packing)
    #   - "rectangular": Traditional square grid (legacy behavior)
    #
    "candidate_grid": {
        "grid_type": "hexagonal",  # "hexagonal" | "rectangular"
        # Density factor for hexagonal grid (controls overlap between coverage circles)
        # - 1.0: Minimal overlap (circles just touch - may leave tiny gaps)
        # - 1.5: Moderate overlap (RECOMMENDED - guarantees coverage)
        # - 2.0: Dense overlap (similar candidate count to rectangular)
        "hexagonal_density": 1.5,
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🧮 ILP SOLVER CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    # Settings for Integer Linear Programming solver (used when solver_mode="ilp")
    # ILP provides optimal/near-optimal solutions by minimizing borehole count
    "ilp_solver": {
        # Coverage target percentage (100% = full coverage required)
        "coverage_target_pct": 100.0,
        # ═══════════════════════════════════════════════════════════════════
        # CANDIDATE SPACING (as multiplier of max_spacing_m)
        # ═══════════════════════════════════════════════════════════════════
        # Controls the grid density for candidate borehole locations.
        # Expressed as a fraction of max_spacing_m for automatic scaling.
        #
        # - 0.25: Dense grid (25m at 100m max_spacing) - very precise, slower
        # - 0.50: Standard grid (50m at 100m max_spacing) - RECOMMENDED
        # - 0.75: Sparse grid (75m at 100m max_spacing) - faster, less optimal
        #
        # NOTE: Candidate grid uses hexagonal packing (see candidate_grid.grid_type)
        # ENV OVERRIDE: EC7_CANDIDATE_SPACING_MULT (float, default: 0.5)
        "candidate_spacing_mult": _env_or_default(
            "EC7_CANDIDATE_SPACING_MULT", 0.5, float
        ),
        # ═══════════════════════════════════════════════════════════════════
        # TEST POINT SPACING (as multiplier of max_spacing_m)
        # ═══════════════════════════════════════════════════════════════════
        # Controls the grid density for coverage verification test points.
        # Expressed as a fraction of max_spacing_m for automatic scaling.
        #
        # - 0.10: Dense verification (10m at 100m max_spacing) - very precise
        # - 0.20: Standard verification (20m at 100m max_spacing) - RECOMMENDED
        # - 0.30: Sparse verification (30m at 100m max_spacing) - faster
        #
        # NOTE: Test points use HEXAGONAL grid (same as candidate grid).
        # This provides optimal coverage verification with uniform point density.
        # ENV OVERRIDE: EC7_TEST_SPACING_MULT (float, default: 0.2)
        "test_spacing_mult": _env_or_default("EC7_TEST_SPACING_MULT", 0.2, float),
        # Maximum solver time in seconds before falling back to greedy
        "time_limit_s": 120,
        # MIP gap tolerance (0.01 = allow 1% suboptimality for faster solve)
        "mip_gap": 0.05,
        # ═══════════════════════════════════════════════════════════════════
        # STALL DETECTION (TIME-BASED EARLY TERMINATION)
        # ═══════════════════════════════════════════════════════════════════
        # Detects when the solver is making no meaningful progress by comparing
        # gap values over time using kCallbackMipInterrupt.
        #
        # TECHNICAL BACKGROUND:
        # HiGHS has two types of progress:
        #   1. Primal progress: Finding better solutions (BestSol improves)
        #   2. Dual progress: Proving optimality (BestBound tightens)
        #
        # The kCallbackMipImprovingSolution callback only fires for primal
        # progress. When solver stalls on proving optimality, no new solutions
        # are found and that callback never fires - leading to false "no stall"
        # detection.
        #
        # IMPLEMENTATION:
        # Uses kCallbackMipInterrupt with time-based sampling (~1 second intervals).
        # This captures BOTH primal and dual progress via the mip_gap field.
        #
        # How it works:
        # 1. Wait for solver "warmup" (skip first warmup_seconds)
        # 2. Every ~1 second, compare current gap to gap from comparison_seconds ago
        # 3. If the absolute gap difference is < min_improvement_pct points,
        #    solver is "stalled" and terminates with current best solution
        #
        # Example with defaults (warmup=5s, comparison=5s, improvement=5):
        # - At t=15s: gap 11.3%, compared to t=10s gap 11.8%
        # - Improvement: 11.8 - 11.3 = 0.5 points < 5 → STALL DETECTED
        # - Solver terminates, saving ~100s of proving optimality
        #
        # Key difference from gap threshold (mip_rel_gap):
        # - mip_rel_gap: "Stop when gap reaches X%" (target-based)
        # - Stall detection: "Stop when gap hasn't dropped by Y points
        #   over the last Z seconds" (progress-based)
        #
        "stall_detection": {
            # Master switch for stall detection
            # ENV OVERRIDE: EC7_STALL_ENABLED (bool, default: true)
            "enabled": _env_or_default("EC7_STALL_ENABLED", True, bool),
            # ═══════════════════════════════════════════════════════════════
            # GAP THRESHOLD (WHEN TO START STALL DETECTION)
            # ═══════════════════════════════════════════════════════════════
            # Stall detection only activates once the gap drops BELOW this value.
            # This prevents premature termination while the solver is still making
            # rapid progress on high-gap solutions.
            #
            # Example: gap_threshold_pct=15 means:
            #   - Gap at 20%: Stall detection NOT active (still searching)
            #   - Gap at 12%: Stall detection ACTIVE (now monitoring progress)
            #
            # ENV OVERRIDE: EC7_STALL_GAP_THRESHOLD (float, default: 15.0)
            "gap_threshold_pct": _env_or_default(
                "EC7_STALL_GAP_THRESHOLD", 15.0, float
            ),
            # Warmup period in seconds AFTER gap threshold is reached
            # Gives solver time to stabilize before checking for stalls
            # ENV OVERRIDE: EC7_STALL_WARMUP_S (float, default: 15.0)
            "warmup_seconds": _env_or_default("EC7_STALL_WARMUP_S", 15.0, float),
            # Time window in seconds for comparison (compare current gap to N seconds ago)
            # Higher values = more tolerance for slow progress
            # ENV OVERRIDE: EC7_STALL_WINDOW_S (float, default: 10.0)
            "comparison_seconds": _env_or_default("EC7_STALL_WINDOW_S", 10.0, float),
            # Minimum ABSOLUTE gap improvement (percentage points) required
            # If gap hasn't dropped by at least this many points, solver stalls
            # Example: 5.0 means gap must drop from 12% to 7% (5 point drop)
            # ENV OVERRIDE: EC7_STALL_MIN_IMPROVEMENT (float, default: 5.0)
            "min_improvement_pct": _env_or_default(
                "EC7_STALL_MIN_IMPROVEMENT", 5.0, float
            ),
            # ═══════════════════════════════════════════════════════════════
            # UNIFIED SETTINGS: Apply to CZRC (second pass) as well?
            # ═══════════════════════════════════════════════════════════════
            # When True: CZRC second pass uses these SAME settings (ignores
            #            czrc_optimization.ilp.stall_detection values)
            # When False: CZRC uses its own settings from czrc_optimization.ilp
            # ENV OVERRIDE: EC7_STALL_APPLY_TO_CZRC (bool, default: true)
            "apply_to_czrc": _env_or_default("EC7_STALL_APPLY_TO_CZRC", True, bool),
        },
        # Number of threads for ILP solver (keep at 1 for single-threaded solving)
        # IMPORTANT: Do NOT increase this value. Parallelization should happen at a
        # higher level (e.g., parallel filter combinations) rather than within the ILP
        # solver. Using threads=1 allows better resource utilization when running
        # multiple ILP instances in parallel across different filter scenarios.
        # BENCHMARKED 2026-01-08: W=14×T=1 is optimal. Hybrid configs are 31-87% slower.
        "threads": 1,
        # Verbose mode for ILP solver progress output
        # 0 = silent (default), 1 = progress output, 2 = detailed debug
        # Set to 1 or 2 to see solver progress during long runs
        "verbose": 1,
        # ═══════════════════════════════════════════════════════════════════
        # MIP HEURISTIC EFFORT (HiGHS solver parameter)
        # ═══════════════════════════════════════════════════════════════════
        # Fraction of solver effort devoted to primal heuristics (finding
        # feasible solutions). Higher values find solutions faster but may
        # sacrifice optimality proof. Range: 0.0 to 1.0
        #
        # - 0.05: Default (balanced)
        # - 0.10: Low (2× default)
        # - 0.15: Medium (3× default, often optimal for set cover)
        # - 0.30: High (6× default)
        # - 0.50: Aggressive (10× default, prioritizes feasibility)
        #
        # For borehole placement (set cover with many near-equivalent optima),
        # higher values often improve performance.
        # ENV OVERRIDE: EC7_HEURISTIC_EFFORT (float 0.0-1.0)
        "mip_heuristic_effort": _env_or_default("EC7_HEURISTIC_EFFORT", 0.05, float),
        # DEPRECATED: Use optimization.solver_mode instead
        # Kept for backwards compatibility only
        "use_ilp": True,
        # Minimum gap area to consider (m²) - smaller gaps are ignored
        "min_gap_area_m2": 100.0,
        # Fill remaining fragments after ILP solve (adds boreholes to cover small gaps)
        # Set to False to use only the ILP solution without fragment fill
        "fill_remaining_fragments": False,
        # ═══════════════════════════════════════════════════════════════════
        # 🔷 CONFLICT CONSTRAINTS (CRITICAL FOR LARGE GAPS)
        # ═══════════════════════════════════════════════════════════════════
        # Without conflict constraints, standard set cover allows selecting many
        # nearby candidates with massive overlap, causing 20× overkill for large gaps.
        #
        # Conflict constraints force the solver to choose between nearby candidates,
        # naturally producing sparse hexagonal-like layouts.
        #
        # Enable conflict constraints for sparse hexagonal-like layout
        "use_conflict_constraints": True,
        # ═══════════════════════════════════════════════════════════════════
        # CONSTRAINT MODE: "clique" (RECOMMENDED) or "pairwise" (legacy)
        # ═══════════════════════════════════════════════════════════════════
        # Clique constraints are mathematically stronger than pairwise:
        # - Replace O(k²) pairwise constraints with O(1) clique per neighborhood
        # - LP bound improvement: 71-85% for typical clique sizes (7-13)
        # - Constraint reduction: 55× observed on hexagonal grids
        #
        # Pairwise constraints (legacy): each pair (j,k) gets x[j]+x[k] <= 1
        # - Simpler but weaker LP bounds
        # - More constraints (7000+ vs 1400 for typical problems)
        # ENV OVERRIDE: EC7_CONSTRAINT_MODE ("clique" or "pairwise")
        "conflict_constraint_mode": _env_or_default("EC7_CONSTRAINT_MODE", "pairwise"),
        # Exclusion factor: minimum separation as multiple of max_spacing (200m)
        #
        # KEY INSIGHT: Lower factor = more overlap = MORE optimizer flexibility
        # = typically FEWER boreholes needed (counter-intuitive but correct!)
        #
        # The optimizer can place boreholes closer together, allowing it to
        # cover edges and corners more efficiently without needing extra boreholes.
        #
        # - 0.6: Very aggressive (120m), max flexibility, typically fewest BHs
        # - 0.7: Aggressive (140m), good for irregular shapes
        # - 0.8: Balanced (160m), reliable coverage - RECOMMENDED
        # - 1.0: Circles just touch (200m), reduced flexibility
        # - >1.0: Gaps between circles, may need MORE boreholes to fill gaps
        #
        # See: _reports/exclusion_factor_explained.md for visual explanation
        # ENV OVERRIDE: EC7_EXCLUSION_FACTOR (float)
        "exclusion_factor": _env_or_default("EC7_EXCLUSION_FACTOR", 0.9, float),
        # ═══════════════════════════════════════════════════════════════════
        # CROSS-ZONE EXCLUSION METHOD (for multi-zone optimization)
        # ═══════════════════════════════════════════════════════════════════
        # When candidates span multiple zones with different max_spacing values,
        # determines how to calculate the exclusion distance for conflict constraints.
        #
        # Methods:
        # - "min": Use minimum zone spacing (strictest - prevents ANY close pairs)
        #          exclusion_dist = exclusion_factor × min(zone_spacings)
        #          Result: Conservative, may over-constrain larger-spacing zones
        #
        # - "max": Use maximum zone spacing (most permissive for cross-zone pairs)
        #          exclusion_dist = exclusion_factor × max(zone_spacings)
        #          Result: Allows cross-zone boreholes to be closer together,
        #          better for zones with different spacing requirements
        #
        # - "average": Use average of zone spacings (balanced approach)
        #              exclusion_dist = exclusion_factor × mean(zone_spacings)
        #              Result: Middle ground, suitable for mixed scenarios
        #
        # Note: For same-zone pairs, always uses that zone's spacing regardless
        # of this setting. This setting only affects CROSS-ZONE conflict pairs.
        #
        # ENV OVERRIDE: EC7_CROSS_ZONE_EXCLUSION_METHOD ("min", "max", "average")
        "cross_zone_exclusion_method": _env_or_default(
            "EC7_CROSS_ZONE_EXCLUSION_METHOD", "average"
        ),
        # ═══════════════════════════════════════════════════════════════════
        # CLIQUE MODE SETTINGS (only used when conflict_constraint_mode="clique")
        # ═══════════════════════════════════════════════════════════════════
        # Minimum clique size to include as constraint
        # 3 = include triangles, 4+ = only larger groups
        "min_clique_size": 3,
        # Maximum cliques to enumerate (memory/time safety limit)
        "max_cliques": 50000,
        # ═══════════════════════════════════════════════════════════════════
        # CONNECTED COMPONENTS DECOMPOSITION (SPEEDUP FOR DISTRIBUTED GAPS)
        # ═══════════════════════════════════════════════════════════════════
        # Decomposes gaps into independent subproblems based on geometry.
        # Gaps that cannot share a borehole (distance > 2 × max_spacing) are
        # solved separately, then results are merged.
        #
        # Benefits:
        # - 2-10× speedup for spatially distributed gap configurations
        # - Better scaling with number of gaps
        # - Each component optimizes independently (smaller ILP instances)
        #
        # Technical: Uses scipy.sparse.csgraph.connected_components with
        # connectivity_threshold = 2 × max_spacing (400m for 200m radius)
        # ENV OVERRIDE: EC7_CONNECTED_COMPS ("true" or "false")
        "use_connected_components": _env_bool("EC7_CONNECTED_COMPS", False),
        # Minimum gaps per component to use ILP (smaller → greedy fallback)
        "min_component_gaps_for_ilp": 1,
        # ═══════════════════════════════════════════════════════════════════
        # ZONE-BASED DECOMPOSITION (HIERARCHICAL SPLITTING)
        # ═══════════════════════════════════════════════════════════════════
        # Splits large gaps by zone boundaries BEFORE applying connected components.
        # This is critical for problems where a single gap spans multiple zones,
        # causing 10K+ variables that exceed solver capacity.
        #
        # Benefits:
        # - Dramatic complexity reduction (N-fold problem → N smaller problems)
        # - Enables parallelization across zones
        # - Aligns with engineering practice (per-zone investigation programs)
        #
        # Trade-offs:
        # - May produce 5-15% more boreholes than global optimum
        # - Boreholes near zone boundaries won't share coverage cross-zone
        #
        # When enabled, the decomposition hierarchy is:
        # 1. Split gaps by zone boundaries (use_zone_decomposition)
        # 2. Within each zone, split by connected components (use_connected_components)
        # 3. Solve ILP for each resulting subproblem
        #
        # ENV OVERRIDE: EC7_ZONE_DECOMP ("true" or "false")
        "use_zone_decomposition": _env_bool("EC7_ZONE_DECOMP", True),
        # Minimum gap area within a zone to include (filters boundary slivers)
        # Slivers can occur when gaps exactly follow zone boundaries
        "min_zone_gap_area_m2": 100.0,
        # ═══════════════════════════════════════════════════════════════════
        # PAIRWISE MODE SETTINGS (only used when conflict_constraint_mode="pairwise")
        # ═══════════════════════════════════════════════════════════════════
        # Maximum conflict pairs before truncation (memory/performance safety)
        # CRITICAL: If truncation occurs, the optimization model is INCOMPLETE
        # and will produce suboptimal results.
        "max_conflict_pairs": 200000,
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🔄 GREEDY SOLVER CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    # Settings for greedy disk cover heuristic (used when solver_mode="greedy")
    # Greedy is faster but produces ~10-30% more boreholes than optimal
    "greedy_solver": {
        # Maximum iterations before stopping (safety limit)
        "max_iterations": 1000,
        # Minimum area gain (m²) to continue adding boreholes
        # CRITICAL: This must be proportional to borehole coverage area!
        # With 200m radius, borehole covers ~125,664 m² (12.57 ha)
        # Value of 1.0 m² causes catastrophic over-placement (adding boreholes
        # for tiny 1m² fragments, resulting in 20-50× too many boreholes!)
        #
        # Recommended values:
        #   - 6,283 m² (0.63 ha) = 5% efficiency threshold (strict)
        #   - 12,566 m² (1.26 ha) = 10% efficiency threshold (moderate)
        #   - 31,416 m² (3.14 ha) = 25% efficiency threshold (lenient)
        #
        # Formula: min_gain ≈ π × radius² × efficiency_threshold
        #          For 200m radius, 10% threshold: π × 200² × 0.10 = 12,566 m²
        "min_coverage_gain_m2": 10000.0,  # ~8% of borehole area (~1.0 ha)
        # Minimum efficiency percentage (coverage_gain / borehole_area × 100)
        # This provides a proportional stopping criterion that adapts to radius.
        # The algorithm uses the STRICTER of min_coverage_gain_m2 and this.
        # Recommended: 5-15% (lower = more thorough, higher = faster)
        "min_efficiency_pct": 8.0,  # Stop when best coverage < 8% of borehole area
        # Candidate grid uses the same spacing as ILP (computed from multiplier)
        # See ilp_solver.candidate_spacing_mult for configuration
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🔧 BORDER CONSOLIDATION (SECOND PASS)
    # ═══════════════════════════════════════════════════════════════════════
    # After zone decomposition merges per-zone solutions, a second pass can
    # remove redundant boreholes near zone boundaries where adjacent zones
    # placed overlapping coverage.
    #
    # Modes:
    #   - "disabled": No second pass (current behavior)
    #   - "ilp": ILP re-optimization using proposed boreholes as candidates
    #   - "buffer_zone": Lock interior boreholes, re-optimize buffer region only
    #                    (recommended for multi-zone with different spacing)
    #
    # ENV OVERRIDE: EC7_BORDER_CONSOLIDATION_MODE ("disabled", "ilp", "buffer_zone")
    "border_consolidation": {
        # Master toggle - mode selection
        # "disabled": Skip second pass (fastest, current behavior)
        # "ilp": ILP solve using first-pass boreholes as candidates
        # "buffer_zone": Lock interior, re-solve buffer only (recommended)
        "mode": _env_or_default("EC7_BORDER_CONSOLIDATION_MODE", "disabled"),
        # ═══════════════════════════════════════════════════════════════════
        # ILP SECOND PASS SETTINGS (shared by "ilp" and "buffer_zone" modes)
        # ═══════════════════════════════════════════════════════════════════
        # Time limit for second-pass ILP solve (seconds)
        # Since candidates are just proposed boreholes (~100 vs ~5000),
        # solve is very fast - typically <1 second
        "time_limit": 90,
        # MIP gap tolerance - can be looser since problem is smaller
        "mip_gap": 0.03,
        # Coverage target percentage (97% allows for minor gaps)
        "coverage_target_pct": 100.0,
        # Use conflict constraints to prevent boreholes too close together
        "use_conflict_constraints": True,
        # Exclusion factor for conflict constraints
        # 0.8 = boreholes must be at least 80% of max_spacing apart
        "exclusion_factor": 0.9,
        # Logging verbosity for HiGHS solver
        # 0 = silent, 1 = summary, 2+ = detailed
        "verbose": 1,
        # ═══════════════════════════════════════════════════════════════════
        # BUFFER ZONE MODE SETTINGS (only used when mode="buffer_zone")
        # ═══════════════════════════════════════════════════════════════════
        # Buffer width as multiple of max_spacing
        # 1.5 = 1.5 × max_spacing from zone boundaries classifies as "buffer"
        # Interior boreholes (outside buffer) are locked and not re-optimized
        "buffer_width_factor": 3,
        # Generate fresh candidates in buffer zone (vs using first-pass only)
        # True = dense grid at min zone spacing for guaranteed feasibility
        # False = use only first-pass boreholes as candidates (faster but may fail)
        "use_fresh_candidates": True,
        # Candidate grid spacing multiplier for buffer zone (relative to min zone spacing)
        # 0.25 = 4 candidates per diameter (dense for high coverage)
        "buffer_candidate_spacing_mult": 0.25,
        # ═══════════════════════════════════════════════════════════════════
        # REGION SPLITTING SETTINGS (buffer_zone mode optimization)
        # ═══════════════════════════════════════════════════════════════════
        # Enable splitting disconnected buffer regions into independent ILPs
        # When True: Regions separated by 2× max zone spacing are solved separately
        # This reduces ILP complexity when buffer zones are far apart
        "splitting_enabled": True,
        # DEBUG: Show buffer zone outline on map (for testing buffer width)
        # This adds a magenta dashed outline showing the buffer zone polygon
        "show_buffer_zone_outline": False,
        # DEBUG: Show split region boundaries on map (for testing region splitting)
        # This adds colored dashed outlines showing independent consolidation regions
        # Each region is solved with a separate ILP (faster than one large ILP)
        "show_split_regions": False,  # Set to True to visualize split regions
        # ═══════════════════════════════════════════════════════════════════
        # CROSS-ZONE REACHABILITY CONSOLIDATION (CZRC) - VISUALIZATION
        # ═══════════════════════════════════════════════════════════════════
        # CZRC replaces the buffer strip approach with precise geometric computation.
        # Instead of buffering zone borders, it computes coverage cloud intersections
        # to identify exactly where cross-zone boreholes are possible.
        #
        # CZRC Two-Tier Expansion Architecture:
        # - Tier 1 (CZRC + 1×R_max): Test points + candidate grid + first-pass BHs as candidates
        # - Tier 2 (CZRC + tier2_rmax_multiplier×R_max): First-pass BHs as locked constants
        # where R_max = max(zone spacings) for each pairwise CZRC region
        "czrc_tier2_rmax_multiplier": 2.5,  # Multiplier for Tier 2 expansion (default: 2.0)
        #
        # Show expanded ILP visibility regions (tier2_rmax_multiplier × R_max expansion boundaries)
        # Displays: Tier 1 (candidate grid boundary) and Tier 2 (visibility boundary) as lines
        "show_czrc_ilp_visibility": True,
        # Colors for CZRC visualization (Zone Overlap checkbox controls visibility)
        "czrc_cloud_opacity": 0.20,  # Alpha for zone coverage clouds (was 0.15, too faint)
        "czrc_cell_cloud_opacity": 0.20,  # Alpha for cell coverage clouds (higher for visibility)
        "czrc_pairwise_color": "cyan",  # Color for pairwise intersections
        "czrc_pairwise_opacity": 0.40,  # Higher opacity for overlap regions (more visible than fills)
        "czrc_line_width": 2,
        # CZRC candidate grid styling (hexagonal cells - same style as second_pass_grid)
        "czrc_grid": {
            "color": "rgba(68, 67, 65, 1)",  # Dark orange - distinct from second pass grid
            "line_width": 0.5,  # Thin lines matching second_pass_grid style
        },
        # CZRC ILP visibility boundary styling (tier 2 boundary - controlled by CZRC Grid checkbox)
        "czrc_ilp_visibility": {
            "tier2_color": "rgba(138, 43, 226, 0.8)",  # BlueViolet for Tier 2 (visibility boundary)
            "tier2_dash": "longdash",  # Long dash for Tier 2
            "line_width": 2,
        },
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🔗 CZRC OPTIMIZATION (NEW SECOND PASS - REPLACES BORDER CONSOLIDATION)
    # ═══════════════════════════════════════════════════════════════════════
    # CZRC (Cross-Zone Redundancy Check) uses precise coverage cloud intersections
    # to identify and optimize cross-zone borehole placement. This is the
    # recommended replacement for border_consolidation.
    #
    # Architecture:
    # - Tier 1 (CZRC + tier1_mult × R_max): Active optimization region
    #   - Test points filtered to this region
    #   - Fresh candidate grid generated here
    #   - First-pass boreholes here are ILP CANDIDATES
    # - Tier 2 (CZRC + tier2_mult × R_max): Coverage context region
    #   - First-pass boreholes here are LOCKED CONSTANTS
    #   - Provides pre-coverage for Tier 1 test points
    #
    # R_max = max(zone spacings) for each pairwise CZRC region
    "czrc_optimization": {
        # Master switch - enables CZRC geometry computation (for visualization)
        # When True: Computes CZRC regions and shows Zone Overlap layer in HTML
        # When False: Skips CZRC entirely (no visualization, no optimization)
        "enabled": True,  # Set True to enable (requires border_consolidation.mode="disabled")
        # ═══════════════════════════════════════════════════════════════════
        # SKIP ILP OPTIMIZATION (visualization only mode)
        # ═══════════════════════════════════════════════════════════════════
        # When True: Computes and visualizes CZRC regions BUT skips the ILP solver
        # Use this when you want Zone Overlap visualization without the slow optimization
        # Useful for: debugging shapefile overlaps, testing zone configurations
        "skip_ilp": False,  # Set True to show visualization but skip optimization
        # ═══════════════════════════════════════════════════════════════════
        # TIER EXPANSION MULTIPLIERS (override with EC7_TIER1_RMAX_MULT, EC7_TIER2_RMAX_MULT)
        # ═══════════════════════════════════════════════════════════════════
        # Tier 1: Active optimization region (test points + candidates + BH candidates)
        "tier1_rmax_multiplier": _env_or_default(
            "EC7_TIER1_RMAX_MULT", 1.0, float
        ),  # Tier 1 = CZRC + mult × R_max
        # Tier 2: Coverage context region (locked boreholes provide pre-coverage)
        "tier2_rmax_multiplier": _env_or_default(
            "EC7_TIER2_RMAX_MULT", 2.0, float
        ),  # Tier 2 = CZRC + mult × R_max
        # ═══════════════════════════════════════════════════════════════════
        # CANDIDATE GRID SETTINGS
        # ═══════════════════════════════════════════════════════════════════
        # Candidate grid spacing as multiple of min zone spacing in the pair
        # 0.5 = 2 candidates per coverage diameter (balanced density)
        "candidate_grid_spacing_mult": 0.5,
        # ═══════════════════════════════════════════════════════════════════
        # TIER 2 TEST POINT PROTECTION (prevents Tier 2 coverage gaps)
        # ═══════════════════════════════════════════════════════════════════
        # When enabled, CZRC generates sparse test points in Tier 2 zones to
        # prevent removing boreholes that are the sole coverage for Tier 2.
        # This protects cross-tier coverage from inadvertent removal.
        #
        # The Tier 2 test point multiplier is relative to Tier 1's test_spacing_mult:
        #   Tier 2 spacing = test_spacing_mult × tier2_test_spacing_multiplier × max_spacing
        #   e.g., 0.2 × 3.0 × 200m = 120m grid spacing for 200m-spaced zones
        "tier2_test_point_protection": {
            "enabled": True,  # Enable Tier 2 test point constraints in ILP
            "tier2_test_spacing_multiplier": 3.0,  # 3× sparser than Tier 1 test points
        },
        # ═══════════════════════════════════════════════════════════════════
        # CZRC RESULT CACHING (Intra-run caching for ILP results)
        # ═══════════════════════════════════════════════════════════════════
        # When enabled, CZRC caches ILP solutions based on problem definition:
        # - Tier 1 geometry (determines candidate grid)
        # - Zone spacings (determines spacing constraints)
        # - Unsatisfied test points (the actual coverage requirements)
        #
        # Different first-pass borehole sets may produce the SAME unsatisfied
        # test points after pre-coverage computation. Caching allows reuse.
        #
        # Expected hit rate: 20-50% depending on filter combination diversity
        # Time savings: 10-30s per cache hit (skips full ILP solve)
        "cache_enabled": False,  # DISABLED for exclusion factor study
        "cache_lock_timeout_s": 600.0,  # Lock timeout in seconds (10 min default)
        # ═══════════════════════════════════════════════════════════════════
        # ILP SOLVER SETTINGS
        # ═══════════════════════════════════════════════════════════════════
        "ilp": {
            # Time limit per pairwise region (seconds)
            "time_limit": 90,
            # MIP gap tolerance
            "mip_gap": 0.03,
            # Coverage target (100% = must cover all test points)
            "coverage_target_pct": 100.0,
            # Use conflict constraints to prevent boreholes too close
            "use_conflict_constraints": True,
            # Exclusion factor (0.9 = 90% of min spacing between boreholes)
            # ENV OVERRIDE: EC7_CZRC_EXCLUSION_FACTOR (float, default: 0.9)
            "exclusion_factor": _env_or_default(
                "EC7_CZRC_EXCLUSION_FACTOR", 0.9, float
            ),
            # Cross-zone exclusion method: "min", "max", or "average"
            # Controls how exclusion distance is calculated for cross-zone pairs.
            # Inherits from ilp_solver.cross_zone_exclusion_method by default.
            # See ilp_solver.cross_zone_exclusion_method for detailed documentation.
            # ENV OVERRIDE: EC7_CZRC_CROSS_ZONE_EXCLUSION_METHOD
            "cross_zone_exclusion_method": _env_or_default(
                "EC7_CZRC_CROSS_ZONE_EXCLUSION_METHOD", None
            ),  # None = inherit from ilp_solver
            # ═══════════════════════════════════════════════════════════════
            # STALL DETECTION (TIME-BASED EARLY TERMINATION)
            # ═══════════════════════════════════════════════════════════════
            # CZRC stall detection settings. By default, uses the SAME settings
            # as first-pass (ilp_solver.stall_detection) when apply_to_czrc=True.
            #
            # These settings are ONLY used when ilp_solver.stall_detection.apply_to_czrc=False
            # See ilp_solver.stall_detection for detailed documentation.
            "stall_detection": {
                # Master switch for CZRC stall detection
                # NOTE: Ignored when ilp_solver.stall_detection.apply_to_czrc=True
                # ENV OVERRIDE: EC7_CZRC_STALL_ENABLED (bool, default: true)
                "enabled": _env_or_default("EC7_CZRC_STALL_ENABLED", True, bool),
                # Warmup period in seconds before checking for stalls
                # NOTE: Ignored when ilp_solver.stall_detection.apply_to_czrc=True
                # ENV OVERRIDE: EC7_CZRC_STALL_WARMUP_S (float, default: 5.0)
                "warmup_seconds": _env_or_default(
                    "EC7_CZRC_STALL_WARMUP_S", 5.0, float
                ),
                # Time window in seconds for comparison
                # NOTE: Ignored when ilp_solver.stall_detection.apply_to_czrc=True
                # ENV OVERRIDE: EC7_CZRC_STALL_WINDOW_S (float, default: 10.0)
                "comparison_seconds": _env_or_default(
                    "EC7_CZRC_STALL_WINDOW_S", 10.0, float
                ),
                # Minimum ABSOLUTE gap improvement (percentage points) required
                # NOTE: Ignored when ilp_solver.stall_detection.apply_to_czrc=True
                # ENV OVERRIDE: EC7_CZRC_STALL_MIN_IMPROVEMENT (float, default: 5.0)
                "min_improvement_pct": _env_or_default(
                    "EC7_CZRC_STALL_MIN_IMPROVEMENT", 5.0, float
                ),
            },
            # Logging verbosity (0=silent, 1=summary, 2+=detailed)
            "verbose": 1,
        },
        # ═══════════════════════════════════════════════════════════════════
        # CELL SPLITTING (Large CZRC Region Decomposition)
        # ═══════════════════════════════════════════════════════════════════
        # When enabled, splits large CZRC regions into cells before ILP
        # solving. This prevents HiGHS solver stalls on regions with >400-500
        # candidates.
        #
        # Trigger: If unified_tier1.area > max_area_for_direct_ilp_m2, split.
        # Each cell is processed independently using the existing
        # solve_czrc_ilp_for_cluster() function, then results are merged.
        #
        # Methods:
        #   - "kmeans_voronoi": K-means clustering + Voronoi tessellation
        #     Produces balanced, geometry-aware cells. RECOMMENDED.
        #   - "grid": Fixed-size grid (legacy fallback)
        "cell_splitting": {
            # Master switch for cell splitting
            "enabled": True,
            # Maximum area (m²) before triggering cell split
            # 1,000,000 m² = 1 km² (approx 400-500 candidates at typical density)
            "max_area_for_direct_ilp_m2": 1_000_000,
            # Splitting method: "kmeans_voronoi" (recommended) or "grid" (legacy)
            "method": "kmeans_voronoi",
            # Minimum cell area to process (skip tiny slivers)
            "min_cell_area_m2": 100,
            # === Grid method settings (legacy fallback) ===
            "grid": {
                "cell_size_m": 2000,
            },
            # === K-means + Voronoi settings (recommended) ===
            "kmeans_voronoi": {
                # Target average cell area (determines number of cells)
                # K = ceil(region_area / target_cell_area)
                "target_cell_area_m2": 1_000_000,  # 1 km² per cell
                # Minimum number of cells (even for small regions)
                "min_cells": 2,
                # Maximum number of cells (safety cap)
                "max_cells": 50,
                # Random seed for K-means reproducibility
                "random_state": 42,
            },
        },
        # ═══════════════════════════════════════════════════════════════════
        # 🔗 THIRD PASS: CELL-CELL BOUNDARY CONSOLIDATION
        # ═══════════════════════════════════════════════════════════════════
        # After splitting large clusters into Voronoi cells and solving each
        # cell independently, there may be redundant boreholes at cell boundaries.
        # This third pass applies the same CZRC methodology to cell-cell boundaries:
        #
        # 1. Compute coverage cloud for each cell
        # 2. Find pairwise intersections (cell-cell CZRC regions)
        # 3. Re-optimize each intersection to remove redundant boreholes
        #
        # This is mathematically identical to zone-zone CZRC but operates on
        # cells within the same cluster (uniform spacing).
        #
        # ILP SETTINGS INHERITANCE: Third Pass automatically inherits ILP settings
        # (exclusion_factor, coverage_target_pct, etc.) from czrc_optimization.ilp
        # unless explicitly overridden in this section's "ilp" subsection.
        "cell_boundary_consolidation": {
            # Master switch for third pass cell-cell CZRC
            "enabled": True,
            # Tier 1 multiplier (same as zone CZRC - inherit from parent)
            "tier1_rmax_multiplier": 1.0,
            # Tier 2 multiplier (same as zone CZRC - inherit from parent)
            "tier2_rmax_multiplier": 2.0,
            # Test point spacing multiplier (same as zone CZRC)
            "test_spacing_mult": 0.2,
            # NOTE: ILP settings (exclusion_factor, coverage_target_pct, etc.)
            # are inherited from czrc_optimization.ilp section automatically.
            # Only add an "ilp" subsection here to override specific settings.
        },
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🎚️ FILTER UI SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    # Depth slider step size in meters (also controls filter combination count)
    # Lower values = more granular filtering but more pre-computed combinations
    # Example: 50m step with max depth 100m = 3 depth values (0, 50, 100)
    #          × 8 checkbox combos = 24 total filter combinations
    "depth_slider_step_m": 50.0,
    # ═══════════════════════════════════════════════════════════════════════
    # 🧮 OPTIMIZATION SOLVER SELECTION
    # ═══════════════════════════════════════════════════════════════════════
    # Controls which algorithm is used for borehole placement optimization.
    #
    # solver_mode options:
    #   - "ilp": Use Integer Linear Programming (optimal solution, requires solver)
    #            Falls back to greedy automatically if ILP fails/times out
    #            This is the recommended mode - always tries for optimal first
    #   - "greedy": Use greedy disk cover heuristic (fast, approximate solution)
    #               No ILP attempt, pure greedy optimization
    #               Use only when you explicitly want to skip ILP entirely
    #
    # NOTE: "auto" mode was removed. ILP with greedy fallback provides the
    # best balance of quality and reliability for both single and parallel runs.
    #
    "optimization": {
        "solver_mode": "ilp",  # "ilp" | "greedy"
    },
    # ═══════════════════════════════════════════════════════════════════════
    # ⚡ PARALLEL PROCESSING
    # ═══════════════════════════════════════════════════════════════════════
    # Settings for parallel pre-computation of coverage for all filter combinations
    "parallel": {
        # Master toggle - set False to use n_jobs=1 (sequential execution)
        "enabled": True,
        # Number of worker processes (-1 = auto, based on CPU cores)
        "max_workers": -1,
        # Default optimal worker count when auto-detecting
        # Benchmark results (50m step, 24 combos): 10w=43s, 12w=47s, 14w=41s, 18w=42s
        "optimal_workers_default": 14,
        # Minimum combinations needed to justify parallel overhead
        "min_combinations_for_parallel": 10,
        # Timeout per combination (seconds) - coverage is faster than variogram
        "timeout_per_combo_seconds": 60,
        # Fall back to sequential on any parallel error
        "fallback_on_error": True,
        # Joblib backend ("loky" = process-based, safe for CPU-bound)
        "backend": "loky",
        # Verbosity level for progress output (0-10)
        "verbose": 10,
        # ═══════════════════════════════════════════════════════════════════
        # 🗂️ INTRA-RUN ZONE CACHE
        # ═══════════════════════════════════════════════════════════════════
        # Cache zone-level ILP solutions to avoid redundant solves within a run.
        # Multiple filter combinations often produce identical zone-gap geometries,
        # especially when boreholes span multiple depth thresholds.
        #
        # When enabled (production mode only):
        # - Creates temp directory shared across all workers
        # - Uses per-key file locking for race-condition-free access
        # - Automatically cleaned up when run completes
        #
        # Performance impact: 30-50% reduction in solve time for 80 filter combos
        #
        # NOTE: Zone cache is automatically DISABLED in testing mode regardless
        # of this setting (testing_mode.enabled = True disables zone cache).
        "zone_cache_enabled": True,
    },
    # Solver/ILP settings for parallel workers (reduced timeouts for faster throughput)
    # These override the main solver settings when running in parallel context
    "parallel_solver_overrides": {
        # Override solver_mode for parallel (None = use main setting)
        # Set to "greedy" for maximum parallel throughput
        "solver_mode": None,  # None | "ilp" | "greedy"
        # Reduced timeout per combination (vs 120s for single run)
        "time_limit_s": 60,
        # Looser tolerance for faster termination (vs 0.03)
        "mip_gap": 0.05,
    },
    # Backwards compatibility alias (DEPRECATED - use parallel_solver_overrides)
    "parallel_ilp_overrides": {
        # Reduced timeout per combination (vs 120s for single run)
        "time_limit_s": 60,
        # Looser tolerance for faster termination (vs 0.03)
        "mip_gap": 0.05,
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 💾 CACHE SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    # Settings for caching precomputed coverage results
    # Cache eliminates 5-minute wait when only testing HTML/visualization changes
    "cache": {
        # Master toggle - set False to always recompute
        "enabled": True,
        # Force overwrite - recompute and overwrite existing cache even if valid
        # Use when you've changed algorithm implementation details (ILP logic, etc.)
        # that aren't captured in the cache fingerprint
        "force_overwrite": True,
        # Cache storage directory (relative to workspace root)
        "cache_dir": "Gap_Analysis_EC7/cache",
        # Maximum cache entries to keep (oldest are pruned)
        "max_cache_entries": 10,
        # Auto-expire cache entries older than this
        "max_cache_age_days": 30,
        # Log cache hits/misses
        "log_cache_hits": True,
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 📂 FILE PATHS (Rarely changed - at bottom)
    # ═══════════════════════════════════════════════════════════════════════
    # MODIFICATION POINT: Update these paths for your project
    "file_paths": {
        "embankment_shp": "LocationGroupsZones/LocationGroupsZones.shp",
        "boreholes_csv": "Openground CSVs/GIR Location Group/Location Details.csv",
        "output_dir": "Gap_Analysis_EC7/Output",
        "log_dir": "Gap_Analysis_EC7/logs",
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🎨 VISUALIZATION SETTINGS (Rarely changed - at bottom)
    # ═══════════════════════════════════════════════════════════════════════
    "visualization": {
        # Panel visibility toggles
        "show_coverage_stats_panel": False,  # Show/hide coverage statistics panel in HTML
        # Coverage zone colors (buffer union visualization)
        # NOTE: Coverage must have high opacity (0.85+) to properly cover gaps
        # underneath due to Plotly's inability to render polygon holes as transparent.
        # Gaps renders first (underneath), coverage renders second (on top).
        "coverage_colors": {
            "covered": "rgba(95, 235, 95, 0.85)",  # Green with 85% opacity (must be high to cover gaps)
            "gap": "rgba(255, 255, 255, 0.407)",  # Grey with 49% opacity for uncovered gaps
        },
        # Grid cell styling (legacy - kept for compatibility)
        "grid_line_color": "rgba(0, 0, 0, 0.3)",
        "grid_line_width": 1.0,
        # Borehole marker styling (existing boreholes)
        "borehole_marker": {
            "size": 6,
            "color": "rgba(0, 0, 0, 0.82)",  # Black with 82% opacity
            "symbol": "circle",
            "line_color": "rgba(0, 0, 0, 0)",  # Transparent outline
            "line_width": 0,
        },
        # Proposed/suggested borehole marker styling (optimized locations)
        "proposed_marker": {
            "size": 7,
            "color": "rgba(0, 85, 255, 0.769)",  # Blue with 90% opacity
            "symbol": "x",  # Cross marker for proposed boreholes
            "buffer_color": "rgba(4, 0, 255, 0.173)",  # Blue (#0400FF) with 30% opacity
            "line_color": "rgba(0, 85, 255, 1)",  # Blue outline for buffer
            "line_width": 0.5,  # Buffer outline width in pixels
        },
        # Removed borehole marker styling (consolidation removed these)
        "removed_marker": {
            "size": 6,
            "color": "rgba(220, 53, 70, 0.914)",  # Red with 85% opacity
            "symbol": "x",  # Cross marker for removed boreholes
            "buffer_color": "rgba(220, 53, 70, 0)",  # Red with 25% opacity
            "line_color": "rgba(220, 53, 70, 1)",  # Red outline for buffer
            "line_width": 1,  # Buffer outline width in pixels
        },
        # Added borehole marker styling (consolidation added these - new positions)
        "added_marker": {
            "size": 6,
            "color": "rgba(40, 167, 70, 0.915)",  # Green with 85% opacity
            "symbol": "x",  # Cross marker for added boreholes
            "buffer_color": "rgba(40, 167, 70, 0)",  # Green with 25% opacity
            "line_color": "rgba(40, 167, 70, 1)",  # Green outline for buffer
            "line_width": 1,  # Buffer outline width in pixels
        },
        # First-pass candidate marker styling (border boreholes used as ILP candidates)
        # These appear as black X markers when Second Pass Grid layer is enabled
        # NOTE: Plotly Scattergl requires rgba() format, not 8-char hex
        "first_pass_candidate_marker": {
            "size": 9,
            "color": "rgba(0, 0, 0, 0.53)",  # #000000 with 0x87/255 ≈ 0.53 opacity
            "symbol": "x",
            "line_width": 1,
        },
        # Hexagonal candidate grid overlay styling (thin grid showing placement options)
        "hexgrid_overlay": {
            "color": "rgba(26, 26, 60, 1)",
            "line_width": 0.5,  # Thin lines for subtle overlay
        },
        # Second pass candidate grid overlay styling (consolidation candidates)
        "second_pass_grid": {
            "color": "rgba(26, 26, 60, 0.55)",
            "line_width": 0.5,  # Thin lines for subtle overlay
        },
        # CZRC test points styling (Tier 1 + Tier 2 ring test points)
        "czrc_test_points": {
            "tier1_color": "rgba(255, 0, 0, 1)",  # Red for Tier 1 uncovered
            "tier2_color": "rgba(174, 0, 255, 1)",  # Purple for Tier 2 ring uncovered
            "tier1_covered_color": "rgba(0, 200, 0, 1)",  # Green for Tier 1 covered by locked BHs
            "tier2_covered_color": "rgba(0, 180, 0, 1)",  # Green for Tier 2 ring covered by locked BHs
            "size": 5,  # Small markers
            "symbol": "circle",  # Circle markers
        },
        # Figure dimensions
        "figure_width": 1400,
        "figure_height": 900,
        # Figure background color
        "figure_background": "rgba(161, 150, 150, 0.076)",
        # Panel layout settings
        "panel_layout": {
            "left_panel_width": 220,
            "right_panel_width": 220,
            "top_offset": 80,
            "vertical_gap": 20,
            "sidebar_spacing": 100,
        },
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🗺️ BGS BEDROCK GEOLOGY LAYER (Rarely changed - at bottom)
    # ═══════════════════════════════════════════════════════════════════════
    "bgs_bedrock": {
        "enabled": True,  # Master toggle for this layer
        "layer_name": "BGS Bedrock",  # Display name in Layers panel
        "shapefile_path": "BGS_Shapefiles/Bedrock_geology_625000_scale.shp",
        "buffer_m": 500.0,  # Buffer around borehole extent (meters)
        "opacity": 0.4,  # Fill opacity for geology polygons
        "line_color": "rgba(50, 50, 50, 0.3)",  # Polygon outline color
        "line_width": 0.5,  # Polygon outline width
        "color_column": "LITHOSTRAT",  # Column to use for coloring formations
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🗺️ BGS SUPERFICIAL DEPOSITS LAYER (Rarely changed - at bottom)
    # ═══════════════════════════════════════════════════════════════════════
    "bgs_deposits": {
        "enabled": True,  # Master toggle for this layer
        "layer_name": "BGS Deposits",  # Display name in Layers panel
        "shapefile_path": "BGS_Shapefiles/Superficial_deposits_625000_scale.shp",
        "buffer_m": 500.0,  # Buffer around borehole extent (meters)
        "opacity": 0.4,  # Fill opacity for geology polygons
        "line_color": "rgba(50, 50, 50, 0.3)",  # Polygon outline color
        "line_width": 0.5,  # Polygon outline width
        "color_column": "LITHOSTRAT",  # Column to use for coloring formations
    },
    # ═══════════════════════════════════════════════════════════════════════
    # 🔍 QUALITY CONTROL
    # ═══════════════════════════════════════════════════════════════════════
    "quality_control": {
        "target_crs": "EPSG:27700",  # British National Grid
        "min_boreholes_for_analysis": 3,  # Minimum boreholes required
    },
    # Shortcut alias for backwards compatibility
    "target_crs": "EPSG:27700",
}


# ═══════════════════════════════════════════════════════════════════════════
# 📐 COMPUTED SPACING VALUES (Derived from multipliers × max_spacing_m)
# ═══════════════════════════════════════════════════════════════════════════
def get_candidate_spacing_m() -> float:
    """
    Compute candidate grid spacing in meters.

    Returns:
        Candidate spacing = max_spacing_m × candidate_spacing_mult
        Default: 100m × 0.5 = 50m
    """
    max_spacing = CONFIG.get("max_spacing_m", 100.0)
    mult = CONFIG.get("ilp_solver", {}).get("candidate_spacing_mult", 0.5)
    return max_spacing * mult


def get_test_spacing_m() -> float:
    """
    Compute test point grid spacing in meters.

    Returns:
        Test spacing = max_spacing_m × test_spacing_mult
        Default: 100m × 0.2 = 20m
    """
    max_spacing = CONFIG.get("max_spacing_m", 100.0)
    mult = CONFIG.get("ilp_solver", {}).get("test_spacing_mult", 0.2)
    return max_spacing * mult


# ═══════════════════════════════════════════════════════════════════════════
# 🚀 MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Running config.py directly launches main.py
    import subprocess
    import sys
    from pathlib import Path

    main_py = Path(__file__).parent / "main.py"
    sys.exit(subprocess.call([sys.executable, str(main_py)]))
