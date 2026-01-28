#!/usr/bin/env python3
"""
EC7 Borehole Placement Solvers Module

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Core optimization algorithms for borehole placement.
Contains ILP and greedy solvers without orchestration logic.

Key Functions:
- _solve_ilp: ILP solver dispatcher (highspy primary, PuLP fallback)
- _solve_ilp_highspy: Native highspy implementation (primary)
- _solve_ilp_pulp_fallback: PuLP fallback (when highspy unavailable)
- _solve_greedy: Greedy disk cover heuristic
- resolve_solver_mode: Convert solver mode string to boolean

SOLVER ARCHITECTURE:
- Primary: Native highspy backend (direct HiGHS API, true warm start support)
- Fallback: PuLP backend (if highspy not installed)
- Install highspy: pip install highspy

CONFIGURATION ARCHITECTURE:
- No CONFIG access - all functions accept explicit parameters
- Solver parameters (time_limit, mip_gap, etc.) passed explicitly
- Coverage/constraint data passed from orchestrator

PARALLELIZATION NOTE:
Keep threads=1 at ILP level. Parallelize at higher level instead.

For Navigation: Use VS Code outline (Ctrl+Shift+O) to jump between sections.
"""

import logging
import math
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

if TYPE_CHECKING:
    import pulp

# Check for highspy availability at module load
# When available, _solve_ilp() uses native highspy backend for:
# - Direct HiGHS API access (faster than PuLP abstraction)
# - True warm start support via setSolution()
# - Full control over all HiGHS options
HIGHSPY_AVAILABLE = False
try:
    import highspy

    HIGHSPY_AVAILABLE = True
except ImportError:
    # Will use PuLP fallback - slower but functional
    pass


# ===========================================================================
# 🔧 SOLVER SELECTION SECTION
# ===========================================================================


def _get_best_solver(
    time_limit: int,
    mip_gap: float,
    threads: int,
    verbose: int = 0,
    mip_heuristic_effort: float = 0.05,
    warm_start: bool = False,
    logger: Optional[logging.Logger] = None,
    logPath: Optional[str] = None,
) -> "pulp.apis.LpSolver":
    """
    Get the best available ILP solver.

    Priority order:
    1. HiGHS API (2-5× faster than CBC, uses highspy)
    2. HiGHS CMD (command-line version)
    3. CBC (default PuLP solver)

    PARALLELIZATION NOTE:
    ---------------------
    Keep threads=1 here. Do NOT parallelize at the ILP solver level.
    Instead, parallelize at a higher level by running multiple filter
    combinations concurrently. Each filter run should use threads=1
    to avoid CPU contention between ILP solver threads.

    Args:
        time_limit: Maximum solve time in seconds
        mip_gap: MIP gap tolerance
        threads: Number of threads (should be 1 for batch processing)
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        mip_heuristic_effort: Fraction of effort for primal heuristics (0.0-1.0)
        logger: Optional logger
        logPath: Optional path to write HiGHS solver output log file

    Returns:
        PuLP solver instance
    """
    import pulp

    # Convert verbose level to msg parameter
    # msg=0: silent, msg=1: normal output, msg=2+: detailed
    msg_level = min(verbose, 1)  # PuLP uses 0/1 for msg

    # Try HiGHS API first (fastest - uses highspy directly)
    # NOTE: HiGHS API (highspy) does NOT support warmStart parameter - only HiGHS_CMD does
    # If warm start is requested, skip HiGHS API and use HiGHS_CMD instead
    if not warm_start:
        try:
            # Build solver params, optionally including log_file
            solver_params = {"mip_heuristic_effort": mip_heuristic_effort}
            if logPath:
                solver_params["log_file"] = logPath

            solver = pulp.HiGHS(
                msg=msg_level,
                timeLimit=time_limit,
                gapRel=mip_gap,
                threads=threads,
                **solver_params,
            )
            if solver.available():
                if logger:
                    verbose_str = " (verbose)" if verbose > 0 else ""
                    heur_str = f", heuristic_effort={mip_heuristic_effort}"
                    logger.info(
                        f"   🚀 Using HiGHS solver (faster){verbose_str}{heur_str}"
                    )
                return solver
        except Exception:
            pass

    # Try HiGHS CMD (command-line version)
    # Note: HiGHS_CMD doesn't support mip_heuristic_effort via options
    # Note: HiGHS_CMD DOES support warmStart
    # Note: HiGHS_CMD requires the highs binary in PATH (not just highspy pip package)
    try:
        solver = pulp.HiGHS_CMD(
            msg=msg_level,
            timeLimit=time_limit,
            gapRel=mip_gap,
            threads=threads,
            warmStart=warm_start,
            logPath=logPath,
        )
        if solver.available():
            if logger:
                verbose_str = " (verbose)" if verbose > 0 else ""
                ws_str = ", warmStart" if warm_start else ""
                logger.info(
                    f"   🚀 Using HiGHS CMD solver (faster){verbose_str}{ws_str}"
                )
            return solver
        elif warm_start and logger:
            logger.warning(
                "   ⚠️ HiGHS_CMD not available (binary not in PATH). "
                "Warm start requested but falling back to solver without warm start support."
            )
    except Exception:
        pass

    # If warm start was requested but HiGHS_CMD not available, try HiGHS API anyway
    # (better to have faster solver without warm start than slower CBC)
    if warm_start:
        try:
            # Build solver params, optionally including log_file
            solver_params = {"mip_heuristic_effort": mip_heuristic_effort}
            if logPath:
                solver_params["log_file"] = logPath

            solver = pulp.HiGHS(
                msg=msg_level,
                timeLimit=time_limit,
                gapRel=mip_gap,
                threads=threads,
                **solver_params,
            )
            if solver.available():
                if logger:
                    verbose_str = " (verbose)" if verbose > 0 else ""
                    heur_str = f", heuristic_effort={mip_heuristic_effort}"
                    logger.info(
                        f"   🚀 Using HiGHS solver (no warmStart support){verbose_str}{heur_str}"
                    )
                return solver
        except Exception:
            pass

    # Fall back to CBC
    if logger:
        verbose_str = " (verbose)" if verbose > 0 else ""
        logger.info(f"   📦 Using CBC solver (default){verbose_str}")
    return pulp.PULP_CBC_CMD(
        msg=msg_level,
        timeLimit=time_limit,
        threads=threads,
        options=[f"ratioGap={mip_gap}"],
    )


# ===========================================================================
# 🔧 SOLVER MODE RESOLUTION
# ===========================================================================


def resolve_solver_mode(
    solver_mode: str = "ilp",
    is_parallel_context: bool = False,
    legacy_use_ilp: Optional[bool] = None,
) -> Tuple[bool, str]:
    """
    Resolve solver mode to use_ilp boolean.

    Supports both new solver_mode string and legacy use_ilp boolean.

    BEHAVIOR:
    - "ilp": Always use ILP first, falls back to greedy on timeout/failure
    - "greedy": Always use greedy directly, skip ILP entirely

    Args:
        solver_mode: "ilp" or "greedy"
        is_parallel_context: No longer affects solver selection (kept for API compat)
        legacy_use_ilp: Deprecated use_ilp boolean (for backwards compatibility)

    Returns:
        Tuple of (use_ilp: bool, solver_reason: str)
    """
    # Handle legacy use_ilp parameter (backwards compatibility)
    if legacy_use_ilp is not None:
        if legacy_use_ilp:
            return True, "ILP (legacy use_ilp=True)"
        else:
            return False, "Greedy (legacy use_ilp=False)"

    # Resolve solver_mode string
    solver_mode_lower = solver_mode.lower().strip()

    if solver_mode_lower == "ilp":
        return True, "ILP (solver_mode=ilp)"
    elif solver_mode_lower == "greedy":
        return False, "Greedy (solver_mode=greedy)"
    else:
        # Unknown mode (including old "auto") - default to ILP
        return True, f"ILP (defaulting from unknown solver_mode: {solver_mode})"


# ===========================================================================
# 🧮 ILP SOLVER DISPATCHER SECTION
# ===========================================================================


def _solve_ilp(
    test_points: List[Point],
    candidates: List[Point],
    coverage: Dict[int, List[int]],
    time_limit: int,
    mip_gap: float = 0.03,
    threads: int = 1,
    coverage_target_pct: float = 97.0,
    use_conflict_constraints: bool = True,
    conflict_constraint_mode: str = "clique",
    exclusion_factor: float = 0.8,
    max_spacing: float = 200.0,
    max_conflict_pairs: int = 200000,
    min_clique_size: int = 3,
    max_cliques: int = 50000,
    verbose: int = 0,
    mip_heuristic_effort: float = 0.05,
    warm_start_indices: Optional[List[int]] = None,
    logger: Optional[logging.Logger] = None,
    highs_log_file: Optional[str] = None,
    stall_detection_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    """Solve borehole placement ILP using highspy (primary) or PuLP (fallback).

    This function dispatches to the appropriate backend:
    - Primary: Native highspy for direct HiGHS API access
    - Fallback: PuLP if highspy is not installed

    Args:
        test_points: Points that must be covered
        candidates: Candidate borehole positions
        coverage: Dict mapping test_point_index -> list of candidate indices
        time_limit: Maximum solve time in seconds
        mip_gap: MIP gap tolerance (default 3%)
        threads: Number of threads (keep at 1 for batch processing)
        coverage_target_pct: Target coverage % (97% = cover 97% of coverable points)
        use_conflict_constraints: Whether to add exclusion constraints
        conflict_constraint_mode: "clique" or "pairwise"
        exclusion_factor: Fraction of max_spacing for exclusion distance
        max_spacing: Maximum spacing for exclusion calculation
        max_conflict_pairs: Max pairwise constraints (if mode="pairwise")
        min_clique_size: Min clique size to add (if mode="clique")
        max_cliques: Max cliques to add (if mode="clique")
        verbose: Verbosity level (0=silent, 1=progress)
        mip_heuristic_effort: Fraction of effort for primal heuristics
        warm_start_indices: Candidate indices to seed as initial solution
        logger: Optional logger
        highs_log_file: Optional path to write HiGHS output log
        stall_detection_config: Stall detection config dict (None=read from CONFIG)

    Returns:
        Tuple of (selected_candidate_indices, stats_dict) or (None, error_stats)
    """
    # Primary: Use native highspy backend (always preferred when available)
    if HIGHSPY_AVAILABLE:
        return _solve_ilp_highspy(
            test_points=test_points,
            candidates=candidates,
            coverage=coverage,
            time_limit=time_limit,
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
            warm_start_indices=warm_start_indices,
            logger=logger,
            highs_log_file=highs_log_file,
            stall_detection_config=stall_detection_config,
        )

    # Fallback: Use PuLP if highspy not installed
    if logger:
        logger.info("   ⚠️ highspy not available, using PuLP fallback")
    if warm_start_indices and logger:
        logger.warning("   ⚠️ Warm start not supported in PuLP fallback")
    return _solve_ilp_pulp_fallback(
        test_points=test_points,
        candidates=candidates,
        coverage=coverage,
        time_limit=time_limit,
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
        highs_log_file=highs_log_file,
    )


# ===========================================================================
# 📦 PULP FALLBACK SOLVER SECTION (when highspy not available)
# ===========================================================================


def _solve_ilp_pulp_fallback(
    test_points: List[Point],
    candidates: List[Point],
    coverage: Dict[int, List[int]],
    time_limit: int,
    mip_gap: float = 0.03,
    threads: int = 1,
    coverage_target_pct: float = 97.0,
    use_conflict_constraints: bool = True,
    conflict_constraint_mode: str = "clique",
    exclusion_factor: float = 0.8,
    max_spacing: float = 200.0,
    max_conflict_pairs: int = 200000,
    min_clique_size: int = 3,
    max_cliques: int = 50000,
    verbose: int = 0,
    mip_heuristic_effort: float = 0.05,
    logger: Optional[logging.Logger] = None,
    highs_log_file: Optional[str] = None,
) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    """
    Simplified PuLP fallback when highspy is not available.

    Note: Does not support warm start. For full functionality, install highspy.
    Uses n_coverable (not n_test_points) to prevent infeasibility when points
    have no covering candidates.
    """
    try:
        import pulp
    except ImportError:
        if logger:
            logger.warning("⚠️ Neither highspy nor PuLP installed")
        return None, {"method": "ilp_failed", "reason": "No solver available"}

    n_candidates = len(candidates)
    n_test_points = len(test_points)
    n_coverable = len(coverage)  # Only points with at least one covering candidate

    use_full_coverage = coverage_target_pct >= 99.9
    # Use n_coverable to prevent infeasibility when points are uncoverable
    min_covered = int(n_coverable * coverage_target_pct / 100.0)

    if logger:
        form_type = "FULL" if use_full_coverage else "PARTIAL"
        logger.info(
            f"   🧮 ILP {form_type} COVERAGE (PuLP fallback): "
            f"{n_candidates} candidates, {n_test_points} test points"
        )

    # Create problem
    prob = pulp.LpProblem("MinimumBoreholes", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(n_candidates)]

    if use_full_coverage:
        y = None
        prob += pulp.lpSum(x), "TotalBoreholes"
        for i in range(n_test_points):
            covering = coverage.get(i, [])
            if covering:
                prob += pulp.lpSum(x[j] for j in covering) >= 1, f"Cover_{i}"
    else:
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n_test_points)]
        prob += pulp.lpSum(x), "TotalBoreholes"
        prob += pulp.lpSum(y) >= min_covered, "MinCoverage"
        for i in range(n_test_points):
            covering = coverage.get(i, [])
            if covering:
                prob += y[i] <= pulp.lpSum(x[j] for j in covering), f"Link_{i}"
            else:
                prob += y[i] == 0, f"Uncoverable_{i}"

    # Add conflict constraints if enabled
    conflict_stats: Dict[str, Any] = {
        "constraint_mode": (
            conflict_constraint_mode if use_conflict_constraints else "disabled"
        ),
        "enabled": use_conflict_constraints,
    }

    if use_conflict_constraints:
        from Gap_Analysis_EC7.solvers.optimization_geometry import (
            _generate_conflict_pairs,
            _generate_clique_constraints,
        )

        exclusion_dist = exclusion_factor * max_spacing
        conflict_stats["exclusion_factor"] = exclusion_factor
        conflict_stats["exclusion_dist_m"] = exclusion_dist

        if conflict_constraint_mode == "clique":
            cliques, clique_stats = _generate_clique_constraints(
                candidates, exclusion_dist, min_clique_size, max_cliques, logger
            )
            for idx, clique in enumerate(cliques):
                prob += pulp.lpSum(x[j] for j in clique) <= 1, f"Clique_{idx}"
            conflict_stats.update(clique_stats)
            conflict_stats["n_conflict_constraints"] = len(cliques)
        else:
            pairs, pairs_gen, truncated = _generate_conflict_pairs(
                candidates, exclusion_dist, max_conflict_pairs, logger
            )
            for j, k in pairs:
                prob += x[j] + x[k] <= 1, f"Conflict_{j}_{k}"
            conflict_stats.update(
                {
                    "conflict_pairs_count": len(pairs),
                    "conflict_pairs_generated": pairs_gen,
                    "conflict_was_truncated": truncated,
                    "n_conflict_constraints": len(pairs),
                }
            )

    # Solve using best available solver
    solver = _get_best_solver(
        time_limit,
        mip_gap,
        threads,
        verbose,
        mip_heuristic_effort,
        warm_start=False,
        logger=logger,
        logPath=highs_log_file,
    )

    try:
        prob.solve(solver)
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.warning(f"⚠️ ILP solver error: {e}")
        return None, {"method": "ilp_failed", "reason": str(e)}

    if prob.status != pulp.LpStatusOptimal:
        status_name = pulp.LpStatus.get(prob.status, "Unknown")
        if logger:
            logger.warning(f"⚠️ ILP solver status: {status_name}")
        return None, {"method": "ilp_failed", "reason": status_name}

    # Extract solution
    selected = [j for j in range(n_candidates) if x[j].value() and x[j].value() > 0.5]

    if use_full_coverage:
        covered, pct = n_coverable, 100.0
        form = "full_coverage"
    else:
        covered = sum(1 for i in range(n_test_points) if y[i].value() == 1)
        pct = (covered / n_coverable * 100) if n_coverable else 0
        form = "partial_coverage"

    stats = {
        "method": "ilp_pulp_fallback",
        "formulation": form,
        "solver_status": "optimal",
        "objective_value": pulp.value(prob.objective),
        "coverage_target_pct": coverage_target_pct,
        "actual_coverage_pct": pct,
        "test_points_covered": covered,
        "test_points_total": n_test_points,
        "test_points_coverable": n_coverable,
        "conflict_stats": conflict_stats,
        "conflict_constraints_enabled": use_conflict_constraints,
        "warm_start_used": False,
        "warm_start_count": 0,
    }

    if logger:
        logger.info(
            f"   ✅ ILP optimal (PuLP fallback): {len(selected)} boreholes, "
            f"{pct:.1f}% coverage"
        )

    return selected, stats


# ===========================================================================
# 🧮 HIGHSPY NATIVE ILP SOLVER SECTION
# ===========================================================================


def _solve_ilp_highspy(
    test_points: List[Point],
    candidates: List[Point],
    coverage: Dict[int, List[int]],
    time_limit: int,
    mip_gap: float = 0.03,
    threads: int = 1,
    coverage_target_pct: float = 97.0,
    use_conflict_constraints: bool = True,
    conflict_constraint_mode: str = "clique",
    exclusion_factor: float = 0.8,
    max_spacing: float = 200.0,
    max_conflict_pairs: int = 200000,
    min_clique_size: int = 3,
    max_cliques: int = 50000,
    verbose: int = 0,
    mip_heuristic_effort: float = 0.05,
    warm_start_indices: Optional[List[int]] = None,
    logger: Optional[logging.Logger] = None,
    highs_log_file: Optional[str] = None,
    stall_detection_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    """
    Solve borehole placement using native highspy ILP with warm start support.

    This is an alternative to _solve_ilp() that uses highspy directly instead of
    PuLP. The main advantage is full warm start support via setSolution().

    Args:
        test_points: Points that must be covered
        candidates: Candidate borehole positions
        coverage: Dict mapping test_point_index -> list of candidate indices that cover it
        time_limit: Maximum solve time in seconds
        mip_gap: MIP gap tolerance
        threads: Number of threads (should be 1 for batch processing)
        coverage_target_pct: Target coverage percentage (97% = cover 97% of test points)
        use_conflict_constraints: Whether to add exclusion constraints
        conflict_constraint_mode: "clique" or "pairwise"
        exclusion_factor: Fraction of max_spacing for exclusion distance
        max_spacing: Maximum spacing for exclusion calculation
        max_conflict_pairs: Max pairwise constraints (if mode="pairwise")
        min_clique_size: Min clique size to add (if mode="clique")
        max_cliques: Max cliques to add (if mode="clique")
        verbose: Verbosity level (0=silent, 1=progress)
        mip_heuristic_effort: Fraction of effort for primal heuristics
        warm_start_indices: Candidate indices to seed as initial solution
        logger: Optional logger
        highs_log_file: Optional path to write HiGHS solver output to a file.
        stall_detection_config: Configuration for stall detection (dict with keys:
            enabled, min_solutions_before_check, stall_window_seconds,
            min_improvement_pct, absolute_gap_threshold).
            If not provided, reads from CONFIG["ilp_solver"]["stall_detection"].

    Returns:
        Tuple of (selected_candidate_indices, stats_dict) or (None, error_stats)
    """
    # Note: highspy import is done at module level (HIGHSPY_AVAILABLE check)
    # This function is only called when HIGHSPY_AVAILABLE is True

    # If stall_detection_config not explicitly provided, read from CONFIG
    if stall_detection_config is None:
        try:
            from Gap_Analysis_EC7.config import CONFIG

            stall_detection_config = CONFIG.get("ilp_solver", {}).get(
                "stall_detection", {}
            )
        except (ImportError, KeyError):
            stall_detection_config = {}  # Empty = disabled

    n_candidates = len(candidates)
    n_test_points = len(test_points)

    # Count coverable test points (those with at least one covering candidate)
    n_coverable = len(coverage)  # coverage dict only contains coverable points

    use_full_coverage = coverage_target_pct >= 99.9
    # Calculate min_covered based on COVERABLE test points, not total
    # This prevents infeasibility when some test points are uncoverable
    min_covered = int(n_coverable * coverage_target_pct / 100.0)

    # === STEP 1: Initialize HiGHS solver ===
    h = highspy.Highs()
    h.setOptionValue("output_flag", verbose > 0)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", mip_gap)
    h.setOptionValue("threads", threads)
    h.setOptionValue("mip_heuristic_effort", mip_heuristic_effort)

    # Write HiGHS output to log file if path provided
    if highs_log_file:
        h.setOptionValue("log_file", highs_log_file)

    # === STEP 2: Add candidate selection variables (x_j) ===
    # minimize sum(x_j) - minimize total boreholes
    cost = np.ones(n_candidates, dtype=np.double)
    lower = np.zeros(n_candidates, dtype=np.double)
    upper = np.ones(n_candidates, dtype=np.double)

    # Add columns with empty constraint matrix initially
    h.addCols(n_candidates, cost, lower, upper, 0, [], [], [])

    # Make them binary (integer with 0-1 bounds)
    for j in range(n_candidates):
        h.changeColIntegrality(j, highspy.HighsVarType.kInteger)

    if logger:
        form_type = "FULL" if use_full_coverage else "PARTIAL"
        logger.info(
            f"   🧮 ILP {form_type} COVERAGE (highspy): {n_candidates} candidates, "
            f"{n_test_points} test points"
        )

    # === STEP 3: Add coverage constraints ===
    if use_full_coverage:
        # Full coverage: for each test point, at least one covering candidate
        for i in range(n_test_points):
            covering = coverage.get(i, [])
            if covering:
                h.addRow(
                    1.0,
                    highspy.kHighsInf,  # >= 1
                    len(covering),
                    np.array(covering, dtype=np.int32),
                    np.ones(len(covering), dtype=np.double),
                )
    else:
        # Partial coverage: add y_i indicator variables
        # Need to extend column count: x_0..x_{n-1}, y_0..y_{n_test_points-1}

        # Add y variables with zero cost (we only minimize x)
        y_cost = np.zeros(n_test_points, dtype=np.double)
        y_lower = np.zeros(n_test_points, dtype=np.double)
        y_upper = np.ones(n_test_points, dtype=np.double)
        h.addCols(n_test_points, y_cost, y_lower, y_upper, 0, [], [], [])

        # Make y variables binary
        for i in range(n_test_points):
            h.changeColIntegrality(n_candidates + i, highspy.HighsVarType.kInteger)

        # Constraint: sum(y) >= min_covered
        y_indices = np.arange(
            n_candidates, n_candidates + n_test_points, dtype=np.int32
        )
        y_coeffs = np.ones(n_test_points, dtype=np.double)
        h.addRow(
            float(min_covered), highspy.kHighsInf, n_test_points, y_indices, y_coeffs
        )

        # Linking constraints: y_i <= sum(x_j for j covering i)
        for i in range(n_test_points):
            covering = coverage.get(i, [])
            if covering:
                # y_i - sum(x_j for j covering i) <= 0
                indices = np.array([n_candidates + i] + list(covering), dtype=np.int32)
                coeffs = np.array([1.0] + [-1.0] * len(covering), dtype=np.double)
                h.addRow(-highspy.kHighsInf, 0.0, len(indices), indices, coeffs)
            else:
                # Uncoverable point: y_i = 0
                h.changeColBounds(n_candidates + i, 0.0, 0.0)

    # === STEP 4: Add conflict constraints ===
    conflict_stats: Dict[str, Any] = {
        "constraint_mode": (
            conflict_constraint_mode if use_conflict_constraints else "disabled"
        ),
        "enabled": use_conflict_constraints,
    }

    if use_conflict_constraints:
        from Gap_Analysis_EC7.solvers.optimization_geometry import (
            _generate_conflict_pairs,
            _generate_clique_constraints,
        )

        exclusion_dist = exclusion_factor * max_spacing
        conflict_stats["exclusion_factor"] = exclusion_factor
        conflict_stats["exclusion_dist_m"] = exclusion_dist

        if conflict_constraint_mode == "clique":
            cliques, clique_stats = _generate_clique_constraints(
                candidates, exclusion_dist, min_clique_size, max_cliques, logger
            )
            for clique in cliques:
                # sum(x_j for j in clique) <= 1
                h.addRow(
                    -highspy.kHighsInf,
                    1.0,
                    len(clique),
                    np.array(clique, dtype=np.int32),
                    np.ones(len(clique), dtype=np.double),
                )
            conflict_stats.update(clique_stats)
            conflict_stats["n_conflict_constraints"] = len(cliques)
        else:
            pairs, pairs_gen, truncated = _generate_conflict_pairs(
                candidates, exclusion_dist, max_conflict_pairs, logger
            )
            for j, k in pairs:
                # x_j + x_k <= 1
                h.addRow(
                    -highspy.kHighsInf,
                    1.0,
                    2,
                    np.array([j, k], dtype=np.int32),
                    np.array([1.0, 1.0], dtype=np.double),
                )
            conflict_stats.update(
                {
                    "conflict_pairs_count": len(pairs),
                    "conflict_pairs_generated": pairs_gen,
                    "conflict_was_truncated": truncated,
                    "n_conflict_constraints": len(pairs),
                }
            )

    # === STEP 5: Apply warm start ===
    if warm_start_indices:
        warm_set = set(warm_start_indices)

        # Build solution vector
        n_total_cols = (
            n_candidates if use_full_coverage else n_candidates + n_test_points
        )
        col_values = []

        # x values: 1 for warm start indices, 0 otherwise
        for j in range(n_candidates):
            col_values.append(1.0 if j in warm_set else 0.0)

        # y values (if partial coverage): 1 if covered by warm start
        if not use_full_coverage:
            for i in range(n_test_points):
                covering = coverage.get(i, [])
                covered_by_warm = any(j in warm_set for j in covering)
                col_values.append(1.0 if covered_by_warm else 0.0)

        # Create HighsSolution and set it
        solution = highspy.HighsSolution()
        solution.col_value = col_values
        status = h.setSolution(solution)

        if logger:
            logger.info(
                f"   🔥 Warm start: {len(warm_start_indices)} candidates seeded"
            )

    # === STEP 6: Set up stall detection callback (if enabled) ===
    # ══════════════════════════════════════════════════════════════════════════
    # STALL DETECTION (Log-Based)
    # ══════════════════════════════════════════════════════════════════════════
    # Monitors MIP gap improvement rate, not just absolute gap level.
    #
    # Key Insight: HiGHS has two types of progress:
    #   1. Primal progress: Finding better feasible solutions (BestSol improves)
    #   2. Dual progress: Proving optimality (BestBound tightens, gap decreases)
    #
    # The kCallbackMipImprovingSolution callback ONLY fires for primal progress.
    # When solver stalls on proving optimality (dual progress only), no new
    # solutions are found, so that callback never fires.
    #
    # Solution: Use kCallbackMipLogging which fires every time HiGHS prints
    # a log row (controlled by mip_min_logging_interval, default 5 seconds).
    # This captures BOTH types of progress via the mip_gap field.
    #
    # Detection Logic:
    #   - Track gap at each log row (every ~5 seconds)
    #   - After warmup_log_rows, compare current gap to comparison_window rows ago
    #   - If improvement < min_improvement_pct gap points → STALL DETECTED
    # ══════════════════════════════════════════════════════════════════════════

    stall_enabled = stall_detection_config.get("enabled", False)
    stall_stats = {
        "stall_detected": False,
        "final_gap_pct": None,
        "log_rows_processed": 0,
        "gap_history": [],
        "termination_reason": None,
    }

    if stall_enabled:
        # Extract stall detection parameters (now in seconds directly)
        # gap_threshold_pct: only start stall detection once gap is below this
        # warmup_seconds: skip first N seconds AFTER threshold reached
        # comparison_seconds: compare current gap to gap from N seconds ago
        # min_improvement_pct: require at least this many gap points improvement
        gap_threshold = stall_detection_config.get("gap_threshold_pct", 15.0)
        warmup_seconds = stall_detection_config.get("warmup_seconds", 15.0)
        comparison_seconds = stall_detection_config.get("comparison_seconds", 10.0)
        min_improvement = stall_detection_config.get("min_improvement_pct", 5.0)

        # Mutable state for callback (time-based tracking)
        import time as time_module

        callback_state = {
            "gap_times": [],  # List of (time, gap) tuples - only after threshold reached
            "threshold_reached": False,  # True once gap drops below threshold
            "threshold_reached_time": None,  # Time when threshold was first reached
            "terminated": False,
            "termination_reason": None,
            "last_check_time": 0.0,
            "start_time": time_module.perf_counter(),
        }

        hscb = highspy.cb

        def stall_detection_callback(
            callback_type: int,
            message: str,
            data_out: Any,
            data_in: Any,
            user_callback_data: Any,
        ) -> None:
            """Time-based stall detection callback with gap threshold.

            Uses kCallbackMipInterrupt which respects user_interrupt.
            Stall detection only activates once gap drops below gap_threshold_pct.
            After threshold is reached, waits warmup_seconds then starts monitoring.
            """
            nonlocal callback_state

            if callback_state["terminated"]:
                data_in.user_interrupt = True  # Keep signaling interrupt
                return

            # Only process MipInterrupt events
            if callback_type != hscb.HighsCallbackType.kCallbackMipInterrupt:
                return

            current_time = time_module.perf_counter() - callback_state["start_time"]
            current_gap_pct = data_out.mip_gap * 100.0

            # Skip infinite gaps (no dual bound computed yet)
            if current_gap_pct == float("inf"):
                return

            # Only sample every 1 second to avoid too many entries
            if current_time - callback_state["last_check_time"] < 1.0:
                return
            callback_state["last_check_time"] = current_time

            # Check if gap threshold has been reached
            if not callback_state["threshold_reached"]:
                if current_gap_pct <= gap_threshold:
                    callback_state["threshold_reached"] = True
                    callback_state["threshold_reached_time"] = current_time
                    if logger:
                        logger.info(
                            f"   📉 Gap threshold reached: {current_gap_pct:.2f}% <= {gap_threshold:.1f}% "
                            f"at t={current_time:.1f}s (warmup starts now)"
                        )
                return  # Don't record samples until threshold is reached

            # Record this gap sample (only after threshold reached)
            time_since_threshold = (
                current_time - callback_state["threshold_reached_time"]
            )

            # Only start recording samples AFTER warmup is complete
            # This ensures comparison window only looks at post-warmup data
            if time_since_threshold < warmup_seconds:
                return  # Still in warmup, don't record or compare

            callback_state["gap_times"].append((time_since_threshold, current_gap_pct))

            # Find gap from comparison_seconds ago (must be after warmup)
            comparison_time = time_since_threshold - comparison_seconds

            # Need at least comparison_seconds of history after warmup
            if comparison_time < warmup_seconds:
                return  # Not enough post-warmup history yet

            old_gap = None
            for t, g in reversed(callback_state["gap_times"]):
                if t <= comparison_time and g != float("inf"):
                    old_gap = g
                    break

            if old_gap is None:
                return  # Not enough history yet

            improvement = old_gap - current_gap_pct  # Absolute gap point improvement

            if improvement < min_improvement:
                # Stall detected! Gap hasn't improved enough over time window
                data_in.user_interrupt = True
                callback_state["terminated"] = True
                callback_state["termination_reason"] = "stall_detected"
                if logger:
                    logger.info(
                        f"   🛑 Stall detected at t={current_time:.1f}s: "
                        f"gap {current_gap_pct:.2f}% (improvement: {improvement:.2f} pts < "
                        f"threshold: {min_improvement:.1f} pts over {comparison_seconds:.0f}s)"
                    )

        h.setCallback(stall_detection_callback, None)
        # Use MipInterrupt callback - respects user_interrupt for early termination
        h.startCallback(hscb.HighsCallbackType.kCallbackMipInterrupt)

        if logger:
            # Explain the time-based detection mechanism with gap threshold
            logger.info(
                f"   📊 Stall detection: gap threshold {gap_threshold:.1f}%, then "
                f"require {min_improvement:.1f} pt improvement over {comparison_seconds:.0f}s "
                f"(after {warmup_seconds:.0f}s warmup)"
            )

    # === STEP 7: Solve ===
    h.run()

    # Stop callbacks if started
    if stall_enabled:
        h.stopCallback(highspy.cb.HighsCallbackType.kCallbackMipInterrupt)
        # Capture final state
        stall_stats["stall_detected"] = callback_state["terminated"]
        stall_stats["log_rows_processed"] = len(callback_state["gap_times"])
        stall_stats["gap_history"] = [g for _, g in callback_state["gap_times"]]
        stall_stats["termination_reason"] = callback_state["termination_reason"]
        # Set final gap from history if available
        if callback_state["gap_times"]:
            stall_stats["final_gap_pct"] = callback_state["gap_times"][-1][1]

        # Log stall detection outcome
        if logger and stall_stats["stall_detected"]:
            final_time = (
                callback_state["gap_times"][-1][0] if callback_state["gap_times"] else 0
            )
            logger.info(
                f"   ✅ Early termination via stall detection at t={final_time:.1f}s "
                f"(gap: {stall_stats['final_gap_pct']:.1f}%, samples: {stall_stats['log_rows_processed']})"
            )

    # === STEP 8: Extract solution ===
    model_status = h.getModelStatus()

    # Check if we have a feasible solution (optimal, early termination, or time limit)
    # HiGHS provides incumbent solution for these statuses
    has_feasible_solution = model_status in (
        highspy.HighsModelStatus.kOptimal,
        highspy.HighsModelStatus.kInterrupt,  # Stall detection callback
        highspy.HighsModelStatus.kTimeLimit,  # Time limit reached but has solution
        highspy.HighsModelStatus.kSolutionLimit,  # Max improving solutions reached
    )

    if has_feasible_solution:
        sol = h.getSolution()
        info = h.getInfo()

        # Check that we actually have a solution
        if sol.col_value is None or len(sol.col_value) < n_candidates:
            if logger:
                logger.warning("⚠️ No valid solution available after early termination")
            return None, {"method": "ilp_failed", "reason": "no_solution"}

        # Extract selected candidates (x_j > 0.5)
        selected = [j for j in range(n_candidates) if sol.col_value[j] > 0.5]

        # Calculate actual coverage (percentage of COVERABLE points covered)
        if use_full_coverage:
            covered_count = n_coverable
            actual_pct = 100.0
        else:
            covered_count = sum(
                1 for i in range(n_test_points) if sol.col_value[n_candidates + i] > 0.5
            )
            actual_pct = (covered_count / n_coverable * 100) if n_coverable else 0

        # Determine solver status description
        if model_status == highspy.HighsModelStatus.kOptimal:
            solver_status = "optimal"
        elif model_status == highspy.HighsModelStatus.kInterrupt:
            solver_status = "early_termination"
        elif model_status == highspy.HighsModelStatus.kTimeLimit:
            solver_status = "time_limit"
        else:
            solver_status = "feasible"

        stats = {
            "method": "ilp_highspy",
            "formulation": "full_coverage" if use_full_coverage else "partial_coverage",
            "solver_status": solver_status,
            "objective_value": info.objective_function_value,
            "coverage_target_pct": coverage_target_pct,
            "actual_coverage_pct": actual_pct,
            "test_points_covered": covered_count,
            "test_points_total": n_test_points,
            "test_points_coverable": n_coverable,
            "n_variables": (
                n_candidates if use_full_coverage else n_candidates + n_test_points
            ),
            "conflict_stats": conflict_stats,
            "conflict_constraints_enabled": use_conflict_constraints,
            "warm_start_used": warm_start_indices is not None
            and len(warm_start_indices) > 0,
            "warm_start_count": len(warm_start_indices) if warm_start_indices else 0,
            "stall_detection": stall_stats,
        }

        status_emoji = "✅" if solver_status == "optimal" else "⚡"
        if logger:
            gap_info = (
                f" (gap: {stall_stats['final_gap_pct']:.1f}%)"
                if stall_stats["final_gap_pct"]
                else ""
            )
            logger.info(
                f"   {status_emoji} ILP {solver_status} (highspy): {len(selected)} boreholes, "
                f"{actual_pct:.1f}% coverage{gap_info}"
            )

        return selected, stats
    else:
        # Handle non-optimal status without feasible solution
        status_names = {
            highspy.HighsModelStatus.kInfeasible: "Infeasible",
            highspy.HighsModelStatus.kUnbounded: "Unbounded",
            highspy.HighsModelStatus.kUnboundedOrInfeasible: "UnboundedOrInfeasible",
            highspy.HighsModelStatus.kNotset: "NotSet",
            highspy.HighsModelStatus.kLoadError: "LoadError",
            highspy.HighsModelStatus.kModelError: "ModelError",
            highspy.HighsModelStatus.kPresolveError: "PresolveError",
            highspy.HighsModelStatus.kSolveError: "SolveError",
            highspy.HighsModelStatus.kPostsolveError: "PostsolveError",
            highspy.HighsModelStatus.kModelEmpty: "ModelEmpty",
            highspy.HighsModelStatus.kObjectiveBound: "ObjectiveBound",
            highspy.HighsModelStatus.kObjectiveTarget: "ObjectiveTarget",
            highspy.HighsModelStatus.kTimeLimit: "TimeLimit",
            highspy.HighsModelStatus.kIterationLimit: "IterationLimit",
            highspy.HighsModelStatus.kSolutionLimit: "SolutionLimit",
            highspy.HighsModelStatus.kInterrupt: "Interrupt",
            highspy.HighsModelStatus.kMemoryLimit: "MemoryLimit",
        }
        status_name = status_names.get(model_status, str(model_status))

        if logger:
            logger.warning(f"⚠️ ILP solver status (highspy): {status_name}")

        return None, {"method": "ilp_failed", "reason": status_name}


# ===========================================================================
# 🔄 GREEDY SOLVER SECTION
# ===========================================================================


def _solve_greedy(
    gap_polys: List[Polygon],
    candidates: List[Point],
    radius: float,
    max_iterations: int = 1000,
    min_coverage_gain: float = 1.0,
    min_efficiency_pct: float = 5.0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Solve borehole placement using greedy disk cover heuristic.

    Iteratively selects candidate that covers maximum remaining uncovered area.
    Continues until all gaps are covered or no candidate provides sufficient gain.

    STOPPING CONDITIONS (any of these stops the algorithm):
    1. Remaining area < min_coverage_gain (explicit minimum)
    2. Best coverage efficiency < min_efficiency_pct (proportional to borehole area)
    3. max_iterations reached (safety limit)
    4. No remaining candidates

    The efficiency-based stopping is critical for large gaps: with 200m radius
    boreholes (125,664 m² each), we should stop adding boreholes when the best
    remaining coverage is < 5-10% of borehole area (6,283-12,566 m²).

    Args:
        gap_polys: List of gap Polygon objects
        candidates: List of candidate borehole locations
        radius: Coverage radius per borehole
        max_iterations: Maximum iterations before stopping (safety limit)
        min_coverage_gain: Minimum area gain (m²) per iteration to continue
        min_efficiency_pct: Minimum coverage efficiency % (gain/borehole_area × 100)
        logger: Optional logger

    Returns:
        Tuple of (selected_candidate_indices, stats_dict)
    """
    borehole_area = math.pi * radius * radius
    efficiency_threshold = borehole_area * min_efficiency_pct / 100.0

    # Use the stricter of min_coverage_gain and efficiency_threshold
    effective_min_gain = max(min_coverage_gain, efficiency_threshold)

    if logger:
        logger.info(f"   🔄 Solving with greedy heuristic...")
        logger.info(
            f"      max_iterations={max_iterations}, "
            f"min_gain={min_coverage_gain:.0f}m², "
            f"efficiency_threshold={efficiency_threshold:.0f}m² ({min_efficiency_pct}%)"
        )
        logger.info(
            f"      effective_min_gain={effective_min_gain:.0f}m² (stricter of the two)"
        )

    remaining = unary_union(gap_polys)
    initial_area = remaining.area if not remaining.is_empty else 0
    selected = []
    iterations = 0
    last_coverage_gain = borehole_area  # Start with full coverage assumed

    while (
        not remaining.is_empty
        and remaining.area > effective_min_gain
        and iterations < max_iterations
    ):
        iterations += 1

        best_idx = None
        best_coverage = 0

        for j, cand in enumerate(candidates):
            if j in selected:
                continue

            disk = cand.buffer(radius)
            covered_area = disk.intersection(remaining).area

            if covered_area > best_coverage:
                best_coverage = covered_area
                best_idx = j

        if best_idx is None or best_coverage < effective_min_gain:
            if logger and iterations <= 10:
                logger.info(
                    f"      Stopping at iteration {iterations}: best_coverage={best_coverage:.0f}m² < threshold"
                )
            break

        selected.append(best_idx)
        remaining = remaining.difference(candidates[best_idx].buffer(radius))
        last_coverage_gain = best_coverage

        # Log progress for first few and every 10th iteration
        if logger and (iterations <= 5 or iterations % 10 == 0):
            efficiency = (best_coverage / borehole_area) * 100
            logger.info(
                f"      Iter {iterations}: added BH, coverage={best_coverage:.0f}m² "
                f"({efficiency:.1f}% eff), remaining={remaining.area/10000:.1f}ha"
            )

    remaining_area = remaining.area if not remaining.is_empty else 0
    covered_area = initial_area - remaining_area
    coverage_pct = (covered_area / initial_area * 100) if initial_area > 0 else 100.0

    stats = {
        "method": "greedy",
        "iterations": iterations,
        "initial_gap_area": initial_area,
        "remaining_area": remaining_area,
        "coverage_achieved_pct": coverage_pct,
        "max_iterations_config": max_iterations,
        "min_coverage_gain_config": min_coverage_gain,
        "efficiency_threshold_m2": efficiency_threshold,
        "effective_min_gain_m2": effective_min_gain,
        "last_coverage_gain": last_coverage_gain,
    }

    if logger:
        logger.info(
            f"   ✅ Greedy: {len(selected)} boreholes in {iterations} iterations "
            f"({coverage_pct:.1f}% coverage)"
        )

    return selected, stats
