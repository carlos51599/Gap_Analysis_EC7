#!/usr/bin/env python3
"""
EC7 Borehole Placement Solvers Module

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Core optimization algorithms for borehole placement.
Contains ILP and greedy solvers without orchestration logic.

Key Functions:
- _get_best_solver: Select best available ILP solver (HiGHS > CBC)
- resolve_solver_mode: Convert solver mode string to boolean
- _solve_ilp: Integer Linear Programming solver implementation
- _solve_greedy: Greedy disk cover heuristic implementation

CONFIGURATION ARCHITECTURE:
- No CONFIG access - all functions accept explicit parameters
- Solver parameters (time_limit, mip_gap, etc.) passed explicitly
- Coverage/constraint data passed from orchestrator

SOLVER NOTES:
- HiGHS is preferred when available (2-5× faster than CBC)
- Install HiGHS: pip install highspy
- Falls back to CBC if HiGHS not available
- PARALLELIZATION: Keep threads=1, parallelize at higher level instead

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
# 🧮 ILP SOLVER SECTION
# ===========================================================================


def _create_ilp_problem(
    n_candidates: int,
    n_test_points: int,
    coverage: Dict[int, List[int]],
    coverage_target_pct: float,
    logger: Optional[logging.Logger] = None,
) -> Tuple[
    "pulp.LpProblem",
    List["pulp.LpVariable"],
    Optional[List["pulp.LpVariable"]],
    bool,
    int,
]:
    """Create ILP problem with full or partial coverage formulation."""
    import pulp

    use_full_coverage = coverage_target_pct >= 99.9
    min_covered = int(n_test_points * coverage_target_pct / 100.0)

    prob = pulp.LpProblem("MinimumBoreholes", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(n_candidates)]

    if use_full_coverage:
        if logger:
            logger.info(
                f"   🧮 ILP FULL COVERAGE: {n_candidates} vars, "
                f"{n_test_points} constraints (simple set cover)"
            )
        y = None
        prob += pulp.lpSum(x), "TotalBoreholes"

        for i in range(n_test_points):
            covering = coverage.get(i, [])
            if covering:
                prob += pulp.lpSum(x[j] for j in covering) >= 1, f"Cover_{i}"
    else:
        if logger:
            logger.info(
                f"   🧮 ILP PARTIAL COVERAGE: {n_candidates + n_test_points} vars, "
                f"target={min_covered}/{n_test_points} = {coverage_target_pct}%"
            )
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n_test_points)]
        prob += pulp.lpSum(x), "TotalBoreholes"
        prob += pulp.lpSum(y) >= min_covered, "MinCoverage"

        for i in range(n_test_points):
            covering = coverage.get(i, [])
            if covering:
                prob += y[i] <= pulp.lpSum(x[j] for j in covering), f"Link_{i}"
            else:
                prob += y[i] == 0, f"Uncoverable_{i}"

    return prob, x, y, use_full_coverage, min_covered


def _add_conflict_constraints(
    prob: "pulp.LpProblem",
    x: List["pulp.LpVariable"],
    candidates: List[Point],
    conflict_constraint_mode: str,
    exclusion_factor: float,
    max_spacing: float,
    min_clique_size: int,
    max_cliques: int,
    max_conflict_pairs: int,
    n_test_points: int,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Add conflict constraints (clique or pairwise) to ILP problem."""
    from Gap_Analysis_EC7.solvers.optimization_geometry import (
        _generate_conflict_pairs,
        _generate_clique_constraints,
    )
    import pulp

    exclusion_dist = exclusion_factor * max_spacing
    stats: Dict[str, Any] = {
        "constraint_mode": conflict_constraint_mode,
        "enabled": True,
        "exclusion_factor": exclusion_factor,
        "exclusion_dist_m": exclusion_dist,
    }

    if conflict_constraint_mode == "clique":
        cliques, clique_stats = _generate_clique_constraints(
            candidates,
            exclusion_dist,
            min_clique_size=min_clique_size,
            max_cliques=max_cliques,
            logger=logger,
        )
        for idx, clique in enumerate(cliques):
            prob += pulp.lpSum(x[j] for j in clique) <= 1, f"Clique_{idx}"
        stats.update(clique_stats)
        stats["n_conflict_constraints"] = len(cliques)
        if logger and cliques:
            logger.info(
                f"   📊 Total ILP: {n_test_points + len(cliques)} constraints "
                f"({n_test_points} coverage + {len(cliques)} clique)"
            )
    else:
        pairs, pairs_gen, truncated = _generate_conflict_pairs(
            candidates, exclusion_dist, max_conflict_pairs, logger
        )
        for j, k in pairs:
            prob += x[j] + x[k] <= 1, f"Conflict_{j}_{k}"
        stats.update(
            {
                "conflict_pairs_count": len(pairs),
                "conflict_pairs_generated": pairs_gen,
                "conflict_was_truncated": truncated,
                "max_conflict_pairs": max_conflict_pairs,
                "n_conflict_constraints": len(pairs),
            }
        )
        if logger:
            logger.info(
                f"   📊 Total ILP: {n_test_points + len(pairs)} constraints "
                f"({n_test_points} coverage + {len(pairs)} pairwise)"
            )

    return stats


def _execute_ilp_solver(
    prob: "pulp.LpProblem",
    time_limit: int,
    mip_gap: float,
    threads: int,
    verbose: int,
    mip_heuristic_effort: float,
    warm_start: bool = False,
    logger: Optional[logging.Logger] = None,
    highs_log_file: Optional[str] = None,
) -> Tuple[int, Optional[str]]:
    """Execute ILP solver, returns (status, error_msg)."""
    solver = _get_best_solver(
        time_limit,
        mip_gap,
        threads,
        verbose,
        mip_heuristic_effort,
        warm_start,
        logger,
        logPath=highs_log_file,
    )
    try:
        prob.solve(solver)
        return prob.status, None
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.warning(f"⚠️ ILP solver error: {e}")
        return -1, str(e)


def _extract_ilp_solution(
    prob: "pulp.LpProblem",
    x: List["pulp.LpVariable"],
    y: Optional[List["pulp.LpVariable"]],
    solve_status: int,
    n_candidates: int,
    n_test_points: int,
    use_full_coverage: bool,
    coverage_target_pct: float,
    use_conflict_constraints: bool,
    exclusion_factor: float,
    max_spacing: float,
    conflict_stats: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    """Extract solution from solved ILP and build statistics."""
    import pulp

    if solve_status != pulp.LpStatusOptimal:
        status_name = pulp.LpStatus.get(solve_status, "Unknown")
        if logger:
            logger.warning(f"⚠️ ILP solver status: {status_name}")
        return None, {"method": "ilp_failed", "reason": status_name}

    selected = [
        j
        for j in range(n_candidates)
        if x[j].value() is not None and x[j].value() > 0.5
    ]

    if use_full_coverage:
        covered, pct, form = n_test_points, 100.0, "full_coverage"
    else:
        covered = sum(1 for i in range(n_test_points) if y[i].value() == 1)
        pct = (covered / n_test_points * 100) if n_test_points else 0
        form = "partial_coverage"

    stats = {
        "method": "ilp",
        "formulation": form,
        "solver_status": "optimal",
        "objective_value": pulp.value(prob.objective),
        "coverage_target_pct": coverage_target_pct,
        "actual_coverage_pct": pct,
        "test_points_covered": covered,
        "test_points_total": n_test_points,
        "n_variables": (
            n_candidates if use_full_coverage else n_candidates + n_test_points
        ),
        "n_constraints": n_test_points if use_full_coverage else n_test_points + 1,
        "conflict_stats": conflict_stats,
        "conflict_constraints_enabled": use_conflict_constraints,
        "conflict_constraint_mode": conflict_stats.get("constraint_mode", "disabled"),
    }
    if use_conflict_constraints:
        stats["exclusion_factor"] = exclusion_factor
        stats["exclusion_dist_m"] = exclusion_factor * max_spacing
        stats["n_conflict_constraints"] = conflict_stats.get(
            "n_conflict_constraints", 0
        )

    if logger:
        logger.info(
            f"   ✅ ILP optimal: {len(selected)} boreholes, {pct:.1f}% coverage ({form})"
        )

    return selected, stats


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
) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    """Solve borehole placement using ILP. Minimizes boreholes with coverage target.

    Backend Selection:
        - If warm_start_indices is provided, uses native highspy for true warm start
        - Otherwise, uses PuLP (slightly more feature-rich logging)

    Args:
        warm_start_indices: Optional list of candidate indices to use as warm start.
            These are set to 1 initially, all others to 0. HiGHS will use this as
            the initial primal solution.
        highs_log_file: Optional path to write HiGHS solver output to a file.
    """
    # Use highspy backend when warm start is requested (PuLP doesn't support it properly)
    use_warm_start = warm_start_indices is not None and len(warm_start_indices) > 0
    if use_warm_start:
        try:
            import highspy  # noqa: F401

            if logger:
                logger.info("   🔧 Using native highspy backend (warm start enabled)")
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
            )
        except ImportError:
            if logger:
                logger.warning(
                    "   ⚠️ highspy not available, falling back to PuLP (no true warm start)"
                )

    # PuLP backend (default when no warm start)
    try:
        import pulp
    except ImportError:
        if logger:
            logger.warning("⚠️ PuLP not installed - cannot use ILP solver")
        return None, {"method": "ilp_failed", "reason": "PuLP not installed"}

    n_candidates, n_test_points = len(candidates), len(test_points)

    # Step 1: Create ILP problem with appropriate formulation
    prob, x, y, use_full_coverage, _ = _create_ilp_problem(
        n_candidates, n_test_points, coverage, coverage_target_pct, logger
    )

    # Step 1.5: Set warm start values if provided
    use_warm_start = warm_start_indices is not None and len(warm_start_indices) > 0
    if use_warm_start:
        warm_set = set(warm_start_indices)
        for j in range(n_candidates):
            x[j].setInitialValue(1 if j in warm_set else 0)
        if logger:
            logger.info(
                f"   🔥 Warm start: {len(warm_start_indices)} candidates seeded"
            )

    # Step 2: Add conflict constraints (optional)
    conflict_stats: Dict[str, Any] = {
        "constraint_mode": (
            conflict_constraint_mode if use_conflict_constraints else "disabled"
        ),
        "enabled": use_conflict_constraints,
    }

    if use_conflict_constraints:
        conflict_stats = _add_conflict_constraints(
            prob,
            x,
            candidates,
            conflict_constraint_mode,
            exclusion_factor,
            max_spacing,
            min_clique_size,
            max_cliques,
            max_conflict_pairs,
            n_test_points,
            logger,
        )

    # Step 3: Execute solver
    status, err = _execute_ilp_solver(
        prob,
        time_limit,
        mip_gap,
        threads,
        verbose,
        mip_heuristic_effort,
        warm_start=use_warm_start,
        logger=logger,
        highs_log_file=highs_log_file,
    )
    if err:
        return None, {"method": "ilp_failed", "reason": err}

    # Step 4: Extract solution and build statistics
    return _extract_ilp_solution(
        prob,
        x,
        y,
        status,
        n_candidates,
        n_test_points,
        use_full_coverage,
        coverage_target_pct,
        use_conflict_constraints,
        exclusion_factor,
        max_spacing,
        conflict_stats,
        logger,
    )


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

    Returns:
        Tuple of (selected_candidate_indices, stats_dict) or (None, error_stats)
    """
    try:
        import highspy
        import numpy as np
    except ImportError:
        if logger:
            logger.warning("⚠️ highspy not installed - cannot use native ILP solver")
        return None, {"method": "ilp_failed", "reason": "highspy not installed"}

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

    # === STEP 6: Solve ===
    h.run()

    # === STEP 7: Extract solution ===
    model_status = h.getModelStatus()

    if model_status == highspy.HighsModelStatus.kOptimal:
        sol = h.getSolution()
        info = h.getInfo()

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

        stats = {
            "method": "ilp_highspy",
            "formulation": "full_coverage" if use_full_coverage else "partial_coverage",
            "solver_status": "optimal",
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
        }

        if logger:
            logger.info(
                f"   ✅ ILP optimal (highspy): {len(selected)} boreholes, "
                f"{actual_pct:.1f}% coverage"
            )

        return selected, stats
    else:
        # Handle non-optimal status
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
