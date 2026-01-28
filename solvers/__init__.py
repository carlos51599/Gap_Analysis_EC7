"""
═══════════════════════════════════════════════════════════════════════════════
📦 SOLVERS PACKAGE
═══════════════════════════════════════════════════════════════════════════════

This package contains the borehole placement optimization components:

Modules:
- solver_config: SolverConfig dataclasses and factory functions
- solver_orchestration: Main compute_optimal_boreholes entry point
- solver_algorithms: ILP and greedy solver implementations
- optimization_geometry: Candidate grid generation and coverage mapping
- consolidation: Second pass consolidation (used by parallel/coverage_worker.py)

Public API:
- compute_optimal_boreholes: Main entry point for optimization
- optimize_boreholes: Simplified API using SolverConfig
- SolverConfig, GridConfig, ILPConfig, etc.: Configuration dataclasses
- create_default_config, create_fast_config, etc.: Factory functions
- generate_hexagon_grid_polygons: Hex grid visualization helper

Note: Border consolidation is handled by coverage_worker.py using
consolidate_boreholes() from consolidation.py, not in solver_orchestration.

Usage:
    from Gap_Analysis_EC7.solvers import compute_optimal_boreholes
    from Gap_Analysis_EC7.solvers import create_default_config, SolverConfig

    config = create_default_config(max_spacing=200.0)
    boreholes, stats = compute_optimal_boreholes(gaps, ...)

═══════════════════════════════════════════════════════════════════════════════
"""

# Configuration dataclasses and factories
from Gap_Analysis_EC7.solvers.solver_config import (
    GridConfig,
    ILPConfig,
    GreedyConfig,
    DecompositionConfig,
    ConflictConfig,
    SolverConfig,
    create_default_config,
    create_fast_config,
    create_parallel_config,
    create_precision_config,
    config_from_legacy_params,
    config_from_project_config,
    config_from_ilp_dict,
)

# Main orchestration entry points
from Gap_Analysis_EC7.solvers.solver_orchestration import (
    compute_optimal_boreholes,
    optimize_boreholes,
    verify_coverage,
)

# Solver algorithms (typically used internally)
from Gap_Analysis_EC7.solvers.solver_algorithms import (
    resolve_solver_mode,
    _solve_ilp,
    _solve_greedy,
    _get_best_solver,
)

# Geometry utilities
from Gap_Analysis_EC7.solvers.optimization_geometry import (
    generate_hexagon_grid_polygons,
    _normalize_gaps,
    _decompose_into_components,
    _generate_candidate_grid,
    _generate_test_points,
    _build_coverage_dict,
)

# Border consolidation moved to parallel workers
# (see Gap_Analysis_EC7/parallel/coverage_worker.py)

__all__ = [
    # Config classes
    "GridConfig",
    "ILPConfig",
    "GreedyConfig",
    "DecompositionConfig",
    "ConflictConfig",
    "SolverConfig",
    # Config factories
    "create_default_config",
    "create_fast_config",
    "create_parallel_config",
    "create_precision_config",
    "config_from_legacy_params",
    "config_from_project_config",
    "config_from_ilp_dict",
    # Orchestration
    "compute_optimal_boreholes",
    "optimize_boreholes",
    "verify_coverage",
    # Solvers
    "resolve_solver_mode",
    "_solve_ilp",
    "_solve_greedy",
    "_get_best_solver",
    # Geometry
    "generate_hexagon_grid_polygons",
    "_normalize_gaps",
    "_decompose_into_components",
    "_generate_candidate_grid",
    "_generate_test_points",
    "_build_coverage_dict",
]
