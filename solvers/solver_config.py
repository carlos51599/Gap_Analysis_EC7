"""
═══════════════════════════════════════════════════════════════════════════════
📋 SOLVER CONFIGURATION MODULE
═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURAL OVERVIEW
----------------------
Purpose: Centralized configuration dataclasses for the borehole optimizer.
         Reduces 31-parameter function signatures to structured config objects.

Key Interactions:
- solver_orchestration.py: Uses SolverConfig for compute_optimal_boreholes
- solvers.py: Uses ILPConfig for _solve_ilp, GreedyConfig for _solve_greedy
- optimization_geometry.py: Uses GridConfig for candidate generation

Design Philosophy:
- Immutable configs (frozen=True) prevent accidental modification
- Sensible defaults match EC7 borehole spacing requirements
- Grouped by concern: Grid, ILP, Greedy, Decomposition, Conflict

NAVIGATION GUIDE
----------------
# ═════ 1. GRID CONFIGURATION
# ═════ 2. ILP SOLVER CONFIGURATION
# ═════ 3. GREEDY SOLVER CONFIGURATION
# ═════ 4. DECOMPOSITION CONFIGURATION
# ═════ 5. CONFLICT CONSTRAINT CONFIGURATION
# ═════ 6. MAIN SOLVER CONFIGURATION (FACADE)
# ═════ 7. FACTORY FUNCTIONS

═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import logging


# ═══════════════════════════════════════════════════════════════════════════════
# 📐 1. GRID CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GridConfig:
    """
    Configuration for candidate grid generation.

    The grid determines potential borehole locations. Hexagonal grids provide
    ~15% better coverage than rectangular grids at the same density.

    Attributes:
        candidate_spacing: Grid spacing for candidate boreholes (m). Smaller = more
            candidates = better coverage but slower solve. Default 50m.
        test_spacing: Grid spacing for test points within gaps (m). Used to
            evaluate coverage. Smaller = more accurate but slower. Default 15m.
        grid_type: "hexagonal" (optimal packing) or "rectangular" (legacy).
        hexagonal_density: Density multiplier for hexagonal grid. 1.0 = standard,
            1.5 = 50% more candidates in dense areas. Default 1.5.
    """

    candidate_spacing: float = 50.0
    test_spacing: float = 15.0
    grid_type: str = "hexagonal"
    hexagonal_density: float = 1.5

    def __post_init__(self) -> None:
        """Validate grid configuration."""
        if self.candidate_spacing <= 0:
            raise ValueError(
                f"candidate_spacing must be > 0, got {self.candidate_spacing}"
            )
        if self.test_spacing <= 0:
            raise ValueError(f"test_spacing must be > 0, got {self.test_spacing}")
        if self.grid_type not in ("hexagonal", "rectangular"):
            raise ValueError(
                f"grid_type must be 'hexagonal' or 'rectangular', got {self.grid_type}"
            )
        if self.hexagonal_density <= 0:
            raise ValueError(
                f"hexagonal_density must be > 0, got {self.hexagonal_density}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 🛑 2a. STALL DETECTION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StallDetectionConfig:
    """
    Configuration for HiGHS solver stall detection (early termination).

    Stall detection monitors MIP gap improvement across HiGHS log rows.
    When the gap fails to improve by min_improvement_pct over comparison_window
    log rows, the solver terminates early with the current best solution.

    This is critical for preventing long runs when the solver is stuck
    proving optimality without finding better solutions.

    Attributes:
        enabled: Master switch for stall detection. Default True.
        warmup_solutions: Number of log rows to skip before checking (warmup
            phase). Each log row is ~5 seconds. Default 1.
        comparison_window: Compare gap to this many log rows ago. Default 2.
        min_improvement_pct: Minimum absolute gap improvement (percentage points)
            required. If gap drops from 15% to 14%, that's 1 point improvement.
            Default 5.0 points.
    """

    enabled: bool = True
    warmup_solutions: int = 1
    comparison_window: int = 2
    min_improvement_pct: float = 5.0

    def __post_init__(self) -> None:
        """Validate stall detection configuration."""
        if self.warmup_solutions < 0:
            raise ValueError(
                f"warmup_solutions must be >= 0, got {self.warmup_solutions}"
            )
        if self.comparison_window < 1:
            raise ValueError(
                f"comparison_window must be >= 1, got {self.comparison_window}"
            )
        if self.min_improvement_pct < 0:
            raise ValueError(
                f"min_improvement_pct must be >= 0, got {self.min_improvement_pct}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for passing to _solve_ilp_highspy."""
        return {
            "enabled": self.enabled,
            "warmup_solutions": self.warmup_solutions,
            "comparison_window": self.comparison_window,
            "min_improvement_pct": self.min_improvement_pct,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StallDetectionConfig":
        """Create from dictionary (e.g., from CONFIG)."""
        return cls(
            enabled=d.get("enabled", True),
            warmup_solutions=d.get("warmup_solutions", 1),
            comparison_window=d.get("comparison_window", 2),
            min_improvement_pct=d.get("min_improvement_pct", 5.0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 🧮 2b. ILP SOLVER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ILPConfig:
    """
    Configuration for Integer Linear Programming solver.

    ILP provides optimal solutions but may be slow for large problems.
    HiGHS is preferred (2-5× faster than CBC).

    Attributes:
        time_limit: Maximum solve time in seconds. Default 120s.
        mip_gap: MIP gap tolerance (0.03 = 3% suboptimality allowed). Lower =
            closer to optimal but slower. Default 0.03.
        threads: Solver threads. Keep at 1 for multiprocessing; parallelize at
            higher level instead. Default 1.
        verbose: Solver verbosity (0=silent, 1=moderate, 2+=detailed).
        mip_heuristic_effort: Fraction of solve time for primal heuristics.
            Higher = better initial solutions but less branching. Default 0.05.
        stall_detection: Configuration for early termination on solver stall.
    """

    time_limit: int = 120
    mip_gap: float = 0.03
    threads: int = 1
    verbose: int = 0
    mip_heuristic_effort: float = 0.05
    stall_detection: StallDetectionConfig = field(
        default_factory=StallDetectionConfig
    )

    def __post_init__(self) -> None:
        """Validate ILP configuration."""
        if self.time_limit <= 0:
            raise ValueError(f"time_limit must be > 0, got {self.time_limit}")
        if not 0 <= self.mip_gap <= 1:
            raise ValueError(f"mip_gap must be in [0, 1], got {self.mip_gap}")
        if self.threads < 1:
            raise ValueError(f"threads must be >= 1, got {self.threads}")


# ═══════════════════════════════════════════════════════════════════════════════
# 🔄 3. GREEDY SOLVER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GreedyConfig:
    """
    Configuration for greedy heuristic solver.

    Greedy is fast but may not find optimal solutions. Used as fallback
    when ILP times out or for parallel contexts.

    Attributes:
        max_iterations: Maximum iterations. Each iteration places one borehole.
            Default 1000.
        min_gain: Minimum area gain (m²) per iteration to continue. Stops early
            if remaining coverage gains are negligible. Default 1.0.
        min_efficiency_pct: Minimum coverage efficiency to continue. Stops if
            the best candidate covers less than this percentage of its
            theoretical maximum. Default 8.0.
        fill_remaining: After main optimization, fill any remaining fragments
            with additional greedy passes. Default False.
    """

    max_iterations: int = 1000
    min_gain: float = 1.0
    min_efficiency_pct: float = 8.0
    fill_remaining: bool = False

    def __post_init__(self) -> None:
        """Validate greedy configuration."""
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.min_gain < 0:
            raise ValueError(f"min_gain must be >= 0, got {self.min_gain}")
        if not 0 <= self.min_efficiency_pct <= 100:
            raise ValueError(
                f"min_efficiency_pct must be in [0, 100], got {self.min_efficiency_pct}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 🗂️ 4. DECOMPOSITION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DecompositionConfig:
    """
    Configuration for problem decomposition strategies.

    Decomposition breaks large problems into smaller, independent subproblems.
    Hierarchy: Zone → Connected Components → Single Solve

    Attributes:
        use_zone_decomposition: Split gaps by zone boundaries first. Requires
            zones_gdf to be provided at solve time. Default False.
        zones_gdf: GeoDataFrame with zone polygon geometries. If None when
            use_zone_decomposition=True, zone decomposition is skipped with
            a warning. Default None.
        min_zone_gap_area_m2: Minimum gap area (m²) within a zone to process.
            Smaller gaps are skipped. Default 100.
        use_connected_components: Decompose by spatial proximity. Gaps that
            don't overlap when buffered by max_spacing are independent.
            Default True.
        min_component_gaps_for_ilp: Minimum gaps per component to use ILP.
            Smaller components use greedy. Default 1.
        zone_cache_dir: Path to shared zone cache directory for intra-run caching.
            When set, identical zone-gap geometries across filter combinations
            reuse cached ILP results. Default None (caching disabled).
    """

    use_zone_decomposition: bool = False
    zones_gdf: Optional[Any] = None  # GeoDataFrame
    min_zone_gap_area_m2: float = 100.0
    use_connected_components: bool = True
    min_component_gaps_for_ilp: int = 1
    zone_cache_dir: Optional[str] = None  # Path to shared zone cache

    def __post_init__(self) -> None:
        """Validate decomposition configuration."""
        # NOTE: We no longer require zones_gdf at config creation time.
        # If use_zone_decomposition=True but zones_gdf=None, the solver
        # orchestration will skip zone decomposition with a warning.
        # This allows creating a config from project CONFIG and providing
        # zones_gdf later when calling the optimizer.
        if self.min_zone_gap_area_m2 < 0:
            raise ValueError(
                f"min_zone_gap_area_m2 must be >= 0, got {self.min_zone_gap_area_m2}"
            )
        if self.min_component_gaps_for_ilp < 1:
            raise ValueError(
                f"min_component_gaps_for_ilp must be >= 1, got {self.min_component_gaps_for_ilp}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ⚔️ 5. CONFLICT CONSTRAINT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ConflictConfig:
    """
    Configuration for conflict constraints (spacing enforcement).

    Conflict constraints ensure boreholes maintain minimum separation.
    EC7 requires 200m spacing; exclusion_factor=0.8 → 160m minimum.

    Attributes:
        enabled: Enable conflict constraints. Default True.
        mode: "clique" (stronger, O(n³) worst case) or "pairwise" (simpler,
            O(n²)). Clique mode finds maximal cliques and adds sum(x) <= 1
            constraints. Default "clique".
        exclusion_factor: Minimum separation as fraction of max_spacing.
            0.8 = 80% of max_spacing = 160m for 200m spacing. Default 0.8.
        max_conflict_pairs: Maximum conflict pairs for pairwise mode.
            Prevents explosion on dense grids. Default 200000.
        min_clique_size: Minimum clique size for clique mode. Smaller cliques
            are dropped. Default 3.
        max_cliques: Maximum cliques to enumerate. Default 50000.
    """

    enabled: bool = True
    mode: str = "clique"
    exclusion_factor: float = 0.8
    max_conflict_pairs: int = 200000
    min_clique_size: int = 3
    max_cliques: int = 50000

    def __post_init__(self) -> None:
        """Validate conflict configuration."""
        if self.mode not in ("clique", "pairwise"):
            raise ValueError(f"mode must be 'clique' or 'pairwise', got {self.mode}")
        if not 0 < self.exclusion_factor <= 1:
            raise ValueError(
                f"exclusion_factor must be in (0, 1], got {self.exclusion_factor}"
            )
        if self.max_conflict_pairs < 0:
            raise ValueError(
                f"max_conflict_pairs must be >= 0, got {self.max_conflict_pairs}"
            )
        if self.min_clique_size < 2:
            raise ValueError(
                f"min_clique_size must be >= 2, got {self.min_clique_size}"
            )
        if self.max_cliques < 0:
            raise ValueError(f"max_cliques must be >= 0, got {self.max_cliques}")


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 6. MAIN SOLVER CONFIGURATION (FACADE)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SolverConfig:
    """
    Main configuration facade for compute_optimal_boreholes.

    Groups all configuration into logical sub-configs. Use factory functions
    for common configurations.

    Attributes:
        max_spacing: Coverage radius per borehole (m). EC7 default 200m.
        coverage_target_pct: Target coverage percentage. 100 = full coverage.
        solver_mode: "ilp" | "greedy" | "auto" | None.
            - "ilp": Use ILP, fallback to greedy on failure.
            - "greedy": Use greedy only.
            - "auto": ILP for single runs, greedy for parallel.
            - None: Use legacy use_ilp boolean.
        is_parallel_context: True if running in parallel worker.
        grid: Grid generation configuration.
        ilp: ILP solver configuration.
        greedy: Greedy solver configuration.
        decomposition: Problem decomposition configuration.
        conflict: Conflict constraint configuration.
        border_consolidation: Border consolidation config dict (optional).
        highs_log_folder: Optional folder path to write HiGHS solver logs.
        logger: Optional logger instance.

    Example:
        >>> config = SolverConfig.default()
        >>> boreholes, stats = compute_optimal_boreholes(gaps, config=config)

        >>> fast_config = SolverConfig.fast_greedy()
        >>> boreholes, stats = compute_optimal_boreholes(gaps, config=fast_config)
    """

    max_spacing: float = 200.0
    coverage_target_pct: float = 100.0
    solver_mode: Optional[str] = "ilp"
    is_parallel_context: bool = False
    grid: GridConfig = field(default_factory=GridConfig)
    ilp: ILPConfig = field(default_factory=ILPConfig)
    greedy: GreedyConfig = field(default_factory=GreedyConfig)
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)
    border_consolidation: Optional[Dict[str, Any]] = (
        None  # Raw config dict for border consolidation
    )
    highs_log_folder: Optional[str] = None  # Folder for HiGHS solver log files
    logger: Optional[logging.Logger] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Validate main configuration."""
        if self.max_spacing <= 0:
            raise ValueError(f"max_spacing must be > 0, got {self.max_spacing}")
        if not 0 < self.coverage_target_pct <= 100:
            raise ValueError(
                f"coverage_target_pct must be in (0, 100], got {self.coverage_target_pct}"
            )
        if self.solver_mode not in (None, "ilp", "greedy", "auto"):
            raise ValueError(
                f"solver_mode must be None, 'ilp', 'greedy', or 'auto', got {self.solver_mode}"
            )

    def with_modifications(
        self,
        max_spacing: Optional[float] = None,
        candidate_spacing: Optional[float] = None,
        test_spacing: Optional[float] = None,
        use_zone_decomposition: Optional[bool] = None,
        zones_gdf: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "SolverConfig":
        """
        Create a modified copy of this config.

        Used by zone decomposition to adjust spacing parameters for each zone
        while preserving all other settings. Since SolverConfig is frozen
        (immutable), this method creates a new instance with the changes.

        Args:
            max_spacing: Override max_spacing if provided
            candidate_spacing: Override grid.candidate_spacing if provided
            test_spacing: Override grid.test_spacing if provided
            use_zone_decomposition: Override decomposition.use_zone_decomposition
            zones_gdf: Override decomposition.zones_gdf (use object() sentinel
                       to explicitly set to None vs not changing)
            logger: Override logger (use object() sentinel to set to None)

        Returns:
            New SolverConfig with specified modifications

        Example:
            >>> zone_config = config.with_modifications(
            ...     max_spacing=150.0,
            ...     candidate_spacing=75.0,
            ...     use_zone_decomposition=False,
            ... )
        """
        # Build modified GridConfig if spacing changed
        new_grid = GridConfig(
            candidate_spacing=(
                candidate_spacing
                if candidate_spacing is not None
                else self.grid.candidate_spacing
            ),
            test_spacing=(
                test_spacing if test_spacing is not None else self.grid.test_spacing
            ),
            grid_type=self.grid.grid_type,
            hexagonal_density=self.grid.hexagonal_density,
        )

        # Build modified DecompositionConfig if decomposition settings changed
        new_decomposition = DecompositionConfig(
            use_connected_components=self.decomposition.use_connected_components,
            min_component_gaps_for_ilp=self.decomposition.min_component_gaps_for_ilp,
            use_zone_decomposition=(
                use_zone_decomposition
                if use_zone_decomposition is not None
                else self.decomposition.use_zone_decomposition
            ),
            zones_gdf=(
                zones_gdf if zones_gdf is not None else self.decomposition.zones_gdf
            ),
            min_zone_gap_area_m2=self.decomposition.min_zone_gap_area_m2,
        )

        return SolverConfig(
            max_spacing=max_spacing if max_spacing is not None else self.max_spacing,
            coverage_target_pct=self.coverage_target_pct,
            solver_mode=self.solver_mode,
            is_parallel_context=self.is_parallel_context,
            grid=new_grid,
            ilp=self.ilp,
            greedy=self.greedy,
            decomposition=new_decomposition,
            conflict=self.conflict,
            logger=logger if logger is not None else self.logger,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 7. FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def create_default_config(
    max_spacing: float = 200.0,
    verbose: int = 0,
    logger: Optional[logging.Logger] = None,
) -> SolverConfig:
    """
    Create default EC7-compliant solver configuration.

    Uses ILP solver with 200m spacing, connected components decomposition,
    and clique-based conflict constraints.

    Args:
        max_spacing: Coverage radius per borehole. Default 200m for EC7.
        verbose: Solver verbosity level. Default 0 (silent).
        logger: Optional logger instance.

    Returns:
        SolverConfig with sensible defaults.
    """
    return SolverConfig(
        max_spacing=max_spacing,
        solver_mode="ilp",
        ilp=ILPConfig(verbose=verbose),
        logger=logger,
    )


def create_fast_config(
    max_spacing: float = 200.0,
    verbose: int = 0,
    logger: Optional[logging.Logger] = None,
) -> SolverConfig:
    """
    Create fast configuration for quick results.

    Uses greedy solver with relaxed stopping criteria.
    ~10× faster than default but may not be optimal.

    Args:
        max_spacing: Coverage radius per borehole. Default 200m.
        verbose: Solver verbosity level. Default 0 (silent).
        logger: Optional logger instance.

    Returns:
        SolverConfig optimized for speed.
    """
    return SolverConfig(
        max_spacing=max_spacing,
        solver_mode="greedy",
        greedy=GreedyConfig(
            max_iterations=500,
            min_gain=5.0,
            min_efficiency_pct=10.0,
        ),
        decomposition=DecompositionConfig(
            use_connected_components=True,
            min_component_gaps_for_ilp=999999,  # Never use ILP
        ),
        conflict=ConflictConfig(enabled=False),  # Skip conflict constraints
        ilp=ILPConfig(verbose=verbose),
        logger=logger,
    )


def create_parallel_config(
    max_spacing: float = 200.0,
    logger: Optional[logging.Logger] = None,
) -> SolverConfig:
    """
    Create configuration for parallel worker contexts.

    Uses greedy solver (safer for multiprocessing), single thread,
    no zone decomposition.

    Args:
        max_spacing: Coverage radius per borehole. Default 200m.
        logger: Optional logger instance.

    Returns:
        SolverConfig safe for multiprocessing.
    """
    return SolverConfig(
        max_spacing=max_spacing,
        solver_mode="greedy",
        is_parallel_context=True,
        ilp=ILPConfig(threads=1, verbose=0),
        decomposition=DecompositionConfig(
            use_zone_decomposition=False,
            use_connected_components=False,
        ),
        logger=logger,
    )


def create_precision_config(
    max_spacing: float = 200.0,
    time_limit: int = 600,
    verbose: int = 1,
    logger: Optional[logging.Logger] = None,
) -> SolverConfig:
    """
    Create precision configuration for optimal results.

    Uses ILP with tight MIP gap, longer time limit, finer grid.
    ~5-10× slower than default but produces near-optimal solutions.

    Args:
        max_spacing: Coverage radius per borehole. Default 200m.
        time_limit: Maximum ILP solve time. Default 600s (10 minutes).
        verbose: Solver verbosity level. Default 1.
        logger: Optional logger instance.

    Returns:
        SolverConfig optimized for solution quality.
    """
    return SolverConfig(
        max_spacing=max_spacing,
        solver_mode="ilp",
        grid=GridConfig(
            candidate_spacing=30.0,  # Finer grid
            test_spacing=10.0,  # More test points
            hexagonal_density=2.0,  # Higher density
        ),
        ilp=ILPConfig(
            time_limit=time_limit,
            mip_gap=0.01,  # Tighter gap
            verbose=verbose,
            mip_heuristic_effort=0.1,  # More heuristic effort
        ),
        greedy=GreedyConfig(fill_remaining=True),  # Fill gaps
        conflict=ConflictConfig(
            mode="clique",
            min_clique_size=2,  # Include smaller cliques
            max_cliques=100000,
        ),
        logger=logger,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 🔄 LEGACY PARAMETER CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════


def config_from_project_config(
    config_dict: Dict[str, Any],
    max_spacing: Optional[float] = None,
    zones_gdf: Optional[Any] = None,
    is_parallel_context: bool = False,
    logger: Optional[logging.Logger] = None,
) -> SolverConfig:
    """
    Create SolverConfig from project-level CONFIG dictionary.

    This is the RECOMMENDED way to create configuration - it ensures all
    project settings (including environment variable overrides) are honored.

    Args:
        config_dict: The project CONFIG dictionary from config.py.
        max_spacing: Override max_spacing (uses config_dict["max_spacing_m"] if None).
        zones_gdf: GeoDataFrame with zone geometries for zone decomposition.
        is_parallel_context: True if running in parallel worker context.
        logger: Optional logger instance.

    Returns:
        SolverConfig populated from project CONFIG settings.

    Example:
        >>> from Gap_Analysis_EC7.config import CONFIG
        >>> config = config_from_project_config(CONFIG, zones_gdf=zones_gdf)
        >>> boreholes, stats = compute_optimal_boreholes(gaps, config=config)
    """
    # Extract sub-dictionaries
    ilp_config = config_dict.get("ilp_solver", {})
    greedy_config = config_dict.get("greedy_solver", {})
    grid_config = config_dict.get("candidate_grid", {})
    opt_config = config_dict.get("optimization", {})

    # Resolve max_spacing (parameter override > config > default)
    effective_max_spacing = max_spacing or config_dict.get("max_spacing_m", 100.0)

    # Compute candidate/test spacing from multipliers
    candidate_mult = ilp_config.get("candidate_spacing_mult", 0.5)
    test_mult = ilp_config.get("test_spacing_mult", 0.2)
    candidate_spacing = effective_max_spacing * candidate_mult
    test_spacing = effective_max_spacing * test_mult

    # Resolve solver mode
    solver_mode = opt_config.get("solver_mode", "ilp")

    # Apply parallel overrides if in parallel context
    if is_parallel_context:
        parallel_overrides = config_dict.get("parallel_solver_overrides", {})
        if parallel_overrides.get("solver_mode"):
            solver_mode = parallel_overrides["solver_mode"]

    # Extract border consolidation config (raw dict for post-processing)
    border_consolidation_config = config_dict.get("border_consolidation", None)

    return SolverConfig(
        max_spacing=effective_max_spacing,
        coverage_target_pct=ilp_config.get("coverage_target_pct", 100.0),
        solver_mode=solver_mode,
        is_parallel_context=is_parallel_context,
        grid=GridConfig(
            candidate_spacing=candidate_spacing,
            test_spacing=test_spacing,
            grid_type=grid_config.get("grid_type", "hexagonal"),
            hexagonal_density=grid_config.get("hexagonal_density", 1.5),
        ),
        ilp=ILPConfig(
            time_limit=ilp_config.get("time_limit_s", 120),
            mip_gap=ilp_config.get("mip_gap", 0.15),
            threads=ilp_config.get("threads", 1),
            verbose=ilp_config.get("verbose", 1),
            mip_heuristic_effort=ilp_config.get("mip_heuristic_effort", 0.05),
        ),
        greedy=GreedyConfig(
            max_iterations=greedy_config.get("max_iterations", 1000),
            min_gain=greedy_config.get("min_coverage_gain_m2", 10000.0),
            min_efficiency_pct=greedy_config.get("min_efficiency_pct", 8.0),
            fill_remaining=ilp_config.get("fill_remaining_fragments", False),
        ),
        decomposition=DecompositionConfig(
            use_zone_decomposition=ilp_config.get("use_zone_decomposition", True),
            zones_gdf=zones_gdf,
            min_zone_gap_area_m2=ilp_config.get("min_zone_gap_area_m2", 100.0),
            use_connected_components=ilp_config.get("use_connected_components", False),
            min_component_gaps_for_ilp=ilp_config.get("min_component_gaps_for_ilp", 1),
        ),
        conflict=ConflictConfig(
            enabled=ilp_config.get("use_conflict_constraints", True),
            mode=ilp_config.get("conflict_constraint_mode", "pairwise"),
            exclusion_factor=ilp_config.get("exclusion_factor", 0.8),
            max_conflict_pairs=ilp_config.get("max_conflict_pairs", 200000),
            min_clique_size=ilp_config.get("min_clique_size", 3),
            max_cliques=ilp_config.get("max_cliques", 50000),
        ),
        border_consolidation=border_consolidation_config,
        logger=logger,
    )


def config_from_legacy_params(
    max_spacing: float = 200.0,
    candidate_spacing: float = 50.0,
    test_spacing: float = 15.0,
    time_limit: int = 120,
    mip_gap: float = 0.03,
    coverage_target_pct: float = 100.0,
    threads: int = 1,
    use_ilp: bool = True,
    solver_mode: Optional[str] = None,
    is_parallel_context: bool = False,
    fill_remaining_fragments: bool = False,
    greedy_max_iterations: int = 1000,
    greedy_min_gain: float = 1.0,
    greedy_min_efficiency_pct: float = 8.0,
    candidate_grid_type: str = "hexagonal",
    hexagonal_density: float = 1.5,
    use_conflict_constraints: bool = True,
    conflict_constraint_mode: str = "clique",
    exclusion_factor: float = 0.8,
    max_conflict_pairs: int = 200000,
    min_clique_size: int = 3,
    max_cliques: int = 50000,
    use_connected_components: bool = True,
    min_component_gaps_for_ilp: int = 1,
    use_zone_decomposition: bool = False,
    zones_gdf: Optional[Any] = None,
    min_zone_gap_area_m2: float = 100.0,
    verbose: int = 0,
    mip_heuristic_effort: float = 0.05,
    logger: Optional[logging.Logger] = None,
) -> SolverConfig:
    """
    Convert legacy 31-parameter signature to SolverConfig.

    This enables backwards compatibility - existing code can continue
    using positional/keyword arguments, which get converted to config.

    Args:
        All legacy parameters from compute_optimal_boreholes.

    Returns:
        SolverConfig equivalent to the legacy parameters.
    """
    # Resolve solver_mode from legacy use_ilp if not specified
    effective_solver_mode = solver_mode
    if solver_mode is None:
        effective_solver_mode = "ilp" if use_ilp else "greedy"

    return SolverConfig(
        max_spacing=max_spacing,
        coverage_target_pct=coverage_target_pct,
        solver_mode=effective_solver_mode,
        is_parallel_context=is_parallel_context,
        grid=GridConfig(
            candidate_spacing=candidate_spacing,
            test_spacing=test_spacing,
            grid_type=candidate_grid_type,
            hexagonal_density=hexagonal_density,
        ),
        ilp=ILPConfig(
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            verbose=verbose,
            mip_heuristic_effort=mip_heuristic_effort,
        ),
        greedy=GreedyConfig(
            max_iterations=greedy_max_iterations,
            min_gain=greedy_min_gain,
            min_efficiency_pct=greedy_min_efficiency_pct,
            fill_remaining=fill_remaining_fragments,
        ),
        decomposition=DecompositionConfig(
            use_zone_decomposition=use_zone_decomposition,
            zones_gdf=zones_gdf,
            min_zone_gap_area_m2=min_zone_gap_area_m2,
            use_connected_components=use_connected_components,
            min_component_gaps_for_ilp=min_component_gaps_for_ilp,
        ),
        conflict=ConflictConfig(
            enabled=use_conflict_constraints,
            mode=conflict_constraint_mode,
            exclusion_factor=exclusion_factor,
            max_conflict_pairs=max_conflict_pairs,
            min_clique_size=min_clique_size,
            max_cliques=max_cliques,
        ),
        logger=logger,
    )


def config_from_ilp_dict(
    ilp_config: Dict[str, Any],
    max_spacing: float,
    zones_gdf: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    highs_log_folder: Optional[str] = None,
) -> SolverConfig:
    """
    Create SolverConfig from the ILP config dict used in parallel workers.

    This bridges the gap between coverage_worker.py's dict-based config
    and the typed SolverConfig API. The ilp_config dict is produced by
    _build_solver_config() in coverage_orchestrator.py.

    Args:
        ilp_config: Dict with solver settings, including nested dicts:
            - "greedy_solver": GreedyConfig fields
            - "candidate_grid": GridConfig fields
            Plus flat fields for ILP, conflict, and decomposition settings.
        max_spacing: EC7 maximum borehole spacing in meters
        zones_gdf: Optional GeoDataFrame for zone decomposition
        logger: Optional logger for solver output
        highs_log_folder: Optional folder path for HiGHS solver log files

    Returns:
        Fully configured SolverConfig object

    Example:
        >>> # In coverage_worker.py:
        >>> config = config_from_ilp_dict(ilp_config, max_spacing, zones_gdf)
        >>> boreholes, stats = optimize_boreholes(gaps, config)
    """
    greedy_dict = ilp_config.get("greedy_solver", {}) or {}
    grid_dict = ilp_config.get("candidate_grid", {}) or {}
    stall_dict = ilp_config.get("stall_detection", {}) or {}

    return SolverConfig(
        max_spacing=max_spacing,
        coverage_target_pct=ilp_config.get("coverage_target_pct", 100.0),
        solver_mode=ilp_config.get("solver_mode", None),
        is_parallel_context=True,  # Always True for worker context
        grid=GridConfig(
            candidate_spacing=ilp_config.get("candidate_spacing_m", max_spacing * 0.5),
            test_spacing=ilp_config.get("test_spacing_m", max_spacing * 0.2),
            grid_type=grid_dict.get("grid_type", "hexagonal"),
            hexagonal_density=grid_dict.get("hexagonal_density", 1.5),
        ),
        ilp=ILPConfig(
            time_limit=ilp_config.get("time_limit_s", 30),
            mip_gap=ilp_config.get("mip_gap", 0.05),
            threads=1,  # Single thread per worker (critical for parallel safety)
            verbose=ilp_config.get("verbose", 0),
            mip_heuristic_effort=ilp_config.get("mip_heuristic_effort", 0.05),
            stall_detection=StallDetectionConfig.from_dict(stall_dict),
        ),
        greedy=GreedyConfig(
            max_iterations=greedy_dict.get("max_iterations", 1000),
            min_gain=greedy_dict.get("min_coverage_gain_m2", 10000.0),
            min_efficiency_pct=greedy_dict.get("min_efficiency_pct", 8.0),
            fill_remaining=False,
        ),
        decomposition=DecompositionConfig(
            use_zone_decomposition=ilp_config.get("use_zone_decomposition", False),
            zones_gdf=zones_gdf,
            min_zone_gap_area_m2=ilp_config.get("min_zone_gap_area_m2", 100.0),
            use_connected_components=ilp_config.get("use_connected_components", False),
            min_component_gaps_for_ilp=ilp_config.get("min_component_gaps_for_ilp", 1),
            zone_cache_dir=ilp_config.get("zone_cache_dir"),  # Intra-run caching
        ),
        conflict=ConflictConfig(
            enabled=ilp_config.get("use_conflict_constraints", True),
            mode=ilp_config.get("conflict_constraint_mode", "clique"),
            exclusion_factor=ilp_config.get("exclusion_factor", 0.8),
            min_clique_size=ilp_config.get("min_clique_size", 3),
            max_cliques=ilp_config.get("max_cliques", 50000),
            max_conflict_pairs=ilp_config.get("max_conflict_pairs", 200000),
        ),
        highs_log_folder=highs_log_folder,
        logger=logger,
    )
