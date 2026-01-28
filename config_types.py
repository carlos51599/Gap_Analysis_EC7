"""
═══════════════════════════════════════════════════════════════════════════════
📋 UNIFIED CONFIGURATION TYPES
═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURAL OVERVIEW
----------------------
Responsibility: Define all configuration dataclasses for the EC7 application.
Replaces scattered CONFIG dictionary access with typed, validated config objects.

This is Phase 1 of the configuration unification plan. It provides:
1. Type-safe configuration dataclasses
2. A single AppConfig facade that wraps all settings
3. Factory methods to create configs from the existing CONFIG dictionary

Usage:
    from Gap_Analysis_EC7.config import CONFIG
    from Gap_Analysis_EC7.config_types import AppConfig

    # Create once at application startup
    app_config = AppConfig.from_dict(CONFIG)

    # Use throughout the application
    solver_config = app_config.solver_config(zones_gdf=zones_gdf)

For Navigation: Use VS Code outline (Ctrl+Shift+O)

NAVIGATION GUIDE
----------------
# ═════ 1. FILE PATHS CONFIGURATION
# ═════ 2. TESTING MODE CONFIGURATION
# ═════ 3. CANDIDATE GRID CONFIGURATION
# ═════ 4. VISUALIZATION CONFIGURATION
# ═════ 5. PARALLEL PROCESSING CONFIGURATION
# ═════ 6. CACHE CONFIGURATION
# ═════ 7. QUALITY CONTROL CONFIGURATION
# ═════ 7b. BORDER CONSOLIDATION CONFIGURATION
# ═════ 8. APP CONFIG (MASTER FACADE)

═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import logging


# ═══════════════════════════════════════════════════════════════════════════════
# 📁 1. FILE PATHS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FilePathsConfig:
    """
    File path configuration for inputs and outputs.

    Attributes:
        embankment_shp: Path to embankment/zone shapefile.
        boreholes_csv: Path to borehole locations CSV.
        output_dir: Directory for output files.
        log_dir: Directory for log files.
    """

    embankment_shp: str = ""
    boreholes_csv: str = ""
    output_dir: str = "Output"
    log_dir: str = "logs"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FilePathsConfig":
        """Create FilePathsConfig from CONFIG['file_paths'] dictionary."""
        return cls(
            embankment_shp=d.get("embankment_shp", ""),
            boreholes_csv=d.get("boreholes_csv", ""),
            output_dir=d.get("output_dir", "Output"),
            log_dir=d.get("log_dir", "logs"),
        )

    @property
    def output_path(self) -> Path:
        """Get output directory as relative Path object."""
        return Path(self.output_dir)

    @property
    def log_path(self) -> Path:
        """Get log directory as relative Path object."""
        return Path(self.log_dir)

    def output_dir_path(self, workspace_root: Path) -> Path:
        """Get output directory resolved against workspace root."""
        return workspace_root / self.output_dir

    def log_dir_path(self, workspace_root: Path) -> Path:
        """Get log directory resolved against workspace root."""
        return workspace_root / self.log_dir


# ═══════════════════════════════════════════════════════════════════════════════
# 🧪 2. TESTING MODE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TestingFilterConfig:
    """Filter settings for testing mode."""

    min_depth: int = 0
    require_spt: bool = False
    require_triaxial_total: bool = False
    require_triaxial_effective: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestingFilterConfig":
        """Create TestingFilterConfig from filter dictionary."""
        return cls(
            min_depth=d.get("min_depth", 0),
            require_spt=d.get("require_spt", False),
            require_triaxial_total=d.get("require_triaxial_total", False),
            require_triaxial_effective=d.get("require_triaxial_effective", False),
        )


@dataclass(frozen=True)
class TestingModeConfig:
    """
    Configuration for testing/debug mode.

    When testing mode is enabled, it overrides various settings to create
    a controlled, reproducible test environment.

    Attributes:
        enabled: Master toggle for testing mode.
        filter: Fixed filter settings for testing.
        force_cache_overwrite: Always recompute instead of using cache.
        force_single_worker: Run with single worker for debugging.
        solver_overrides: Override solver settings for testing.
    """

    enabled: bool = False
    filter: TestingFilterConfig = field(default_factory=TestingFilterConfig)
    force_cache_overwrite: bool = True
    force_single_worker: bool = True
    solver_overrides: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestingModeConfig":
        """Create TestingModeConfig from CONFIG['testing_mode'] dictionary."""
        return cls(
            enabled=d.get("enabled", False),
            filter=TestingFilterConfig.from_dict(d.get("filter", {})),
            force_cache_overwrite=d.get("force_cache_overwrite", True),
            force_single_worker=d.get("force_single_worker", True),
            solver_overrides=d.get("solver_overrides", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 🔷 3. CANDIDATE GRID CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CandidateGridConfig:
    """
    Configuration for candidate borehole grid generation.

    Attributes:
        grid_type: "hexagonal" (optimal packing) or "rectangular" (legacy).
        hexagonal_density: Density factor for hexagonal grid (1.0-2.0).
    """

    grid_type: str = "hexagonal"
    hexagonal_density: float = 1.5

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidateGridConfig":
        """Create CandidateGridConfig from CONFIG['candidate_grid'] dictionary."""
        return cls(
            grid_type=d.get("grid_type", "hexagonal"),
            hexagonal_density=d.get("hexagonal_density", 1.5),
        )

    def __post_init__(self) -> None:
        """Validate grid configuration."""
        if self.grid_type not in ("hexagonal", "rectangular"):
            raise ValueError(
                f"grid_type must be 'hexagonal' or 'rectangular', got {self.grid_type}"
            )
        if self.hexagonal_density <= 0:
            raise ValueError(
                f"hexagonal_density must be > 0, got {self.hexagonal_density}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 4. VISUALIZATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BoreholeMarkerConfig:
    """Styling configuration for borehole markers on maps."""

    size: int = 6
    color: str = "rgba(0, 0, 0, 0.82)"
    symbol: str = "circle"
    line_color: str = "rgba(0, 0, 0, 0)"
    line_width: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoreholeMarkerConfig":
        """Create BoreholeMarkerConfig from dictionary."""
        return cls(
            size=d.get("size", 6),
            color=d.get("color", "rgba(0, 0, 0, 0.82)"),
            symbol=d.get("symbol", "circle"),
            line_color=d.get("line_color", "rgba(0, 0, 0, 0)"),
            line_width=d.get("line_width", 0),
        )


@dataclass(frozen=True)
class ProposedMarkerConfig:
    """Styling for proposed/optimized borehole markers."""

    size: int = 8
    color: str = "rgba(0, 85, 255, 0.769)"
    symbol: str = "x"
    buffer_color: str = "rgba(4, 0, 255, 0.173)"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProposedMarkerConfig":
        """Create ProposedMarkerConfig from dictionary."""
        return cls(
            size=d.get("size", 8),
            color=d.get("color", "rgba(0, 85, 255, 0.769)"),
            symbol=d.get("symbol", "x"),
            buffer_color=d.get("buffer_color", "rgba(4, 0, 255, 0.173)"),
        )


@dataclass(frozen=True)
class CoverageColorsConfig:
    """Color configuration for coverage visualization."""

    covered: str = "rgba(95, 235, 95, 0.85)"
    gap: str = "rgba(245, 109, 109, 0.398)"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CoverageColorsConfig":
        """Create CoverageColorsConfig from dictionary."""
        return cls(
            covered=d.get("covered", "rgba(95, 235, 95, 0.85)"),
            gap=d.get("gap", "rgba(245, 109, 109, 0.398)"),
        )


@dataclass(frozen=True)
class PanelLayoutConfig:
    """Panel layout dimensions for HTML reports."""

    left_panel_width: int = 220
    right_panel_width: int = 220
    top_offset: int = 80
    vertical_gap: int = 20
    sidebar_spacing: int = 100

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PanelLayoutConfig":
        """Create PanelLayoutConfig from dictionary."""
        return cls(
            left_panel_width=d.get("left_panel_width", 220),
            right_panel_width=d.get("right_panel_width", 220),
            top_offset=d.get("top_offset", 80),
            vertical_gap=d.get("vertical_gap", 20),
            sidebar_spacing=d.get("sidebar_spacing", 100),
        )


@dataclass(frozen=True)
class HexGridOverlayConfig:
    """Styling for hexagonal candidate grid overlay."""

    color: str = "rgba(26, 26, 60, 1)"
    line_width: float = 0.5

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HexGridOverlayConfig":
        """Create HexGridOverlayConfig from dictionary."""
        return cls(
            color=d.get("color", "rgba(26, 26, 60, 1)"),
            line_width=d.get("line_width", 0.5),
        )


@dataclass(frozen=True)
class VisualizationConfig:
    """
    All visualization-related settings for HTML reports and maps.

    Attributes:
        show_coverage_stats_panel: Show/hide the coverage statistics panel.
        borehole_marker: Styling for existing borehole markers.
        proposed_marker: Styling for proposed/optimized boreholes.
        coverage_colors: Colors for coverage visualization.
        panel_layout: Panel dimensions.
        hexgrid_overlay: Styling for hex grid overlay.
        figure_width: Figure width in pixels.
        figure_height: Figure height in pixels.
        figure_background: Figure background color (CSS color string).
    """

    show_coverage_stats_panel: bool = False
    borehole_marker: BoreholeMarkerConfig = field(default_factory=BoreholeMarkerConfig)
    proposed_marker: ProposedMarkerConfig = field(default_factory=ProposedMarkerConfig)
    coverage_colors: CoverageColorsConfig = field(default_factory=CoverageColorsConfig)
    panel_layout: PanelLayoutConfig = field(default_factory=PanelLayoutConfig)
    hexgrid_overlay: HexGridOverlayConfig = field(default_factory=HexGridOverlayConfig)
    figure_width: int = 1200
    figure_height: int = 900
    figure_background: str = "white"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisualizationConfig":
        """Create VisualizationConfig from CONFIG['visualization'] dictionary."""
        return cls(
            show_coverage_stats_panel=d.get("show_coverage_stats_panel", False),
            borehole_marker=BoreholeMarkerConfig.from_dict(
                d.get("borehole_marker", {})
            ),
            proposed_marker=ProposedMarkerConfig.from_dict(
                d.get("proposed_marker", {})
            ),
            coverage_colors=CoverageColorsConfig.from_dict(
                d.get("coverage_colors", {})
            ),
            panel_layout=PanelLayoutConfig.from_dict(d.get("panel_layout", {})),
            hexgrid_overlay=HexGridOverlayConfig.from_dict(
                d.get("hexgrid_overlay", {})
            ),
            figure_width=d.get("figure_width", 1200),
            figure_height=d.get("figure_height", 900),
            figure_background=d.get("figure_background", "white"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ⚡ 5. PARALLEL PROCESSING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ParallelConfig:
    """
    Configuration for parallel/multicore processing.

    Attributes:
        enabled: Master toggle for parallel processing.
        max_workers: Number of worker processes (-1 = auto).
        optimal_workers_default: Default worker count when auto-detecting.
        min_combinations_for_parallel: Minimum combinations to justify parallel.
        timeout_per_combo_seconds: Timeout per combination in seconds.
        fallback_on_error: Fall back to sequential on errors.
        backend: Joblib backend ("loky" = process-based).
        verbose: Verbosity level (0-10).
    """

    enabled: bool = True
    max_workers: int = -1
    optimal_workers_default: int = 14
    min_combinations_for_parallel: int = 10
    timeout_per_combo_seconds: int = 60
    fallback_on_error: bool = True
    backend: str = "loky"
    verbose: int = 10

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParallelConfig":
        """Create ParallelConfig from CONFIG['parallel'] dictionary."""
        return cls(
            enabled=d.get("enabled", True),
            max_workers=d.get("max_workers", -1),
            optimal_workers_default=d.get("optimal_workers_default", 14),
            min_combinations_for_parallel=d.get("min_combinations_for_parallel", 10),
            timeout_per_combo_seconds=d.get("timeout_per_combo_seconds", 60),
            fallback_on_error=d.get("fallback_on_error", True),
            backend=d.get("backend", "loky"),
            verbose=d.get("verbose", 10),
        )


@dataclass(frozen=True)
class ParallelSolverOverridesConfig:
    """Solver overrides for parallel context (reduced timeouts)."""

    solver_mode: Optional[str] = None
    time_limit_s: int = 60
    mip_gap: float = 0.05

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParallelSolverOverridesConfig":
        """Create ParallelSolverOverridesConfig from dictionary."""
        return cls(
            solver_mode=d.get("solver_mode"),
            time_limit_s=d.get("time_limit_s", 60),
            mip_gap=d.get("mip_gap", 0.05),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 💾 6. CACHE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CacheConfig:
    """
    Configuration for result caching.

    Cache eliminates recomputation when only testing HTML/visualization changes.

    Attributes:
        enabled: Master toggle for caching.
        force_overwrite: Force recompute even if cache exists.
        cache_dir: Directory for cache files.
        max_cache_entries: Maximum entries before pruning.
        max_cache_age_days: Auto-expire entries older than this.
        log_cache_hits: Log cache hits/misses.
    """

    enabled: bool = True
    force_overwrite: bool = False
    cache_dir: str = "Gap_Analysis_EC7/cache"
    max_cache_entries: int = 10
    max_cache_age_days: int = 30
    log_cache_hits: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CacheConfig":
        """Create CacheConfig from CONFIG['cache'] dictionary."""
        return cls(
            enabled=d.get("enabled", True),
            force_overwrite=d.get("force_overwrite", False),
            cache_dir=d.get("cache_dir", "Gap_Analysis_EC7/cache"),
            max_cache_entries=d.get("max_cache_entries", 10),
            max_cache_age_days=d.get("max_cache_age_days", 30),
            log_cache_hits=d.get("log_cache_hits", True),
        )

    @property
    def cache_path(self) -> Path:
        """Get cache directory as Path object."""
        return Path(self.cache_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# 🔍 7. QUALITY CONTROL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class QualityControlConfig:
    """Quality control settings."""

    target_crs: str = "EPSG:27700"
    min_boreholes_for_analysis: int = 3

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QualityControlConfig":
        """Create QualityControlConfig from dictionary."""
        return cls(
            target_crs=d.get("target_crs", "EPSG:27700"),
            min_boreholes_for_analysis=d.get("min_boreholes_for_analysis", 3),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 🔄 7b. BORDER CONSOLIDATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GreedyBorderConfig:
    """
    Configuration for greedy border consolidation algorithm.

    The greedy algorithm iteratively removes redundant boreholes at zone
    boundaries while maintaining minimum coverage requirements.

    Attributes:
        min_coverage_factor: Multiplier for minimum coverage (1.0 = exact EC7).
        removal_priority: "fewer_unique" or "most_redundant" strategy.
    """

    min_coverage_factor: float = 1.0
    removal_priority: str = "fewer_unique"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GreedyBorderConfig":
        """Create GreedyBorderConfig from dictionary."""
        return cls(
            min_coverage_factor=d.get("min_coverage_factor", 1.0),
            removal_priority=d.get("removal_priority", "fewer_unique"),
        )

    def __post_init__(self) -> None:
        """Validate greedy config."""
        valid_priorities = ("fewer_unique", "most_redundant")
        if self.removal_priority not in valid_priorities:
            raise ValueError(
                f"removal_priority must be one of {valid_priorities}, "
                f"got '{self.removal_priority}'"
            )
        if self.min_coverage_factor <= 0:
            raise ValueError(
                f"min_coverage_factor must be > 0, got {self.min_coverage_factor}"
            )


@dataclass(frozen=True)
class GlobalILPBorderConfig:
    """
    Configuration for global ILP border consolidation algorithm.

    The global ILP re-optimizes border regions using integer linear programming
    with warm start from existing placements.

    Attributes:
        time_limit_s: Maximum solver time in seconds.
        mip_gap: Acceptable optimality gap (0.05 = 5%).
        fix_interior_boreholes: Keep non-border boreholes fixed.
        use_warm_start: Initialize from existing placement.
    """

    time_limit_s: int = 120
    mip_gap: float = 0.05
    fix_interior_boreholes: bool = True
    use_warm_start: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GlobalILPBorderConfig":
        """Create GlobalILPBorderConfig from dictionary."""
        return cls(
            time_limit_s=d.get("time_limit_s", 120),
            mip_gap=d.get("mip_gap", 0.05),
            fix_interior_boreholes=d.get("fix_interior_boreholes", True),
            use_warm_start=d.get("use_warm_start", True),
        )

    def __post_init__(self) -> None:
        """Validate global ILP config."""
        if self.time_limit_s <= 0:
            raise ValueError(f"time_limit_s must be > 0, got {self.time_limit_s}")
        if not 0 < self.mip_gap < 1:
            raise ValueError(f"mip_gap must be in (0, 1), got {self.mip_gap}")


@dataclass(frozen=True)
class BorderConsolidationConfig:
    """
    Configuration for border consolidation second pass.

    Border consolidation is a post-processing step that removes redundant
    boreholes placed at zone boundaries during zone decomposition. Adjacent
    zones may place boreholes that are closer together than necessary when
    processed independently.

    Attributes:
        mode: "disabled", "ilp", or "buffer_zone".
        time_limit: Maximum solver time in seconds.
        mip_gap: Acceptable optimality gap (0.03 = 3%).
        coverage_target_pct: Minimum coverage percentage to maintain.
        use_conflict_constraints: Enable conflict constraints.
        exclusion_factor: Minimum spacing factor between boreholes.
        verbose: HiGHS verbosity level (0=silent, 1=summary).
        buffer_width_factor: Buffer width as multiple of max_spacing (buffer_zone mode).
        use_fresh_candidates: Generate fresh candidate grid in buffer zone.
        buffer_candidate_spacing_mult: Candidate grid spacing multiplier.
    """

    mode: str = "disabled"
    time_limit: int = 60
    mip_gap: float = 0.03
    coverage_target_pct: float = 97.0
    use_conflict_constraints: bool = True
    exclusion_factor: float = 0.8
    verbose: int = 1
    # Buffer zone mode settings
    buffer_width_factor: float = 1.5
    use_fresh_candidates: bool = True
    buffer_candidate_spacing_mult: float = 0.25
    # Legacy fields for backward compatibility
    greedy: GreedyBorderConfig = field(default_factory=GreedyBorderConfig)
    global_ilp: GlobalILPBorderConfig = field(default_factory=GlobalILPBorderConfig)

    def __post_init__(self) -> None:
        """Validate mode and parameters."""
        valid_modes = ("disabled", "ilp", "greedy", "global_ilp", "buffer_zone")
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{self.mode}'")
        if self.time_limit <= 0:
            raise ValueError(f"time_limit must be > 0, got {self.time_limit}")
        if not 0 < self.mip_gap < 1:
            raise ValueError(f"mip_gap must be in (0, 1), got {self.mip_gap}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BorderConsolidationConfig":
        """Create BorderConsolidationConfig from dictionary."""
        return cls(
            mode=d.get("mode", "disabled"),
            time_limit=d.get("time_limit", 60),
            mip_gap=d.get("mip_gap", 0.03),
            coverage_target_pct=d.get("coverage_target_pct", 97.0),
            use_conflict_constraints=d.get("use_conflict_constraints", True),
            exclusion_factor=d.get("exclusion_factor", 0.8),
            verbose=d.get("verbose", 1),
            buffer_width_factor=d.get("buffer_width_factor", 1.5),
            use_fresh_candidates=d.get("use_fresh_candidates", True),
            buffer_candidate_spacing_mult=d.get("buffer_candidate_spacing_mult", 0.25),
            greedy=GreedyBorderConfig.from_dict(d.get("greedy", {})),
            global_ilp=GlobalILPBorderConfig.from_dict(d.get("global_ilp", {})),
        )

    @property
    def is_enabled(self) -> bool:
        """Check if border consolidation is enabled."""
        return self.mode != "disabled"


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 8. APP CONFIG (MASTER FACADE)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AppConfig:
    """
    Master configuration object for the EC7 Gap Analysis application.

    This is the single source of truth for all typed configuration. Create it
    once at application startup using AppConfig.from_dict(CONFIG) and pass it
    to all functions that need settings.

    Attributes:
        max_spacing_m: EC7 maximum borehole spacing in meters.
        target_crs: Coordinate reference system (e.g., "EPSG:27700").
        depth_slider_step_m: Depth slider step size in meters.
        file_paths: File path configuration.
        testing_mode: Testing mode configuration.
        candidate_grid: Candidate grid configuration.
        visualization: Visualization configuration.
        parallel: Parallel processing configuration.
        parallel_solver_overrides: Solver overrides for parallel context.
        cache: Cache configuration.
        quality_control: Quality control settings.
        border_consolidation: Border consolidation second pass configuration.

    Example:
        from Gap_Analysis_EC7.config import CONFIG
        from Gap_Analysis_EC7.config_types import AppConfig

        app_config = AppConfig.from_dict(CONFIG)
        solver_config = app_config.solver_config(zones_gdf=zones_gdf)
    """

    # Core settings
    max_spacing_m: float = 100.0
    target_crs: str = "EPSG:27700"
    depth_slider_step_m: float = 10.0

    # Sub-configurations
    file_paths: FilePathsConfig = field(default_factory=FilePathsConfig)
    testing_mode: TestingModeConfig = field(default_factory=TestingModeConfig)
    candidate_grid: CandidateGridConfig = field(default_factory=CandidateGridConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    parallel_solver_overrides: ParallelSolverOverridesConfig = field(
        default_factory=ParallelSolverOverridesConfig
    )
    cache: CacheConfig = field(default_factory=CacheConfig)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)
    border_consolidation: BorderConsolidationConfig = field(
        default_factory=BorderConsolidationConfig
    )

    # Raw config dict for legacy access (deprecated - use typed properties)
    _raw_config: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AppConfig":
        """
        Create AppConfig from the CONFIG dictionary.

        This is the primary factory method. Pass the CONFIG dict from config.py.

        Args:
            config_dict: The CONFIG dictionary from config.py.

        Returns:
            AppConfig instance with all settings populated.

        Example:
            from Gap_Analysis_EC7.config import CONFIG
            app_config = AppConfig.from_dict(CONFIG)
        """
        return cls(
            max_spacing_m=config_dict.get("max_spacing_m", 100.0),
            target_crs=config_dict.get("target_crs", "EPSG:27700"),
            depth_slider_step_m=config_dict.get("depth_slider_step_m", 10.0),
            file_paths=FilePathsConfig.from_dict(config_dict.get("file_paths", {})),
            testing_mode=TestingModeConfig.from_dict(
                config_dict.get("testing_mode", {})
            ),
            candidate_grid=CandidateGridConfig.from_dict(
                config_dict.get("candidate_grid", {})
            ),
            visualization=VisualizationConfig.from_dict(
                config_dict.get("visualization", {})
            ),
            parallel=ParallelConfig.from_dict(config_dict.get("parallel", {})),
            parallel_solver_overrides=ParallelSolverOverridesConfig.from_dict(
                config_dict.get("parallel_solver_overrides", {})
            ),
            cache=CacheConfig.from_dict(config_dict.get("cache", {})),
            quality_control=QualityControlConfig.from_dict(
                config_dict.get("quality_control", {})
            ),
            border_consolidation=BorderConsolidationConfig.from_dict(
                config_dict.get("border_consolidation", {})
            ),
            _raw_config=config_dict,
        )

    def solver_config(
        self,
        zones_gdf: Optional[Any] = None,
        is_parallel_context: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> Any:
        """
        Get a SolverConfig object for borehole optimization.

        This bridges to the existing SolverConfig infrastructure in solver_config.py.

        Args:
            zones_gdf: Optional GeoDataFrame of zones (for zone-specific spacing).
            is_parallel_context: True if running in parallel worker.
            logger: Optional logger instance.

        Returns:
            SolverConfig object ready for optimize_boreholes().

        Example:
            solver_config = app_config.solver_config(zones_gdf=zones_gdf)
            boreholes, stats = optimize_boreholes(gaps, solver_config)
        """
        from Gap_Analysis_EC7.solvers.solver_config import config_from_project_config

        return config_from_project_config(
            config_dict=self._raw_config,
            zones_gdf=zones_gdf,
            is_parallel_context=is_parallel_context,
            logger=logger,
        )

    @property
    def log_dir(self) -> Path:
        """Get log directory as Path object."""
        return self.file_paths.log_path

    @property
    def output_dir(self) -> Path:
        """Get output directory as Path object."""
        return self.file_paths.output_path

    def get_raw(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the raw CONFIG dictionary.

        DEPRECATED: Use typed properties instead. This method exists only for
        backwards compatibility during the migration to typed config objects.

        Args:
            key: Top-level key in CONFIG dictionary.
            default: Default value if key not found.

        Returns:
            Value from CONFIG or default.
        """
        return self._raw_config.get(key, default)

    def get_ilp_config(self) -> Dict[str, Any]:
        """
        Get ILP solver config dictionary.

        DEPRECATED: Use solver_config() instead.
        """
        return self._raw_config.get("ilp_solver", {})

    def get_greedy_config(self) -> Dict[str, Any]:
        """
        Get greedy solver config dictionary.

        DEPRECATED: Use solver_config() instead.
        """
        return self._raw_config.get("greedy_solver", {})

    def get_optimization_config(self) -> Dict[str, Any]:
        """
        Get optimization config dictionary.

        DEPRECATED: Use solver_config() instead.
        """
        return self._raw_config.get("optimization", {})
