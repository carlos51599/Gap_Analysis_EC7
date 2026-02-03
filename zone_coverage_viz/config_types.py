"""
Configuration types for zone coverage visualization.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURAL OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

Responsibility: Define typed configuration dataclasses for the zone coverage
visualization module.

Follows AI-Optimized Monolith Guidelines:
- Frozen dataclasses for immutability
- from_dict() class methods for dict compatibility
- Complete type hints on all fields

For Navigation: Use VS Code outline (Ctrl+Shift+O)

═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any


# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 STYLE CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ZoneStyle:
    """Styling for zone polygons.

    Attributes:
        fill_color: RGBA tuple for zone fill (0-255 per channel)
        line_color: RGBA tuple for zone boundary line
        line_width: Width of zone boundary line in pixels
    """

    fill_color: Tuple[int, int, int, int] = (200, 200, 200, 50)
    line_color: Tuple[int, int, int, int] = (100, 100, 100, 255)
    line_width: float = 2.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ZoneStyle":
        """Create ZoneStyle from dictionary."""
        return cls(
            fill_color=tuple(d.get("fill_color", (200, 200, 200, 50))),
            line_color=tuple(d.get("line_color", (100, 100, 100, 255))),
            line_width=d.get("line_width", 2.0),
        )


@dataclass(frozen=True)
class BoreholeStyle:
    """Styling for borehole points.

    Attributes:
        radius_pixels: Radius of borehole marker in pixels
        fill_color: RGBA tuple for borehole fill
        line_color: RGBA tuple for borehole outline
        line_width: Width of borehole outline in pixels
    """

    radius_pixels: int = 8
    fill_color: Tuple[int, int, int, int] = (65, 105, 225, 255)  # Royal Blue
    line_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    line_width: float = 2.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoreholeStyle":
        """Create BoreholeStyle from dictionary."""
        return cls(
            radius_pixels=d.get("radius_pixels", 8),
            fill_color=tuple(d.get("fill_color", (65, 105, 225, 255))),
            line_color=tuple(d.get("line_color", (255, 255, 255, 255))),
            line_width=d.get("line_width", 2.0),
        )


@dataclass(frozen=True)
class CoverageStyle:
    """Styling for coverage polygons.

    Attributes:
        fill_color: RGBA tuple for coverage area fill
        line_color: RGBA tuple for coverage boundary line
        line_width: Width of coverage boundary line in pixels
    """

    fill_color: Tuple[int, int, int, int] = (100, 149, 237, 120)  # Cornflower Blue
    line_color: Tuple[int, int, int, int] = (70, 130, 180, 200)
    line_width: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CoverageStyle":
        """Create CoverageStyle from dictionary."""
        return cls(
            fill_color=tuple(d.get("fill_color", (100, 149, 237, 120))),
            line_color=tuple(d.get("line_color", (70, 130, 180, 200))),
            line_width=d.get("line_width", 1.0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ⚙️ MAIN CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ZoneCoverageConfig:
    """
    Main configuration for zone coverage visualization.

    Attributes:
        title: HTML page title
        default_zoom: Initial map zoom level
        enable_drag: Whether boreholes can be dragged
        show_zone_labels: Whether to display zone name labels
        pre_compute_coverage: Whether to pre-compute initial coverage server-side
        zone_style: Styling for zone polygons
        borehole_style: Styling for borehole markers
        coverage_style: Styling for coverage polygons
        max_spacing_field: Zone property field containing max spacing value
        default_max_spacing: Default max spacing if field not found
        use_cdn_libs: Use CDN for deck.gl/Turf.js instead of bundled
    """

    title: str = "Zone-Aware Coverage Visualization"
    default_zoom: float = 13.0
    enable_drag: bool = True
    show_zone_labels: bool = True
    pre_compute_coverage: bool = True
    zone_style: ZoneStyle = field(default_factory=ZoneStyle)
    borehole_style: BoreholeStyle = field(default_factory=BoreholeStyle)
    coverage_style: CoverageStyle = field(default_factory=CoverageStyle)
    max_spacing_field: str = "max_spacing_m"
    default_max_spacing: float = 100.0
    use_cdn_libs: bool = True  # Use CDN for smaller file size

    def __post_init__(self) -> None:
        """Initialize nested config objects if None."""
        # Frozen dataclass - use object.__setattr__ for initialization
        if self.zone_style is None:
            object.__setattr__(self, "zone_style", ZoneStyle())
        if self.borehole_style is None:
            object.__setattr__(self, "borehole_style", BoreholeStyle())
        if self.coverage_style is None:
            object.__setattr__(self, "coverage_style", CoverageStyle())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ZoneCoverageConfig":
        """Create ZoneCoverageConfig from dictionary."""
        return cls(
            title=d.get("title", "Zone-Aware Coverage Visualization"),
            default_zoom=d.get("default_zoom", 13.0),
            enable_drag=d.get("enable_drag", True),
            show_zone_labels=d.get("show_zone_labels", True),
            pre_compute_coverage=d.get("pre_compute_coverage", True),
            zone_style=ZoneStyle.from_dict(d.get("zone_style", {})),
            borehole_style=BoreholeStyle.from_dict(d.get("borehole_style", {})),
            coverage_style=CoverageStyle.from_dict(d.get("coverage_style", {})),
            max_spacing_field=d.get("max_spacing_field", "max_spacing_m"),
            default_max_spacing=d.get("default_max_spacing", 100.0),
            use_cdn_libs=d.get("use_cdn_libs", True),
        )
