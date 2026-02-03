"""
Zone-Aware Dynamic Coverage Visualization Module

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURAL OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

Responsibility: Generate standalone HTML with deck.gl-based interactive
visualization for zone-aware borehole coverage with variable radii per zone.

Key Features:
- Draggable boreholes with real-time coverage updates
- Zone-aware coverage radii (different max_spacing per zone)
- Multi-zone "flower petal" coverage shapes at zone boundaries
- Web Worker-based geometry computation for smooth performance
- Export modified positions to CSV

Usage:
    from zone_coverage_viz import generate_zone_coverage_html

    html_path = generate_zone_coverage_html(
        zones_gdf=zones,
        boreholes_gdf=boreholes,
        output_path="output/zone_coverage.html",
        config=ZoneCoverageConfig()
    )

For Navigation: Use VS Code outline (Ctrl+Shift+O)

═══════════════════════════════════════════════════════════════════════════════
"""

from .deckgl_builder import generate_zone_coverage_html
from .config_types import (
    ZoneCoverageConfig,
    ZoneStyle,
    BoreholeStyle,
    CoverageStyle,
)

__all__ = [
    "generate_zone_coverage_html",
    "ZoneCoverageConfig",
    "ZoneStyle",
    "BoreholeStyle",
    "CoverageStyle",
]
