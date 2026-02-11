#!/usr/bin/env python3
"""
Visualization Package

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Provide modular visualization components for EC7 HTML reports.

This package extracts visualization logic from html_builder_EC7.py into
focused modules:
- satellite_tiles: Satellite imagery fetching and caching
- geometry_utils: Color conversions, statistics, coordinate utilities
- plotly_traces: Plotly trace builders for maps
- html_panels: HTML/CSS panel generators

Usage:
    from Gap_Analysis_EC7.visualization import (
        fetch_satellite_tiles_base64,
        hex_to_rgb,
        build_merged_polygon_trace,
        generate_legend_panel_html,
    )

Navigation Guide:
- Each module has its own docstring with function list
- Use VS Code outline (Ctrl+Shift+O) to jump between functions
"""

# ===========================================================================
# SATELLITE TILES
# ===========================================================================

from Gap_Analysis_EC7.visualization.satellite_tiles import (
    fetch_satellite_tiles_base64,
    get_cache_path,
    load_from_cache,
    save_to_cache,
    # Backward compatibility aliases
    _fetch_satellite_tiles_base64,
    _get_satellite_cache_path,
    _load_satellite_cache,
    _save_satellite_cache,
)

# ===========================================================================
# GEOMETRY UTILITIES
# ===========================================================================

from Gap_Analysis_EC7.visualization.geometry_utils import (
    hex_to_rgb,
    rgb_to_hex,
    interpolate_color,
    generate_diverging_colorscale,
    bounds_to_center,
    expand_bounds,
    # Backward compatibility aliases
    _hex_to_rgb,
)

# ===========================================================================
# PLOTLY TRACES
# ===========================================================================

from Gap_Analysis_EC7.visualization.plotly_traces import (
    # Trace builders
    build_hexagon_grid_trace,
    build_zone_boundary_traces,
    build_boreholes_trace,
    build_proposed_boreholes_trace,
    build_per_pass_snapshot_trace,
    build_centreline_traces,
    build_single_polygon_trace,
    # Coverage zone trace builders (moved from coverage_zones.py)
    build_coverage_polygon_trace,
    build_coverage_marker_trace,
    build_coverage_buffer_trace,
    build_borehole_circles_trace,
    # Layout builders
    build_map_layout,
    # Figure modifiers (add traces to fig)
    add_grid_cells_trace,
    add_hexagon_grid_overlay,
    add_zone_boundary_traces,
    add_boreholes_trace,
    # Coverage zone figure modifiers (moved from coverage_zones.py)
    add_coverage_zone_traces,
    add_proposed_borehole_traces,
    # Backward compatibility aliases
    _add_grid_cells_trace,
    _add_boreholes_trace,
    _add_hexagon_grid_overlay,
    _add_zone_boundary_traces,
    _build_layout_without_dropdown,
    # Legacy coverage zone trace names (backward compatibility)
    build_merged_marker_trace,
    build_merged_buffer_trace,
)

# ===========================================================================
# HTML PANELS
# ===========================================================================

from Gap_Analysis_EC7.visualization.html_panels import (
    # Style constants
    LIQUID_GLASS_STYLE,
    PANEL_STYLE_ITEM,
    PANEL_STYLE_HEADER,
    DEFAULT_LEFT_PANEL_WIDTH,
    DEFAULT_RIGHT_PANEL_WIDTH,
    DEFAULT_PANEL_VERTICAL_GAP,
    # Panel generators
    generate_legend_panel_html,
    generate_coverage_stats_panel_html,
    generate_layers_panel_html,
    generate_filters_panel_html,
    generate_panel_styles_css,
    generate_slider_styles_css,
    # Backward compatibility aliases
    _generate_legend_panel_html,
    _generate_coverage_stats_panel_html,
    _generate_layers_panel_html,
    _generate_filters_panel_html,
)

# ===========================================================================
# CLIENT-SIDE SCRIPTS
# ===========================================================================

from Gap_Analysis_EC7.visualization.client_scripts import (
    # Layer toggle scripts
    generate_layer_toggle_scripts,
    # Filter panel scripts
    generate_filter_panel_scripts,
    # Coverage data initialization
    generate_coverage_data_script,
    # Click-to-copy tooltip
    generate_click_to_copy_script,
)

# ===========================================================================
# HTML BUILDER (Main report generator)
# ===========================================================================

from Gap_Analysis_EC7.visualization.html_builder import (
    generate_multi_layer_html,
)

# ===========================================================================
# PUBLIC API
# ===========================================================================

__all__ = [
    # HTML Builder (main entry point)
    "generate_multi_layer_html",
    # Satellite
    "fetch_satellite_tiles_base64",
    "get_cache_path",
    "load_from_cache",
    "save_to_cache",
    # Geometry
    "hex_to_rgb",
    "rgb_to_hex",
    "interpolate_color",
    "generate_diverging_colorscale",
    "bounds_to_center",
    "expand_bounds",
    # Trace builders
    "build_hexagon_grid_trace",
    "build_zone_boundary_traces",
    "build_boreholes_trace",
    "build_proposed_boreholes_trace",
    "build_per_pass_snapshot_trace",
    "build_centreline_traces",
    "build_single_polygon_trace",
    # Coverage zone trace builders
    "build_coverage_polygon_trace",
    "build_coverage_marker_trace",
    "build_coverage_buffer_trace",
    "build_borehole_circles_trace",
    # Legacy names (backward compatibility)
    "build_merged_marker_trace",
    "build_merged_buffer_trace",
    # Layout builders
    "build_map_layout",
    # Figure modifiers
    "add_grid_cells_trace",
    "add_hexagon_grid_overlay",
    "add_zone_boundary_traces",
    "add_boreholes_trace",
    # Coverage zone figure modifiers
    "add_coverage_zone_traces",
    "add_proposed_borehole_traces",
    # Panels
    "LIQUID_GLASS_STYLE",
    "PANEL_STYLE_ITEM",
    "PANEL_STYLE_HEADER",
    "DEFAULT_LEFT_PANEL_WIDTH",
    "DEFAULT_RIGHT_PANEL_WIDTH",
    "DEFAULT_PANEL_VERTICAL_GAP",
    "generate_legend_panel_html",
    "generate_coverage_stats_panel_html",
    "generate_layers_panel_html",
    "generate_filters_panel_html",
    "generate_panel_styles_css",
    # Client-side scripts
    "generate_layer_toggle_scripts",
    "generate_filter_panel_scripts",
    "generate_coverage_data_script",
    "generate_click_to_copy_script",
    "generate_slider_styles_css",
]
