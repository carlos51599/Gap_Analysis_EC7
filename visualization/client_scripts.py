#!/usr/bin/env python3
"""
Client-Side JavaScript Generators

ARCHITECTURAL OVERVIEW:
=======================
Responsibility: Generate JavaScript code for HTML panel interactivity.
Pure functions that return JavaScript strings - no Plotly dependency.

Script Categories:
- Layer visibility toggles (BGS, satellite, proposed boreholes, candidate grid)
- Filter panel scripts (depth slider, test data checkboxes, coverage switching)

Dependencies:
- json (for JavaScript data embedding)

Navigation Guide:
- Use VS Code outline (Ctrl+Shift+O) to jump between functions
- LAYER_TOGGLE_SCRIPTS: Visibility controls for map layers
- FILTER_SCRIPTS: Depth filtering and test data filtering
"""

import json
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ LAYER TOGGLE SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════


def generate_layer_toggle_scripts(
    bgs_layers_json: str,
    proposed_range_json: str,
    hexgrid_range_json: str,
    has_second_pass: bool = False,
    has_czrc_second_pass: bool = False,
    has_czrc_zone_overlap: bool = False,
    has_czrc_grid: bool = False,
    has_third_pass: bool = False,
    has_third_pass_overlap: bool = False,
    has_third_pass_grid: bool = False,
    has_third_pass_test_points: bool = False,
) -> str:
    """
    Generate JavaScript for layer visibility toggles.

    Handles:
    - BGS layer checkboxes (multiple layers)
    - Satellite layer checkbox (toggles image opacity)
    - Proposed boreholes checkbox (shows/hides proposed borehole markers + buffers)
    - Second pass checkbox (shows/hides removed and added borehole traces together)
    - CZRC second pass checkbox (shows/hides CZRC removed/added traces)
    - CZRC zone overlap checkbox (shows/hides coverage clouds and pairwise overlaps)
    - CZRC grid checkbox (shows/hides visibility boundary and hex candidate grid)
    - Third pass checkbox (shows/hides cell-cell CZRC removed/added traces)
    - Third pass overlap checkbox (shows/hides cell clouds and intersections)
    - Third pass grid checkbox (shows/hides cell-cell candidate grid)
    - Third pass test points checkbox (shows/hides test points used in cell-cell optimization)
    - Candidate grid checkbox (shows/hides hexagon overlay)

    Args:
        bgs_layers_json: JSON string mapping layer names to [startIdx, endIdx]
        proposed_range_json: JSON string for proposed boreholes trace range [start, end] or null
        hexgrid_range_json: JSON string for candidate grid trace range [start, end] or null
        has_second_pass: Whether second pass traces are available
        has_czrc_second_pass: Whether CZRC second pass traces are available
        has_czrc_zone_overlap: Whether CZRC zone overlap traces are available
        has_czrc_grid: Whether CZRC grid traces are available
        has_third_pass: Whether third pass traces (cell-cell CZRC) are available
        has_third_pass_overlap: Whether third pass overlap traces (cell clouds/intersections) are available
        has_third_pass_grid: Whether third pass grid traces (cell-cell candidate grid) are available
        has_third_pass_test_points: Whether third pass test points trace is available

    Returns:
        JavaScript code block as string (without <script> tags)
    """
    # Generate second pass toggle script if needed
    second_pass_script = ""
    if has_second_pass:
        second_pass_script = """
    const secondPassCheckbox = document.getElementById('secondPassCheckbox');
    if (secondPassCheckbox) {
        secondPassCheckbox.addEventListener('change', function() {
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            const traceIndices = [];
            
            if (typeof COVERAGE_TRACE_RANGES !== 'undefined' && typeof currentCoverageCombo !== 'undefined') {
                const ranges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
                if (ranges) {
                    // Collect removed buffer and marker traces
                    const removedBufferRange = ranges.removed_buffers || [0, 0];
                    const removedMarkerRange = ranges.removed_markers || [0, 0];
                    for (let i = removedBufferRange[0]; i < removedBufferRange[1]; i++) {
                        traceIndices.push(i);
                    }
                    for (let i = removedMarkerRange[0]; i < removedMarkerRange[1]; i++) {
                        traceIndices.push(i);
                    }
                    
                    // Collect added buffer and marker traces
                    const addedBufferRange = ranges.added_buffers || [0, 0];
                    const addedMarkerRange = ranges.added_markers || [0, 0];
                    for (let i = addedBufferRange[0]; i < addedBufferRange[1]; i++) {
                        traceIndices.push(i);
                    }
                    for (let i = addedMarkerRange[0]; i < addedMarkerRange[1]; i++) {
                        traceIndices.push(i);
                    }
                }
            }
            
            if (traceIndices.length > 0) {
                Plotly.restyle(plotDiv, {'visible': this.checked}, traceIndices);
            }
        });
    }
"""

    # Generate Zone Overlap toggle script if needed (independent of CZRC second pass)
    zone_overlap_script = ""
    if has_czrc_zone_overlap:
        zone_overlap_script = """
    const zoneOverlapCheckbox = document.getElementById('zoneOverlapCheckbox');
    if (zoneOverlapCheckbox) {
        zoneOverlapCheckbox.addEventListener('change', function() {
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            const traceIndices = [];
            
            if (typeof COVERAGE_TRACE_RANGES !== 'undefined' && typeof currentCoverageCombo !== 'undefined') {
                const ranges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
                if (ranges) {
                    // Toggle CZRC coverage clouds (per-zone coverage areas)
                    if (ranges.czrc_clouds) {
                        const [startIdx, endIdx] = ranges.czrc_clouds;
                        for (let i = startIdx; i < endIdx; i++) {
                            traceIndices.push(i);
                        }
                    }
                    // Toggle CZRC pairwise regions (2-zone overlaps)
                    if (ranges.czrc_pairwise) {
                        const [startIdx, endIdx] = ranges.czrc_pairwise;
                        for (let i = startIdx; i < endIdx; i++) {
                            traceIndices.push(i);
                        }
                    }
                }
            }
            
            if (traceIndices.length > 0) {
                Plotly.restyle(plotDiv, {'visible': this.checked}, traceIndices);
            }
        });
    }
"""

    # Generate CZRC grid toggle script if needed (independent of CZRC second pass)
    czrc_grid_script = ""
    if has_czrc_grid:
        czrc_grid_script = """
    const czrcGridCheckbox = document.getElementById('czrcGridCheckbox');
    if (czrcGridCheckbox) {
        czrcGridCheckbox.addEventListener('change', function() {
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            const traceIndices = [];
            
            if (typeof COVERAGE_TRACE_RANGES !== 'undefined' && typeof currentCoverageCombo !== 'undefined') {
                const ranges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
                // Toggle CZRC grid (candidate hexagons in overlap regions)
                if (ranges && ranges.czrc_grid) {
                    const [startIdx, endIdx] = ranges.czrc_grid;
                    for (let i = startIdx; i < endIdx; i++) {
                        traceIndices.push(i);
                    }
                }
                // Toggle CZRC first pass candidates (black X markers)
                if (ranges && ranges.czrc_first_pass_candidates) {
                    const [startIdx, endIdx] = ranges.czrc_first_pass_candidates;
                    for (let i = startIdx; i < endIdx; i++) {
                        traceIndices.push(i);
                    }
                }
            }
            
            if (traceIndices.length > 0) {
                Plotly.restyle(plotDiv, {'visible': this.checked}, traceIndices);
            }
        });
    }
"""

    # Generate CZRC second pass toggle scripts if needed
    czrc_second_pass_script = ""
    if has_czrc_second_pass:
        czrc_second_pass_script = """
    const czrcSecondPassCheckbox = document.getElementById('czrcSecondPassCheckbox');
    if (czrcSecondPassCheckbox) {
        czrcSecondPassCheckbox.addEventListener('change', function() {
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            const traceIndices = [];
            
            if (typeof COVERAGE_TRACE_RANGES !== 'undefined' && typeof currentCoverageCombo !== 'undefined') {
                const ranges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
                if (ranges) {
                    // Collect CZRC removed buffer and marker traces
                    const czrcRemovedBufferRange = ranges.czrc_removed_buffers || [0, 0];
                    const czrcRemovedMarkerRange = ranges.czrc_removed_markers || [0, 0];
                    for (let i = czrcRemovedBufferRange[0]; i < czrcRemovedBufferRange[1]; i++) {
                        traceIndices.push(i);
                    }
                    for (let i = czrcRemovedMarkerRange[0]; i < czrcRemovedMarkerRange[1]; i++) {
                        traceIndices.push(i);
                    }
                    
                    // Collect CZRC added buffer and marker traces
                    const czrcAddedBufferRange = ranges.czrc_added_buffers || [0, 0];
                    const czrcAddedMarkerRange = ranges.czrc_added_markers || [0, 0];
                    for (let i = czrcAddedBufferRange[0]; i < czrcAddedBufferRange[1]; i++) {
                        traceIndices.push(i);
                    }
                    for (let i = czrcAddedMarkerRange[0]; i < czrcAddedMarkerRange[1]; i++) {
                        traceIndices.push(i);
                    }
                }
            }
            
            if (traceIndices.length > 0) {
                Plotly.restyle(plotDiv, {'visible': this.checked}, traceIndices);
            }
        });
    }

    // CZRC Test Points checkbox handler
    const czrcTestPointsCheckbox = document.getElementById('czrcTestPointsCheckbox');
    if (czrcTestPointsCheckbox) {
        czrcTestPointsCheckbox.addEventListener('change', function() {
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            const traceIndices = [];
            
            if (typeof COVERAGE_TRACE_RANGES !== 'undefined' && typeof currentCoverageCombo !== 'undefined') {
                const ranges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
                if (ranges) {
                    // Collect CZRC test points trace
                    const czrcTestPointsRange = ranges.czrc_test_points || [0, 0];
                    for (let i = czrcTestPointsRange[0]; i < czrcTestPointsRange[1]; i++) {
                        traceIndices.push(i);
                    }
                }
            }
            
            if (traceIndices.length > 0) {
                Plotly.restyle(plotDiv, {'visible': this.checked}, traceIndices);
            }
        });
    }
"""

    # Generate Third Pass (cell-cell CZRC) toggle script if needed
    third_pass_script = ""
    if has_third_pass:
        third_pass_script = """
    // Third Pass (cell-cell CZRC) checkbox handler
    const thirdPassCheckbox = document.getElementById('thirdPassCheckbox');
    if (thirdPassCheckbox) {
        thirdPassCheckbox.addEventListener('change', function() {
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            const traceIndices = [];
            
            if (typeof COVERAGE_TRACE_RANGES !== 'undefined' && typeof currentCoverageCombo !== 'undefined') {
                const ranges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
                if (ranges) {
                    // Collect Third Pass removed buffer and marker traces
                    const thirdPassRemovedBufferRange = ranges.third_pass_removed_buffers || [0, 0];
                    const thirdPassRemovedMarkerRange = ranges.third_pass_removed_markers || [0, 0];
                    for (let i = thirdPassRemovedBufferRange[0]; i < thirdPassRemovedBufferRange[1]; i++) {
                        traceIndices.push(i);
                    }
                    for (let i = thirdPassRemovedMarkerRange[0]; i < thirdPassRemovedMarkerRange[1]; i++) {
                        traceIndices.push(i);
                    }
                    
                    // Collect Third Pass added buffer and marker traces
                    const thirdPassAddedBufferRange = ranges.third_pass_added_buffers || [0, 0];
                    const thirdPassAddedMarkerRange = ranges.third_pass_added_markers || [0, 0];
                    for (let i = thirdPassAddedBufferRange[0]; i < thirdPassAddedBufferRange[1]; i++) {
                        traceIndices.push(i);
                    }
                    for (let i = thirdPassAddedMarkerRange[0]; i < thirdPassAddedMarkerRange[1]; i++) {
                        traceIndices.push(i);
                    }
                }
            }
            
            if (traceIndices.length > 0) {
                Plotly.restyle(plotDiv, {'visible': this.checked}, traceIndices);
            }
        });
    }
"""

    # Generate Third Pass Overlap (cell clouds + intersections) toggle script if needed
    third_pass_overlap_script = ""
    if has_third_pass_overlap:
        third_pass_overlap_script = """
    // Third Pass Overlap (cell clouds + intersections) checkbox handler
    const thirdPassOverlapCheckbox = document.getElementById('thirdPassOverlapCheckbox');
    if (thirdPassOverlapCheckbox) {
        thirdPassOverlapCheckbox.addEventListener('change', function() {
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            const traceIndices = [];
            
            if (typeof COVERAGE_TRACE_RANGES !== 'undefined' && typeof currentCoverageCombo !== 'undefined') {
                const ranges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
                if (ranges) {
                    // Toggle third pass cell clouds (per-cell coverage areas)
                    if (ranges.third_pass_clouds) {
                        const [startIdx, endIdx] = ranges.third_pass_clouds;
                        for (let i = startIdx; i < endIdx; i++) {
                            traceIndices.push(i);
                        }
                    }
                    // Toggle third pass cell intersections (cell-cell overlaps)
                    if (ranges.third_pass_intersections) {
                        const [startIdx, endIdx] = ranges.third_pass_intersections;
                        for (let i = startIdx; i < endIdx; i++) {
                            traceIndices.push(i);
                        }
                    }
                }
            }
            
            if (traceIndices.length > 0) {
                Plotly.restyle(plotDiv, {'visible': this.checked}, traceIndices);
            }
        });
    }
"""

    # Generate Third Pass Grid toggle script if needed
    third_pass_grid_script = ""
    if has_third_pass_grid:
        third_pass_grid_script = """
    // Third Pass Grid checkbox handler
    const thirdPassGridCheckbox = document.getElementById('thirdPassGridCheckbox');
    if (thirdPassGridCheckbox) {
        thirdPassGridCheckbox.addEventListener('change', function() {
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            const traceIndices = [];
            
            if (typeof COVERAGE_TRACE_RANGES !== 'undefined' && typeof currentCoverageCombo !== 'undefined') {
                const ranges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
                // Toggle third pass grid (cell-cell candidate hexagons)
                if (ranges && ranges.third_pass_grid) {
                    const [startIdx, endIdx] = ranges.third_pass_grid;
                    for (let i = startIdx; i < endIdx; i++) {
                        traceIndices.push(i);
                    }
                }
            }
            
            if (traceIndices.length > 0) {
                Plotly.restyle(plotDiv, {'visible': this.checked}, traceIndices);
            }
        });
    }
"""

    # Generate Third Pass Test Points toggle script if needed
    third_pass_test_points_script = ""
    if has_third_pass_test_points:
        third_pass_test_points_script = """
    // Third Pass Test Points checkbox handler
    const thirdPassTestPointsCheckbox = document.getElementById('thirdPassTestPointsCheckbox');
    if (thirdPassTestPointsCheckbox) {
        thirdPassTestPointsCheckbox.addEventListener('change', function() {
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            const traceIndices = [];
            
            if (typeof COVERAGE_TRACE_RANGES !== 'undefined' && typeof currentCoverageCombo !== 'undefined') {
                const ranges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
                // Toggle third pass test points (cell-cell CZRC test points)
                if (ranges && ranges.third_pass_test_points) {
                    const [startIdx, endIdx] = ranges.third_pass_test_points;
                    for (let i = startIdx; i < endIdx; i++) {
                        traceIndices.push(i);
                    }
                }
            }
            
            if (traceIndices.length > 0) {
                Plotly.restyle(plotDiv, {'visible': this.checked}, traceIndices);
            }
        });
    }
"""

    return f"""
    // === LAYER STATE ===
    const bgsLayers = {bgs_layers_json};
    const bgsLayerStates = {{}};
    
    const proposedBoreholesRange = {proposed_range_json};
    const candidateGridRange = {hexgrid_range_json};
    
    // Initialize BGS layer states
    Object.keys(bgsLayers).forEach(layerName => {{
        bgsLayerStates[layerName] = false;
    }});
    
    // === VISIBILITY UPDATE FUNCTIONS ===
    
    function updateBgsLayerVisibility(layerName) {{
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv || !bgsLayers[layerName]) return;
        
        const [startIdx, endIdx] = bgsLayers[layerName];
        
        const traceIndices = [];
        for (let i = startIdx; i < endIdx; i++) {{
            traceIndices.push(i);
        }}
        
        const visibility = bgsLayerStates[layerName];
        Plotly.restyle(plotDiv, {{'visible': visibility}}, traceIndices);
    }}
    
    // === CHECKBOX EVENT LISTENERS ===
    
    document.querySelectorAll('.bgsLayerCheckbox').forEach(checkbox => {{
        checkbox.addEventListener('change', function() {{
            const layerName = this.getAttribute('data-layer');
            bgsLayerStates[layerName] = this.checked;
            updateBgsLayerVisibility(layerName);
        }});
    }});
    
    const satelliteCheckbox = document.getElementById('satelliteLayerCheckbox');
    if (satelliteCheckbox) {{
        satelliteCheckbox.addEventListener('change', function() {{
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            const currentLayout = plotDiv.layout;
            if (currentLayout && currentLayout.images && currentLayout.images.length > 0) {{
                const newOpacity = this.checked ? 1.0 : 0;
                const updatedImages = currentLayout.images.map(img => ({{
                    ...img,
                    opacity: newOpacity
                }}));
                Plotly.relayout(plotDiv, {{'images': updatedImages}});
            }}
        }});
    }}
    
    const proposedCheckbox = document.getElementById('proposedBoreholesCheckbox');
    if (proposedCheckbox) {{
        proposedCheckbox.addEventListener('change', function() {{
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            let traceIndices = [];
            
            if (typeof COVERAGE_TRACE_RANGES !== 'undefined' && typeof currentCoverageCombo !== 'undefined') {{
                const ranges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
                if (ranges) {{
                    const bufferRange = ranges.proposed_buffers || [0, 0];
                    const markerRange = ranges.proposed_markers || [0, 0];
                    for (let i = bufferRange[0]; i < bufferRange[1]; i++) {{
                        traceIndices.push(i);
                    }}
                    for (let i = markerRange[0]; i < markerRange[1]; i++) {{
                        traceIndices.push(i);
                    }}
                }}
            }} else if (proposedBoreholesRange !== null) {{
                const [startIdx, endIdx] = proposedBoreholesRange;
                for (let i = startIdx; i < endIdx; i++) {{
                    traceIndices.push(i);
                }}
            }}
            
            if (traceIndices.length > 0) {{
                Plotly.restyle(plotDiv, {{'visible': this.checked}}, traceIndices);
            }}
        }});
    }}
    
    // Borehole circles toggle (outline-only circles showing coverage radii)
    const boreholeCirclesCheckbox = document.getElementById('boreholeCirclesCheckbox');
    if (boreholeCirclesCheckbox) {{
        boreholeCirclesCheckbox.addEventListener('change', function() {{
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            const traceIndices = [];
            
            if (typeof COVERAGE_TRACE_RANGES !== 'undefined' && typeof currentCoverageCombo !== 'undefined') {{
                const ranges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
                if (ranges && ranges.borehole_circles) {{
                    const [startIdx, endIdx] = ranges.borehole_circles;
                    for (let i = startIdx; i < endIdx; i++) {{
                        traceIndices.push(i);
                    }}
                }}
            }}
            
            if (traceIndices.length > 0) {{
                Plotly.restyle(plotDiv, {{'visible': this.checked}}, traceIndices);
            }}
        }});
    }}
    
    const candidateGridCheckbox = document.getElementById('candidateGridCheckbox');
    if (candidateGridCheckbox) {{
        candidateGridCheckbox.addEventListener('change', function() {{
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            const traceIndices = [];
            
            if (typeof COVERAGE_TRACE_RANGES !== 'undefined' && typeof currentCoverageCombo !== 'undefined') {{
                const ranges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
                if (ranges && ranges.hexagon_grid) {{
                    const [startIdx, endIdx] = ranges.hexagon_grid;
                    for (let i = startIdx; i < endIdx; i++) {{
                        traceIndices.push(i);
                    }}
                }}
            }} else if (candidateGridRange !== null) {{
                const [startIdx, endIdx] = candidateGridRange;
                for (let i = startIdx; i < endIdx; i++) {{
                    traceIndices.push(i);
                }}
            }}
            
            if (traceIndices.length > 0) {{
                Plotly.restyle(plotDiv, {{'visible': this.checked}}, traceIndices);
            }}
        }});
    }}
{second_pass_script}
{zone_overlap_script}
{czrc_grid_script}
{czrc_second_pass_script}
{third_pass_script}
{third_pass_overlap_script}
{third_pass_grid_script}
{third_pass_test_points_script}
"""


# ═══════════════════════════════════════════════════════════════════════════
# 🔍 FILTER PANEL SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════


def _js_filter_state(borehole_trace_idx: int, slider_min: int, slider_max: int) -> str:
    """Generate JS filter state constants and variables."""
    return f"""
    // === FILTER STATE ===
    const BOREHOLE_TRACE_IDX = {borehole_trace_idx};
    const SLIDER_MIN = {slider_min};
    const SLIDER_MAX = {slider_max};
    let boreholeData = null;
    let totalBoreholes = 0;
    let filterInitialized = false;
    
    let minDepthFilter = SLIDER_MIN;
    let filterSPT = false;
    let filterTriaxialTotal = false;
    let filterTriaxialEffective = false;
    
    function getPlotDiv() {{
        return document.querySelector('.plotly-graph-div');
    }}"""


def _js_initialization() -> str:
    """Generate JS borehole data initialization function."""
    return """
    // === INITIALIZATION ===
    function initializeBoreholeFilter() {
        if (filterInitialized) return true;
        const plotDiv = getPlotDiv();
        if (!plotDiv || !plotDiv.data || !plotDiv.data[BOREHOLE_TRACE_IDX]) return false;
        
        const trace = plotDiv.data[BOREHOLE_TRACE_IDX];
        if (!trace.customdata || trace.customdata.length === 0) return false;
        
        boreholeData = trace.customdata.map(function(d) {
            if (Array.isArray(d) && d.length >= 5) {
                return {
                    depth: parseFloat(d[1]) || 0,
                    has_spt: d[2] === 1 || d[2] === true,
                    has_triaxial_total: d[3] === 1 || d[3] === true,
                    has_triaxial_effective: d[4] === 1 || d[4] === true
                };
            }
            return {
                depth: Array.isArray(d) && d.length > 1 ? parseFloat(d[1]) || 0 : 0,
                has_spt: false, has_triaxial_total: false, has_triaxial_effective: false
            };
        });
        
        totalBoreholes = boreholeData.length;
        filterInitialized = true;
        updateCountDisplay(totalBoreholes);
        return true;
    }"""


def _js_display_updates() -> str:
    """Generate JS display update functions."""
    return """
    // === DISPLAY UPDATES ===
    function updateCountDisplay(shown) {
        const countDiv = document.getElementById('boreholeCount');
        if (countDiv) {
            countDiv.textContent = shown === totalBoreholes 
                ? 'Showing all ' + totalBoreholes + ' boreholes'
                : 'Showing ' + shown + ' of ' + totalBoreholes + ' boreholes';
        }
    }
    
    function updateDepthDisplay() {
        const valueSpan = document.getElementById('depthFilterValue');
        if (valueSpan) {
            valueSpan.textContent = minDepthFilter === 0 ? 'All depths' : '≥ ' + minDepthFilter + 'm';
        }
        const sliderTrack = document.getElementById('sliderTrack');
        if (sliderTrack) {
            const pct = ((minDepthFilter - SLIDER_MIN) / (SLIDER_MAX - SLIDER_MIN)) * 100;
            sliderTrack.style.background = 'linear-gradient(to right, #4a90d9 ' + pct + '%, #ddd ' + pct + '%)';
        }
    }"""


def _js_coverage_switching() -> str:
    """Generate JS coverage trace switching function."""
    return """
    // === COVERAGE SWITCHING ===
    function updateCoverageVisibility() {
        const plotDiv = getPlotDiv();
        if (!plotDiv || typeof COVERAGE_TRACE_RANGES === 'undefined') return;
        
        const newComboKey = 'd' + minDepthFilter + '_spt' + (filterSPT ? 1 : 0) + 
                          '_txt' + (filterTriaxialTotal ? 1 : 0) + '_txe' + (filterTriaxialEffective ? 1 : 0);
        if (newComboKey === currentCoverageCombo) return;
        
        const newRanges = COVERAGE_TRACE_RANGES[newComboKey];
        const oldRanges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
        if (!newRanges) { console.warn('No precomputed coverage for combo:', newComboKey); return; }
        
        const traceIndices = [], visibilities = [];
        const traceTypes = ['covered', 'gaps', 'proposed_buffers', 'proposed_markers', 'hexagon_grid'];
        
        // Hide old traces
        if (oldRanges) {
            traceTypes.forEach(function(tt) {
                const r = oldRanges[tt];
                if (r) for (let i = r[0]; i < r[1]; i++) { traceIndices.push(i); visibilities.push(false); }
            });
        }
        
        // Show new traces (respecting checkbox states)
        const showProposed = document.getElementById('proposedBoreholesCheckbox')?.checked ?? true;
        const showGrid = document.getElementById('candidateGridCheckbox')?.checked ?? false;
        const showCircles = document.getElementById('boreholeCirclesCheckbox')?.checked ?? false;
        
        traceTypes.forEach(function(tt) {
            const r = newRanges[tt];
            if (!r) return;
            let show = true;
            if (tt === 'proposed_buffers' || tt === 'proposed_markers') show = showProposed;
            else if (tt === 'hexagon_grid') show = showGrid;
            else if (tt === 'borehole_circles') show = showCircles;
            for (let i = r[0]; i < r[1]; i++) {
                const idx = traceIndices.indexOf(i);
                if (idx >= 0) visibilities[idx] = show;
                else { traceIndices.push(i); visibilities.push(show); }
            }
        });
        
        if (traceIndices.length > 0) Plotly.restyle(plotDiv, {'visible': visibilities}, traceIndices);
        currentCoverageCombo = newComboKey;
    }"""


def _js_filter_application() -> str:
    """Generate JS filter application function."""
    return """
    // === FILTER APPLICATION ===
    function applyAllFilters() {
        if (!initializeBoreholeFilter()) { setTimeout(applyAllFilters, 200); return; }
        const plotDiv = getPlotDiv();
        if (!plotDiv) return;
        
        const opacities = boreholeData.map(function(bh) {
            if (bh.depth < minDepthFilter) return 0;
            if (filterSPT && !bh.has_spt) return 0;
            if (filterTriaxialTotal && !bh.has_triaxial_total) return 0;
            if (filterTriaxialEffective && !bh.has_triaxial_effective) return 0;
            return 1;
        });
        
        Plotly.restyle(plotDiv, {'marker.opacity': [opacities]}, [BOREHOLE_TRACE_IDX]);
        updateCountDisplay(opacities.filter(function(o) { return o > 0; }).length);
        updateCoverageVisibility();
    }"""


def _js_event_listeners() -> str:
    """Generate JS event listeners and deferred initialization."""
    return """
    // === EVENT LISTENERS ===
    const depthSlider = document.getElementById('depthSlider');
    if (depthSlider) {
        // Initialize minDepthFilter from the slider's current value (may be preset from testing mode)
        minDepthFilter = parseInt(depthSlider.value);
        depthSlider.addEventListener('input', function() {
            minDepthFilter = parseInt(this.value);
            updateDepthDisplay();
            applyAllFilters();
        });
        updateDepthDisplay();
    }
    
    // Initialize checkbox filter states from their current checked state (may be preset from testing mode)
    document.querySelectorAll('.test-data-filter').forEach(function(checkbox) {
        const ft = checkbox.getAttribute('data-filter');
        const checked = checkbox.checked;
        if (ft === 'spt') filterSPT = checked;
        else if (ft === 'triaxial_total') filterTriaxialTotal = checked;
        else if (ft === 'triaxial_effective') filterTriaxialEffective = checked;
        
        checkbox.addEventListener('change', function() {
            const ft = this.getAttribute('data-filter');
            const checked = this.checked;
            if (ft === 'spt') filterSPT = checked;
            else if (ft === 'triaxial_total') filterTriaxialTotal = checked;
            else if (ft === 'triaxial_effective') filterTriaxialEffective = checked;
            applyAllFilters();
        });
    });
    
    // === DEFERRED INITIALIZATION ===
    function tryInit() {
        if (!initializeBoreholeFilter()) {
            setTimeout(tryInit, 300);
        } else {
            // After initialization, apply filters if any preset values are non-default
            // This handles testing mode preset filter values
            if (minDepthFilter > 0 || filterSPT || filterTriaxialTotal || filterTriaxialEffective) {
                applyAllFilters();
            }
        }
    }
    if (document.readyState === 'complete') setTimeout(tryInit, 500);
    else window.addEventListener('load', function() { setTimeout(tryInit, 500); });
"""


def generate_filter_panel_scripts(
    boreholes_trace_idx: int,
    slider_min: int,
    slider_max: int,
) -> str:
    """Generate JavaScript for filter panel with depth slider and test data checkboxes."""
    return f"""
(function() {{
{_js_filter_state(boreholes_trace_idx, slider_min, slider_max)}
{_js_initialization()}
{_js_display_updates()}
{_js_coverage_switching()}
{_js_filter_application()}
{_js_event_listeners()}
}})();
"""


# ═══════════════════════════════════════════════════════════════════════════
# 🔴 BOREHOLE CIRCLE REMOVAL SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════


def generate_circle_removal_script() -> str:
    """
    Generate JavaScript for interactive removal of borehole circles.

    When Shift+Click is performed on a borehole circle (while the layer is visible),
    the circle is removed from the visualization. Features:

    - Tracks removed borehole IDs in a Set
    - Maintains removal history for Undo functionality
    - Stores master data for regenerating trace after removal
    - Provides Export functionality for removed circle list
    - Shows visual feedback via toast notifications
    - Updates counter in removal controls panel

    The removal is data-driven: we filter the master borehole list and reconstruct
    the trace data rather than nullifying individual points.

    Returns:
        JavaScript code block as string (without <script> tags)
    """
    return """
(function() {
    // === CIRCLE REMOVAL STATE ===
    // Global flag to indicate removal handling is active
    window.circleRemovalActive = false;
    const removedBoreholeIds = new Set();
    const removalHistory = [];  // Stack for undo functionality
    let masterBoreholeData = null;  // Will be populated from trace customdata
    
    // === TOAST NOTIFICATION ===
    function showRemovalToast(message, isSuccess) {
        const existingToast = document.getElementById('removalToast');
        if (existingToast) existingToast.remove();
        
        const toast = document.createElement('div');
        toast.id = 'removalToast';
        toast.style.cssText = `
            position: fixed;
            bottom: 60px;
            left: 50%;
            transform: translateX(-50%);
            padding: 12px 24px;
            background: ${isSuccess ? 'rgba(220, 53, 69, 0.95)' : 'rgba(108, 117, 125, 0.95)'};
            color: white;
            border-radius: 8px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            z-index: 10000;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            transition: opacity 0.3s ease-out;
            pointer-events: none;
        `;
        toast.textContent = message;
        document.body.appendChild(toast);
        
        setTimeout(function() {
            toast.style.opacity = '0';
            setTimeout(function() { toast.remove(); }, 300);
        }, 2000);
    }
    
    // === UI UPDATE ===
    function updateRemovalUI() {
        const countDisplay = document.getElementById('removedCount');
        if (countDisplay) {
            countDisplay.textContent = `Removed: ${removedBoreholeIds.size} circle${removedBoreholeIds.size !== 1 ? 's' : ''}`;
        }
        
        const undoBtn = document.getElementById('undoLastRemoval');
        if (undoBtn) {
            undoBtn.disabled = removalHistory.length === 0;
        }
        
        const resetBtn = document.getElementById('resetRemovals');
        if (resetBtn) {
            resetBtn.disabled = removedBoreholeIds.size === 0;
        }
    }
    
    // === CIRCLE TRACE HELPERS ===
    function getCircleTraceIndex() {
        if (typeof COVERAGE_TRACE_RANGES === 'undefined' || typeof currentCoverageCombo === 'undefined') {
            return null;
        }
        const ranges = COVERAGE_TRACE_RANGES[currentCoverageCombo];
        if (ranges && ranges.borehole_circles) {
            return ranges.borehole_circles[0];  // Start index of borehole circles trace
        }
        return null;
    }
    
    function isBoreholeCirclesVisible() {
        const checkbox = document.getElementById('boreholeCirclesCheckbox');
        return checkbox && checkbox.checked;
    }
    
    // === EXTRACT MASTER DATA FROM TRACE ===
    function extractMasterDataFromTrace() {
        const plotDiv = document.querySelector('.plotly-graph-div');
        const traceIdx = getCircleTraceIndex();
        if (!plotDiv || traceIdx === null || !plotDiv.data || !plotDiv.data[traceIdx]) {
            return;
        }
        
        const trace = plotDiv.data[traceIdx];
        if (!trace.customdata || trace.customdata.length === 0) {
            console.warn('No customdata in borehole circles trace');
            return;
        }
        
        // Extract unique boreholes from customdata
        // customdata format: [borehole_id, center_x, center_y, radius] or null (separator)
        const boreholes = new Map();  // borehole_id -> {x, y, radius}
        
        for (const cd of trace.customdata) {
            if (cd !== null && Array.isArray(cd) && cd.length >= 4) {
                const [bhId, cx, cy, radius] = cd;
                if (!boreholes.has(bhId)) {
                    boreholes.set(bhId, { id: bhId, x: cx, y: cy, radius: radius });
                }
            }
        }
        
        masterBoreholeData = Array.from(boreholes.values());
        console.log(`📊 Extracted master data: ${masterBoreholeData.length} boreholes`);
    }
    
    // === REGENERATE TRACE ===
    function regenerateCircleTrace() {
        const plotDiv = document.querySelector('.plotly-graph-div');
        const traceIdx = getCircleTraceIndex();
        if (!plotDiv || traceIdx === null || !masterBoreholeData) {
            return;
        }
        
        // Filter out removed boreholes
        const visibleBoreholes = masterBoreholeData.filter(bh => !removedBoreholeIds.has(bh.id));
        
        // Generate new coordinate arrays
        const numPoints = 64;
        const allX = [];
        const allY = [];
        const allCustomdata = [];
        
        for (const bh of visibleBoreholes) {
            const angles = [];
            for (let i = 0; i <= numPoints; i++) {
                angles.push(2 * Math.PI * i / numPoints);
            }
            
            for (const angle of angles) {
                const x = bh.x + bh.radius * Math.cos(angle);
                const y = bh.y + bh.radius * Math.sin(angle);
                allX.push(x);
                allY.push(y);
                allCustomdata.push([bh.id, bh.x, bh.y, bh.radius]);
            }
            
            // Separator
            allX.push(null);
            allY.push(null);
            allCustomdata.push(null);
        }
        
        // Update trace
        Plotly.restyle(plotDiv, {
            x: [allX],
            y: [allY],
            customdata: [allCustomdata]
        }, [traceIdx]);
    }
    
    // === REMOVAL HANDLER ===
    function removeCircle(boreholeId) {
        if (removedBoreholeIds.has(boreholeId)) {
            return;  // Already removed
        }
        
        removedBoreholeIds.add(boreholeId);
        removalHistory.push(boreholeId);
        regenerateCircleTrace();
        updateRemovalUI();
        showRemovalToast(`🗑️ Removed circle #${boreholeId}`, true);
    }
    
    // === UNDO/RESET HANDLERS ===
    window.undoLastCircleRemoval = function() {
        if (removalHistory.length === 0) return;
        
        const lastId = removalHistory.pop();
        removedBoreholeIds.delete(lastId);
        regenerateCircleTrace();
        updateRemovalUI();
        showRemovalToast(`↩️ Restored circle #${lastId}`, false);
    };
    
    window.resetAllCircleRemovals = function() {
        if (removedBoreholeIds.size === 0) return;
        
        const count = removedBoreholeIds.size;
        removedBoreholeIds.clear();
        removalHistory.length = 0;
        regenerateCircleTrace();
        updateRemovalUI();
        showRemovalToast(`↩️ Restored ${count} circle${count !== 1 ? 's' : ''}`, false);
    };
    
    // === EXPORT HANDLER ===
    window.exportRemovedCircles = function() {
        if (removedBoreholeIds.size === 0) {
            showRemovalToast('No circles removed to export', false);
            return;
        }
        
        // Get removed borehole details
        const removedDetails = [];
        for (const bhId of removedBoreholeIds) {
            const bh = masterBoreholeData ? masterBoreholeData.find(b => b.id === bhId) : null;
            if (bh) {
                removedDetails.push({
                    id: bh.id,
                    x: bh.x,
                    y: bh.y,
                    radius: bh.radius
                });
            } else {
                removedDetails.push({ id: bhId });
            }
        }
        
        const exportData = {
            timestamp: new Date().toISOString(),
            removed_count: removedBoreholeIds.size,
            removed_circles: removedDetails
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'removed_circles.json';
        a.click();
        URL.revokeObjectURL(url);
        
        showRemovalToast(`📥 Exported ${removedBoreholeIds.size} removed circles`, true);
    };
    
    // === PLOTLY CLICK HANDLER FOR REMOVAL ===
    function attachRemovalHandler() {
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv) {
            setTimeout(attachRemovalHandler, 500);
            return;
        }
        
        // Extract master data on first load
        setTimeout(function() {
            extractMasterDataFromTrace();
        }, 1000);
        
        // Add click handler
        plotDiv.on('plotly_click', function(data) {
            // Only handle Shift+Click when borehole circles layer is visible
            if (!data || !data.event || !data.event.shiftKey) return;
            if (!isBoreholeCirclesVisible()) return;
            
            const point = data.points && data.points[0];
            if (!point || !point.customdata) return;
            
            // Check if click is on borehole circles trace
            const circleTraceIdx = getCircleTraceIndex();
            if (circleTraceIdx === null || point.curveNumber !== circleTraceIdx) return;
            
            // customdata format: [borehole_id, center_x, center_y, radius]
            const cd = point.customdata;
            if (!Array.isArray(cd) || cd.length < 4) return;
            
            const boreholeId = cd[0];
            
            // Set flag to prevent copy-to-clipboard from running
            window.circleRemovalActive = true;
            
            // Extract master data if not done yet
            if (!masterBoreholeData) {
                extractMasterDataFromTrace();
            }
            
            removeCircle(boreholeId);
            
            // Reset flag after short delay
            setTimeout(function() {
                window.circleRemovalActive = false;
            }, 100);
        });
        
        console.log('🔴 Circle removal handler attached');
    }
    
    // === SHOW/HIDE REMOVAL CONTROLS ===
    function toggleRemovalControlsVisibility() {
        const controls = document.getElementById('circleRemovalControls');
        if (controls) {
            controls.style.display = isBoreholeCirclesVisible() ? 'block' : 'none';
        }
    }
    
    // Watch for checkbox changes
    function watchBoreholeCirclesCheckbox() {
        const checkbox = document.getElementById('boreholeCirclesCheckbox');
        if (checkbox) {
            checkbox.addEventListener('change', function() {
                toggleRemovalControlsVisibility();
                // Re-extract master data when layer is toggled on
                if (this.checked && !masterBoreholeData) {
                    setTimeout(extractMasterDataFromTrace, 500);
                }
            });
        }
    }
    
    // === BUTTON EVENT BINDINGS ===
    function bindButtonEvents() {
        const undoBtn = document.getElementById('undoLastRemoval');
        if (undoBtn) {
            undoBtn.addEventListener('click', window.undoLastCircleRemoval);
        }
        
        const resetBtn = document.getElementById('resetRemovals');
        if (resetBtn) {
            resetBtn.addEventListener('click', window.resetAllCircleRemovals);
        }
        
        const exportBtn = document.getElementById('exportRemoved');
        if (exportBtn) {
            exportBtn.addEventListener('click', window.exportRemovedCircles);
        }
    }
    
    // === INITIALIZATION ===
    if (document.readyState === 'complete') {
        setTimeout(function() {
            attachRemovalHandler();
            watchBoreholeCirclesCheckbox();
            bindButtonEvents();
            updateRemovalUI();
        }, 500);
    } else {
        window.addEventListener('load', function() {
            setTimeout(function() {
                attachRemovalHandler();
                watchBoreholeCirclesCheckbox();
                bindButtonEvents();
                updateRemovalUI();
            }, 500);
        });
    }
})();
"""


# ═══════════════════════════════════════════════════════════════════════════
# 📋 CLICK-TO-COPY TOOLTIP SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════


def generate_click_to_copy_script() -> str:
    """
    Generate JavaScript for copying hover tooltip text to clipboard on click.

    When a user clicks on a data point in the Plotly plot, this script captures
    the current hover tooltip content and copies it to the system clipboard.
    A visual feedback toast notification confirms the copy action.

    Features:
    - Captures tooltip text from Plotly's hover events
    - Uses modern Clipboard API with fallback for older browsers
    - Shows toast notification with copy confirmation
    - Handles multi-line tooltips (preserves line breaks)
    - Works with all trace types (boreholes, coverage zones, proposed locations)

    Returns:
        JavaScript code block as string (without <script> tags)
    """
    return """
(function() {
    // === CLICK-TO-COPY TOOLTIP ===
    // Copies the hover tooltip text to clipboard when user clicks on a data point
    
    let lastHoverText = null;
    
    // === TOAST NOTIFICATION ===
    function showCopyToast(message, isSuccess) {
        // Remove any existing toast
        const existingToast = document.getElementById('copyToast');
        if (existingToast) existingToast.remove();
        
        // Create toast element
        const toast = document.createElement('div');
        toast.id = 'copyToast';
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 12px 24px;
            background: ${isSuccess ? 'rgba(40, 167, 69, 0.95)' : 'rgba(220, 53, 69, 0.95)'};
            color: white;
            border-radius: 8px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            z-index: 10000;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            transition: opacity 0.3s ease-out;
            pointer-events: none;
        `;
        toast.textContent = message;
        document.body.appendChild(toast);
        
        // Auto-remove after 2 seconds
        setTimeout(function() {
            toast.style.opacity = '0';
            setTimeout(function() { toast.remove(); }, 300);
        }, 2000);
    }
    
    // === COPY TO CLIPBOARD ===
    function copyToClipboard(text) {
        if (!text) {
            showCopyToast('No tooltip text to copy', false);
            return;
        }
        
        // Modern Clipboard API (preferred)
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text)
                .then(function() {
                    showCopyToast('📋 Copied to clipboard!', true);
                })
                .catch(function(err) {
                    console.warn('Clipboard API failed, using fallback:', err);
                    fallbackCopy(text);
                });
        } else {
            // Fallback for older browsers
            fallbackCopy(text);
        }
    }
    
    function fallbackCopy(text) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.cssText = 'position: fixed; left: -9999px; top: 0;';
        document.body.appendChild(textarea);
        textarea.select();
        try {
            const success = document.execCommand('copy');
            showCopyToast(success ? '📋 Copied to clipboard!' : 'Copy failed', success);
        } catch (err) {
            showCopyToast('Copy failed: ' + err.message, false);
        }
        document.body.removeChild(textarea);
    }
    
    // === PLOTLY EVENT HANDLERS ===
    function attachPlotlyEvents() {
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv) {
            setTimeout(attachPlotlyEvents, 500);
            return;
        }
        
        // Capture hover text
        plotDiv.on('plotly_hover', function(data) {
            if (data && data.points && data.points.length > 0) {
                const point = data.points[0];
                // Extract text from hovertemplate or hovertext
                let hoverText = '';
                
                // Try to get the formatted hover text
                if (point.hovertext) {
                    hoverText = point.hovertext;
                } else if (point.text) {
                    hoverText = point.text;
                } else if (point.customdata) {
                    // Build text from customdata (used by boreholes)
                    const cd = point.customdata;
                    if (Array.isArray(cd)) {
                        // Format: [LocationID, Depth, has_spt, has_tx_total, has_tx_effective]
                        hoverText = `Location: ${cd[0] || 'Unknown'}`;
                        if (cd.length > 1 && cd[1] !== null) {
                            hoverText += `\\nDepth: ${cd[1]}m`;
                        }
                        if (cd.length > 2) {
                            const tests = [];
                            if (cd[2]) tests.push('SPT');
                            if (cd[3]) tests.push('Triaxial Total');
                            if (cd[4]) tests.push('Triaxial Effective');
                            if (tests.length > 0) {
                                hoverText += `\\nTests: ${tests.join(', ')}`;
                            }
                        }
                    } else {
                        hoverText = String(cd);
                    }
                } else {
                    // Fallback: build from x, y coordinates
                    const x = point.x !== undefined ? point.x.toFixed(1) : '?';
                    const y = point.y !== undefined ? point.y.toFixed(1) : '?';
                    hoverText = `Easting: ${x}\\nNorthing: ${y}`;
                    if (point.data && point.data.name) {
                        hoverText = `${point.data.name}\\n${hoverText}`;
                    }
                }
                
                // Clean up HTML tags if any
                hoverText = hoverText.replace(/<br>/gi, '\\n').replace(/<[^>]*>/g, '');
                lastHoverText = hoverText;
            }
        });
        
        // Clear hover text when mouse leaves
        plotDiv.on('plotly_unhover', function() {
            // Keep lastHoverText for a short time to allow click
            setTimeout(function() { lastHoverText = null; }, 500);
        });
        
        // Copy on click (skip if circle removal is active)
        plotDiv.on('plotly_click', function(data) {
            // Skip copy-to-clipboard if circle removal handler just processed this click
            if (window.circleRemovalActive) {
                return;
            }
            
            // Also skip if Shift is pressed (reserved for circle removal)
            if (data && data.event && data.event.shiftKey) {
                return;
            }
            
            if (data && data.points && data.points.length > 0) {
                const point = data.points[0];
                let clickText = '';
                const traceName = point.data && point.data.name ? point.data.name : '';
                const x = point.x !== undefined ? point.x.toFixed(1) : '?';
                const y = point.y !== undefined ? point.y.toFixed(1) : '?';
                
                // Build text similar to hover - prioritize hovertext which has full formatting
                if (point.hovertext) {
                    clickText = point.hovertext;
                } else if (point.text) {
                    clickText = point.text;
                } else if (lastHoverText) {
                    // Use captured hover text which has proper formatting from hovertemplate
                    clickText = lastHoverText;
                } else if (point.customdata) {
                    const cd = point.customdata;
                    // Check if this is a Proposed Boreholes trace (customdata is [borehole_number])
                    if (traceName.includes('Proposed') && Array.isArray(cd) && cd.length === 1) {
                        clickText = `Proposed Borehole #${cd[0]}\\nEasting: ${x}\\nNorthing: ${y}`;
                    } else if (Array.isArray(cd)) {
                        // Existing borehole format: [location_id, depth, has_spt, has_tx_total, has_tx_eff]
                        clickText = `Location: ${cd[0] || 'Unknown'}`;
                        if (cd.length > 1 && cd[1] !== null) {
                            clickText += `\\nDepth: ${cd[1]}m`;
                        }
                        if (cd.length > 2) {
                            const tests = [];
                            if (cd[2]) tests.push('SPT');
                            if (cd[3]) tests.push('Triaxial Total');
                            if (cd[4]) tests.push('Triaxial Effective');
                            if (tests.length > 0) {
                                clickText += `\\nTests: ${tests.join(', ')}`;
                            }
                        }
                    } else {
                        clickText = String(cd);
                    }
                } else {
                    clickText = `Easting: ${x}\\nNorthing: ${y}`;
                    if (traceName) {
                        clickText = `${traceName}\\n${clickText}`;
                    }
                }
                
                // Clean up and copy
                clickText = clickText.replace(/<br>/gi, '\\n').replace(/<[^>]*>/g, '');
                copyToClipboard(clickText);
            }
        });
        
        console.log('📋 Click-to-copy tooltip handler attached');
    }
    
    // === INITIALIZATION ===
    if (document.readyState === 'complete') {
        setTimeout(attachPlotlyEvents, 500);
    } else {
        window.addEventListener('load', function() {
            setTimeout(attachPlotlyEvents, 500);
        });
    }
})();
"""


# ═══════════════════════════════════════════════════════════════════════════
# �📊 COVERAGE DATA INITIALIZATION SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════


def generate_coverage_data_script(
    trace_ranges_json: str,
    stats_json: str,
    max_spacing: float,
    default_combo_key: str,
) -> str:
    """
    Generate JavaScript for coverage trace data initialization.

    This script defines global constants used by filter panel scripts
    to switch between precomputed coverage traces.

    Args:
        trace_ranges_json: JSON string mapping combo keys to trace index ranges
        stats_json: JSON string mapping combo keys to coverage statistics
        max_spacing: EC7 maximum spacing value in meters
        default_combo_key: Default filter combination key (e.g., "d0_spt0_txt0_txe0")

    Returns:
        JavaScript code block as string (without <script> tags)
    """
    return f"""// Coverage trace index ranges for visibility toggling
// Keys: d{{depth}}_spt{{0|1}}_txt{{0|1}}_txe{{0|1}}
// Values: {{covered: [start, end], gaps: [start, end], proposed_buffers: [start, end], proposed_markers: [start, end]}}
const COVERAGE_TRACE_RANGES = {trace_ranges_json};
const COVERAGE_STATS = {stats_json};
const COVERAGE_MAX_SPACING = {max_spacing};
const DEFAULT_COVERAGE_COMBO = "{default_combo_key}";
let currentCoverageCombo = DEFAULT_COVERAGE_COMBO;
"""
