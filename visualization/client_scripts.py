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
            
            setTraceVisibility(plotDiv, traceIndices, this.checked);
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
            
            setTraceVisibility(plotDiv, traceIndices, this.checked);
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
            
            setTraceVisibility(plotDiv, traceIndices, this.checked);
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
            
            setTraceVisibility(plotDiv, traceIndices, this.checked);
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
            
            setTraceVisibility(plotDiv, traceIndices, this.checked);
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
            
            setTraceVisibility(plotDiv, traceIndices, this.checked);
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
            
            setTraceVisibility(plotDiv, traceIndices, this.checked);
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
            
            setTraceVisibility(plotDiv, traceIndices, this.checked);
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
            
            setTraceVisibility(plotDiv, traceIndices, this.checked);
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
    
    // Helper to toggle trace visibility AND hoverinfo together
    // When hidden, hoverinfo is set to 'skip' so tooltips don't show
    function setTraceVisibility(plotDiv, traceIndices, isVisible) {{
        if (!plotDiv || traceIndices.length === 0) return;
        
        // Get current hoverinfo for each trace to restore when showing
        const hoverInfoUpdates = traceIndices.map(idx => {{
            const trace = plotDiv.data[idx];
            if (isVisible) {{
                // Restore original hoverinfo (use 'all' if not stored)
                return trace._originalHoverinfo || trace.hoverinfo || 'all';
            }} else {{
                // Store original hoverinfo if not already stored
                if (!trace._originalHoverinfo && trace.hoverinfo !== 'skip') {{
                    trace._originalHoverinfo = trace.hoverinfo || 'all';
                }}
                return 'skip';
            }}
        }});
        
        Plotly.restyle(plotDiv, {{
            'visible': isVisible,
            'hoverinfo': hoverInfoUpdates
        }}, traceIndices);
    }}
    
    function updateBgsLayerVisibility(layerName) {{
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv || !bgsLayers[layerName]) return;
        
        const [startIdx, endIdx] = bgsLayers[layerName];
        
        const traceIndices = [];
        for (let i = startIdx; i < endIdx; i++) {{
            traceIndices.push(i);
        }}
        
        const visibility = bgsLayerStates[layerName];
        setTraceVisibility(plotDiv, traceIndices, visibility);
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
            
            setTraceVisibility(plotDiv, traceIndices, this.checked);
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
            
            setTraceVisibility(plotDiv, traceIndices, this.checked);
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
            
            setTraceVisibility(plotDiv, traceIndices, this.checked);
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
# � CIRCLE EDITING SCRIPTS (INTERACTIVE DRAG/EXPORT)
# ═══════════════════════════════════════════════════════════════════════════


def generate_circle_data_script(
    circle_coordinates: List[Dict[str, float]],
    buffer_radius: float,
) -> str:
    """
    Generate JavaScript for circle master data initialization.

    Creates a global `masterBoreholeData` array containing all circle
    centre coordinates and radii, enabling drag operations and exports.

    Args:
        circle_coordinates: List of {"x": float, "y": float, "coverage_radius"?: float}
        buffer_radius: Default radius for circles without explicit coverage_radius

    Returns:
        JavaScript code block as string (without <script> tags)
    """
    # Build the master data array
    master_data = []
    for i, coord in enumerate(circle_coordinates):
        cx = coord["x"]
        cy = coord["y"]
        radius = coord.get("coverage_radius", buffer_radius)
        master_data.append({
            "id": i,
            "x": cx,
            "y": cy,
            "radius": radius,
            "originalX": cx,
            "originalY": cy,
            "moved": False,
        })

    master_data_json = json.dumps(master_data, indent=2)

    return f"""// Master data for proposed borehole circles (interactive editing)
const masterBoreholeData = {master_data_json};
const CIRCLE_POINTS = 64;  // Points per circle for regeneration
"""


def generate_circle_drag_script() -> str:
    """
    Generate JavaScript for circle drag-and-drop functionality.

    Implements:
    - Drag mode state management
    - Mouse event handling (click to select, move to drag, click to drop)
    - Real-time circle regeneration during drag
    - Coordinate conversion (pixel <-> data coordinates)
    - Performance throttling for smooth drag
    - Export to JSON/CSV
    - Reset to original positions

    Returns:
        JavaScript code block as string (without <script> tags)
    """
    return """
(function() {
    // === STATE MANAGEMENT ===
    let dragModeEnabled = false;
    let isDragging = false;
    let draggedCircleId = null;
    let lastMoveTime = 0;
    const THROTTLE_MS = 50;  // 20 FPS for smooth drag

    // === COORDINATE CONVERSION ===
    function pixelToDataCoords(plotDiv, pixelX, pixelY) {
        const xaxis = plotDiv._fullLayout.xaxis;
        const yaxis = plotDiv._fullLayout.yaxis;
        const margin = plotDiv._fullLayout.margin;

        // Account for plot margins
        const plotX = pixelX - margin.l;
        const plotY = pixelY - margin.t;

        // Convert pixel position to data coordinates
        const dataX = xaxis.p2d(plotX);
        const dataY = yaxis.p2d(plotY);

        return { x: dataX, y: dataY };
    }

    // === CIRCLE IDENTIFICATION ===
    function findCircleAtPoint(dataX, dataY) {
        if (typeof masterBoreholeData === 'undefined') return null;
        
        // Check if click is within any circle
        for (const circle of masterBoreholeData) {
            const dx = dataX - circle.x;
            const dy = dataY - circle.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist <= circle.radius) {
                return circle.id;
            }
        }
        return null;
    }

    // === CIRCLE REGENERATION ===
    function generateCirclePoints(centerX, centerY, radius, numPoints) {
        numPoints = numPoints || CIRCLE_POINTS;
        const xPoints = [];
        const yPoints = [];
        const angleStep = (2 * Math.PI) / numPoints;

        for (let i = 0; i <= numPoints; i++) {
            const angle = i * angleStep;
            xPoints.push(centerX + radius * Math.cos(angle));
            yPoints.push(centerY + radius * Math.sin(angle));
        }

        return { x: xPoints, y: yPoints };
    }

    function findBoreholeCirclesTraceIndex(plotDiv) {
        // Find trace by name or legendgroup
        for (let i = 0; i < plotDiv.data.length; i++) {
            const trace = plotDiv.data[i];
            if (trace.legendgroup === 'borehole_circles' || 
                (trace.name && trace.name.toLowerCase().includes('circle'))) {
                return i;
            }
        }
        return -1;
    }

    function updateCircleInTrace(plotDiv, circleId, newX, newY) {
        const circleTraceIdx = findBoreholeCirclesTraceIndex(plotDiv);
        if (circleTraceIdx === -1) return;

        const trace = plotDiv.data[circleTraceIdx];
        const circle = masterBoreholeData[circleId];
        const radius = circle.radius;

        // Calculate start/end indices for this circle's points in the trace
        // Each circle has CIRCLE_POINTS+1 points plus 1 null separator = CIRCLE_POINTS+2
        const pointsPerCircle = CIRCLE_POINTS + 2;
        const startIdx = circleId * pointsPerCircle;
        const endIdx = startIdx + pointsPerCircle - 1;  // Exclude the null

        // Generate new circle points
        const newPoints = generateCirclePoints(newX, newY, radius, CIRCLE_POINTS);

        // Create updated x/y arrays (copy existing)
        const newXArray = [...trace.x];
        const newYArray = [...trace.y];

        // Replace circle points
        for (let i = 0; i < newPoints.x.length && (startIdx + i) < endIdx; i++) {
            newXArray[startIdx + i] = newPoints.x[i];
            newYArray[startIdx + i] = newPoints.y[i];
        }

        // Update trace with new coordinates
        Plotly.restyle(plotDiv, {
            x: [newXArray],
            y: [newYArray]
        }, [circleTraceIdx]);

        // Update master data
        masterBoreholeData[circleId].x = newX;
        masterBoreholeData[circleId].y = newY;
        masterBoreholeData[circleId].moved = true;

        updateCircleStats();
    }

    // === UI HELPERS ===
    function updateStatus(message) {
        const statusDiv = document.getElementById('circleEditStatus');
        if (statusDiv) statusDiv.textContent = message;
    }

    function showDragCoords(x, y) {
        const coordsDiv = document.getElementById('dragCoords');
        if (coordsDiv) coordsDiv.style.display = 'block';
        const xSpan = document.getElementById('dragX');
        const ySpan = document.getElementById('dragY');
        if (xSpan) xSpan.textContent = x.toFixed(1);
        if (ySpan) ySpan.textContent = y.toFixed(1);
    }

    function hideDragCoords() {
        const coordsDiv = document.getElementById('dragCoords');
        if (coordsDiv) coordsDiv.style.display = 'none';
    }

    function countMovedCircles() {
        if (typeof masterBoreholeData === 'undefined') return 0;
        return masterBoreholeData.filter(c => c.moved).length;
    }

    function updateCircleStats() {
        if (typeof masterBoreholeData === 'undefined') return;
        const totalSpan = document.getElementById('totalCircles');
        const movedSpan = document.getElementById('movedCircles');
        if (totalSpan) totalSpan.textContent = masterBoreholeData.length;
        if (movedSpan) movedSpan.textContent = countMovedCircles();
    }

    // === EXPORT FUNCTIONS ===
    function exportCirclesJSON() {
        if (typeof masterBoreholeData === 'undefined') {
            showExportToast('No circle data available', false);
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            totalCircles: masterBoreholeData.length,
            movedCircles: countMovedCircles(),
            circles: masterBoreholeData.map(c => ({
                id: c.id,
                x: c.x,
                y: c.y,
                radius: c.radius,
                originalX: c.originalX,
                originalY: c.originalY,
                moved: c.moved
            }))
        };

        const jsonStr = JSON.stringify(exportData, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'borehole_coordinates_' + new Date().toISOString().slice(0,10) + '.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showExportToast('Coordinates exported to JSON', true);
    }

    function exportCirclesCSV() {
        if (typeof masterBoreholeData === 'undefined') {
            showExportToast('No circle data available', false);
            return;
        }

        const headers = ['id', 'x', 'y', 'radius', 'original_x', 'original_y', 'moved'];
        const rows = masterBoreholeData.map(c => [
            c.id, c.x, c.y, c.radius, c.originalX, c.originalY, c.moved
        ]);
        
        const csvContent = [headers.join(',')]
            .concat(rows.map(r => r.join(',')))
            .join('\\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'borehole_coordinates_' + new Date().toISOString().slice(0,10) + '.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showExportToast('Coordinates exported to CSV', true);
    }

    function showExportToast(message, isSuccess) {
        const toast = document.createElement('div');
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
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 2000);
    }

    // === RESET FUNCTION ===
    function resetAllCircles() {
        if (typeof masterBoreholeData === 'undefined') return;
        
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv) return;

        masterBoreholeData.forEach(circle => {
            if (circle.moved) {
                updateCircleInTrace(plotDiv, circle.id, circle.originalX, circle.originalY);
                circle.x = circle.originalX;
                circle.y = circle.originalY;
                circle.moved = false;
            }
        });

        updateCircleStats();
        updateStatus('All circles reset to original positions');
    }

    // === DRAG MODE INITIALIZATION ===
    function initDragMode() {
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv) {
            setTimeout(initDragMode, 500);
            return;
        }

        // Toggle button handler
        const toggleBtn = document.getElementById('toggleDragMode');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', function() {
                dragModeEnabled = !dragModeEnabled;
                
                const indicator = document.getElementById('dragModeIndicator');
                if (dragModeEnabled) {
                    this.textContent = 'Disable Drag Mode';
                    this.style.background = '#dc3545';
                    if (indicator) indicator.style.display = 'block';
                    // Disable Plotly's default dragmode
                    Plotly.relayout(plotDiv, { dragmode: false });
                    updateStatus('Click on a circle to start dragging');
                } else {
                    this.textContent = 'Enable Drag Mode';
                    this.style.background = '#28a745';
                    if (indicator) indicator.style.display = 'none';
                    // Re-enable pan mode
                    Plotly.relayout(plotDiv, { dragmode: 'pan' });
                    isDragging = false;
                    draggedCircleId = null;
                    hideDragCoords();
                    updateStatus('Click "Enable Drag Mode" to reposition circles');
                }
            });
        }

        // Click handler for selecting/placing circles
        plotDiv.addEventListener('click', function(event) {
            if (!dragModeEnabled) return;
            
            const rect = plotDiv.getBoundingClientRect();
            const pixelX = event.clientX - rect.left;
            const pixelY = event.clientY - rect.top;
            const dataCoords = pixelToDataCoords(plotDiv, pixelX, pixelY);

            if (!isDragging) {
                // Try to select a circle
                const circleId = findCircleAtPoint(dataCoords.x, dataCoords.y);
                if (circleId !== null) {
                    isDragging = true;
                    draggedCircleId = circleId;
                    updateStatus('Dragging circle #' + (circleId + 1) + ' - click to place');
                    showDragCoords(dataCoords.x, dataCoords.y);
                }
            } else {
                // Place the circle at current position
                if (draggedCircleId !== null) {
                    updateCircleInTrace(plotDiv, draggedCircleId, dataCoords.x, dataCoords.y);
                    updateStatus('Circle #' + (draggedCircleId + 1) + ' placed - click another to drag');
                }
                isDragging = false;
                draggedCircleId = null;
                hideDragCoords();
            }
        });

        // Mouse move handler for drag preview
        plotDiv.addEventListener('mousemove', function(event) {
            if (!dragModeEnabled || !isDragging || draggedCircleId === null) return;
            
            const now = Date.now();
            if (now - lastMoveTime < THROTTLE_MS) return;
            lastMoveTime = now;

            const rect = plotDiv.getBoundingClientRect();
            const pixelX = event.clientX - rect.left;
            const pixelY = event.clientY - rect.top;
            const dataCoords = pixelToDataCoords(plotDiv, pixelX, pixelY);

            // Update circle position in real-time
            updateCircleInTrace(plotDiv, draggedCircleId, dataCoords.x, dataCoords.y);
            showDragCoords(dataCoords.x, dataCoords.y);
        });

        // Escape key to cancel drag
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && isDragging && draggedCircleId !== null) {
                // Reset to position before this drag started
                const circle = masterBoreholeData[draggedCircleId];
                updateCircleInTrace(plotDiv, draggedCircleId, circle.originalX, circle.originalY);
                circle.moved = false;
                isDragging = false;
                draggedCircleId = null;
                hideDragCoords();
                updateStatus('Drag cancelled');
            }
        });

        // Export button handlers
        document.getElementById('exportCoordsJSON')?.addEventListener('click', exportCirclesJSON);
        document.getElementById('exportCoordsCSV')?.addEventListener('click', exportCirclesCSV);
        document.getElementById('resetAllCircles')?.addEventListener('click', resetAllCircles);

        // Initialize stats
        updateCircleStats();
        console.log('🔄 Circle drag mode initialized');
    }

    // === INITIALIZATION ===
    if (document.readyState === 'complete') {
        setTimeout(initDragMode, 500);
    } else {
        window.addEventListener('load', function() {
            setTimeout(initDragMode, 500);
        });
    }
})();
"""


# ═══════════════════════════════════════════════════════════════════════════
# �🔍 FILTER PANEL SCRIPTS
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
    // Store original hover templates for filtered boreholes
    let originalBoreholeHovertemplates = null;
    
    function initializeBoreholeFilter() {
        if (filterInitialized) return true;
        const plotDiv = getPlotDiv();
        if (!plotDiv || !plotDiv.data || !plotDiv.data[BOREHOLE_TRACE_IDX]) return false;
        
        const trace = plotDiv.data[BOREHOLE_TRACE_IDX];
        if (!trace.customdata || trace.customdata.length === 0) return false;
        
        // Store original hover templates (build array from template string + customdata)
        const templateStr = trace.hovertemplate || '';
        originalBoreholeHovertemplates = trace.customdata.map(function(d, idx) {
            // Build per-point hover text from customdata
            const locId = Array.isArray(d) ? d[0] : '';
            const depth = Array.isArray(d) && d.length > 1 ? parseFloat(d[1]) || 0 : 0;
            const x = trace.x[idx];
            const y = trace.y[idx];
            return '<b>' + locId + '</b><br>Easting: ' + Math.round(x).toLocaleString() + 
                   '<br>Northing: ' + Math.round(y).toLocaleString() + 
                   '<br>Final Depth: ' + depth.toFixed(1) + 'm<extra></extra>';
        });
        
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
        
        // Update visibility and hoverinfo together (hide tooltips for hidden traces)
        if (traceIndices.length > 0) {
            const hoverInfoUpdates = visibilities.map(function(v, idx) {
                const trace = plotDiv.data[traceIndices[idx]];
                if (v) {
                    return trace._originalHoverinfo || trace.hoverinfo || 'all';
                } else {
                    if (!trace._originalHoverinfo && trace.hoverinfo !== 'skip') {
                        trace._originalHoverinfo = trace.hoverinfo || 'all';
                    }
                    return 'skip';
                }
            });
            Plotly.restyle(plotDiv, {'visible': visibilities, 'hoverinfo': hoverInfoUpdates}, traceIndices);
        }
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
        
        const opacities = [];
        const hovertemplates = [];
        
        boreholeData.forEach(function(bh, idx) {
            let visible = true;
            if (bh.depth < minDepthFilter) visible = false;
            if (filterSPT && !bh.has_spt) visible = false;
            if (filterTriaxialTotal && !bh.has_triaxial_total) visible = false;
            if (filterTriaxialEffective && !bh.has_triaxial_effective) visible = false;
            
            opacities.push(visible ? 1 : 0);
            // Set hovertemplate to empty string for hidden points (disables tooltip)
            hovertemplates.push(visible ? originalBoreholeHovertemplates[idx] : '');
        });
        
        Plotly.restyle(plotDiv, {
            'marker.opacity': [opacities],
            'hovertemplate': [hovertemplates]
        }, [BOREHOLE_TRACE_IDX]);
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
# � CLICK-TO-COPY TOOLTIP SCRIPTS
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
        
        // Copy on Ctrl+Click (changed from regular click to allow circle dragging)
        plotDiv.on('plotly_click', function(data) {
            // Only copy if Ctrl key is held down
            if (!window.event || !window.event.ctrlKey) {
                return;  // Allow normal click for circle dragging
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
        
        console.log('📋 Ctrl+Click-to-copy tooltip handler attached');
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
