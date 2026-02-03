# Interactive Circle Movement - Implementation Plan

## Branch: `feature/interactive-circle-movement`

## Overview

This document details the implementation plan for adding draggable circle functionality to the EC7 HTML visualization, enabling users to manually reposition proposed borehole circles and export updated coordinates.

---

## Architecture Context

### Current System Structure

```
visualization/
├── client_scripts.py    # JavaScript generation (layer toggles, filters, click-to-copy)
├── html_panels.py       # HTML panel generators (filters, layers, legend, stats)
├── html_builder.py      # Main assembly: combines traces, panels, scripts into final HTML
└── plotly_traces.py     # Plotly trace builders (boreholes, coverage, circles)
```

### Existing Circle Implementation

Circles are rendered via `build_borehole_circles_trace()` in `plotly_traces.py`:
- Each circle = 64 points forming a closed polygon
- Rendered as `go.Scattergl` trace with `mode="lines"`
- No `customdata` currently stored (only `hoverinfo="skip"`)

### Key Insight

The analysis document references `masterBoreholeData` but this doesn't exist in the current implementation. We need to create this data structure and the accompanying drag/export functionality.

---

## Implementation Phases

### Phase 1: Data Infrastructure

**Goal:** Establish JavaScript-accessible circle data structure

#### 1.1 Modify `plotly_traces.py`

Add `customdata` to circle traces for identification:

```python
# In build_borehole_circles_trace()
# Add customdata storing: [borehole_id, center_x, center_y, radius]
customdata = []
for i, coord in enumerate(coordinates):
    cx, cy = coord["x"], coord["y"]
    radius = coord.get("coverage_radius", buffer_radius)
    # Store same data for each of the 64 points in the circle
    customdata.extend([[i, cx, cy, radius]] * (n_circle_points + 1))
```

#### 1.2 Create `generate_circle_data_script()` in `client_scripts.py`

```python
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
        JavaScript code block as string
    """
```

Output format:
```javascript
// Master data for proposed borehole circles
const masterBoreholeData = [
    { id: 0, x: 451234.5, y: 173456.2, radius: 150.0, originalX: 451234.5, originalY: 173456.2, moved: false },
    { id: 1, x: 451567.8, y: 173890.1, radius: 150.0, originalX: 451567.8, originalY: 173890.1, moved: false },
    // ...
];
const CIRCLE_POINTS = 64;  // Points per circle for regeneration
```

---

### Phase 2: Drag Mode UI

**Goal:** Add user controls for drag mode activation

#### 2.1 Add to `html_panels.py`

Create new function `generate_circle_editing_panel_html()`:

```python
def generate_circle_editing_panel_html(
    panel_width: int = DEFAULT_RIGHT_PANEL_WIDTH,
    vertical_gap: int = DEFAULT_PANEL_VERTICAL_GAP,
) -> str:
    """
    Generate HTML for circle editing controls panel.

    Provides:
    - Drag mode toggle button
    - Real-time coordinate display during drag
    - Export buttons (JSON and CSV)
    - Reset button to restore original positions
    """
```

HTML structure:
```html
<div id="circleEditingPanel" style="{LIQUID_GLASS_STYLE} ...">
    <div style="{PANEL_STYLE_HEADER}">Circle Editing</div>

    <!-- Drag Mode Toggle -->
    <button id="toggleDragMode" style="...">
        Enable Drag Mode
    </button>
    <div id="dragModeIndicator" style="display: none; color: #28a745; font-weight: bold;">
        DRAG MODE ACTIVE
    </div>

    <!-- Real-time Coordinates -->
    <div id="dragCoords" style="display: none; margin-top: 10px;">
        <div style="font-size: 10px; color: #666;">Dragging circle:</div>
        <div>X: <span id="dragX">-</span></div>
        <div>Y: <span id="dragY">-</span></div>
    </div>

    <!-- Status Messages -->
    <div id="circleEditStatus" style="margin-top: 10px; font-size: 10px; color: #666;">
        Click "Enable Drag Mode" to reposition circles
    </div>

    <!-- Export Section -->
    <div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid #ddd;">
        <div style="font-weight: 500; margin-bottom: 6px;">Export Coordinates</div>
        <button id="exportCoordsJSON" style="...">Export JSON</button>
        <button id="exportCoordsCSV" style="...">Export CSV</button>
    </div>

    <!-- Reset Section -->
    <div style="margin-top: 8px;">
        <button id="resetAllCircles" style="...">Reset All Positions</button>
    </div>

    <!-- Statistics -->
    <div id="circleStats" style="margin-top: 10px; font-size: 10px;">
        <span id="totalCircles">0</span> circles |
        <span id="movedCircles">0</span> moved
    </div>
</div>
```

---

### Phase 3: Core Drag Functionality

**Goal:** Implement drag interactions in JavaScript

#### 3.1 Create `generate_circle_drag_script()` in `client_scripts.py`

```python
def generate_circle_drag_script() -> str:
    """
    Generate JavaScript for circle drag-and-drop functionality.

    Implements:
    - Drag mode state management
    - Mouse event handling (click to select, move to drag, click to drop)
    - Real-time circle regeneration during drag
    - Coordinate conversion (pixel <-> data coordinates)
    - Performance throttling for smooth drag
    """
```

Key JavaScript functions:

```javascript
// === STATE MANAGEMENT ===
let dragModeEnabled = false;
let isDragging = false;
let draggedCircleId = null;
let dragOffset = { x: 0, y: 0 };  // Offset from click point to circle center

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

function dataToPixelCoords(plotDiv, dataX, dataY) {
    const xaxis = plotDiv._fullLayout.xaxis;
    const yaxis = plotDiv._fullLayout.yaxis;
    const margin = plotDiv._fullLayout.margin;

    const pixelX = xaxis.d2p(dataX) + margin.l;
    const pixelY = yaxis.d2p(dataY) + margin.t;

    return { x: pixelX, y: pixelY };
}

// === CIRCLE IDENTIFICATION ===
function findCircleAtPoint(dataX, dataY) {
    // Check if click is within any circle
    for (const circle of masterBoreholeData) {
        const dx = dataX - circle.x;
        const dy = dataY - circle.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance <= circle.radius) {
            return circle.id;
        }
    }
    return null;
}

// === CIRCLE REGENERATION ===
function generateCirclePoints(centerX, centerY, radius, numPoints = 64) {
    const xPoints = [];
    const yPoints = [];
    const angleStep = (2 * Math.PI) / numPoints;

    for (let i = 0; i <= numPoints; i++) {
        const angle = i * angleStep;
        xPoints.push(centerX + radius * Math.cos(angle));
        yPoints.push(centerY + radius * Math.sin(angle));
    }

    // Add null to separate from other circles in same trace
    xPoints.push(null);
    yPoints.push(null);

    return { x: xPoints, y: yPoints };
}

function updateCircleInTrace(plotDiv, circleId, newX, newY) {
    // Find the borehole circles trace
    const circleTraceIdx = findBoreholeCirclesTraceIndex(plotDiv);
    if (circleTraceIdx === -1) return;

    const trace = plotDiv.data[circleTraceIdx];
    const circle = masterBoreholeData[circleId];
    const radius = circle.radius;

    // Calculate start/end indices for this circle's points in the trace
    const pointsPerCircle = CIRCLE_POINTS + 2;  // 64 points + closing point + null separator
    const startIdx = circleId * pointsPerCircle;
    const endIdx = startIdx + pointsPerCircle;

    // Generate new circle points
    const newPoints = generateCirclePoints(newX, newY, radius, CIRCLE_POINTS);

    // Create updated x/y arrays
    const newXArray = [...trace.x];
    const newYArray = [...trace.y];

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
}

// === DRAG EVENT HANDLERS ===
function initDragMode() {
    const plotDiv = document.querySelector('.plotly-graph-div');
    if (!plotDiv) return;

    // Toggle button handler
    document.getElementById('toggleDragMode').addEventListener('click', function() {
        dragModeEnabled = !dragModeEnabled;
        this.textContent = dragModeEnabled ? 'Disable Drag Mode' : 'Enable Drag Mode';
        this.style.background = dragModeEnabled ? '#dc3545' : '#28a745';

        document.getElementById('dragModeIndicator').style.display = dragModeEnabled ? 'block' : 'none';

        // Disable Plotly's default drag mode when in circle drag mode
        if (dragModeEnabled) {
            Plotly.relayout(plotDiv, { dragmode: false });
            plotDiv.style.cursor = 'crosshair';
            updateStatus('Click on a circle to drag it');
        } else {
            Plotly.relayout(plotDiv, { dragmode: 'pan' });
            plotDiv.style.cursor = 'default';
            updateStatus('Drag mode disabled');
        }
    });

    // Click handler for selecting circles
    plotDiv.on('plotly_click', function(data) {
        if (!dragModeEnabled) return;

        const point = data.points[0];
        const clickX = point.x;
        const clickY = point.y;

        if (!isDragging) {
            // First click: select circle
            const circleId = findCircleAtPoint(clickX, clickY);
            if (circleId !== null) {
                isDragging = true;
                draggedCircleId = circleId;
                const circle = masterBoreholeData[circleId];
                dragOffset = {
                    x: clickX - circle.x,
                    y: clickY - circle.y
                };
                highlightCircle(circleId);
                updateStatus(`Dragging circle #${circleId + 1}`);
                showDragCoords(circle.x, circle.y);
            }
        } else {
            // Second click: drop circle
            const newX = clickX - dragOffset.x;
            const newY = clickY - dragOffset.y;
            updateCircleInTrace(plotDiv, draggedCircleId, newX, newY);

            isDragging = false;
            draggedCircleId = null;
            unhighlightCircle();
            hideDragCoords();
            updateStatus(`Circle moved. ${countMovedCircles()} circles modified.`);
            updateCircleStats();
        }
    });

    // Mouse move handler for drag preview
    let lastMoveTime = 0;
    const throttleMs = 50;  // Update max 20 times per second

    plotDiv.addEventListener('mousemove', function(e) {
        if (!isDragging || !dragModeEnabled) return;

        const now = Date.now();
        if (now - lastMoveTime < throttleMs) return;
        lastMoveTime = now;

        const rect = plotDiv.getBoundingClientRect();
        const pixelX = e.clientX - rect.left;
        const pixelY = e.clientY - rect.top;

        const dataCoords = pixelToDataCoords(plotDiv, pixelX, pixelY);
        const newX = dataCoords.x - dragOffset.x;
        const newY = dataCoords.y - dragOffset.y;

        // Update circle position in real-time
        updateCircleInTrace(plotDiv, draggedCircleId, newX, newY);
        showDragCoords(newX, newY);
    });

    // Escape key to cancel drag
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && isDragging) {
            // Restore original position
            const circle = masterBoreholeData[draggedCircleId];
            updateCircleInTrace(plotDiv, draggedCircleId, circle.originalX, circle.originalY);
            circle.x = circle.originalX;
            circle.y = circle.originalY;
            circle.moved = false;

            isDragging = false;
            draggedCircleId = null;
            unhighlightCircle();
            hideDragCoords();
            updateStatus('Drag cancelled');
        }
    });
}

// === UI HELPERS ===
function updateStatus(message) {
    const statusDiv = document.getElementById('circleEditStatus');
    if (statusDiv) statusDiv.textContent = message;
}

function showDragCoords(x, y) {
    const coordsDiv = document.getElementById('dragCoords');
    const xSpan = document.getElementById('dragX');
    const ySpan = document.getElementById('dragY');
    if (coordsDiv) coordsDiv.style.display = 'block';
    if (xSpan) xSpan.textContent = x.toFixed(1);
    if (ySpan) ySpan.textContent = y.toFixed(1);
}

function hideDragCoords() {
    const coordsDiv = document.getElementById('dragCoords');
    if (coordsDiv) coordsDiv.style.display = 'none';
}

function highlightCircle(circleId) {
    // Change circle color to indicate selection
    const plotDiv = document.querySelector('.plotly-graph-div');
    const circleTraceIdx = findBoreholeCirclesTraceIndex(plotDiv);
    if (circleTraceIdx === -1) return;

    // Store original color and set highlight
    Plotly.restyle(plotDiv, {
        'line.color': 'rgba(255, 165, 0, 1)',  // Orange highlight
        'line.width': 3
    }, [circleTraceIdx]);
}

function unhighlightCircle() {
    const plotDiv = document.querySelector('.plotly-graph-div');
    const circleTraceIdx = findBoreholeCirclesTraceIndex(plotDiv);
    if (circleTraceIdx === -1) return;

    Plotly.restyle(plotDiv, {
        'line.color': 'rgba(0, 100, 255, 0.7)',  // Restore original
        'line.width': 2
    }, [circleTraceIdx]);
}

function countMovedCircles() {
    return masterBoreholeData.filter(c => c.moved).length;
}

function updateCircleStats() {
    const totalSpan = document.getElementById('totalCircles');
    const movedSpan = document.getElementById('movedCircles');
    if (totalSpan) totalSpan.textContent = masterBoreholeData.length;
    if (movedSpan) movedSpan.textContent = countMovedCircles();
}

function findBoreholeCirclesTraceIndex(plotDiv) {
    // Find trace by name or legendgroup
    for (let i = 0; i < plotDiv.data.length; i++) {
        const trace = plotDiv.data[i];
        if (trace.legendgroup === 'borehole_circles' ||
            (trace.name && trace.name.includes('Circles'))) {
            return i;
        }
    }
    return -1;
}
```

---

### Phase 4: Export Functionality

**Goal:** Enable coordinate export in JSON and CSV formats

#### 4.1 Add to `generate_circle_drag_script()`

```javascript
// === EXPORT FUNCTIONS ===
function exportCirclesJSON() {
    const exportData = {
        timestamp: new Date().toISOString(),
        projection: "EPSG:27700",
        total_circles: masterBoreholeData.length,
        circles_moved: countMovedCircles(),
        circles: masterBoreholeData.map(c => ({
            id: c.id,
            x: c.x,
            y: c.y,
            radius: c.radius,
            original_x: c.originalX,
            original_y: c.originalY,
            moved: c.moved
        }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `borehole_coordinates_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showExportToast('Coordinates exported to JSON');
}

function exportCirclesCSV() {
    const headers = ['id', 'x', 'y', 'radius', 'original_x', 'original_y', 'moved'];
    const rows = masterBoreholeData.map(c =>
        [c.id, c.x, c.y, c.radius, c.originalX, c.originalY, c.moved].join(',')
    );

    const csvContent = [headers.join(','), ...rows].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `borehole_coordinates_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showExportToast('Coordinates exported to CSV');
}

function showExportToast(message) {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        padding: 12px 24px;
        background: rgba(40, 167, 69, 0.95);
        color: white;
        border-radius: 8px;
        font-family: sans-serif;
        z-index: 10000;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 2000);
}

// === RESET FUNCTION ===
function resetAllCircles() {
    const plotDiv = document.querySelector('.plotly-graph-div');

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

// === EVENT LISTENER SETUP ===
document.getElementById('exportCoordsJSON')?.addEventListener('click', exportCirclesJSON);
document.getElementById('exportCoordsCSV')?.addEventListener('click', exportCirclesCSV);
document.getElementById('resetAllCircles')?.addEventListener('click', resetAllCircles);
```

---

### Phase 5: Integration

**Goal:** Wire everything together in `html_builder.py`

#### 5.1 Update `html_builder.py`

Add imports:
```python
from Gap_Analysis_EC7.visualization.client_scripts import (
    # ... existing imports ...
    generate_circle_data_script,
    generate_circle_drag_script,
)
from Gap_Analysis_EC7.visualization.html_panels import (
    # ... existing imports ...
    generate_circle_editing_panel_html,
)
```

In the main build function, add:
```python
# Generate circle editing panel (if circles are present)
if has_borehole_circles:
    circle_editing_panel = generate_circle_editing_panel_html(
        panel_width=RIGHT_PANEL_WIDTH,
        vertical_gap=PANEL_VERTICAL_GAP,
    )
    right_panels.append(circle_editing_panel)

# Add circle data script before other scripts
if has_borehole_circles:
    circle_data_script = generate_circle_data_script(
        circle_coordinates=proposed_coordinates,
        buffer_radius=max_spacing,
    )
    scripts.append(f"<script>{circle_data_script}</script>")

    circle_drag_script = generate_circle_drag_script()
    scripts.append(f"<script>{circle_drag_script}</script>")
```

---

## File Modification Summary

| File | Changes |
|------|---------|
| `visualization/plotly_traces.py` | Add `customdata` to `build_borehole_circles_trace()` |
| `visualization/client_scripts.py` | Add `generate_circle_data_script()` and `generate_circle_drag_script()` |
| `visualization/html_panels.py` | Add `generate_circle_editing_panel_html()` |
| `visualization/html_builder.py` | Import new functions, add panel and scripts to build output |
| `visualization/__init__.py` | Export new functions |

---

## Testing Plan

### Unit Tests

1. **Circle data script generation**
   - Verify JSON structure of `masterBoreholeData`
   - Test with varying numbers of circles
   - Test with mixed coverage radii

2. **Coordinate conversion**
   - Test `pixelToDataCoords()` and `dataToPixelCoords()` round-trip
   - Verify accuracy at plot edges

3. **Circle regeneration**
   - Verify 64-point circle geometry
   - Test with different radii

### Integration Tests

1. **Drag interaction**
   - Click to select circle
   - Move mouse to drag
   - Click to drop
   - Escape to cancel

2. **Export validation**
   - JSON format matches specification
   - CSV headers and data correct
   - Timestamps in ISO format

3. **Reset functionality**
   - All circles return to original positions
   - `moved` flags reset to false

### Manual Testing Checklist

- [ ] Enable drag mode toggles Plotly dragmode off
- [ ] Clicking circle highlights it
- [ ] Mouse move updates circle position in real-time
- [ ] Dropping circle updates master data
- [ ] Escape cancels drag and restores position
- [ ] Export JSON contains all fields
- [ ] Export CSV matches JSON data
- [ ] Reset returns all circles to original positions
- [ ] Statistics update correctly
- [ ] Works after zoom/pan operations
- [ ] Multiple drag operations work in sequence

---

## Performance Considerations

### Throttling

Mouse move events fire rapidly (60+ times/second). The implementation uses a 50ms throttle to limit updates to 20 FPS, balancing smoothness with performance.

### Trace Updates

`Plotly.restyle()` is more efficient than `Plotly.react()` for updating a single trace. However, if performance is still an issue:

**Option A: Drag Preview Mode**
Show only the circle center point during drag, render full circle on drop:
```javascript
// During drag: show center marker only
Plotly.restyle(plotDiv, {
    x: [[newX]],
    y: [[newY]],
    mode: 'markers'
}, [dragPreviewTraceIdx]);

// On drop: render full circle
updateCircleInTrace(plotDiv, circleId, newX, newY);
```

**Option B: Canvas Overlay**
Render a temporary canvas circle during drag, then update Plotly on drop. More complex but potentially smoother.

---

## Alternative: Click-to-Move Mode

If continuous drag proves problematic, implement simpler click-to-move:

1. Click circle to select (highlight)
2. Click destination to move circle there
3. Enter to confirm, Escape to cancel

This eliminates continuous mouse tracking but is less intuitive.

---

## Export Format Specification

### JSON Format
```json
{
    "timestamp": "2026-02-03T10:30:45.123Z",
    "projection": "EPSG:27700",
    "total_circles": 25,
    "circles_moved": 3,
    "circles": [
        {
            "id": 0,
            "x": 451234.5,
            "y": 173456.2,
            "radius": 150.0,
            "original_x": 451234.5,
            "original_y": 173456.2,
            "moved": false
        }
    ]
}
```

### CSV Format
```csv
id,x,y,radius,original_x,original_y,moved
0,451234.5,173456.2,150.0,451234.5,173456.2,false
1,451567.8,173890.1,150.0,451500.0,173800.0,true
```

---

## Implementation Order

1. **Phase 1** - Data infrastructure (foundation)
2. **Phase 2** - UI panel (visible progress for testing)
3. **Phase 3** - Core drag functionality (main feature)
4. **Phase 4** - Export functionality (completes feature)
5. **Phase 5** - Integration and testing

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Plotly coordinate conversion issues | High | Test thoroughly at various zoom levels |
| Performance with many circles | Medium | Implement throttling, consider preview mode |
| WebGL rendering artifacts | Low | Test on different browsers/GPUs |
| Zoom/pan breaks drag state | Medium | Cancel drag on zoom/pan events |
| Circle overlap during drag | Low | Allow overlaps, user responsibility |

---

## Success Criteria

1. Users can drag circles to new positions smoothly
2. Coordinate exports include all required fields
3. Reset restores all original positions
4. Feature works on Chrome, Firefox, Edge
5. No performance degradation with 50+ circles
6. Existing functionality (removal, layer toggles) unaffected
