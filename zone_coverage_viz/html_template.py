"""
HTML template generator for zone coverage visualization.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURAL OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

Responsibility: Generate a complete, self-contained HTML file with embedded
deck.gl application, data, and dependencies.

Key Features:
- Standalone HTML that works offline
- CDN-loaded or bundled libraries
- Embedded zone/borehole data
- Pre-computed initial coverage
- Full application JavaScript

For Navigation: Use VS Code outline (Ctrl+Shift+O)

═══════════════════════════════════════════════════════════════════════════════
"""

import json
from pathlib import Path
from typing import Dict, Any

from .config_types import ZoneCoverageConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 📄 TEMPLATE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def generate_html(
    zones_json: str,
    boreholes_json: str,
    positions_json: str,
    coverage_json: str,
    config: ZoneCoverageConfig,
    main_app_code: str,
    worker_code: str,
    ui_code: str,
    center: tuple,
    turf_code: str = "",
) -> str:
    """
    Generate complete standalone HTML with embedded code and data.

    Args:
        zones_json: Compact JSON string of zone GeoJSON
        boreholes_json: Compact JSON string of borehole GeoJSON
        positions_json: Compact JSON array of [lon, lat] positions
        coverage_json: JSON object of borehole_id -> geometry
        config: Visualization configuration
        main_app_code: Main deck.gl application JavaScript
        worker_code: Web Worker JavaScript code
        ui_code: UI controls JavaScript code
        center: (lon, lat) center for initial view
        turf_code: Turf.js library code (for worker)

    Returns:
        Complete HTML string
    """

    # === BUILD CONFIG OBJECT ===
    viz_config = {
        "center": list(center),
        "zoom": config.default_zoom,
        "enableDrag": config.enable_drag,
        "defaultMaxSpacing": config.default_max_spacing,
        "zoneStyle": {
            "fillColor": list(config.zone_style.fill_color),
            "lineColor": list(config.zone_style.line_color),
            "lineWidth": config.zone_style.line_width,
        },
        "boreholeStyle": {
            "radiusPixels": config.borehole_style.radius_pixels,
            "fillColor": list(config.borehole_style.fill_color),
            "lineColor": list(config.borehole_style.line_color),
            "lineWidth": config.borehole_style.line_width,
        },
        "coverageStyle": {
            "fillColor": list(config.coverage_style.fill_color),
            "lineColor": list(config.coverage_style.line_color),
            "lineWidth": config.coverage_style.line_width,
        },
    }

    viz_config_json = json.dumps(viz_config, separators=(",", ":"))

    # === LIBRARY LOADING ===
    if config.use_cdn_libs:
        lib_scripts = """
    <!-- deck.gl from CDN -->
    <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
    <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />
"""
        turf_cdn = """
    <!-- Turf.js from CDN (for worker initialization) -->
    <script>
        // Fetch Turf.js code for Web Worker
        fetch('https://unpkg.com/@turf/turf@6.5.0/turf.min.js')
            .then(r => r.text())
            .then(code => { window.TURF_CODE = code; })
            .catch(e => console.warn('Failed to load Turf.js:', e));
    </script>
"""
    else:
        # Bundled libraries (would need to be embedded)
        lib_scripts = """
    <!-- Bundled libraries -->
    <script>/* deck.gl bundled */</script>
"""
        turf_cdn = ""

    # === INLINE TURF CODE ===
    turf_script = ""
    if turf_code:
        # Escape for embedding
        escaped_turf = (
            turf_code.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
        )
        turf_script = f"""
    <script>
        // Turf.js code for Web Worker
        const TURF_CODE = `{escaped_turf}`;
    </script>
"""

    # === BUILD HTML ===
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config.title}</title>
    
    {lib_scripts}
    
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            overflow: hidden;
        }}
        
        #deckgl-container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
        
        /* Control Panel */
        .control-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
            padding: 15px;
            min-width: 220px;
            max-height: calc(100vh - 40px);
            overflow-y: auto;
            z-index: 1000;
        }}
        
        .control-panel h3 {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #333;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }}
        
        .control-section {{
            margin-bottom: 15px;
        }}
        
        .control-section h4 {{
            font-size: 12px;
            font-weight: 600;
            color: #666;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 13px;
        }}
        
        .stat-label {{
            color: #666;
        }}
        
        .stat-value {{
            font-weight: 600;
            color: #333;
        }}
        
        .btn {{
            display: block;
            width: 100%;
            padding: 8px 12px;
            margin-top: 8px;
            border: none;
            border-radius: 4px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        
        .btn-primary {{
            background: #4169E1;
            color: white;
        }}
        
        .btn-primary:hover {{
            background: #3157c7;
        }}
        
        .btn-secondary {{
            background: #e0e0e0;
            color: #333;
        }}
        
        .btn-secondary:hover {{
            background: #d0d0d0;
        }}
        
        .btn-danger {{
            background: #dc3545;
            color: white;
        }}
        
        .btn-danger:hover {{
            background: #c82333;
        }}
        
        /* Layer toggles */
        .layer-toggle-row {{
            display: flex;
            align-items: center;
            padding: 4px 0;
        }}
        
        .layer-toggle-row input {{
            margin-right: 8px;
        }}
        
        .layer-toggle-row label {{
            font-size: 13px;
            color: #333;
        }}
        
        /* Instructions */
        .instructions {{
            font-size: 11px;
            color: #888;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }}
        
        .instructions p {{
            margin-bottom: 4px;
        }}
        
        .kbd {{
            display: inline-block;
            padding: 2px 5px;
            font-family: monospace;
            font-size: 10px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}
        
        /* Tooltip */
        #tooltip {{
            position: absolute;
            pointer-events: none;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            max-width: 300px;
            z-index: 1001;
            display: none;
        }}
        
        /* Loading overlay */
        .loading-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 2000;
        }}
        
        .loading-overlay.hidden {{
            display: none;
        }}
        
        .spinner {{
            width: 40px;
            height: 40px;
            border: 4px solid #f0f0f0;
            border-top-color: #4169E1;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <!-- Map container -->
    <div id="deckgl-container"></div>
    
    <!-- Control Panel -->
    <div class="control-panel">
        <h3>🗺️ Zone Coverage</h3>
        
        <!-- Statistics -->
        <div class="control-section">
            <h4>Statistics</h4>
            <div class="stat-row">
                <span class="stat-label">Zones:</span>
                <span class="stat-value" id="zone-count">-</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Boreholes:</span>
                <span class="stat-value" id="borehole-count">-</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Modified:</span>
                <span class="stat-value"><span id="modified-count">0</span></span>
            </div>
        </div>
        
        <!-- Layer Toggles -->
        <div class="control-section">
            <h4>Layers</h4>
            <div class="layer-toggle-row">
                <input type="checkbox" id="toggle-zones" class="layer-toggle" data-layer="zones" checked>
                <label for="toggle-zones">Zone Boundaries</label>
            </div>
            <div class="layer-toggle-row">
                <input type="checkbox" id="toggle-coverage" class="layer-toggle" data-layer="coverage" checked>
                <label for="toggle-coverage">Coverage Areas</label>
            </div>
            <div class="layer-toggle-row">
                <input type="checkbox" id="toggle-boreholes" class="layer-toggle" data-layer="boreholes" checked>
                <label for="toggle-boreholes">Boreholes</label>
            </div>
        </div>
        
        <!-- Actions -->
        <div class="control-section">
            <h4>Actions</h4>
            <button id="fit-btn" class="btn btn-secondary">🔍 Zoom to Fit</button>
            <button id="recalc-btn" class="btn btn-secondary">🔄 Recalculate</button>
            <button id="export-btn" class="btn btn-primary">📥 Export CSV</button>
            <button id="reset-btn" class="btn btn-danger">↩️ Reset All</button>
        </div>
        
        <!-- Instructions -->
        <div class="instructions">
            <p><strong>Drag</strong> boreholes to reposition</p>
            <p><span class="kbd">R</span> Reset positions</p>
            <p><span class="kbd">F</span> Zoom to fit</p>
            <p><span class="kbd">Esc</span> Cancel drag</p>
        </div>
    </div>
    
    <!-- Tooltip -->
    <div id="tooltip"></div>
    
    <!-- Loading overlay -->
    <div class="loading-overlay" id="loading">
        <div class="spinner"></div>
    </div>
    
    {turf_cdn}
    {turf_script}
    
    <!-- Embedded Data -->
    <script>
        // Zone GeoJSON
        const ZONE_DATA = {zones_json};
        
        // Borehole GeoJSON
        const BOREHOLE_DATA = {boreholes_json};
        
        // Borehole positions [lon, lat]
        const POSITION_DATA = {positions_json};
        
        // Pre-computed coverage (borehole_id -> geometry)
        const COVERAGE_DATA = {coverage_json};
        
        // Visualization config
        const VIZ_CONFIG = {viz_config_json};
    </script>
    
    <!-- Worker Code (as string for Blob) -->
    <script>
        const WORKER_CODE = `{_escape_js_string(worker_code)}`;
    </script>
    
    <!-- Main Application -->
    <script>
{main_app_code}
    </script>
    
    <!-- UI Controls -->
    <script>
{ui_code}
    </script>
    
    <!-- Hide loading overlay once ready -->
    <script>
        window.addEventListener('load', function() {{
            setTimeout(function() {{
                document.getElementById('loading').classList.add('hidden');
            }}, 500);
        }});
    </script>
</body>
</html>"""

    return html


def _escape_js_string(code: str) -> str:
    """Escape JavaScript code for embedding in template literal.

    Args:
        code: JavaScript code to escape

    Returns:
        Escaped code safe for template literal
    """
    # Escape backslashes first
    code = code.replace("\\", "\\\\")
    # Escape backticks
    code = code.replace("`", "\\`")
    # Escape template literal interpolation
    code = code.replace("${", "\\${")
    return code


# ═══════════════════════════════════════════════════════════════════════════════
# 📁 FILE LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_js_template(template_name: str) -> str:
    """Load a JavaScript template file.

    Args:
        template_name: Name of template file (without path)

    Returns:
        JavaScript code as string
    """
    template_dir = Path(__file__).parent / "js_templates"
    template_path = template_dir / template_name

    if not template_path.exists():
        raise FileNotFoundError(f"JS template not found: {template_path}")

    return template_path.read_text(encoding="utf-8")


def load_all_js_templates() -> Dict[str, str]:
    """Load all JavaScript templates.

    Returns:
        Dict mapping template name to code
    """
    templates = {}
    template_dir = Path(__file__).parent / "js_templates"

    for js_file in template_dir.glob("*.js"):
        templates[js_file.stem] = js_file.read_text(encoding="utf-8")

    return templates
