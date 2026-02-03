/**
 * Zone-Aware Coverage Visualization
 * 
 * Main deck.gl application for interactive borehole coverage visualization.
 * Features:
 * - Draggable coverage circles (filled areas)
 * - LIVE smooth radius changes when crossing zone boundaries
 * - Zone-aware radius lookup using Turf.js point-in-polygon
 * - Export modified positions
 */

(function() {
    'use strict';
    
    // =========================================================================
    // APPLICATION STATE
    // =========================================================================
    
    const state = {
        deck: null,
        map: null,
        
        // Data
        zones: null,
        boreholes: null,
        positions: [],  // Mutable positions array [lon, lat]
        originalPositions: [],  // Original positions for reset
        radii: [],  // Current radius per borehole (in METERS)
        
        // Drag state
        dragging: false,
        dragIndex: null,
        dragOffset: null,  // Offset from click point to center
        
        // Config
        config: {
            enableDrag: true,
            defaultMaxSpacing: 100,
            zoneStyle: {
                fillColor: [200, 200, 200, 50],
                lineColor: [100, 100, 100, 255],
                lineWidth: 2
            },
            coverageStyle: {
                fillColor: [100, 149, 237, 120],
                lineColor: [70, 130, 180, 255],
                lineWidth: 2
            }
        }
    };
    
    // =========================================================================
    // INITIALIZATION
    // =========================================================================
    
    /**
     * Initialize the application
     */
    function init() {
        console.log('🚀 Initializing Zone Coverage Visualization');
        
        // Load embedded data
        loadEmbeddedData();
        
        // Compute initial radii based on zone positions
        computeAllRadii();
        
        // Initialize deck.gl
        initDeck();
        
        // Setup UI handlers
        setupUI();
        
        console.log('✅ Initialization complete');
    }
    
    /**
     * Load data embedded in the page
     */
    function loadEmbeddedData() {
        // These are embedded by the Python generator
        if (typeof ZONE_DATA !== 'undefined') {
            state.zones = ZONE_DATA;
            console.log(`📍 Loaded ${state.zones.features.length} zones`);
        }
        
        if (typeof BOREHOLE_DATA !== 'undefined') {
            state.boreholes = BOREHOLE_DATA;
            console.log(`📍 Loaded ${state.boreholes.features.length} boreholes`);
        }
        
        if (typeof POSITION_DATA !== 'undefined') {
            state.positions = JSON.parse(JSON.stringify(POSITION_DATA));
            state.originalPositions = JSON.parse(JSON.stringify(POSITION_DATA));
            // Initialize radii array (will be computed in computeAllRadii)
            state.radii = new Array(state.positions.length).fill(state.config.defaultMaxSpacing);
            console.log(`📍 Loaded ${state.positions.length} positions`);
        }
        
        if (typeof VIZ_CONFIG !== 'undefined') {
            Object.assign(state.config, VIZ_CONFIG);
        }
    }
    
    /**
     * Compute radii for all boreholes based on their current zone
     */
    function computeAllRadii() {
        if (!state.zones || !state.positions.length) return;
        
        for (let i = 0; i < state.positions.length; i++) {
            state.radii[i] = getRadiusForPosition(state.positions[i]);
        }
        console.log('📐 Computed initial radii for all boreholes');
    }
    
    /**
     * Get the radius (max_spacing) for a position based on which zone it's in
     * Uses Turf.js point-in-polygon for zone detection
     */
    function getRadiusForPosition(position) {
        if (!state.zones || !state.zones.features || typeof turf === 'undefined') {
            return state.config.defaultMaxSpacing;
        }
        
        const [lon, lat] = position;
        const point = turf.point([lon, lat]);
        
        // Find which zone contains this point
        for (const zone of state.zones.features) {
            try {
                if (turf.booleanPointInPolygon(point, zone)) {
                    const maxSpacing = zone.properties.max_spacing_m || state.config.defaultMaxSpacing;
                    return maxSpacing;
                }
            } catch (e) {
                // Skip invalid geometries
            }
        }
        
        // Not in any zone - use default
        return state.config.defaultMaxSpacing;
    }
    
    /**
     * Get borehole ID by position index
     */
    function getBhIdByIndex(index) {
        if (!state.boreholes || !state.boreholes.features) return null;
        const feature = state.boreholes.features[index];
        return feature ? feature.id || feature.properties?.id : null;
    }
    
    // =========================================================================
    // DECK.GL INITIALIZATION
    // =========================================================================
    
    /**
     * Initialize deck.gl with MapLibre basemap
     */
    function initDeck() {
        const { GeoJsonLayer, ScatterplotLayer, PolygonLayer } = deck;
        
        // Get initial view from embedded config
        const initialView = {
            longitude: VIZ_CONFIG?.center?.[0] || -1.45,
            latitude: VIZ_CONFIG?.center?.[1] || 51.45,
            zoom: VIZ_CONFIG?.zoom || 13,
            pitch: 0,
            bearing: 0
        };
        
        state.deck = new deck.DeckGL({
            container: 'deckgl-container',
            mapStyle: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
            initialViewState: initialView,
            controller: true,
            layers: buildLayers(),
            
            // Drag handling
            onDragStart: handleDragStart,
            onDrag: handleDrag,
            onDragEnd: handleDragEnd,
            
            // Cursor
            getCursor: ({isDragging}) => isDragging ? 'grabbing' : 'grab'
        });
    }
    
    // =========================================================================
    // LAYER BUILDING - Using ScatterplotLayer for smooth radius transitions
    // =========================================================================
    
    /**
     * Build all deck.gl layers
     */
    function buildLayers() {
        const layers = [];
        
        // Zone boundaries (bottom)
        if (state.zones) {
            layers.push(buildZoneLayer());
        }
        
        // Coverage circles (draggable) - using ScatterplotLayer with METERS
        layers.push(buildCoverageLayer());
        
        // Small center dots for visual reference
        layers.push(buildCenterDotsLayer());
        
        return layers;
    }
    
    /**
     * Build zone boundary layer
     */
    function buildZoneLayer() {
        return new deck.GeoJsonLayer({
            id: 'zones',
            data: state.zones,
            stroked: true,
            filled: true,
            pickable: false,
            getFillColor: state.config.zoneStyle.fillColor,
            getLineColor: state.config.zoneStyle.lineColor,
            getLineWidth: state.config.zoneStyle.lineWidth,
            lineWidthUnits: 'pixels'
        });
    }
    
    /**
     * Build coverage circles as ScatterplotLayer with METER-based radius
     * This allows smooth, live radius changes when crossing zones
     */
    function buildCoverageLayer() {
        // Build data array with current positions and radii
        const data = state.positions.map((pos, i) => ({
            position: pos,
            radius: state.radii[i],
            index: i,
            id: getBhIdByIndex(i)
        }));
        
        return new deck.ScatterplotLayer({
            id: 'coverage',
            data: data,
            pickable: true,  // DRAGGABLE
            radiusUnits: 'meters',  // KEY: Use meters for geographic radius
            radiusScale: 1,
            getPosition: d => d.position,
            getRadius: d => d.radius,  // Dynamic radius per circle
            getFillColor: state.config.coverageStyle.fillColor,
            getLineColor: state.config.coverageStyle.lineColor,
            getLineWidth: state.config.coverageStyle.lineWidth,
            lineWidthUnits: 'pixels',
            stroked: true,
            // Smooth transitions
            transitions: {
                getRadius: {
                    duration: 150,  // 150ms smooth transition
                    easing: t => t * (2 - t)  // Ease out
                }
            },
            // Highlight on hover
            autoHighlight: true,
            highlightColor: [100, 149, 237, 180]
        });
    }
    
    /**
     * Build small center dots for visual reference (not pickable)
     */
    function buildCenterDotsLayer() {
        return new deck.ScatterplotLayer({
            id: 'center-dots',
            data: state.positions.map((pos, i) => ({
                position: pos,
                index: i
            })),
            pickable: false,  // NOT pickable
            radiusUnits: 'pixels',
            getPosition: d => d.position,
            getRadius: 4,
            getFillColor: [0, 0, 0, 200],
            stroked: false
        });
    }
    
    /**
     * Update all layers (called after position changes)
     */
    function updateLayers() {
        if (state.deck) {
            state.deck.setProps({ layers: buildLayers() });
        }
    }
    
    // =========================================================================
    // DRAG HANDLING - Coverage circles are draggable
    // =========================================================================
    
    /**
     * Handle drag start on coverage circle
     */
    function handleDragStart(info, event) {
        if (!state.config.enableDrag) return;
        
        // Check if we're starting on a coverage circle
        if (info.layer && info.layer.id === 'coverage' && info.object) {
            const index = info.object.index;
            if (index !== undefined && index >= 0) {
                state.dragging = true;
                state.dragIndex = index;
                
                // Store the drag offset (distance from click to center)
                const centerPos = state.positions[index];
                state.dragOffset = [
                    info.coordinate[0] - centerPos[0],
                    info.coordinate[1] - centerPos[1]
                ];
                
                // Disable map pan
                state.deck.setProps({
                    controller: { dragPan: false }
                });
                
                return true;
            }
        }
        return false;
    }
    
    /**
     * Handle drag movement - LIVE radius updates
     */
    function handleDrag(info, event) {
        if (!state.dragging || state.dragIndex === null) return;
        
        // Calculate new center position accounting for drag offset
        const newLng = info.coordinate[0] - (state.dragOffset?.[0] || 0);
        const newLat = info.coordinate[1] - (state.dragOffset?.[1] || 0);
        
        // Update position
        state.positions[state.dragIndex] = [newLng, newLat];
        
        // LIVE: Update radius based on current zone - this is INSTANT
        state.radii[state.dragIndex] = getRadiusForPosition([newLng, newLat]);
        
        // Update layers immediately for smooth visual feedback
        updateLayers();
    }
    
    /**
     * Handle drag end
     */
    function handleDragEnd(info, event) {
        if (!state.dragging) return;
        
        // Final position/radius update
        if (state.dragIndex !== null) {
            const pos = state.positions[state.dragIndex];
            state.radii[state.dragIndex] = getRadiusForPosition(pos);
        }
        
        state.dragging = false;
        state.dragIndex = null;
        state.dragOffset = null;
        
        // Re-enable map pan
        state.deck.setProps({
            controller: true
        });
        
        // Update modified count
        updateModifiedCount();
        
        // Final layer update
        updateLayers();
    }
    
    // =========================================================================
    // UI HANDLERS
    // =========================================================================
    
    /**
     * Setup UI event handlers
     */
    function setupUI() {
        // Reset button
        const resetBtn = document.getElementById('reset-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', resetPositions);
        }
        
        // Export button
        const exportBtn = document.getElementById('export-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', exportPositions);
        }
        
        // Recalculate button
        const recalcBtn = document.getElementById('recalc-btn');
        if (recalcBtn) {
            recalcBtn.addEventListener('click', recalculateAllRadii);
        }
    }
    
    /**
     * Reset all positions to original
     */
    function resetPositions() {
        state.positions = JSON.parse(JSON.stringify(state.originalPositions));
        computeAllRadii();  // Recalculate radii for original positions
        updateModifiedCount();
        updateLayers();
        console.log('🔄 Positions reset to original');
    }
    
    /**
     * Recalculate radii for all boreholes based on current zones
     */
    function recalculateAllRadii() {
        computeAllRadii();
        updateLayers();
        console.log('📐 Recalculated all radii');
    }
    
    /**
     * Export modified positions to CSV
     */
    function exportPositions() {
        const lines = ['borehole_id,original_lon,original_lat,modified_lon,modified_lat,moved'];
        
        for (let i = 0; i < state.positions.length; i++) {
            const bhId = getBhIdByIndex(i) || `BH_${i}`;
            const orig = state.originalPositions[i];
            const curr = state.positions[i];
            const moved = (orig[0] !== curr[0] || orig[1] !== curr[1]) ? 'yes' : 'no';
            
            lines.push(`${bhId},${orig[0]},${orig[1]},${curr[0]},${curr[1]},${moved}`);
        }
        
        const csv = lines.join('\n');
        downloadFile('modified_positions.csv', csv, 'text/csv');
        
        console.log('📥 Exported positions to CSV');
    }
    
    /**
     * Download a file
     */
    function downloadFile(filename, content, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    /**
     * Update the modified count display
     */
    function updateModifiedCount() {
        let count = 0;
        for (let i = 0; i < state.positions.length; i++) {
            const orig = state.originalPositions[i];
            const curr = state.positions[i];
            if (orig[0] !== curr[0] || orig[1] !== curr[1]) {
                count++;
            }
        }
        
        const countEl = document.getElementById('modified-count');
        if (countEl) {
            countEl.textContent = count;
        }
    }
    
    // =========================================================================
    // ENTRY POINT
    // =========================================================================
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
    // Expose for debugging
    window.ZoneCoverage = {
        state,
        recalculateAllRadii,
        resetPositions,
        exportPositions,
        getRadiusForPosition
    };
    
})();
