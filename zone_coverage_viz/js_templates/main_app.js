/**
 * Zone-Aware Coverage Visualization
 * 
 * Main deck.gl application for interactive borehole coverage visualization.
 * Features:
 * - Draggable borehole markers
 * - Real-time zone-aware coverage updates
 * - Multi-zone "flower petal" coverage shapes
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
        worker: null,
        workerReady: false,
        
        // Data
        zones: null,
        boreholes: null,
        positions: [],  // Mutable positions array [lon, lat]
        originalPositions: [],  // Original positions for reset
        coverage: {},  // borehole_id -> geometry
        
        // Drag state
        dragging: false,
        dragIndex: null,
        
        // Config
        config: {
            enableDrag: true,
            defaultMaxSpacing: 100,
            zoneStyle: {
                fillColor: [200, 200, 200, 50],
                lineColor: [100, 100, 100, 255],
                lineWidth: 2
            },
            boreholeStyle: {
                radiusPixels: 8,
                fillColor: [65, 105, 225, 255],
                lineColor: [255, 255, 255, 255],
                lineWidth: 2
            },
            coverageStyle: {
                fillColor: [100, 149, 237, 120],
                lineColor: [70, 130, 180, 200],
                lineWidth: 1
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
        
        // Initialize Web Worker
        initWorker();
        
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
            console.log(`📍 Loaded ${state.positions.length} positions`);
        }
        
        if (typeof COVERAGE_DATA !== 'undefined') {
            state.coverage = COVERAGE_DATA;
            console.log(`📐 Loaded ${Object.keys(state.coverage).length} coverage geometries`);
        }
        
        if (typeof VIZ_CONFIG !== 'undefined') {
            Object.assign(state.config, VIZ_CONFIG);
        }
    }
    
    /**
     * Initialize the Web Worker for geometry computation
     */
    function initWorker() {
        if (typeof WORKER_CODE === 'undefined') {
            console.warn('⚠️ Worker code not embedded, coverage updates will be disabled');
            return;
        }
        
        try {
            // Create worker from embedded code
            const blob = new Blob([WORKER_CODE], { type: 'application/javascript' });
            const workerUrl = URL.createObjectURL(blob);
            state.worker = new Worker(workerUrl);
            
            state.worker.onmessage = handleWorkerMessage;
            state.worker.onerror = (e) => {
                console.error('Worker error:', e);
            };
            
            // Initialize worker with Turf.js and zones
            if (typeof TURF_CODE !== 'undefined') {
                state.worker.postMessage({
                    type: 'INIT',
                    id: 'init',
                    payload: {
                        turfCode: TURF_CODE,
                        zones: state.zones
                    }
                });
            }
            
            console.log('🔧 Web Worker initialized');
        } catch (e) {
            console.error('Failed to initialize worker:', e);
        }
    }
    
    /**
     * Handle messages from the Web Worker
     */
    function handleWorkerMessage(e) {
        const { type, id, payload } = e.data;
        
        switch (type) {
            case 'INIT_COMPLETE':
                state.workerReady = true;
                console.log('✅ Worker ready');
                break;
                
            case 'COVERAGE_RESULT':
                // Update coverage for single borehole
                if (payload && state.dragIndex !== null) {
                    const bhId = getBhIdByIndex(state.dragIndex);
                    if (bhId) {
                        state.coverage[bhId] = payload;
                        updateLayers();
                    }
                }
                break;
                
            case 'ALL_COVERAGE_RESULT':
                state.coverage = payload;
                updateLayers();
                break;
                
            case 'ERROR':
                console.error('Worker error:', payload.message);
                break;
        }
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
    // LAYER BUILDING
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
        
        // Coverage polygons (middle)
        layers.push(buildCoverageLayer());
        
        // Borehole markers (top)
        layers.push(buildBoreholeLayer());
        
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
     * Build coverage polygon layer from coverage geometries
     */
    function buildCoverageLayer() {
        // Convert coverage dict to GeoJSON FeatureCollection
        const features = Object.entries(state.coverage).map(([bhId, geometry]) => ({
            type: 'Feature',
            id: bhId,
            properties: { borehole_id: bhId },
            geometry: geometry
        }));
        
        const data = {
            type: 'FeatureCollection',
            features: features
        };
        
        return new deck.GeoJsonLayer({
            id: 'coverage',
            data: data,
            stroked: true,
            filled: true,
            pickable: false,
            getFillColor: state.config.coverageStyle.fillColor,
            getLineColor: state.config.coverageStyle.lineColor,
            getLineWidth: state.config.coverageStyle.lineWidth,
            lineWidthUnits: 'pixels'
        });
    }
    
    /**
     * Build borehole scatter layer
     */
    function buildBoreholeLayer() {
        return new deck.ScatterplotLayer({
            id: 'boreholes',
            data: state.positions.map((pos, i) => ({
                position: pos,
                index: i,
                id: getBhIdByIndex(i)
            })),
            pickable: true,
            radiusUnits: 'pixels',
            getPosition: d => d.position,
            getRadius: state.config.boreholeStyle.radiusPixels,
            getFillColor: state.config.boreholeStyle.fillColor,
            getLineColor: state.config.boreholeStyle.lineColor,
            getLineWidth: state.config.boreholeStyle.lineWidth,
            lineWidthUnits: 'pixels',
            stroked: true
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
    // DRAG HANDLING
    // =========================================================================
    
    /**
     * Handle drag start on borehole
     */
    function handleDragStart(info, event) {
        if (!state.config.enableDrag) return;
        
        // Check if we're starting on a borehole
        if (info.layer && info.layer.id === 'boreholes' && info.object) {
            state.dragging = true;
            state.dragIndex = info.object.index;
            
            // Disable map pan
            state.deck.setProps({
                controller: { dragPan: false }
            });
            
            return true;
        }
        return false;
    }
    
    /**
     * Handle drag movement
     */
    function handleDrag(info, event) {
        if (!state.dragging || state.dragIndex === null) return;
        
        // Update position
        const [lng, lat] = info.coordinate;
        state.positions[state.dragIndex] = [lng, lat];
        
        // Request coverage update (throttled)
        requestCoverageUpdate(state.dragIndex, [lng, lat]);
        
        // Update layers immediately for smooth visual feedback
        updateLayers();
    }
    
    /**
     * Handle drag end
     */
    function handleDragEnd(info, event) {
        if (!state.dragging) return;
        
        // Final coverage update
        if (state.dragIndex !== null) {
            const pos = state.positions[state.dragIndex];
            requestCoverageUpdate(state.dragIndex, pos, true);
        }
        
        state.dragging = false;
        state.dragIndex = null;
        
        // Re-enable map pan
        state.deck.setProps({
            controller: true
        });
        
        // Update modified count
        updateModifiedCount();
    }
    
    // Throttle for coverage updates during drag
    let coverageUpdateTimeout = null;
    
    /**
     * Request coverage update from worker (throttled during drag)
     */
    function requestCoverageUpdate(index, position, immediate = false) {
        if (!state.worker || !state.workerReady) return;
        
        const doUpdate = () => {
            state.worker.postMessage({
                type: 'COMPUTE_COVERAGE',
                id: `coverage_${index}`,
                payload: {
                    position: position,
                    zones: state.zones,
                    defaultMaxSpacing: state.config.defaultMaxSpacing
                }
            });
        };
        
        if (immediate) {
            if (coverageUpdateTimeout) {
                clearTimeout(coverageUpdateTimeout);
                coverageUpdateTimeout = null;
            }
            doUpdate();
        } else {
            // Throttle to 60fps (16ms)
            if (!coverageUpdateTimeout) {
                coverageUpdateTimeout = setTimeout(() => {
                    coverageUpdateTimeout = null;
                    doUpdate();
                }, 16);
            }
        }
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
            recalcBtn.addEventListener('click', recalculateAllCoverage);
        }
    }
    
    /**
     * Reset all positions to original
     */
    function resetPositions() {
        state.positions = JSON.parse(JSON.stringify(state.originalPositions));
        recalculateAllCoverage();
        updateModifiedCount();
        console.log('🔄 Positions reset to original');
    }
    
    /**
     * Recalculate coverage for all boreholes
     */
    function recalculateAllCoverage() {
        if (!state.worker || !state.workerReady) {
            console.warn('Worker not ready');
            return;
        }
        
        // Build borehole list with current positions
        const boreholes = state.positions.map((pos, i) => ({
            id: getBhIdByIndex(i),
            position: pos
        }));
        
        state.worker.postMessage({
            type: 'COMPUTE_ALL',
            id: 'compute_all',
            payload: {
                boreholes: boreholes,
                zones: state.zones,
                defaultMaxSpacing: state.config.defaultMaxSpacing
            }
        });
        
        console.log('📐 Recalculating all coverage...');
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
        recalculateAllCoverage,
        resetPositions,
        exportPositions
    };
    
})();
