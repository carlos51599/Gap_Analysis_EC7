/**
 * Zone-Aware Coverage Visualization - HYBRID APPROACH
 * 
 * Main deck.gl application for interactive borehole coverage visualization.
 * 
 * DUAL-LAYER ARCHITECTURE:
 * 1. ScatterplotLayer: Shows PREVIEW circles during drag (fast, responsive)
 * 2. GeoJsonLayer: Shows ACCURATE clipped coverage after drag-end (computed by Worker)
 * 
 * Features:
 * - Draggable borehole markers
 * - PREVIEW: Simple circles that change radius by zone (during drag)
 * - ACCURATE: Clipped polygons computed by Web Worker (on drag-end)
 * - Zone-aware coverage using Turf.js buffer + intersection
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
        radii: [],  // Current radius per borehole (in METERS) - for preview
        
        // Accurate coverage geometries (computed on main thread)
        accurateCoverage: [],  // Array of GeoJSON features per borehole
        showAccurateCoverage: true,  // Toggle between preview and accurate
        
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
            },
            // Accurate coverage uses slightly different style
            accurateCoverageStyle: {
                fillColor: [34, 139, 34, 100],  // Forest green
                lineColor: [34, 139, 34, 200],
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
        console.log('🚀 Initializing Zone Coverage Visualization (Hybrid Mode)');
        
        // Load embedded data
        loadEmbeddedData();
        
        // Compute initial radii based on zone positions (for preview)
        computeAllRadii();
        
        // Initialize deck.gl
        initDeck();
        
        // Setup UI handlers
        setupUI();
        
        // Compute accurate coverage on main thread (after UI is ready)
        setTimeout(() => {
            computeAllAccurateCoverage();
        }, 100);
        
        console.log('✅ Initialization complete');
    }
    
    // =========================================================================
    // ACCURATE COVERAGE COMPUTATION (Main Thread)
    // =========================================================================
    
    /**
     * Compute zone-aware coverage for all boreholes
     * Uses Turf.js on the main thread
     */
    function computeAllAccurateCoverage() {
        if (typeof turf === 'undefined') {
            console.warn('⚠️ Cannot compute coverage - Turf.js not loaded');
            return;
        }
        if (!state.zones) {
            console.warn('⚠️ Cannot compute coverage - zones not loaded');
            return;
        }
        if (!state.zones.features || state.zones.features.length === 0) {
            console.warn('⚠️ Cannot compute coverage - zones.features empty');
            return;
        }
        if (!state.positions.length) {
            console.warn('⚠️ Cannot compute coverage - positions empty');
            return;
        }
        
        console.log('📐 Computing accurate zone-clipped coverage...');
        console.log(`   - ${state.zones.features.length} zones available`);
        console.log(`   - ${state.positions.length} boreholes to process`);
        console.log(`   - First zone max_spacing_m: ${state.zones.features[0].properties?.max_spacing_m}`);
        console.log(`   - First borehole position: ${state.positions[0]}`);
        const startTime = performance.now();
        
        const coverages = [];
        for (let i = 0; i < state.positions.length; i++) {
            const coverage = computeSingleAccurateCoverage(i, state.positions[i]);
            if (coverage) {
                coverages.push(coverage);
            }
        }
        
        state.accurateCoverage = coverages;
        const elapsed = performance.now() - startTime;
        console.log(`✅ Computed ${coverages.length} accurate coverages in ${elapsed.toFixed(0)}ms`);
        
        updateLayers();
    }
    
    /**
     * Compute zone-aware coverage for a single borehole position
     * 
     * Algorithm:
     * 1. For each zone within range of the borehole
     * 2. Buffer the borehole point by that zone's max_spacing_m
     * 3. Intersect the buffer with the zone polygon
     * 4. Union all intersected pieces
     * 5. Return the final coverage geometry
     */
    function computeSingleAccurateCoverage(boreholeIndex, position) {
        if (!position) return null;
        
        const isFirstBorehole = boreholeIndex === 0;
        
        try {
            const [lon, lat] = position;
            const point = turf.point([lon, lat]);
            const fragments = [];
            
            if (isFirstBorehole) {
                console.log(`🔍 Debug first borehole at [${lon}, ${lat}]`);
            }
            
            let zonesChecked = 0;
            let zonesSkipped = 0;
            
            for (const zone of state.zones.features) {
                const maxSpacing = zone.properties.max_spacing_m || state.config.defaultMaxSpacing;
                
                // Quick bbox check - skip zones too far away
                const bbox = turf.bbox(zone);
                const margin = degreesFromMeters(maxSpacing, lat);
                
                if (lon < bbox[0] - margin || lon > bbox[2] + margin ||
                    lat < bbox[1] - margin || lat > bbox[3] + margin) {
                    zonesSkipped++;
                    continue;
                }
                
                zonesChecked++;
                
                // Buffer the point by this zone's spacing (Turf uses km)
                const buffer = turf.buffer(point, maxSpacing / 1000, { units: 'kilometers' });
                if (!buffer) {
                    if (isFirstBorehole) console.log(`   Buffer failed for zone ${zone.properties?.zone_name}`);
                    continue;
                }
                
                // Intersect with zone polygon (Turf.js v7 API: single FeatureCollection)
                try {
                    if (isFirstBorehole && zonesChecked === 1) {
                        console.log(`   Buffer type: ${buffer.type}, geometry.type: ${buffer.geometry?.type}`);
                        console.log(`   Zone type: ${zone.type}, geometry.type: ${zone.geometry?.type}`);
                        console.log(`   Buffer coords length: ${buffer.geometry?.coordinates?.[0]?.length}`);
                        console.log(`   Zone coords length: ${zone.geometry?.coordinates?.[0]?.length}`);
                        console.log(`   First zone coord: ${JSON.stringify(zone.geometry?.coordinates?.[0]?.[0])}`);
                        
                        // Try the operation step by step
                        const fc = turf.featureCollection([buffer, zone]);
                        console.log(`   FeatureCollection features: ${fc.features?.length}`);
                        console.log(`   FC feature 0 type: ${fc.features?.[0]?.geometry?.type}`);
                        console.log(`   FC feature 1 type: ${fc.features?.[1]?.geometry?.type}`);
                    }
                    
                    const intersection = turf.intersect(
                        turf.featureCollection([buffer, zone])
                    );
                    
                    if (isFirstBorehole) {
                        console.log(`   Zone: ${zone.properties?.zone_name}, maxSpacing: ${maxSpacing}m, intersection: ${intersection ? 'YES' : 'NO'}`);
                    }
                    
                    if (intersection && !isEmptyGeometry(intersection)) {
                        fragments.push(intersection);
                    }
                } catch (e) {
                    // Skip invalid intersections
                    if (isFirstBorehole) console.log(`   Intersection error: ${e.message}`);
                    console.debug(`Intersection failed for zone:`, e.message);
                }
            }
            
            if (isFirstBorehole) {
                console.log(`   Zones checked: ${zonesChecked}, skipped (bbox): ${zonesSkipped}, fragments: ${fragments.length}`);
            }
            
            // Union all fragments (Turf.js v7 API: single FeatureCollection)
            let coverage = null;
            if (fragments.length === 1) {
                coverage = fragments[0];
            } else if (fragments.length > 1) {
                try {
                    // Turf.js v7: union takes a single FeatureCollection
                    coverage = turf.union(turf.featureCollection(fragments));
                } catch (e) {
                    console.debug(`Union failed:`, e.message);
                    coverage = fragments[0];
                }
            }
            
            // Add borehole index as property
            if (coverage) {
                coverage.properties = coverage.properties || {};
                coverage.properties.boreholeIndex = boreholeIndex;
            }
            
            return coverage;
            
        } catch (e) {
            console.warn(`Failed to compute coverage for borehole ${boreholeIndex}:`, e);
            return null;
        }
    }
    
    /**
     * Convert meters to approximate degrees at a given latitude
     */
    function degreesFromMeters(meters, latitude) {
        const metersPerDegree = 111320 * Math.cos(latitude * Math.PI / 180);
        return meters / metersPerDegree;
    }
    
    /**
     * Check if a geometry is empty
     */
    function isEmptyGeometry(geom) {
        if (!geom) return true;
        if (geom.type === 'GeometryCollection' && (!geom.geometries || geom.geometries.length === 0)) {
            return true;
        }
        if (geom.geometry) {
            return isEmptyGeometry(geom.geometry);
        }
        return false;
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
     * 
     * Layer order (bottom to top):
     * 1. Zone boundaries
     * 2. Accurate coverage (GeoJsonLayer) - when not dragging
     * 3. Preview circles (ScatterplotLayer) - during drag
     * 4. Center dots (for reference)
     */
    function buildLayers() {
        const layers = [];
        
        // Zone boundaries (bottom)
        if (state.zones) {
            layers.push(buildZoneLayer());
        }
        
        // Accurate coverage from Web Worker (when available and not dragging)
        if (state.showAccurateCoverage && state.accurateCoverage.length > 0 && !state.dragging) {
            layers.push(buildAccurateCoverageLayer());
        }
        
        // Preview circles (during drag or when no accurate coverage available)
        if (state.dragging || state.accurateCoverage.length === 0) {
            layers.push(buildPreviewCoverageLayer());
        }
        
        // Small center dots for visual reference and picking
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
     * Build ACCURATE coverage layer using Web Worker computed geometries
     * These are proper zone-clipped polygons
     */
    function buildAccurateCoverageLayer() {
        // Build a FeatureCollection from all coverage geometries
        const features = state.accurateCoverage.filter(f => f != null);
        
        const data = {
            type: 'FeatureCollection',
            features: features
        };
        
        return new deck.GeoJsonLayer({
            id: 'accurate-coverage',
            data: data,
            stroked: true,
            filled: true,
            pickable: false,  // Not pickable - we pick on center dots
            getFillColor: state.config.accurateCoverageStyle.fillColor,
            getLineColor: state.config.accurateCoverageStyle.lineColor,
            getLineWidth: state.config.accurateCoverageStyle.lineWidth,
            lineWidthUnits: 'pixels'
        });
    }
    
    /**
     * Build PREVIEW coverage circles as ScatterplotLayer
     * Shows during drag for responsive feedback
     */
    function buildPreviewCoverageLayer() {
        // Build data array with current positions and radii
        const data = state.positions.map((pos, i) => ({
            position: pos,
            radius: state.radii[i],
            index: i,
            id: getBhIdByIndex(i)
        }));
        
        return new deck.ScatterplotLayer({
            id: 'preview-coverage',
            data: data,
            pickable: false,  // Not pickable - we pick on center dots
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
            }
        });
    }
    
    /**
     * Build small center dots - THESE are pickable for dragging
     */
    function buildCenterDotsLayer() {
        return new deck.ScatterplotLayer({
            id: 'center-dots',
            data: state.positions.map((pos, i) => ({
                position: pos,
                index: i
            })),
            pickable: true,  // PICKABLE for drag
            radiusUnits: 'pixels',
            getPosition: d => d.position,
            getRadius: 8,  // Larger for easier picking
            getFillColor: [0, 0, 0, 200],
            stroked: true,
            getLineColor: [255, 255, 255, 255],
            getLineWidth: 2,
            lineWidthUnits: 'pixels',
            // Highlight on hover
            autoHighlight: true,
            highlightColor: [255, 100, 100, 255]
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
    // DRAG HANDLING - Center dots are draggable, coverage updates accordingly
    // =========================================================================
    
    /**
     * Handle drag start on center dot
     */
    function handleDragStart(info, event) {
        if (!state.config.enableDrag) return;
        
        // Check if we're starting on a center dot
        if (info.layer && info.layer.id === 'center-dots' && info.object) {
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
                
                // Show preview layer during drag
                updateLayers();
                
                return true;
            }
        }
        return false;
    }
    
    /**
     * Handle drag movement - LIVE preview radius updates
     */
    function handleDrag(info, event) {
        if (!state.dragging || state.dragIndex === null) return;
        
        // Calculate new center position accounting for drag offset
        const newLng = info.coordinate[0] - (state.dragOffset?.[0] || 0);
        const newLat = info.coordinate[1] - (state.dragOffset?.[1] || 0);
        
        // Update position
        state.positions[state.dragIndex] = [newLng, newLat];
        
        // LIVE: Update PREVIEW radius based on current zone
        state.radii[state.dragIndex] = getRadiusForPosition([newLng, newLat]);
        
        // Update layers immediately for smooth visual feedback
        updateLayers();
    }
    
    /**
     * Handle drag end - Compute ACCURATE coverage on main thread
     */
    function handleDragEnd(info, event) {
        if (!state.dragging) return;
        
        const draggedIndex = state.dragIndex;
        
        // Final position/radius update for preview
        if (draggedIndex !== null) {
            const pos = state.positions[draggedIndex];
            state.radii[draggedIndex] = getRadiusForPosition(pos);
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
        
        // Compute ACCURATE coverage for this borehole on main thread
        if (draggedIndex !== null) {
            const coverage = computeSingleAccurateCoverage(draggedIndex, state.positions[draggedIndex]);
            if (coverage) {
                // Find and replace this borehole's coverage
                const existingIdx = state.accurateCoverage.findIndex(
                    c => c?.properties?.boreholeIndex === draggedIndex
                );
                if (existingIdx >= 0) {
                    state.accurateCoverage[existingIdx] = coverage;
                } else {
                    state.accurateCoverage.push(coverage);
                }
            }
        }
        
        // Final layer update (shows accurate coverage)
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
        
        // Request accurate coverage for all reset positions
        requestAllCoverage();
        
        console.log('🔄 Positions reset to original');
    }
    
    /**
     * Recalculate radii for all boreholes based on current zones
     */
    function recalculateAllRadii() {
        computeAllRadii();
        updateLayers();
        
        // Also request accurate coverage
        requestAllCoverage();
        
        console.log('📐 Recalculated all radii');
    }
    
    /**
     * Toggle between preview and accurate coverage display
     */
    function toggleAccurateCoverage() {
        state.showAccurateCoverage = !state.showAccurateCoverage;
        updateLayers();
        console.log(`🔄 Accurate coverage: ${state.showAccurateCoverage ? 'ON' : 'OFF'}`);
    }
    
    /**
     * Request recomputation of all accurate coverage geometries
     * Called after reset or recalculate operations
     */
    function requestAllCoverage() {
        console.log('📐 Requesting full coverage recomputation...');
        computeAllAccurateCoverage();
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
        getRadiusForPosition,
        toggleAccurateCoverage,
        requestAllCoverage
    };
    
})();
