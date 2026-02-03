/**
 * UI Controls for Zone Coverage Visualization
 * 
 * Handles the control panel interactions and export functionality.
 */

(function() {
    'use strict';
    
    // =========================================================================
    // LAYER VISIBILITY
    // =========================================================================
    
    /**
     * Toggle layer visibility
     */
    function toggleLayer(layerId, visible) {
        if (!window.ZoneCoverage || !window.ZoneCoverage.state.deck) return;
        
        const deck = window.ZoneCoverage.state.deck;
        const layers = deck.props.layers;
        
        // Update layer visibility
        const updatedLayers = layers.map(layer => {
            if (layer.id === layerId) {
                return layer.clone({ visible: visible });
            }
            return layer;
        });
        
        deck.setProps({ layers: updatedLayers });
    }
    
    /**
     * Setup layer toggle checkboxes
     */
    function setupLayerToggles() {
        const toggles = document.querySelectorAll('.layer-toggle');
        
        toggles.forEach(toggle => {
            toggle.addEventListener('change', function() {
                const layerId = this.dataset.layer;
                toggleLayer(layerId, this.checked);
            });
        });
    }
    
    // =========================================================================
    // ZOOM CONTROLS
    // =========================================================================
    
    /**
     * Zoom to fit all data
     */
    function zoomToFit() {
        if (!window.ZoneCoverage) return;
        
        const state = window.ZoneCoverage.state;
        if (!state.zones || !state.deck) return;
        
        // Get bounds from zones
        const bounds = getBoundsFromGeoJSON(state.zones);
        if (!bounds) return;
        
        const [minLng, minLat, maxLng, maxLat] = bounds;
        
        // Calculate center
        const centerLng = (minLng + maxLng) / 2;
        const centerLat = (minLat + maxLat) / 2;
        
        // Calculate zoom (approximate)
        const latDiff = maxLat - minLat;
        const lngDiff = maxLng - minLng;
        const maxDiff = Math.max(latDiff, lngDiff);
        
        // Rough zoom calculation (adjust as needed)
        let zoom = 14;
        if (maxDiff > 0.1) zoom = 11;
        else if (maxDiff > 0.05) zoom = 12;
        else if (maxDiff > 0.02) zoom = 13;
        else if (maxDiff > 0.01) zoom = 14;
        else zoom = 15;
        
        state.deck.setProps({
            initialViewState: {
                longitude: centerLng,
                latitude: centerLat,
                zoom: zoom,
                pitch: 0,
                bearing: 0,
                transitionDuration: 500
            }
        });
    }
    
    /**
     * Get bounds from GeoJSON FeatureCollection
     */
    function getBoundsFromGeoJSON(geojson) {
        if (!geojson || !geojson.features) return null;
        
        let minLng = Infinity, minLat = Infinity;
        let maxLng = -Infinity, maxLat = -Infinity;
        
        function processCoords(coords) {
            if (typeof coords[0] === 'number') {
                // [lng, lat] pair
                minLng = Math.min(minLng, coords[0]);
                maxLng = Math.max(maxLng, coords[0]);
                minLat = Math.min(minLat, coords[1]);
                maxLat = Math.max(maxLat, coords[1]);
            } else {
                // Nested array
                coords.forEach(processCoords);
            }
        }
        
        geojson.features.forEach(feature => {
            if (feature.geometry && feature.geometry.coordinates) {
                processCoords(feature.geometry.coordinates);
            }
        });
        
        if (minLng === Infinity) return null;
        
        return [minLng, minLat, maxLng, maxLat];
    }
    
    // =========================================================================
    // STATS PANEL
    // =========================================================================
    
    /**
     * Update statistics display
     */
    function updateStats() {
        if (!window.ZoneCoverage) return;
        
        const state = window.ZoneCoverage.state;
        
        // Borehole count
        const bhCountEl = document.getElementById('borehole-count');
        if (bhCountEl) {
            bhCountEl.textContent = state.positions.length;
        }
        
        // Zone count
        const zoneCountEl = document.getElementById('zone-count');
        if (zoneCountEl && state.zones) {
            zoneCountEl.textContent = state.zones.features.length;
        }
        
        // Coverage count
        const coverageCountEl = document.getElementById('coverage-count');
        if (coverageCountEl) {
            coverageCountEl.textContent = Object.keys(state.coverage).length;
        }
    }
    
    // =========================================================================
    // KEYBOARD SHORTCUTS
    // =========================================================================
    
    /**
     * Setup keyboard shortcuts
     */
    function setupKeyboardShortcuts() {
        document.addEventListener('keydown', function(e) {
            // Escape - deselect
            if (e.key === 'Escape') {
                // Cancel any drag
                if (window.ZoneCoverage && window.ZoneCoverage.state.dragging) {
                    window.ZoneCoverage.state.dragging = false;
                    window.ZoneCoverage.state.dragIndex = null;
                }
            }
            
            // R - Reset positions
            if (e.key === 'r' && !e.ctrlKey && !e.metaKey) {
                if (window.ZoneCoverage) {
                    window.ZoneCoverage.resetPositions();
                }
            }
            
            // F - Zoom to fit
            if (e.key === 'f' && !e.ctrlKey && !e.metaKey) {
                zoomToFit();
            }
        });
    }
    
    // =========================================================================
    // TOOLTIP
    // =========================================================================
    
    /**
     * Setup hover tooltip
     */
    function setupTooltip() {
        const tooltip = document.getElementById('tooltip');
        if (!tooltip) return;
        
        // Listen for hover events from deck.gl
        // This would need integration with deck.gl's getTooltip
    }
    
    // =========================================================================
    // INITIALIZATION
    // =========================================================================
    
    function initUI() {
        setupLayerToggles();
        setupKeyboardShortcuts();
        setupTooltip();
        
        // Update stats after a short delay to ensure data is loaded
        setTimeout(updateStats, 500);
        
        // Zoom to fit button
        const fitBtn = document.getElementById('fit-btn');
        if (fitBtn) {
            fitBtn.addEventListener('click', zoomToFit);
        }
        
        console.log('🎮 UI controls initialized');
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initUI);
    } else {
        initUI();
    }
    
    // Expose utilities
    window.ZoneCoverageUI = {
        toggleLayer,
        zoomToFit,
        updateStats
    };
    
})();
