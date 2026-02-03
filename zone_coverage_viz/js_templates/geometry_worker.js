/**
 * Geometry Worker for Zone-Aware Coverage Calculation
 * 
 * This Web Worker performs Turf.js geometry operations off the main thread
 * to ensure smooth borehole dragging without UI jank.
 * 
 * Messages:
 * - INIT: Initialize with zone geometries and Turf.js code
 * - COMPUTE_COVERAGE: Compute zone-aware coverage for a single borehole
 * - COMPUTE_ALL: Batch compute coverage for all boreholes
 */

// Worker state
let turfLoaded = false;
let zoneFeatures = [];

/**
 * Message handler - routes messages to appropriate functions
 */
self.onmessage = function(e) {
    const { type, payload, id } = e.data;
    
    try {
        switch (type) {
            case 'INIT':
                handleInit(payload);
                self.postMessage({ type: 'INIT_COMPLETE', id });
                break;
                
            case 'COMPUTE_COVERAGE':
                const coverage = computeCoverage(payload);
                self.postMessage({ type: 'COVERAGE_RESULT', id, payload: coverage });
                break;
                
            case 'COMPUTE_ALL':
                const allCoverage = computeAllCoverage(payload);
                self.postMessage({ type: 'ALL_COVERAGE_RESULT', id, payload: allCoverage });
                break;
                
            default:
                self.postMessage({ 
                    type: 'ERROR', 
                    id, 
                    payload: { message: `Unknown message type: ${type}` }
                });
        }
    } catch (error) {
        self.postMessage({
            type: 'ERROR',
            id,
            payload: { message: error.message, stack: error.stack }
        });
    }
};

/**
 * Initialize worker with Turf.js and zone data
 */
function handleInit(payload) {
    const { turfCode, zones } = payload;
    
    // Load Turf.js by evaluating the code
    if (turfCode && !turfLoaded) {
        try {
            // Create a function that returns turf and execute it
            const loadTurf = new Function(turfCode + '; return turf;');
            self.turf = loadTurf();
            turfLoaded = true;
        } catch (e) {
            throw new Error('Failed to load Turf.js: ' + e.message);
        }
    }
    
    // Store zone features
    if (zones && zones.features) {
        zoneFeatures = zones.features;
    }
}

/**
 * Convert meters to approximate degrees at given latitude
 */
function metersToDegrees(meters, latitude) {
    const earthRadius = 6371000.0;
    const latRad = latitude * Math.PI / 180.0;
    
    const degLat = meters / (earthRadius * Math.PI / 180.0);
    const degLon = meters / (earthRadius * Math.cos(latRad) * Math.PI / 180.0);
    
    return (degLat + degLon) / 2.0;
}

/**
 * Compute zone-aware coverage for a single borehole
 * 
 * Creates "flower petal" coverage by:
 * 1. For each zone, create a circle with zone's max_spacing radius
 * 2. Intersect circle with zone boundary
 * 3. Union all intersections
 */
function computeCoverage(payload) {
    const { position, zones, defaultMaxSpacing } = payload;
    
    if (!turfLoaded || !self.turf) {
        throw new Error('Turf.js not loaded');
    }
    
    const [lon, lat] = position;
    const point = self.turf.point([lon, lat]);
    
    // Use provided zones or stored zones
    const zoneList = (zones && zones.features) ? zones.features : zoneFeatures;
    
    if (!zoneList || zoneList.length === 0) {
        return null;
    }
    
    const coverageParts = [];
    
    for (const zone of zoneList) {
        // Get zone's max spacing (in meters)
        const maxSpacingM = zone.properties.max_spacing_m || defaultMaxSpacing || 100;
        
        // Convert to degrees at borehole latitude
        const radiusDeg = metersToDegrees(maxSpacingM, lat);
        
        // Create coverage circle (in kilometers for turf.circle)
        const radiusKm = maxSpacingM / 1000.0;
        const circle = self.turf.circle([lon, lat], radiusKm, {
            steps: 64,
            units: 'kilometers'
        });
        
        // Intersect with zone
        try {
            const intersection = self.turf.intersect(
                self.turf.featureCollection([circle, zone])
            );
            
            if (intersection && intersection.geometry) {
                coverageParts.push(intersection);
            }
        } catch (e) {
            // Intersection failed (topological error), skip this zone
            console.warn('Intersection failed for zone:', zone.properties.zone_id);
        }
    }
    
    if (coverageParts.length === 0) {
        return null;
    }
    
    // Union all parts
    if (coverageParts.length === 1) {
        return coverageParts[0].geometry;
    }
    
    try {
        const fc = self.turf.featureCollection(coverageParts);
        const unioned = self.turf.union(fc);
        return unioned ? unioned.geometry : null;
    } catch (e) {
        // Union failed, return first part
        return coverageParts[0].geometry;
    }
}

/**
 * Compute coverage for all boreholes (batch operation)
 */
function computeAllCoverage(payload) {
    const { boreholes, zones, defaultMaxSpacing } = payload;
    
    const results = {};
    
    for (const bh of boreholes) {
        const bhId = bh.id || bh.properties?.id;
        const position = bh.geometry?.coordinates || bh.position;
        
        if (!position || !bhId) continue;
        
        const coverage = computeCoverage({
            position,
            zones,
            defaultMaxSpacing
        });
        
        if (coverage) {
            results[bhId] = coverage;
        }
    }
    
    return results;
}
