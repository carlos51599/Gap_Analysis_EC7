/**
 * Coverage Worker - Computes accurate zone-clipped coverage polygons
 * 
 * This Web Worker handles computationally intensive Turf.js operations
 * off the main thread to keep the UI responsive during drag interactions.
 * 
 * Input message: { type: 'compute', boreholeIndex: number, position: [lon, lat] }
 * Output message: { type: 'result', boreholeIndex: number, coverage: GeoJSON }
 */

// Load Turf.js immediately on worker start
let turfLoaded = false;
try {
    importScripts('https://unpkg.com/@turf/turf@6.5.0/turf.min.js');
    turfLoaded = typeof turf !== 'undefined';
    console.log('✅ Turf.js loaded in worker:', turfLoaded);
} catch (e) {
    console.error('❌ Failed to load Turf.js in worker:', e);
    turfLoaded = false;
}

// Worker state
let zones = null;

/**
 * Handle messages from main thread
 */
self.onmessage = function(e) {
    const msg = e.data;
    
    switch (msg.type) {
        case 'init':
            // Initialize with zone data
            initWorker(msg);
            break;
            
        case 'compute':
            // Compute coverage for a single borehole
            computeCoverage(msg.boreholeIndex, msg.position);
            break;
            
        case 'computeAll':
            // Compute coverage for all boreholes
            computeAllCoverage(msg.positions);
            break;
            
        default:
            console.warn('Unknown message type:', msg.type);
    }
};

/**
 * Initialize the worker with zone data
 */
function initWorker(msg) {
    // Store zone data
    zones = msg.zones;
    
    self.postMessage({ 
        type: 'ready', 
        turfLoaded: turfLoaded,
        zoneCount: zones ? zones.features.length : 0
    });
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
function computeCoverage(boreholeIndex, position) {
    if (!turfLoaded || !zones || !position) {
        self.postMessage({
            type: 'result',
            boreholeIndex: boreholeIndex,
            coverage: null,
            error: 'Worker not ready or missing data'
        });
        return;
    }
    
    try {
        const [lon, lat] = position;
        const point = turf.point([lon, lat]);
        
        // Collect coverage fragments from each zone
        const fragments = [];
        
        for (const zone of zones.features) {
            const maxSpacing = zone.properties.max_spacing_m || 100;
            
            // Quick bounding box check - skip zones too far away
            const bbox = turf.bbox(zone);
            const expandedBbox = [
                bbox[0] - degreesFromMeters(maxSpacing, lat),
                bbox[1] - degreesFromMeters(maxSpacing, lat),
                bbox[2] + degreesFromMeters(maxSpacing, lat),
                bbox[3] + degreesFromMeters(maxSpacing, lat)
            ];
            
            // Check if point is within expanded bbox
            if (lon < expandedBbox[0] || lon > expandedBbox[2] ||
                lat < expandedBbox[1] || lat > expandedBbox[3]) {
                continue;  // Skip this zone - too far
            }
            
            // Buffer the point by this zone's spacing
            // Note: Turf.js buffer uses kilometers by default
            const buffer = turf.buffer(point, maxSpacing / 1000, { units: 'kilometers' });
            
            if (!buffer) continue;
            
            // Intersect with zone polygon
            try {
                const intersection = turf.intersect(
                    turf.featureCollection([buffer]),
                    turf.featureCollection([zone])
                );
                
                if (intersection && !isEmpty(intersection)) {
                    fragments.push(intersection);
                }
            } catch (e) {
                // Skip invalid intersections
            }
        }
        
        // Union all fragments
        let coverage = null;
        if (fragments.length === 1) {
            coverage = fragments[0];
        } else if (fragments.length > 1) {
            try {
                // Progressively union fragments
                coverage = fragments[0];
                for (let i = 1; i < fragments.length; i++) {
                    const union = turf.union(
                        turf.featureCollection([coverage]),
                        turf.featureCollection([fragments[i]])
                    );
                    if (union) {
                        coverage = union;
                    }
                }
            } catch (e) {
                // If union fails, use first fragment
                coverage = fragments[0];
            }
        }
        
        // Add borehole index as property
        if (coverage) {
            coverage.properties = coverage.properties || {};
            coverage.properties.boreholeIndex = boreholeIndex;
        }
        
        self.postMessage({
            type: 'result',
            boreholeIndex: boreholeIndex,
            coverage: coverage
        });
        
    } catch (e) {
        self.postMessage({
            type: 'result',
            boreholeIndex: boreholeIndex,
            coverage: null,
            error: e.message
        });
    }
}

/**
 * Compute coverage for all boreholes at once (initial load)
 */
function computeAllCoverage(positions) {
    if (!turfLoaded || !zones || !positions) {
        self.postMessage({
            type: 'allResults',
            coverages: [],
            error: 'Worker not ready or missing data'
        });
        return;
    }
    
    const coverages = [];
    const startTime = performance.now();
    
    for (let i = 0; i < positions.length; i++) {
        const coverage = computeSingleCoverage(i, positions[i]);
        if (coverage) {
            coverages.push(coverage);
        }
        
        // Report progress every 50 boreholes
        if (i > 0 && i % 50 === 0) {
            self.postMessage({
                type: 'progress',
                current: i,
                total: positions.length
            });
        }
    }
    
    const elapsed = performance.now() - startTime;
    
    self.postMessage({
        type: 'allResults',
        coverages: coverages,
        elapsed: elapsed
    });
}

/**
 * Compute coverage for a single borehole (internal, no postMessage)
 */
function computeSingleCoverage(boreholeIndex, position) {
    if (!position) return null;
    
    try {
        const [lon, lat] = position;
        const point = turf.point([lon, lat]);
        const fragments = [];
        
        for (const zone of zones.features) {
            const maxSpacing = zone.properties.max_spacing_m || 100;
            
            // Quick bbox check
            const bbox = turf.bbox(zone);
            const margin = degreesFromMeters(maxSpacing, lat);
            
            if (lon < bbox[0] - margin || lon > bbox[2] + margin ||
                lat < bbox[1] - margin || lat > bbox[3] + margin) {
                continue;
            }
            
            const buffer = turf.buffer(point, maxSpacing / 1000, { units: 'kilometers' });
            if (!buffer) continue;
            
            try {
                const intersection = turf.intersect(
                    turf.featureCollection([buffer]),
                    turf.featureCollection([zone])
                );
                
                if (intersection && !isEmpty(intersection)) {
                    fragments.push(intersection);
                }
            } catch (e) {
                // Skip
            }
        }
        
        let coverage = null;
        if (fragments.length === 1) {
            coverage = fragments[0];
        } else if (fragments.length > 1) {
            try {
                coverage = fragments[0];
                for (let i = 1; i < fragments.length; i++) {
                    const union = turf.union(
                        turf.featureCollection([coverage]),
                        turf.featureCollection([fragments[i]])
                    );
                    if (union) coverage = union;
                }
            } catch (e) {
                coverage = fragments[0];
            }
        }
        
        if (coverage) {
            coverage.properties = coverage.properties || {};
            coverage.properties.boreholeIndex = boreholeIndex;
        }
        
        return coverage;
        
    } catch (e) {
        return null;
    }
}

/**
 * Convert meters to approximate degrees at a given latitude
 */
function degreesFromMeters(meters, latitude) {
    // At equator: 1 degree ≈ 111,320 meters
    // This decreases with cos(latitude) for longitude
    const metersPerDegree = 111320 * Math.cos(latitude * Math.PI / 180);
    return meters / metersPerDegree;
}

/**
 * Check if a geometry is empty
 */
function isEmpty(geom) {
    if (!geom) return true;
    if (geom.type === 'GeometryCollection' && (!geom.geometries || geom.geometries.length === 0)) {
        return true;
    }
    if (geom.geometry) {
        return isEmpty(geom.geometry);
    }
    return false;
}
