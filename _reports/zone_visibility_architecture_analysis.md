# Zone Visibility Architecture Analysis

## Current Problem

**Symptom**: When hiding zones using checkboxes, the behavior is inconsistent - sometimes boreholes and coverage hide correctly, sometimes they don't.

**Root Cause**: The architecture has **three separate data models** that aren't inherently linked:

1. **Zone polygons** (Leaflet GeoJSON layer)
2. **Borehole markers** (Leaflet CircleMarkers in a FeatureGroup)
3. **Coverage polygons** (Leaflet GeoJSON layer)

Each layer is managed independently, and the relationship between them is computed on-the-fly using point-in-polygon tests. This creates race conditions and synchronization issues.

---

## Current Architecture Flow

```mermaid
flowchart TD
    subgraph Frontend["Frontend (index.html)"]
        ZV[zoneVisibility dict]
        HBZ[hiddenBoreholesPerZone dict]
        ZL[zonesLayer]
        BL[boreholesLayer]
        CL[coverageLayer]
    end
    
    subgraph Backend["Backend (geometry_service.py)"]
        GS[CoverageService]
        CC[_coverage_cache]
    end
    
    CB[Checkbox Toggle] --> TZV[toggleZoneVisibility]
    TZV -->|"1. Update opacity"| ZL
    TZV -->|"2. Capture boreholes in zone"| HBZ
    TZV -->|"3. Call update"| UBVZ[updateBoreholeVisibilityForZones]
    
    UBVZ -->|"4. Show/hide markers"| BL
    UBVZ -->|"5. Fetch filtered coverage"| API[/api/coverage/filtered]
    API --> GS
    GS -->|"6. Compute clipped coverage"| CL
    
    style CB fill:#f9f,stroke:#333
    style TZV fill:#faa,stroke:#333
    style UBVZ fill:#faa,stroke:#333
```

### Problems with Current Flow

1. **Temporal Coupling**: Boreholes are captured "when the zone is hidden" - but if the zone geometry loads slowly or the point-in-polygon test fails, boreholes might be missed.

2. **Redundant Point-in-Polygon Tests**: The frontend does point-in-polygon tests in JavaScript, while the backend does similar tests in Python. Results can differ due to floating-point precision.

3. **No Single Source of Truth**: Each layer has its own visibility state:
   - `zoneVisibility` for zones
   - `hiddenBoreholesPerZone` for boreholes
   - Coverage gets recomputed from server

4. **Async Race Conditions**: When `toggleGroupVisibility()` calls `toggleZoneVisibility()` in a loop, each call triggers an async `updateBoreholeVisibilityForZones()`, potentially interleaving.

5. **Coverage Layer Doesn't Track Borehole Ownership**: The coverage GeoJSON has `borehole_index` but hiding a coverage by setting opacity doesn't work with the count/export logic.

---

## Proposed Architecture Options

### Option A: Server-Side Unified State (Recommended)

**Principle**: Server is the single source of truth. All visibility filtering happens server-side.

```mermaid
flowchart TD
    subgraph Frontend["Frontend"]
        VS[visibleZones array]
        CB[Checkbox] -->|"Update visibleZones"| VS
        VS -->|"Single API call"| API
    end
    
    subgraph Backend["Backend"]
        API[/api/filtered-view] --> FS[FilterService]
        FS --> FZ[Filter zones]
        FS --> FB[Filter boreholes by zone containment]
        FS --> FC[Compute clipped coverage]
        FS -->|"Return unified GeoJSON"| R[Response]
    end
    
    R -->|"Replace all layers"| Frontend
```

**Implementation**:

```python
# New endpoint: /api/filtered-view
@app.route('/api/filtered-view', methods=['POST'])
def get_filtered_view():
    """Return zones, boreholes, and coverage filtered by visible zones."""
    visible_zones = request.json.get('visibleZones', [])
    
    return jsonify({
        'zones': filter_zones(visible_zones),
        'boreholes': filter_boreholes_by_zones(visible_zones),
        'coverage': compute_clipped_coverage(visible_zones),
        'stats': compute_stats_for_visible(visible_zones)
    })
```

**Pros**:
- Single source of truth (server)
- Atomic updates (all layers update together)
- No race conditions
- Backend can optimize with spatial indexing
- Easy to debug (one endpoint, one response)

**Cons**:
- More network traffic (full layer replacement)
- Latency on each toggle (could be mitigated with debouncing)

---

### Option B: Borehole-Zone Association at Data Load

**Principle**: Pre-compute which zones each borehole belongs to at load time. Store this mapping in the borehole data.

```javascript
// At data load time, add zone associations to each borehole
boreholesData.features.forEach(borehole => {
    borehole.properties.zones = []; // e.g., ['Embankment_0', 'Embankment_1']
    zonesData.features.forEach(zone => {
        if (isPointInPolygon(borehole.geometry, zone.geometry)) {
            borehole.properties.zones.push(zone.properties.zone_name);
        }
    });
});

// Filtering becomes trivial
function getVisibleBoreholes() {
    const hiddenZones = new Set(Object.keys(zoneVisibility).filter(z => !zoneVisibility[z]));
    return boreholesData.features.filter(bh => 
        !bh.properties.zones.some(z => hiddenZones.has(z))
    );
}
```

**Pros**:
- Fast filtering (no geometry computation on toggle)
- Deterministic (association computed once)
- Works offline (no server calls for visibility)

**Cons**:
- Borehole moves require re-computing associations
- Doesn't handle coverage clipping (still needs server)

---

### Option C: Unified Layer Container with Data Binding

**Principle**: Create a single data model that binds zones, boreholes, and coverage together. Changes propagate automatically.

```javascript
class ZoneCoverageModel {
    constructor() {
        this.zones = new Map(); // zone_name -> {geometry, visible, boreholes, coverage}
        this.boreholes = new Map(); // borehole_id -> {geometry, zone_associations}
    }
    
    setZoneVisibility(zoneName, visible) {
        const zone = this.zones.get(zoneName);
        zone.visible = visible;
        this._updateDependentLayers(zoneName);
    }
    
    _updateDependentLayers(zoneName) {
        const zone = this.zones.get(zoneName);
        
        // Update zone layer
        this.zoneLayer.setStyle(zoneName, zone.visible ? {} : {opacity: 0});
        
        // Update all boreholes in this zone
        zone.boreholes.forEach(bhId => {
            const bh = this.boreholes.get(bhId);
            const anyZoneVisible = bh.zone_associations.some(z => this.zones.get(z).visible);
            this.boreholeLayer.setVisibility(bhId, anyZoneVisible);
        });
        
        // Request coverage re-clip
        this._fetchClippedCoverage();
    }
}
```

**Pros**:
- Clean separation of concerns
- Reactive updates (change propagates automatically)
- Testable (model can be unit tested)

**Cons**:
- Significant refactor required
- Need to handle edge cases (borehole in multiple zones)

---

### Option D: Simplify to "Hide Zone = Hide Everything Inside" (Quickest Fix)

**Principle**: When a zone is hidden, simply hide everything geometrically inside it at that moment. No special tracking.

```javascript
function toggleZoneVisibility(zoneName, visible) {
    zoneVisibility[zoneName] = visible;
    
    // Update zone polygon visibility (CSS only, keep in DOM)
    zonesLayer.eachLayer(layer => {
        if (layer.feature.properties.zone_name === zoneName) {
            layer.setStyle({ opacity: visible ? 0.8 : 0, fillOpacity: visible ? 1 : 0 });
        }
    });
    
    // Get zone geometry for containment tests
    const zoneGeom = zonesData.features.find(f => f.properties.zone_name === zoneName)?.geometry;
    
    // Toggle boreholes inside this zone
    boreholeMarkers.forEach(marker => {
        const point = [marker.getLatLng().lng, marker.getLatLng().lat];
        const inThisZone = isPointInPolygon(point, zoneGeom);
        
        if (inThisZone) {
            if (visible) {
                boreholesLayer.addLayer(marker.markerGroup);
            } else {
                marker.markerGroup.remove();
            }
        }
    });
    
    // Toggle coverage for boreholes inside this zone
    coverageLayer.eachLayer(layer => {
        const bhIndex = layer.feature.properties.borehole_index;
        const bhMarker = boreholeMarkers.find(m => m.boreholeIndex === bhIndex);
        if (bhMarker) {
            const point = [bhMarker.getLatLng().lng, bhMarker.getLatLng().lat];
            if (isPointInPolygon(point, zoneGeom)) {
                layer.setStyle({ opacity: visible ? 0.5 : 0, fillOpacity: visible ? 0.3 : 0 });
            }
        }
    });
    
    updateBoreholeCountDisplay();
}
```

**Pros**:
- Simplest to implement
- No async operations
- Predictable behavior

**Cons**:
- Coverage doesn't get geometrically clipped (just hidden/shown)
- Borehole on zone boundary could have inconsistent behavior

---

## Recommendation

**For production reliability**: **Option A (Server-Side Unified State)**

- Centralizes all visibility logic on the server
- Eliminates client-side race conditions
- Makes coverage clipping deterministic
- Easier to test and debug

**Implementation steps**:

1. Create `/api/filtered-view` endpoint that accepts `visibleZones` array
2. Server returns filtered zones, boreholes, and pre-clipped coverage
3. Frontend replaces all layer data atomically
4. Add debouncing (300ms) for rapid checkbox toggles
5. Show loading indicator during filter updates

**For quick fix**: **Option D (Simplify)**

- Can be implemented in 1-2 hours
- Removes the complex `hiddenBoreholesPerZone` tracking
- Trade-off: No geometric coverage clipping (just hide/show)

---

## Technical Debt in Current System

| Issue                                   | Severity | Option A Fixes | Option D Fixes |
| --------------------------------------- | -------- | -------------- | -------------- |
| Race conditions in async updates        | High     | ✅              | ✅              |
| Redundant point-in-polygon logic        | Medium   | ✅              | ❌              |
| No single source of truth               | High     | ✅              | Partial        |
| Complex hiddenBoreholesPerZone tracking | Medium   | ✅              | ✅              |
| Coverage clipping inconsistency         | High     | ✅              | ❌              |
| Group toggle fires multiple async calls | High     | ✅              | ✅              |

---

## Recommended Approach: Option A+ (Server Truth + Optimistic UI)

**Validated by Zen MCP expert review** ✅

Based on user requirements for **instant feedback** during interactive operations (moving, deleting, adding boreholes), we need a hybrid approach:

### Core Principles

1. **Server is source of truth** for zone-borehole-coverage relationships
2. **Optimistic UI updates** for instant feedback
3. **Background sync** to verify/reconcile state
4. **Pre-computed associations** to avoid runtime geometry tests
5. **Coverage derives from borehole** - no separate zone tracking for coverage

### Architecture

```mermaid
flowchart TD
    subgraph Frontend["Frontend (Optimistic UI)"]
        BHD[boreholeData with zone_ids]
        ZV[zoneVisibility]
        UI[Instant UI Update]
    end
    
    subgraph Backend["Backend (Source of Truth)"]
        API[/api/sync-state]
        ZBS[Zone-Borehole Service]
        CC[Coverage Cache]
    end
    
    User[User Action] --> UI
    UI -->|"1. Instant feedback"| Frontend
    UI -->|"2. Background sync"| API
    API --> ZBS
    ZBS -->|"3. Verify & reconcile"| Frontend
```

### Data Model

Each borehole carries its zone associations:

```javascript
// Borehole data structure (loaded once from server)
{
    "type": "Feature",
    "properties": {
        "borehole_id": "BH-001",
        "borehole_index": 0,
        "zone_ids": ["Embankment_0", "Embankment_1"],  // Pre-computed on server
        "coverage_radius_m": 100
    },
    "geometry": { "type": "Point", "coordinates": [x, y] }
}
```

### Visibility Logic (Instant, No Server Call)

```javascript
// Toggle zone - instant UI update
function toggleZoneVisibility(zoneName, visible) {
    zoneVisibility[zoneName] = visible;
    
    // Instant zone opacity update
    updateZoneLayerOpacity(zoneName, visible);
    
    // Instant borehole visibility (uses pre-computed zone_ids)
    boreholeData.features.forEach(bh => {
        if (bh.properties.zone_ids.includes(zoneName)) {
            const anyZoneVisible = bh.properties.zone_ids.some(zid => zoneVisibility[zid] !== false);
            setBoreholeVisibility(bh.properties.borehole_index, anyZoneVisible);
        }
    });
    
    // Instant coverage visibility (using borehole association)
    updateCoverageVisibility();
    
    // Background: request clipped coverage for visible zones
    debouncedFetchClippedCoverage();
}
```

### Borehole Operations (Instant + Background Sync)

```javascript
// Move borehole - instant feedback + background reconciliation
function moveBorehole(boreholeIndex, newPosition) {
    // 1. Instant UI update
    updateBoreholeMarkerPosition(boreholeIndex, newPosition);
    updateCoverageCirclePosition(boreholeIndex, newPosition);
    
    // 2. Background sync - server re-computes zone associations
    fetch('/api/borehole/move', {
        method: 'POST',
        body: JSON.stringify({ borehole_index: boreholeIndex, position: newPosition })
    })
    .then(response => response.json())
    .then(data => {
        // 3. Reconcile zone associations if changed
        boreholeData.features[boreholeIndex].properties.zone_ids = data.zone_ids;
        
        // 4. Update visibility if moved into hidden zone
        const anyZoneVisible = data.zone_ids.some(zid => zoneVisibility[zid] !== false);
        setBoreholeVisibility(boreholeIndex, anyZoneVisible);
    });
}
```

### Server Endpoints

| Endpoint | Purpose | When Called |
|----------|---------|-------------|
| `/api/boreholes/with-zones` | Load boreholes with pre-computed zone_ids | Page load |
| `/api/borehole/move` | Update position, return new zone_ids | After drag |
| `/api/borehole/add` | Create borehole, return with zone_ids | After add |
| `/api/borehole/delete` | Remove borehole | After delete |
| `/api/coverage/clipped` | Get clipped coverage for visible zones | Debounced on toggle |

### Benefits

1. **Instant Feedback**: UI updates immediately without waiting for server
2. **Correct State**: Server computes zone associations correctly (no JS geometry bugs)
3. **No Race Conditions**: Pre-computed associations mean no runtime containment tests
4. **Background Clipping**: Coverage geometry clipping happens asynchronously
5. **Resilient**: If server is slow, UI still responsive

### Implementation Plan

| Phase | Task | Effort |
|-------|------|--------|
| 1 | Add `zone_ids` to borehole GeoJSON on server | 1 hour |
| 2 | Modify frontend to use `zone_ids` for visibility | 1 hour |
| 3 | Add `/api/borehole/move` endpoint | 1 hour |
| 4 | Wire up optimistic move with background sync | 1 hour |
| 5 | Add/Delete operations with zone association | 1 hour |
| 6 | Debounced coverage clipping for zone toggle | 1 hour |

**Total: ~6 hours for full implementation**

---

## Expert Review Notes (Zen MCP)

### Key Validations
- ✅ Server-computed `zone_ids` is the **gold standard** for consistency
- ✅ Optimistic UI with server reconciliation is the correct pattern
- ✅ `zone_ids` array correctly handles multi-zone boreholes

### Critical Clarifications

**1. Coverage Visibility - Derive from Borehole (SIMPLIFY)**

> "If a borehole is hidden by the zone toggle, its entire associated coverage is also hidden."

**DO NOT** attempt geometric clipping of coverage on the frontend. This reintroduces complexity.

```javascript
// CORRECT: Coverage visibility follows borehole visibility
function isBoreholeVisible(borehole) {
    return !borehole.properties.zone_ids.some(zid => zoneVisibility[zid] === false);
}

// Coverage visibility = parent borehole visibility
coverageFeatures.forEach(cov => {
    const parentBorehole = boreholes[cov.properties.borehole_index];
    cov.visible = isBoreholeVisible(parentBorehole);
});
```

**2. Multi-Zone Boreholes - Hide if ANY zone is hidden**

```javascript
// Borehole in Zone A and Zone B
// If Zone A is hidden → borehole is hidden (even if Zone B is visible)
const shouldHide = borehole.properties.zone_ids.some(zid => zoneVisibility[zid] === false);
```

**3. Boundary Edge Cases - Server Defines "In Zone"**

Use `ST_Intersects` (includes boundary) or `ST_Within` consistently on server. Client trusts `zone_ids` from server - no local geometry tests.

**4. Reconciliation Strategy**

When server returns updated `zone_ids` after a move:
1. Update local borehole data
2. Re-evaluate visibility based on current zone toggles
3. If borehole moved into a hidden zone, it disappears (expected behavior)

### What NOT to Use (Over-Engineering)

- ❌ CRDTs/OT (for real-time multi-user editing - not needed for single user)
- ❌ Client-side polygon clipping (Turf.js etc. - slow, inconsistent)
- ❌ Separate `zone_ids` for coverage (derive from borehole instead)

---

## Next Steps

Ready to implement Option A+ with:
- Pre-computed zone associations in borehole data
- Instant UI updates using those associations  
- Background sync for operations that change geometry
