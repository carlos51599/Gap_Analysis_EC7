# Zone Visibility Implementation Plan (Option A+)

## Overview

**Goal**: Implement rock-solid single source of truth with instant UI feedback for zone visibility filtering.

**Key Principles**:
1. Server computes and stores `zone_ids` for each borehole
2. Frontend uses `zone_ids` for instant visibility filtering (no geometry tests)
3. Coverage visibility derives from parent borehole visibility
4. Move/add/delete operations: instant UI update → background server sync

---

## Phase 1: Backend - Add Zone IDs to Borehole Data

### 1.1 Modify `geometry_service.py` - Add `_compute_borehole_zone_ids()`

**File**: `zone_coverage_viz/geometry_service.py`

**New Function**:
```python
def _compute_borehole_zone_ids(self, borehole_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Pre-compute which zones contain each borehole.
    
    Returns GeoDataFrame with new 'zone_ids' column containing list of zone names.
    """
    from shapely.ops import unary_union
    
    if self._zones_gdf is None or self._zones_gdf.empty:
        borehole_gdf['zone_ids'] = [[] for _ in range(len(borehole_gdf))]
        return borehole_gdf
    
    zone_ids_list = []
    for idx, bh in borehole_gdf.iterrows():
        bh_point = bh.geometry
        containing_zones = []
        
        for zone_idx, zone in self._zones_gdf.iterrows():
            # Use contains() - point must be inside or on boundary
            if zone.geometry.contains(bh_point) or zone.geometry.touches(bh_point):
                containing_zones.append(zone.zone_name)
        
        zone_ids_list.append(containing_zones)
    
    borehole_gdf['zone_ids'] = zone_ids_list
    return borehole_gdf
```

**Location**: After `_ensure_zones_loaded()` method

### 1.2 Modify `get_coverage_geojson()` to Include Zone IDs

**File**: `zone_coverage_viz/geometry_service.py`

**Modify**: The part that builds borehole features

**Before** (current):
```python
borehole_features.append({
    "type": "Feature",
    "properties": {
        "borehole_id": row.borehole_id,
        "borehole_index": idx,
        ...
    },
    ...
})
```

**After**:
```python
borehole_features.append({
    "type": "Feature",
    "properties": {
        "borehole_id": row.borehole_id,
        "borehole_index": idx,
        "zone_ids": row.zone_ids if hasattr(row, 'zone_ids') else [],
        ...
    },
    ...
})
```

### 1.3 Call `_compute_borehole_zone_ids()` at Load Time

**Location**: In `compute_coverage()` or wherever boreholes are loaded

**Add**:
```python
# After loading boreholes
self._borehole_gdf = self._compute_borehole_zone_ids(self._borehole_gdf)
```

---

## Phase 2: Frontend - Use Zone IDs for Instant Visibility

### 2.1 Store Borehole Data with Zone IDs

**File**: `zone_coverage_viz/templates/index.html`

**Modify** data storage section:

```javascript
// Global data storage
let boreholesData = null;        // GeoJSON with zone_ids in properties
let zoneVisibility = {};         // { 'Zone_0': true, 'Zone_1': false, ... }
```

### 2.2 New Visibility Helper Function

**Add** new function:

```javascript
/**
 * Check if a borehole should be visible based on zone visibility state.
 * Rule: Borehole is hidden if ANY of its containing zones is hidden.
 * 
 * @param {Array<string>} zoneIds - Array of zone names containing this borehole
 * @returns {boolean} - True if borehole should be visible
 */
function isBoreholeVisibleByZones(zoneIds) {
    if (!zoneIds || zoneIds.length === 0) {
        return true; // Borehole not in any zone = always visible
    }
    // Hidden if ANY zone is hidden
    return !zoneIds.some(zid => zoneVisibility[zid] === false);
}
```

### 2.3 Rewrite `toggleZoneVisibility()` - Instant, No Async

**Replace** existing function with:

```javascript
/**
 * Toggle zone visibility - instant UI update using pre-computed zone_ids.
 * No server calls, no geometry tests.
 */
function toggleZoneVisibility(zoneName, visible) {
    console.log(`🔄 Toggle zone ${zoneName}: ${visible}`);
    
    // 1. Update state
    zoneVisibility[zoneName] = visible;
    
    // 2. Update zone polygon opacity (instant)
    zonesLayer.eachLayer(layer => {
        if (layer.feature.properties.zone_name === zoneName) {
            layer.setStyle({
                opacity: visible ? 0.8 : 0,
                fillOpacity: visible ? 1 : 0
            });
        }
    });
    
    // 3. Update borehole visibility using zone_ids (instant)
    boreholeMarkers.forEach(marker => {
        const zoneIds = marker.feature.properties.zone_ids || [];
        const shouldBeVisible = isBoreholeVisibleByZones(zoneIds);
        
        if (shouldBeVisible) {
            if (!boreholesLayer.hasLayer(marker.markerGroup)) {
                boreholesLayer.addLayer(marker.markerGroup);
            }
        } else {
            if (boreholesLayer.hasLayer(marker.markerGroup)) {
                marker.markerGroup.remove();
            }
        }
    });
    
    // 4. Update coverage visibility using parent borehole (instant)
    coverageLayer.eachLayer(layer => {
        const bhIndex = layer.feature.properties.borehole_index;
        const parentMarker = boreholeMarkers.find(m => m.boreholeIndex === bhIndex);
        if (parentMarker) {
            const zoneIds = parentMarker.feature.properties.zone_ids || [];
            const shouldBeVisible = isBoreholeVisibleByZones(zoneIds);
            layer.setStyle({
                opacity: shouldBeVisible ? 0.5 : 0,
                fillOpacity: shouldBeVisible ? 0.3 : 0
            });
        }
    });
    
    // 5. Update count display
    updateBoreholeCountDisplay();
}
```

### 2.4 Remove Old Async Functions

**Delete or comment out**:
- `hiddenBoreholesPerZone` variable
- `updateBoreholeVisibilityForZones()` async function
- Any point-in-polygon test functions used for zone containment

### 2.5 Simplify `toggleGroupVisibility()`

**Modify** to call `toggleZoneVisibility()` synchronously:

```javascript
function toggleGroupVisibility(groupKey, visible) {
    const group = document.querySelector(`.zone-group[data-group="${groupKey}"]`);
    if (!group) return;
    
    // Toggle all zones in this group
    const zoneCheckboxes = group.querySelectorAll('.zone-toggle-checkbox');
    zoneCheckboxes.forEach(checkbox => {
        const zoneName = checkbox.dataset.zone;
        checkbox.checked = visible;
        toggleZoneVisibility(zoneName, visible);  // Instant, no async
    });
    
    // Update group checkbox state
    const groupCheckbox = group.querySelector('.group-checkbox');
    if (groupCheckbox) {
        groupCheckbox.checked = visible;
        groupCheckbox.indeterminate = false;
    }
}
```

---

## Phase 3: Borehole Operations with Background Sync

### 3.1 New Endpoint: `/api/borehole/move`

**File**: `zone_coverage_viz/server.py`

**Add**:
```python
@app.route('/api/borehole/move', methods=['POST'])
def move_borehole():
    """
    Update borehole position and return updated zone associations.
    """
    data = request.get_json()
    borehole_index = data.get('borehole_index')
    new_lat = data.get('latitude')
    new_lng = data.get('longitude')
    
    # Update position in service
    updated_bh = coverage_service.move_borehole(borehole_index, new_lat, new_lng)
    
    # Return updated borehole with new zone_ids
    return jsonify({
        'success': True,
        'borehole_index': borehole_index,
        'zone_ids': updated_bh.get('zone_ids', []),
        'position': {'lat': new_lat, 'lng': new_lng}
    })
```

### 3.2 Add `move_borehole()` to Geometry Service

**File**: `zone_coverage_viz/geometry_service.py`

**Add**:
```python
def move_borehole(self, borehole_index: int, lat: float, lng: float) -> dict:
    """
    Update borehole position and re-compute zone associations.
    """
    from shapely.geometry import Point
    
    # Update geometry
    new_point = Point(lng, lat)
    self._borehole_gdf.at[borehole_index, 'geometry'] = new_point
    
    # Re-compute zone_ids for this borehole
    containing_zones = []
    for zone_idx, zone in self._zones_gdf.iterrows():
        if zone.geometry.contains(new_point) or zone.geometry.touches(new_point):
            containing_zones.append(zone.zone_name)
    
    self._borehole_gdf.at[borehole_index, 'zone_ids'] = containing_zones
    
    # Invalidate coverage cache
    self._coverage_cache = None
    
    return {
        'borehole_index': borehole_index,
        'zone_ids': containing_zones,
        'lat': lat,
        'lng': lng
    }
```

### 3.3 Frontend: Optimistic Move with Sync

**File**: `zone_coverage_viz/templates/index.html`

**Modify** the drag end handler:

```javascript
marker.on('dragend', async function(e) {
    const newLatLng = e.target.getLatLng();
    const bhIndex = marker.boreholeIndex;
    
    // 1. INSTANT: Update marker position (already done by drag)
    
    // 2. INSTANT: Update coverage circle position
    updateCoverageCirclePosition(bhIndex, newLatLng);
    
    // 3. BACKGROUND: Server sync for zone_ids update
    try {
        const response = await fetch('/api/borehole/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                borehole_index: bhIndex,
                latitude: newLatLng.lat,
                longitude: newLatLng.lng
            })
        });
        
        const result = await response.json();
        
        // 4. RECONCILE: Update zone_ids from server
        marker.feature.properties.zone_ids = result.zone_ids;
        
        // 5. RE-EVALUATE: Visibility based on new zone_ids
        const shouldBeVisible = isBoreholeVisibleByZones(result.zone_ids);
        if (!shouldBeVisible) {
            // Borehole moved into hidden zone - hide it
            marker.markerGroup.remove();
            hideCoverageForBorehole(bhIndex);
        }
        
        console.log(`✅ Borehole ${bhIndex} moved, new zones: ${result.zone_ids.join(', ')}`);
    } catch (error) {
        console.error('Failed to sync borehole move:', error);
        // Could revert position here if needed
    }
});
```

---

## Phase 4: Testing & Validation

### 4.1 Test Cases

| Test | Expected Behavior |
|------|-------------------|
| Toggle zone off | Boreholes in that zone disappear instantly |
| Toggle zone on | Boreholes reappear instantly |
| Toggle group off | All zones in group hidden, all boreholes in those zones hidden |
| Drag borehole within same zone | Position updates, visibility unchanged |
| Drag borehole into hidden zone | Position updates, then borehole disappears after server sync |
| Drag borehole out of hidden zone | Position updates, borehole stays visible |
| Borehole in multiple zones, one hidden | Borehole is hidden |
| Borehole in multiple zones, all visible | Borehole is visible |

### 4.2 Performance Targets

| Operation | Target Latency |
|-----------|----------------|
| Zone toggle | < 50ms (instant feel) |
| Group toggle (10 zones) | < 100ms |
| Borehole drag feedback | < 16ms (60fps) |
| Server zone_ids sync | < 500ms background |

---

## Implementation Checklist

### Backend (geometry_service.py)
- [ ] Add `_compute_borehole_zone_ids()` method
- [ ] Call it after loading boreholes
- [ ] Include `zone_ids` in borehole GeoJSON properties
- [ ] Add `move_borehole()` method

### Backend (server.py)
- [ ] Add `/api/borehole/move` endpoint

### Frontend (index.html)
- [ ] Add `isBoreholeVisibleByZones()` helper function
- [ ] Rewrite `toggleZoneVisibility()` to use zone_ids
- [ ] Remove `hiddenBoreholesPerZone` tracking
- [ ] Remove `updateBoreholeVisibilityForZones()` async function
- [ ] Simplify `toggleGroupVisibility()`
- [ ] Add optimistic move handler with background sync

### Testing
- [ ] Verify zone toggle hides/shows correctly
- [ ] Verify group toggle works
- [ ] Verify borehole drag with zone_ids update
- [ ] Verify multi-zone borehole behavior
- [ ] Performance testing

---

## Files to Modify

| File | Changes |
|------|---------|
| `zone_coverage_viz/geometry_service.py` | Add zone_ids computation, move_borehole() |
| `zone_coverage_viz/server.py` | Add /api/borehole/move endpoint |
| `zone_coverage_viz/templates/index.html` | Rewrite toggle functions, add optimistic move |

**Estimated Time**: 4-6 hours total

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Server sync fails | Keep last-known zone_ids, retry on next operation |
| Slow zone computation | Use spatial index (rtree) for large zone counts |
| Edge case: borehole on boundary | Use consistent `contains()` or `intersects()` - server decides |
| Coverage circle extends outside zone | Don't clip - entire coverage follows borehole visibility |

---

## Ready to Implement

Start with **Phase 1** (backend zone_ids computation) as it establishes the foundation for all other phases.
