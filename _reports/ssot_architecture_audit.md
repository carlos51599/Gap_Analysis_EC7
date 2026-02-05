# SSOT Architecture Audit - zone_coverage_viz

## Audit Scope
Review of `Main/Gap_Analysis_EC7/zone_coverage_viz/` for Single Source of Truth (SSOT) compliance.

## SSOT Principles Checklist

| Principle                             | Status      | Details                                            |
| ------------------------------------- | ----------- | -------------------------------------------------- |
| Server computes geometry/clipping     | ✅ Compliant | All coverage clipping via `/api/coverage/filtered` |
| Server computes zone associations     | ✅ Compliant | `zone_ids` computed by server at load time         |
| Server handles filtering logic        | ✅ Compliant | `mode` parameter sent to server, server decides    |
| Client only renders what server sends | ⚠️ Gray Area | Borehole marker visibility uses client-side toggle |

---

## ✅ COMPLIANT: Coverage Geometry Clipping

**Code:** `server.py` → `/api/coverage/filtered`
```python
coverages_geojson = coverage_service.compute_all_coverages_filtered(
    boreholes, exclude_zones
)
```

**Verdict:** SSOT compliant. Server performs all Shapely intersection/clipping operations.

---

## ✅ COMPLIANT: Existing Coverage Clipping

**Code:** `server.py` → `/api/existing-coverage/filtered`
```python
clipped = coverage_service.clip_coverage_to_visible_zones(existing, exclude_zones)
```

**Verdict:** SSOT compliant. Server clips existing coverage to visible zones.

---

## ✅ COMPLIANT: Zone ID Computation

**Code:** `data_loader.py` → `_compute_zone_ids()`
```python
def _compute_zone_ids(self) -> None:
    """Compute which zones each borehole belongs to (zone_ids)."""
    # Uses Shapely point-in-polygon, stores result in borehole properties
```

**Verdict:** SSOT compliant. Server computes zone associations once at load time.

---

## ⚠️ GRAY AREA: Borehole Marker Visibility

**Code:** `index.html` → `toggleZoneVisibility()`
```javascript
// 3. Update borehole visibility based on mode (uses pre-computed zone_ids from server)
boreholeMarkers.forEach((grabCircle) => {
    let shouldBeVisible;
    if (mode === 'hide_zone_boreholes') {
        shouldBeVisible = isBoreholeVisibleByLocationZone(locationZoneIds);
    } else {
        shouldBeVisible = isBoreholeVisibleByZones(coverageZoneIds);
    }
    // Add/remove marker from map based on shouldBeVisible
});
```

**Analysis:**
- **Data source:** Server (zone_ids, coverageZoneIds)
- **Decision logic:** Client (checks UI checkbox state)
- **Justification:** Instant UI responsiveness (no API call per toggle)

**Verdict:** Acceptable trade-off for UX. The **data** is from server (SSOT), only the **visibility toggle** is client-side.

**Alternative (pure SSOT):** Server endpoint that accepts hidden zones and returns visible borehole indices. Would add ~200ms latency per checkbox click.

---

## ⚠️ GRAY AREA: Point-in-Polygon for Marker Styling

**Code:** `index.html` → `isPointInAnyZone()`, `updateMarkerForZoneChange()`
```javascript
function isPointInAnyZone(lon, lat) {
    for (const feature of zonesData.features) {
        if (isPointInPolygon(point, feature.geometry)) {
            return true;
        }
    }
    return false;
}
```

**Usage:** Only for visual feedback when dragging markers (change color if borehole moves outside zones).

**Verdict:** Acceptable. This is **visual feedback only**, not filtering or data decisions. The actual zone associations are still server-computed.

---

## Summary

| Area                       | SSOT Status | Recommendation                      |
| -------------------------- | ----------- | ----------------------------------- |
| Coverage clipping          | ✅ Compliant | No changes needed                   |
| Existing coverage clipping | ✅ Compliant | No changes needed                   |
| Zone ID computation        | ✅ Compliant | No changes needed                   |
| Borehole marker visibility | ⚠️ Gray area | Keep for UX; document as acceptable |
| Marker styling on drag     | ⚠️ Gray area | Keep; visual-only                   |

## Potential Future Enhancement

If pure SSOT becomes a requirement, consider:

1. **Server-side borehole visibility endpoint:**
   ```
   POST /api/boreholes/visibility
   Body: { excludeZones: [...], mode: "..." }
   Response: { visibleIndices: [0, 2, 5, ...] }
   ```
   
2. **Debounced calls:** Bundle multiple checkbox clicks into a single API call.

3. **WebSocket for real-time sync:** Push visibility changes instead of polling.

For the current use case (interactive map with instant feedback), the hybrid approach is the best balance of SSOT principles and user experience.
