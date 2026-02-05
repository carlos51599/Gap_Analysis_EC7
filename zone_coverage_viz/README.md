# Zone Coverage Visualization

Interactive map for visualizing and adjusting proposed borehole positions with zone-aware coverage.

## Features

- **Draggable Boreholes**: Click and drag proposed borehole markers to new positions
- **Zone-Aware Coverage**: Coverage radius automatically adjusts based on each zone's `max_spacing_m`
- **Existing Coverage Layer**: Shows current borehole coverage (green) from main.py output
- **Proposed Coverage Layer**: Shows coverage from proposed/optimized boreholes (blue)
- **Real-time Updates**: Coverage recalculates when boreholes are moved
- **Export**: Download modified positions as CSV

## Quick Start

1. **Generate data** (run main.py first to create Output files):
   ```bash
   cd Main/Gap_Analysis_EC7
   python main.py
   ```

2. **Generate zone coverage data**:
   ```bash
   python zone_coverage_viz/generate_zone_data.py
   ```

3. **Start the server**:
   ```bash
   python zone_coverage_viz/server.py
   ```
   Or use the batch file:
   ```bash
   run_zone_coverage_viz.bat
   ```

4. **Open browser** at http://localhost:5051

## Architecture

```
zone_coverage_viz/
├── server.py           # Flask web server
├── data_loader.py      # Loads zone_coverage_data.json
├── geometry_service.py # Shapely-based coverage computation
├── generate_zone_data.py  # Creates JSON from main.py output
└── templates/
    └── index.html      # Leaflet map frontend
```

## Data Flow

1. `main.py` → generates shapefiles, proposed boreholes, coverage polygons
2. `generate_zone_data.py` → converts to `zone_coverage_data.json` (WGS84)
3. `server.py` → serves JSON via REST API
4. `index.html` → Leaflet map with draggable markers

## API Endpoints

| Endpoint                 | Method | Description                           |
| ------------------------ | ------ | ------------------------------------- |
| `/`                      | GET    | Map interface                         |
| `/api/zones`             | GET    | Zone polygons with max_spacing_m      |
| `/api/boreholes`         | GET    | Proposed borehole positions           |
| `/api/coverage/all`      | GET    | All proposed coverage polygons        |
| `/api/coverage/existing` | GET    | Existing borehole coverage            |
| `/api/coverage/update`   | POST   | Recompute coverage for moved borehole |

## Requirements

- Python 3.8+
- Flask, Flask-CORS
- Shapely, GeoPandas, PyProj
- (Frontend uses CDN-hosted Leaflet)

## Configuration

Zone spacing is defined in `shapefile_config.py`:
- Embankment zones: 100m max_spacing
- Highway zones: 200m max_spacing

Coverage is computed using the **target zone's** spacing requirement, not the borehole's origin zone.

## Performance Optimization

### Delta Updates Pattern (February 2026)

The application uses a **lazy stats update pattern** for responsive UI:

**Problem:** Original implementation computed coverage stats synchronously on every operation:
- Delete/Add/Move → Compute stats (~230ms) → Return response
- Result: 230ms+ latency felt sluggish

**Solution:** Separate visual updates from stats computation:
1. **Visual operations return immediately** (~3-5ms)
   - Delete: Remove borehole marker + coverage polygon from map
   - Add/Move: Update marker position + single coverage polygon
2. **Stats computed lazily** (~200-250ms, non-blocking)
   - Frontend requests stats after 50ms delay
   - Stats panel updates while user continues working

**Performance Comparison:**
| Operation | Before | After  | Improvement |
| --------- | ------ | ------ | ----------- |
| Delete    | ~230ms | ~3ms   | **77x**     |
| Add       | ~250ms | ~50ms  | **5x**      |
| Move      | ~200ms | ~200ms | -           |

**Architecture Components:**

1. **Server-side caching** (`geometry_service.py`):
   - `CachedCoverage` dataclass stores BNG geometries per borehole
   - `_coverage_cache: Dict[str, CachedCoverage]` - indexed by borehole_id
   - `_stats_dirty` flag tracks when stats need recomputation
   - `invalidate_cache(borehole_id)` removes single entry on delete

2. **Lazy stats endpoint** (`server.py`):
   - Delete returns `stats_pending: true` instead of computing stats
   - Frontend fetches `/api/coverage/stats` separately

3. **Frontend delta handling** (`index.html`):
   - `removeCoverageByIndex()` - efficiently removes single coverage + re-indexes
   - `setTimeout(() => refreshCoverageStats(), 50)` - deferred stats update

**Future Optimization Opportunities:**
- [ ] Incremental stats: Update stats by subtracting removed coverage area (avoid full recompute)
- [ ] WebSocket push: Server computes stats in background, pushes to client when ready
- [ ] Pre-computed zone intersections: Cache coverage-zone intersections for fast stats
- [ ] Batch operations: Group multiple deletes into single stats recompute

## Borehole Filtering System

### Filter-Agnostic Visibility Architecture

The application uses a **generic visibility detection pattern** that automatically works with any current or future filter:

**Key Principle:** Borehole count and CSV export check which markers are actually on the map, not which filters are active.

**Helper Functions** (in `templates/index.html`):
```javascript
getVisibleBoreholeIndices()   // Returns indices of boreholes currently on map
getHiddenBoreholeIndices()    // Returns indices of hidden boreholes  
updateBoreholeCountDisplay()  // Updates "visible/total" display
```

### Adding New Filters

When implementing new filters, follow this pattern to ensure the count and export work automatically:

**✅ CORRECT - Use Leaflet layer management:**
```javascript
// To HIDE a borehole:
grabCircle.markerGroup.remove();
// OR
boreholesLayer.removeLayer(grabCircle.markerGroup);

// To SHOW a borehole:
boreholesLayer.addLayer(grabCircle.markerGroup);

// Then update count:
updateBoreholeCountDisplay();
```

**❌ WRONG - These approaches will NOT work:**
- CSS visibility (`opacity: 0`, `display: none`) - marker still "on map"
- Moving markers off-screen - marker still "on map"
- Separate tracking without layer removal - markers still "on map"

**Reference Implementation:** See `updateBoreholeVisibilityForZones()` in `templates/index.html`

### Current Filters

| Filter          | Description                          | Location                             |
| --------------- | ------------------------------------ | ------------------------------------ |
| Zone Visibility | Hide/show boreholes by zone checkbox | `updateBoreholeVisibilityForZones()` |

### CSV Export Behavior

The export endpoint accepts an `excludeIndices` query parameter with comma-separated borehole indices to exclude. The frontend automatically passes hidden borehole indices when exporting, ensuring only visible boreholes are exported.

## Single Source of Truth (SSOT) Architecture

### Overview

Zone-borehole associations are maintained as a **server-authoritative single source of truth** via the `zone_ids` property. This enables instant UI feedback without client-side geometry tests.

### Architecture Layers

```
┌────────────────────────────────────────────────────────────────┐
│                   DATA LOADER (Authoritative)                   │
│                      data_loader.py                             │
├────────────────────────────────────────────────────────────────┤
│  _compute_borehole_zone_ids()     - Initial computation         │
│  update_borehole_position()       - Incremental update          │
│                                                                 │
│  Both use identical: zone_geom.intersects(bh_point)             │
│                           │                                     │
│                           ▼                                     │
│              zone_ids: ["Zone_0", "Zone_3"]                     │
└────────────────────────────────────────────────────────────────┘
                           │
                           │ /api/boreholes, /api/coverage/update
                           ▼
┌────────────────────────────────────────────────────────────────┐
│                      FRONTEND (Consumer Only)                   │
│                      templates/index.html                       │
├────────────────────────────────────────────────────────────────┤
│  grabCircle.zoneIds = feature.properties.zone_ids               │
│                                                                 │
│  isBoreholeVisibleByZones(zoneIds):                             │
│    • Returns true if NO containing zones are hidden             │
│    • Never computes geometry - only reads zone_ids              │
│                                                                 │
│  toggleZoneVisibility(zoneName, visible):                       │
│    • Pure state lookup - instant, no async, no server calls     │
└────────────────────────────────────────────────────────────────┘
```

### Data Flow

**Initial Load:**
```
zone_coverage_data.json
        │
        ▼
DataLoader._load_from_json()
        │
        ▼
_compute_borehole_zone_ids()  ← SSOT computation
        │
        │  For each borehole:
        │  └─ For each zone: if zone.intersects(point)
        │                    → add zone_name to zone_ids
        ▼
_boreholes_gdf['zone_ids'] = [["Zone_0"], ["Zone_1"], ...]
        │
        │  GET /api/boreholes
        ▼
Frontend stores: grabCircle.zoneIds = feature.properties.zone_ids
```

**Update Flow (Move Borehole):**
```
[User drags marker]
        │
        ▼
POST /api/coverage/update { index, lon, lat }
        │
        ├─► data_loader.update_borehole_position()
        │       └─► Re-computes zone_ids using same intersects() logic
        │
        ├─► Returns { zone_ids: [...], coverage: {...} }
        │
[Frontend receives response]
        │
        ▼
marker.zoneIds = response.zone_ids  ← SSOT sync
        │
        ▼
isBoreholeVisibleByZones() re-evaluated if zones hidden
```

### Key Components

| Component                      | File             | Responsibility               |
| ------------------------------ | ---------------- | ---------------------------- |
| `_compute_borehole_zone_ids()` | `data_loader.py` | Initial zone assignment      |
| `update_borehole_position()`   | `data_loader.py` | Re-compute on move           |
| `get_boreholes_geojson()`      | `data_loader.py` | Include zone_ids in response |
| `isBoreholeVisibleByZones()`   | `index.html`     | Visibility check (read-only) |
| `toggleZoneVisibility()`       | `index.html`     | Instant toggle (no server)   |

### Visibility Rule (Updated February 2026)

**New Requirements:**
1. **R1 (Coverage-Based Visibility):** If a borehole's COVERAGE touches ANY visible zone, the borehole stays visible (even if its point is in a hidden zone)
2. **R2 (Hidden Zone Ignorance):** New/moved boreholes should ignore hidden zones for zone assignment

**Key Insight - Two Zone Concepts:**
| Concept             | Source                    | Used For                                |
| ------------------- | ------------------------- | --------------------------------------- |
| `zone_ids`          | Point containment test    | Zone transitions, outside-zone coloring |
| `coverage_zone_ids` | Buffer-intersection zones | Visibility filtering                    |

**Implementation (Hybrid A+B):**
```javascript
function isBoreholeVisibleByZones(coverageZoneIds) {
    // R1: Visible if coverage touches ANY visible zone
    if (!coverageZoneIds || coverageZoneIds.length === 0) return true;
    return coverageZoneIds.some(zid => zoneVisibility[zid] !== false);
}
```

**Server Support for R2:**
- `_compute_borehole_zone_ids(exclude_zones)` - excludes hidden zones from assignment
- When moving/adding with hidden zones, server treats those zones as non-existent

### Benefits

| Benefit         | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| **Instant UI**  | No async calls for zone toggle - pure state lookup           |
| **Consistency** | Server uses same `intersects()` for initial load and updates |
| **No Drift**    | Frontend never computes zone associations                    |
| **Predictable** | Zone changes during drag are detected server-side            |

### Zone Change Detection

When a borehole is moved, the server logs zone transitions:
```
[ZONE CHANGE] BLACK->ORANGE | old_zones=["Zone_0"] -> new_zones=[]
```

Color semantics:
- **BLACK (in-zone)**: `zone_ids.length > 0` 
- **ORANGE (outside)**: `zone_ids.length === 0`

### Implementation Checklist

For maintainers extending the system:

- [x] Server computes zone_ids at load time
- [x] zone_ids included in `/api/boreholes` response
- [x] `update_borehole_position()` returns updated zone_ids
- [x] Frontend stores zone_ids on marker objects
- [x] `toggleZoneVisibility()` uses zone_ids (no geometry tests)
- [x] Coverage visibility derives from parent borehole
- [x] Old async function deprecated as no-op
