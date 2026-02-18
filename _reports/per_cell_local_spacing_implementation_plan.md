# Implementation Plan: Per-Cell Local Zone Spacing for CZRC

## Goal

Replace the current cluster-wide `min(zone_spacings)` with per-cell local zone spacing in Second and Third Pass candidate grid generation. Cell areas remain static at 2 km² (current config base values). Only grid density changes.

---

## Problem Summary

Today, `check_and_split_large_cluster()` computes a single `min_zone_spacing` from ALL zones in the cluster (line 1862). This value drives:

1. **Candidate grid density** via `candidate_grid_spacing = min_zone_spacing × 0.5` — stamped identically across every cell
2. **Spacing-relative cell thresholds** via `_compute_cluster_cell_thresholds()` — a single threshold for the whole cluster
3. **Third Pass grid density** via `_aggregate_zone_spacings()` at line 2136 — same cluster-wide aggregation

A cell that sits entirely within a 200m-spacing zone gets a 25m grid (if another zone in the cluster has 50m spacing), producing 16× more candidates than needed.

---

## Design Decision: Static Cell Sizes, Local Grid Density

**Cell areas stay at static 2 km²** (the base config values `max_area_for_direct_ilp_m2 = 1_000_000` and `target_cell_area_m2 = 1_000_000`). The spacing-relative scaling of cell thresholds is kept as-is — this plan does NOT change cell sizing. It only changes what happens INSIDE each cell: the candidate grid density.

**What changes:**
- After cells are created by K-means/Voronoi, each cell determines which zones overlap it
- Each cell's ILP solve uses a `local_min_spacing` derived only from those overlapping zones
- Third Pass similarly uses per-pair local spacing

---

## Data Already Available

`czrc_data` (built in `czrc_geometry.py` lines 398-401) contains:

```python
zone_geometries = {zone_name: Shapely_polygon}  # Raw zone boundaries
zone_spacings   = {zone_name: max_spacing_m}     # Zone spacings
```

These are passed to `run_czrc_optimization()` via the `czrc_data` dict but are NOT currently plumbed through to `check_and_split_large_cluster()` or `solve_czrc_ilp_for_cluster()`.

---

## Changes Required

### Change 1: New helper function `_compute_local_zone_spacing()`

**File:** `czrc_solver.py`  
**Location:** After `_aggregate_zone_spacings()` (line ~545), in the same section  
**Size:** ~25 lines

```
def _compute_local_zone_spacing(
    cell_geometry: BaseGeometry,
    zone_geometries: Dict[str, BaseGeometry],
    zone_spacings: Dict[str, float],
    method: str = "min",
) -> float:
```

**Logic:**
1. For each zone in `zone_geometries`, check if `cell_geometry.intersects(zone_geom)`
2. Collect spacings of overlapping zones
3. Aggregate using `method` (min/max/average — same as `_aggregate_zone_spacings`)
4. If no zones overlap (edge case), fall back to `min(zone_spacings.values())`

**Returns:** Single float — the local min spacing for this cell

This is intentionally a thin spatial filter. Shapely `.intersects()` is cheap for pre-built polygons.

---

### Change 2: Plumb `zone_geometries` into `check_and_split_large_cluster()`

**File:** `czrc_solver.py`  
**Function:** `check_and_split_large_cluster()` (line 1817)

Add parameter:
```python
zone_geometries: Optional[Dict[str, BaseGeometry]] = None,
```

**No change to cell splitting logic itself.** Cells are still created using the current cluster-wide `min_zone_spacing` for K-means sample grid and thresholds. This is correct because cell SHAPES should not depend on local spacing — they are geometric partitions of the cluster.

**After cells are created** (line ~1977), before the per-cell `solve_czrc_ilp_for_cluster()` call:

```python
# Compute local spacing for this cell
if zone_geometries:
    local_spacing = _compute_local_zone_spacing(
        cell_geom, zone_geometries, zone_spacings, exclusion_method
    )
else:
    local_spacing = None  # Fall back to cluster-wide aggregation
```

Pass `local_spacing` as a new `override_min_spacing` parameter to `solve_czrc_ilp_for_cluster()`.

---

### Change 3: Add `override_min_spacing` to `solve_czrc_ilp_for_cluster()`

**File:** `czrc_solver.py`  
**Function:** `solve_czrc_ilp_for_cluster()` (line 2898)

Add parameter:
```python
override_min_spacing: Optional[float] = None,
```

At line ~3002, where spacing is currently computed:

```python
# Current code:
all_zones = set()
for pk in pair_keys:
    all_zones.update(parse_pair_key(pk, zone_spacings))
exclusion_method = _get_cross_zone_exclusion_method(config)
min_spacing = _aggregate_zone_spacings(zone_spacings, list(all_zones), exclusion_method)

# New code:
if override_min_spacing is not None:
    min_spacing = override_min_spacing
else:
    all_zones = set()
    for pk in pair_keys:
        all_zones.update(parse_pair_key(pk, zone_spacings))
    exclusion_method = _get_cross_zone_exclusion_method(config)
    min_spacing = _aggregate_zone_spacings(zone_spacings, list(all_zones), exclusion_method)
```

This override is ONLY used for candidate grid density (passed to `_prepare_candidates_for_ilp`). The coverage radius logic in `_build_coverage_dict_variable_test_radii` still uses per-test-point `required_radius` — unchanged.

---

### Change 4: Plumb `zone_geometries` from `run_czrc_optimization()` to `check_and_split_large_cluster()`

**File:** `czrc_solver.py`  
**Function:** `run_czrc_optimization()` (line 3350)

At line ~3397, extract zone_geometries from czrc_data:

```python
zone_geometries_raw = czrc_data.get("zone_geometries", {})
# Deserialize WKT strings to Shapely geometries if needed
zone_geometries = {}
for name, geom_or_wkt in zone_geometries_raw.items():
    if isinstance(geom_or_wkt, str):
        zone_geometries[name] = wkt.loads(geom_or_wkt)
    else:
        zone_geometries[name] = geom_or_wkt
```

At line ~3447, pass to `check_and_split_large_cluster()`:

```python
selected, removed, added, cluster_stats = check_and_split_large_cluster(
    cluster,
    zone_spacings,
    all_test_points,
    first_pass_boreholes,
    config,
    zones_clip_geometry,
    logger,
    czrc_cache,
    highs_log_folder,
    cluster_idx,
    zone_geometries=zone_geometries,  # NEW
)
```

---

### Change 5: Third Pass local spacing

**File:** `czrc_solver.py`  
**Location:** Inside `check_and_split_large_cluster()`, around line 2130-2150

Currently Third Pass computes `min_grid_spacing` using cluster-wide `_aggregate_zone_spacings()`. Replace with per-pair local spacing:

For each `(cell_i, cell_j, czrc_region)` adjacency in `run_cell_czrc_pass()`, the CZRC region geometry itself determines which zones are relevant. Compute local spacing for each pair:

**Option A (simpler):** In `check_and_split_large_cluster()`, compute `min_grid_spacing` using the same local spacing logic but for the full cluster (not per-pair). This is a middle ground — better than cluster-wide min but not fully per-pair.

**Option B (precise):** In `run_cell_czrc_pass()`, for each pair, compute local spacing from the CZRC region. This requires plumbing `zone_geometries` into `run_cell_czrc_pass()` and `solve_cell_cell_czrc()`.

**Recommendation:** Option A for first iteration, Option B as follow-up if needed.

For Option A, change lines 2132-2147:
```python
# Current:
min_grid_spacing = _aggregate_zone_spacings(
    zone_spacings, list(all_cluster_zones), exclusion_method
)

# New:
if zone_geometries:
    # Use unified_tier1 as the region to determine local zones
    min_grid_spacing = _compute_local_zone_spacing(
        unified_tier1, zone_geometries, zone_spacings, exclusion_method
    )
else:
    min_grid_spacing = _aggregate_zone_spacings(
        zone_spacings, list(all_cluster_zones), exclusion_method
    )
```

---

### Change 6: Logging

Add a log line after computing local spacing in the per-cell loop:

```python
log.info(
    f"   📐 Cell {i}: local_spacing={local_spacing:.0f}m "
    f"(cluster-wide={cluster_min_spacing:.0f}m, "
    f"zones={list(local_zones)})"
)
```

This makes it visible in the output which cells got different grid densities.

---

## Files Modified

| File             | Changes                                                          | Lines Added | Lines Modified |
| ---------------- | ---------------------------------------------------------------- | ----------- | -------------- |
| `czrc_solver.py` | New `_compute_local_zone_spacing()`, modify 3 existing functions | ~25 new     | ~30 modified   |

Total: ~55 lines changed in 1 file.

---

## What Does NOT Change

| Component                                    | Why                                                                    |
| -------------------------------------------- | ---------------------------------------------------------------------- |
| Cell splitting thresholds                    | Static 2 km² — no spacing-relative cell sizing change                  |
| K-means sample grid density                  | Still uses cluster-wide min for seed placement (cell shapes unchanged) |
| Coverage radius logic                        | Still uses per-test-point `required_radius`                            |
| `_build_coverage_dict_variable_test_radii()` | Unchanged — already correct                                            |
| Config values                                | No config changes needed                                               |
| First Pass                                   | Unrelated — already uses per-zone spacing                              |
| Visualization                                | No changes — same borehole markers, same layers                        |

---

## Testing Strategy

### 1. Unit test for `_compute_local_zone_spacing()`
- Cell overlapping one zone → returns that zone's spacing
- Cell overlapping two zones → returns min of both (when method="min")
- Cell overlapping no zones → returns fallback
- Cell partially overlapping → still detects intersection

### 2. Integration test: multi-zone cluster
- Create cluster with zones at 50m and 200m spacing
- After splitting, verify cells in 200m-only area get `local_spacing=200`
- Verify cells in 50m area get `local_spacing=50`
- Verify cells spanning both get `local_spacing=50`

### 3. Regression test: single-zone cluster
- When all zones have the same spacing, behavior should be identical to current
- Compare output borehole counts before/after

### 4. Third Pass test
- Verify `min_grid_spacing` passed to `run_cell_czrc_pass()` reflects local zones

---

## Sequence of Implementation

```
Step 1: Write _compute_local_zone_spacing() + unit tests
Step 2: Add zone_geometries parameter to check_and_split_large_cluster()
Step 3: Plumb zone_geometries from run_czrc_optimization()
Step 4: Add override_min_spacing to solve_czrc_ilp_for_cluster()
Step 5: Compute local spacing per cell and pass as override
Step 6: Third Pass local spacing (Option A)
Step 7: Add logging
Step 8: Run full pipeline and compare output
```

---

## Risk Assessment

| Risk                                                          | Likelihood | Impact   | Mitigation                                                  |
| ------------------------------------------------------------- | ---------- | -------- | ----------------------------------------------------------- |
| `zone_geometries` not in `czrc_data` for some configs         | Low        | Medium   | Fallback to cluster-wide min when `zone_geometries` is None |
| Shapely `.intersects()` returns True for tangent-only contact | Low        | Low      | Filter by `intersection.area > min_area`                    |
| Some cells overlap zero zones (geometric edge cases)          | Low        | Medium   | Fallback to cluster-wide min spacing                        |
| Performance regression from per-cell intersection checks      | Very Low   | Very Low | N_zones × N_cells intersections — typically <50 total       |

---

## Appendix: Current Code Locations

| Component                               | File             | Line      |
| --------------------------------------- | ---------------- | --------- |
| `_aggregate_zone_spacings()`            | czrc_solver.py   | 506       |
| `_compute_cluster_cell_thresholds()`    | czrc_solver.py   | 1761      |
| `check_and_split_large_cluster()`       | czrc_solver.py   | 1817      |
| Cluster-wide min_zone_spacing           | czrc_solver.py   | 1862      |
| K-means sample grid                     | czrc_solver.py   | 1918-1940 |
| Per-cell ILP call                       | czrc_solver.py   | 1977      |
| Third Pass min_grid_spacing             | czrc_solver.py   | 2132-2147 |
| `run_cell_czrc_pass()`                  | czrc_solver.py   | 2632      |
| `solve_cell_cell_czrc()`                | czrc_solver.py   | 2312      |
| `solve_czrc_ilp_for_cluster()`          | czrc_solver.py   | 2898      |
| Spacing aggregation in cluster solver   | czrc_solver.py   | 2987-3002 |
| `run_czrc_optimization()`               | czrc_solver.py   | 3350      |
| Call to `check_and_split_large_cluster` | czrc_solver.py   | 3447      |
| `zone_geometries` source                | czrc_geometry.py | 398-401   |
| `_prepare_candidates_for_ilp()`         | czrc_solver.py   | 1003      |
