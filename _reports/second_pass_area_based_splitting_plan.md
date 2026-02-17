# Implementation Plan: Area-Based Cell Splitting for Second Pass

## Goal

Remove spacing-relative cell sizing from Second Pass `check_and_split_large_cluster()`. Cell splitting decisions should be purely area-based (static thresholds), while candidate grid density inside each cell should use local zone spacing from spatial intersection with `zone_geometries`.

---

## The Two Separate Problems

### Problem 1: Cell Sizing (When and How to Split)

**Current behaviour:** `check_and_split_large_cluster()` at line 1862 computes `candidate_grid_spacing = min(zone_spacings) × 0.5`, then feeds it into `_compute_cluster_cell_thresholds()` which scales the split threshold and target cell area with `spacing²`:

```
effective_threshold = max(base_threshold, 400 × candidate_spacing²)
effective_target    = max(base_target,    200 × candidate_spacing²)
```

This means the cluster-wide minimum spacing (e.g. 50m from one small zone) inflates or deflates cell areas for the entire cluster. A 200m-only area forced through a 50m-derived threshold gets unnecessarily small cells.

**Required behaviour:** Split decision based purely on area. Use the static config values directly:
- `max_area_for_direct_ilp_m2 = 1,000,000` (1 km²) — if cluster area exceeds this, split
- `target_cell_area_m2 = 1,000,000` (1 km²) — target size for each cell after splitting

No scaling. No spacing dependency. The same thresholds regardless of which zones are in the cluster.

### Problem 2: Candidate Grid Density (How Dense the Grid Is Inside Each Cell)

**Current behaviour:** `solve_czrc_ilp_for_cluster()` at line 2997 computes `min_spacing` from ALL zones in the cluster (via `_aggregate_zone_spacings`), then passes it to `_prepare_candidates_for_ilp()` which generates a hexagonal grid at `min_spacing × 0.5`. Every cell in the cluster gets identical grid density, driven by the cluster-wide aggregated spacing.

**Required behaviour:** Each cell determines which zones spatially overlap it, computes a local min spacing from only those zones, and generates its candidate grid at that local density. A cell overlapping only 200m zones gets a 100m grid. A cell overlapping a 50m zone gets a 25m grid.

---

## Changes Required

### Change 1: Remove spacing-relative from Second Pass cell splitting

**File:** `czrc_solver.py`  
**Function:** `check_and_split_large_cluster()` (line 1817)  
**Lines:** 1865-1884

**Remove** the spacing-relative threshold computation. The split decision uses `base_max_area` directly (the static config value). The K-means target uses `base_target` directly.

Before:
```python
# Compute candidate grid spacing for this cluster (same as used in ILP)
min_zone_spacing = min(zone_spacings.values()) if zone_spacings else 100.0
candidate_mult = config.get("candidate_grid_spacing_mult", 0.5)
candidate_grid_spacing = min_zone_spacing * candidate_mult

# Apply spacing-relative scaling to thresholds
sr = cell_config.get("spacing_relative", {})
max_area, effective_target = _compute_cluster_cell_thresholds(
    candidate_grid_spacing_m=candidate_grid_spacing,
    base_threshold_m2=base_max_area,
    base_target_area_m2=base_target,
    spacing_relative_config=sr,
)
```

After:
```python
# Second Pass uses purely area-based cell splitting.
# No spacing-relative scaling — thresholds are static config values.
# (First Pass zone_auto_splitting retains spacing-relative sizing.)
max_area = base_max_area
effective_target = base_target
```

This also removes the log line about spacing-relative sizing (lines 1879-1884) and the `candidate_grid_spacing` variable that was only used for threshold computation.

**Also remove** the `_override_target_cell_area(cell_config, effective_target)` call at line 1946 — since `effective_target == base_target`, `_override_target_cell_area` still works but the spacing-relative override it was designed for is now a no-op. Can simplify to just pass `cell_config` directly, or keep as-is since the function is harmless.

### Change 2: K-means sample grid — use a fixed density

**File:** `czrc_solver.py`  
**Function:** `check_and_split_large_cluster()` (lines 1918-1940)

Currently the K-means sample grid is generated at `min_zone_spacing × 0.5`, which means a cluster with mixed 50m/200m zones gets a globally dense 25m grid for seed placement.

**Change:** Use a fixed sample grid spacing for K-means seeding that doesn't depend on zone spacing. A reasonable default: derive from the target cell area.

```python
# Fixed sample grid for K-means seeding
# Grid spacing = sqrt(target_cell_area) / 10 gives ~100 seeds per target cell
# For 1 km² target: sqrt(1e6) / 10 = 100m spacing
import math
sample_spacing = math.sqrt(effective_target) / 10.0
```

This gives uniform cell shapes regardless of zone composition. The sample grid is only for K-means seed placement — it has nothing to do with the ILP candidate grid.

### Change 3: New helper `_compute_local_zone_spacing()`

**File:** `czrc_solver.py`  
**Location:** After `_aggregate_zone_spacings()` (line ~545)  
**Size:** ~25 lines

```python
def _compute_local_zone_spacing(
    cell_geometry: BaseGeometry,
    zone_geometries: Dict[str, BaseGeometry],
    zone_spacings: Dict[str, float],
    method: str = "min",
) -> float:
    """
    Compute the controlling zone spacing for a specific cell geometry.

    Intersects cell_geometry with each zone's geometry to find which zones
    are actually present in this cell. Aggregates their spacings using the
    specified method (min/max/average).

    Falls back to min(zone_spacings.values()) if no zones overlap.
    """
    local_spacings = []
    for zone_name, zone_geom in zone_geometries.items():
        if cell_geometry.intersects(zone_geom):
            spacing = zone_spacings.get(zone_name)
            if spacing is not None:
                local_spacings.append(spacing)

    if not local_spacings:
        # Fallback: no zones overlap this cell (geometric edge case)
        return min(zone_spacings.values()) if zone_spacings else 100.0

    if method == "min":
        return min(local_spacings)
    elif method == "max":
        return max(local_spacings)
    elif method == "average":
        return sum(local_spacings) / len(local_spacings)
    return min(local_spacings)
```

### Change 4: Plumb `zone_geometries` into the call chain

**File:** `czrc_solver.py`

**4a. `run_czrc_optimization()` (line 3350):**  
Extract `zone_geometries` from `czrc_data` and pass to `check_and_split_large_cluster()`.

```python
# After line ~3397 (where zone_spacings is extracted)
zone_geometries_raw = czrc_data.get("zone_geometries", {})
zone_geometries = {}
for name, geom_or_wkt in zone_geometries_raw.items():
    if isinstance(geom_or_wkt, str):
        zone_geometries[name] = wkt.loads(geom_or_wkt)
    else:
        zone_geometries[name] = geom_or_wkt
```

At line ~3447, add `zone_geometries` to the call:
```python
check_and_split_large_cluster(
    cluster, zone_spacings, ...,
    zone_geometries=zone_geometries,  # NEW
)
```

**4b. `check_and_split_large_cluster()` (line 1817):**  
Add parameter:
```python
zone_geometries: Optional[Dict[str, BaseGeometry]] = None,
```

**4c. `solve_czrc_ilp_for_cluster()` (line 2898):**  
Add parameter:
```python
override_min_spacing: Optional[float] = None,
```

### Change 5: Per-cell local spacing in the cell processing loop

**File:** `czrc_solver.py`  
**Function:** `check_and_split_large_cluster()`  
**Location:** Lines 1971-1985 (the per-cell loop)

Before each `solve_czrc_ilp_for_cluster()` call, compute local spacing:

```python
for i, cell_geom in enumerate(cells):
    cell_cluster = _create_cell_cluster(cell_geom, cluster, i)

    # Compute local zone spacing for this cell
    local_spacing = None
    if zone_geometries:
        exclusion_method = _get_cross_zone_exclusion_method(config)
        local_spacing = _compute_local_zone_spacing(
            cell_geom, zone_geometries, zone_spacings, exclusion_method
        )
        log.info(
            f"   📐 Cell {i}: local_spacing={local_spacing:.0f}m"
        )

    selected, removed, added, stats = solve_czrc_ilp_for_cluster(
        cell_cluster, zone_spacings, ...,
        override_min_spacing=local_spacing,  # NEW
    )
```

### Change 6: `solve_czrc_ilp_for_cluster()` uses override when provided

**File:** `czrc_solver.py`  
**Function:** `solve_czrc_ilp_for_cluster()` (line 2898)  
**Location:** Lines 2997-3005

```python
# Current:
all_zones = set()
for pk in pair_keys:
    all_zones.update(parse_pair_key(pk, zone_spacings))
exclusion_method = _get_cross_zone_exclusion_method(config)
min_spacing = _aggregate_zone_spacings(
    zone_spacings, list(all_zones), exclusion_method
)

# New:
if override_min_spacing is not None:
    min_spacing = override_min_spacing
else:
    all_zones = set()
    for pk in pair_keys:
        all_zones.update(parse_pair_key(pk, zone_spacings))
    exclusion_method = _get_cross_zone_exclusion_method(config)
    min_spacing = _aggregate_zone_spacings(
        zone_spacings, list(all_zones), exclusion_method
    )
```

When called from the per-cell loop with `override_min_spacing`, the local value is used for `_prepare_candidates_for_ilp()` grid density. When called directly (non-split clusters), the existing aggregation logic applies unchanged.

### Change 7: Third Pass `min_grid_spacing` uses local spacing

**File:** `czrc_solver.py`  
**Function:** `check_and_split_large_cluster()`  
**Location:** Lines 2126-2139

Third Pass does NOT split cells — it processes adjacencies between cells already created by Second Pass. The only thing it needs is `min_grid_spacing` for candidate grid density in `solve_cell_cell_czrc()`.

Currently it aggregates across ALL cluster zones. Change to use `zone_geometries` intersection with `unified_tier1`:

```python
# Current:
min_grid_spacing = _aggregate_zone_spacings(
    zone_spacings, list(all_cluster_zones), exclusion_method
)

# New:
if zone_geometries:
    min_grid_spacing = _compute_local_zone_spacing(
        unified_tier1, zone_geometries, zone_spacings, exclusion_method
    )
else:
    min_grid_spacing = _aggregate_zone_spacings(
        zone_spacings, list(all_cluster_zones), exclusion_method
    )
```

Note: this is still cluster-level, not per-pair. For a first iteration this is sufficient — the Third Pass boundary regions are small geographically and the local spacing of the cluster's unified_tier1 is already more accurate than the global min. Per-pair local spacing (intersecting the CZRC region with zone_geometries) could be a follow-up.

### Change 8: Config cleanup

**File:** `config.py`  
**Location:** Lines 817-823 (cell_splitting.spacing_relative)

Either:
- **Option A:** Remove the `spacing_relative` block from `cell_splitting` config entirely
- **Option B:** Set `"enabled": False` and add a comment explaining why

**Recommendation:** Option B — keeps the config structure visible for reference and makes the change reversible.

```python
"spacing_relative": {
    "enabled": False,  # Disabled: Second Pass uses purely area-based splitting.
    # First Pass zone_auto_splitting retains spacing-relative sizing.
    "cell_area_multiplier": 200,
    "threshold_multiplier": 400,
},
```

---

## What Changes vs What Stays

| Component | Before | After |
|-----------|--------|-------|
| Second Pass split threshold | `max(1M, 400 × s²)` — spacing-dependent | `1,000,000 m²` — static |
| Second Pass target cell area | `max(1M, 200 × s²)` — spacing-dependent | `1,000,000 m²` — static |
| K-means sample grid spacing | `min(zone_spacings) × 0.5` — cluster-wide min | `sqrt(target) / 10` — fixed from area |
| Per-cell candidate grid density | Cluster-wide `_aggregate_zone_spacings()` | `_compute_local_zone_spacing()` per cell |
| Third Pass candidate grid density | Cluster-wide `_aggregate_zone_spacings()` | `_compute_local_zone_spacing()` for unified_tier1 |
| First Pass zone_auto_splitting | Spacing-relative (unchanged) | Spacing-relative (unchanged) |
| Coverage radius logic | Per-test-point `required_radius` | Per-test-point `required_radius` (unchanged) |
| `_build_coverage_dict_variable_test_radii()` | Unchanged | Unchanged |
| Config `cell_splitting.spacing_relative` | `enabled: True` | `enabled: False` |
| Config `zone_auto_splitting.spacing_relative` | `enabled: True` | `enabled: True` (unchanged) |

---

## Files Modified

| File | What Changes |
|------|-------------|
| `czrc_solver.py` | New `_compute_local_zone_spacing()` (~25 lines), remove spacing-relative from `check_and_split_large_cluster()`, add `zone_geometries` parameter plumbing, per-cell local spacing, `override_min_spacing` in `solve_czrc_ilp_for_cluster()`, Third Pass local spacing |
| `config.py` | Set `cell_splitting.spacing_relative.enabled = False` |

---

## Sequence of Implementation

```
Step 1: Remove spacing-relative from check_and_split_large_cluster()
        - Delete threshold computation lines 1865-1884
        - Use base_max_area and base_target directly
        - Fix K-means sample grid to use area-derived spacing
Step 2: Disable spacing_relative in config.py
Step 3: Write _compute_local_zone_spacing() function
Step 4: Plumb zone_geometries from run_czrc_optimization() → check_and_split_large_cluster()
Step 5: Add override_min_spacing to solve_czrc_ilp_for_cluster()
Step 6: Per-cell local spacing in the cell loop
Step 7: Third Pass local spacing
Step 8: Add logging for local spacing per cell
Step 9: Run full pipeline and compare output
```

---

## Testing Strategy

### 1. Unit test: `_compute_local_zone_spacing()`
- Cell overlapping one zone → returns that zone's spacing
- Cell overlapping two zones → returns min (when method="min")
- Cell overlapping no zones → fallback

### 2. Regression test: single-zone cluster
- All zones same spacing → identical behaviour to current
- Cell count unchanged, borehole count unchanged

### 3. Multi-zone test
- Create cluster with 50m and 200m zones
- Verify cell count is based on area (not spacing)
- Verify cells in 200m-only areas get coarser candidate grid
- Verify cells in 50m areas get dense candidate grid

### 4. Config test
- Verify `spacing_relative.enabled = False` in cell_splitting
- Verify `spacing_relative.enabled = True` still in zone_auto_splitting (unchanged)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `zone_geometries` missing from `czrc_data` | Low | Medium | Fallback to cluster-wide aggregation when None |
| Tangent-only intersection false positives | Low | Low | Accept — `intersects()` is correct for "which zones touch this cell" |
| Too many cells for large-spacing clusters (1 km² static threshold) | Medium | Low | This is correct — area-based means more cells in large clusters, but each cell has sparse grid. Total candidate count may decrease. |
| K-means seeding at fixed density produces bad cell shapes | Low | Low | 100m sample spacing gives ~100 seeds per km² — plenty for good Voronoi cells |

---

## Code Location Reference

| Component | File | Line |
|-----------|------|------|
| `_aggregate_zone_spacings()` | czrc_solver.py | 506 |
| `_compute_cluster_cell_thresholds()` | czrc_solver.py | 1761 |
| `check_and_split_large_cluster()` | czrc_solver.py | 1817 |
| Spacing-relative computation (REMOVE) | czrc_solver.py | 1865-1884 |
| K-means sample grid (FIX) | czrc_solver.py | 1918-1940 |
| Per-cell ILP call | czrc_solver.py | 1977 |
| Third Pass `min_grid_spacing` | czrc_solver.py | 2126-2139 |
| `solve_czrc_ilp_for_cluster()` spacing | czrc_solver.py | 2997-3005 |
| `run_czrc_optimization()` | czrc_solver.py | 3350 |
| Call to `check_and_split_large_cluster` | czrc_solver.py | 3447 |
| `zone_geometries` source | czrc_geometry.py | 398-401 |
| `_prepare_candidates_for_ilp()` | czrc_solver.py | 1003 |
| Config `cell_splitting` | config.py | 807-840 |
| Config `zone_auto_splitting` | config.py | 758-800 |
