# Portable Deployment Options for zone_coverage_viz

## Goal
Create a portable folder that can:
- Be zipped and shared with non-technical colleagues
- Run on restricted work laptops (**no admin rights, cannot run .exe files**)
- Require no installations
- View in a standard browser

## Constraint: Restricted Work Laptops
- ❌ Cannot run .exe files
- ❌ Cannot install software
- ❌ No admin rights
- ✅ Can open HTML files in browser
- ✅ Can access network shares (maybe)

This eliminates PyInstaller, embedded Python, and Tauri options.

---

## Viable Options for Restricted Environments

### Option A: Pre-Computed Static HTML (View-Only)

**Approach:** Pre-compute all data server-side, generate a self-contained HTML file

**Structure:**
```
zone_coverage_viewer/
├── index.html          # Self-contained map with embedded data
├── data/
│   ├── zones.geojson
│   ├── boreholes.geojson
│   └── coverages.geojson   # Pre-computed coverages
└── VIEW.bat            # Just opens the HTML file
```

**VIEW.bat:**
```batch
@echo off
start "" "%~dp0index.html"
```

**Capabilities:**
- ✅ View zones, boreholes, coverages on map
- ✅ Pan, zoom, layer toggles
- ✅ Zone visibility checkboxes (client-side filtering of pre-computed data)
- ❌ No drag-and-drop (would need server for recomputation)
- ❌ No export (would need server)

**Pros:**
- ✅ Zero dependencies - just double-click HTML or .bat
- ✅ Works on any restricted laptop
- ✅ Tiny file size (1-5 MB)
- ✅ No antivirus issues
- ✅ Can run from email attachment or USB

**Cons:**
- ❌ Static data (no live editing)
- ❌ If borehole positions change, must regenerate

**Size:** 1-5 MB
**Complexity:** Low
**Interactivity:** View-only

---

### Option B: Internal Web Server (IT Hosted)

**Approach:** IT department hosts Flask server on internal network

**Access:**
```
http://internal-server.company.com:5051
```

**Pros:**
- ✅ Full interactivity (drag, export, etc.)
- ✅ No installation on user laptops
- ✅ Central data management
- ✅ Updates automatic

**Cons:**
- ⚠️ Requires IT involvement
- ⚠️ Only works on company network (or VPN)
- ⚠️ Dependency on IT for updates

**Size:** N/A (server-hosted)
**Complexity:** Medium (IT coordination)
**Interactivity:** Full

---

### Option C: SharePoint/OneDrive Hosted Static Site

**Approach:** Host static files on SharePoint/Teams/OneDrive

**Structure:**
```
SharePoint > Sites > SESRO > zone_coverage_viewer/
├── index.html
└── data/*.geojson
```

**Access:**
- Users navigate to SharePoint folder
- Click index.html to view

**Pros:**
- ✅ Already part of company infrastructure
- ✅ Easy to update (just upload new files)
- ✅ Access control via SharePoint permissions

**Cons:**
- ⚠️ Some organizations block JavaScript in SharePoint HTML
- ⚠️ May need to configure CORS or permissions
- ❌ Still view-only (no server)

**Size:** 1-5 MB
**Complexity:** Low-Medium
**Interactivity:** View-only

---

### Option D: Pure JavaScript Client-Side (Full Interactivity)

**Approach:** Port geometry computations to JavaScript (Turf.js)

**What would need to change:**
1. Replace Shapely/GeoPandas with Turf.js (JavaScript library)
2. Compute coverage circles client-side
3. Perform zone intersection client-side
4. All data processing in browser

**Structure:**
```
zone_coverage_interactive/
├── index.html          # Full application
├── js/
│   ├── turf.min.js     # Geometry library
│   └── app.js          # Application logic
└── data/*.geojson
```

**Pros:**
- ✅ Full interactivity (drag, drop, export)
- ✅ No server needed
- ✅ Works offline
- ✅ Just double-click HTML

**Cons:**
- ⚠️ Significant development effort (port Python → JS)
- ⚠️ Browser performance limits with large datasets
- ⚠️ Potential accuracy differences between Turf.js and Shapely

**Size:** 2-10 MB
**Complexity:** High (rewrite backend)
**Interactivity:** Full

---

## Recommendation Matrix (Restricted Laptops)

| Option         | Size    | Interactivity | Effort | Works Offline | Recommended         |
| -------------- | ------- | ------------- | ------ | ------------- | ------------------- |
| A: Static HTML | 1-5 MB  | View-only     | Low    | ✅             | ⭐ **Quick win**     |
| B: IT Server   | N/A     | Full          | Medium | ❌             | Best if IT supports |
| C: SharePoint  | 1-5 MB  | View-only     | Low    | ❌             | If SharePoint works |
| D: Pure JS     | 2-10 MB | Full          | High   | ✅             | Long-term if needed |

---

## Recommended Strategy

### Short-term: Option A (Static HTML Bundle)

1. Generate static HTML with pre-computed coverages
2. Embed data as JavaScript variables or load from local .geojson files
3. Use Leaflet for map + layer controls
4. Allow zone visibility toggles (filter pre-computed features client-side)

**User workflow:**
1. Receive zip file
2. Extract to any folder
3. Double-click `VIEW.bat` or `index.html`
4. Map opens in default browser

### Long-term: Option D if interactivity critical

If drag-and-drop and live updates are essential:
- Port geometry logic to Turf.js
- Create fully client-side application
- ~2-4 weeks development effort

---

## Static Bundle Implementation Plan

1. **Modify `main.py` output:**
   - Generate `coverages.geojson` (all boreholes' coverages pre-computed)
   - Generate `zones.geojson` (all zones)
   - Generate `boreholes.geojson` (all borehole positions)

2. **Create standalone `index.html`:**
   - Load Leaflet from CDN or embed
   - Load data files via `fetch()` or embed as variables
   - Render zones, boreholes, coverages as layers
   - Add zone visibility checkboxes (filter pre-loaded data)

3. **Package:**
   ```
   zone_coverage_viewer/
   ├── index.html
   ├── data/
   │   ├── zones.geojson
   │   ├── boreholes.geojson
   │   ├── proposed_coverages.geojson
   │   └── existing_coverage.geojson
   ├── VIEW.bat           # Opens index.html
   └── README.txt         # Instructions
   ```

4. **Test on restricted laptop**

5. **Distribute via email/SharePoint**

---

## Next Steps

1. [ ] Confirm: Is view-only acceptable, or is interactivity essential?
2. [ ] If view-only OK: Implement Option A (1-2 days)
3. [ ] If interactivity needed: Evaluate effort for Option D (Turf.js port)
4. [ ] Test on actual restricted work laptop
5. [ ] Document user instructions
