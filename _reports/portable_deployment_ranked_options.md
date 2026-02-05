# Portable Deployment Options - Ranked by Recommendation

## Quick Reference (Ranked Best → Worst)

| Rank | Option          | Interactivity | Size       | Dev Time  | Likely to Work? |
| ---- | --------------- | ------------- | ---------- | --------- | --------------- |
| ⭐1   | PyInstaller     | ✅ Full        | 50-150 MB  | 1-2 days  | 60-70%          |
| 2    | Embedded Python | ✅ Full        | 100-200 MB | 1-2 days  | 50-60%          |
| 3    | HTA Wrapper     | ⚠️ Limited     | 5-20 MB    | 1 day     | 40-50%          |
| 4    | Static HTML     | ❌ View-only   | 1-5 MB     | 1 day     | 99%             |
| 5    | Pure JS Rewrite | ✅ Full        | 2-10 MB    | 2-4 weeks | 99%             |

---

## ⭐ Rank 1: PyInstaller Bundle (BEST SOLUTION)

**What it is:** Single `.exe` file containing Python + all dependencies

**Interactivity:** ✅ **FULL** - drag boreholes, export CSV, toggle zones, real-time recalculation

**How it works:**
1. User double-clicks `START.bat`
2. Browser opens to `http://127.0.0.1:5051`
3. Executable runs Flask server in background
4. User experience identical to current development version

**Pros:**
- Single clean executable
- All dependencies bundled (Shapely, GeoPandas, Flask)
- Fast startup
- Identical functionality to current app

**Cons:**
- May be blocked by Windows SmartScreen ("Unknown publisher")
- May be blocked by corporate antivirus
- May be blocked by Group Policy

**What to test:**
- Double-click a simple `.exe` file
- If SmartScreen appears, click "More info" → "Run anyway"
- If blocked by policy, try right-click → Properties → Unblock

**Workarounds if blocked:**
- IT can whitelist the executable
- Code signing certificate (~$200/year) removes SmartScreen warning

---

## Rank 2: Embedded Python (Portable Python)

**What it is:** Full Python runtime + dependencies in a folder

**Interactivity:** ✅ **FULL** - same as PyInstaller

**How it works:**
1. User double-clicks `START.bat`
2. Batch file sets `PATH` to portable Python folder
3. Runs `python.exe server.py`
4. Browser opens

**Folder structure:**
```
zone_coverage_portable/
├── python/
│   ├── python.exe
│   └── Lib/site-packages/    ← All dependencies
├── zone_coverage_viz/        ← Application code
├── Output/                   ← Data files
└── START.bat
```

**Pros:**
- Full Python environment
- Easier to update/debug
- No compilation needed

**Cons:**
- Larger size (more files)
- `python.exe` subject to same restrictions as any .exe
- Messier folder structure

**What to test:**
- Same as Rank 1 - if `.exe` runs, this will too

---

## Rank 3: HTA (HTML Application)

**What it is:** HTML file with `.hta` extension that runs in Windows' built-in `mshta.exe`

**Interactivity:** ⚠️ **LIMITED** - View + some JavaScript, no Python backend

**How it works:**
1. User double-clicks `ZoneCoverageViewer.hta`
2. Windows opens in `mshta.exe` (built-in Windows component)
3. Can display map and pre-computed data
4. Cannot run Python (no server)

**Pros:**
- Uses built-in Windows executable (`mshta.exe`)
- No downloaded executable
- Some enhanced permissions vs plain HTML

**Cons:**
- Cannot run Python/Flask - view-only unless geometry ported to JS
- Often blocked by corporate Group Policy (security risk)
- VBScript/JScript limitations

**What to test:**
- Double-click a simple `.hta` file
- If it opens → HTA is allowed
- If security warning or block → HTA disabled by policy

---

## Rank 4: Static HTML Bundle (Guaranteed to Work)

**What it is:** Pre-computed data + Leaflet map in pure HTML/JavaScript

**Interactivity:** ❌ **VIEW-ONLY**
- ✅ View zones, boreholes, coverage circles on map
- ✅ Pan, zoom, toggle layers
- ✅ Zone visibility checkboxes (filter pre-computed data)
- ❌ NO drag-and-drop
- ❌ NO export to CSV
- ❌ NO real-time recalculation

**How it works:**
1. User double-clicks `index.html` (or `VIEW.bat`)
2. Opens in default browser
3. Displays pre-computed coverage visualization

**Pros:**
- Zero chance of being blocked
- Works on any machine with browser
- Tiny file size
- No server, no antivirus concerns

**Cons:**
- Static snapshot only
- Must regenerate if positions change
- No interactivity beyond viewing

**Use case:** Sharing a "snapshot" of current coverage analysis with colleagues who just need to view it.

---

## Rank 5: Pure JavaScript Rewrite (Long-term Best)

**What it is:** Port all Python geometry logic to Turf.js (JavaScript library)

**Interactivity:** ✅ **FULL** - runs entirely in browser

**How it works:**
1. User double-clicks `index.html`
2. Turf.js does all geometry calculations client-side
3. Full interactivity without any server

**Pros:**
- Full interactivity (drag, export, recalculate)
- Zero chance of being blocked
- Works offline
- No server, no Python

**Cons:**
- **2-4 weeks development effort** to port Python geometry code
- Browser performance limits with large datasets
- Potential accuracy differences vs Shapely

**Verdict:** Best long-term solution if interactivity is essential and all .exe options fail.

---

## Testing Strategy

Test in order of recommendation:

```
TEST 1: PyInstaller .exe
├── Works → USE THIS (Rank 1) ✅
└── Blocked → Test 2

TEST 2: Embedded Python
├── Works → USE THIS (Rank 2)
└── Blocked → Test 3

TEST 3: HTA file
├── Works → Limited use (Rank 3)
└── Blocked → Test 4

TEST 4: Static HTML
└── Always works → USE IF VIEW-ONLY OK (Rank 4)

If interactivity essential and all blocked:
→ Consider Rank 5 (JS rewrite) or IT-hosted server
```

---

## What I Can Create for Testing

Tell me which you want first:

1. **PyInstaller test exe** - Simple "Hello from PyInstaller" window
2. **Embedded Python test** - Folder with `python.exe` + simple script
3. **HTA test** - Simple `.hta` file with a button
4. **Static HTML test** - Leaflet map with sample zones/boreholes

Just say "create test 1" (or 2, 3, 4) and I'll make it.
