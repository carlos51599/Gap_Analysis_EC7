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
