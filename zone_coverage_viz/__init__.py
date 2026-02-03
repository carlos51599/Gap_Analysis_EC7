#!/usr/bin/env python3
"""
Zone Coverage Visualization Module

Interactive visualization for zone-aware borehole coverage with
draggable markers and real-time coverage updates.

Usage:
    python -m zone_coverage_viz.server [data_dir]

Or from Python:
    from zone_coverage_viz.server import main
    main()
"""

from zone_coverage_viz.data_loader import DataLoader
from zone_coverage_viz.geometry_service import CoverageService

__all__ = ["DataLoader", "CoverageService"]
