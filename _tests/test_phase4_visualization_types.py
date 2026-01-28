#!/usr/bin/env python3
"""
Phase 4 Tests: Visualization Type Compatibility

Tests that visualization functions accept both:
1. Dict[str, Any] (backwards compatibility)
2. Typed config objects (VisualizationConfig, BoreholeMarkerConfig)

These tests verify the Union type support added in Phase 4.
"""

import pytest
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

from Gap_Analysis_EC7.config_types import (
    VisualizationConfig,
    BoreholeMarkerConfig,
    ProposedMarkerConfig,
)
from Gap_Analysis_EC7.visualization.plotly_traces import (
    build_boreholes_trace,
    build_proposed_boreholes_trace,
    add_boreholes_trace,
    _normalize_borehole_marker_config,
    _normalize_proposed_marker_config,
)
from Gap_Analysis_EC7.visualization.html_builder import (
    _normalize_visualization_config,
    _get_borehole_marker_from_config,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_boreholes_gdf():
    """Create sample boreholes GeoDataFrame for testing."""
    data = {
        "LocationID": ["BH001", "BH002", "BH003"],
        "Final Depth": [15.0, 20.0, 25.0],
        "geometry": [
            Point(450000, 175000),
            Point(450100, 175100),
            Point(450200, 175200),
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:27700")


@pytest.fixture
def borehole_marker_dict():
    """Borehole marker config as dictionary (old format)."""
    return {
        "size": 8,
        "color": "rgba(0, 0, 0, 0.8)",
        "symbol": "circle",
        "line_color": "white",
        "line_width": 1,
    }


@pytest.fixture
def borehole_marker_typed():
    """Borehole marker config as typed dataclass (new format)."""
    return BoreholeMarkerConfig(
        size=8,
        color="rgba(0, 0, 0, 0.8)",
        symbol="circle",
        line_color="white",
        line_width=1,
    )


@pytest.fixture
def proposed_marker_dict():
    """Proposed marker config as dictionary (old format)."""
    return {
        "size": 10,
        "color": "blue",
        "symbol": "x",
        "buffer_color": "rgba(0, 0, 255, 0.2)",
    }


@pytest.fixture
def proposed_marker_typed():
    """Proposed marker config as typed dataclass (new format)."""
    return ProposedMarkerConfig(
        size=10,
        color="blue",
        symbol="x",
        buffer_color="rgba(0, 0, 255, 0.2)",
    )


@pytest.fixture
def visualization_config_dict():
    """Visualization config as dictionary (old format)."""
    return {
        "show_coverage_stats_panel": True,
        "borehole_marker": {
            "size": 6,
            "color": "rgba(0, 0, 0, 0.82)",
            "symbol": "circle",
        },
        "proposed_marker": {
            "size": 8,
            "color": "rgba(0, 85, 255, 0.769)",
            "symbol": "x",
        },
        "figure_width": 1200,
        "figure_height": 900,
    }


@pytest.fixture
def visualization_config_typed():
    """Visualization config as typed dataclass (new format)."""
    return VisualizationConfig(
        show_coverage_stats_panel=True,
        borehole_marker=BoreholeMarkerConfig(
            size=6,
            color="rgba(0, 0, 0, 0.82)",
            symbol="circle",
        ),
        figure_width=1200,
        figure_height=900,
    )


# ============================================================================
# NORMALIZATION HELPER TESTS
# ============================================================================


class TestBoreholeMarkerNormalization:
    """Tests for _normalize_borehole_marker_config helper."""

    def test_normalize_from_dict(self, borehole_marker_dict):
        """Dict input should be converted to BoreholeMarkerConfig."""
        result = _normalize_borehole_marker_config(borehole_marker_dict)
        assert isinstance(result, BoreholeMarkerConfig)
        assert result.size == 8
        assert result.color == "rgba(0, 0, 0, 0.8)"
        assert result.symbol == "circle"

    def test_normalize_from_typed(self, borehole_marker_typed):
        """Typed input should pass through unchanged."""
        result = _normalize_borehole_marker_config(borehole_marker_typed)
        assert result is borehole_marker_typed  # Same object
        assert isinstance(result, BoreholeMarkerConfig)


class TestProposedMarkerNormalization:
    """Tests for _normalize_proposed_marker_config helper."""

    def test_normalize_from_dict(self, proposed_marker_dict):
        """Dict input should be converted to ProposedMarkerConfig."""
        result = _normalize_proposed_marker_config(proposed_marker_dict)
        assert isinstance(result, ProposedMarkerConfig)
        assert result.size == 10
        assert result.color == "blue"
        assert result.symbol == "x"

    def test_normalize_from_typed(self, proposed_marker_typed):
        """Typed input should pass through unchanged."""
        result = _normalize_proposed_marker_config(proposed_marker_typed)
        assert result is proposed_marker_typed  # Same object


class TestVisualizationConfigNormalization:
    """Tests for _normalize_visualization_config helper."""

    def test_normalize_from_dict(self, visualization_config_dict):
        """Dict input should be converted to VisualizationConfig."""
        result = _normalize_visualization_config(visualization_config_dict)
        assert isinstance(result, VisualizationConfig)
        assert result.show_coverage_stats_panel is True
        assert result.figure_width == 1200

    def test_normalize_from_typed(self, visualization_config_typed):
        """Typed input should pass through unchanged."""
        result = _normalize_visualization_config(visualization_config_typed)
        assert result is visualization_config_typed


class TestGetBoreholeMarkerFromConfig:
    """Tests for _get_borehole_marker_from_config helper."""

    def test_extract_from_dict(self, visualization_config_dict):
        """Should extract BoreholeMarkerConfig from dict."""
        result = _get_borehole_marker_from_config(visualization_config_dict)
        assert isinstance(result, BoreholeMarkerConfig)
        assert result.size == 6

    def test_extract_from_typed(self, visualization_config_typed):
        """Should extract BoreholeMarkerConfig from typed config."""
        result = _get_borehole_marker_from_config(visualization_config_typed)
        assert isinstance(result, BoreholeMarkerConfig)
        assert result.size == 6


# ============================================================================
# PLOTLY TRACE FUNCTION TESTS
# ============================================================================


class TestBuildBoreholesTrace:
    """Tests for build_boreholes_trace Union type support."""

    def test_accepts_dict_config(self, sample_boreholes_gdf, borehole_marker_dict):
        """Function should work with dict config (backwards compatibility)."""
        trace = build_boreholes_trace(sample_boreholes_gdf, borehole_marker_dict)
        assert trace is not None
        assert trace.marker.size == 8
        assert trace.marker.color == "rgba(0, 0, 0, 0.8)"

    def test_accepts_typed_config(self, sample_boreholes_gdf, borehole_marker_typed):
        """Function should work with typed BoreholeMarkerConfig."""
        trace = build_boreholes_trace(sample_boreholes_gdf, borehole_marker_typed)
        assert trace is not None
        assert trace.marker.size == 8
        assert trace.marker.color == "rgba(0, 0, 0, 0.8)"

    def test_produces_same_result(
        self, sample_boreholes_gdf, borehole_marker_dict, borehole_marker_typed
    ):
        """Dict and typed config should produce equivalent traces."""
        trace_dict = build_boreholes_trace(sample_boreholes_gdf, borehole_marker_dict)
        trace_typed = build_boreholes_trace(sample_boreholes_gdf, borehole_marker_typed)
        assert trace_dict.marker.size == trace_typed.marker.size
        assert trace_dict.marker.color == trace_typed.marker.color


class TestBuildProposedBoreholesTrace:
    """Tests for build_proposed_boreholes_trace Union type support."""

    def test_accepts_dict_config(self, proposed_marker_dict):
        """Function should work with dict config (backwards compatibility)."""
        locations = [{"x": 450000, "y": 175000}, {"x": 450100, "y": 175100}]
        trace = build_proposed_boreholes_trace(locations, proposed_marker_dict)
        assert trace is not None
        assert trace.marker.size == 10

    def test_accepts_typed_config(self, proposed_marker_typed):
        """Function should work with typed ProposedMarkerConfig."""
        locations = [{"x": 450000, "y": 175000}, {"x": 450100, "y": 175100}]
        trace = build_proposed_boreholes_trace(locations, proposed_marker_typed)
        assert trace is not None
        assert trace.marker.size == 10


class TestAddBoreholesTrace:
    """Tests for add_boreholes_trace Union type support."""

    def test_accepts_dict_config(self, sample_boreholes_gdf, borehole_marker_dict):
        """Function should work with dict config (backwards compatibility)."""
        import plotly.graph_objects as go

        fig = go.Figure()
        count = add_boreholes_trace(fig, sample_boreholes_gdf, borehole_marker_dict)
        assert count == 1
        assert len(fig.data) == 1

    def test_accepts_typed_config(self, sample_boreholes_gdf, borehole_marker_typed):
        """Function should work with typed BoreholeMarkerConfig."""
        import plotly.graph_objects as go

        fig = go.Figure()
        count = add_boreholes_trace(fig, sample_boreholes_gdf, borehole_marker_typed)
        assert count == 1
        assert len(fig.data) == 1


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Edge case tests for type handling."""

    def test_empty_dict_uses_defaults(self):
        """Empty dict should use default values."""
        result = _normalize_borehole_marker_config({})
        assert isinstance(result, BoreholeMarkerConfig)
        assert result.size == 6  # Default value

    def test_partial_dict_fills_defaults(self):
        """Partial dict should fill missing values with defaults."""
        partial = {"size": 12}  # Only size specified
        result = _normalize_borehole_marker_config(partial)
        assert result.size == 12
        assert result.color == "rgba(0, 0, 0, 0.82)"  # Default

    def test_empty_boreholes_returns_none(self, borehole_marker_typed):
        """Empty GeoDataFrame should return None trace."""
        empty_gdf = gpd.GeoDataFrame(columns=["LocationID", "geometry"])
        trace = build_boreholes_trace(empty_gdf, borehole_marker_typed)
        assert trace is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
