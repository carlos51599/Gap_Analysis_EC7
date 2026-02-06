"""Data models package for typed borehole and optimization data structures."""

from .data_models import (
    Borehole,
    BoreholePass,
    BoreholeStatus,
    PassResult,
    # Batch conversion utilities
    boreholes_from_dicts,
    boreholes_to_dicts,
    # Duck-typed accessor functions
    get_bh_coords,
    get_bh_position,
    get_bh_radius,
    get_bh_source_pass,
)

from .cluster_stats import (
    CellStats,
    ClusterStats,
    ThirdPassPairStats,
    ThirdPassStats,
)

__all__ = [
    # Borehole models
    "Borehole",
    "BoreholePass",
    "BoreholeStatus",
    "PassResult",
    # Batch conversion utilities
    "boreholes_from_dicts",
    "boreholes_to_dicts",
    # Duck-typed accessor functions
    "get_bh_coords",
    "get_bh_position",
    "get_bh_radius",
    "get_bh_source_pass",
    # Cluster stats models
    "CellStats",
    "ClusterStats",
    "ThirdPassPairStats",
    "ThirdPassStats",
]
