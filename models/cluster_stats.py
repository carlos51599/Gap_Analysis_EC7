"""
Typed dataclasses for CZRC cluster statistics.

Architectural Overview:
=======================
This module provides typed access to cluster optimization results, handling
the complexity of split vs unsplit clusters transparently. The key abstraction
is `get_tier2_geometry()` which returns the correct tier2 boundary regardless
of whether the cluster was partitioned during optimization.

Key Interactions:
-----------------
- Input: czrc_solver.py creates dicts, consumers convert via ClusterStats.from_dict()
- Output: Visualization and export code uses typed attributes and unified geometry access
- Navigation: Use VS Code outline (Ctrl+Shift+O) for quick navigation

Data Flow:
----------
1. czrc_solver.py creates cluster_stats dict with all optimization results
2. Consumers call ClusterStats.from_dict() at boundary
3. Business logic uses typed attributes and get_tier2_geometry()
4. If serialization needed, call to_dict()

MODIFICATION POINT: Add new fields here when solver output changes
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Shapely imports - handle missing gracefully for type hints
try:
    from shapely.geometry.base import BaseGeometry
    from shapely import wkt
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    BaseGeometry = Any  # type: ignore
    SHAPELY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# 📦 CELL-LEVEL DATACLASSES (Second Pass / Split Clusters)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CellStats:
    """Statistics for a single cell in a split cluster (Second Pass output).
    
    When a cluster is partitioned due to size, each cell gets its own 
    Second Pass optimization. This dataclass captures the per-cell results.
    
    Key Field:
    ----------
    tier2_wkt: The cell-specific Tier 2 boundary. For split clusters,
               the overall tier2 is the UNION of all cell tier2 geometries.
    """
    cell_index: int
    tier2_wkt: Optional[str] = None
    tier1_wkt: Optional[str] = None
    selected_count: int = 0
    boreholes_removed: int = 0
    boreholes_added: int = 0
    solve_time: float = 0.0
    candidates_count: int = 0
    precovered_count: int = 0
    area_ha: float = 0.0
    r_max: float = 0.0
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CellStats":
        """Create CellStats from solver dict output."""
        return cls(
            cell_index=d.get("cell_index", 0),
            tier2_wkt=d.get("tier2_wkt"),
            tier1_wkt=d.get("tier1_wkt"),
            selected_count=d.get("selected_count", 0),
            boreholes_removed=d.get("boreholes_removed", 0),
            boreholes_added=d.get("boreholes_added", 0),
            solve_time=d.get("solve_time", 0.0),
            candidates_count=d.get("candidates_count", 0),
            precovered_count=d.get("precovered_count", 0),
            area_ha=d.get("area_ha", 0.0),
            r_max=d.get("r_max", 0.0),
        )
    
    def get_tier2_geometry(self) -> Optional["BaseGeometry"]:
        """Parse tier2_wkt into a Shapely geometry.
        
        Returns:
            Shapely geometry or None if tier2_wkt is missing/invalid.
        """
        if not self.tier2_wkt or not SHAPELY_AVAILABLE:
            return None
        try:
            return wkt.loads(self.tier2_wkt)
        except Exception:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dict for serialization."""
        result: Dict[str, Any] = {
            "cell_index": self.cell_index,
            "selected_count": self.selected_count,
            "boreholes_removed": self.boreholes_removed,
            "boreholes_added": self.boreholes_added,
            "solve_time": self.solve_time,
            "candidates_count": self.candidates_count,
            "precovered_count": self.precovered_count,
            "area_ha": self.area_ha,
            "r_max": self.r_max,
        }
        if self.tier2_wkt:
            result["tier2_wkt"] = self.tier2_wkt
        if self.tier1_wkt:
            result["tier1_wkt"] = self.tier1_wkt
        return result


# ═══════════════════════════════════════════════════════════════════════════
# 📦 THIRD PASS DATACLASSES (Cell-Cell Pair Optimization)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ThirdPassPairStats:
    """Statistics for a single cell-cell pair in Third Pass.
    
    Third Pass optimizes the boundaries between adjacent cells,
    creating new tier2 boundaries for each cell pair.
    """
    cell_key: str = ""
    tier2_wkt: Optional[str] = None
    bh_count: int = 0
    status: str = "unknown"
    r_max: float = 0.0
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThirdPassPairStats":
        """Create from solver dict output."""
        return cls(
            cell_key=d.get("cell_key", ""),
            tier2_wkt=d.get("tier2_wkt"),
            bh_count=d.get("bh_count", 0),
            status=d.get("status", "unknown"),
            r_max=d.get("r_max", 0.0),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dict for serialization."""
        result: Dict[str, Any] = {
            "cell_key": self.cell_key,
            "bh_count": self.bh_count,
            "status": self.status,
            "r_max": self.r_max,
        }
        if self.tier2_wkt:
            result["tier2_wkt"] = self.tier2_wkt
        return result


@dataclass
class ThirdPassStats:
    """Aggregated Third Pass statistics for a split cluster.
    
    Contains results from optimizing all adjacent cell pairs.
    """
    status: str = "not_run"
    pairs_processed: int = 0
    pair_stats: List[ThirdPassPairStats] = field(default_factory=list)
    tier_geometries: Dict[str, Dict[str, str]] = field(default_factory=dict)
    third_pass_removed: List[Dict[str, Any]] = field(default_factory=list)
    third_pass_added: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThirdPassStats":
        """Create from solver dict output."""
        pair_stats = [
            ThirdPassPairStats.from_dict(ps) 
            for ps in d.get("pair_stats", [])
        ]
        return cls(
            status=d.get("status", "not_run"),
            pairs_processed=d.get("pairs_processed", 0),
            pair_stats=pair_stats,
            tier_geometries=d.get("tier_geometries", {}),
            third_pass_removed=d.get("third_pass_removed", []),
            third_pass_added=d.get("third_pass_added", []),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dict for serialization."""
        return {
            "status": self.status,
            "pairs_processed": self.pairs_processed,
            "pair_stats": [ps.to_dict() for ps in self.pair_stats],
            "tier_geometries": self.tier_geometries,
            "third_pass_removed": self.third_pass_removed,
            "third_pass_added": self.third_pass_added,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 📊 MAIN CLUSTER STATS DATACLASS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ClusterStats:
    """
    Typed container for CZRC cluster optimization statistics.
    
    Handles both split and unsplit clusters with a unified interface.
    Use `get_tier2_geometry()` to access the Tier 2 boundary regardless
    of whether the cluster was partitioned during optimization.
    
    Why This Exists:
    ----------------
    The tier2 boundary visualization bug (where split cluster tier2 wasn't 
    shown) happened because:
    - Unsplit clusters store tier2 in `stats["tier2_wkt"]`
    - Split clusters store tier2 in `stats["cell_stats"][i]["tier2_wkt"]`
    
    This dataclass provides `get_tier2_geometry()` which handles both cases
    transparently, preventing future bugs.
    
    Usage:
    ------
    ```python
    # Convert from solver output at module boundary
    stats = ClusterStats.from_dict(solver_output["cluster_stats"]["ZoneA+ZoneB"])
    
    # Access tier2 without knowing if cluster was split
    tier2_geom = stats.get_tier2_geometry()
    if tier2_geom:
        _add_boundary_trace(fig, geometry=tier2_geom, ...)
    
    # Use typed attributes (IDE autocomplete works!)
    print(f"Cluster {stats.cluster_index}: {stats.selected_count} boreholes")
    ```
    """
    
    # === Identity ===
    cluster_key: str = ""
    cluster_index: int = 0
    pair_keys: List[str] = field(default_factory=list)
    
    # === Cluster Type ===
    was_split: bool = False
    is_unified_cluster: bool = False
    
    # === Geometry (Unsplit Clusters Only) ===
    tier1_wkt: Optional[str] = None
    tier2_wkt: Optional[str] = None  # Only populated for unsplit clusters!
    
    # === Spacing ===
    overall_r_max: float = 0.0
    r_max: float = 0.0
    max_spacing_m: float = 0.0
    area_ha: float = 0.0
    
    # === Test Point Counts ===
    tier1_test_points: int = 0
    tier2_ring_test_points: int = 0
    precovered_count: int = 0
    unsatisfied_count: int = 0
    
    # === Borehole Counts ===
    candidates_count: int = 0
    selected_count: int = 0
    boreholes_removed: int = 0
    boreholes_added: int = 0
    
    # === Timing ===
    solve_time: float = 0.0
    
    # === Status ===
    status: str = "unknown"
    
    # === Split Cluster Data ===
    cell_wkts: List[str] = field(default_factory=list)
    cell_stats: List[CellStats] = field(default_factory=list)
    cell_czrc_stats: Optional[ThirdPassStats] = None
    
    # === ILP Details (kept as dict for flexibility) ===
    ilp_stats: Dict[str, Any] = field(default_factory=dict)
    
    # === Borehole Lists (expensive to type fully, kept as dicts) ===
    first_pass_candidates: List[Dict[str, Any]] = field(default_factory=list)
    czrc_test_points: List[Dict[str, Any]] = field(default_factory=list)
    second_pass_boreholes: List[Dict[str, Any]] = field(default_factory=list)
    
    # ═══════════════════════════════════════════════════════════════════
    # UNIFIED GEOMETRY ACCESS - The main reason this dataclass exists!
    # ═══════════════════════════════════════════════════════════════════
    
    def get_tier2_geometry(self) -> Optional["BaseGeometry"]:
        """
        Get the Tier 2 boundary geometry, handling split/unsplit transparently.
        
        For unsplit clusters: Returns the single tier2 polygon from tier2_wkt
        For split clusters: Returns the union of all cell tier2 polygons
        
        This method eliminates the need for callers to check `was_split` and
        handle the two data access patterns separately.
        
        Returns:
            Shapely geometry representing the Tier 2 boundary, or None if
            the geometry is unavailable or Shapely is not installed.
        
        Example:
            >>> stats = ClusterStats.from_dict(solver_output)
            >>> tier2 = stats.get_tier2_geometry()
            >>> if tier2:
            ...     fig.add_trace(create_polygon_trace(tier2))
        """
        if not SHAPELY_AVAILABLE:
            return None
            
        if not self.was_split:
            # Unsplit: tier2_wkt is at cluster level
            if self.tier2_wkt:
                try:
                    return wkt.loads(self.tier2_wkt)
                except Exception:
                    return None
        else:
            # Split: tier2_wkt is in each cell_stats entry
            tier2_geoms = []
            for cs in self.cell_stats:
                geom = cs.get_tier2_geometry()
                if geom is not None and not geom.is_empty:
                    tier2_geoms.append(geom)
            
            if tier2_geoms:
                return unary_union(tier2_geoms)
        
        return None
    
    def get_tier1_geometry(self) -> Optional["BaseGeometry"]:
        """Get the Tier 1 boundary geometry.
        
        Returns:
            Shapely geometry or None if tier1_wkt is missing/invalid.
        """
        if not SHAPELY_AVAILABLE or not self.tier1_wkt:
            return None
        try:
            return wkt.loads(self.tier1_wkt)
        except Exception:
            return None
    
    def get_cell_geometries(self) -> List["BaseGeometry"]:
        """Get the cell partition geometries (split clusters only).
        
        Returns:
            List of Shapely geometries for each cell, or empty list if
            cluster is not split or Shapely is unavailable.
        """
        if not SHAPELY_AVAILABLE or not self.cell_wkts:
            return []
        geoms = []
        for cell_wkt in self.cell_wkts:
            try:
                geoms.append(wkt.loads(cell_wkt))
            except Exception:
                continue
        return geoms
    
    # ═══════════════════════════════════════════════════════════════════
    # FACTORY METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClusterStats":
        """
        Create ClusterStats from a dictionary (solver output).
        
        This method handles the full complexity of split vs unsplit cluster
        data, parsing nested cell_stats and cell_czrc_stats automatically.
        
        Args:
            d: Dictionary from czrc_solver.py output
        
        Returns:
            Typed ClusterStats instance
        
        Example:
            >>> raw_stats = czrc_result["cluster_stats"]["ZoneA+ZoneB"]
            >>> stats = ClusterStats.from_dict(raw_stats)
            >>> print(stats.selected_count)  # IDE autocomplete works!
        """
        # Parse cell_stats for split clusters
        cell_stats = [
            CellStats.from_dict(cs) 
            for cs in d.get("cell_stats", [])
        ]
        
        # Parse Third Pass stats if present
        cell_czrc_raw = d.get("cell_czrc_stats")
        cell_czrc_stats = ThirdPassStats.from_dict(cell_czrc_raw) if cell_czrc_raw else None
        
        return cls(
            cluster_key=d.get("cluster_key", ""),
            cluster_index=d.get("cluster_index", 0),
            pair_keys=d.get("pair_keys", []),
            was_split=d.get("was_split", False),
            is_unified_cluster=d.get("is_unified_cluster", False),
            tier1_wkt=d.get("tier1_wkt"),
            tier2_wkt=d.get("tier2_wkt"),
            overall_r_max=d.get("overall_r_max", 0.0),
            r_max=d.get("r_max", 0.0),
            max_spacing_m=d.get("max_spacing_m", 0.0),
            area_ha=d.get("area_ha", 0.0),
            tier1_test_points=d.get("tier1_test_points", 0),
            tier2_ring_test_points=d.get("tier2_ring_test_points", 0),
            precovered_count=d.get("precovered_count", 0),
            unsatisfied_count=d.get("unsatisfied_count", 0),
            candidates_count=d.get("candidates_count", 0),
            selected_count=d.get("selected_count", 0),
            boreholes_removed=d.get("boreholes_removed", 0),
            boreholes_added=d.get("boreholes_added", 0),
            solve_time=d.get("solve_time", 0.0),
            status=d.get("status", "unknown"),
            cell_wkts=d.get("cell_wkts", []),
            cell_stats=cell_stats,
            cell_czrc_stats=cell_czrc_stats,
            ilp_stats=d.get("ilp_stats", {}),
            first_pass_candidates=d.get("first_pass_candidates", []),
            czrc_test_points=d.get("czrc_test_points", []),
            second_pass_boreholes=d.get("second_pass_boreholes", []),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert back to dictionary for JSON serialization.
        
        Maintains backward compatibility with existing code paths that
        expect dict format.
        
        Returns:
            Dictionary representation matching solver output format
        """
        result: Dict[str, Any] = {
            "cluster_key": self.cluster_key,
            "cluster_index": self.cluster_index,
            "pair_keys": self.pair_keys,
            "was_split": self.was_split,
            "is_unified_cluster": self.is_unified_cluster,
            "overall_r_max": self.overall_r_max,
            "r_max": self.r_max,
            "max_spacing_m": self.max_spacing_m,
            "area_ha": self.area_ha,
            "tier1_test_points": self.tier1_test_points,
            "tier2_ring_test_points": self.tier2_ring_test_points,
            "precovered_count": self.precovered_count,
            "unsatisfied_count": self.unsatisfied_count,
            "candidates_count": self.candidates_count,
            "selected_count": self.selected_count,
            "boreholes_removed": self.boreholes_removed,
            "boreholes_added": self.boreholes_added,
            "solve_time": self.solve_time,
            "status": self.status,
            "ilp_stats": self.ilp_stats,
            "first_pass_candidates": self.first_pass_candidates,
            "czrc_test_points": self.czrc_test_points,
            "second_pass_boreholes": self.second_pass_boreholes,
        }
        
        # Only include geometry fields if present
        if self.tier1_wkt:
            result["tier1_wkt"] = self.tier1_wkt
        if self.tier2_wkt:
            result["tier2_wkt"] = self.tier2_wkt
            
        # Only include split cluster fields if applicable
        if self.cell_wkts:
            result["cell_wkts"] = self.cell_wkts
        if self.cell_stats:
            result["cell_stats"] = [cs.to_dict() for cs in self.cell_stats]
        if self.cell_czrc_stats:
            result["cell_czrc_stats"] = self.cell_czrc_stats.to_dict()
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def get_total_boreholes(self) -> int:
        """Get total borehole count across all cells.
        
        For split clusters, sums across cell_stats. For unsplit, returns
        selected_count directly.
        """
        if not self.was_split:
            return self.selected_count
        return sum(cs.selected_count for cs in self.cell_stats)
    
    def get_cell_count(self) -> int:
        """Get number of cells (1 for unsplit clusters)."""
        if not self.was_split:
            return 1
        return len(self.cell_stats)
    
    def summary(self) -> str:
        """Human-readable summary of cluster optimization results."""
        split_indicator = " (split)" if self.was_split else ""
        return (
            f"Cluster {self.cluster_index}{split_indicator}: "
            f"{self.get_total_boreholes()} boreholes, "
            f"r_max={self.overall_r_max:.1f}m, "
            f"area={self.area_ha:.2f}ha"
        )
