"""
Typed data models for borehole optimization.

Architectural Overview:
=======================
This module contains immutable dataclasses that replace Dict[str, Any]
throughout the codebase. The key benefit is that source_pass is REQUIRED,
eliminating the reconstruction bugs where boreholes showed incorrect pass labels.

Key Interactions:
-----------------
- Input: Optimization functions create Borehole instances with explicit source_pass
- Output: as_dict() method provides backward compatibility with existing code
- Navigation: Use VS Code outline (Ctrl+Shift+O) for quick navigation

Data Flow:
----------
1. First Pass creates boreholes with source_pass=FIRST
2. Second Pass CZRC creates boreholes with source_pass=SECOND
3. Third Pass Cell-Cell creates boreholes with source_pass=THIRD
4. Removed boreholes KEEP their original source_pass (where they came from)

MODIFICATION POINT: Add new BoreholeStatus values here for future pass types
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# 🏷️ ENUMS SECTION
# ═══════════════════════════════════════════════════════════════════════════

class BoreholePass(Enum):
    """Which optimization pass created this borehole.
    
    This enum replaces string literals like "First Pass", "Second Pass", etc.
    Using an enum prevents typos and enables IDE autocomplete.
    
    MODIFICATION POINT: Add new pass types here (e.g., FOURTH = "Fourth Pass")
    """
    FIRST = "First Pass"
    SECOND = "Second Pass"
    THIRD = "Third Pass"
    
    @classmethod
    def from_string(cls, s: str) -> "BoreholePass":
        """Convert string to BoreholePass, with fallback to FIRST.
        
        Args:
            s: String like "First Pass", "Second Pass", "Third Pass"
            
        Returns:
            Matching BoreholePass enum member, or FIRST if not found
        """
        for member in cls:
            if member.value == s:
                return member
        return cls.FIRST


class BoreholeStatus(Enum):
    """Current status of a borehole in the optimization.
    
    Tracks whether a borehole is proposed, added, removed, or locked.
    """
    PROPOSED = "proposed"      # Initial proposal from optimization
    ADDED = "added"            # Added by CZRC optimization
    REMOVED = "removed"        # Eliminated by optimization
    LOCKED = "locked"          # Fixed, cannot be removed
    
    @classmethod
    def from_string(cls, s: str) -> "BoreholeStatus":
        """Convert string to BoreholeStatus, with fallback to PROPOSED.
        
        Args:
            s: String like "proposed", "added", "removed", "locked"
            
        Returns:
            Matching BoreholeStatus enum member, or PROPOSED if not found
        """
        for member in cls:
            if member.value == s:
                return member
        return cls.PROPOSED


# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ BOREHOLE DATACLASS SECTION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Borehole:
    """Immutable borehole with full provenance tracking.
    
    Key Differences from Dict[str, Any]:
    ------------------------------------
    1. source_pass is REQUIRED - cannot create a borehole without it
    2. frozen=True prevents accidental mutation
    3. position property provides normalized coordinates for comparison
    4. Type hints enable IDE autocomplete and error detection
    
    Why Immutable (frozen=True):
    ----------------------------
    When a borehole is "removed" by an optimization pass, we create a NEW
    Borehole instance with status=REMOVED, preserving the original source_pass.
    This prevents bugs where removed boreholes incorrectly show the current pass.
    
    Usage Examples:
    ---------------
    ```python
    # Create from optimization result
    bh = Borehole(
        x=123.456, y=789.012, coverage_radius=100.0,
        source_pass=BoreholePass.SECOND,
        status=BoreholeStatus.ADDED
    )
    
    # Convert to dict for backward compatibility
    d = bh.as_dict()
    
    # Create from existing dict (legacy code)
    bh2 = Borehole.from_dict(d)
    
    # Use position for set operations
    positions = {bh.position for bh in boreholes}
    ```
    """
    x: float
    y: float
    coverage_radius: float
    source_pass: BoreholePass  # REQUIRED - this is the key fix for Bug 1
    status: BoreholeStatus = BoreholeStatus.PROPOSED
    zone_id: Optional[str] = None
    cluster_id: Optional[str] = None
    tier: Optional[int] = None
    
    @property
    def position(self) -> Tuple[float, float]:
        """Normalized position for set operations and comparison.
        
        Rounds to 6 decimal places to avoid floating-point comparison issues.
        This fixes Bug 3 where position-based reconstruction failed due to
        tiny floating-point differences.
        
        Returns:
            Tuple of (x, y) rounded to 6 decimal places
        """
        return (round(self.x, 6), round(self.y, 6))
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dict for backward compatibility with existing code.
        
        This method allows gradual migration - new code creates Borehole
        instances, but existing code can still work with dicts.
        
        Returns:
            Dict with all fields, source_pass and status as string values
        """
        result = {
            "x": self.x,
            "y": self.y,
            "coverage_radius": self.coverage_radius,
            "source_pass": self.source_pass.value,
            "status": self.status.value,
        }
        # Only include optional fields if set
        if self.zone_id is not None:
            result["zone_id"] = self.zone_id
        if self.cluster_id is not None:
            result["cluster_id"] = self.cluster_id
        if self.tier is not None:
            result["tier"] = self.tier
        return result
    
    @classmethod
    def from_dict(
        cls, 
        d: Dict[str, Any], 
        default_pass: BoreholePass = BoreholePass.FIRST
    ) -> "Borehole":
        """Create Borehole from dict, handling legacy dicts without source_pass.
        
        This method enables backward compatibility with existing code that
        uses Dict[str, Any] for boreholes.
        
        Args:
            d: Dictionary with at least x, y, coverage_radius
            default_pass: Used only if source_pass is missing (legacy data)
            
        Returns:
            Borehole instance with all fields populated
            
        Raises:
            KeyError: If x, y, or coverage_radius are missing
        """
        # Handle source_pass - can be string, BoreholePass, or missing
        source_pass_value = d.get("source_pass")
        if source_pass_value is None:
            source_pass = default_pass
        elif isinstance(source_pass_value, BoreholePass):
            source_pass = source_pass_value
        else:
            source_pass = BoreholePass.from_string(str(source_pass_value))
        
        # Handle status - can be string, BoreholeStatus, or missing
        status_value = d.get("status", "proposed")
        if isinstance(status_value, BoreholeStatus):
            status = status_value
        else:
            status = BoreholeStatus.from_string(str(status_value))
        
        return cls(
            x=float(d["x"]),
            y=float(d["y"]),
            coverage_radius=float(d["coverage_radius"]),
            source_pass=source_pass,
            status=status,
            zone_id=d.get("zone_id"),
            cluster_id=d.get("cluster_id"),
            tier=d.get("tier"),
        )
    
    def with_status(self, new_status: BoreholeStatus) -> "Borehole":
        """Create a copy with a new status, preserving all other fields.
        
        Since Borehole is frozen, we can't mutate it. This method creates
        a new instance with the updated status.
        
        Args:
            new_status: The new status for the borehole
            
        Returns:
            New Borehole instance with updated status
        """
        return Borehole(
            x=self.x,
            y=self.y,
            coverage_radius=self.coverage_radius,
            source_pass=self.source_pass,  # Preserve original!
            status=new_status,
            zone_id=self.zone_id,
            cluster_id=self.cluster_id,
            tier=self.tier,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 📊 PASS RESULT DATACLASS SECTION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PassResult:
    """Output from a single optimization pass.
    
    Tracks exactly what came in and what came out, with full provenance.
    This provides a clear audit trail for debugging and visualization.
    
    Key Design Decision:
    --------------------
    - added: Boreholes CREATED by this pass (source_pass = this pass)
    - removed: Boreholes ELIMINATED by this pass (source_pass = ORIGINAL pass!)
    
    This distinction is critical for correct visualization labels.
    
    Usage:
    ------
    ```python
    result = PassResult(
        pass_type=BoreholePass.SECOND,
        input_boreholes=first_pass_output,
        output_boreholes=optimized_boreholes,
        added=[bh for bh in optimized if bh not in first_pass_output],
        removed=[bh for bh in first_pass_output if bh not in optimized],
    )
    
    print(f"Second Pass: {result.net_change:+d} boreholes")
    ```
    """
    pass_type: BoreholePass
    input_boreholes: List[Borehole]
    output_boreholes: List[Borehole]
    added: List[Borehole]      # Created by this pass (source_pass = this pass)
    removed: List[Borehole]    # Eliminated by this pass (keep original source_pass!)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def net_change(self) -> int:
        """Net boreholes added (positive) or removed (negative).
        
        Returns:
            len(added) - len(removed)
        """
        return len(self.added) - len(self.removed)
    
    @property
    def input_count(self) -> int:
        """Number of boreholes that entered this pass."""
        return len(self.input_boreholes)
    
    @property
    def output_count(self) -> int:
        """Number of boreholes that exited this pass."""
        return len(self.output_boreholes)
    
    def summary(self) -> str:
        """Human-readable summary of this pass's results.
        
        Returns:
            String like "Second Pass: 10 in → 8 out (+2 added, -4 removed)"
        """
        return (
            f"{self.pass_type.value}: {self.input_count} in → {self.output_count} out "
            f"(+{len(self.added)} added, -{len(self.removed)} removed)"
        )
