"""
EC7 Simple Gap Analysis

Ultra-simplified EC7-compliant borehole spacing analysis.
Single site-wide spacing, single grid generator, minimal code.
"""

from Gap_Analysis_EC7.main import run_ec7_analysis
from Gap_Analysis_EC7.config import CONFIG

__all__ = ["run_ec7_analysis", "CONFIG"]
