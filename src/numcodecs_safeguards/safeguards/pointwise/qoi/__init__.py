"""
Implementations for the provided pointwise quantity of interest (QoI) safeguards.
"""

__all__ = ["PointwiseExpr"]

from typing import NewType

PointwiseExpr = NewType("PointwiseExpr", str)
