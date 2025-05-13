"""
Implementations for the provided pointwise quantity of interest (QoI) safeguards.
"""

__all__ = ["Expr"]

from typing import NewType

Expr = NewType("Expr", str)
