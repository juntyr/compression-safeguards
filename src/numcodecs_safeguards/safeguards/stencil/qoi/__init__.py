"""
Implementations for the provided stencil quantity of interest (QoI) safeguards.
"""

__all__ = ["StencilExpr"]

from typing import NewType

StencilExpr = NewType("StencilExpr", str)
