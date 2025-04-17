"""
Bias reducing safeguard.
"""

__all__ = ["BiasReductionSafeguard"]

import numpy as np

from .abc import StochasticSafeguard
from ...intervals import IntervalUnion, Interval, Lower, Upper, Minimum, Maximum


class BiasReductionSafeguard(StochasticSafeguard):
    __slots__ = ()

    kind = "bias"

    def __init__(self):
        pass

    def compute_safe_intervals(self, data: np.ndarray) -> IntervalUnion:
        data = data.flatten()
        valid = Interval.empty_like(data)

        # high is exclusive, so to sample [False, True], we need high = True+1
        mask = np.random.randint(2, size=data.size, dtype=bool)

        Lower(data) <= valid[mask] <= Maximum
        Minimum <= valid[~mask] <= Upper(data)

        return valid.into_union()

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(kind=type(self).kind)
