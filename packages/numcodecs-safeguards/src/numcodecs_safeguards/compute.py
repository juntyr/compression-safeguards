"""
Helper classes for configuring the compute behaviour of the codec with safeguards.
"""

__all__ = ["Compute"]

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Self

import numpy as np
from compression_safeguards.api import Safeguards
from compression_safeguards.safeguards.pointwise.abc import PointwiseSafeguard
from compression_safeguards.safeguards.stencil.abc import StencilSafeguard
from compression_safeguards.utils.bindings import Bindings
from compression_safeguards.utils.error import SafeguardsSafetyBug, ctx
from compression_safeguards.utils.typing import JSON, C, S, T


@dataclass(kw_only=True)
class Compute:
    """
    Compute configuration with options that may affect the compression
    ratio and time cost of computing the safeguards corrections.

    While these options can change the particular corrections that are
    produced, the resulting corrections always satisfy the safety
    requirements.

    Some configuration options are unstable, i.e. they should not be relied
    upon in production code since they might be removed or changed without a
    breaking major version bump.
    """

    unstable_iterative: bool = False
    """
    Unstable option to use an iterative algorithm that can reduce the number of
    corrections that need to be applied, which can improve the compression
    ratio at the cost of requiring additional time to compute the corrections.
    """

    def get_config(self) -> dict[str, JSON]:
        """
        Returns the compute configuration.

        Returns
        -------
        config : dict[str, JSON]
            Compute configuration.
        """

        return dataclasses.asdict(self)

    @classmethod
    def from_config(cls, config: dict[str, JSON]) -> Self:
        """
        Instantiate the compute configuration from a [`dict`][dict].

        Parameters
        ----------
        config : dict[str, JSON]
            Compute configuration.

        Returns
        -------
        safeguards : Self
            Instantiated compute configuration.
        """

        FIELDS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(cls))
        STABILISED: dict[str, str] = dict()
        REMOVED_UNSTABLE: tuple[str, ...] = ()

        clean_config = dict()

        for name, value in config.items():
            if name in FIELDS:
                clean_config[name] = value
            elif name in STABILISED:
                # automatically upgrade unstable options
                warnings.warn(
                    f"compute configuration `{name}` has been stabilised as `{STABILISED[name]}`",
                    category=PendingDeprecationWarning,
                )
                clean_config[STABILISED[name]] = value
            elif name in REMOVED_UNSTABLE:
                # skip removed unstable options
                warnings.warn(
                    f"compute configuration `{name}` has been removed",
                    category=DeprecationWarning,
                )
            else:
                raise TypeError(f"unknown compute configuration `{name}`") | ctx

        return cls(**clean_config)  # type: ignore


def _refine_correction_iteratively(
    safeguards: Safeguards,
    data: np.ndarray[S, np.dtype[T]],
    prediction: np.ndarray[S, np.dtype[T]],
    correction: np.ndarray[S, np.dtype[C]],
    late_bound: Bindings,
) -> np.ndarray[S, np.dtype[C]]:
    if np.all(correction == 0):
        return correction

    safeguards_: list[PointwiseSafeguard | StencilSafeguard] = []

    for safeguard in safeguards.safeguards:
        if not isinstance(safeguard, PointwiseSafeguard | StencilSafeguard):
            with ctx.parameter("compute"), ctx.parameter("unstable_iterative"):
                raise (
                    ValueError(
                        "only supported for pointwise and stencil safeguards, "
                        + f"but {type(safeguard).kind} is neither"
                    )
                    | ctx
                )
        safeguards_.append(safeguard)

    # full correction
    correction_full = correction
    corrected_full = safeguards.apply_correction(prediction, correction_full)

    # iterative correction, starting with no correction at all
    correction_iterative = np.zeros_like(correction_full)
    corrected_iterative = prediction.copy()

    # resolve the late-bound bindings using the Safeguards API, since we use
    #  lower-level APIs from now on
    late_bound_resolved = safeguards._prepare_non_chunked_bindings(
        data=data,
        prediction=prediction,
        late_bound=late_bound,
        description="checking the safeguards",
        chunked_method_name="check_chunk",
    )

    # check for which data points all pointwise checks succeed
    check_pointwise = np.ones(data.shape, dtype=np.bool)
    for safeguard in safeguards_:
        check_pointwise &= safeguard.check_pointwise(
            data,
            corrected_iterative,
            late_bound=late_bound_resolved,
            where=True,  # start with a complete check
        )

    # refine while not all checks succeed
    while not np.all(check_pointwise):
        # find points that failed the check but have already been corrected
        # for these sticky failures, expand to the failure inverse footprint,
        #  to correct all data points that may have contributed to the failure
        sticky_needs_correction_pointwise = (~check_pointwise) & (
            correction_iterative == correction_full
        )
        sticky_needs_correction_inverse_footprint = np.zeros_like(
            sticky_needs_correction_pointwise
        )
        for safeguard in safeguards_:
            sticky_needs_correction_inverse_footprint |= (
                safeguard.compute_inverse_footprint(
                    sticky_needs_correction_pointwise,
                    late_bound=late_bound_resolved,
                    where=True,  # complete inverse footprint
                )
            )

        # determine the data points that need a correction
        needs_correction = sticky_needs_correction_inverse_footprint
        needs_correction |= ~check_pointwise

        # determine the data points that get a new correction
        correction_changed = correction_iterative != correction_full
        correction_changed &= needs_correction

        # use the pre-computed correction where needed
        correction_iterative[needs_correction] = correction_full[needs_correction]
        corrected_iterative[needs_correction] = corrected_full[needs_correction]

        # expand the footprint of the changed corrections to find the points
        #  that need to be rechecked
        where = np.zeros(data.shape, dtype=np.bool)
        for safeguard in safeguards_:
            where |= safeguard.compute_footprint(
                correction_changed,
                late_bound=late_bound_resolved,
                where=True,  # complete footprint
            )

        # sanity check: we recheck at least where the check previously failed
        assert not np.any(~check_pointwise & ~where)

        # update for which data points all pointwise checks succeed
        check_pointwise.fill(True)
        for safeguard in safeguards_:
            check_pointwise &= safeguard.check_pointwise(
                data,
                corrected_iterative,
                late_bound=late_bound_resolved,
                # minimal where only includes the footprint of the data points
                #  that were newly corrected just now,
                # i.e. the points that might now have re-evaluate their checks
                #  since they depend on these newly corrected point
                where=where,
            )

        # continue with the next loop iteration to see if all checks succeeded
        continue

    # all checks succeeded, so a reduced correction has been found

    # safety check that the refined correction is in fact valid
    if not safeguards.check(
        data,
        safeguards.apply_correction(prediction, correction_iterative),
        late_bound=late_bound,
        where=True,  # complete check check
    ):
        raise (
            SafeguardsSafetyBug(
                "the iteratively refined corrections fail the " + "safeguards check"
            )
            | ctx
        )

    return correction_iterative
