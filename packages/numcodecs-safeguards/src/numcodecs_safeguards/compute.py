"""
Helper classes for configuring the compute behaviour of the codec with safeguards.
"""

__all__ = ["Compute"]

from dataclasses import dataclass

import numpy as np
from compression_safeguards.api import Safeguards
from compression_safeguards.safeguards.pointwise.abc import PointwiseSafeguard
from compression_safeguards.safeguards.stencil.abc import StencilSafeguard
from compression_safeguards.utils.bindings import Bindings
from compression_safeguards.utils.error import SafeguardsSafetyBug, ctx
from compression_safeguards.utils.typing import C, S, T


@dataclass
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

    _unstable_iterative: bool = False
    """
    Unstable option to use an iterative algorithm that can reduce the number of
    corrections that need to be applied, which can improve the compression
    ratio at the cost of requiring additional time to compute the corrections.
    """


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
            with ctx.parameter("compute"), ctx.parameter("_unstable_iterative"):
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
    correction_iterative = np.zeros_like(correction)
    corrected_iterative = prediction.copy()
    last_needs_correction = np.zeros(data.shape, dtype=np.bool)

    # resolve the late-bound bindings using the Safeguards API, since we use
    #  lower-level APIs from now on
    late_bound_resolved = safeguards._prepare_non_chunked_bindings(
        data=data,
        prediction=prediction,
        late_bound=late_bound,
        description="checking the safeguards",
        chunked_method_name="check_chunk",
    )

    while True:
        # check for data points all pointwise checks succeed
        check_pointwise = np.ones(data.shape, dtype=np.bool)
        for safeguard in safeguards_:
            check_pointwise &= safeguard.check_pointwise(
                data,
                corrected_iterative,
                late_bound=late_bound_resolved,
                # a reduced where would only include the inverse footprint of
                #  the data points that were newly corrected in the last round,
                # i.e. which points might now have re-evaluate their checks
                #  since they depend on these point
                where=True,
            )

        # all checks succeed, so a reduced correction has been found
        if np.all(check_pointwise):
            if not safeguards.check(
                data,
                safeguards.apply_correction(prediction, correction_iterative),
                late_bound=late_bound,
                where=True,
            ):
                raise (
                    SafeguardsSafetyBug(
                        "the iteratively refined corrections fail the "
                        + "safeguards check"
                    )
                    | ctx
                )

            return correction_iterative

        # find points that failed the check during the previous iteration and
        #  still fail the check
        # for these sticky failures, expand the failure footprint to correct
        #  all data points that could have contributed to the failure
        sticky_needs_correction_pointwise = (~check_pointwise) & last_needs_correction
        sticky_needs_correction_footprint = np.zeros_like(
            sticky_needs_correction_pointwise
        )
        for safeguard in safeguards_:
            sticky_needs_correction_footprint |= safeguard.compute_footprint(
                sticky_needs_correction_pointwise,
                late_bound=late_bound_resolved,
                where=True,
            )

        # determine the data points that need a correction
        needs_correction = sticky_needs_correction_footprint
        needs_correction |= ~check_pointwise
        last_needs_correction[:] = needs_correction

        # use the pre-computed correction where needed
        correction_iterative[needs_correction] = correction_full[needs_correction]
        corrected_iterative[needs_correction] = corrected_full[needs_correction]
