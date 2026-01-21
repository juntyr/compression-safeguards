"""
# Fearless (chunked) lossy compression with `xarray-safeguards`

Lossy compression can be *scary* as valuable information or features of the
data may be lost.

By using [`Safeguards`][compression_safeguards.api.Safeguards] to **guarantee**
your safety requirements, lossy compression can be applied safely and
*without fear*.

## Overview

This package provides functionality to use safeguards with (chunked)
[`xarray.DataArray`][xarray.DataArray]s and cross-chunk boundary conditions.
Since applying safeguards to chunked data by yourself is very difficult and
error-prone, this package handles all of the complexity for you with a simple
and safe API.

In particular, `xarray-safeguards` provides the
[`produce_data_array_correction`][.produce_data_array_correction]
function to produce a correction such that certain properties of the original
data are preserved after compression, which can be stored in the same or a
different dataset (or file).

This correction can be applied to the decompressed data using the
[`apply_data_array_correction`][.apply_data_array_correction] function or the
[`.safeguarded`][.DatasetSafeguardedAccessor] accessor on datasets.

This package also provides the [`.safeguards`][.DataArraySafeguardsAccessor]
accessor on correction or corrected data arrays to inspect the safeguards that
were applied.

By applying the corrections produced by `produce_data_array_correction`, data
that was compressed with badly-behaving lossy compressors can be safely used,
at the cost of potentially less efficient compression, and lossy compression
can be applied *without fear*.

## Example

You can produce and apply the corrections to uphold an absolute error bound of
$eb_{abs} = 0.1$ for an already-compressed data array as follows:

```py
import numpy as np
import xarray as xr
from xarray_safeguards import apply_data_array_correction, produce_data_array_correction

# some (chunked) n-dimensional data array
da = xr.DataArray(np.linspace(-10, 10, 21), name="da").chunk(10)
# lossy-compressed prediction for the data, here all zeros
da_prediction = xr.DataArray(np.zeros_like(da.values), name="da").chunk(10)

da_correction = produce_data_array_correction(
    data=da,
    prediction=da_prediction,
    # guarantee an absolute error bound of 0.1:
    #   |x - x'| <= 0.1
    safeguards=[dict(kind="eb", type="abs", eb=0.1)],
)

## (a) manual correction ##

da_corrected = apply_data_array_correction(da_prediction, da_correction)
np.testing.assert_allclose(da_corrected.values, da.values, rtol=0, atol=0.1)

## (b) automatic correction with xarray accessors ##

# combine the lossy prediction and the correction into one dataset
#  e.g. by loading them from different files using `xarray.open_mfdataset`
ds = xr.Dataset({
    da_prediction.name: da_prediction,
    da_correction.name: da_correction,
})

# access the safeguarded dataset that applies all corrections
ds_safeguarded: xr.Dataset = ds.safeguarded
np.testing.assert_allclose(ds_safeguarded["da"].values, da.values, rtol=0, atol=0.1)
```

Please refer to the
[`compression_safeguards.SafeguardKind`][compression_safeguards.safeguards.SafeguardKind]
for an enumeration of all supported safeguards.
"""

__all__ = [
    "produce_data_array_correction",
    "apply_data_array_correction",
    "DataValue",
    "DatasetSafeguardedAccessor",
    "DataArraySafeguardsAccessor",
]

import json
import warnings
from collections.abc import Collection, Mapping
from types import MappingProxyType
from typing import Literal, TypeAlias, assert_never

import dask
import dask.array
import numpy as np
import xarray as xr
from compression_safeguards.api import Safeguards
from compression_safeguards.safeguards.abc import Safeguard
from compression_safeguards.safeguards.stencil import (
    BoundaryCondition,
    NeighbourhoodAxis,
)
from compression_safeguards.utils.bindings import Parameter, Value
from compression_safeguards.utils.error import (
    LateBoundParameterResolutionError,
    TypeCheckError,
    TypeSetError,
    ctx,
)
from compression_safeguards.utils.typing import JSON, S, T

DataValue: TypeAlias = int | float | np.number | xr.DataArray
"""
Parameter value type that includes scalar numbers and data arrays thereof.
"""


def produce_data_array_correction(
    data: xr.DataArray,
    prediction: xr.DataArray,
    safeguards: Collection[dict[str, JSON] | Safeguard],
    late_bound: Mapping[str, DataValue] = MappingProxyType(dict()),
    *,
    allow_unsafe_safeguards_override: bool = False,
) -> xr.DataArray:
    """
    Produce the correction required to make the `prediction` data array satisfy the `safeguards` relative to the `data` array.

    The `data` array may be chunked[^1] and the `prediction` array must use the
    same chunking. Importantly, the `data` array must contain the complete data,
    i.e. not just a sub-chunk of the data, so that non-pointwise safeguards are
    correctly applied.

    If the the `data` array is chunked, the correction is produced lazily,
    otherwise it is computed eagerly.

    The `data` array must have a name and the `prediction` array must use the
    same name.

    [^1]: At the moment, only chunking with `dask` is supported.

    Parameters
    ----------
    data : xr.DataArray
        The data array, relative to which the safeguards are enforced.
    prediction : xr.DataArray
        The prediction array for which the correction is produced.
    safeguards : Collection[dict[str, JSON] | Safeguard]
        The safeguards that will be applied relative to the `data` array.

        They can either be
        passed as a safeguard configuration [`dict`][dict] or an already
        initialized
        [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard].

        Please refer to the
        [`SafeguardKind`][compression_safeguards.safeguards.SafeguardKind]
        for an enumeration of all supported safeguards.
    late_bound : Mapping[str, DataValue]
        The bindings for all late-bound parameters of the `safeguards`.

        The bindings must resolve all late-bound parameters and include no
        extraneous parameters.

        If a binding resolves to a [`xr.DataArray`][xarray.DataArray], its
        dimensions must be a subset of `data.dims` and the shape must be
        broadcastable to the data shape, i.e. each axis must have either
        size 1 or the same size as the matching dimension in the `data`.

        This method automatically provides the following built-in constants
        to the safeguards, which must not be included:

        - `$x` and `$X`: the original `data` as a constant
        - `$x_min` and `$x_max`: the global minimum/maximum of the data
        - `$d_DIM` for each dimension `DIM` of the `data` array
    allow_unsafe_safeguards_override : bool
        **WARNING:** This option is *unsafe* and must only be passed if
        instructed. Do *not* pass this option otherwise.

    Returns
    -------
    correction : xr.DataArray
        The correction array

    Raises
    ------
    ValueError
        if the `data` has already been corrected by safeguards, i.e. if it is
        not the original uncorrected data, which will create a printer problem.
    ValueError
        if the `prediction` has already been corrected, which may create a
        printer problem.
    TypeSetError
        if the `data` uses an unsupported data type.
    ValueError
        if the `data`'s dimensions, shape, dtype, chunks, or name do not match
        the `prediction`'s, or if the data name is [`None`][None].
    LateBoundParameterResolutionError
        if `late_bound` does not resolve all late-bound parameters of the
        safeguards or includes any extraneous parameters.
    TypeError
        if a late-bound parameter is not a scalar or
        [`xarray.DataArray`][xarray.DataArray].
    ValueError
        if a late-bound parameter's dimensions are not a subset of the `data`
        dimensions, or it is not broadcastable to the `data` shape.
    ...
        if instantiating a safeguard, checking a safeguard, or computing the
        correction for a safeguard raises an exception.
    """

    # small safeguard against the printer problem
    if "safeguards" in data.attrs:
        with ctx.parameter("data"):
            raise (
                ValueError(
                    "computing the safeguards correction relative to a `data` "
                    + "array that has *already* been safeguards-corrected "
                    + "before is unsafe as compression errors can accumulate "
                    + "when the original uncompressed data is not known; this "
                    + "is also known as the printer problem; please pass the "
                    + "original uncompressed and uncorrected `data` to ensure "
                    + "that the safeguards can be applied correctly"
                )
                | ctx
            )

    if "safeguards" in prediction.attrs:
        if allow_unsafe_safeguards_override:
            warnings.warn("`allow_unsafe_safeguards_override`=True")
        elif "safeguards" in prediction.attrs:
            with ctx.parameter("prediction"):
                raise (
                    ValueError(
                        "computing the safeguards correction for a "
                        + "`prediction` array that has *already* been "
                        + "safeguards-corrected before is unsafe as the "
                        + "safety guarantees provided by the previously "
                        + "applied safeguards may not be provided by the newly "
                        + "applied safeguards, thus violating the earlier "
                        + "guarantees; this is also known as the printer "
                        + "problem; please manually inspect the `.safeguards` "
                        + "property of the `prediction` array and ensure that "
                        + "they are included in the new `safeguards` and then "
                        + "pass `allow_unsafe_safeguards_override=True` to "
                        + "this function"
                    )
                    | ctx
                )
    elif allow_unsafe_safeguards_override is not False:
        with ctx.parameter("allow_unsafe_safeguards_override"):
            raise (
                ValueError("unsafe option that must only be passed if instructed") | ctx
            )

    TypeSetError.check_dtype_or_raise(data.dtype, Safeguards.supported_dtypes())

    with ctx.parameter("prediction"):
        if prediction.dims != data.dims:
            raise ValueError("prediction.dims must match data.dims") | ctx
        if prediction.shape != data.shape:
            raise ValueError("prediction.shape must match data.shape") | ctx
        if prediction.dtype != data.dtype:
            raise ValueError("prediction.dtype must match data.dtype") | ctx
        if prediction.chunks != data.chunks:
            raise ValueError("prediction.chunks must match data.chunks") | ctx
        if prediction.name != data.name:
            raise ValueError("prediction.name must match data.name") | ctx
    with ctx.parameter("data"):
        if data.name is None:
            raise ValueError("data.name must not be None") | ctx

    safeguards_: Safeguards = Safeguards(safeguards=safeguards)

    builtin_late_bound: frozenset[Parameter] = frozenset(
        safeguards_.builtin_late_bound
    ) | frozenset(
        [Parameter("$x_min"), Parameter("$x_max")]
        + [Parameter(f"$d_{d}") for d in data.dims]
    )

    safeguards_late_bound_reqs = frozenset(safeguards_.late_bound)
    late_bound_reqs = frozenset(safeguards_late_bound_reqs - builtin_late_bound)
    late_bound_keys = frozenset(Parameter(k) for k in late_bound.keys())

    LateBoundParameterResolutionError.check_or_raise(late_bound_reqs, late_bound_keys)

    # create the global built-in late-bound bindings with $x_min and $x_max
    #  and split-out the late-bound data array bindings that require chunking
    late_bound_global: dict[str, int | float | np.number] = dict()
    late_bound_data_arrays: dict[str, xr.DataArray] = dict()
    if "$x_min" in safeguards_late_bound_reqs:
        da_min = (
            np.nanmin(data)
            if data.size > 0 and not np.all(np.isnan(data))
            else data.dtype.type(0)
        )
        late_bound_global["$x_min"] = da_min
    if "$x_max" in safeguards_late_bound_reqs:
        da_max = (
            np.nanmax(data)
            if data.size > 0 and not np.all(np.isnan(data))
            else data.dtype.type(0)
        )
        late_bound_global["$x_max"] = da_max
    with ctx.parameter("late_bound"):
        for k, v in late_bound.items():
            with ctx.parameter(k):
                TypeCheckError.check_instance_or_raise(
                    v, int | float | np.number | xr.DataArray
                )
                if isinstance(v, int | float | np.number):
                    late_bound_global[k] = v
                else:
                    if not frozenset(v.dims).issubset(data.dims):
                        raise ValueError("dims are not a subset of data dims") | ctx
                    for d, ds in v.sizes.items():
                        if (ds != 1) and (ds != data.sizes[d]):
                            raise (
                                ValueError(
                                    f"dim {d} is not broadcastable to the data shape"
                                )
                                | ctx
                            )
                    late_bound_data_arrays[k] = v

    correction_name = f"{data.name}_sg"
    correction_attrs = dict(
        safeguarded=data.name, safeguards=json.dumps(safeguards_.get_config())
    )

    da_correction: xr.DataArray

    # special case for no chunking: just compute eagerly
    if data.chunks is None:
        # provide built-in late-bound bindings $d for the data dimensions
        late_bound_full: dict[str, Value] = dict(**late_bound_global)
        for i, d in enumerate(data.dims):
            if f"$d_{d}" in safeguards_late_bound_reqs:
                shape = [1 for _ in range(len(data.dims))]
                shape[i] = data.shape[i]
                late_bound_full[f"$d_{d}"] = data[d].values.reshape(shape)
        for k, dv in late_bound_data_arrays.items():
            axes = sorted(
                [i for i in range(len(dv.dims))],
                key=lambda i: data.dims.index(dv.dims[i]),
            )
            shape = [1 for _ in range(len(data.dims))]
            for d in dv.dims:
                i = data.dims.index(d)
                shape[i] = data.shape[i]
            late_bound_full[k] = dv.values.transpose(axes).reshape(shape)

        da_correction = (
            data.copy(
                data=safeguards_.compute_correction(
                    data.values,
                    prediction.values,
                    late_bound=late_bound_full,
                    where=True,
                )
            )
            .rename(correction_name)
            .assign_attrs(**correction_attrs)
        )

        # drop the previous encoding from the variable but not the coords
        return da_correction._replace(variable=da_correction.variable.drop_encoding())

    # provide built-in late-bound bindings $d for the data dimensions
    chunked_late_bound: dict[str, dask.array.Array] = dict()
    for i, d in enumerate(data.dims):
        if f"$d_{d}" in safeguards_late_bound_reqs:
            shape = [1 for _ in range(len(data.dims))]
            shape[i] = data.shape[i]
            chunked_late_bound[f"$d_{d}"] = dask.array.broadcast_to(
                data[d].data.reshape(shape),
                shape=data.shape,
                meta=np.array((), dtype=data[d].dtype),
            ).rechunk(data.chunks)
    for k, v in late_bound_data_arrays.items():
        dims = sorted(v.dims, key=lambda i: data.dims.index(i))
        shape = [1 for _ in range(len(data.dims))]
        for d in v.dims:
            i = data.dims.index(d)
            shape[i] = data.shape[i]
        chunked_late_bound[k] = dask.array.broadcast_to(
            v.transpose(*dims, transpose_coords=False).data.reshape(shape),
            shape=data.shape,
            meta=np.array((), dtype=v.dtype),
        ).rechunk(data.chunks)

    required_stencil = safeguards_.compute_required_stencil_for_chunked_correction(
        data.shape
    )
    correction_dtype: np.dtype[np.unsignedinteger] = (
        safeguards_.correction_dtype_for_data(data.dtype)
    )

    # special case for no stencil: just apply independently to each chunk
    if all(s.before == 0 and s.after == 0 for b, s in required_stencil):

        def _compute_independent_chunk_correction(
            data_chunk: np.ndarray[S, np.dtype[T]],
            prediction_chunk: np.ndarray[S, np.dtype[T]],
            *late_bound_chunks: np.ndarray[S, np.dtype[T]],
            late_bound_names: tuple[str, ...],
            late_bound_global: dict[str, int | float | np.number],
            safeguards: Safeguards,
        ) -> np.ndarray[S, np.dtype[np.unsignedinteger]]:
            assert len(late_bound_chunks) == len(late_bound_names), (
                "late-bound chunks and names mismatch"
            )

            late_bound_chunk: dict[str, Value] = dict(
                **late_bound_global,
                **{p: v for p, v in zip(late_bound_names, late_bound_chunks)},
            )

            return safeguards.compute_correction(
                data_chunk, prediction_chunk, late_bound=late_bound_chunk, where=True
            )

        da_correction = (
            data.copy(
                data=data.data.map_blocks(
                    _compute_independent_chunk_correction,
                    prediction.data,
                    *chunked_late_bound.values(),
                    dtype=correction_dtype,
                    chunks=None,
                    enforce_ndim=True,
                    meta=np.array((), dtype=correction_dtype),
                    late_bound_names=tuple(chunked_late_bound.keys()),
                    late_bound_global=late_bound_global,
                    safeguards=safeguards_,
                )
            )
            .rename(correction_name)
            .assign_attrs(**correction_attrs)
        )

        # drop the previous encoding from the variable but not the coords
        return da_correction._replace(variable=da_correction.variable.drop_encoding())

    boundary: Literal["none", "periodic"] = "none"
    depth_: list[tuple[int, int]] = []
    for a, (b, s) in zip(data.shape, required_stencil):
        # dask doesn't support depths larger than the axes,
        # so clip that axis and prefer no boundary condition
        #  as it will anyways be rechunked to just a single chunk
        if s.before >= a or s.after >= a:
            depth_.append((a, a))
        else:
            depth_.append((s.before, s.after))
            match b:
                case BoundaryCondition.valid:
                    pass
                case BoundaryCondition.wrap:
                    boundary = "periodic"
                case _:
                    assert_never(b)

    match boundary:
        case "none":
            depth: tuple[int | tuple[int, int], ...] = tuple(
                b if b == a else (a, b) for a, b in depth_
            )
        case "periodic":
            # dask only supports asymmetric depths for the none boundary
            depth = tuple(max(a, b) for a, b in depth_)
        case _:
            assert_never(boundary)

    def _check_overlapping_stencil_chunk(
        data_chunk: np.ndarray[S, np.dtype[T]],
        prediction_chunk: np.ndarray[S, np.dtype[T]],
        data_indices_chunk: np.ndarray[S, np.dtype[np.intp]],
        *late_bound_chunks: np.ndarray[S, np.dtype[T]],
        late_bound_names: tuple[str, ...],
        late_bound_global: dict[str, int | float | np.number],
        safeguards: Safeguards,
        data_shape: tuple[int, ...],
        depth_: tuple[int | tuple[int, int], ...],
        boundary_: Literal["none", "periodic"],
        block_info=None,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.bool]]:
        assert block_info is not None, "missing block info"
        assert len(late_bound_chunks) == len(late_bound_names), (
            "late-bound chunks and names mismatch"
        )
        assert len(depth_) == data_chunk.ndim, "overlap depth length mismatch"

        late_bound_chunk: dict[str, Value] = dict(
            **late_bound_global,
            **{p: v for p, v in zip(late_bound_names, late_bound_chunks)},
        )

        depth: tuple[tuple[int, int], ...] = tuple(
            (d, d) if isinstance(d, int) else d for d in depth_
        )

        match boundary_:
            case "none":
                # map_overlap passes the incorrect block_info,
                #  see https://github.com/dask/dask/issues/11349
                # so we only trust chunk locations (but not array locations)
                chunk_index: tuple[int, ...] = block_info[None]["chunk-location"]
                num_chunks: tuple[int, ...] = block_info[None]["num-chunks"]

                # the none boundary does not extend beyond the data boundary
                #  and so some stencils may be truncated
                # dask guarantees that the chunks are at least as large as the
                #  stencil, so only the edge chunks have truncated stencils
                chunk_stencil: tuple[
                    tuple[
                        Literal[BoundaryCondition.valid, BoundaryCondition.wrap],
                        NeighbourhoodAxis,
                    ],
                    ...,
                ] = tuple(
                    (
                        BoundaryCondition.valid,
                        NeighbourhoodAxis(
                            before=b if ci > 0 else 0, after=a if (ci + 1) < cn else 0
                        ),
                    )
                    for (b, a), ci, cn in zip(depth, chunk_index, num_chunks)
                )
            case "periodic":
                # the periodic boundary guarantees that we always have the
                #  stencil we asked for
                chunk_stencil = tuple(
                    (BoundaryCondition.wrap, NeighbourhoodAxis(before=b, after=a))
                    for b, a in depth
                )
            case _:
                assert_never(boundary_)

        # extract the indices of the non-stencil-extended data indices chunk
        data_indices_chunk_: np.ndarray[tuple[int, ...], np.dtype[np.intp]] = (
            data_indices_chunk[
                tuple(
                    slice(a.before, None if a.after == 0 else -a.after)
                    for b, a in chunk_stencil
                )
            ]
        )
        chunk_shape = data_indices_chunk_.shape

        # extract the offset of the chunk from the data indices chunk
        if data_indices_chunk_.size > 0:
            chunk_offset_index = int(data_indices_chunk_.flatten()[0])
            chunk_offset_: list[int] = []
            for s in data_shape[::-1]:
                chunk_offset_.append(chunk_offset_index % s)
                chunk_offset_index //= s
            chunk_offset: tuple[int, ...] = tuple(chunk_offset_[::-1])
        else:
            chunk_offset = tuple(0 for _ in data_shape)

        # this is safe because
        # - map_overlap ensures we get chunks including their required stencil
        chunk_is_ok = safeguards.check_chunk(
            data_chunk,
            prediction_chunk,
            data_shape=data_shape,
            chunk_offset=chunk_offset,
            chunk_stencil=chunk_stencil,
            late_bound_chunk=late_bound_chunk,
            where_chunk=True,
        )

        # broadcast the boolean check scalar to the output chunk shape
        return np.broadcast_to(chunk_is_ok, chunk_shape)

    data_indices = dask.array.arange(data.size).reshape(data.shape).rechunk(data.chunks)

    # first check each chunk to find out if any failed the check
    # for stencil safeguards, one chunk failing requires all to be corrected
    #  (since incremental corrections could spread)
    # if no chunk check failed, the chunked correction can take a fast path
    any_chunk_check_failed = (
        not dask.array.map_overlap(
            _check_overlapping_stencil_chunk,
            data.data,
            prediction.data,
            data_indices,
            *chunked_late_bound.values(),
            dtype=np.bool,
            # we cannot output 1x...x1 chunks here since we later use the block
            #  info to compute the data size and chunk location
            chunks=None,
            enforce_ndim=True,
            meta=np.array((), dtype=np.bool),
            depth=depth,
            boundary=boundary,
            trim=False,
            align_arrays=False,
            # if the stencil is larger than the smallest chunk, temporary rechunking may be necessary
            allow_rechunk=True,
            late_bound_names=tuple(chunked_late_bound.keys()),
            late_bound_global=late_bound_global,
            safeguards=safeguards_,
            data_shape=data.shape,
            depth_=depth,
            boundary_=boundary,
        )
        .all()
        .compute()
    )

    def _compute_overlapping_stencil_chunk_correction(
        data_chunk: np.ndarray[S, np.dtype[T]],
        prediction_chunk: np.ndarray[S, np.dtype[T]],
        data_indices_chunk: np.ndarray[S, np.dtype[np.intp]],
        *late_bound_chunks: np.ndarray[S, np.dtype[T]],
        late_bound_names: tuple[str, ...],
        late_bound_global: dict[str, int | float | np.number],
        safeguards: Safeguards,
        data_shape: tuple[int, ...],
        depth_: tuple[int | tuple[int, int], ...],
        boundary_: Literal["none", "periodic"],
        any_chunk_check_failed: bool,
        block_info=None,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.unsignedinteger]]:
        assert block_info is not None, "missing block info"
        assert len(late_bound_chunks) == len(late_bound_names), (
            "late-bound chunks and names mismatch"
        )
        assert len(depth_) == data_chunk.ndim, "overlap depth length mismatch"

        late_bound_chunk: dict[str, Value] = dict(
            **late_bound_global,
            **{p: v for p, v in zip(late_bound_names, late_bound_chunks)},
        )

        depth: tuple[tuple[int, int], ...] = tuple(
            (d, d) if isinstance(d, int) else d for d in depth_
        )

        match boundary_:
            case "none":
                # map_overlap passes the wrong block_info,
                #  see https://github.com/dask/dask/issues/11349
                # so we only trust it just enough to compute the effective
                #  stencil size
                extended_data_shape: tuple[int, ...] = tuple(block_info[None]["shape"])
                extended_chunk_location: tuple[tuple[int, int], ...] = tuple(
                    block_info[None]["array-location"]
                )

                # the none boundary does not extend beyond the data boundary,
                #  so we need to check by how much the stencil was cut off
                chunk_stencil: tuple[
                    tuple[
                        Literal[BoundaryCondition.valid, BoundaryCondition.wrap],
                        NeighbourhoodAxis,
                    ],
                    ...,
                ] = tuple(
                    (
                        BoundaryCondition.valid,
                        NeighbourhoodAxis(
                            before=s - max(0, s - b), after=min(e + a, d) - e
                        ),
                    )
                    for (b, a), (s, e), d in zip(
                        depth, extended_chunk_location, extended_data_shape
                    )
                )
            case "periodic":
                # the periodic boundary guarantees that we always have the
                #  stencil we asked for
                chunk_stencil = tuple(
                    (BoundaryCondition.wrap, NeighbourhoodAxis(before=b, after=a))
                    for b, a in depth
                )
            case _:
                assert_never(boundary_)

        # extract the indices of the non-stencil-extended data indices chunk
        data_indices_chunk_: np.ndarray[tuple[int, ...], np.dtype[np.intp]] = (
            data_indices_chunk[
                tuple(
                    slice(a.before, None if a.after == 0 else -a.after)
                    for b, a in chunk_stencil
                )
            ]
        )
        chunk_shape = data_indices_chunk_.shape

        # extract the offset of the chunk from the data indices chunk
        if data_indices_chunk_.size > 0:
            chunk_offset_index = int(data_indices_chunk_.flatten()[0])
            chunk_offset_: list[int] = []
            for s in data_shape[::-1]:
                chunk_offset_.append(chunk_offset_index % s)
                chunk_offset_index //= s
            chunk_offset: tuple[int, ...] = tuple(chunk_offset_[::-1])
        else:
            chunk_offset = tuple(0 for _ in data_shape)

        # this is safe because
        # - map_overlap ensures we get chunks including their required stencil
        # - compute_chunked_correction only returns the correction for the non-
        #   overlapping non-stencil parts of the chunk
        correction: np.ndarray[tuple[int, ...], np.dtype[np.unsignedinteger]] = (
            safeguards.compute_chunked_correction(
                data_chunk,
                prediction_chunk,
                data_shape=data_shape,
                chunk_offset=chunk_offset,
                chunk_stencil=chunk_stencil,
                any_chunk_check_failed=any_chunk_check_failed,
                late_bound_chunk=late_bound_chunk,
                where_chunk=True,
            )
        )
        assert correction.shape == chunk_shape, "invalid correction chunk shape"

        return correction

    da_correction = (
        data.copy(
            data=dask.array.map_overlap(
                _compute_overlapping_stencil_chunk_correction,
                data.data,
                prediction.data,
                data_indices,
                *chunked_late_bound.values(),
                dtype=correction_dtype,
                chunks=None,
                enforce_ndim=True,
                meta=np.array((), dtype=correction_dtype),
                depth=depth,
                boundary=boundary,
                trim=False,
                align_arrays=False,
                # if the stencil is larger than the smallest chunk, temporary rechunking may be necessary
                allow_rechunk=True,
                late_bound_names=tuple(chunked_late_bound.keys()),
                late_bound_global=late_bound_global,
                safeguards=safeguards_,
                data_shape=data.shape,
                depth_=depth,
                boundary_=boundary,
                any_chunk_check_failed=any_chunk_check_failed,
            )
            .compute_chunk_sizes()  # shape and chunk size may be wrong, recompute
            .rechunk(data.chunks)  # undo temporary rechunking
        )
        .rename(correction_name)
        .assign_attrs(**correction_attrs)
    )

    # drop the previous encoding from the variable but not the coords
    return da_correction._replace(variable=da_correction.variable.drop_encoding())


def apply_data_array_correction(
    prediction: xr.DataArray,
    correction: xr.DataArray,
) -> xr.DataArray:
    """
    Apply the `correction` to the `prediction` array to satisfy the safeguards for which the `correction` was produced.

    The `prediction` must be bitwise equivalent to the `prediction` that
    was used to produce the `correction`.

    The `prediction` array may be chunked[^2] and the `correction` array must
    use the same chunking, though this chunking may differ from the one that
    was used to produce the `correction`.

    If the the `prediction` array is chunked, the correction is applied lazily,
    otherwise it its application is computed eagerly.

    [^2]: Any chunking supported by `xarray` is supported, including but not
    limited to `dask`, please see
    <https://docs.xarray.dev/en/stable/internals/chunked-arrays.html>.

    Parameters
    ----------
    prediction : xr.DataArray
        The prediction array for which the correction has been produced.
    correction : xr.DataArray
        The correction array.

    Returns
    -------
    corrected : xr.DataArray
        The corrected array, which satisfies the safeguards.

    Raises
    ------
    ValueError
        if the `correction`'s dimensions, shape, or chunks do not match the
        `prediction`'s, or if the `correction`'s dtype does not match the
        correction dtype for the `prediction`'s dtype.
    ValueError
        if the `correction` does not contain metadata about the safeguards that
        it was produced with.
    ...
        if re-instantiating the safeguards from their configuration metadata
        raises an exception.
    """

    with ctx.parameter("correction"):
        if correction.dims != prediction.dims:
            raise ValueError("correction.dims must match prediction.dims") | ctx
        if correction.shape != prediction.shape:
            raise ValueError("correction.shape must match prediction.shape") | ctx
        if correction.chunks != prediction.chunks:
            raise ValueError("correction.chunks must match prediction.chunks") | ctx

        if "safeguards" not in correction.attrs:
            raise (
                ValueError(
                    "correction must contain metadata about the safeguards "
                    + "that it was produced with"
                )
                | ctx
            )
    safeguards = Safeguards.from_config(json.loads(correction.attrs["safeguards"]))

    with ctx.parameter("correction"):
        if correction.dtype != safeguards.correction_dtype_for_data(prediction.dtype):
            raise (
                ValueError(
                    "correction.dtype must match the correction dtype for "
                    + "prediction.dtype"
                )
                | ctx
            )

    def _apply_independent_chunk_correction(
        prediction_chunk: xr.DataArray,
        correction_chunk: xr.DataArray,
        safeguards: Safeguards,
    ) -> xr.DataArray:
        return prediction_chunk.copy(
            data=safeguards.apply_correction(
                prediction_chunk.values, correction_chunk.values
            )
        )

    return xr.map_blocks(
        _apply_independent_chunk_correction,
        prediction,
        args=(correction,),
        kwargs=dict(safeguards=safeguards),
        template=prediction,
    ).assign_attrs(safeguards=correction.attrs["safeguards"])


@xr.register_dataset_accessor("safeguarded")
class DatasetSafeguardedAccessor:
    """
    An extension for an [`xarray.Dataset`][xarray.Dataset] that provides the
    `.safeguarded` property that applies safeguards corrections in the dataset
    to their respective variables.

    The safeguarded variables must be bitwise equivalent to the variables
    for which their corrections were originally produced.

    For instance, for a dataset `ds` that contains both a variable `ds.foo` and
    its correction, you can access the corrected variable using
    `ds.safeguarded.foo`.

    Raises
    ------
    AttributeError
        if the dataset does not contain not-yet-applied safeguards corrections.
    RuntimeError
        if the attributes of the safeguards correction are invalid.
    ValueError
        if applying a safeguards correction raises an exception.
    ...
        if re-instantiating the safeguards from their configuration metadata
        raises an exception.
    """

    __slots__: tuple[str, ...] = ()

    def __new__(cls, ds: xr.Dataset) -> xr.Dataset:  # type: ignore
        corrected: dict[str, xr.DataArray] = dict()

        for vn, v in ds.data_vars.items():
            if "safeguarded" in v.attrs:
                kp = v.attrs["safeguarded"]
                with ctx.parameter(repr(vn)):
                    if kp is None:
                        raise (
                            RuntimeError(
                                "data variable is a safeguards correction but "
                                + "its `safeguarded` attribute must not be None"
                            )
                            | ctx
                        )
                    if kp not in ds.data_vars:
                        raise (
                            RuntimeError(
                                "data variable is a safeguards correction for "
                                + f"{kp!r} but there is no variable of that "
                                + "name"
                            )
                            | ctx
                        )
                    if "safeguards" not in v.attrs:
                        raise (
                            RuntimeError(
                                "data variable is missing the `safeguards` "
                                + "attribute"
                            )
                            | ctx
                        )

                corrected[kp] = apply_data_array_correction(ds.data_vars[kp], v)

        if len(corrected) == 0:
            raise (
                AttributeError(
                    "not a dataset with not-yet-applied safeguards corrections"
                )
                | ctx
            )

        return xr.Dataset(corrected, attrs=ds.attrs)


@xr.register_dataarray_accessor("safeguards")
class DataArraySafeguardsAccessor:
    """
    An extension for an [`xarray.DataArray`][xarray.DataArray] that provides
    the `.safeguards` property that exposes the collection of
    [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s for
    a safeguards correction or corrected array.

    Raises
    ------
    AttributeError
        if the data array does not have associated safeguards.
    ...
        if re-instantiating the safeguards from their configuration metadata
        raises an exception.
    """

    __slots__: tuple[str, ...] = ()

    def __new__(cls, da: xr.DataArray) -> Collection[Safeguard]:  # type: ignore
        if "safeguards" not in da.attrs:
            raise AttributeError("not a data array with safeguards") | ctx
        return Safeguards.from_config(json.loads(da.attrs["safeguards"])).safeguards
