import atheris
from timeoutcontext import timeout

with atheris.instrument_imports():
    import sys
    import types
    import typing
    import warnings
    from collections.abc import Collection, Sequence
    from enum import Enum
    from inspect import signature

    import numpy as np
    import xarray as xr
    from xarray_safeguards import produce_data_array_correction

    from compression_safeguards.api import Safeguards
    from compression_safeguards.safeguards._qois.expr.hashing import (
        HashingExpr,
        _patch_for_hashing_qoi_dev_only,
    )
    from compression_safeguards.safeguards.qois import (
        StencilQuantityOfInterestExpression,
    )
    from compression_safeguards.safeguards.stencil import NeighbourhoodBoundaryAxis
    from compression_safeguards.safeguards.stencil.qoi.eb import (
        StencilQuantityOfInterestErrorBoundSafeguard,
    )
    from compression_safeguards.utils.bindings import Parameter


warnings.filterwarnings("error")


np.set_printoptions(floatmode="unique")


_patch_for_hashing_qoi_dev_only().__enter__()


def generate_parameter(
    data: atheris.FuzzedDataProvider, ty: type, depth: int, late_bound_params: set[str]
):
    if ty is types.NoneType:
        return None
    if ty is float:
        return data.ConsumeFloat()
    if ty is int:
        return data.ConsumeInt(1)
    if ty is bool:
        return data.ConsumeBool()

    if typing.get_origin(ty) in (Collection, Sequence):
        if len(typing.get_args(ty)) == 1:
            return [
                generate_parameter(
                    data, typing.get_args(ty)[0], depth, late_bound_params
                )
                for _ in range(data.ConsumeIntInRange(0, 3 - depth))
            ]

    if typing.get_origin(ty) in (typing.Union, types.UnionType):
        tys = typing.get_args(ty)

        if len(tys) == 2 and tys[0] is str and issubclass(tys[1], Enum):
            return list(tys[1])[data.ConsumeIntInRange(0, len(tys[1]) - 1)]

        if len(tys) == 2 and tys[0] is dict and tys[1] is NeighbourhoodBoundaryAxis:
            return {
                p: generate_parameter(data, v.annotation, depth, late_bound_params)
                for p, v in signature(NeighbourhoodBoundaryAxis).parameters.items()
            }

        if len(tys) == 2 and tys[0] is str and tys[1] is Parameter:
            i = data.ConsumeIntInRange(0, 3)
            if i == 0:
                p = data.ConsumeString(2)
                late_bound_params.add(p)
                return p
            return ["$x", "$x_min", "$x_max"][i - 1]

        if len(tys) > 2 and str in tys and Parameter in tys:
            # ensure that str | Parameter stay together during the union pick
            tys = tuple(t for t in tys if t not in (str, Parameter)) + (
                str | Parameter,
            )

        ty = tys[data.ConsumeIntInRange(0, len(tys) - 1)]

        return generate_parameter(data, ty, depth, late_bound_params)

    if ty is StencilQuantityOfInterestExpression:
        return "x"

    assert False, f"unknown parameter type {ty!r}"


def check_one_input(data) -> None:
    data = atheris.FuzzedDataProvider(data)

    late_bound_params: set[str] = {"foo"}

    safeguard_config = {
        p: generate_parameter(data, v.annotation, 0, late_bound_params)
        for p, v in signature(
            StencilQuantityOfInterestErrorBoundSafeguard
        ).parameters.items()
        if p != "qoi"
    }

    dtype: np.dtype = np.dtype(
        sorted([d.name for d in Safeguards.supported_dtypes()])[
            data.ConsumeIntInRange(0, len(Safeguards.supported_dtypes()) - 1)
        ]
    )
    sizea: int = data.ConsumeIntInRange(0, 20)
    sizeb: int = data.ConsumeIntInRange(0, 20 // max(1, sizea))
    size = sizea * sizeb

    chunksa: int = data.ConsumeIntInRange(1, sizea) if sizea > 0 else 0
    chunksb: int = data.ConsumeIntInRange(1, sizeb) if sizeb > 0 else 0
    chunks = dict(a=chunksa)

    # input data and the decoded data
    raw = data.ConsumeBytes(size * dtype.itemsize)

    if len(raw) != size * dtype.itemsize:
        return

    raw = np.frombuffer(raw, dtype=dtype)

    # skip all-zero raw inputs since we use that to sidechannel-communicate in
    #  the fuzzer if a prediction is already corrected
    if np.all(raw == 0):
        return

    if sizeb != 0:
        raw = raw.reshape((sizea, sizeb))
        chunks = dict(a=chunksa, b=chunksb)

    late_bound = dict()
    for p in late_bound_params:
        c = data.ConsumeIntInRange(0, 4)
        if c == 0:
            late_bound[p] = data.ConsumeInt(1)
        elif c == 1:
            late_bound[p] = data.ConsumeFloat()
        elif c == 2:
            b = data.ConsumeBytes(size * np.dtype(int).itemsize)
            if len(b) != size * np.dtype(int).itemsize:
                return
            late_bound[p] = np.frombuffer(b, dtype=int).reshape(raw.shape)
        elif c == 3:
            b = data.ConsumeBytes(size * np.dtype(float).itemsize)
            if len(b) != size * np.dtype(float).itemsize:
                return
            late_bound[p] = np.frombuffer(b, dtype=float).reshape(raw.shape)
        else:
            b = data.ConsumeBytes(size * dtype.itemsize)
            if len(b) != size * dtype.itemsize:
                return
            late_bound[p] = np.frombuffer(b, dtype=dtype).reshape(raw.shape)
    late_bound_da = {
        p: xr.DataArray(v, dims=list(chunks.keys())).chunk(chunks)
        if isinstance(v, np.ndarray)
        else v
        for p, v in late_bound.items()
    }

    try:
        with timeout(1):
            safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
                qoi="x",
                **safeguard_config,
            )
            safeguard._qoi_expr._expr = HashingExpr.from_data_shape(
                data_shape=safeguard._qoi_expr._stencil_shape,
                late_bound_constants=frozenset(["foo"]),
            )
            safeguard._qoi_expr._late_bound_constants = (
                safeguard._qoi_expr._expr.late_bound_constants
            )
    except (AssertionError, Warning, TimeoutError):
        return

    # xarray-safeguards provides `$x_min` and `$x_max`,
    #  but the compression-safeguards do not
    if "$x_min" in safeguard.late_bound:
        late_bound["$x_min"] = (
            np.nanmin(raw)
            if raw.size > 0 and not np.all(np.isnan(raw))
            else np.array(0, dtype=raw.dtype)
        )
    if "$x_max" in safeguard.late_bound:
        late_bound["$x_max"] = (
            np.nanmax(raw)
            if raw.size > 0 and not np.all(np.isnan(raw))
            else np.array(0, dtype=raw.dtype)
        )

    da = xr.DataArray(raw, dims=list(chunks.keys())).chunk(chunks)
    da_prediction = xr.DataArray(np.zeros_like(raw), dims=list(chunks.keys())).chunk(
        chunks
    )

    try:
        global_hash = Safeguards(safeguards=[safeguard]).compute_correction(
            data=da.values, prediction=da_prediction.values, late_bound=late_bound
        )
        chunked_hash = produce_data_array_correction(
            data=da,
            prediction=da_prediction,
            safeguards=[safeguard],
            late_bound=late_bound_da,
        )
        np.testing.assert_array_equal(global_hash, chunked_hash.values)
    except Exception as err:
        if (
            (
                isinstance(err, IndexError)
                and ("axis index" in str(err))
                and ("out of bounds for array of shape" in str(err))
            )
            or (
                isinstance(err, IndexError)
                and ("duplicate axis index" in str(err))
                and ("normalised to" in str(err))
                and ("for array of shape" in str(err))
            )
            or (
                isinstance(err, TypeError | ValueError)
                and ("cannot losslessly cast" in str(err))
            )
            or (
                isinstance(err, ValueError)
                and ("cannot cast non-finite" in str(err))
                and ("to saturating finite" in str(err))
            )
            or (isinstance(err, AssertionError) and str(err).startswith("eb must be"))
            or (
                isinstance(err, AssertionError)
                and ("fuzzer hash is all zeros" in str(err))
            )
        ):
            return
        print(  # noqa: T201
            f"\n===\n\nsafeguard = {safeguard!r}\n\ndata = {raw!r}\n\nchunks = {chunks!r}\n\n===\n"
        )
        raise err


atheris.Setup(sys.argv, check_one_input)
atheris.Fuzz()
