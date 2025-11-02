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
    from typing import ClassVar, Generic

    import numcodecs
    import numcodecs.compat
    import numcodecs.registry
    import numpy as np
    from numcodecs.abc import Codec
    from numcodecs_safeguards import SafeguardsCodec
    from typing_extensions import override  # MSPV 3.12

    from compression_safeguards import SafeguardKind, Safeguards
    from compression_safeguards.safeguards.abc import Safeguard
    from compression_safeguards.safeguards.combinators.select import SelectSafeguard
    from compression_safeguards.safeguards.pointwise.sign import SignPreservingSafeguard
    from compression_safeguards.safeguards.qois import (
        PointwiseQuantityOfInterestExpression,
        StencilQuantityOfInterestExpression,
    )
    from compression_safeguards.safeguards.stencil import NeighbourhoodBoundaryAxis
    from compression_safeguards.utils._compat import _ensure_array
    from compression_safeguards.utils.bindings import Parameter
    from compression_safeguards.utils.error import (
        ErrorContextMixin,
        IndexContextLayer,
        LateBoundParameterContextLayer,
        ParameterContextLayer,
        SafeguardTypeContextLayer,
    )
    from compression_safeguards.utils.typing import S, T


warnings.filterwarnings("error")


np.set_printoptions(floatmode="unique")


# the fuzzer *somehow* messes up np.nanmin and np.nanmax, so patch them
def nanmin(x: np.ndarray[S, np.dtype[T]]) -> T:
    x = _ensure_array(x)
    if np.all(np.isnan(x)):
        warnings.warn("All-NaN slice encountered", RuntimeWarning)
        return x.dtype.type(np.nan)
    if np.any(np.isnan(x)):
        x = _ensure_array(x, copy=True)
        x[np.isnan(x)] = np.inf
    return np.amin(x)


np.nanmin = nanmin


def nanmax(x: np.ndarray[S, np.dtype[T]]) -> T:
    x = _ensure_array(x)
    if np.all(np.isnan(x)):
        warnings.warn("All-NaN slice encountered", RuntimeWarning)
        return x.dtype.type(np.nan)
    if np.any(np.isnan(x)):
        x = _ensure_array(x, copy=True)
        x[np.isnan(x)] = -np.inf
    return np.amax(x)


np.nanmax = nanmax


class FuzzCodec(Codec, Generic[S, T]):
    __slots__: tuple[str, ...] = ("data", "decoded")
    data: np.ndarray[S, np.dtype[T]]
    decoded: np.ndarray[S, np.dtype[T]]

    codec_id: ClassVar[str] = "fuzz"  # type: ignore

    def __init__(
        self, data: np.ndarray[S, np.dtype[T]], decoded: np.ndarray[S, np.dtype[T]]
    ):
        self.data = data
        self.decoded = decoded

    def encode(self, buf):
        return b""

    def decode(self, buf, out=None):
        assert len(buf) == 0
        return numcodecs.compat.ndarray_copy(
            _ensure_array(self.decoded, copy=True), out
        )

    def get_config(self):
        return dict(id=type(self).codec_id, data=self.data, decoded=self.decoded)

    @override
    def __repr__(self):
        config = {k: v for k, v in self.get_config().items() if k != "id"}
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in config.items())})"


numcodecs.registry.register_codec(FuzzCodec)


def generate_parameter(
    data: atheris.FuzzedDataProvider, ty: type, depth: int, late_bound: set[str]
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
                generate_parameter(data, typing.get_args(ty)[0], depth, late_bound)
                for _ in range(data.ConsumeIntInRange(0, 3 - depth))
            ]

    if typing.get_origin(ty) in (typing.Union, types.UnionType):
        tys = typing.get_args(ty)

        if len(tys) == 2 and tys[0] is str and issubclass(tys[1], Enum):
            return list(tys[1])[data.ConsumeIntInRange(0, len(tys[1]) - 1)]

        if (
            len(tys) == 2
            and (tys[0] is dict or typing.get_origin(tys[0]) is dict)
            and tys[1] is NeighbourhoodBoundaryAxis
        ):
            return {
                p: generate_parameter(data, v.annotation, depth, late_bound)
                for p, v in signature(NeighbourhoodBoundaryAxis).parameters.items()
            }

        if len(tys) == 2 and tys[0] is str and tys[1] is Parameter:
            i = data.ConsumeIntInRange(0, 3)
            if i == 0:
                p = data.ConsumeString(2)
                late_bound.add(p)
                return p
            return ["$x", "$x_min", "$x_max"][i - 1]

        if (
            len(tys) > 1
            and (tys[0] is dict or typing.get_origin(tys[0]) is dict)
            and all(issubclass(t, Safeguard) for t in tys[1:])
        ):
            return generate_safeguard_config(data, depth + 1, late_bound)

        if len(tys) > 2 and str in tys and Parameter in tys:
            # ensure that str | Parameter stay together during the union pick
            tys = tuple(t for t in tys if t not in (str, Parameter)) + (
                str | Parameter,
            )

        ty = tys[data.ConsumeIntInRange(0, len(tys) - 1)]

        return generate_parameter(data, ty, depth, late_bound)

    if ty in (
        PointwiseQuantityOfInterestExpression,
        StencilQuantityOfInterestExpression,
    ):

        def consume_float_str(data: atheris.FuzzedDataProvider) -> str:
            return str(data.ConsumeFloat()).replace("nan", "NaN").replace("inf", "Inf")

        ATOMS = [
            # number literals
            int,
            float,
            # pointwise data
            "x",
            # constants
            "e",
            "pi",
            # variables
            "variable",
        ]
        OPS = {
            # variable assignment
            "assignment": 2,
            # unary operators
            "negate": 1,
            # binary operators
            "+": 2,
            "-": 2,
            "*": 2,
            "/": 2,
            "**": 2,
            # logarithms and exponentials
            "ln": 1,
            "log2": 1,
            "log10": 1,
            "log": 2,
            "exp": 1,
            "exp2": 1,
            "exp10": 1,
            # exponentiation
            "sqrt": 1,
            "square": 1,
            "reciprocal": 1,
            # absolute value
            "abs": 1,
            # sign and rounding
            "sign": 1,
            "floor": 1,
            "ceil": 1,
            "trunc": 1,
            "round_ties_even": 1,
            # trigonometric
            "sin": 1,
            "cos": 1,
            "tan": 1,
            "asin": 1,
            "acos": 1,
            "atan": 1,
            # hyperbolic
            "sinh": 1,
            "cosh": 1,
            "tanh": 1,
            "asinh": 1,
            "acosh": 1,
            "atanh": 1,
            # classification
            "isfinite": 1,
            "isinf": 1,
            "isnan": 1,
            # conditional
            "where": 3,
        }

        if ty is StencilQuantityOfInterestExpression:
            ATOMS += [
                # data stencil neighbourhood
                "X",
            ]
            OPS = {
                **OPS,
                # stencil centre index
                "I": 1,
                # array index
                "indexI": 1,
                "index1": 2,
                "index2": 3,
                # array transpose
                "transpose": 1,
                # array operations
                "size": 1,
                "sum": 1,
                "matmul": 2,
                # finite difference
                "finite_difference": 1,
            }

        atoms = []
        for _ in range(data.ConsumeIntInRange(2, 4)):
            atom = ATOMS[data.ConsumeIntInRange(0, len(ATOMS) - 1)]
            if atom is int:
                atom = str(data.ConsumeInt(2))
            elif atom is float:
                atom = consume_float_str(data)
            elif atom == "variable":
                atom = f'{"v" if ty is PointwiseQuantityOfInterestExpression else "V"}["{data.ConsumeString(2)}"]'
            atoms.append(atom)

        done = False
        assignments = []
        while not done:
            done = len(atoms) == 1
            op = list(OPS.keys())[data.ConsumeIntInRange(0, len(OPS) - 1)]
            atom1, *atomn = [
                atoms.pop() if len(atoms) > 0 else "1" for _ in range(OPS[op])
            ]

            if op == "assignment":
                assignments.append(
                    f'{"v" if ty is PointwiseQuantityOfInterestExpression else "V"}["{data.ConsumeString(2)}"] = {atomn[0]};'
                )
                atoms.append(atom1)
            elif op == "neg":
                atoms.append(f"-({atom1})")
            elif op in ["+", "-", "*", "/", "**"]:
                atoms.append(f"{atom1} {op} {atomn[0]}")
            elif op == "log":
                atoms.append(f"log({atom1}, base={atomn[0]})")
            elif op == "I":
                atoms.append(f"I[{data.ConsumeIntInRange(0, 1)}]")
            elif op == "indexI":
                atoms.append(f"({atom1})[I]")
            elif op == "index1":
                atoms.append(f"({atom1})[{atomn[0]}]")
            elif op == "index2":
                atoms.append(f"({atom1})[{atomn[0]}, {atomn[1]}]")
            elif op == "transpose":
                atoms.append(f"({atom1}).T")
            elif op == "finite_difference":
                atoms.append(
                    f"finite_difference({atom1}, order={data.ConsumeIntInRange(0, 3)}, accuracy={data.ConsumeIntInRange(1, 4)}, type={data.ConsumeIntInRange(-1, 1)}, axis={data.ConsumeIntInRange(0, 1)}, grid_spacing={consume_float_str(data)})"
                )
            else:
                atoms.append(f"{op}({atom1}, {', '.join(atomn)})")
        [atom] = atoms

        if len(assignments) == 0:
            return atom

        nl = "\n"
        return f"{nl.join(assignments)}\nreturn {atom};"

    assert False, f"unknown parameter type {ty!r}"


def generate_safeguard_config(
    data: atheris.FuzzedDataProvider, depth: int, late_bound: set[str]
):
    kind = list(SafeguardKind)[data.ConsumeIntInRange(0, len(SafeguardKind) - 1)]

    return {
        "kind": kind.name,
        **{
            p: generate_parameter(data, v.annotation, depth, late_bound)
            for p, v in signature(kind.value).parameters.items()
        },
    }


def check_one_input(data) -> None:
    data = atheris.FuzzedDataProvider(data)

    late_bound: set[str] = set()

    safeguards = [
        generate_safeguard_config(data, 0, late_bound)
        for _ in range(data.ConsumeIntInRange(0, 8))
    ]

    dtype: np.dtype[np.number] = np.dtype(
        sorted([d.name for d in Safeguards.supported_dtypes()])[
            data.ConsumeIntInRange(0, len(Safeguards.supported_dtypes()) - 1)
        ]
    )
    sizea: int = data.ConsumeIntInRange(0, 20)
    sizeb: int = data.ConsumeIntInRange(0, 20 // max(1, sizea))
    size = sizea * sizeb

    # input data and the decoded data
    raw = data.ConsumeBytes(size * dtype.itemsize)
    decoded = data.ConsumeBytes(size * dtype.itemsize)

    if len(raw) != size * dtype.itemsize:
        return

    if len(decoded) != size * dtype.itemsize:
        return

    raw = np.frombuffer(raw, dtype=dtype)
    decoded = np.frombuffer(decoded, dtype=dtype)

    if sizeb != 0:
        raw = raw.reshape((sizea, sizeb))
        decoded = decoded.reshape((sizea, sizeb))

    fixed_constants = dict()
    for p in late_bound:
        c = data.ConsumeIntInRange(0, 4)
        if c == 0:
            fixed_constants[p] = data.ConsumeInt(1)
        elif c == 1:
            fixed_constants[p] = data.ConsumeFloat()
        elif c == 2:
            b = data.ConsumeBytes(size * np.dtype(int).itemsize)
            if len(b) != size * np.dtype(int).itemsize:
                return
            fixed_constants[p] = np.frombuffer(b, dtype=int).reshape(raw.shape)
        elif c == 3:
            b = data.ConsumeBytes(size * np.dtype(float).itemsize)
            if len(b) != size * np.dtype(float).itemsize:
                return
            fixed_constants[p] = np.frombuffer(b, dtype=float).reshape(raw.shape)
        else:
            b = data.ConsumeBytes(size * dtype.itemsize)
            if len(b) != size * dtype.itemsize:
                return
            fixed_constants[p] = np.frombuffer(b, dtype=dtype).reshape(raw.shape)

    try:
        with timeout(1):
            safeguard = SafeguardsCodec(
                codec=FuzzCodec(raw, decoded),
                safeguards=safeguards,
                fixed_constants=fixed_constants,
            )
    except (ValueError, TypeError, SyntaxError, TimeoutError):
        return
    except RuntimeWarning as err:
        # skip expressions that try to perform a**b with excessive digits
        if ("symbolic integer evaluation" in str(err)) and (
            "excessive number of digits" in str(err)
        ):
            return
        raise

    grepr = repr(safeguard)
    gconfig = safeguard.get_config()

    safeguard = numcodecs.registry.get_codec(gconfig)
    assert safeguard.get_config() == gconfig, (
        f"{safeguard.get_config()!r} vs {gconfig!r}"
    )
    assert repr(safeguard) == grepr, f"{safeguard!r} vs {grepr}"

    try:
        encoded = safeguard.encode(raw)
        safeguard.decode(encoded, out=np.empty_like(raw))
    except Exception as err:
        if isinstance(err, ErrorContextMixin):
            match err.context.layers:
                case (
                    *_,
                    ParameterContextLayer("neighbourhood"),
                    IndexContextLayer(_),
                    ParameterContextLayer("axis"),
                ) if isinstance(err, IndexError) and (
                    "is out of bounds for array of shape" in str(err)
                ):
                    return
                case (
                    *_,
                    ParameterContextLayer("neighbourhood"),
                    IndexContextLayer(_),
                    ParameterContextLayer("axis"),
                ) | (
                    *_,
                    ParameterContextLayer("eb"),
                    LateBoundParameterContextLayer(_),
                ) if (
                    isinstance(err, IndexError)
                    and ("duplicate axis index" in str(err))
                    and ("normalised to" in str(err))
                    and ("for array of shape" in str(err))
                ):
                    return
                case (*_, ParameterContextLayer(_)) | (
                    *_,
                    ParameterContextLayer(_),
                    LateBoundParameterContextLayer(_),
                ) if isinstance(err, TypeError | ValueError) and (
                    "cannot losslessly cast" in str(err)
                ):
                    return
                case (
                    *_,
                    ParameterContextLayer("eb"),
                    LateBoundParameterContextLayer(_),
                ) if (
                    isinstance(err, ValueError)
                    and ("cannot cast non-finite" in str(err))
                    and ("to saturating finite" in str(err))
                ):
                    return
                case (
                    *_,
                    SafeguardTypeContextLayer(safeguard),
                    ParameterContextLayer("selector"),
                    LateBoundParameterContextLayer(_),
                ) if (
                    isinstance(err, ValueError)
                    and ("invalid entry in choice array" in str(err))
                    and safeguard is SelectSafeguard
                ):
                    return
                case (
                    *_,
                    ParameterContextLayer("eb"),
                    LateBoundParameterContextLayer(_),
                ) if isinstance(err, ValueError) and ("must be" in str(err)):
                    return
                case (
                    *_,
                    SafeguardTypeContextLayer(safeguard),
                    ParameterContextLayer("offset"),
                    LateBoundParameterContextLayer(_),
                ) if (
                    isinstance(err, ValueError)
                    and ("must not contain any NaN values" in str(err))
                    and safeguard is SignPreservingSafeguard
                ):
                    return
                case (
                    *_,
                    ParameterContextLayer(_),
                    LateBoundParameterContextLayer(_),
                ) if (
                    isinstance(err, ValueError)
                    and ("cannot broadcast from shape" in str(err))
                    and ("to shape ()" in str(err))
                ):
                    return
                case _:
                    pass
        print(f"\n===\n\ncodec = {grepr}\n\n===\n")  # noqa: T201
        raise

    # test using the safeguards with the zero codec
    safeguard = SafeguardsCodec(
        codec=dict(id="zero"),
        safeguards=safeguards,
        fixed_constants=fixed_constants,
    )

    grepr = repr(safeguard)
    gconfig = safeguard.get_config()

    safeguard = numcodecs.registry.get_codec(gconfig)
    assert safeguard.get_config() == gconfig

    try:
        encoded = safeguard.encode(raw)
        safeguard.decode(encoded, out=np.empty_like(raw))
    except Exception:
        print(f"\n===\n\ncodec = {grepr}\n\ndata = {raw!r}\n\n===\n")  # noqa: T201
        raise


atheris.Setup(sys.argv, check_one_input)
atheris.Fuzz()
