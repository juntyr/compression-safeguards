import functools
from typing import Callable

import numpy as np
import sympy as sp
import sympy.tensor.array.expressions  # noqa: F401
from typing_extensions import Never  # MSPV 3.11

from ...utils.cast import _float128, _float128_dtype, _float128_precision, _sign


def sympy_expr_to_numpy(
    symbols: list[sp.Symbol | sp.tensor.array.expressions.ArraySymbol],
    expr: sp.Basic,
    dtype: np.dtype,
) -> Callable[..., np.ndarray]:
    """
    Compile the SymPy expression `expr` over a list of `symbols` into a
    function that uses NumPy for numerical evaluation.

    The function evaluates to a numpy array of the provided `dtype` if all
    variable inputs are numpy arrays of the same `dtype`.
    """

    return sp.lambdify(
        symbols,
        expr,
        modules=(
            (
                # polyfill operations that numpy_quaddtype does not yet support
                #  but that numpy supports (otherwise sympy polyfills)
                [
                    dict(
                        # special functions
                        sign=_sign,
                        # hyperbolic functions
                        sinh=lambda x: (np.exp(x) - np.exp(-x)) / 2,
                        cosh=lambda x: (np.exp(x) + np.exp(-x)) / 2,
                        tanh=lambda x: (np.exp(x * 2) - 1) / (np.exp(x * 2) + 1),
                        arcsinh=lambda x: np.log(x + np.sqrt(x**2 + 1)),
                        arccosh=lambda x: np.log(x + np.sqrt(x**2 - 1)),
                        arctanh=lambda x: np.log((1 + x) / (1 - x)) / 2,
                    )
                ]
                if dtype == _float128_dtype
                else []
            )
            + ["numpy"]
            + ([{_float128_dtype.name: _float128}] if dtype == _float128_dtype else [])
        ),
        printer=_create_sympy_numpy_printer_class(dtype),
        docstring_limit=0,
    )


@functools.cache
def _create_sympy_numpy_printer_class(
    dtype: np.dtype,
) -> type[sp.printing.numpy.NumPyPrinter]:
    """
    Create a SymPy to NumPy printer class that outputs numerical values and
    constants with the provided `dtype` and sufficient precision.
    """

    class NumPyDtypePrinter(sp.printing.numpy.NumPyPrinter):
        __slots__ = ("_dtype",)

        # remove default printing of known constants
        _kc: dict = dict()

        def __init__(self, settings=None) -> None:
            self._dtype = dtype.name
            if settings is None:
                settings = dict()
            settings["fully_qualified_modules"] = False
            if dtype == _float128_dtype:
                settings["precision"] = _float128_precision * 2
            else:
                settings["precision"] = np.finfo(dtype).precision * 2
            super().__init__(settings)

        def _print_Integer(self, expr) -> str:
            return f"{self._dtype}({str(expr.p)!r})"

        def _print_Rational(self, expr) -> str:
            return f"{self._dtype}({str(expr.p)!r}) / {self._dtype}({str(expr.q)!r})"

        def _print_Float(self, expr) -> str:
            # explicitly create the float from its string representation
            #  e.g. 1.2 -> float16('1.2')
            s = super()._print_Float(expr)
            return f"{self._dtype}({s!r})"

        def _print_Exp1(self, expr) -> str:
            return self._print_NumberSymbol(expr)

        def _print_Pi(self, expr) -> str:
            return self._print_NumberSymbol(expr)

        def _print_NaN(self, expr) -> str:  # type: ignore
            return f"{self._dtype}(nan)"

        def _print_Infinity(self, expr) -> str:  # type: ignore
            return f"{self._dtype}(inf)"

        def _print_ImaginaryUnit(self, expr) -> Never:
            raise ValueError(
                "cannot evaluate an expression containing an imaginary number"
            )

        def _print_ArrayElement(self, expr) -> str:
            indices = []
            for i in expr.indices:
                assert isinstance(i, sp.Integer), (
                    "data neighbourhood only supports integer indices"
                )
                indices.append(i.p)
            printed = f"{expr.name}[..., {', '.join([str(i) for i in indices])}]"
            return printed

        def _print_trunc(self, expr) -> str:
            (x,) = expr.args
            return f"trunc({self._print(x)})"

        def _print_round_ties_even(self, expr) -> str:
            (x,) = expr.args
            return f"rint({self._print(x)})"

    return NumPyDtypePrinter
