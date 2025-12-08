"""
## Safeguarding Quantities of Interest

We are often not just interested in data itself, but also in quantities derived
from it. For instance, we might later plot the data logarithm, compute a
derivative, or apply a smoothing kernel. In these cases, we often want to
safeguard not just properties on the data but also on these derived quantities
of interest (QoIs).

The `compression-safeguards` package provides the
[`PointwiseQuantityOfInterestErrorBoundSafeguard`][..pointwise.qoi.eb.PointwiseQuantityOfInterestErrorBoundSafeguard]
and
[`StencilQuantityOfInterestErrorBoundSafeguard`][..stencil.qoi.eb.StencilQuantityOfInterestErrorBoundSafeguard]
safeguards to preserve various [`ErrorBound`][..eb.ErrorBound]s on pointwise[^1]
and stencil[^2] quantities of interest, respectively.

[^1]: A pointwise QoI is computed independently for each data point, taking
    only the value of this data point as input.
[^2]: A stencil QoI is computed for a local neighbourhood or stencil around
    each data point, e.g. to compute a 5x5 2D block mean over the data.

## Grammar

These QoI safeguards are configured with the quantity of interest expression,
which must be given in [`str`][str]ing form using the following EBNF
grammar[^3], where some rules are only available in pointwise or in stencil
QoIs:

[^3]: You can visualise the EBNF grammar at <https://matthijsgroen.github.io/ebnf2railroad/try-yourself.html>.

```ebnf
qoi =
    expr
  | { assignment }, "return", expr, ";"
;

assignment =
    variable, "=", expr, ";"
;

expr =
    number
  | array
  | unary_operator
  | binary_operator
  | binary_comparison
  | array_transpose
  | subexpression
  | constant
  | data
  | late_bound_constant
  | variable
  | array_indexing
  | arithmetic_functions
  | classification_functions
  | logical_combinators
  | array_functions
  | finite_difference
;

number =
    integer
  | float
  | "Inf"  (* infinity *)
  | "NaN"  (* not a number *)
;

integer =
    [ sign ], digit, { digit }
;
float =
    [ sign ], digit, { digit }, [
        ".", digit, { digit }
    ], [
        "e", [ sign ], digit, { digit }
    ]
;

sign =
    "+" | "-"
;
digit =
    "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
;

array =
    "[", expr, { ",", expr }, [","], "]"
;

unary_operator =
    "+", expr  (* positive / no-op *)
  | "-", expr  (* negation *)
;

binary_operator =
    expr, "+", expr  (* addition *)
  | expr, "-", expr  (* subtraction *)
  | expr, "*", expr  (* multiplication *)
  | expr, "/", expr  (* division *)
  | expr, "**", expr  (* exponentiation / power *)
;

binary_comparison =
    expr, "<", expr  (* 1 if a less than b, 0 otherwise (e.g. with NaN) *)
  | expr, "<=", expr  (* 1 if a less than or equal to b, 0 otherwise (e.g. with NaN) *)
  | expr, "==", expr  (* 1 if a equal to b, 0 otherwise (e.g. with NaN) *)
  | expr, "!=", expr  (* 1 if a not equal to b, 0 otherwise; 1 for NaN != NaN *)
  | expr, ">=", expr  (* 1 if a greater than or equal to b, 0 otherwise (e.g. with NaN) *)
  | expr, ">", expr  (* 1 if a greater than b, 0 otherwise (e.g. with NaN) *)
;

array_transpose =
    expr, ".T"
;

subexpression =
    "(", expr, ")"
;

constant =
    "e"  (* Euler's number *)
  | "pi"  (* pi *)
;

data =
    "x"  (* pointwise data value, x = X[I] *)
  | "X"  (* stencil data neighbourhood, only available in stencil QoIs *)
;

late_bound_constant =
    "c", "[", '"', identifier, '"', "]"  (* late-bound constant pointwise value *)
  | "C", "[", '"', identifier, '"', "]"  (* late-bound constant stencil neighbourhood, only available in stencil QoIs *)
  | "c", "[", '"', "$", identifier, '"', "]"  (* built-in late-bound constant pointwise value *)
  | "C", "[", '"', "$", identifier, '"', "]"  (* built-in late-bound constant stencil neighbourhood, only available in stencil QoIs *)
;

variable =
    "v", "[", '"', identifier, '"', "]"    (* user-defined variable, only available in pointwise QoIs *)
  | "V", "[", '"', identifier, '"', "]"    (* user-defined variable, only available in stencil QoIs *)
;

identifier =
    ( letter | "_" ), { letter | digit | "_" }
;
letter =
    "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k"
  | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v"
  | "w" | "x" | "y" | "z" | "A" | "B" | "C" | "D" | "E" | "F" | "G"
  | "H" | "I" | "J" | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R"
  | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z"
;

array_indexing =
    expr, "[", "I", "]"  (* stencil neighbourhood centre, only available in stencil QoIs *)
  | expr, "[", index, { ",", index }, [","], "]"  (* array indexing *)
  | "I", "[", index, { ",", index }, [","], "]"  (* indexed 1D array over the per-axis stencil neighbourhood centre indices, only available in stencil QoIs *)
;

index =
    integer_expr  (* single index *)
  | [ integer_expr ], ":", [ integer_expr ]  (* slice index from (inclusive) to (exclusive) *)
  | [ integer_expr ], ":", [ integer_expr ], ":", [ integer_expr ]  (* slice index with step *)
;

integer_expr =
    ? symbolic integer-only expression ?
;

arithmetic_functions =
    "ln", "(", expr, [","], ")"  (* natural logarithm *)
  | "log2", "(", expr, [","], ")"  (* binary logarithm *)
  | "log10", "(", expr, [","], ")"  (* decimal logarithm *)
  | "log", "(", expr, ",", "base", "=", expr, [","], ")"  (* logarithm with arbitrary base *)
  | "exp", "(", expr, [","], ")"  (* exponential e^x *)
  | "exp2", "(", expr, [","], ")"  (* binary exponentiation 2^x *)
  | "exp10", "(", expr, [","], ")"  (* decimal exponentiation 10^x *)
  | "sqrt", "(", expr, [","], ")"  (* square root *)
  | "square", "(", expr, [","], ")"  (* square x^2 *)
  | "reciprocal", "(", expr, [","], ")"  (* reciprocal 1/x *)
  | "abs", "(", expr, [","], ")"   (* absolute value *)
  | "sign", "(", expr, [","], ")"  (* sign function, signed NaN for NaNs *)
  | "floor", "(", expr, [","], ")"  (* round down, towards negative infinity *)
  | "ceil", "(", expr, [","], ")"  (* round up, towards positive infinity *)
  | "trunc", "(", expr, [","], ")"  (* round towards zero *)
  | "round_ties_even", "(", expr, [","], ")"  (* round to nearest integer, ties to even *)
  | "sin", "(", expr, [","], ")"  (* sine sin(x) *)
  | "cos", "(", expr, [","], ")"  (* cosine cos(x) *)
  | "tan", "(", expr, [","], ")"  (* tangent tan(x) *)
  | "asin", "(", expr, [","], ")"  (* inverse sine asin(x) *)
  | "acos", "(", expr, [","], ")"  (* inverse cosine acos(x) *)
  | "atan", "(", expr, [","], ")"  (* inverse tangent atan(x) *)
  | "sinh", "(", expr, [","], ")"  (* hyperbolic sine sinh(x) *)
  | "cosh", "(", expr, [","], ")"  (* hyperbolic cosine cosh(x) *)
  | "tanh", "(", expr, [","], ")"  (* hyperbolic tangent tanh(x) *)
  | "asinh", "(", expr, [","], ")"  (* inverse hyperbolic sine asinh(x) *)
  | "acosh", "(", expr, [","], ")"  (* inverse hyperbolic cosine acosh(x) *)
  | "atanh", "(", expr, [","], ")"  (* inverse hyperbolic tangent atanh(x) *)
;

classification_functions =
    "isfinite", "(", expr, [","], ")"  (* 1 if finite, 0 if inf or NaN *)
  | "isinf", "(", expr, [","], ")"  (* 1 if inf, 0 if finite or NaN *)
  | "isnan", "(", expr, [","], ")"  (* 1 if NaN, 0 if finite or inf *)
;

logical_combinators =
    "not", "(", expr, [","], ")"  (* 1 if == 0, 0 if != 0 *)
  | "all", "(", expr, [","], ")"  (* 1 if all array elements != 0, 0 if any array element == 0 *)
  | "any", "(", expr, [","], ")"  (* 1 if any array element != 0, 0 if all array elements == 0 *)
  | "where", "(", expr, ",", expr, ",", expr, [","], ")"  (* where(c, x, y) = x if (c != 0) else y *)
;

array_functions =
    "size", "(", expr, [","], ")"  (* array size *)
  | "shape", "(", expr, [","], ")"  (* array shape as a 1D array *)
  | "sum", "(", expr, [","], ")"  (* array sum *)
  | "matmul", "(", expr, ",", expr, [","], ")"  (* matrix (2d array) multiplication *)
;

finite_difference =
    "finite_difference", "("  (* finite difference over an expression, only available in stencil QoIs *)
      , expr, ","
      , "order", "=", integer, ","  (* order of the derivative *)
      , "accuracy", "=", integer, ","  (* order of accuracy of the approximation *)
      , "type", "=", ("-1" | "0" | "1"), ","  (* backwards | central | forward difference *)
      , "axis", "=", integer, ","  (* axis, relative to the neighbourhood *)
      , (
            "grid_spacing", "=", expr  (* scalar uniform grid spacing along the axis *)
          | "grid_centre", "=", expr  (* centre of an arbitrary grid along the axis *)
        )
      , (
            [","]
          | ",", "grid_period", "=", expr, [","]  (* optional grid period, e.g. 2*pi or 360 *)
        )
  , ")"
;
```

The QoI expression can also contain whitespaces (space ` `, tab `\\t`,
newline `\\n`) and single-line inline comments starting with a hash `#`.

## Numerical Evaluation

### Floating-point data type

QoIs can be evaluated on any data type supported by the safeguards (see
[`Safeguards.supported_dtypes`][...api.Safeguards.supported_dtypes]).
Since the QoIs support many functions with floating-point outputs, they
are evaluated using floating-point arithmetic.

Importantly, the floating-point evaluation data type *must* be able to
represent all values of the input data type losslessly.

* For floating-point data, this is at least the input data type.

* For integer data, the data is first losslessly upcast to a floating-point
type with sufficient precision to represent all integer values, i.e. a type
whose mantissa has more bits than the integer type. For the below floating-
point types, this corresponds to choosing a floating-point data type with a
larger bit width (e.g. at least [`np.float64`][numpy.float64] for
[`np.int32`][numpy.int32] or [`np.uint32`][numpy.uint32] data).

The specific floating-point data type in which the quantities of interest are
evaluated is configured using the [`ToFloatMode`][...utils.cast.ToFloatMode]
enum, please refer to its documentation for further information.

### Literals

The quantities of interest can contain both integer and floating-point
literals. During numerical evaluation, both are evaluated for the chosen
floating-point type. For instance, if evaluation occurs in
[`np.float64`][numpy.float64] format, a literal `0.33` is evaluated using
[`np.float64("0.33")`], which returns the closest representable
[`np.float64`][numpy.float64] value for the symbolic value `0.33`. The
constants `e` and `pi` are provided as the closest representable value in the
chosen floating-point type.

### Symbolic integer constant folding

For integer-only subexpressions, the quantities of interest perform symbolic
constant folding based on operator associativity and explicit grouping with
parentheses. Like in Python, `1 + 2 + 3.5` (addition is left-associative) is
first evaluated to `3 + 3.5` and `3.5 + 2 * 3` to `3.5 + 6`. However, `1.0 + 1`
is left as-is. The following symbolic integer constant-folding operations are
provided:

- group `(a)`: iff `a` is an integer, `(a)` is folded
- negation `-a` (right associative): iff `a` is an integer, `-a` is folded
- addition `a + b` (left associative): iff both `a` and `b` are integers,
  `a + b` is folded
- subtraction `a - b` (left associative): iff both `a` and `b` are integers,
  `a - b` is folded
- multiplication `a * b` (left associative): iff both `a` and `b` are integers,
  `a * b` is folded
- exponentiation / power `a ** b` (right associative): iff both `a` and `b` are
  integers and b is non-negative, `a ** b` is folded

Since (true) division (left associative) in Python always produces a floating
point number, even for `a / 1`, division does not perform symbolic integer
constant-folding in general. However, if for `a / b` both `a` and `b` are
integers and have a greatest common denominator / factor g, i.e. `a = g * c`
and `b = g * d`, the division is symbolically simplified to `c / d`.
Furthermore, `a / 1` is evaluated to `a.0` and `a / -1` to `-a.0`. Since `a.0`
is a floating-point literal, symbolic integer constant folding stops there.

### Numerical evaluation order

Numerical evaluation (after symbolic integer constant folding) evaluates the
quantity of interest literally as written. No clever rewritings are performed.
Therefore, the quantities of interest have strictly defined and reproducible
evaluation rules, which allow writing them to reproduce the same evaluation as
existing (Python) code. For example (where `t1` ... are temporaries):

1. `3.0 + 1 + 7 * (x - 3)`
2. `4.0 + 7 * (x - 3)`
3. `t1 = x - 3.0`
4. `4.0 + 7 * t1`
5. `t2 = 7.0 * t1`
6. `4.0 + t2`

### Numerical functions

Numerical evaluation of the quantities of interest in `compression-safeguards`
is provided by `numpy`.

Some mathematical expressions such as the square root can be written using (a)
exponentiation `x ** 0.5` or (b) the built-in `sqrt(x)` function. It is
preferable to use special built-in functions, where available, since the
safeguards can better understand their meaning and provide better corrections
and higher compression ratios for them.

The operators and functions in the above QoI grammar are evaluated using
`numpy` ufuncs and follow the specification of the `math.h` ISO C standard
(see e.g. <https://pubs.opengroup.org/onlinepubs/9799919799/>):

| QoI function | `numpy` ufunc | `math.h` equivalent | -0.0 behaviour |
| ------------ | ------------- | ------------------- | -------------- |
| `+a` | no-op | |  -0.0 |
| `-a` | `np.negative` | | +0.0 |
| `a + b` | `np.add` | | recessive[^5] |
| `a - b` | `np.subtract` | | recessive[^5] |
| `a * b` | `np.multiply` | | as expected[^6] |
| `a / b` | `np.divide` | | as expected[^7] |
| `a ** b` | `np.power` | `pow` | as expected[^8] |
| `ln` | `np.log` | `log` | -inf |
| `log2` | `np.log2` | `log2` | -inf |
| `log10` | `np.log10` | `log10` | -inf |
| `log(a, base=b)` | `np.divide(np.log(a), np.log(b))` | | as expected |
| `exp` | `np.exp` | `exp` | 1.0 |
| `exp2` | `np.exp2` | `exp2` | 1.0 |
| `exp10` | `np.power(10, a)` | | 1.0 |
| `sqrt` | `np.sqrt` | `sqrt` | -0.0 |
| `square` | `np.square` | | +0.0 |
| `reciprocal` | `np.reciprocal` | | -inf |
| `abs` | `np.abs` | | +0.0 |
| `sign` | `np.sign` | | +0.0 |
| `floor` | `np.floor` | `floor` | -0.0 |
| `ceil` | `np.ceil` | `ceil` | -0.0 |
| `trunc` | `np.trunc` | `trunc` | -0.0 |
| `round_ties_even` | `np.rint` | `rint`[^4] | -0.0 |
| `sin` | `np.sin` | `sin` | -0.0 |
| `cos` | `np.cos` | `cos` | 1.0 |
| `tan` | `np.tan` | `tan` | -0.0 |
| `asin` | `np.arcsin` | `asin` | -0.0 |
| `acos` | `np.arccos` | `acos` | pi/2 |
| `atan` | `np.arctan` | `atan` | -0.0 |
| `sinh` | `np.sinh` | `sinh` | -0.0 |
| `cosh` | `np.cosh` | `cosh` | 1.0 |
| `tanh` | `np.tanh` | `tanh` | -0.0 |
| `asinh` | `np.arcsinh` | `asinh` | -0.0 |
| `acosh` | `np.arccosh` | `acosh` | NaN |
| `atanh` | `np.arctanh` | `atanh` | -0.0 |
| `isfinite` | `np.isfinite` | `isfinite` | 1.0 |
| `isinf` | `np.isinf` | `isinf` | +0.0 |
| `isnan` | `np.isnan` | `isnan` | +0.0 |
| `where` | `np.where` | | as expected |

[^4]: with the round-to-nearest rounding mode
[^5]: -0.0 + -0.0 = -0.0, -0.0 - +0.0 = -0.0, otherwise -0.0 is treated like +0.0
[^6]: a * -0.0 = -a * +0.0
[^7]: a / -0.0 = -a / +0.0, -0.0 / b = +0.0 / -b
[^8]: -0.0 ** (2k+1) = -0.0, -0.0 ** +b = +0.0 otherwise, -0.0 ** (-2k=1) = -inf, -0.0 ** -b = +inf otherwise, a ** -0.0 = 1.0

Furthermore, the array `sum` and `matmul` functions are implemented as explicit
sums over the array elements in natural order, e.g.
`sum([[1, 2], [3, 4]]) = (1 + 2 + 3 + 4)` and
`matmul([[1, 2]], [[3], [4]]) = [[(1 * 3 + 2 * 4)]]`.

### Finite differences

The `finite_difference` function can be used to compute the finite-difference-
approximated derivative over an expression. The finite difference coefficients
for arbitrary orders, accuracies, and grid spacings are derived using the
algorithm from:

> Fornberg, B. (1988). Generation of finite difference formulas on arbitrarily
spaced grids. *Mathematics of Computation*, 51(184), 699-706. Available from:
[doi:10.1090/s0025-5718-1988-0935077-0](https://doi.org/10.1090/s0025-5718-1988-0935077-0).

The computation of the coefficients uses symbolic integer constant folding,
where possible, to produce accurate coefficients on a best-effort basis. Since
the quantities of interest perform no symbolic constant folding on floating
point literals, however, numerical rounding errors can occur when computing the
coefficients for large orders or when using non-integer grid spacings,
custom grid periods, or arbitrary late-bound constant grids. Therefore, the
bitwise exact evaluation of the `finite_difference` function is not yet
specified.
"""

__all__ = [
    "PointwiseQuantityOfInterestExpression",
    "StencilQuantityOfInterestExpression",
]

from typing import NewType

PointwiseQuantityOfInterestExpression = NewType(
    "PointwiseQuantityOfInterestExpression", str
)
"""
Pointwise quantity of interest expression in [`str`][str]ing form, following
the above EBNF grammar for pointwise QoIs.
"""


StencilQuantityOfInterestExpression = NewType(
    "StencilQuantityOfInterestExpression", str
)
"""
Stencil quantity of interest expression in [`str`][str]ing form, following
the above EBNF grammar for stencil QoIs.
"""
