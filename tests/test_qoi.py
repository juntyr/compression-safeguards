import numpy as np

from compression_safeguards.safeguards._qois.expr.abs import ScalarAbs
from compression_safeguards.safeguards._qois.expr.data import Data
from compression_safeguards.safeguards._qois.expr.reciprocal import ScalarReciprocal
from compression_safeguards.safeguards._qois.expr.sign import ScalarSign
from compression_safeguards.safeguards._qois.expr.square import ScalarSquare
from compression_safeguards.safeguards._qois.expr.trigonometric import (
    ScalarAsin,
    ScalarSin,
)
from compression_safeguards.safeguards._qois.interval import (
    compute_safe_data_lower_upper_interval_union,
)


def test_abs():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarAbs(Data(index=()))

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.abs(X) - 1,
            np.abs(X) + 1,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    assert np.all(
        (
            valid._lower[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    -43.0,
                    -3.0,
                    -2.0,
                    -1.5,
                    -1.0,
                    -1.0,
                    -1.5,
                    -2.0,
                    1.0,
                    41.0,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._lower[0]))
    )
    assert np.all(np.isnan(valid._lower[1]))
    assert np.all(
        (
            valid._upper[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    -41.0,
                    -1.0,
                    2.0,
                    1.5,
                    1.0,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    43.0,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._upper[0]))
    )
    assert np.all(np.isnan(valid._upper[1]))


def test_sign_same():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarSign(Data(index=()))

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.sign(X),
            np.sign(X),
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    fmax = np.finfo(X.dtype).max
    fmin = np.finfo(X.dtype).smallest_subnormal

    assert np.all(
        (
            valid._lower[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -0.0,
                    -0.0,
                    fmin,
                    fmin,
                    fmin,
                    fmin,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._lower[0]))
    )
    assert np.all(np.isnan(valid._lower[1]))
    assert np.all(
        (
            valid._upper[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    -fmin,
                    -fmin,
                    -fmin,
                    -fmin,
                    +0.0,
                    +0.0,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._upper[0]))
    )
    assert np.all(np.isnan(valid._upper[1]))


def test_sign_one_off():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarSign(Data(index=()))

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.sign(X) - 1,
            np.sign(X) + 1,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    fmax = np.finfo(X.dtype).max

    assert np.all(
        (
            valid._lower[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -0.0,
                    -0.0,
                    -0.0,
                    -0.0,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._lower[0]))
    )
    assert np.all(np.isnan(valid._lower[1]))
    assert np.all(
        (
            valid._upper[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    +0.0,
                    +0.0,
                    +0.0,
                    +0.0,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._upper[0]))
    )
    assert np.all(np.isnan(valid._upper[1]))


def test_sign_any():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarSign(Data(index=()))

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.sign(X) - 2,
            np.sign(X) + 2,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    fmax = np.finfo(X.dtype).max

    assert np.all(
        (
            valid._lower[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._lower[0]))
    )
    assert np.all(np.isnan(valid._lower[1]))
    assert np.all(
        (
            valid._upper[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._upper[0]))
    )
    assert np.all(np.isnan(valid._upper[1]))


def test_square():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarSquare(Data(index=()))

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.square(X) - 1,
            np.square(X) + 1,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    np.testing.assert_allclose(
        valid._lower[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -np.sqrt(np.square(42.0) + 1),
                -np.sqrt(5.0),
                -np.sqrt(2.0),
                -np.sqrt(1.25),
                -1.0,
                -1.0,
                -np.sqrt(1.25),
                -np.sqrt(2.0),
                np.sqrt(3.0),
                np.sqrt(np.square(42.0) - 1),
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._lower[1]))
    np.testing.assert_allclose(
        valid._upper[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -np.sqrt(np.square(42.0) - 1),
                -np.sqrt(3.0),
                np.sqrt(2.0),
                np.sqrt(1.25),
                1.0,
                1.0,
                np.sqrt(1.25),
                np.sqrt(2.0),
                np.sqrt(5.0),
                np.sqrt(np.square(42.0) + 1),
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._upper[1]))


def test_reciprocal():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarReciprocal(Data(index=()))

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.reciprocal(X) - 1,
            np.reciprocal(X) + 1,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        assert np.all(
            (np.abs(np.reciprocal(X) - np.reciprocal(valid._lower[0])) <= 1.0)
            | (np.reciprocal(X) == np.reciprocal(valid._lower[0]))
            | (np.isnan(X) & np.isnan(valid._lower[0]))
        )
    assert np.all(np.isnan(valid._lower[1]))
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        assert np.all(
            (np.abs(np.reciprocal(X) - np.reciprocal(valid._upper[0])) <= 1.0)
            | (np.reciprocal(X) == np.reciprocal(valid._upper[0]))
            | (np.isnan(X) & np.isnan(valid._upper[0]))
        )
    assert np.all(np.isnan(valid._upper[1]))


def test_sin():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarSin(Data(index=()))

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.sin(X) - 1,
            np.sin(X) + 1,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    np.testing.assert_allclose(
        valid._lower[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -42.0 - (np.asin(1.0) - np.asin(np.sin(-42.0))),
                -2.0 - (np.asin(np.sin(-2.0) + 1.0) - np.asin(np.sin(-2.0))),
                -np.pi / 2,
                -np.pi / 2,
                -np.pi / 2,
                -np.pi / 2,
                0.5 + (np.asin(np.sin(0.5) - 1) - np.asin(np.sin(0.5))),
                1.0 + (np.asin(np.sin(1.0) - 1) - np.asin(np.sin(1.0))),
                np.pi / 2,
                42.0 - (np.asin(np.sin(42.0) + 1) - np.asin(np.sin(42.0))),
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._lower[1]))
    np.testing.assert_allclose(
        valid._upper[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -42.0 - (np.asin(np.sin(-42.0) - 1) - np.asin(np.sin(-42.0))),
                -np.pi / 2,
                -1.0 + (np.asin(np.sin(-1.0) + 1) - np.asin(np.sin(-1.0))),
                -0.5 + (np.asin(np.sin(-0.5) + 1) - np.asin(np.sin(-0.5))),
                np.pi / 2,
                np.pi / 2,
                np.pi / 2,
                np.pi / 2,
                2.0 - (np.asin(np.sin(2.0) - 1.0) - np.asin(np.sin(2.0))),
                42.0 - (np.asin(-1.0) - np.asin(np.sin(42.0))),
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._upper[1]))


def test_asin():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -2.0,
            -1.0,
            -0.75,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            0.75,
            1.0,
            2.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarAsin(Data(index=()))

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.asin(X) - 1,
            np.asin(X) + 1,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    fmax = np.finfo(X.dtype).max
    one_eps = np.nextafter(X.dtype.type(1), 2)

    np.testing.assert_allclose(
        valid._lower[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -fmax,
                -1.0,
                -1.0,
                np.sin(np.asin(-0.5) - 1.0),
                np.sin(-1.0),
                np.sin(-1.0),
                np.sin(np.asin(0.5) - 1.0),
                np.sin(np.asin(0.75) - 1.0),
                np.sin(np.asin(1.0) - 1.0),
                one_eps,
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._lower[1]))
    np.testing.assert_allclose(
        valid._upper[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -one_eps,
                np.sin(np.asin(-1.0) + 1.0),
                np.sin(np.asin(-0.75) + 1.0),
                np.sin(np.asin(-0.5) + 1.0),
                np.sin(1.0),
                np.sin(1.0),
                np.sin(np.asin(0.5) + 1.0),
                1.0,
                1.0,
                fmax,
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._upper[1]))
