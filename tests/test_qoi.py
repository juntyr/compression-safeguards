import numpy as np

from compression_safeguards.safeguards._qois.expr.abs import ScalarAbs
from compression_safeguards.safeguards._qois.expr.data import Data
from compression_safeguards.safeguards._qois.expr.reciprocal import ScalarReciprocal
from compression_safeguards.safeguards._qois.expr.sign import ScalarSign
from compression_safeguards.safeguards._qois.expr.square import ScalarSquare
from compression_safeguards.safeguards._qois.interval import (
    compute_safe_eb_lower_upper_interval_union,
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
            0.0,
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
        eb_X_lower, eb_X_upper = expr.compute_data_error_bound(
            -np.ones_like(X),
            np.ones_like(X),
            X,
            X,
            dict(),
        )
        valid = compute_safe_eb_lower_upper_interval_union(
            X,
            X,
            eb_X_lower,
            eb_X_upper,
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
            0.0,
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
        eb_X_lower, eb_X_upper = expr.compute_data_error_bound(
            -np.zeros_like(X),
            np.zeros_like(X),
            X,
            X,
            dict(),
        )
        valid = compute_safe_eb_lower_upper_interval_union(
            X,
            X,
            eb_X_lower,
            eb_X_upper,
        )

    fmax = np.finfo(X.dtype).max
    fmin = np.finfo(X.dtype).smallest_subnormal

    # FIXME: interval based error bounds would produce these exact values
    np.testing.assert_allclose(
        valid._lower[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -fmax,
                -fmax,
                -fmax,
                -fmax,
                0.0,
                fmin,
                fmin,
                fmin,
                fmin,
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._lower[1]))
    # FIXME: interval based error bounds would produce these exact values
    np.testing.assert_allclose(
        valid._upper[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -fmin,
                -fmin,
                -fmin,
                -fmin,
                0.0,
                fmax,
                fmax,
                fmax,
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


def test_sign_one_off():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            0.0,
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
        eb_X_lower, eb_X_upper = expr.compute_data_error_bound(
            -np.ones_like(X),
            np.ones_like(X),
            X,
            X,
            dict(),
        )
        valid = compute_safe_eb_lower_upper_interval_union(
            X,
            X,
            eb_X_lower,
            eb_X_upper,
        )

    fmax = np.finfo(X.dtype).max

    # FIXME: interval based error bounds would produce these exact values
    np.testing.assert_allclose(
        valid._lower[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -fmax,
                -fmax,
                -fmax,
                -fmax,
                -fmax,
                0.0,
                0.0,
                0.0,
                0.0,
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._lower[1]))
    # FIXME: interval based error bounds would produce these exact values
    np.testing.assert_allclose(
        valid._upper[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                0.0,
                0.0,
                0.0,
                0.0,
                fmax,
                fmax,
                fmax,
                fmax,
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


def test_sign_any():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            0.0,
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
        eb_X_lower, eb_X_upper = expr.compute_data_error_bound(
            -np.ones_like(X) * 2,
            np.ones_like(X) * 2,
            X,
            X,
            dict(),
        )
        valid = compute_safe_eb_lower_upper_interval_union(
            X,
            X,
            eb_X_lower,
            eb_X_upper,
        )

    fmax = np.finfo(X.dtype).max

    # FIXME: interval based error bounds would produce these exact values
    np.testing.assert_allclose(
        valid._lower[0],
        np.array(
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
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._lower[1]))
    # FIXME: interval based error bounds would produce these exact values
    np.testing.assert_allclose(
        valid._upper[0],
        np.array(
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
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
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
            0.0,
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
        eb_X_lower, eb_X_upper = expr.compute_data_error_bound(
            -np.ones_like(X),
            np.ones_like(X),
            X,
            X,
            dict(),
        )
        valid = compute_safe_eb_lower_upper_interval_union(
            X,
            X,
            eb_X_lower,
            eb_X_upper,
        )

    # FIXME: interval based error bounds would produce these exact values
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
    # FIXME: interval based error bounds would produce these exact values
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
            0.0,
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
        eb_X_lower, eb_X_upper = expr.compute_data_error_bound(
            -np.ones_like(X),
            np.ones_like(X),
            X,
            X,
            dict(),
        )
        valid = compute_safe_eb_lower_upper_interval_union(
            X,
            X,
            eb_X_lower,
            eb_X_upper,
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
            | (np.isnan(X) == np.isnan(valid._upper[0]))
            | (np.isnan(X) & np.isnan(valid._upper[0]))
        )
    assert np.all(np.isnan(valid._upper[1]))
