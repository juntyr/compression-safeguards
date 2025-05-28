import numpy as np

from compression_safeguards.cast import (
    _float128,
    _float128_eps,
    _float128_max,
    _float128_min,
    _float128_precision,
    _float128_smallest_normal,
    _float128_smallest_subnormal,
    _isnan,
    _nextafter,
)


def test_float128():
    assert _float128_max == _float128("1.189731495357231765085759326628007e+4932")
    assert _float128_min == _float128("-1.189731495357231765085759326628007e+4932")
    assert _float128_precision == 33
    assert _float128_smallest_subnormal == _float128("6e-4966")
    assert _float128_smallest_normal == _float128(
        "3.3621031431120935062626778173217526e-4932"
    )
    assert _float128_eps == _float128("1.9259299443872358530559779425849273e-34")


def test_nextafter_float128():
    assert _nextafter(np.array(_float128(0)), np.inf)[()] == _float128("6e-4966")
    assert _nextafter(np.array(-_float128(0)), np.inf)[()] == _float128("6e-4966")
    assert _nextafter(np.array(_float128(0)), -np.inf)[()] == _float128("-6e-4966")
    assert _nextafter(np.array(-_float128(0)), -np.inf)[()] == _float128("-6e-4966")
    assert _nextafter(np.array(_float128(0)), 0)[()] == _float128(0)
    assert _nextafter(np.array(-_float128(0)), -0.0)[()] == _float128(-0.0)

    assert _nextafter(np.array(_float128("6e-4966")), np.inf)[()] == _float128(
        "1e-4965"
    )
    assert _nextafter(np.array(_float128("6e-4966")), -np.inf)[()] == _float128(0)
    assert _nextafter(np.array(_float128("-6e-4966")), np.inf)[()] == -_float128(0)
    assert _nextafter(np.array(_float128("-6e-4966")), -np.inf)[()] == _float128(
        "-1e-4965"
    )

    assert _nextafter(
        np.array(_float128("3.3621031431120935062626778173217526e-4932")), np.inf
    ) == _float128("3.362103143112093506262677817321753e-4932")
    assert _nextafter(
        np.array(_float128("3.3621031431120935062626778173217526e-4932")), -np.inf
    ) == _float128("3.362103143112093506262677817321752e-4932")
    assert _nextafter(
        np.array(_float128("-3.3621031431120935062626778173217526e-4932")), np.inf
    ) == _float128("-3.362103143112093506262677817321752e-4932")
    assert _nextafter(
        np.array(_float128("-3.3621031431120935062626778173217526e-4932")), -np.inf
    ) == _float128("-3.362103143112093506262677817321753e-4932")

    assert _nextafter(np.array(_float128(1)), np.inf)[()] == _float128(
        "1.0000000000000000000000000000000002"
    )
    assert _nextafter(np.array(_float128(1)), -np.inf)[()] == _float128(
        "0.9999999999999999999999999999999999"
    )
    assert _nextafter(np.array(_float128(-1)), np.inf)[()] == _float128(
        "-0.9999999999999999999999999999999999"
    )
    assert _nextafter(np.array(_float128(-1)), -np.inf)[()] == _float128(
        "-1.0000000000000000000000000000000002"
    )

    assert _nextafter(
        np.array(_float128("1.0000000000000000000000000000000002")), np.inf
    )[()] == _float128("1.0000000000000000000000000000000004")
    assert _nextafter(
        np.array(_float128("1.0000000000000000000000000000000002")), -np.inf
    )[()] == _float128(1)
    assert _nextafter(
        np.array(_float128("-1.0000000000000000000000000000000002")), np.inf
    )[()] == _float128(-1)
    assert _nextafter(
        np.array(_float128("-1.0000000000000000000000000000000002")), -np.inf
    )[()] == _float128("-1.0000000000000000000000000000000004")

    assert _nextafter(
        np.array(_float128("1.0000000000000000000000000000000004")), np.inf
    )[()] == _float128("1.0000000000000000000000000000000006")
    assert _nextafter(
        np.array(_float128("1.0000000000000000000000000000000004")), -np.inf
    )[()] == _float128("1.0000000000000000000000000000000002")
    assert _nextafter(
        np.array(_float128("-1.0000000000000000000000000000000004")), np.inf
    )[()] == _float128("-1.0000000000000000000000000000000002")
    assert _nextafter(
        np.array(_float128("-1.0000000000000000000000000000000004")), -np.inf
    )[()] == _float128("-1.0000000000000000000000000000000006")

    assert _nextafter(np.array(_float128(424242)), np.inf)[()] == _float128(
        "424242.00000000000000000000000000005"
    )
    assert _nextafter(np.array(_float128(424242)), -np.inf)[()] == _float128(
        "424241.99999999999999999999999999995"
    )
    assert _nextafter(np.array(_float128(-424242)), np.inf)[()] == _float128(
        "-424241.99999999999999999999999999995"
    )
    assert _nextafter(np.array(_float128(-424242)), -np.inf)[()] == _float128(
        "-424242.00000000000000000000000000005"
    )

    assert _nextafter(
        np.array(_float128("1.189731495357231765085759326628007e+4932")), np.inf
    )[()] == _float128(np.inf)
    assert _nextafter(
        np.array(_float128("1.189731495357231765085759326628007e+4932")), -np.inf
    )[()] == _float128("1.1897314953572317650857593266280069e+4932")
    assert _nextafter(
        np.array(_float128("-1.189731495357231765085759326628007e+4932")), np.inf
    )[()] == _float128("-1.1897314953572317650857593266280069e+4932")
    assert _nextafter(
        np.array(_float128("-1.189731495357231765085759326628007e+4932")), -np.inf
    )[()] == _float128(-np.inf)

    assert _nextafter(np.array(_float128(np.inf)), np.inf)[()] == _float128(np.inf)
    assert _nextafter(np.array(_float128(np.inf)), -np.inf)[()] == _float128(
        "1.189731495357231765085759326628007e+4932"
    )
    assert _nextafter(np.array(_float128(-np.inf)), np.inf)[()] == _float128(
        "-1.189731495357231765085759326628007e+4932"
    )
    assert _nextafter(np.array(_float128(-np.inf)), -np.inf)[()] == _float128(-np.inf)

    assert _isnan(np.array(_nextafter(np.array(_float128(np.nan)), 0)[()]))
    assert _isnan(np.array(_nextafter(np.array(_float128(np.nan)), np.inf)[()]))
    assert _isnan(np.array(_nextafter(np.array(_float128(0)), np.nan)[()]))
    assert _isnan(np.array(_nextafter(np.array(_float128(np.inf)), np.nan)[()]))
    assert _isnan(np.array(_nextafter(np.array(_float128(np.nan)), np.nan)[()]))
