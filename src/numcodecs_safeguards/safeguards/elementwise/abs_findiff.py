import numpy as np


def _finite_difference_coefficients(
    order: int,
    offsets: np.ndarray,
) -> np.ndarray:
    # x0 = 0
    M = order
    a = offsets
    N = len(a) - 1

    coeffs = {
        (0, 0, 0): 1,
    }

    c1 = 1

    for n in range(1, N + 1):
        c2 = 1
        for v in range(0, n):
            c3 = a[n] - a[v]
            c2 *= c3
            if n <= M:
                coeffs[(n, n - 1, v)] = 0
            for m in range(0, min(n, M) + 1):
                if m > 0:
                    coeffs[(m, n, v)] = (
                        (a[n] * coeffs[(m, n - 1, v)]) - (m * coeffs[(m - 1, n - 1, v)])
                    ) / c3
                else:
                    coeffs[(m, n, v)] = (a[n] * coeffs[(m, n - 1, v)]) / c3
        for m in range(0, min(n, M) + 1):
            if m > 0:
                coeffs[(m, n, n)] = (c1 / c2) * (
                    (m * coeffs[(m - 1, n - 1, n - 1)])
                    - (a[n - 1] * coeffs[(m, n - 1, n - 1)])
                )
            else:
                coeffs[(m, n, n)] = -(c1 / c2) * (a[n - 1] * coeffs[(m, n - 1, n - 1)])
        c1 = c2

    return np.array([coeffs[M, N, v] for v in range(0, N + 1)])
