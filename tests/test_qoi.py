from numcodecs_safeguards.safeguards.pointwise.qoi import QuantityOfInterestSafeguard


def test():
    safeguard = QuantityOfInterestSafeguard(qoi="3*x**3 + 2*x**2 + x", eb_abs=1)
    assert False, f"{safeguard}"
