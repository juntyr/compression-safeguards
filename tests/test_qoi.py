from numcodecs_safeguards.safeguards.pointwise.qoi import QuantityOfInterestSafeguard


def test():
    safeguard = QuantityOfInterestSafeguard(qoi="log(x, 2)", eb_abs=1)
    assert False, f"{safeguard}"
