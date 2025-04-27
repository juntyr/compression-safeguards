from numcodecs_safeguards.safeguards.pointwise.qoi import QuantityOfInterestSafeguard


def test():
    safeguard = QuantityOfInterestSafeguard(qoi="x**5", eb_abs=1)
    assert safeguard._expr is None, f"{safeguard._expr}\n{safeguard._eb_abs_qoi}"
