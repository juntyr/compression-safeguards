from numcodecs_safeguards import Safeguards


def test_safeguard_kinds():
    for kind in Safeguards:
        assert kind.name == kind.value.kind
