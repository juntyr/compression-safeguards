from compression_safeguards import SafeguardKind


def test_safeguard_kinds():
    for kind in SafeguardKind:
        assert kind.name == kind.value.kind
