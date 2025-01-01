from numcodecs_guardrails import GuardrailKind


def test_guardrail_kinds():
    for kind in GuardrailKind:
        assert kind.name == kind.value.kind
