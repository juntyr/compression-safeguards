from numcodecs_guardrails import Guardrails


def test_guardrail_kinds():
    for kind in Guardrails:
        assert kind.name == kind.value.kind
