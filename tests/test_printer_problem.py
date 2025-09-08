import pytest
from numcodecs_safeguards import SafeguardsCodec


def test_direct_wrap():
    SafeguardsCodec(codec=dict(id="zero"), safeguards=[])

    with pytest.raises(AssertionError, match="printer problem"):
        SafeguardsCodec(
            codec=SafeguardsCodec(codec=dict(id="zero"), safeguards=[]), safeguards=[]
        )


def test_codec_stack_wrap():
    with pytest.raises(AssertionError, match="printer problem"):
        SafeguardsCodec(
            codec=dict(
                id="combinators.stack",
                codecs=[
                    dict(id="bitround", keepbits=10),
                    SafeguardsCodec(codec=dict(id="zero"), safeguards=[]),
                ],
            ),
            safeguards=[],
        )
