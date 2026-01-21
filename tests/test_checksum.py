import numpy as np
from numcodecs_safeguards.checksum import checksum


def test_rfc_1071():
    # the example in section 3 is given in big-endian order
    data = np.array(
        [0x00, 0x01, 0xF2, 0x03, 0xF4, 0xF5, 0xF6, 0xF7], dtype=np.uint8
    ).view(">u2")

    cs = checksum(data)
    # undo the 1s complement since RFC 1071's example doesn't apply it
    cs1 = (~np.frombuffer(cs, dtype="<u2", count=1)).tobytes()

    assert cs1 == b"\xdd\xf2"

    cs = checksum(data.astype("<i2"))
    cs1 = (~np.frombuffer(cs, dtype="<u2", count=1)).tobytes()
    assert cs1 == b"\xdd\xf2"

    cs = checksum(data.astype(">i2"))
    cs1 = (~np.frombuffer(cs, dtype="<u2", count=1)).tobytes()
    assert cs1 == b"\xdd\xf2"

    cs = checksum(data.view(">i4").astype("<i4"))
    cs1 = (~np.frombuffer(cs, dtype="<u2", count=1)).tobytes()
    assert cs1 == b"\xdd\xf2"

    cs = checksum(data.view(">i4"))
    cs1 = (~np.frombuffer(cs, dtype="<u2", count=1)).tobytes()
    assert cs1 == b"\xdd\xf2"

    cs = checksum(data.view(">i8").astype("<i8"))
    cs1 = (~np.frombuffer(cs, dtype="<u2", count=1)).tobytes()
    assert cs1 == b"\xdd\xf2"

    cs = checksum(data.view(">i8"))
    cs1 = (~np.frombuffer(cs, dtype="<u2", count=1)).tobytes()
    assert cs1 == b"\xdd\xf2"


# adapted from https://docs.rs/crate/internet-checksum/0.2.1/source/src/lib.rs#768-784
def test_ipv4_checksums():
    # fmt: off
    IPV4_HEADERS = [
        [
            0x45, 0x00, 0x00, 0x34, 0x00, 0x00, 0x40, 0x00, 0x40, 0x06,
            0xae, 0xea, 0xc0, 0xa8, 0x01, 0x0f, 0xc0, 0xb8, 0x09, 0x6a,
        ],
        [
            0x45, 0x20, 0x00, 0x74, 0x5b, 0x6e, 0x40, 0x00, 0x37, 0x06,
            0x5c, 0x1c, 0xc0, 0xb8, 0x09, 0x6a, 0xc0, 0xa8, 0x01, 0x0f,
        ],
        [
            0x45, 0x20, 0x02, 0x8f, 0x00, 0x00, 0x40, 0x00, 0x3b, 0x11,
            0xc9, 0x3f, 0xac, 0xd9, 0x05, 0x6e, 0xc0, 0xa8, 0x01, 0x0f,
        ]
    ]
    # fmt: on

    for data in IPV4_HEADERS:
        assert checksum(np.array(data, np.uint8)) == b"\x00\x00"

        assert checksum(np.frombuffer(bytes(data), "<u2")) == b"\x00\x00"
        assert checksum(np.frombuffer(bytes(data), "<u4")) == b"\x00\x00"
