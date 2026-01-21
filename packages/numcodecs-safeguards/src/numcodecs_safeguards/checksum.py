__all__ = ["checksum"]

import numpy as np
from compression_safeguards.utils.typing import S, T


# adapted from https://gist.github.com/drpresq/ecf699f05178dd280ca14a2577e004f2
def checksum(data: np.ndarray[S, np.dtype[T]]) -> bytes:
    """
    Compute the [RFC 1071] "Internet Checksum" over the little-endian C-order
    bytes of the `data` array.

    The 16bit checksum is returned as two bytes in big-endian order.

    Parameters
    ----------
    data : np.ndarray[S, np.dtype[T]]
        The array to compute the checksum for.

    Returns
    -------
    checksum : bytes
        Two byte checksum in big-endian order.

    [RFC 1071]: https://datatracker.ietf.org/doc/html/rfc1071
    """

    # read the data as little endian bytes
    bytes_le = data.astype(data.dtype.newbyteorder("<")).tobytes()
    # add padding if needed
    if len(bytes_le) % 2 != 0:
        bytes_le += b"\0"

    # reinterpret the bytes as little endian unsigned 16bit integers
    u16sle: np.ndarray[tuple[int]] = np.frombuffer(
        bytes_le, dtype=np.dtype("<u2"), count=len(bytes_le) // 2
    )

    # sum up using unsigned 32bit integer wrapping arithmetic
    acc32: np.uint32 = np.sum(u16sle.astype(np.uint32))

    # fold into unsigned 16bit integer: add the carrier
    while (acc32 >> 16) > 0:
        acc32 = (acc32 & 0xFFFF) + (acc32 >> 16)
    acc16: np.uint16 = acc32.astype(np.uint16)

    # one's complement, in big-endian 16bit unsigned integer
    checksum = np.array(~acc16).astype(">u2")

    return checksum.tobytes()
