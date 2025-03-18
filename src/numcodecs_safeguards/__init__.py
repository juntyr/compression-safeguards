"""
# Fearless lossy compression with `numcodecs-safeguards`

Lossy compression can be scary as valuable information may be lost.

This package provides the
[`SafeguardsCodec`][numcodecs_safeguards.SafeguardsCodec] adapter and several
[`Safeguards`][numcodecs_safeguards.Safeguards] that can be applied to *any*
existing (lossy) compressor to *guarantee* that certain properties about the
compression error are upheld.

Note that the wrapped compressor is treated as a blackbox and the decompressed
data is postprocessed to re-establish the properties, if necessary.

By using the [`SafeguardsCodec`][numcodecs_safeguards.SafeguardsCodec] adapter,
badly behaving lossy compressors become safe to use, at the cost of potentially
less efficient compression, and lossy compression can be applied without fear.
"""

__all__ = ["SafeguardsCodec", "Safeguards"]

from collections.abc import Sequence
from enum import Enum
from io import BytesIO
from typing import Callable
from typing_extensions import Buffer  # MSPV 3.12

import numcodecs
import numcodecs.compat
import numcodecs.registry
import numcodecs.zlib
import numpy as np
import varint

from numcodecs.abc import Codec
from numcodecs_combinators.abc import CodecCombinatorMixin

from .lossless import Lossless
from .safeguards import Safeguard
from .safeguards.elementwise import ElementwiseSafeguard
from .safeguards.elementwise.abs import AbsoluteErrorBoundSafeguard
from .safeguards.elementwise.decimal import DecimalErrorBoundSafeguard
from .safeguards.elementwise.findiff.abs import (
    FiniteDifferenceAbsoluteErrorBoundSafeguard,
)

# from .safeguards.elementwise.monotonicity import MonotonicityPreservingSafeguard
from .safeguards.elementwise.rel_or_abs import RelativeOrAbsoluteErrorBoundSafeguard
from .safeguards.elementwise.sign import SignPreservingSafeguard
from .safeguards.elementwise.zero import ZeroIsZeroSafeguard


class Safeguards(Enum):
    """
    Enumeration of all supported safeguards:
    """

    # exact values
    zero = ZeroIsZeroSafeguard
    """Enforce that zero (or another constant) is exactly preserved."""

    # error bounds
    abs = AbsoluteErrorBoundSafeguard
    """Enforce an absolute error bound."""

    rel_or_abs = RelativeOrAbsoluteErrorBoundSafeguard
    """Enforce a relative error bound, fall back to an absolute error bound close to zero."""

    decimal = DecimalErrorBoundSafeguard
    """Enforce a decimal error bound."""

    # finite difference error bounds
    findiff_abs = FiniteDifferenceAbsoluteErrorBoundSafeguard
    """Enforce an absolute error bound for the finite differences."""

    # # monotonicity
    # monotonicity = MonotonicityPreservingSafeguard
    # """Enforce that monotonic sequences remain monotonic."""

    # sign
    sign = SignPreservingSafeguard
    """Enforce that the sign (-1, 0, +1) of each element is preserved."""


_FORMAT_VERSION: str = "0.1.x"


_SUPPORTED_DTYPES: set[np.dtype] = {
    np.dtype("int8"),
    np.dtype("int16"),
    np.dtype("int32"),
    np.dtype("int64"),
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("uint32"),
    np.dtype("uint64"),
    np.dtype("float32"),
    np.dtype("float64"),
}


class SafeguardsCodec(Codec, CodecCombinatorMixin):
    """
    An adaptor codec that uses safeguards to guarantee certain properties are
    upheld by the wrapped codec.

    Parameters
    ----------
    codec : None | dict | Codec
        The codec to wrap with safeguards. It can either be passed as a codec
        configuration [`dict`][dict], which is passed to
        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec],
        or an already initialized [`Codec`][numcodecs.abc.Codec].

        The codec must encode to a 1D buffer of bytes. It is desirable to
        perform lossless compression after applying the safeguards (rather than
        before), e.g. by customising the
        [`Lossless.for_codec`][numcodecs_safeguards.Lossless.for_codec]
        field of the `lossless` parameter.

        If [`None`][None], *only* the safeguards are used to encode the data.
        Note that using a codec likely provides a better compression ratio. If
        no safeguards are provided, the encoded data can be decoded to *any*
        output.
    safeguards : Sequence[dict | Safeguard]
        The safeguards that will be applied to the codec. They can either be
        passed as a safeguard configuration [`dict`][dict] or an already
        initialized [`Safeguard`][numcodecs_safeguards.safeguards.Safeguard].

        Please refer to [`Safeguards`][numcodecs_safeguards.Safeguards] for an
        enumeration of all supported safeguards.
    lossless : None | dict | Lossless, optional
        The lossless encoding that is applied after the codec and the
        safeguards:

        - [`Lossless.for_codec`][numcodecs_safeguards.Lossless.for_codec]
          specifies the lossless encoding that is applied to the encoded output
          of the wrapped `codec`. By default, no additional lossless encoding
          is applied.
        - [`Lossless.for_safeguards`][numcodecs_safeguards.Lossless.for_safeguards]
          specifies the lossless encoding that is applied to the encoded
          correction that the safeguards produce. By default, Huffman encoding
          followed by Zstandard is applied.
    _version : ...
        Internal, do not provide this paramter explicitly.
    """

    __slots__ = (
        "_version",
        "_codec",
        "_safeguards",
        "_lossless_for_codec",
        "_lossless_for_safeguards",
    )
    _version: str
    _codec: None | Codec
    _elementwise_safeguards: tuple[ElementwiseSafeguard, ...]
    _lossless_for_codec: None | Codec
    _lossless_for_safeguards: Codec

    codec_id: str = "safeguards"  # type: ignore

    def __init__(
        self,
        *,
        codec: None | dict | Codec,
        safeguards: Sequence[dict | Safeguard],
        lossless: None | dict | Lossless = None,
        _version: None | str = None,
    ):
        if _version is not None:
            assert _version == _FORMAT_VERSION

        self._codec = (
            codec
            if isinstance(codec, Codec)
            else numcodecs.registry.get_codec(codec)
            if codec is not None
            else None
        )

        lossless = (
            lossless
            if isinstance(lossless, Lossless)
            else Lossless(**lossless)
            if lossless is not None
            else Lossless()
        )
        self._lossless_for_codec = (
            lossless.for_codec
            if isinstance(lossless.for_codec, Codec)
            else numcodecs.registry.get_codec(codec)
            if lossless.for_codec is not None
            else None
        )
        self._lossless_for_safeguards = (
            lossless.for_safeguards
            if isinstance(lossless.for_safeguards, Codec)
            else numcodecs.registry.get_codec(codec)
        )

        safeguards = [
            safeguard
            if isinstance(safeguard, Safeguard)
            else Safeguards[safeguard["kind"]].value(
                **{p: v for p, v in safeguard.items() if p != "kind"}
            )
            for safeguard in safeguards
        ]

        self._elementwise_safeguards = tuple(
            safeguard
            for safeguard in safeguards
            if isinstance(safeguard, ElementwiseSafeguard)
        )
        safeguards = [
            safeguard
            for safeguard in safeguards
            if not isinstance(safeguard, ElementwiseSafeguard)
        ]

        assert len(safeguards) == 0, f"unsupported safeguards {safeguards:!r}"

    def encode(self, buf: Buffer) -> Buffer:
        """Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        """

        data = numcodecs.compat.ensure_ndarray(buf)

        assert data.dtype in _SUPPORTED_DTYPES, (
            f"can only encode arrays of dtype {', '.join(d.str for d in _SUPPORTED_DTYPES)}"
        )

        if self._codec is None:
            encoded_bytes = b""
            decoded = np.zeros_like(data)
        else:
            encoded = self._codec.encode(np.copy(data))
            encoded = numcodecs.compat.ensure_ndarray(encoded)

            assert encoded.dtype == np.dtype("uint8"), "codec must encode to bytes"
            assert len(encoded.shape) <= 1, "codec must encode to 1D bytes"
            encoded_bytes = numcodecs.compat.ensure_bytes(encoded)

            decoded = np.empty_like(data)
            decoded = self._codec.decode(
                np.frombuffer(
                    np.copy(encoded_bytes), dtype="uint8", count=len(encoded_bytes)
                ),
                out=decoded,
            )
            decoded = numcodecs.compat.ensure_ndarray(decoded)

            if self._lossless_for_codec is not None:
                encoded_bytes = numcodecs.compat.ensure_bytes(
                    self._lossless_for_codec.encode(encoded_bytes)
                )

        assert decoded.dtype == data.dtype, "codec must roundtrip dtype"
        assert decoded.shape == data.shape, "codec must roundtrip shape"

        all_ok = True
        for safeguard in self._elementwise_safeguards:
            if not safeguard.check(data, decoded):
                all_ok = False
                break

        if all_ok:
            correction_bytes = b""
        else:
            all_intervals = []
            for safeguard in self._elementwise_safeguards:
                intervals = safeguard.compute_safe_intervals(data)
                assert np.all(intervals.contains(data)), (
                    f"elementwise safeguard {safeguard!r}'s intervals must contain the original data"
                )
                all_intervals.append(intervals)

            combined_intervals = all_intervals[0]
            for intervals in all_intervals[1:]:
                combined_intervals = combined_intervals.intersect(intervals)
            correction = combined_intervals.encode(decoded)

            for safeguard, intervals in zip(
                self._elementwise_safeguards, all_intervals
            ):
                assert np.all(intervals.contains(correction)), (
                    f"{safeguard!r} interval does not contain the correction {correction!r}"
                )
                assert safeguard.check(data, correction), (
                    f"{safeguard!r} check fails after correction {correction!r}"
                )

            correction_bytes = ElementwiseSafeguard._encode_correction(
                decoded,
                correction,
                self._lossless_for_safeguards,
            )

        correction_len = varint.encode(len(correction_bytes))

        return correction_len + encoded_bytes + correction_bytes

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        out : None | Buffer
            Writeable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style
            buffer protocol.
        """

        assert out is not None, "can only decode into known dtype and shape"

        buf_array = numcodecs.compat.ensure_ndarray(buf)
        assert buf_array.dtype == np.dtype("uint8"), "codec must decode from bytes"
        assert len(buf_array.shape) <= 1, "codec must decode from 1D bytes"
        buf_bytes = numcodecs.compat.ensure_bytes(buf)

        buf_io = BytesIO(buf_bytes)
        correction_len = varint.decode_stream(buf_io)
        assert correction_len >= 0

        if correction_len > 0:
            encoded = buf_bytes[buf_io.tell() : -correction_len]
            correction = buf_bytes[-correction_len:]
        else:
            encoded = buf_bytes[buf_io.tell() :]
            correction = b""

        if self._codec is None:
            assert encoded == b"", "can only decode empy message without a codec"
            decoded = np.zeros_like(out)
        else:
            if self._lossless_for_codec is not None:
                encoded = numcodecs.compat.ensure_bytes(
                    self._lossless_for_codec.decode(encoded)
                )

            decoded = self._codec.decode(
                np.frombuffer(encoded, dtype="uint8", count=len(encoded)), out=out
            )

        if correction_len > 0:
            corrected = ElementwiseSafeguard._apply_correction(
                decoded,
                correction,
                self._lossless_for_safeguards,
            )
        else:
            corrected = decoded

        return numcodecs.compat.ndarray_copy(corrected, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of the codec with safeguards.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this stack from the returned config.

        Returns
        -------
        config : dict
            Configuration of the codec with safeguards.
        """

        return dict(
            id=type(self).codec_id,
            _version=_FORMAT_VERSION,
            codec=None if self._codec is None else self._codec.get_config(),
            safeguards=[
                safeguard.get_config() for safeguard in self._elementwise_safeguards
            ],
            lossless=dict(
                for_codec=None
                if self._lossless_for_codec is None
                else self._lossless_for_codec.get_config(),
                for_safeguards=self._lossless_for_safeguards.get_config(),
            ),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(codec={self._codec!r}, safeguards={list(self._elementwise_safeguards)!r}, lossless={Lossless(for_codec=self._lossless_for_codec, for_safeguards=self._lossless_for_safeguards)!r})"

    def map(self, mapper: Callable[[Codec], Codec]) -> "SafeguardsCodec":
        """
        Apply the `mapper` to this codec with safeguards.
        In the returned
        [`SafeguardsCodec`][numcodecs_safeguards.SafeguardsCodec], the codec is
        replaced by its mapped codec.

        The `mapper` should recursively apply itself to any inner codecs that
        also implement the
        [`CodecCombinatorMixin`][numcodecs_combinators.abc.CodecCombinatorMixin]
        mixin.

        To automatically handle the recursive application as a caller, you can
        use
        ```python
        numcodecs_combinators.map_codec(codec, mapper)
        ```
        instead.

        Parameters
        ----------
        mapper : Callable[[Codec], Codec]
            The callable that should be applied to this codec to map over this
            codec with safeguards.

        Returns
        -------
        mapped : SafeguardsCodec
            The mapped codec with safeguards.
        """

        return SafeguardsCodec(
            codec=None if self._codec is None else mapper(self._codec),
            safeguards=self._elementwise_safeguards,
            lossless=Lossless(
                for_codec=None
                if self._lossless_for_codec is None
                else mapper(self._lossless_for_codec),
                for_safeguards=mapper(self._lossless_for_safeguards),
            ),
            _version=self._version,
        )


numcodecs.registry.register_codec(SafeguardsCodec)
