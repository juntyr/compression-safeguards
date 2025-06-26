r"""
# Fearless lossy compression with `numcodecs-safeguards`

Lossy compression can be *scary* as valuable information or features of the
data may be lost.

By using [`Safeguards`][compression_safeguards.Safeguards] to **guarantee**
your safety requirements, lossy compression can be applied safely and
*without fear*.

## Overview

This package provides the
[`SafeguardsCodec`][numcodecs_safeguards.SafeguardsCodec] adapter /
meta-compressor that can be wrapped around *any* existing (lossy)
[`numcodecs.abc.Codec`][numcodecs.abc.Codec] to *guarantee* that certain
properties of the original data are preserved by compression.

The `SafeguardsCodec` treats the wrapped inner codec as a blackbox. To
guarantee the user's safety requirements, it post-processes the decompressed
data, if necessary. If no correction is needed, the `SafeguardsCodec` only has
a one-byte overhead for the compressed data and a computational overhead at
compression time.

By using the `SafeguardsCodec` adapter, badly behaving lossy codecs become safe
to use, at the cost of potentially less efficient compression, and lossy
compression can be applied *without fear*.

## Example

You can wrap an existing codec with e.g. a relative error bound of
$eb_{rel} = 1\%$ and preserve data signs as follows:

```py
import numpy as np
from numcodecs.fixedscaleoffset import FixedScaleOffset
from numcodecs_safeguards import SafeguardsCodec

# use any numcodecs-compatible codec
# here we quantize data >= -10 with one decimal digit
lossy_codec = FixedScaleOffset(
    offset=-10, scale=10, dtype="float64", astype="uint8",
)

# wrap the codec in the `SafeguardsCodec` and specify the safeguards to apply
sg_codec = SafeguardsCodec(codec=lossy_codec, safeguards=[
    # guarantee a relative error bound of 1%:
    #   |x - x'| <= |x| * 0.01
    dict(kind="eb", type="rel", eb=0.01),
    # guarantee that the sign is preserved:
    #   sign(x) = sign(x')
    dict(kind="sign"),
])

# some n-dimensional data
data = np.linspace(-10, 10, 21)

# encode and decode the data
encoded = sg_codec.encode(data)
decoded = sg_codec.decode(encoded)

# the safeguard properties are guaranteed to hold
assert np.all(np.abs(data - decoded) <= np.abs(data) * 0.01)
assert np.all(np.sign(data) == np.sign(decoded))
```

Please refer to the
[`compression_safeguards.SafeguardKind`][compression_safeguards.SafeguardKind]
for an enumeration of all supported safeguards.
"""

__all__ = ["SafeguardsCodec"]

from collections.abc import Collection, Set
from io import BytesIO
from typing import Callable

import numcodecs
import numcodecs.compat
import numcodecs.registry
import numpy as np
import varint
from compression_safeguards.api import Safeguards
from compression_safeguards.safeguards.abc import Safeguard
from compression_safeguards.utils.bindings import Bindings, Parameter
from compression_safeguards.utils.cast import as_bits
from numcodecs.abc import Codec
from numcodecs_combinators.abc import CodecCombinatorMixin
from typing_extensions import Buffer  # MSPV 3.12

from .lossless import Lossless


class SafeguardsCodec(Codec, CodecCombinatorMixin):
    """
    An adaptor codec that uses [`Safeguards`][compression_safeguards.Safeguards]
    to guarantee certain properties / safety requirements are upheld by the
    wrapped codec.

    Parameters
    ----------
    codec : dict | Codec
        The codec to wrap with safeguards. It can either be passed as a codec
        configuration [`dict`][dict], which is passed to
        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec],
        or an already initialized [`Codec`][numcodecs.abc.Codec]. If you want to
        wrap a sequence or stack of codecs, you can use the
        [`numcodecs_combinators.stack.CodecStack`][numcodecs_combinators.stack.CodecStack]
        combinator.

        It is desirable to perform lossless compression after applying the
        safeguards (rather than before), e.g. by customising the
        [`Lossless.for_codec`][numcodecs_safeguards.lossless.Lossless.for_codec]
        field of the `lossless` parameter.

        The `codec` combined with its `lossless` encoding must encode to a 1D
        buffer of bytes. It is also recommended that the `codec` can
        [`decode`][numcodecs.abc.Codec.decode]
        without receiving the output data type and shape via the `out`
        parameter. If the `codec` does not fulfil these requirements, it can be
        wrapped inside the
        [`numcodecs_combinators.framed.FramedCodecStack`][numcodecs_combinators.framed.FramedCodecStack]
        combinator.

        It is also possible to use *only* the safeguards to encode the data by
        passing [`numcodecs_zero.ZeroCodec()`][numcodecs_zero.ZeroCodec] or
        `dict(id="zero")` to `codec`. The zero codec only encodes the data
        type and shape, not the data values themselves, and decodes to all-
        zeros, forcing the safeguards to correct (almost) all values.
    safeguards : Collection[dict | Safeguard]
        The safeguards that will be applied to the codec. They can either be
        passed as a safeguard configuration [`dict`][dict] or an already
        initialized
        [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard].

        Please refer to the
        [`SafeguardKind`][compression_safeguards.safeguards.SafeguardKind]
        for an enumeration of all supported safeguards.

        The `SafeguardsCodec` does not (yet) support safeguards with late-
        bound parameters, e.g. the
        [`SelectSafeguard`][compression_safeguards.safeguards.combinators.select.SelectSafeguard].
    lossless : None | dict | Lossless, optional
        The lossless encoding that is applied after the codec and the
        safeguards:

        - [`Lossless.for_codec`][numcodecs_safeguards.lossless.Lossless.for_codec]
          specifies the lossless encoding that is applied to the encoded output
          of the wrapped `codec`. By default, no additional lossless encoding
          is applied.
        - [`Lossless.for_safeguards`][numcodecs_safeguards.lossless.Lossless.for_safeguards]
          specifies the lossless encoding that is applied to the encoded
          correction that the safeguards produce. By default, Huffman encoding
          followed by Zstandard is applied.

        The lossless encoding must encode to a 1D buffer of bytes.
    _version : ...
        The codecs's version. Do not provide this parameter explicitly.
    """

    __slots__ = (
        "_codec",
        "_safeguards",
        "_lossless_for_codec",
        "_lossless_for_safeguards",
    )
    _codec: Codec
    _safeguards: Safeguards
    _lossless_for_codec: None | Codec
    _lossless_for_safeguards: Codec

    codec_id: str = "safeguards"  # type: ignore

    def __init__(
        self,
        *,
        codec: dict | Codec,
        safeguards: Collection[dict | Safeguard],
        lossless: None | dict | Lossless = None,
        _version: None | str = None,
    ):
        self._safeguards = Safeguards(safeguards=safeguards, _version=_version)

        assert len(self._safeguards.late_bound - self.builtin_late_bound) == 0, (
            "SafeguardsCodec does not (yet) support non-built-in late-bound parameters"
        )

        self._codec = (
            codec if isinstance(codec, Codec) else numcodecs.registry.get_codec(codec)
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
            else numcodecs.registry.get_codec(lossless.for_codec)
            if lossless.for_codec is not None
            else None
        )
        self._lossless_for_safeguards = (
            lossless.for_safeguards
            if isinstance(lossless.for_safeguards, Codec)
            else numcodecs.registry.get_codec(lossless.for_safeguards)
        )

    @property
    def safeguards(self) -> Collection[Safeguard]:
        """
        The collection of safeguards that will be applied.
        """

        return self._safeguards.safeguards

    @property
    def builtin_late_bound(self) -> Set[Parameter]:
        """
        The set of built-in late-bound constants that the numcodecs-safeguards
        provide automatically, which include the safeguards' built-ins as well
        as `$x_min` and `$x_max`.
        """

        return frozenset(self._safeguards.builtin_late_bound) | frozenset(
            [Parameter("$x_min"), Parameter("$x_max")]
        )

    def encode(self, buf: Buffer) -> bytes:
        """Encode the data in `buf`.

        The encoded data is defined by the following *stable* format:

        ```
        ULEB128(len(correction_bytes)), encoded_bytes, correction_bytes
        ```

        where

        - `ULEB128` refers to the
          [unsigned LEB128](https://en.wikipedia.org/wiki/LEB128#Unsigned_LEB128)
          (little endian base 128) variable length encoding for unsigned
          integers
        - `encoded_bytes` refers to the encoded bytes produced by the codec and
          its optional lossless encoding
        - `correction_bytes` refers to the encoded correction bytes produced by
          the safeguards and their lossless encoding

        If no correction is required, `correction_bytes` is empty and there is
        only a single-byte overhead from using the safeguards.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded data as a bytestring.
        """

        data = (
            buf if isinstance(buf, np.ndarray) else numcodecs.compat.ensure_ndarray(buf)
        )

        assert data.dtype in Safeguards.supported_dtypes(), (
            f"can only encode arrays of dtype {', '.join(d.str for d in Safeguards.supported_dtypes())}"
        )

        encoded = self._codec.encode(np.copy(data))
        encoded = numcodecs.compat.ensure_ndarray(encoded)

        # check that decoding with `out=None` works
        try:
            decoded = self._codec.decode(np.copy(encoded), out=None)
        except Exception as err:
            message = (
                "decoding with `out=None` failed\n\n"
                "consider using wrapping the codec in the "
                "`numcodecs_combinators.framed.FramedCodecStack(codec)` "
                "combinator if the codec requires knowing the output data "
                "type and shape for decoding"
            )

            # MSPV 3.11
            if getattr(err, "add_note", None) is not None:
                err.add_note(message)  # type: ignore
                raise err
            else:
                raise ValueError(message) from err
        decoded = numcodecs.compat.ensure_ndarray(decoded)

        if self._lossless_for_codec is not None:
            encoded = self._lossless_for_codec.encode(encoded)

        try:
            assert encoded.dtype == np.dtype("uint8"), (
                "codec and lossless must encode to bytes"
            )
            assert encoded.ndim <= 1, "codec and lossless must encode to 1D bytes"
            encoded_bytes = numcodecs.compat.ensure_bytes(encoded)

            assert decoded.dtype == data.dtype, "codec must roundtrip dtype"
            assert decoded.shape == data.shape, "codec must roundtrip shape"
        except Exception as err:
            message = (
                "consider using wrapping the codec in the "
                "`numcodecs_combinators.framed.FramedCodecStack(codec)` "
                "combinator to encode to bytes and preserve the data dtype and"
                "shape"
            )

            # MSPV 3.11
            if getattr(err, "add_note", None) is not None:
                err.add_note(message)  # type: ignore
                raise err
            else:
                raise ValueError(message) from err

        late_bound = Bindings.empty()
        late_bound_reqs = self._safeguards.late_bound

        if "$x_min" in late_bound_reqs:
            late_bound = late_bound.update(
                **{
                    "$x_min": np.nanmin(data)
                    if data.size > 0
                    else np.array(0, dtype=data.dtype)
                }
            )
        if "$x_max" in late_bound_reqs:
            late_bound = late_bound.update(
                **{
                    "$x_max": np.nanmax(data)
                    if data.size > 0
                    else np.array(0, dtype=data.dtype)
                }
            )

        # the codec always compresses the complete data ... at least chunking
        #  is not our concern
        correction: np.ndarray = self._safeguards.compute_correction(
            data,
            decoded,
            late_bound=late_bound,
        )

        if np.all(correction == 0):
            correction_bytes = b""
        else:
            correction_bytes = numcodecs.compat.ensure_bytes(
                self._lossless_for_safeguards.encode(correction)
            )

        correction_len = varint.encode(len(correction_bytes))

        return correction_len + encoded_bytes + correction_bytes

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        out : None | Buffer
            Writeable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style
            buffer protocol.
        """

        buf_array = numcodecs.compat.ensure_ndarray(buf)
        assert buf_array.dtype == np.dtype("uint8"), "codec must decode from bytes"
        assert buf_array.ndim <= 1, "codec must decode from 1D bytes"
        buf_bytes = numcodecs.compat.ensure_bytes(buf)

        buf_io = BytesIO(buf_bytes)
        correction_len = varint.decode_stream(buf_io)
        assert correction_len >= 0

        if correction_len > 0:
            encoded = buf_bytes[buf_io.tell() : -correction_len]
            correction_bytes = buf_bytes[-correction_len:]
        else:
            encoded = buf_bytes[buf_io.tell() :]
            correction_bytes = b""

        if self._lossless_for_codec is not None:
            encoded = numcodecs.compat.ensure_bytes(
                self._lossless_for_codec.decode(encoded)
            )

        decoded = self._codec.decode(
            np.frombuffer(encoded, dtype="uint8", count=len(encoded)), out=out
        )

        if correction_len > 0:
            correction = (
                numcodecs.compat.ensure_ndarray(
                    self._lossless_for_safeguards.decode(correction_bytes)
                )
                .view(as_bits(decoded).dtype)
                .reshape(decoded.shape)
            )

            corrected = self._safeguards.apply_correction(decoded, correction)
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
            codec=self._codec.get_config(),
            safeguards=[
                safeguard.get_config() for safeguard in self._safeguards.safeguards
            ],
            lossless=dict(
                for_codec=None
                if self._lossless_for_codec is None
                else self._lossless_for_codec.get_config(),
                for_safeguards=self._lossless_for_safeguards.get_config(),
            ),
            _version=self._safeguards.version,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(codec={self._codec!r}, safeguards={list(self._safeguards.safeguards)!r}, lossless={Lossless(for_codec=self._lossless_for_codec, for_safeguards=self._lossless_for_safeguards)!r})"

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
            codec=mapper(self._codec),
            safeguards=self._safeguards.safeguards,
            lossless=Lossless(
                for_codec=None
                if self._lossless_for_codec is None
                else mapper(self._lossless_for_codec),
                for_safeguards=mapper(self._lossless_for_safeguards),
            ),
            _version=self._safeguards.version,
        )


numcodecs.registry.register_codec(SafeguardsCodec)
