import hashlib
import json
from collections import defaultdict
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Any, cast

import numcodecs_observers
from numcodecs.abc import Codec
from numcodecs_observers.bytesize import BytesizeObserver
from numcodecs_observers.hash import HashableCodec
from numcodecs_observers.walltime import WalltimeObserver
from numcodecs_wasm import WasmCodecInstructionCounterObserver
from typing_extensions import Buffer  # MSPV 3.12


# based on https://death.andgravity.com/stable-hashing
def json_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _result_to_json(result: Mapping[HashableCodec, list]) -> dict[str, list]:
    hash_to_codecs = defaultdict(list)
    for c in result.keys():
        hash_to_codecs[json_hash(c.get_config())].append(c)
    codec_to_hash = dict()
    for h, cs in hash_to_codecs.items():
        for i, c in enumerate(cs):
            codec_to_hash[c] = h if len(cs) == 1 else f"{h}#{i}"
    return {codec_to_hash[c]: rs for c, rs in result.items()}


@contextmanager
def _observe(codec: Codec, observations: list[dict]) -> Generator[Codec]:
    codec_class = codec.__class__

    nbytes = BytesizeObserver()
    timing = WalltimeObserver()
    instructions = WasmCodecInstructionCounterObserver()

    with numcodecs_observers.observe(
        codec,
        observers=[
            nbytes,
            instructions,
            timing,
        ],
    ) as codec_:
        assert isinstance(codec_, numcodecs_observers._ObservingCodec)

        def encode(self, buf: Buffer) -> Buffer:
            observers = [
                observer.observe_encode(codec_._codec, buf)
                for observer in codec_._observers
            ]

            encoded = cast("Buffer", codec_class.encode(codec_._codec, buf))

            for observer in reversed(observers):
                observer(encoded)

            return encoded

        def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
            observers = [
                observer.observe_decode(codec_._codec, buf, out=out)
                for observer in codec_._observers
            ]

            decoded = cast("Buffer", codec_class.decode(codec_._codec, buf, out=out))

            for observer in reversed(observers):
                observer(decoded)

            return decoded

        codec.__class__ = type(
            codec.__class__.__name__,
            (codec.__class__,),
            dict(__slots__=(), encode=encode, decode=decode),
        )

        try:
            yield codec
        finally:
            codec.__class__ = codec_class

    observations.append(
        dict(
            codec=codec.get_config(),
            encoded_bytes=_result_to_json(
                {c: [s.post for s in ss] for c, ss in nbytes.encode_sizes.items()}
            ),
            decoded_bytes=_result_to_json(
                {c: [s.post for s in ss] for c, ss in nbytes.decode_sizes.items()}
            ),
            encode_timing=_result_to_json(timing.encode_times),
            decode_timing=_result_to_json(timing.decode_times),
            encode_instructions=_result_to_json(instructions.encode_instructions),
            decode_instructions=_result_to_json(instructions.decode_instructions),
        )
    )


def observe(codec: Codec, observations: list) -> AbstractContextManager[Codec]:
    return _observe(codec, observations)
