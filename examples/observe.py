import hashlib
import json
from collections import defaultdict
from contextlib import contextmanager

import numcodecs_observers
from numcodecs.abc import Codec
from numcodecs_observers.bytesize import BytesizeObserver
from numcodecs_observers.walltime import WalltimeObserver
from numcodecs_wasm import WasmCodecInstructionCounterObserver


# based on https://death.andgravity.com/stable-hashing
def json_hash(j) -> str:
    return hashlib.md5(
        json.dumps(
            j,
            ensure_ascii=False,
            sort_keys=True,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@contextmanager
def observe(codec: Codec, results: list) -> Codec:
    def result_to_json(r):
        hash_to_codecs = defaultdict(list)
        for c in r.keys():
            hash_to_codecs[json_hash(c.get_config())].append(c)
        codec_to_hash = dict()
        for h, cs in hash_to_codecs.items():
            for i, c in enumerate(cs):
                codec_to_hash[c] = h if len(cs) == 1 else f"{h}#{i}"
        return {codec_to_hash[c]: rs for c, rs in r.items()}

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
        codec_class = codec.__class__
        codec_encode = codec_class.encode
        codec_decode = codec_class.decode

        def my_codec_encode(self, buf):
            observers = [
                observer.observe_encode(codec_._codec, buf)
                for observer in codec_._observers
            ]

            encoded = codec_encode(codec_._codec, buf)

            for observer in reversed(observers):
                observer(encoded)

            return encoded

        def my_codec_decode(self, buf, out=None):
            observers = [
                observer.observe_decode(codec_._codec, buf, out=out)
                for observer in codec_._observers
            ]

            decoded = codec_decode(codec_._codec, buf, out=out)

            for observer in reversed(observers):
                observer(decoded)

            return decoded

        my_codec_class = type(
            codec.__class__.__name__,
            (codec.__class__,),
            dict(
                __slots__=(),
                encode=my_codec_encode,
                decode=my_codec_decode,
            ),
        )

        codec.__class__ = my_codec_class

        try:
            yield codec
        finally:
            codec.__class__ = codec_class

    results.append(
        dict(
            codec=codec.get_config(),
            encoded_bytes=result_to_json(
                {c: [s.post for s in ss] for c, ss in nbytes.encode_sizes.items()}
            ),
            decoded_bytes=result_to_json(
                {c: [s.post for s in ss] for c, ss in nbytes.decode_sizes.items()}
            ),
            encode_timing=result_to_json(timing.encode_times),
            decode_timing=result_to_json(timing.decode_times),
            encode_instructions=result_to_json(instructions.encode_instructions),
            decode_instructions=result_to_json(instructions.decode_instructions),
        )
    )
