# Fearless lossy compression with numcodecs-guardrail

Lossy compression can be scary as valuable information may be lost. This package provides several guardrail adapters that can be applied to *any* existing (lossy) compressor to *guarantee* that certain properties about the compression error are upheld. By using these adapters, badly behaving lossy compressors become safe to use, at the cost of potentially less efficient compression, and lossy compression can be applied without fear.

The guardrails implemented in this package are designed to be convenient to apply to any lossy compression task:

1. They are *guaranteed* to *always* uphold the property they provide.
2. They are designed to minimise the overhead in compressed message size for elements where the properties were already satisfied by the wrapped compressor.

If applied to a

- safe compressor that (by chance or design) already satisfies the properties, the overhead is a constant one byte overhead
- unsafe compressor that violates any of the properties, the properties are restored at a byte overhead that scales linearly with the number of elements that had to be corrected

The GuardrailCodec adapter provided by this package can wrap any existing codec[^1] implementing the numcodecs API[^2] that encodes a buffer (e.g. an ndarray or bytes) to bytes. It is desirable to perform lossless compression after applying the guardrails (rather than before).

[^1]: If you want to wrap a sequence or stack of codecs, you can use the StackCodec combinator from the numcodecs-combinators package.
[^2]: The method implemented in this package is not specific to the numcodecs API. Please reach out if you'd like to help bring the guardrails to a different compression API or language.