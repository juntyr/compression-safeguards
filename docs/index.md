# Fearless lossy compression with `numcodecs-guardrails`

Lossy compression can be scary as valuable information may be lost. This package provides several guardrail adapters that can be applied to *any* existing (lossy) compressor to *guarantee* that certain properties about the compression error are upheld. By using these adapters, badly behaving lossy compressors become safe to use, at the cost of potentially less efficient compression, and lossy compression can be applied without fear.


## Design and Guarantees

The guardrails implemented in this package are designed to be convenient to apply to any lossy compression task:

1. They are *guaranteed* to *always* uphold the property they provide.

2. They are designed to minimise the overhead in compressed message size for elements where the properties were already satisfied by the wrapped compressor.

If applied to

- a safe compressor that (by chance or design) already satisfies the properties, there is a constant single-byte overhead

- an unsafe compressor that violates any of the properties, the properties are restored at a byte overhead that scales linearly with the number of elements that have to be corrected


## Provided guardrails

This package currently implements the following guardrails

- `abs` (absolute error bound):

    The absolute elementwise error is guarantees to be less than or equal to the provided bound. In cases where the arithmetic evaluation of the error bound not well-defined, e.g. for infinite or NaN values, producing the exact same bitpattern is defined to satisfy the error bound.

- `rel-or-abs` (relative [or absolute] error bound):

    The absolute elementwise error between the *logarithms*\* of the values is guaranteed to be less than or equal to $\log(1 + eb_{rel})$ where $eb_{rel}$ is e.g. 2%. The logarithm* here is adapted to support positive, negative, and zero values. For values close to zero, where the relative error is not well defined, the absolute elementwise error is guaranteed to be less than or equal to the absolute error bound.
    
    Put simply, each element satisfies the relative or the absolute error bound (or both). In cases where the arithmetic evaluation of the error bound is not well-defined, e.g. for infinite or NaN values, producing the exact same bitpattern is defined to satisfy the error bound.


## Usage

The [`GuardrailsCodec`][numcodecs_guardrails.GuardrailsCodec] adapter provided by this package can wrap any existing [`Codec`][numcodecs.abc.Codec] [^1] implementing the [`numcodecs`][numcodecs] API [^2] that encodes a buffer (e.g. an ndarray or bytes) to bytes. It is desirable to perform lossless compression after applying the guardrails (rather than before).

[^1]: If you want to wrap a sequence or stack of codecs, you can use the [`CodecStack`](https://numcodecs-combinators.readthedocs.io/en/stable/api/#numcodecs_combinators.CodecStack) combinator from the [`numcodecs-combinators`](https://numcodecs-combinators.readthedocs.io/en/stable/) package.
[^2]: The method implemented in this package is not specific to the [`numcodecs`][numcodecs] API. Please reach out if you'd like to help bring the guardrails to a different compression API or language.


## Funding

The Online Laboratory for Climate Science and Meteorology has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
