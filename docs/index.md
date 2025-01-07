# Fearless lossy compression with `numcodecs-guardrails`

Lossy compression can be scary as valuable information may be lost. This package provides the [`GuardrailsCodec`][numcodecs_guardrails.GuardrailsCodec] adapter and several [`Guardrails`][numcodecs_guardrails.Guardrails] that can be applied to *any* existing (lossy) compressor to *guarantee* that certain properties about the compression error are upheld. Note that the wrapped compressor is treated as a blackbox and the decompressed data is postprocessed to re-establish the properties, if necessary. By using the `GuardrailsCodec` adapter, badly behaving lossy compressors become safe to use, at the cost of potentially less efficient compression, and lossy compression can be applied without fear.

## Design and Guarantees

The guardrails implemented in this package are designed to be convenient to apply to any lossy compression task:

1. They are *guaranteed* to *always* uphold the property they provide.

2. They are designed to minimise the overhead in compressed message size for elements where the properties were already satisfied by the wrapped compressor.

3. Their implementations are prioritise correctness in all cases and simplicity of verification over performance (and byte ratio overhead when the properties are violated by many data elements). Please refer to the [related projects](#related-projects) section for alternatives with different design considerations.

If applied to

- a safe compressor that (by chance or design) already satisfies the properties, there is a constant single-byte overhead

- an unsafe compressor that violates any of the properties, the properties are restored at a byte overhead that scales linearly with the number of elements that have to be corrected


## Provided guardrails

This package currently implements the following [guardrails][numcodecs_guardrails.Guardrails]:

- [`abs`][numcodecs_guardrails.guardrails.elementwise.abs.AbsoluteErrorBoundGuardrail] (absolute error bound):

    The absolute elementwise error is guaranteed to be less than or equal to the provided bound. In cases where the arithmetic evaluation of the error bound not well-defined, e.g. for infinite or NaN values, producing the exact same bitpattern is defined to satisfy the error bound.

- [`rel_or_abs`][numcodecs_guardrails.guardrails.elementwise.rel_or_abs.RelativeOrAbsoluteErrorBoundGuardrail] (relative [or absolute] error bound):

    The absolute elementwise error between the *logarithms*\* of the values is guaranteed to be less than or equal to $\log(1 + eb_{rel})$ where $eb_{rel}$ is e.g. 2%. The logarithm* here is adapted to support positive, negative, and zero values. For values close to zero, where the relative error is not well-defined, the absolute elementwise error is guaranteed to be less than or equal to the absolute error bound.
    
    Put simply, each element satisfies the relative or the absolute error bound (or both). In cases where the arithmetic evaluation of the error bound is not well-defined, e.g. for infinite or NaN values, producing the exact same bitpattern is defined to satisfy the error bound.

- [`monotonicity`][numcodecs_guardrails.guardrails.elementwise.monotonicity.MonotonicityPreservingGuardrail] (monotonicity-preserving):

    Sequences that are monotonic in the input are guaranteed to be monotonic in the decompressed output. Monotonic sequences are detected using per-axis moving windows. The guardrail supports enforcing four levels of [monotonicity][numcodecs_guardrails.guardrails.elementwise.monotonicity.Monotonicity]: `strict`, `strict_with_consts`, `strict_to_weak`, `weak`. Windows that are not monotonic or contain non-finite data are skipped. Axes that have fewer elements than the window size are skipped as well.


## Usage

The [`GuardrailsCodec`][numcodecs_guardrails.GuardrailsCodec] adapter provided by this package can wrap any existing [`Codec`][numcodecs.abc.Codec] [^1] implementing the [`numcodecs`][numcodecs] API [^2] that encodes a buffer (e.g. an ndarray or bytes) to bytes. It is desirable to perform lossless compression after applying the guardrails (rather than before).

You can wrap an existing codec with e.g. an absolute error bound of $eb_{abs} = 0.1$  as follows:

```python
from numcodecs_guardrails import Guardrails, GuardrailsCodec

GuardrailsCodec(codec=codec, guardrails=[Guardrails.abs.value(eb_abs=0.1)])
```

You can also provide just the configuration for the codec or any of the guardrails:

```python
from numcodecs_guardrails import GuardrailsCodec

GuardrailsCodec(
    codec=dict(id="my-codec", ...),
    guardrails=[dict(kind="abs", eb_abs=0.1)],
)
```

Finally, you can also use [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec] to instantiate the codec with guardrails from one combined configuration:

```python
import numcodecs

numcodecs.registry.get_codec(dict(
    id="guardrails",
    codec=dict(id="my-codec", ...),
    guardrails=[dict(kind="abs", eb_abs=0.1)],
))
```

[^1]: If you want to wrap a sequence or stack of codecs, you can use the [`CodecStack`](https://numcodecs-combinators.readthedocs.io/en/stable/api/#numcodecs_combinators.CodecStack) combinator from the [`numcodecs-combinators`](https://numcodecs-combinators.readthedocs.io/en/stable/) package.
[^2]: The method implemented in this package is not specific to the [`numcodecs`][numcodecs] API. Please reach out if you'd like to help bring the guardrails to a different compression API or language.


## Related Projects

### SZ3 error compression

[SZ3](https://github.com/szcompressor/SZ3) >=3.2.0 provides the `CmprAlgo=ALGO_NOPRED` option, with which the compression error $error = decompressed - data$ of another lossy compressor can itself be lossy-compressed with e.g. an absolute error bound. Using this option, any compressor can be transformed into an error bounded compressor.

SZ3's error compression can provide higher compression ratios if most data elements are expected to violate the error bound, e.g. when wrapping a lossy compressor that does *not* bound its errors. However, SZ3 has a higher byte overhead than `numcodecs-guardrails` if all elements already satisfy the bound.

**TLDR:** You can SZ3 to transform a *known* *unbounded* lossy compressor into an (absolute) error-bound compressor. Use `numcodecs-guardrails` to wrap *any* compressor (unbounded, best-effort bounded, or strictly bounded) to guarantee it is error bounded.


## Citation

Please cite this work as follows:

> Tyree, J. (2025). `numcodecs-guardrails` &ndash; Fearless lossy compression using guardrails. Available from: <https://github.com/juntyr/numcodecs-guardrails>

Please also refer to the [CITATION.cff](https://github.com/juntyr/numcodecs-guardrails/blob/main/CITATION.cff) file and refer to <https://citation-file-format.github.io> to extract the citation in a format of your choice.


## Funding

The `numcodecs-guardrails` package has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
