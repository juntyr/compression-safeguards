# Fearless lossy compression with `numcodecs-safeguards`

Lossy compression [^1] can be *scary* as valuable information or features of the data may be lost. This package provides several `SafeguardKind`s to express *your* requirements for lossy compression to be safe to use and to *guarantee* that they are upheld by lossy compression. By using safeguards to ensure your safety requirements, lossy compression can be applied safely and *without fear*.

[^1]: Lossy compression methods reduce data size by only storing an approximation of the data. In contrast to lossless compression methods, lossy compression loses information about the data, e.g. by reducing its resolution (only store every $n$th element) precision (only store $n$ digits after the decimal point), smoothing, etc. Therefore, lossy compression methods provide a tradeoff between size reduction and quality preservation.

## (a) Safeguards for users of lossy compression

This package provides the `SafeguardsCodec` adapter that can be wrapped around *any* existing (lossy) compressor to *guarantee* that certain properties about the data are upheld.

Note that the wrapped compressor is treated as a blackbox and the decompressed data is postprocessed to re-establish the properties, if necessary.

By using the `SafeguardsCodec` adapter, badly behaving lossy compressors become safe to use, at the cost of potentially less efficient compression, and lossy compression can be applied *without fear*.

## (b) Safeguards for developers of lossy compressors

Safeguards can also fill the role of a quantizer, which is part of many (predictive) (error-bounded) compressors. If you currently use e.g. a linear quantizer module in your compressor to provide an absolute error bound, you could instead adapt the `Safeguards`, quantize to their computed correction values, and thereby offer a larger selection of safety requirements that your compressor can then guarantee. Note, however, that only pointwise safeguards can be used when quantizing data elements one-by-one.


## Design and Guarantees

The safeguards implemented in this package are designed to be convenient to apply to any lossy compression task:

1. They are *guaranteed* to *always* uphold the property they provide.

2. They are designed to minimise the overhead in compressed message size for elements where the properties were already satisfied by the wrapped compressor.

3. Their implementations are prioritise correctness in all cases and simplicity of verification over performance (and byte ratio overhead when the properties are violated by many data elements). Please refer to the [related projects](#related-projects) section for alternatives with different design considerations.

If applied to

- a safe compressor that (by chance or design) already satisfies the properties, there is a constant single-byte overhead

- an unsafe compressor that violates any of the properties, the properties are restored at a byte overhead that scales linearly with the number of elements that have to be corrected


## Provided safeguards

This package currently implements the following safeguards:

### Error Bounds (pointwise)

- `abs` (absolute error bound):

    The pointwise absolute error is guaranteed to be less than or equal to the provided bound. Infinite values are preserved with the same bit pattern. The safeguard can be configured such that NaN values are preserved with the same bit pattern, or that decoding a NaN value to a NaN value with a different bit pattern also satisfies the error bound.

- `rel` (relative error bound):

    The pointwise relative error is guaranteed to be less than or equal to the provided bound. Zero values are preserved with the same bit pattern. Infinite values are preserved with the same bit pattern. The safeguard can be configured such that NaN values are preserved with the same bit pattern, or that decoding a NaN value to a NaN value with a different bit pattern also satisfies the error bound.

- `ratio` (ratio [decimal] error bound):

    It is guaranteed that the ratios between the original and the decoded values and their inverse ratios are less than or equal to the provided bound. The ratio error is defined to be infinite if the signs of the data and decoded data do not match. Since the provided error bound must be finite, this safeguard also guarantees that the sign of each decoded value matches the sign of each original value and that a decoded value is zero if and only if it is zero in the original data. The ratio error bound is sometimes also known as a decimal error bound if the ratio is expressed as the difference in orders of magnitude. This safeguard can also be used to guarantee a relative-like error bound. Infinite values are preserved with the same bit pattern. The safeguard can be configured such that NaN values are preserved with the same bit pattern, or that decoding a NaN value to a NaN value with a different bit pattern also satisfies the error bound.

### Error Bounds on derived Quantities of Interest (QoIs)

- `qoi_abs_pw` (absolute error bound on quantities of interest):

    The absolute error on a derived pointwise quantity of interest (QoI) is guaranteed to be less than or equal to the provided bound. The non-constant quantity of interest expression can contain the addition, multiplication, division, square root, exponentiation, logarithm, trigonometric, and hyperbolic operations over integer and floating point constants and the pointwise data value. Infinite quantities of interest are preserved with the same bit pattern. NaN quantities of interest remain NaN though not necessarily with the same bit pattern.

- `qoi_abs_stencil` (absolute error bound on quantities of interest over a neighbourhood):

    The absolute error on a derived quantity of interest (QoI) over a neighbourhood of data points is guaranteed to be less than or equal to the provided bound. The non-constant quantity of interest expression can contain the addition, multiplication, division, square root, exponentiation, logarithm, trigonometric, hyperbolic, array sum, matrix transpose, matrix multiplication, and finite difference operations over integer and floating point constants and arrays and the data neighbourhood. If applied to data with more dimensions than the data neighbourhood of the QoI requires, the data neighbourhood is applied independently along these extra axes. If the data neighbourhood uses the `valid` boundary condition along an axis, only data neighbourhoods centred on data points that have sufficient points before and after are safeguarded. If the axis is smaller than required by the neighbourhood along this axis, the data is not safeguarded at all. Using a different boundary condition ensures that all data points are safeguarded. Infinite quantities of interest are preserved with the same bit pattern. NaN quantities of interest remain NaN though not necessarily with the same bit pattern.

### Pointwise properties

- `zero` (zero/constant preserving):

    Values that are zero in the input are guaranteed to also be *exactly* zero in the decompressed output. This safeguard can also be used to enforce that another constant value is bitwise preserved, e.g. a missing value constant or a semantic "zero" value that is represented as a non-zero number. Beware that +0.0 and -0.0 are semantically equivalent in floating point but have different bitwise patterns. If you want to preserve both, you need to use two safeguards, one configured for each zero.

- `sign` (sign-preserving):

    Values are guaranteed to have the same sign (-1, 0, +1) in the decompressed output as they have in the input data. The sign for NaNs is derived from their sign bit, e.g. sign(-NaN) = -1. This safeguard should be combined with e.g. an error bound, as it by itself accepts *any* value with the same sign.

### Relationships between neighboring elements

- `monotonicity` (monotonicity-preserving):

    Sequences that are monotonic in the input are guaranteed to be monotonic in the decompressed output. Monotonic sequences are detected using per-axis moving windows of constant size. Typically, the window size should be chosen to be large enough to ignore noise but small enough to capture details. Four levels of monotonicity can be enforced: `strict`, `strict_with_consts`, `strict_to_weak`, `weak`. Windows that are not monotonic or contain non-finite data are skipped. If the `valid` boundary condition is used, axes that have fewer elements than
    the window size are skipped as well.

### Logical combinators (~pointwise)

- `all` (logical all / and):

    For each element, all of the combined safeguards' guarantees are upheld. At the moment, only pointwise and stencil safeguards and combinations thereof can be combined by this all-combinator.

- `any` (logical any / or):

    For each element, at least one of the combined safeguards' guarantees is upheld. At the moment, only pointwise and stencil safeguards and combinations thereof can be combined by this any-combinator.

- `safe` (logical truth):

    All elements always meet their guarantees and are thus always safe. This truth-combinator can be used, with care, with other logical combinators.

## Usage

The `SafeguardsCodec` adapter provided by this package can wrap any existing [`Codec`] [^2] implementing the [`numcodecs`] API [^3] that encodes a buffer (e.g. an ndarray or bytes) to bytes. It is desirable to perform lossless compression after applying the safeguards (rather than before).

You can wrap an existing codec with e.g. an absolute error bound of $eb_{abs} = 0.1$  as follows:

```python
from compression_safeguards import SafeguardKind
from numcodecs_safeguards import SafeguardsCodec

SafeguardsCodec(codec=codec, safeguards=[SafeguardKind.abs.value(eb_abs=0.1)])
```

You can also provide just the configuration for the codec or any of the safeguards:

```python
from numcodecs_safeguards import SafeguardsCodec

SafeguardsCodec(
    codec=dict(id="my-codec", ...),
    safeguards=[dict(kind="abs", eb_abs=0.1)],
)
```

Finally, you can also use `numcodecs.registry.get_codec(config)` to instantiate the codec with safeguards from one combined configuration:

```python
import numcodecs

numcodecs.registry.get_codec(dict(
    id="safeguards",
    codec=dict(id="my-codec", ...),
    safeguards=[dict(kind="abs", eb_abs=0.1)],
))
```

Please refer to the [API documentation](https://juntyr.github.io/numcodecs-safeguards/_ref/numcodecs_safeguards/codec/#numcodecs_safeguards.codec.SafeguardsCodec) for further information.

[^2]: If you want to wrap a sequence or stack of codecs, you can use the [`CodecStack`] combinator from the [`numcodecs-combinators`] package.
[^3]: The method implemented in this package is not specific to the [`numcodecs`] API. Please reach out if you'd like to help bring the safeguards to a different compression API or language.

[`Codec`]: https://numcodecs.readthedocs.io/en/stable/abc.html#numcodecs.abc.Codec
[`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/
[`CodecStack`]: https://numcodecs-combinators.readthedocs.io/en/stable/_ref/numcodecs_combinators/stack/#numcodecs_combinators.stack.CodecStack
[`numcodecs-combinators`]: https://numcodecs-combinators.readthedocs.io/en/stable/


## Related Projects

### SZ3 error compression

[SZ3](https://github.com/szcompressor/SZ3) >=3.2.0 provides the `CmprAlgo=ALGO_NOPRED` option, with which the compression error $error = decompressed - data$ of another lossy compressor can itself be lossy-compressed with e.g. an absolute error bound. Using this option, any compressor can be transformed into an error bounded compressor.

SZ3's error compression can provide higher compression ratios if most data elements are expected to violate the error bound, e.g. when wrapping a lossy compressor that does *not* bound its errors. However, SZ3 has a higher byte overhead than `numcodecs-safeguards` if all elements already satisfy the bound.

**TLDR:** You can use SZ3 to transform a *known* *unbounded* lossy compressor into an (absolute) error-bound compressor. Use `numcodecs-safeguards` to wrap *any* compressor (unbounded, best-effort bounded, or strictly bounded) to guarantee it is error bounded.


## Citation

Please cite this work as follows:

> Tyree, J. (2025). `numcodecs-safeguards` &ndash; Fearless lossy compression using safeguards. Available from: <https://github.com/juntyr/numcodecs-safeguards>

Please also refer to the [CITATION.cff](CITATION.cff) file and refer to <https://citation-file-format.github.io> to extract the citation in a format of your choice.


## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).


## Funding

The `compression-safeguards` and `numcodecs-safeguards` packages have been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
