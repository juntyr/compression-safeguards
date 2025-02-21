# Fearless lossy compression with `numcodecs-safeguards`

Lossy compression can be scary as valuable information may be lost. This package provides the `SafeguardsCodec` adapter and several safeguards that can be applied to *any* existing (lossy) compressor to *guarantee* that certain properties about the compression error are upheld. Note that the wrapped compressor is treated as a blackbox and the decompressed data is postprocessed to re-establish the properties, if necessary. By using the `SafeguardsCodec` adapter, badly behaving lossy compressors become safe to use, at the cost of potentially less efficient compression, and lossy compression can be applied without fear.


## Design and Guarantees

The safeguards implemented in this package are designed to be convenient to apply to any lossy compression task:

1. They are *guaranteed* to *always* uphold the property they provide.

2. They are designed to minimise the overhead in compressed message size for elements where the properties were already satisfied by the wrapped compressor.

3. Their implementations are prioritise correctness in all cases and simplicity of verification over performance (and byte ratio overhead when the properties are violated by many data elements). Please refer to the [related projects](#related-projects) section for alternatives with different design considerations.

If applied to

- a safe compressor that (by chance or design) already satisfies the properties, there is a constant single-byte overhead

- an unsafe compressor that violates any of the properties, the properties are restored at a byte overhead that scales linearly with the number of elements that have to be corrected


## Provided safeguards

This package currently implements the following safeguards

- `zero` (zero/constant preserving):

    Values that are zero in the input are guaranteed to also be *exactly* zero in the decompressed output. This safeguard can also be used to enforce that another constant value is bitwise preserved, e.g. a missing value constant or a semantic "zero" value that is represented as a non-zero number.

- `abs` (absolute error bound):

    The elementwise absolute error is guaranteed to be less than or equal to the provided bound. In cases where the arithmetic evaluation of the error bound is not well-defined, e.g. for infinite or NaN values, producing the exact same bitpattern is defined to satisfy the error bound. The safeguard can also be configured such that decoding a NaN value to a NaN value with a different bitpattern also satisfies the error bound.

- `rel_or_abs` (relative [or absolute] error bound):

    The elementwise absolute error between the *logarithms*\* of the values is guaranteed to be less than or equal to $\log(1 + eb_{rel})$ where $eb_{rel}$ is e.g. 2%. The logarithm* here is adapted to support positive, negative, and zero values. For values close to zero, where the relative error is not well-defined, the absolute elementwise error is guaranteed to be less than or equal to the absolute error bound.

    Put simply, each element satisfies the relative or the absolute error bound (or both). In cases where the arithmetic evaluation of the error bound is not well-defined, e.g. for infinite or NaN values, producing the exact same bitpattern is defined to satisfy the error bound. The safeguard can also be configured such that decoding a NaN value to a NaN value with a different bitpattern also satisfies the error bound.

- `decimal` (decimal error bound):

    The elementwise decimal error is guaranteed to be less than or equal to the provided bound. The decimal error quantifies the orders of magnitude that the lossy-decoded value is away from the original value, i.e. the difference in their decimal logarithms. It is defined to be infinite if the signs of the data and decoded data do not match. Since the decimal error bound must be finite, this safeguard also guarantees that the sign of each decode value matches the sign of each original value and that a decoded value is zero if and only if it is zero in the original data. In cases where the arithmetic evaluation of the error bound not well-defined, e.g. for infinite or NaN values, producing the exact same bitpattern is defined to satisfy the error bound. The safeguard can also be configured such that decoding a NaN value to a NaN value with a different bitpattern also satisfies the error bound.

- `findiff_abs` (absolute error bound for finite differences):

    The elementwise absolute error of the finite-difference-approximated derivative is guaranteed to be less than or equal to the provided bound. The safeguard supports three types of finite difference: `central`, `forward`, `backward`. The fininite difference is computed with respect to the provided uniform grid spacing. If the spacing is different along different axes, multiple safeguards along specific axes with different spacing can be combined. If the finite difference for an element evaluates to an infinite value, this safeguard guarantees that the finite difference on the decoded value produces the exact same infinite value. For a NaN finite difference, this safeguard guarantees that the finite difference on the decoded value is also NaN, but does not guarantee that it has the same bitpattern.

- `monotonicity` (monotonicity-preserving):

    Sequences that are monotonic in the input are guaranteed to be monotonic in the decompressed output. Monotonic sequences are detected using per-axis moving windows of constant size. Typically, the window size should be chosen to be large enough to ignore noise but small enough to capture details. The safeguard supports enforcing four levels of monotonicity: `strict`, `strict_with_consts`, `strict_to_weak`, `weak`. Windows that are not monotonic or contain non-finite data are skipped. Axes that have fewer elements than the window size are skipped as well.

- `sign` (sign-preserving):

    Values are guaranteed to have the same sign (-1, 0, +1) in the decompressed output as they have in the input data. The sign for NaNs is derived from their sign bit, e.g. sign(-NaN) = -1.


## Usage

The `SafeguardsCodec` adapter provided by this package can wrap any existing [`Codec`] [^1] implementing the [`numcodecs`] API [^2] that encodes a buffer (e.g. an ndarray or bytes) to bytes. It is desirable to perform lossless compression after applying the safeguards (rather than before).

You can wrap an existing codec with e.g. an absolute error bound of $eb_{abs} = 0.1$  as follows:

```python
from numcodecs_safeguards import Safeguards, SafeguardsCodec

SafeguardsCodec(codec=codec, safeguards=[Safeguards.abs.value(eb_abs=0.1)])
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

[^1]: If you want to wrap a sequence or stack of codecs, you can use the [`CodecStack`] combinator from the [`numcodecs-combinators`] package.
[^2]: The method implemented in this package is not specific to the [`numcodecs`] API. Please reach out if you'd like to help bring the safeguards to a different compression API or language.

[`Codec`]: https://numcodecs.readthedocs.io/en/stable/abc.html#numcodecs.abc.Codec
[`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/
[`CodecStack`]: https://numcodecs-combinators.readthedocs.io/en/stable/_ref/numcodecs_combinators/stack/#numcodecs_combinators.stack.CodecStack
[`numcodecs-combinators`]: https://numcodecs-combinators.readthedocs.io/en/stable/


## Related Projects

### SZ3 error compression

[SZ3](https://github.com/szcompressor/SZ3) >=3.2.0 provides the `CmprAlgo=ALGO_NOPRED` option, with which the compression error $error = decompressed - data$ of another lossy compressor can itself be lossy-compressed with e.g. an absolute error bound. Using this option, any compressor can be transformed into an error bounded compressor.

SZ3's error compression can provide higher compression ratios if most data elements are expected to violate the error bound, e.g. when wrapping a lossy compressor that does *not* bound its errors. However, SZ3 has a higher byte overhead than `numcodecs-safeguards` if all elements already satisfy the bound.

**TLDR:** You can SZ3 to transform a *known* *unbounded* lossy compressor into an (absolute) error-bound compressor. Use `numcodecs-safeguards` to wrap *any* compressor (unbounded, best-effort bounded, or strictly bounded) to guarantee it is error bounded.


## Citation

Please cite this work as follows:

> Tyree, J. (2025). `numcodecs-safeguards` &ndash; Fearless lossy compression using safeguards. Available from: <https://github.com/juntyr/numcodecs-safeguards>

Please also refer to the [CITATION.cff](CITATION.cff) file and refer to <https://citation-file-format.github.io> to extract the citation in a format of your choice.


## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).


## Funding

The `numcodecs-safeguards` package has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
