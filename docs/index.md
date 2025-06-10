# Fearless lossy compression with `compression-safeguards`

Lossy compression [^1] can be *scary* as valuable information or features of the data may be lost.

By using safeguards to **guarantee** your safety requirements, lossy compression can be applied safely and *without fear*.

With the `compression-safeguards` package, you can:

- preserve properties over individual data elements (pointwise) or data neighbourhoods (stencil)
- preserve properties over quantities of interest (QoIs) over the data
- preserve regionally varying properties with regions of interest (RoIs)
- combine safeguards arbitrarily with logical combinators

[^1]: Lossy compression methods reduce data size by only storing an approximation of the data. In contrast to lossless compression methods, lossy compression loses information about the data, e.g. by reducing its resolution (only store every $n$th element) precision (only store $n$ digits after the decimal point), smoothing, etc. Therefore, lossy compression methods provide a tradeoff between size reduction and quality preservation.


## What are safeguards?

Safeguards are a declarative way to describe the safety requirements that you have for lossy compression. They range from simple (e.g. error bounds on the data, preserving special values and data signs) to complex (e.g. error bounds on derived quantities over data neighbourhoods, preserving monotonic sequences).

By declaring your safety requirements as safeguards, we can **guarantee** that any lossy compression protected by these safeguards will *always* uphold your safety requirements.

The [`compression-safeguards`][compression_safeguards] package provides several [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s with which you can express *your* safety requirements. Please refer to the [provided safeguards](#provided-safeguards) section or the [`SafeguardKind`][compression_safeguards.SafeguardKind] documentation for a complete list of the supported safeguards.

We also provide the following integrations of the safeguards with popular compression APIs:

- [`numcodecs-safeguards`][numcodecs_safeguards]: provides the [`SafeguardsCodec`][numcodecs_safeguards.SafeguardsCodec] meta-compressor that conveniently applies safeguards to any compressor using the [`numcodecs.abc.Codec`][numcodecs.abc.Codec] API.


## Design and Guarantees

The safeguards are designed to be convenient to apply to any lossy compression task:

1. They are **guaranteed** to *always* uphold the safety property they describe.

2. They are designed to minimise the overhead in compressed message size for elements where the safety requirements were already satisfied.

They should ideally be applied to *every* lossy compression task since they have only a small overhead in the happy case (all safety requirements are already fulfilled) and give you peace of mind by reasserting the requirements if necessary (e.g. if the lossy compressor does not provide them or e.g. has an implementation bug).

Note that the packages in this repository are provided as reference implementations of the compression safeguards framework. Therefore, their implementations prioritise simplicity, portability, and readability over performance. Please refer to the [related projects](#related-projects) section for alternatives with different design considerations.


## Provided safeguards

This package currently implements the following [safeguards][compression_safeguards.SafeguardKind]:

### Error Bounds (pointwise)

- [`eb`][compression_safeguards.safeguards.pointwise.eb.ErrorBoundSafeguard] (error bound):

    The pointwise error is guaranteed to be less than or equal to the provided bound. Three types of [error bounds][compression_safeguards.safeguards.pointwise.eb.ErrorBound] can be enforced: `abs` (absolute), `rel` (relative), and `ratio` (ratio / decimal). For the relative and ratio error bounds, zero values are preserved with the same bit pattern. For the ratio error bound, the sign of the data is preserved. Infinite values are preserved with the same bit pattern. The safeguard can be configured such that NaN values are preserved with the same bit pattern, or that decoding a NaN value to a NaN value with a different bit pattern also satisfies the error bound.

### Error Bounds on derived Quantities of Interest (QoIs)

- [`qoi_abs_pw`][compression_safeguards.safeguards.pointwise.qoi.abs.PointwiseQuantityOfInterestAbsoluteErrorBoundSafeguard] (absolute error bound on pointwise quantities of interest):

    The absolute error on a derived pointwise quantity of interest (QoI) is guaranteed to be less than or equal to the provided bound. The non-constant quantity of interest expression can contain the addition, multiplication, division, square root, exponentiation, logarithm, trigonometric, and hyperbolic operations over integer and floating point constants and the pointwise data value. Infinite quantities of interest are preserved with the same bit pattern. NaN quantities of interest remain NaN though not necessarily with the same bit pattern.

- [`qoi_abs_stencil`][compression_safeguards.safeguards.stencil.qoi.abs.StencilQuantityOfInterestAbsoluteErrorBoundSafeguard] (absolute error bound on quantities of interest over a neighbourhood):

    The absolute error on a derived quantity of interest (QoI) over a neighbourhood of data points is guaranteed to be less than or equal to the provided bound. The non-constant quantity of interest expression can contain the addition, multiplication, division, square root, exponentiation, logarithm, trigonometric, hyperbolic, array sum, matrix transpose, matrix multiplication, and finite difference operations over integer and floating point constants and arrays and the data neighbourhood. If applied to data with more dimensions than the data neighbourhood of the QoI requires, the data neighbourhood is applied independently along these extra axes. If the data neighbourhood uses the [valid][compression_safeguards.safeguards.stencil.BoundaryCondition.valid] boundary condition along an axis, only data neighbourhoods centred on data points that have sufficient points before and after are safeguarded. If the axis is smaller than required by the neighbourhood along this axis, the data is not safeguarded at all. Using a different [`BoundaryCondition`][compression_safeguards.safeguards.stencil.BoundaryCondition] ensures that all data points are safeguarded. Infinite quantities of interest are preserved with the same bit pattern. NaN quantities of interest remain NaN though not necessarily with the same bit pattern.

### Pointwise properties

- [`zero`][compression_safeguards.safeguards.pointwise.zero.ZeroIsZeroSafeguard] (zero/constant preserving):

    Values that are zero in the input are guaranteed to also be *exactly* zero in the decompressed output. This safeguard can also be used to enforce that another constant value is bitwise preserved, e.g. a missing value constant or a semantic "zero" value that is represented as a non-zero number. Beware that +0.0 and -0.0 are semantically equivalent in floating point but have different bitwise patterns. If you want to preserve both, you need to use two safeguards, one configured for each zero.

- [`sign`][compression_safeguards.safeguards.pointwise.sign.SignPreservingSafeguard] (sign-preserving):

    Values are guaranteed to have the same sign (-1, 0, +1) in the decompressed output as they have in the input data. The sign for NaNs is derived from their sign bit, e.g. sign(-NaN) = -1. Sign-preservation should be combined with e.g. an error bound, as it by itself accepts *any* value with the same sign.

### Relationships between neighboring elements

- [`monotonicity`][compression_safeguards.safeguards.stencil.monotonicity.MonotonicityPreservingSafeguard] (monotonicity-preserving):

    Sequences that are monotonic in the input are guaranteed to be monotonic in the decompressed output. Monotonic sequences are detected using per-axis moving windows of constant size. Typically, the window size should be chosen to be large enough to ignore noise but small enough to capture details. Four levels of [monotonicity][compression_safeguards.safeguards.stencil.monotonicity.Monotonicity] can be enforced: `strict`, `strict_with_consts`, `strict_to_weak`, and `weak`. Windows that are not monotonic or contain non-finite data are skipped. If the [valid][compression_safeguards.safeguards.stencil.BoundaryCondition.valid] boundary condition is used, axes that have fewer elements than the window size are skipped as well.

### Logical combinators (~pointwise)

- [`all`][compression_safeguards.safeguards.combinators.all.AllSafeguards] (logical all / and):

    For each element, all of the combined safeguards' guarantees are upheld. At the moment, only pointwise and stencil safeguards and combinations thereof can be combined by this all-combinator.

- [`any`][compression_safeguards.safeguards.combinators.any.AnySafeguard] (logical any / or):

    For each element, at least one of the combined safeguards' guarantees is upheld. At the moment, only pointwise and stencil safeguards and combinations thereof can be combined by this any-combinator.

- [`safe`][compression_safeguards.safeguards.combinators.safe.AlwaysSafeguard] (logical truth):

    All elements always meet their guarantees and are thus always safe. This truth-combinator can be used, with care, with other logical combinators.

- [`select`][compression_safeguards.safeguards.combinators.select.SelectSafeguard] (logical select / switch case):

    For each element, the guarantees of the pointwise selected safeguard are upheld. This combinator allows selecting between several safeguards with per-element granularity. It can be used to describe simple regions of interest where different safeguards, e.g. with different error bounds, are applied to different parts of the data. At the moment, only pointwise and stencil safeguards and combinations thereof can be combined by this select-combinator.


## Installation

The `compression-safeguards` package can be installed from PyPi using pip:

<!--pytest.mark.skip-->
```sh
pip install compression-safeguards
```

You may also need to install the `numpy-quaddtype` dependency from the PyPi test index:

<!--
```sh
pip install numpy
```
-->
<!--pytest-codeblocks:cont-->
```sh
pip install -i https://test.pypi.org/simple/ numpy-quaddtype~=0.0.7
```

The integrations can be installed similarly:

<!--pytest.mark.skip-->
```sh
pip install numcodecs-safeguards
```


## Usage

### (a) Safeguards for users of lossy compression

We recommend using the safeguards through one of their integrations with popular compression APIs, e.g. [`numcodecs-safeguards`][numcodecs_safeguards]:

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

Please refer to the [`numcodecs_safeguards`][numcodecs_safeguards] documentation for further information.

You can also use the lower-level [`compression_safeguards`][compression_safeguards] API directly:

<!--
```py
import numpy as np
def compress(data):
    return data
def decompress(data):
    return np.zeros_like(data)
```
-->
<!--pytest-codeblocks:cont-->
```py
import numpy as np
from compression_safeguards import Safeguards

# create the `Safeguards`
sg = Safeguards(safeguards=[
    # guarantee an absolute error bound of 0.1:
    #   |x - x'| <= 0.1
    dict(kind="eb", type="abs", eb=0.1),
])

# generate some random data to compress
data = np.random.normal(size=(10, 10, 10))

## compression

# compress and decompress the data using *some* compressor
compressed = compress(data)
decompressed = decompress(compressed)

# compute the correction that the safeguards would need to apply to
# guarantee the selected safety requirements
correction = sg.compute_correction(data, decompressed)

# now the compressed data and correction can be stored somewhere
# ...
# and loaded again to decompress

## decompression
decompressed = decompress(compressed)
decompressed = sg.apply_correction(decompressed, correction)

# the safeguard properties are now guaranteed to hold
assert np.all(np.abs(data - decompressed) <= 0.1)
```

Please refer to the [`compression_safeguards`][compression_safeguards] documentation for further examples.

### (b) Safeguards for developers of lossy compressors

The safeguards can also fill the role of a quantizer, which is part of many (predictive) (error-bounded) compressors. If you currently use e.g. a linear quantizer module in your compressor to provide an absolute error bound, you could instead adapt the [`Safeguards`][compression_safeguards.Safeguards], quantize to their [`Safeguards.compute_correction`][compression_safeguards.Safeguards.compute_correction] values, and thereby offer a larger selection of safety requirements that your compressor can then guarantee. Note, however, that only pointwise safeguards can be used when quantizing data elements one-by-one.


## Related Projects

### SZ3 error compression

[SZ3](https://github.com/szcompressor/SZ3) >=3.2.0 provides the `CmprAlgo=ALGO_NOPRED` option, with which the compression error $error = decompressed - data$ of another lossy compressor can itself be lossy-compressed with e.g. an absolute error bound. Using this option, any compressor can be transformed into an error bounded compressor.

SZ3's error compression can provide higher compression ratios if most data elements are expected to violate the error bound, e.g. when wrapping a lossy compressor that does *not* bound its errors. However, SZ3 has a higher byte overhead than `numcodecs-safeguards` if all elements already satisfy the bound.

**TLDR:** You can use SZ3 to transform a *known* *unbounded* lossy compressor into an (absolute) error-bound compressor. Use `compression-safeguards` to guarantee a variety of safety requirements for *any* compressor (unbounded, best-effort bounded, or strictly bounded).


## Citation

Please cite this work as follows:

> Tyree, J. (2025). `compression-safeguards` &ndash; Fearless lossy compression using safeguards. Available from: <https://github.com/juntyr/compression-safeguards>

Please also refer to the [CITATION.cff](https://github.com/juntyr/compression-safeguards/blob/main/CITATION.cff) file and refer to <https://citation-file-format.github.io> to extract the citation in a format of your choice.


## Funding

The `compression-safeguards` and `numcodecs-safeguards` packages have been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
