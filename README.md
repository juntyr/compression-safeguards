# Safe and Fearless lossy compression with `compression-safeguards`

Lossy[^1] compression can be *scary* as valuable information or features of the data may be lost.

By using safeguards to **guarantee** your safety requirements, lossy compression can be applied *safely* and *without fear*.

With the `compression-safeguards` package, you can:

- preserve properties over individual data elements (pointwise) or data neighbourhoods (stencil)
- preserve properties over quantities of interest (QoIs) over the data
- preserve regionally varying properties with regions of interest (RoIs)
- combine safeguards arbitrarily with logical combinators
- apply safeguards to any existing compressor or post-hoc to already-compressed data

[^1]: Lossy compression methods reduce data size by only storing an approximation of the data. In contrast to lossless compression methods, lossy compression loses information about the data, e.g. by reducing its resolution (only store every $n$th element) precision (only store $n$ digits after the decimal point), smoothing, etc. Therefore, lossy compression methods provide a tradeoff between size reduction and quality preservation.



## Table of Contents

* [What are safeguards?](#what-are-safeguards)
* [Design and Guarantees](#design-and-guarantees)
* [Provided safeguards](#provided-safeguards)
* [Installation](#installation)
* [Usage](#usage)
* [How to safeguard ...?](#how-to-safeguard)
* [Limitations](#limitations)
* [Related Projects](#related-projects)
* [Citation](#citation)
* [License](#license)
* [Funding](#funding)


## What are safeguards?

Safeguards are a declarative way to describe the safety requirements that you have for lossy compression. They range from simple (e.g. error bounds on the data, preserving special values and data signs) to complex (e.g. error bounds on derived quantities over data neighbourhoods, preserving monotonic sequences).

By declaring your safety requirements as safeguards, we can **guarantee** that any lossy compression protected by these safeguards will *always* uphold your safety requirements.

The `compression-safeguards` package provides several `Safeguard`s with which you can express *your* safety requirements. Please refer to the [provided safeguards](#provided-safeguards) section for a complete list of the supported safeguards.

We also provide the following integrations of the safeguards with popular compression APIs:

- `numcodecs-safeguards`: provides the `SafeguardsCodec` meta-compressor that conveniently applies safeguards to any compressor using the `numcodecs.abc.Codec` API.
- `xarray-safeguards`: provides functionality to use safeguards with (chunked) `xarray.DataArray`s.

The safeguards can be adopted easily:

- any existing (lossy) compressor can be safeguarded, e.g. with the `numcodecs-safeguards` frontend, allowing users to try out different (untrusted) compressors as the safeguards guarantee that the safety requirements are always upheld
- already compressed data can be safeguarded post-hoc as long as the original uncompressed data still exists, e.g. with the `xarray-safeguards` frontend
- the safeguards-corrections to the compressed data can be stored inline (alongside the lossy-compressed data, e.g. with the `numcodecs-safeguards` frontend) or outline (e.g. in a separate file, with the `xarray-safeguards` frontend)
- the safeguards can be combined with other meta-compression approaches, e.g. progressive data compression and retrieval[^2]

[^2]: See [doi:10.1109/TVCG.2023.3327186](https://doi.org/10.1109/TVCG.2023.3327186) for a general meta-compressor approach that enables progressive decompression to satisfy a compression error that is user-chosen at decompression time.


### Other terminology used by the compression safeguards

- *safeguard*: Declares a safety requirements and enforces that it is met after (lossy) compression.

- *pointwise safeguard*: A safety requirement that concerns just a single data element and can be checked and guaranteed independently for each data point.

- *stencil safeguard*: A safety requirement that is formulated over a neighbourhood of nearby points for each data element.

- *combinator safeguard*: A meta-safeguard that combines over several other safeguard with a logical combinator such as logical 'and' or 'or'.

- *parameter*: A configuration option for a safeguard that is provided when declaring the safeguard and cannot be changed

- *late-bound parameter*: A configuration option for a safeguard that is not constant but depends on the data being compressed. At declaration time, a late-bound parameter is only given a name but not a value. When the safeguards are later applied to data, all late-bound parameters must be resolved by providing their values. The `compression-safeguards`, `numcodecs-safeguards`, and `xarray-safeguards` frontends also provide a few built-in late-bound constants automatically, including `$x` to refer to the data as a constant. When configuring a `numcodecs_safeguards.SafeguardsCodec`, late-bound parameters are provided as *fixed constants* that must be compatible with any data that is encoded by the codec.

- *quantity of interest* (*QoI*): We are often not just interested in data itself, but also in quantities derived from it. For instance, we might later plot the data logarithm, compute a derivative, or apply a smoothing kernel. In these cases, we often want to safeguard not just properties on the data but also on these derived quantities of interest.

- *region of interest* (*RoI*): Sometimes we have regionally varying safety requirements, e.g. because a region has interesting behaviour that we want to especially preserve.


## Design and Guarantees

The safeguards are designed to be convenient to apply to any lossy compression task:

1. They are **guaranteed** to *always* uphold the safety property they describe.

2. They are designed to minimise the overhead in compressed message size for elements where the safety requirements were already satisfied.

They should ideally be applied to *every* lossy compression task since they have only a small overhead in the happy case (all safety requirements are already fulfilled) and give you peace of mind by reasserting the requirements if necessary (e.g. if the lossy compressor does not provide them or e.g. has an implementation bug).

Note that the packages in this repository are provided as reference implementations of the compression safeguards framework. Therefore, their implementations prioritise simplicity, portability, and readability over performance. Please refer to the [related projects](#related-projects) section for alternatives with different design considerations.


## Provided safeguards

This package currently implements the following safeguards:

### Error Bounds (pointwise)

- `eb` (error bound):

    The pointwise error is guaranteed to be less than or equal to the provided bound. Three types of error bounds can be enforced: `abs` (absolute), `rel` (relative), and `ratio` (ratio / decimal). For the relative and ratio error bounds, zero values are preserved with the same bit pattern. For the ratio error bound, the sign of the data is preserved. Infinite values are preserved with the same bit pattern. The safeguard can be configured such that NaN values are preserved with the same bit pattern, or that correcting a NaN value to a NaN value with a different bit pattern also satisfies the error bound.

### Error Bounds on derived Quantities of Interest (QoIs)

- `qoi_eb_pw` (error bound on quantities of interest):

    The error on a derived pointwise quantity of interest (QoI) is guaranteed to be less than or equal to the provided bound. Three types of error bounds can be enforced: `abs` (absolute), `rel` (relative), and `ratio` (ratio / decimal). The non-constant quantity of interest expression can contain the addition, multiplication, division, square root, exponentiation, logarithm, sign, rounding, trigonometric, and hyperbolic operations over integer and floating-point constants and the pointwise data value. For the ratio error bound, the sign of the quantity of interest is preserved. Infinite quantities of interest are preserved with the same bit pattern. NaN quantities of interest remain NaN though not necessarily with the same bit pattern.

- `qoi_eb_stencil` (error bound on quantities of interest over a neighbourhood):

    The error on a derived quantity of interest (QoI) over a neighbourhood of data points is guaranteed to be less than or equal to the provided bound. Three types of error bounds can be enforced: `abs` (absolute), `rel` (relative), and `ratio` (ratio / decimal). The non-constant quantity of interest expression can contain the addition, multiplication, division, square root, exponentiation, logarithm, sign, rounding, trigonometric, hyperbolic, array sum, matrix transpose, matrix multiplication, and finite difference operations over integer and floating-point constants and arrays and the data neighbourhood. If applied to data with more dimensions than the data neighbourhood of the QoI requires, the data neighbourhood is applied independently along these extra axes. If the data neighbourhood uses the `valid` boundary condition along an axis, only data neighbourhoods centred on data points that have sufficient points before and after are safeguarded. If the axis is smaller than required by the neighbourhood along this axis, the data is not safeguarded at all. Using a different boundary condition ensures that all data points are safeguarded. For the ratio error bound, the sign of the quantity of interest is preserved. Infinite quantities of interest are preserved with the same bit pattern. NaN quantities of interest remain NaN though not necessarily with the same bit pattern.

### Pointwise properties

- `same` (value preserving):

    If an element has a special value in the input, that element is guaranteed to also have bitwise the same value in the decompressed output. This safeguard can be used for preserving e.g. zero values, missing values, pre-computed extreme values, or any other value of importance. By default, elements that do *not* have the special value in the input may still have the value in the output. It is also possible to enforce that an element in the output only has the special value if and only if it also has the value in the input, e.g. to ensure that only missing values in the input have the missing value bitpattern in the output. Beware that +0.0 and -0.0 are semantically equivalent in floating-point but have different bitwise patterns. To preserve both, two same value safeguards are needed, one for each bitpattern.

- `sign` (sign-preserving):

    Values are guaranteed to have the same sign (-1, 0, +1) in the decompressed output as they have in the input data. NaN values are preserved as NaN values with the same sign bit. This safeguard can be configured to preserve the sign relative to a custom offset, e.g. to preserve global minima and maxima. This safeguard should be combined with e.g. an error bound, as it by itself accepts *any* value with the same sign.

### Relationships between neighboring elements

- `monotonicity` (monotonicity-preserving):

    Sequences that are monotonic in the input are guaranteed to be monotonic in the decompressed output. Monotonic sequences are detected using per-axis moving windows of constant size. Typically, the window size should be chosen to be large enough to ignore noise but small enough to capture details. Four levels of monotonicity can be enforced: `strict`, `strict_with_consts`, `strict_to_weak`, and `weak`. Windows that are not monotonic or contain non-finite data are skipped. If the `valid` boundary condition is used, axes that have fewer elements than
    the window size are skipped as well.

### Logical combinators (~pointwise)

- `all` (logical all / and):

    For each element, all of the combined safeguards' guarantees are upheld. At the moment, only pointwise and stencil safeguards and combinations thereof can be combined by this all-combinator.

- `any` (logical any / or):

    For each element, at least one of the combined safeguards' guarantees is upheld. At the moment, only pointwise and stencil safeguards and combinations thereof can be combined by this any-combinator.

- `assume_safe` (logical truth):

    All elements are assumed to always meet their guarantees and are thus always safe. This truth-combinator can be used with the `select` combinator to express regions that are *not* of interest, i.e. where no additional safety requirements are imposed.

- `select` (logical select / switch case):

    For each element, the guarantees of the pointwise selected safeguard are upheld. This combinator allows selecting between several safeguards with per-element granularity. It can be used to describe simple regions of interest where different safeguards, e.g. with different error bounds, are applied to different parts of the data. At the moment, only pointwise and stencil safeguards and combinations thereof can be combined by this select-combinator.


## Installation

The `compression-safeguards` package can be installed from PyPi using pip:

<!--pytest.mark.skip-->
```sh
pip install compression-safeguards
```

The integrations can be installed similarly:

<!--pytest.mark.skip-->
```sh
pip install numcodecs-safeguards
pip install xarray-safeguards
```


## Usage

### (a) Safeguards for users of lossy compression

We provide the lower-level `compression-safeguards` package and the user-facing `numcodecs-safeguards` and `xarray-safeguards` frontend packages, which can all be used to apply safeguards. We generally recommend using the safeguards through one of their integrations with popular (compression) APIs, e.g. `numcodecs-safeguards` for quickly getting started with a ready-made compressor for non-chunked arrays, or `xarray-safeguards` for adopting safeguards post-hoc and applying them to already compressed (chunked) data arrays.

#### `numcodecs-safeguards`

You can get started quickly with the `numcodecs`-compatible `SafeguardsCodec` meta-compressor for non-chunked arrays:

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

Please refer to the [`numcodecs-safeguards` documentation](https://juntyr.github.io/compression-safeguards/_ref/numcodecs_safeguards/) for further information.

#### `xarray-safeguards`

If you are working with large chunked datasets, want to post-hoc adopt safeguards for existing already-compressed data, or need extra control over how the safeguards-produced corrections are stored, you can use the `xarray-safeguards` frontend:

```py
import numpy as np
import xarray as xr
from xarray_safeguards import apply_data_array_correction, produce_data_array_correction

# some (chunked) n-dimensional data array
da = xr.DataArray(np.linspace(-10, 10, 21), name="da").chunk(10)
# lossy-compressed prediction for the data, here all zeros
da_prediction = xr.DataArray(np.zeros_like(da.values), name="da").chunk(10)

da_correction = produce_data_array_correction(
    data=da,
    prediction=da_prediction,
    # guarantee an absolute error bound of 0.1:
    #   |x - x'| <= 0.1
    safeguards=[dict(kind="eb", type="abs", eb=0.1)],
)

## (a) manual correction ##

da_corrected = apply_data_array_correction(da_prediction, da_correction)
np.testing.assert_allclose(da_corrected.values, da.values, rtol=0, atol=0.1)

## (b) automatic correction with xarray accessors ##

# combine the lossy prediction and the correction into one dataset
#  e.g. by loading them from different files using `xarray.open_mfdataset`
ds = xr.Dataset({
    da_prediction.name: da_prediction,
    da_correction.name: da_correction,
})

# access the safeguarded dataset that applies all corrections
ds_safeguarded: xr.Dataset = ds.safeguarded
np.testing.assert_allclose(ds_safeguarded["da"].values, da.values, rtol=0, atol=0.1)
```

Please also refer to the [`xarray-safeguards` documentation](https://juntyr.github.io/compression-safeguards/_ref/xarray_safeguards/) and the [`chunked.ipynb`](examples/chunked.ipynb) example for further information.

#### `compression-safeguards`

You can also use the lower-level `compression-safeguards` API directly:

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

Please refer to the [`compression-safeguards` documentation](https://juntyr.github.io/compression-safeguards/_ref/compression_safeguards/) for further examples.

### (b) Safeguards for developers of lossy compressors

The safeguards can also fill the role of a quantizer, which is part of many (predictive) (error-bounded) compressors. If you currently use e.g. a linear quantizer module in your compressor to provide an absolute error bound, you could instead adapt the `Safeguards`, quantize to their `Safeguards.compute_correction` values, and thereby offer a larger selection of safety requirements that your compressor can then guarantee. Note, however, that only pointwise safeguards can be used when quantizing data elements one-by-one.


## How to safeguard ...?

- ... a pointwise absolute / relative / ratio error bound on the data?

    > Use the `eb` safeguard and configure it with the `type` and range of the error bound.

- ... a pointwise normalised (NOA) or range-relative absolute error bound?

    > Use the `eb` safeguard for an absolute error bound but provide a late-bound parameter for the bound value. Since the data range is tightly tied to the data itself, it makes sense to only fill in the actual when applying the safeguards to the actual data. You can either compute the range yourself and then provide it as a `late_bound` binding when computing the safeguard corrections. Alternatively, you can also use the `qoi_eb_pw` safeguard with the `'(x - c["$x_min"]) / (c["$x_max"] - c["$x_min"])'` QoI. Note that we are using the late-bound constants `c["$x_min"]` and `c["$x_max"]` for the data minimum and maximum, which are automatically provided by `numcodecs-safeguards` and `xarray-safeguards`.

- ... a global error bound, e.g. a mean error, mean squared error, root mean square error, or peak signal to noise ratio?

    > The `compression-safeguards` do not currently support global safeguards. However, you can emulate a global error bound using a pointwise error bound, which provides a stricter guarantee. For all of the abovementioned global error bounds, use the `eb` safeguard with a pointwise absolute error bound of
    >
    > - $\epsilon_{abs} = |\epsilon_{ME}|$ for the mean error
    > - $\epsilon_{abs} = \sqrt{\epsilon_{MSE}}$ for the mean square error
    > - $\epsilon_{abs} = \epsilon_{RMSE}$ for the root mean square error
    > - $\epsilon_{abs} = (\text{max}(X) - \text{min}(X)) \cdot {10}^{-\text{PSNR} / 20}$ for the peak signal to noise ratio where $\text{PSNR}$ is given in dB

- ... a missing value?

    > If missing values are encoded as NaNs, the `eb` safeguards already guarantee that NaN values are preserved (if any NaN value works, be sure to enable the `equal_nan` flag). For other values, use the `same` safeguard and enable its `exclusive` flag.

- ... a global extrema (minimum / maximum)?

    > Use the `sign` safeguard with the `offset` corresponding to the extrema to ensure the extrema itself and its relationship to other values is preserved

- ... local extrema or other topological features?

    > Identifying local topological features, especially for noisy data, is a hard problem that you are likely using a custom algorithm for. To safeguard that algorithm's results, you should apply it to the original data before compression and identify how tolerant the algorithm is to errors around the extrema, giving you a regionally varying error bound that is tighter around the topological features, i.e. your regions of interest. Then, you can use the `eb` safeguard but provide a late-bound parameter for the bound value. At compression time, you then bind your regionally varying error tolerance to this parameter. If you also need to preserve whether values around the features are above/below a value, you can use a `sign` safeguard with a matching `offset` for each local feature and select over them using the `select` combinator and a late-bound `selector` mask that is based on your a priori analysis. If regions of interest overlap, you can combine several `select` combinators. For regions that are not of interest, the `select` combinator can fallback to the `assume_safe` safeguard, which imposes no additional safety requirements.

- ... a data distribution histogram?

    > The `compression-safeguards` do not currently support global safeguards. However, we can preserve the histogram bin that each data element falls into using the `qoi_eb_pw` safeguard, which provides a stricter guarantee. For instance, the `'round(100 * (x - c["$x_min"]) / (c["$x_max"] - c["$x_min"]))'` QoI would preserve the index amongst 100 bins. Note that we are using the late-bound constants `c["$x_min"]` and `c["$x_max"]` for the data minimum and maximum, which are automatically provided by `numcodecs-safeguards` and `xarray-safeguards`.


## Limitations

- *printer problem*: The `compression-safeguards` need to know about all safety requirements that they should uphold. If the data is first safeguarded with an absolute error bound, and then later the safeguards-corrected data is safeguarded with a relative error bound, the second safeguard may violate the guarantees provided by the first. Even applying the same safeguard twice in a row can violate the guarantees. This is also known as the printer problem: every time a document is copied (safeguarded) from a previously copied and printed (safeguarded) document, new artifacts are added and accumulate over time. Several safeguards should instead be combined into one using the (logical) combinator safeguards provided by the `compression-safeguards` package. Furthermore, the safeguards should always be given the original, uncompressed and unsafeguarded, reference data in relation to which the safety requirements must be upheld. The `numcodecs-safeguards` and `xarray-safeguards` frontends catch some trivial cases of the printer problem, e.g. wrapping a `SafeguardsCodec` inside a `SafeguardsCodec` or applying safeguards to an already safeguards-corrected `DataArray`. In the future, a community standard for marking lossy-compressed (and safeguarded) data with metadata could help with preventing accidental compression error accumulation.

- *biased corrections*: The `compression-safeguards` do not currently provide a safeguard to guarantee that the compression errors after safeguarding are unbiased. For instance, if a compressor, which produces biased decompressed values that are within the safeguarded error bound, is safeguarded, the biased values are not corrected by the safeguards. Furthermore, the safeguard corrections themselves may introduce bias in the compression error. Please refer to [`error-distribution.ipynb`](examples/error-distribution.ipynb) for some examples. We are working on a bias safeguard that would optionally provide these guarantees.

- *suboptimal one-shot corrections*: The `compression-safeguards` sometimes cannot provide optimal and easily compressible corrections. For instance, using a stencil safeguard that spans a local neighbourhood requires the safeguard to conservatively assume that the worst cases from each individual element could accumulate. Since the `compression-safeguards` compute the corrections for all elements simultaneously (instead of incrementally or by testing an initial correction that is repeatedly adjusted if it leads to a violation elsewhere), even a single violation can require conservative corrections for many data elements. In the future, the `compression-safeguards` API could support computing corrections incrementally such that stencil safeguards could make use of earlier[^3] already-corrected data elements and restrictions imposed by pointwise safeguards to provide better corrections for later elements. If you would like a peek at how safeguards could be applied incrementally, you can have a look at the [`incremental.ipynb`](examples/incremental.ipynb) example.

- *no global safeguards*: The `compression-safeguards` implementation do not currently support global safeguards, such as preserving mean errors or global data distributions. In many cases, it is possible to preserve these properties using stricter pointwise safeguards, at the cost of achieving lower compression ratios. Please refer to the [How to safeguard](#how-to-safeguard) section above for further details and examples.

- *only real data*: The `compression-safeguards` only support data of the following extended[^4] real data types: `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `uint64`, `int64`, `float16`, `float32`, `float64`. We appreciate contributions for supporting further, e.g. complex, data types.

- *single variable only*: The `compression-safeguards` do not support multi-variable safeguarding. If several variables should be safeguarded together, e.g. as inputs to a multi-variable quantity of interest, the variables can be stacked along a new dimension and then used as input for a stencil quantity of interest, e.g. as shown in the [`kinetic-energy.ipynb`](examples/kinetic-energy.ipynb) example. Note that compressing stacked variables with different data distributions might require prior normalisation.

- *no unstructured grids*: The `compression-safeguards` do not support unstructured grids for stencil safeguards (pointwise safeguards can be applied to any data). However, irregularly spaced grids are supported, e.g. by providing the coordinates as a late-bound parameter to a quantity of interest, e.g. for an arbitrary grid spacing to a finite difference. Please reach out if you want to collaborate on bringing support for unstructured grids to the `compression-safeguards`.

- *expensive*: The `compression-safeguards` require substantial computation at compression time, since safety requirements have to be checked, potentially re-established, and then checked again. Safeguarding a quantity of interest can be particularly expensive since rounding errors have to be checked for at every step. However, these extensive checks allow the safeguards to provide hard safety guarantees that users can rely on.

[^3]: See e.g. Figure 4 in [doi:10.14778/3574245.3574255](https://doi.org/10.14778/3574245.3574255) for inspiration on how incremental safeguard corrections might work.

[^4]: The extended real values `-inf` and `+inf` and `NaN` (not a number) are supported for floating-point input data types.

## Related Projects

### SZ3 error compression

[SZ3](https://github.com/szcompressor/SZ3) >=3.2.0 provides the `CmprAlgo=ALGO_NOPRED` option, with which the compression error $error = decompressed - data$ of another lossy compressor can itself be lossy-compressed with e.g. an absolute error bound. Using this option, any compressor can be transformed into an error bounded compressor.

SZ3's error compression can provide higher compression ratios if most data elements are expected to violate the error bound, e.g. when wrapping a lossy compressor that does *not* bound its errors. However, SZ3 has a higher byte overhead than `numcodecs-safeguards` if all elements already satisfy the bound.

**TLDR:** You can use SZ3 to transform a *known* *unbounded* lossy compressor into an (absolute) error-bound compressor. Use `compression-safeguards` to guarantee a variety of safety requirements for *any* compressor (unbounded, best-effort bounded, or strictly bounded).


## Citation

Please cite this work as follows:

> Tyree, J. (2025). `compression-safeguards` &ndash; Safe and Fearless lossy compression using safeguards. Available from: <https://github.com/juntyr/compression-safeguards>

Please also refer to the [CITATION.cff](CITATION.cff) file and refer to <https://citation-file-format.github.io> to extract the citation in a format of your choice.


## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).


## Funding

The `compression-safeguards`, `numcodecs-safeguards`, and `xarray-safeguards` packages have been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
