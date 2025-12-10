"""
# Fearless lossy compression with `compression-safeguards`

Lossy compression can be *scary* as valuable information or features of the
data may be lost.

By using safeguards to **guarantee** your safety requirements, lossy
compression can be applied safely and *without fear*.

## Overview

This package provides several
[`Safeguard`][.safeguards.abc.Safeguard]s (refer to the
[`SafeguardKind`][.safeguards.SafeguardKind] for an enumeration) with which you
can express *your* requirements for lossy compression to be safe to use.

The safeguards are then combined in the
[`Safeguards`][.api.Safeguards], which can be used to compute and apply the
required correction to lossy-compressed data so that it satisfies your safety
guarantees.

This package provides the implementations of the safeguards and the low-level
`Safeguards` API. Please also refer to the following integrations of the
safeguards with popular compression APIs:

- [`numcodecs-safeguards`][numcodecs_safeguards]: provides the
  [`SafeguardsCodec`][numcodecs_safeguards.SafeguardsCodec] meta-compressor that
  conveniently applies safeguards to any compressor using the
  [`numcodecs.abc.Codec`][numcodecs.abc.Codec] API.
- [`xarray-safeguards`][xarray_safeguards]: provides functionality to use
  safeguards with (chunked) [`xarray.DataArray`][xarray.DataArray]s.

## Examples

In the following examples, we assume that there is *some* compressor that can
`compress` and `decompress` n-dimensional `data` using functions of that name.

### Basic Usage

You can guarantee an absolute error bound of $eb_{abs} = 0.1$  as follows:

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

### Instantiating the safeguards

The safeguards can be instantiated from JSON-like configuration:

```py
from compression_safeguards import Safeguards

sg = Safeguards(safeguards=[
    dict(kind="eb", type="abs", eb=0.1),
])
```

or by using the [`SafeguardKind`][.safeguards.SafeguardKind]:

```py
from compression_safeguards import Safeguards, SafeguardKind

sg = Safeguards(safeguards=[
    SafeguardKind.eb.value(type="abs", eb=0.1),
])
```

These two methods can be freely combined.

The entire safeguards can also be turned into a JSON configuration and
recreated from such configuration:

```py
from compression_safeguards import Safeguards

sg = Safeguards(safeguards=[
    dict(kind="eb", type="abs", eb=0.1),
])
config = sg.get_config()

sg = Safeguards.from_config(config)
assert sg.get_config() == config
```

### Combining several safeguards

All of the provided safeguards can be freely combined and are guaranteed to
work together.

Providing several safeguards means that *all* the specified
safety requirements must be upheld:

```py
from compression_safeguards import Safeguards

sg = Safeguards(safeguards=[
    # guarantee an absolute error bound
    dict(kind="eb", type="abs", eb=0.1),
    # and that the data sign is preserved
    dict(kind="sign"),
])
```

This package also provides several combinators that can be used to express
pointwise logical combinations of safeguards:

```py
from compression_safeguards import Safeguards

sg = Safeguards(safeguards=[
    # guarantee that, for each element, *both* an absolute error bound of 0.1
    # *and* a relative error bound of 1% are upheld
    dict(kind="all", safeguards=[
        dict(kind="eb", type="abs", eb=0.1),
        dict(kind="eb", type="rel", eb=0.01),
    ]),
])

sg = Safeguards(safeguards=[
    # guarantee that, for each element, an absolute error bound of 0.1
    # *or* a relative error bound of 1% are upheld
    dict(kind="any", safeguards=[
        dict(kind="eb", type="abs", eb=0.1),
        dict(kind="eb", type="rel", eb=0.01),
    ]),
])
```

### Regionally varying safeguards using late-bound parameters

By default, all safeguards apply the same safety guarantees across the entire
data domain. This package supports two approaches for regionally varying the
guarantees, i.e. applying different guarantees to different data regions.

First, the `select` combinator can be used to switch between two or more
safeguards (or safeguard combinations) using a selection indices array. Unlike
normal safeguard parameters, this selector is a late-bound parameter whose
value is not specified during safeguard initialisation but only later when the
safeguard is applied:

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
from compression_safeguards import Safeguards
from compression_safeguards.utils.bindings import Bindings

sg = Safeguards(safeguards=[
    # select between a coarser, medium, and finer absolute error bound
    #  safeguard based on the late-bound "mask" parameter
    dict(kind="select", selector="mask", safeguards=[
        dict(kind="eb", type="abs", eb=1.0),
        dict(kind="eb", type="abs", eb=0.1),
        dict(kind="eb", type="abs", eb=0.01),
    ]),
])

# generate some random data to compress
data = np.random.normal(size=(10, 10, 10))

## compression (now with late-bound parameters)

# compress and decompress the data using *some* compressor
compressed = compress(data)
decompressed = decompress(compressed)

# compute the correction that the safeguards would need to apply to
# guarantee the selected safety requirements
correction = sg.compute_correction(data, decompressed, late_bound=Bindings(
    # bind the selection mask that selects between the three safeguards
    # the mask must be broadcastable to the data shape
    mask=np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2]).reshape((1, 10, 1)),
))

## decompression (as before)
decompressed = decompress(compressed)
decompressed = sg.apply_correction(decompressed, correction)
```

It is worth noting that the late-bound parameters are only needed at
compression time, decompression is unchanged.

While this first method can combine over any safeguards, it is only convenient
for selecting between a small number of different safeguards.

For the error bound safeguards (pointwise and on quantities of interest), the
error bounds themselves can be provided as late-bound parameters to allow for
smoothly varying error bounds across the data domain. The above example could
then be equivalently expressed as:

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
from compression_safeguards import Safeguards
from compression_safeguards.utils.bindings import Bindings

sg = Safeguards(safeguards=[
    # absolute error bound with a late-bound "eb" parameter
    dict(kind="eb", type="abs", eb="eb"),
])

# generate some random data to compress
data = np.random.normal(size=(10, 10, 10))

## compression (now with late-bound parameters)

# compress and decompress the data using *some* compressor
compressed = compress(data)
decompressed = decompress(compressed)

# compute the correction that the safeguards would need to apply to
# guarantee the selected safety requirements
correction = sg.compute_correction(data, decompressed, late_bound=Bindings(
    # bind the late-bound absolute error bound
    # the bound must be broadcastable to the data shape
    eb=np.array([
        1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01,
    ]).reshape((1, 10, 1)),
))

## decompression (as before)
decompressed = decompress(compressed)
decompressed = sg.apply_correction(decompressed, correction)
```
"""

__all__ = ["Safeguards", "SafeguardKind"]

from .api import Safeguards
from .safeguards import SafeguardKind
