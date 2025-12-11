# API Reference Overview

- [`compression-safeguards`][compression_safeguards]: provides the core [`Safeguards`][compression_safeguards.api.Safeguards] API as well as several [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s with which you can express *your* safety requirements.

- [`numcodecs-safeguards`][numcodecs_safeguards]: provides the [`SafeguardsCodec`][numcodecs_safeguards.SafeguardsCodec] meta-compressor that conveniently applies safeguards to any compressor using the [`numcodecs.abc.Codec`][numcodecs.abc.Codec] API.

- [`xarray-safeguards`][xarray_safeguards]: provides functionality to use safeguards with (chunked) [`xarray.DataArray`][xarray.DataArray]s and cross-chunk boundary conditions.
