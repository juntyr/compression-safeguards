---
edit_uri: docs/try-repl.md
render_macros: true
---

# Try the `compression-safeguards` using JupyterLite

/// details | **Warning:** JupyterLite may not work in every web browser
    type: warning
<img src="https://baseline.js.org/features/wasm-multi-memory/responsive-adaptive.svg" alt="Baseline Status: Multi-memory (WebAssembly)" style="width: 100%; height: auto;" />
///

<iframe id="try-repl-jupyterlite" width="100%" height="750px" referrerpolicy="no-referrer"></iframe>

<script>
  window.addEventListener("load", () => {
    document.getElementById("try-repl-jupyterlite").src = "https://lab.climet.eu/main/repl/index.html?kernel=python&toolbar=1&code=" + encodeURIComponent(`\
# install the compression safeguards
%pip install compression-safeguards=={{ version('compression_safeguards') }}
%pip install numcodecs-safeguards=={{ version('numcodecs_safeguards') }}
%pip install xarray-safeguards=={{ version('xarray_safeguards') }}\
`) + "&code=" + encodeURIComponent(`\
import numpy as np
from matplotlib import pyplot as plt
from numcodecs import Quantize
from numcodecs_combinators.framed import FramedCodecStack
from numcodecs_safeguards import SafeguardedCodec\
`) + "&code=" + encodeURIComponent(`\
# input data
x = np.linspace(-np.pi, np.pi)\
`) + "&code=" + encodeURIComponent(`\
# lossy compression, here linear quantization
codec = Quantize(digits=0, dtype=np.float64)
enc = codec.encode(x)
dec = codec.decode(enc)\
`) + "&code=" + encodeURIComponent(`\
# safeguard an absolute error over a quantity of interest,
#  here sin(x),
# with an absolute error bound s.t.
#  abs( sin(x') - sin(x) ) <= 0.1
sg = SafeguardedCodec(
    codec=FramedCodecStack(codec),
    safeguards=[
        dict(kind="qoi_eb_pw", qoi="sin(x)", type="abs", eb=0.1),
    ]
)
enc_sg = sg.encode(x)
dec_sg = sg.decode(enc_sg)\
`) + "&code=" + encodeURIComponent(`\
# visually compare the lossy compressed and safeguarded data
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(x, np.sin(x))
ax1.plot(x, np.sin(dec))
ax1.plot(x, np.sin(dec_sg))
ax1.legend([], title=r"$\\sin$(...)", ncols=3, loc="upper left")
ax2.axhline(0.1, c="k", ls=":")
ax2.axhline(-0.1, c="k", ls=":")
ax2.plot(x, np.sin(x)-np.sin(x), label="original")
ax2.plot(x, np.sin(dec)-np.sin(x), label="compressed")
ax2.plot(x, np.sin(dec_sg)-np.sin(x), label="safeguarded")
plt.legend(
    title=r"$\\sin(\\hat{x}) - \\sin(x)$",
    ncols=3, loc="lower center",
)
plt.show()\
`) + "&pyodideKernelPackages=" + encodeURIComponent(JSON.stringify({
  "$concat": [
    // example packages
    "matplotlib",
    "numcodecs",
    "numcodecs-bitmap-index",
    "numcodecs-combinators",
    "numcodecs-delta",
    "numcodecs-shuffle",
    "numcodecs-tokenize",
    "numcodecs-zero",
    "numpy",
    "numpy-quaddtype",
    "semver",
    "sly",
    "typing-extensions",
    // example package lazy dependencies
    "crc32c",
    "msgpack",
  ]
}));
  });
</script>
