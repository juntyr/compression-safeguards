---
title: Try Examples using JupyterLite
edit_uri: docs/try-notebook.md
render_macros: true
---

<h1>
    Try the <a id="try-notebook-name"></a> example using JupyterLite
</h1>

/// details | **Warning:** JupyterLite may not work in every web browser
    type: warning
<img src="https://baseline.js.org/features/wasm-multi-memory/responsive-adaptive.svg" alt="Baseline Status: Multi-memory (WebAssembly)" style="width: 100%; height: auto;" />
///

<iframe id="try-notebook-jupyterlite" width="100%" height="750px" referrerpolicy="no-referrer"></iframe>

<script>
  window.addEventListener("load", () => {
    const searchParams = new URL(window.location.href).searchParams;
    const notebook_ = searchParams.get("notebook");

    if (notebook_ === null) {
      return;
    }

    const notebook = new URL(notebook_, "{{ page.canonical_url }}");

    const [, user, repo] = new URL("{{ config.repo_url }}").pathname.split("/");
    const tag = "{{ git.tag }}";
    const name = notebook.pathname.split("/").pop();

    const backlink = new URL(notebook.href);
    backlink.pathname = backlink.pathname.split("/").toSpliced(-1).join("/");

    document.getElementById("try-notebook-name").innerText = name;
    document.getElementById("try-notebook-name").href = backlink;

    document.getElementById("try-notebook-jupyterlite").src = "https://lab.climet.eu/v0.4.0/notebooks/index.html?kernel=python&fromURL=" + notebook.href + "&pyodideKernelEnv=" + encodeURIComponent(JSON.stringify({"$override": {
      "CLIMET_LAB_BOOTSTRAP_CODE": `\
import shutil
from pathlib import Path
from urllib.request import urlopen

import micropip
import pyodide
import pyodide_fs_mount_http

if pyodide.ffi.can_run_sync():
    # we try our best :)
    pyodide.ffi.run_sync(micropip.install([
        "compression-safeguards=={{ version('compression_safeguards') }}",
        "numcodecs-safeguards=={{ version('numcodecs_safeguards') }}",
        "xarray-safeguards=={{ version('xarray_safeguards') }}",
    ] + {{ pyproject()['dependency-groups']['try-examples'] }}))

with urlopen(f"https://cors.climet.eu/https://raw.githubusercontent.com/${user}/${repo}/${tag}/examples/observe.py") as response:
    with open("observe.py", "wb") as file:
        shutil.copyfileobj(response, file)

for folder, files in {
    "cems-ercnfdr": ["data.grib"],
    "era5-lh": ["data.nc"],
    "era5-pr": ["data.nc"],
    "era5-q": ["data.nc"],
    "era5-uv": ["data.nc"],
    "hoaps-c.r30.h06.wvpa.2020-08": ["data.nc"],
    "isabel": ["Pf48.bin", "Uf48.bin"],
    "obs-pr": ["belem.csv", "helsinki.csv"],
    "output": [],
}.items():
    pyodide_fs_mount_http.mount_http_files(Path("data") / folder, {
        name: f"https://media.githubusercontent.com/media/${user}/${repo}/${tag}/examples/data/{folder}/{name}"
        for name in files
    })

for folder in ["observations", "plots", "tables"]:
    Path(folder).mkdir(parents=True, exist_ok=True)
`,
    }}));
  });
</script>
