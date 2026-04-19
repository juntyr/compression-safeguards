---
title: Try Examples using JupyterLite
edit_uri: docs/try-notebook.md
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
    const url = searchParams.get("url");

    if (url === null) {
      return;
    }

    // https://github.com/USER/REPO/blob/BRANCH/...PATH/NAME ->
    // https://raw.githubusercontent.com/USER/REPO/refs/heads/BRANCH/...PATH/NAME
    const rawUrl = new URL(url);
    rawUrl.hostname = "raw.githubusercontent.com";
    rawUrl.pathname = rawUrl.pathname.split("/").toSpliced(3, 1, "refs", "heads").join("/");

    const [, user, repo, , , branch, ...path] = rawUrl.pathname.split("/");
    const name = path.pop();
    const backlink = ["..", ...path, name.split(".")[0]].join("/");

    document.getElementById("try-notebook-name").innerText = name;
    document.getElementById("try-notebook-name").href = backlink;

    document.getElementById("try-notebook-jupyterlite").src = "https://lab.climet.eu/main/notebooks/index.html?kernel=python&fromURL=" + rawUrl.href + "&pyodideKernelEnv=" + encodeURIComponent(JSON.stringify({
      "EARTHKIT_DATA_CACHE_POLICY": "off",
      "EARTHKIT_REGRID_CACHE_POLICY": "off",
      "CLIMET_LAB_BOOTSTRAP_CODE": `\
import shutil
from pathlib import Path
from urllib.request import urlopen

import pyodide_fs_mount_http

with urlopen(f"https://cors.climet.eu/https://raw.githubusercontent.com/${user}/${repo}/refs/heads/${branch}/examples/observe.py") as response:
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
        name: f"https://media.githubusercontent.com/media/${user}/${repo}/refs/heads/${branch}/examples/data/{folder}/{name}"
        for name in files
    })
`,
    }));
  });
</script>
