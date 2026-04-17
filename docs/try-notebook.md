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
    
    // https://github.com/USER/REPO/blob/BRANCH/PATH ->
    // https://raw.githubusercontent.com/USER/REPO/refs/heads/BRANCH/PATH
    const rawUrl = new URL(url);
    rawUrl.hostname = "raw.githubusercontent.com";
    rawUrl.pathname = rawUrl.pathname.split("/").toSpliced(3, 1, "refs", "heads").join("/");

    const parts = rawUrl.pathname.split("/").slice(6);
    const name = parts[parts.length - 1];
    const backlink = ["..", ...parts.slice(0, -1), name.split(".")[0]].join("/");

    document.getElementById("try-notebook-name").innerText = name;
    document.getElementById("try-notebook-name").href = backlink;

    document.getElementById("try-notebook-jupyterlite").src = "https://lab.climet.eu/main/notebooks/index.html?kernel=python&fromURL=" + rawUrl.href + "&pyodideKernelEnv=" + encodeURIComponent(JSON.stringify({
      "EARTHKIT_DATA_CACHE_POLICY": "off",
      "EARTHKIT_REGRID_CACHE_POLICY": "off",
      "CLIMET_LAB_BOOTSTRAP_CODE": `\
import urllib.request
code = urllib.request.urlopen(
    "https://gist.githubusercontent.com/juntyr/f76ca0af41328439bcb40f758d418b7a/raw/5415a603bef6a006d7c889c9f3c226de2f5e04c5/mount-data.py"
).read()
print(code)
try:
    exec(code, globals=dict(), locals=dict())
except Exception as err:
    print(err)
`,
    }));
  });
</script>
