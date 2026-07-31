# Modify the API index page


def on_page_markdown(markdown, page, config, files):
    if page.url.startswith("_ref"):
        markdown = "".join(markdown.splitlines(keepends=True)[1:])

    # keep in sync with compression_safeguards/__init__.py
    if page.url == "_ref/compression_safeguards/":
        markdown = f"""\
{markdown}
    options:
        filters: {config["plugins"]["mkdocstrings"].handlers.get_handler_config("python")["options"]["filters"] + ["!Safeguards", "!SafeguardKind"]!r}

Classes:

- [**`Safeguards`**][compression_safeguards.api.Safeguards] – Collection of [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s.
- [**`SafeguardKind`**][compression_safeguards.safeguards.SafeguardKind] – Enumeration of all supported safeguards.
"""

    return markdown
