# Adapted from: https://github.com/renovatebot/renovatebot.github.io/pull/187

# Allows overriding the edit_url from the yaml frontmatter

def on_page_context(context, page, config, **kwargs):
    if "edit_uri" in page.meta:
        page.edit_url = config["repo_url"] + config["edit_uri"] + page.meta["edit_uri"]
    return context
