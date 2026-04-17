# Collapse index sections such that the section-index plugin can identify them


def on_nav(nav, config, files):
    for page in nav.pages:
        if page.parent is not None and page.parent.title == page.title:
            page.title = None
    return nav
