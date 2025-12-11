import sys
import os

sys.path.insert(0, os.path.abspath(".."))

import nmma

extensions = ["myst_parser", "sphinx_copybutton", "sphinx_github_changelog"]
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

mathjax_config = {
    "tex2jax": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}

sphinx_github_changelog_token = os.getenv("SPHINX_GITHUB_CHANGELOG_TOKEN")


templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme_options = {
    "path_to_docs": "doc",
    "repository_url": "https://github.com/nuclear-multimessenger-astronomy/nmma",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}


master_doc = "index"


project = "nmma"
copyright = "2023, The NMMA Team"
author = "The NMMA Team"


version = nmma.__version__
release = version


language = None


exclude_patterns = ["_build"]


todo_include_todos = False


html_theme = "sphinx_book_theme"


html_favicon = "_static/favicon-light.png"


html_static_path = ["_static"]

html_logo = "_static/light-logo.svg"


html_show_sourcelink = False


htmlhelp_basename = "nmmadoc"


latex_elements = {}


latex_documents = [
    (
        master_doc,
        "nmma.tex",
        "nmma Documentation",
        "The nmma Team",
        "manual",
    ),
]


man_pages = [(master_doc, "nmma", "nmma Documentation", [author], 1)]


texinfo_documents = [
    (
        master_doc,
        "nmma",
        "nmma Documentation",
        author,
        "nmma",
        "One line description of project.",
        "Miscellaneous",
    ),
]


def setup(app):
    app.add_css_file("nmma-docs.css")
