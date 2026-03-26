# -- Project information -------------------------------------------------------
project = "pmf-acls"
copyright = "2025, Jerritt Collord"
author = "Jerritt Collord"
release = "0.1.6"

# -- General configuration ----------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
]

# MyST settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# myst-nb: notebook execution
nb_execution_mode = "auto"          # execute notebooks that have no outputs
nb_execution_timeout = 300
nb_execution_raise_on_error = True

# BibTeX
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# -- HTML output ---------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "pmf-acls"

html_theme_options = {
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pmf-acls/",
            "icon": "fa-solid fa-box",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
}

html_context = {
    "github_user": "collord",
    "github_repo": "pmf-acls-docs",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Suppress warnings --------------------------------------------------------
suppress_warnings = ["myst.header"]
