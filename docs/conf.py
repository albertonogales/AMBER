import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project   = 'AMBER'
author    = 'Alberto Nogales, Álvaro José García-Tejedor'
copyright = '2026, IERU — Universidad de Alcalá'

try:
    import AMBER
    release = AMBER.__version__
except Exception:
    release = '2.2.0'
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
}

autosummary_generate  = True
autodoc_default_options = {
    'members':          True,
    'undoc-members':    False,
    'show-inheritance': True,
}
napoleon_use_param  = True
napoleon_use_rtype  = True

templates_path  = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix   = {'.rst': 'restructuredtext', '.md': 'markdown'}

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth':  4,
    'titles_only':       False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden':     True,
    'logo_only':         False,
    'prev_next_buttons_location': 'bottom',
}
html_title   = f'AMBER {release}'
html_static_path = ['_static']
html_css_files   = ['custom.css']
