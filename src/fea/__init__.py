"""Feature Engineering Impact Analyzer core library.

A modular, Streamlit-independent data-science package that powers the
``app.py`` user interface. Splitting the logic out of the UI makes it
testable, reusable and easy to extend.
"""

from fea import config, data, encoding, features, modeling, pipeline, preprocessing, visualization

__all__ = [
    "config",
    "data",
    "encoding",
    "features",
    "modeling",
    "pipeline",
    "preprocessing",
    "visualization",
]

__version__ = "1.0.0"
