"""Bootstrap talon for modern scikit-learn.

talon 1.4.4 still does ``from sklearn.externals import joblib`` and ships a
classifier pickle that references ``sklearn.svm.classes`` (removed upstream).
This helper patches the import path and (re)trains the signature classifier
into a project-local cache that current sklearn can load.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / ".cache" / "talon" / "classifier"


def init_talon(model_path: Path | None = None):
    """Initialize talon quotations + signature extraction.

    Returns ``(signature, quotations)`` modules ready for use.
    """
    import joblib
    import sklearn.externals

    # Satisfy talon's ``from sklearn.externals import joblib``.
    sklearn.externals.joblib = joblib

    import talon
    from talon import quotations, signature
    from talon.quotations import register_xpath_extensions
    from talon.signature import EXTRACTOR_DATA, extraction
    from talon.signature.learning.classifier import init as clf_init
    from talon.signature.learning.classifier import load, train

    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    register_xpath_extensions()

    try:
        extraction.EXTRACTOR = load(str(path), EXTRACTOR_DATA)
    except Exception:
        extraction.EXTRACTOR = train(clf_init(), EXTRACTOR_DATA, str(path))

    # Keep talon's public init() path consistent if callers invoke it later.
    talon.ML_ENABLED = True
    return signature, quotations
