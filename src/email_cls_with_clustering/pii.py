"""Presidio-based PII redaction for topic-modeling text.

Applied in ``text_clean.prepare_messages`` after MIME/quote stripping (``body``
from ``preprocess``) and boilerplate removal, and before ``embed_text`` /
``ctfidf_text`` are built. That keeps names, phones, and addresses out of
sentence embeddings and cluster assignments while leaving raw headers on the
frame for threading metadata.

spaCy models are loaded lazily on first ``redact`` call. Install:

    python -m spacy download en_core_web_lg
    python -m spacy download es_core_news_sm
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

_ENTITIES = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "LOCATION",
)

_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None


def _engines() -> tuple[Any, Any]:
    global _analyzer, _anonymizer
    if _analyzer is None or _anonymizer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "en", "model_name": "en_core_web_lg"},
                    {"lang_code": "es", "model_name": "es_core_news_sm"},
                ],
            }
        )
        _analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=["en", "es"],
        )
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


def redact(text: str, lang: str = "en") -> str:
    """Replace detected PII in ``text`` with Presidio placeholders."""
    if not isinstance(text, str) or not text.strip():
        return "" if text is None else str(text)

    analyzer, anonymizer = _engines()
    results = analyzer.analyze(
        text=text,
        language=lang,
        entities=list(_ENTITIES),
    )
    return anonymizer.anonymize(text=text, analyzer_results=results).text


def reset_engines() -> None:
    """Clear cached engines (for tests)."""
    global _analyzer, _anonymizer
    _analyzer = None
    _anonymizer = None
