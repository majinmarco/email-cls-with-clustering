"""Expand raw emails, or start from an expanded CSV, then clean text."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from email_cls_with_clustering.paths import notebooks_dir
from email_cls_with_clustering.preprocess import expand_emails
from email_cls_with_clustering.text_clean import prepare_messages

DEFAULT_OUTPUT_NAME = "emails_preprocessed.csv"


def run_preprocess(
    source: Path,
    output: Path,
    *,
    from_expanded: bool,
    progress: bool = True,
    redact_pii: bool = True,
    pii_lang: str = "en",
) -> Path:
    """Write a deduped, cleaned frame ready for topic modeling."""
    frame = pd.read_csv(source)
    print(f"loaded {len(frame)} rows from {source}")
    if not from_expanded:
        frame = expand_emails(frame, progress=progress)
        print(f"expanded to {len(frame)} rows")
    cleaned = prepare_messages(
        frame,
        progress=progress,
        redact_pii_enabled=redact_pii,
        pii_lang=pii_lang,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False)
    print(f"wrote {len(cleaned)} rows → {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expand and clean emails into a topic-modeling input CSV.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Raw emails.csv, or emails_expanded.csv with --from-expanded. "
        "Defaults to notebooks/emails.csv (or emails_expanded.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Cleaned CSV. Defaults to notebooks/{DEFAULT_OUTPUT_NAME}.",
    )
    parser.add_argument(
        "--from-expanded",
        action="store_true",
        help="Skip MIME parsing and start from an already expanded CSV.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide tqdm bars.",
    )
    parser.add_argument(
        "--no-pii-redaction",
        action="store_true",
        help="Skip Presidio PII redaction before embeddings/dedupe.",
    )
    parser.add_argument(
        "--pii-lang",
        default="en",
        choices=("en", "es"),
        help="Presidio/spaCy language for PII detection (default: en).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_dir = notebooks_dir()
    if args.from_expanded:
        source = args.input or (data_dir / "emails_expanded.csv")
    else:
        source = args.input or (data_dir / "emails.csv")
    output = args.output or (data_dir / DEFAULT_OUTPUT_NAME)
    run_preprocess(
        source,
        output,
        from_expanded=args.from_expanded,
        progress=not args.no_progress,
        redact_pii=not args.no_pii_redaction,
        pii_lang=args.pii_lang,
    )
