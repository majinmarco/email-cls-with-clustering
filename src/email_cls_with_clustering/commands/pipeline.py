"""Run preprocessing, then topic modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

from email_cls_with_clustering.commands.preprocess import (
    DEFAULT_OUTPUT_NAME as PREPROCESSED_NAME,
    run_preprocess,
)
from email_cls_with_clustering.commands.topic_model import (
    DEFAULT_OUTPUT_NAME as CLUSTERED_NAME,
    add_document_unit_args,
    add_eval_args,
    add_hyperparam_args,
    params_from_args,
    run_topic_model,
)
from email_cls_with_clustering.paths import notebooks_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess emails, then fit BERTopic into a timestamped CSV.",
    )
    add_hyperparam_args(parser)
    add_eval_args(parser)
    add_document_unit_args(parser)
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=None,
        help="Raw emails.csv, or an expanded CSV with --from-expanded.",
    )
    parser.add_argument(
        "--preprocessed-output",
        type=Path,
        default=None,
        help=f"Cleaned CSV. Defaults to notebooks/{PREPROCESSED_NAME}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Clustered CSV path before the datetime suffix. "
        f"Defaults to notebooks/{CLUSTERED_NAME}.",
    )
    parser.add_argument(
        "--from-expanded",
        action="store_true",
        help="Skip MIME parsing and start from an already expanded CSV.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Hide tqdm bars.")
    parser.add_argument(
        "--no-pii-redaction",
        action="store_true",
        help="Skip Presidio PII redaction during preprocess.",
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
        source = args.raw_input or (data_dir / "emails_expanded.csv")
    else:
        source = args.raw_input or (data_dir / "emails.csv")
    preprocessed = args.preprocessed_output or (data_dir / PREPROCESSED_NAME)
    clustered = args.output or (data_dir / CLUSTERED_NAME)
    env_file = args.env_file or (data_dir / ".env")

    run_preprocess(
        source,
        preprocessed,
        from_expanded=args.from_expanded,
        progress=not args.no_progress,
        redact_pii=not args.no_pii_redaction,
        pii_lang=args.pii_lang,
    )
    written = run_topic_model(
        preprocessed,
        clustered,
        params_from_args(args),
        env_file=env_file,
        coherence=not args.no_coherence,
        dbcv=not args.no_dbcv,
        join_threads=args.join_threads,
    )
    print(f"pipeline output → {written}")
