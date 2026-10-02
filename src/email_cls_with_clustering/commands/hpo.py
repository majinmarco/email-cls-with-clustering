"""CLI for staged topic-model hyperparameter search."""

from __future__ import annotations

import argparse
from pathlib import Path

from email_cls_with_clustering.commands.topic_model import add_document_unit_args
from email_cls_with_clustering.hpo import DEFAULT_SPEC, run_hpo
from email_cls_with_clustering.paths import notebooks_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run staged BERTopic HPO (representation → UMAP → HDBSCAN → outliers).",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
        help="Nested HPO JSON. Defaults to configs/enron-email-topic-model-hpo.json.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Cleaned CSV. Defaults to notebooks/emails_preprocessed_no_spam.csv.",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="Resume from this stage, reading the previous stage's advanced.json.",
    )
    parser.add_argument(
        "--hdbscan-search",
        choices=("grid", "optuna"),
        default="grid",
        help="Stage-3 search. Optuna is the fallback when the full grid is too slow.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Stage-4 clustered CSV stem (trial id is inserted before the suffix).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="dotenv file for the OpenAI backend. Defaults to notebooks/.env.",
    )
    add_document_unit_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_dir = notebooks_dir()
    run_hpo(
        spec_path=args.spec,
        input_path=args.input or (data_dir / "emails_preprocessed_no_spam.csv"),
        from_stage=args.from_stage,
        hdbscan_search=args.hdbscan_search,
        output=args.output,
        env_file=args.env_file or (data_dir / ".env"),
        join_threads=args.join_threads,
    )


if __name__ == "__main__":
    main()
