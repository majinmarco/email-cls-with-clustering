"""Fit BERTopic and write a datetime-suffixed CSV. Runnable on its own."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from email_cls_with_clustering.commands.topic_model import main

if __name__ == "__main__":
    main()
