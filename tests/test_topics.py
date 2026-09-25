# tests/test_topics.py
import numpy as np
import pandas as pd

from email_cls_with_clustering.topics import TopicHyperparams, write_topic_dataset


def test_probabilities_go_to_npy_not_csv(tmp_path):
    frame = pd.DataFrame({"full_message": ["a b", "c d"], "topic": [0, -1]})
    probs = np.array([[0.9, 0.1], [0.4, 0.6]], dtype=np.float32)
    out = write_topic_dataset(
        frame, tmp_path / "clustered.csv", TopicHyperparams(),
        source=tmp_path / "in.csv", cache_path=tmp_path / "emb.npy",
        probabilities=probs,
    )
    assert "prob_vector" not in pd.read_csv(out).columns
    reloaded = np.load(out.with_name(f"{out.stem}_probs.npy"))
    assert reloaded.shape == (2, 2)
    assert reloaded.dtype.kind == "f"   # numbers, not strings