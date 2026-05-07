"""Task t04_reduce_outliers: Two-step noise reduction via BERTopic reduce_outliers"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import tempfile

import numpy as np
import pandas as pd
from bertopic import BERTopic
from clearml import Task
from shared.clearml_utils import get_artifact
from shared.config import CLEARML_PROJECT_NAME, THRESHOLDS_PATH


def _patch_pynndescent():
    try:
        from pynndescent import NNDescent
    except Exception:
        return
    if getattr(NNDescent, "__patched_setstate__", False):
        return
    original_setstate = NNDescent.__setstate__

    def _patched_setstate(self, d):
        defaults = {
            "quantization": None,
            "_min_distance": 0.0,
            "parallel_batch_queries": False,
        }
        for key, val in defaults.items():
            if key not in d:
                d[key] = val
        original_setstate(self, d)

    NNDescent.__setstate__ = _patched_setstate
    NNDescent.__patched_setstate__ = True


def main():
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="t04_reduce_outliers",
        task_type=Task.TaskTypes.data_processing,
    )
    logger = task.get_logger()

    # Load default thresholds from config
    threshold_ctfidf_default = 0.40
    threshold_emb_default = 0.60
    try:
        with open(THRESHOLDS_PATH) as f:
            cfg = json.load(f).get("reduce_outliers", {})
            threshold_ctfidf_default = float(
                cfg.get("threshold_ctfidf", threshold_ctfidf_default)
            )
            threshold_emb_default = float(
                cfg.get("threshold_emb", threshold_emb_default)
            )
    except Exception as e:
        print(f"WARNING: could not load thresholds.json: {e}")

    params = task.connect(
        {
            "upstream_task_ids": "",
            "threshold_ctfidf": threshold_ctfidf_default,
            "threshold_emb": threshold_emb_default,
        }
    )
    threshold_ctfidf = float(params["threshold_ctfidf"])
    threshold_emb = float(params["threshold_emb"])
    print(f"Thresholds: ctfidf={threshold_ctfidf}, embeddings={threshold_emb}")

    # Load artifacts from upstream tasks (t04, t03, t02)
    model_path = get_artifact(task, "bertopic_model.model")
    topics_path = get_artifact(task, "topics.npy")
    embeddings_path = get_artifact(task, "embeddings.npy")
    preprocessed_path = get_artifact(task, "preprocessed.parquet")

    df = pd.read_parquet(preprocessed_path)
    docs_clean = df["text_clean"].tolist()
    docs_lemm = df["text_lemm"].tolist()
    topics_orig = np.load(topics_path).tolist()
    embeddings = np.load(embeddings_path)

    assert len(docs_clean) == len(topics_orig) == len(embeddings), (
        f"Size mismatch: docs={len(docs_clean)}, topics={len(topics_orig)}, emb={len(embeddings)}"
    )

    _patch_pynndescent()
    try:
        topic_model = BERTopic.load(model_path)
    except AttributeError:
        _patch_pynndescent()
        topic_model = BERTopic.load(model_path)

    noise_before = float((np.array(topics_orig) == -1).mean())
    n_outliers_before = int((np.array(topics_orig) == -1).sum())
    print(f"Noise before: {noise_before:.4f} ({n_outliers_before} outlier docs)")

    from sklearn.exceptions import NotFittedError as _NotFittedError

    try:
        topic_model.vectorizer_model.transform(["test"])
    except _NotFittedError:
        print("Vectorizer not fitted - refitting from docs_lemm")
        topic_model.vectorizer_model.fit(docs_lemm)

    # Step 1: c-tf-idf - fast, no embeddings required
    topics_after_ctfidf = topic_model.reduce_outliers(
        docs_clean,
        topics_orig,
        strategy="c-tf-idf",
        threshold=threshold_ctfidf,
    )
    reassigned_step1 = int(
        (np.array(topics_orig) == -1).sum()
        - (np.array(topics_after_ctfidf) == -1).sum()
    )
    noise_step1 = float((np.array(topics_after_ctfidf) == -1).mean())
    print(
        f"Step 1 (c-tf-idf, thr={threshold_ctfidf}): reassigned={reassigned_step1}, noise={noise_step1:.4f}"
    )

    # Step 2: embeddings - cosine similarity to topic centroid
    topics_after_emb = topic_model.reduce_outliers(
        docs_clean,
        topics_after_ctfidf,
        strategy="embeddings",
        embeddings=embeddings,
        threshold=threshold_emb,
    )
    reassigned_step2 = int(
        (np.array(topics_after_ctfidf) == -1).sum()
        - (np.array(topics_after_emb) == -1).sum()
    )
    noise_after = float((np.array(topics_after_emb) == -1).mean())
    print(
        f"Step 2 (embeddings, thr={threshold_emb}): reassigned={reassigned_step2}, noise={noise_after:.4f}"
    )

    # Recompute c-TF-IDF representations with updated topic assignments
    topic_model.update_topics(docs_lemm, topics=topics_after_emb)
    print(f"update_topics done. Topics in model: {len(topic_model.get_topics()) - 1}")

    # Sanity check: reassignment distribution
    orig_arr = np.array(topics_orig)
    after_arr = np.array(topics_after_emb)
    reassigned_mask = (orig_arr == -1) & (after_arr != -1)
    if reassigned_mask.sum() > 0:
        import pandas as _pd

        topic_counts = _pd.Series(after_arr[reassigned_mask]).value_counts()
        top1_share = topic_counts.iloc[0] / reassigned_mask.sum()
        print(
            f"Reassigned {reassigned_mask.sum()} docs across {len(topic_counts)} topics (top-1 share: {top1_share:.1%})"
        )
        if top1_share > 0.30:
            print(
                "WARNING: >30% of outliers went to a single topic - consider lowering threshold_emb"
            )

    # Log metrics to ClearML
    logger.report_scalar(
        "noise_reduction", "noise_before", value=noise_before, iteration=0
    )
    logger.report_scalar(
        "noise_reduction", "noise_after", value=noise_after, iteration=0
    )
    logger.report_scalar(
        "noise_reduction", "delta_noise", value=noise_after - noise_before, iteration=0
    )
    logger.report_scalar(
        "noise_reduction",
        "reassigned_step1_ctfidf",
        value=reassigned_step1,
        iteration=0,
    )
    logger.report_scalar(
        "noise_reduction", "reassigned_step2_emb", value=reassigned_step2, iteration=0
    )
    logger.report_scalar(
        "noise_reduction",
        "reassigned_total",
        value=reassigned_step1 + reassigned_step2,
        iteration=0,
    )

    # Upload updated topics.npy (same artifact name - overrides t04's in upstream chain)
    topics_out_path = os.path.join(tempfile.gettempdir(), "topics.npy")
    np.save(topics_out_path, np.array(topics_after_emb))
    task.upload_artifact("topics.npy", artifact_object=topics_out_path)

    # Upload updated model (same artifact name - overrides t04's in upstream chain)
    model_out_path = tempfile.mkdtemp(prefix="bertopic_reduced_")
    topic_model.save(
        model_out_path,
        serialization="safetensors",
        save_ctfidf=False,
        save_embedding_model=False,
    )
    task.upload_artifact("bertopic_model.model", artifact_object=model_out_path)

    print(
        f"Done. noise {noise_before:.4f} -> {noise_after:.4f} "
        f"(delta={noise_after - noise_before:+.4f}), "
        f"total reassigned={reassigned_step1 + reassigned_step2}"
    )
    task.close()


if __name__ == "__main__":
    main()
