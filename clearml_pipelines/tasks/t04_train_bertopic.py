"""Task t04: Train BERTopic model"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile

import json
import time

import numpy as np
import pandas as pd
from clearml import Task
from shared.bertopic_utils import build_bertopic, compute_metrics, scale_hparams
from shared.clearml_utils import get_artifact, get_artifact_optional, get_best_hparams
from shared.config import CLEARML_PROJECT_NAME, DEFAULT_HPARAMS_PATH


def main():
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="t04_train_bertopic",
        task_type=Task.TaskTypes.training,
    )
    logger = task.get_logger()

    params = task.connect({"upstream_task_ids": "", "heterogeneous_topic_ids": "[]"})
    het_ids_raw = params.get("heterogeneous_topic_ids", "[]")
    if isinstance(het_ids_raw, str):
        heterogeneous_topic_ids = json.loads(het_ids_raw)
    else:
        heterogeneous_topic_ids = list(het_ids_raw)

    preprocessed_path = get_artifact(task, "preprocessed.parquet")
    embeddings_path = get_artifact(task, "embeddings.npy")

    df = pd.read_parquet(preprocessed_path)
    embeddings = np.load(embeddings_path)
    assert len(df) == len(embeddings), "Mismatch between df and embeddings length"

    base_hparams = get_best_hparams(str(DEFAULT_HPARAMS_PATH))
    print(f"Base hparams: {base_hparams}")

    if heterogeneous_topic_ids:
        from math import floor

        adjusted = dict(base_hparams)
        adjusted["min_cluster_size"] = max(
            10, floor(base_hparams.get("min_cluster_size", 53) * 0.7)
        )
        print(
            f"Applying heterogeneous adjustment: min_cluster_size -> {adjusted['min_cluster_size']}"
        )
        hparams_for_build = adjusted
    else:
        hparams_for_build = base_hparams

    corpus_size = len(df)
    scaled = scale_hparams(hparams_for_build, corpus_size)
    print(f"Scaled hparams: {scaled}")

    # Log all hparams
    task.connect(scaled, name="hparams")

    docs_lemm = df["text_lemm"].tolist()
    model = build_bertopic(hparams_for_build, corpus_size)

    t0 = time.time()
    topics, _ = model.fit_transform(docs_lemm, embeddings=embeddings)
    training_duration = time.time() - t0
    print(f"Training done in {training_duration:.1f}s")

    topics_path = os.path.join(tempfile.gettempdir(), "topics.npy")
    np.save(topics_path, np.array(topics))
    task.upload_artifact("topics.npy", artifact_object=topics_path)

    metrics = compute_metrics(model, topics, docs_lemm)
    print(f"Metrics: {metrics}")

    if metrics["num_topics"] < 20:
        print(f"WARNING: only {metrics['num_topics']} topics found")
    if metrics["noise_ratio"] > 0.80:
        print(f"WARNING: high noise ratio {metrics['noise_ratio']:.3f}")

    logger.report_scalar(
        "metrics", "num_topics", value=metrics["num_topics"], iteration=0
    )
    logger.report_scalar(
        "metrics", "noise_ratio", value=metrics["noise_ratio"], iteration=0
    )
    logger.report_scalar(
        "metrics", "dbcv_score", value=metrics["dbcv_score"], iteration=0
    )
    logger.report_scalar(
        "metrics", "cv_coherence", value=metrics["cv_coherence"], iteration=0
    )
    logger.report_scalar(
        "metrics", "topic_diversity", value=metrics["topic_diversity"], iteration=0
    )
    logger.report_scalar(
        "metrics", "training_duration_seconds", value=training_duration, iteration=0
    )

    # Save model using safetensors to avoid cross-platform numba pickle issues
    model_path = tempfile.mkdtemp(prefix="bertopic_model_")
    model.save(
        model_path,
        serialization="safetensors",
        save_ctfidf=False,
        save_embedding_model=False,
    )
    task.upload_artifact("bertopic_model.model", artifact_object=model_path)

    # topic_embeddings
    topic_ids = sorted([t for t in model.get_topics().keys() if t != -1])
    emb_topics = np.array([model.topic_embeddings_[t + 1] for t in topic_ids])
    emb_path = os.path.join(tempfile.gettempdir(), "topic_embeddings.npy")
    np.save(emb_path, emb_topics)
    task.upload_artifact("topic_embeddings.npy", artifact_object=emb_path)

    # topic_info
    topic_info = model.get_topic_info()
    info_path = os.path.join(tempfile.gettempdir(), "topic_info.csv")
    topic_info.to_csv(info_path, index=False)
    task.upload_artifact("topic_info.csv", artifact_object=info_path)

    # training_meta
    emb_meta_local = get_artifact_optional(task, "embedding_meta.json")
    embedding_model_name = ""
    if emb_meta_local:
        try:
            emb_meta = emb_meta_local.get()
            embedding_model_name = emb_meta.get("model_name", "")
        except Exception:
            pass

    from datetime import datetime, timezone

    training_meta = {
        "scaled_hparams": scaled,
        "base_hparams": base_hparams,
        "corpus_size": corpus_size,
        "embedding_model_name": embedding_model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "training_duration_seconds": training_duration,
    }
    meta_path = os.path.join(tempfile.gettempdir(), "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(training_meta, f, indent=2)
    task.upload_artifact("training_meta.json", artifact_object=meta_path)

    print(f"Done. Topics={metrics['num_topics']}, DBCV={metrics['dbcv_score']:.4f}")
    task.close()


if __name__ == "__main__":
    main()
