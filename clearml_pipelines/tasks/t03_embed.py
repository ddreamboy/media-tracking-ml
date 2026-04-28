"""Task t03: Generate sentence embeddings"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile

import json

import numpy as np
import pandas as pd
from clearml import Task
from shared.clearml_utils import get_artifact, get_production_embedding_model_name
from shared.config import (
    CLEARML_PROJECT_NAME,
    EMBEDDING_API_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PROVIDER,
)
from shared.embedder import generate_embeddings


def main():
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="t03_embed",
        task_type=Task.TaskTypes.training,
    )
    logger = task.get_logger()

    params = task.connect(
        {
            "upstream_task_ids": "",
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model_name": EMBEDDING_MODEL_NAME,
            "batch_size": EMBEDDING_BATCH_SIZE,
        }
    )

    provider = params["embedding_provider"]
    model_name = params["embedding_model_name"]
    batch_size = int(params["batch_size"])
    api_key = task.get_parameter("Args/embedding_api_key") or EMBEDDING_API_KEY
    api_base_url = (
        task.get_parameter("Args/embedding_api_base_url") or EMBEDDING_API_BASE_URL
    )

    preprocessed_path = get_artifact(task, "preprocessed.parquet")
    df = pd.read_parquet(preprocessed_path)
    texts = df["text_clean"].tolist()
    print(
        f"Generating embeddings for {len(texts):,} docs, provider={provider}, model={model_name}"
    )

    prod_model_name = get_production_embedding_model_name()
    embedding_changed = prod_model_name is not None and prod_model_name != model_name
    if embedding_changed:
        print(
            f"WARNING: EMBEDDING_MODEL_CHANGED - prod={prod_model_name}, new={model_name}"
        )

    # Log as parameters
    task.connect(
        {"embedding_provider": provider, "embedding_model_name": model_name},
        name="parameters",
    )

    embeddings, meta = generate_embeddings(
        texts, provider, model_name, batch_size, api_key, api_base_url
    )
    meta["embedding_changed"] = embedding_changed

    logger.report_scalar(
        "embedding", "embedding_dim", value=meta["embedding_dim"], iteration=0
    )
    logger.report_scalar(
        "embedding",
        "inference_time_seconds",
        value=meta["inference_time_seconds"],
        iteration=0,
    )
    logger.report_scalar(
        "embedding",
        "throughput_docs_per_sec",
        value=meta["throughput_docs_per_sec"],
        iteration=0,
    )

    emb_path = os.path.join(tempfile.gettempdir(), "embeddings.npy")
    np.save(emb_path, embeddings)
    task.upload_artifact("embeddings.npy", artifact_object=emb_path)

    meta_path = os.path.join(tempfile.gettempdir(), "embedding_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    task.upload_artifact("embedding_meta.json", artifact_object=meta_path)

    print(f"Done. Shape: {embeddings.shape}, changed={embedding_changed}")
    task.close()


if __name__ == "__main__":
    main()
