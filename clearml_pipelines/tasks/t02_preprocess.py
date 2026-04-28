"""Task t02: Preprocess texts (clean + lemmatize)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile

import pandas as pd
from clearml import Task
from shared.clearml_utils import get_artifact
from shared.config import CLEARML_PROJECT_NAME
from shared.preprocessing import preprocess_texts


def main():
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="t02_preprocess",
        task_type=Task.TaskTypes.data_processing,
    )
    logger = task.get_logger()

    params = task.connect({"upstream_task_ids": "", "n_jobs": 4, "chunk_size": 10000})
    n_jobs = int(params["n_jobs"])
    chunk_size = int(params["chunk_size"])

    raw_parquet = get_artifact(task, "raw_data.parquet")
    df = pd.read_parquet(raw_parquet)
    print(f"Loaded {len(df):,} records")

    texts = df["text"].fillna("").tolist()
    clean_texts, lemm_texts = preprocess_texts(
        texts, n_jobs=n_jobs, chunk_size=chunk_size
    )

    df["text_clean"] = clean_texts
    df["text_lemm"] = lemm_texts

    empty_after_lemm = (df["text_lemm"].str.strip() == "").sum()
    df = df[df["text_lemm"].str.strip() != ""].reset_index(drop=True)

    median_token_count = df["text_lemm"].str.split().str.len().median()
    median_char_length = df["text_clean"].str.len().median()

    logger.report_scalar(
        "preprocessing", "empty_after_lemm", value=int(empty_after_lemm), iteration=0
    )
    logger.report_scalar(
        "preprocessing",
        "median_token_count",
        value=float(median_token_count),
        iteration=0,
    )
    logger.report_scalar(
        "preprocessing",
        "median_char_length",
        value=float(median_char_length),
        iteration=0,
    )

    out_cols = ["post_id", "channel", "created_at", "text_clean", "text_lemm"]
    out_cols = [c for c in out_cols if c in df.columns]
    out_path = os.path.join(tempfile.gettempdir(), "preprocessed.parquet")
    df[out_cols].to_parquet(out_path, index=False)
    task.upload_artifact("preprocessed.parquet", artifact_object=out_path)

    print(f"Done. Records: {len(df):,}, empty_lemm={empty_after_lemm}")
    task.close()


if __name__ == "__main__":
    main()
