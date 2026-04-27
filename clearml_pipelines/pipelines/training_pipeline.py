"""Pipeline 1: Training Pipeline - orchestrates t01->t02->t03->t04->t05->t06"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone

from clearml import Task
from clearml.automation.controller import PipelineDecorator
from shared.clearml_utils import is_training_in_progress
from shared.config import (
    CLEARML_PROJECT_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PROVIDER,
)


def _run_task(script_name: str, extra_env: dict = None):
    """Run a task script from clearml_pipelines/tasks/, setting PYTHONPATH correctly."""
    import os
    import subprocess
    import sys

    # Agent runs steps from repo root (working_dir="."), so clearml_pipelines is a subdir
    repo_root = os.path.abspath(".")
    pipelines_dir = os.path.join(repo_root, "clearml_pipelines")
    task_script = os.path.join(pipelines_dir, "tasks", script_name)

    env = {**os.environ, "PYTHONPATH": pipelines_dir}
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [sys.executable, task_script],
        capture_output=True,
        text=True,
        cwd=pipelines_dir,
        env=env,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed:\n{result.stderr}")


@PipelineDecorator.component(
    return_values=["raw_parquet", "meta_json"],
    execution_queue="default",
    task_type=Task.TaskTypes.data_processing,
)
def step_data_fetch(
    start_date: str, end_date: str, min_text_length: int = 50, sample_size: int = 0
) -> tuple:
    _run_task("t01_data_fetch.py", extra_env={"T01_SAMPLE_SIZE": str(sample_size)})
    return "/tmp/raw_data.parquet", "/tmp/dataset_meta.json"


@PipelineDecorator.component(
    return_values=["preprocessed_parquet"],
    execution_queue="default",
    task_type=Task.TaskTypes.data_processing,
)
def step_preprocess(raw_parquet: str) -> str:
    _run_task("t02_preprocess.py")
    return "/tmp/preprocessed.parquet"


@PipelineDecorator.component(
    return_values=["embeddings_npy", "embedding_meta_json"],
    execution_queue="gpu",
    task_type=Task.TaskTypes.training,
)
def step_embed(
    preprocessed_parquet: str, provider: str, model_name: str, batch_size: int
) -> tuple:
    _run_task("t03_embed.py")
    return "/tmp/embeddings.npy", "/tmp/embedding_meta.json"


@PipelineDecorator.component(
    return_values=[
        "model_path",
        "topic_emb_path",
        "topic_info_path",
        "training_meta_path",
    ],
    execution_queue="gpu",
    task_type=Task.TaskTypes.training,
)
def step_train(
    preprocessed_parquet: str, embeddings_npy: str, heterogeneous_ids: str = "[]"
) -> tuple:
    _run_task("t04_train_bertopic.py")
    return (
        "/tmp/bertopic_model",
        "/tmp/topic_embeddings.npy",
        "/tmp/topic_info.csv",
        "/tmp/training_meta.json",
    )


@PipelineDecorator.component(
    return_values=["evolution_report_path"],
    execution_queue="default",
    task_type=Task.TaskTypes.data_processing,
)
def step_evolution(
    model_path: str, topic_emb_path: str, embedding_meta_path: str
) -> str:
    _run_task("t05_topic_evolution.py")
    return "/tmp/evolution_report.json"


@PipelineDecorator.component(
    return_values=["topic_map_path"],
    execution_queue="default",
    task_type=Task.TaskTypes.data_processing,
)
def step_label(
    model_path: str,
    topic_emb_path: str,
    preprocessed_parquet: str,
    evolution_report_path: str,
    embeddings_npy: str,
) -> str:
    _run_task("t06_topic_labeling.py")
    return "/tmp/topic_map_llm.csv"


@PipelineDecorator.pipeline(
    name="Training Pipeline",
    project=CLEARML_PROJECT_NAME,
    version="1.0",
    pipeline_execution_queue="default",
)
def training_pipeline(
    start_date: str,
    end_date: str,
    heterogeneous_ids: str = "[]",
    sample_size: int = 0,
):
    raw_parquet, meta_json = step_data_fetch(
        start_date, end_date, sample_size=sample_size
    )
    preprocessed_parquet = step_preprocess(raw_parquet)
    embeddings_npy, embedding_meta = step_embed(
        preprocessed_parquet,
        provider=EMBEDDING_PROVIDER,
        model_name=EMBEDDING_MODEL_NAME,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    model_path, topic_emb_path, topic_info_path, training_meta_path = step_train(
        preprocessed_parquet, embeddings_npy, heterogeneous_ids
    )
    evolution_report_path = step_evolution(model_path, topic_emb_path, embedding_meta)
    topic_map_path = step_label(
        model_path,
        topic_emb_path,
        preprocessed_parquet,
        evolution_report_path,
        embeddings_npy,
    )
    return topic_map_path


def run_pipeline(start_date: str = None, end_date: str = None, sample_size: int = 0):
    if is_training_in_progress():
        print("Training already in progress - aborting to prevent parallel runs")
        return

    if end_date is None:
        end_date = datetime.now(timezone.utc).date().isoformat()
    if start_date is None:
        from datetime import timedelta

        start_date = (
            (datetime.now(timezone.utc) - timedelta(days=180)).date().isoformat()
        )

    print(
        f"Starting Training Pipeline: {start_date} -> {end_date}, sample_size={sample_size or 'all'}"
    )
    training_pipeline(start_date=start_date, end_date=end_date, sample_size=sample_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--sample-size", type=int, default=1000, help="0 = all records")
    args = parser.parse_args()

    run_pipeline(args.start_date, args.end_date, args.sample_size)
