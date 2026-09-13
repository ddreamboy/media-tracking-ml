"""Training + Validation + Promotion Pipeline: t01 -> t02 -> t03 -> t04 -> t04_reduce_outliers -> t05/t07 -> t06 -> t08"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone

from clearml.automation import PipelineController
from shared.clearml_utils import is_training_in_progress
from shared.config import (
    CLEARML_PROJECT_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MAX_WORKERS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PROVIDER,
    EMBEDDING_TORCH_DTYPE,
)

# Artifact dependency map:
# t01 -> t02(raw_data.parquet)
# t02 -> t03(preprocessed.parquet)
# t02,t03 -> t04(preprocessed.parquet, embeddings.npy, embedding_meta.json)
# t04,t03,t02 -> t04_reduce_outliers(bertopic_model.model*, topics.npy*)  *overrides t04 in upstream chain
# t04_reduce_outliers,t03 -> t05(bertopic_model.model, embedding_meta.json)
# t04_reduce_outliers,t02,t05,t03 -> t06(bertopic_model.model, topics.npy, preprocessed.parquet, evolution_report.json, embeddings.npy)
# t04_reduce_outliers,t03 -> t07(training_meta.json with post-reduction noise_ratio, embedding_meta.json)
# t07,t06,t05,t04_reduce_outliers,t04 -> t08(validation_report.json, topic_map_llm.csv, bertopic_model.model, ...)


def _find_latest_hpo_task():
    """Ищет последнюю завершённую HPO_BERTopic задачу. Возвращает (task_id, sample_size) или (None, 0)"""
    from clearml import Task as ClearMLTask

    tasks = ClearMLTask.get_tasks(
        project_name=CLEARML_PROJECT_NAME,
        task_name="HPO_BERTopic",
        task_filter={"status": ["completed"], "order_by": ["-last_update"]},
    )
    if not tasks:
        return None, 0
    hpo = tasks[0]
    sample_size = int(hpo.get_parameter("Args/sample_size") or hpo.get_parameter("General/sample_size") or 0)
    print(f"Found HPO task: {hpo.id}  sample_size={sample_size}")
    return hpo.id, sample_size


def run_pipeline(
    start_date: str = None,
    end_date: str = None,
    sample_size: int = 0,
):
    if is_training_in_progress():
        print("Training already in progress - aborting to prevent parallel runs")
        return

    if end_date is None:
        end_date = datetime.now(timezone.utc).date().isoformat()
    if start_date is None:
        from datetime import timedelta

        start_date = (datetime.now(timezone.utc) - timedelta(days=180)).date().isoformat()

    hpo_task_id, hpo_sample_size = _find_latest_hpo_task()
    if hpo_task_id and sample_size == 0:
        sample_size = hpo_sample_size
        print(f"Using sample_size={sample_size} from HPO task")

    print(
        f"Starting Training Pipeline: {start_date} -> {end_date}, "
        f"sample_size={sample_size or 'all'}  hpo_task_id={hpo_task_id or 'none'}"
    )

    pipe = PipelineController(
        project=CLEARML_PROJECT_NAME,
        name="Training Pipeline",
        version="1.0",
        add_pipeline_tags=True,
    )
    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="t01_data_fetch",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t01_data_fetch",
        parameter_override={
            "General/start_date": start_date,
            "General/end_date": end_date,
            "General/sample_size": sample_size,
        },
        execution_queue="default",
        cache_executed_step=False,
    )
    pipe.add_step(
        name="t02_preprocess",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t02_preprocess",
        parents=["t01_data_fetch"],
        parameter_override={
            "General/upstream_task_ids": "${t01_data_fetch.id}",
        },
        execution_queue="default",
    )
    pipe.add_step(
        name="t03_embed",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t03_embed",
        parents=["t02_preprocess"],
        parameter_override={
            "General/upstream_task_ids": "${t02_preprocess.id}",
            "General/embedding_provider": EMBEDDING_PROVIDER,
            "General/embedding_model_name": EMBEDDING_MODEL_NAME,
            "General/batch_size": EMBEDDING_BATCH_SIZE,
            "General/max_seq_length": EMBEDDING_MAX_SEQ_LENGTH,
            "General/torch_dtype": EMBEDDING_TORCH_DTYPE,
            "General/dimensions": EMBEDDING_DIMENSIONS,
            "General/max_workers": EMBEDDING_MAX_WORKERS,
        },
        execution_queue="gpu",
    )
    pipe.add_step(
        name="t04_train_bertopic",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t04_train_bertopic",
        parents=["t02_preprocess", "t03_embed"],
        parameter_override={
            "General/upstream_task_ids": "${t02_preprocess.id},${t03_embed.id}",
            "General/hpo_task_id": hpo_task_id or "",
        },
        execution_queue="gpu",
    )
    pipe.add_step(
        name="t04_reduce_outliers",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t04_reduce_outliers",
        parents=["t04_train_bertopic", "t03_embed", "t02_preprocess"],
        parameter_override={
            "General/upstream_task_ids": ("${t04_train_bertopic.id},${t03_embed.id},${t02_preprocess.id}"),
        },
        execution_queue="default",
    )
    pipe.add_step(
        name="t05_topic_evolution",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t05_topic_evolution",
        parents=["t04_reduce_outliers"],
        parameter_override={
            "General/upstream_task_ids": "${t04_reduce_outliers.id},${t03_embed.id}",
        },
        execution_queue="default",
    )
    pipe.add_step(
        name="t06_topic_labeling",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t06_topic_labeling",
        parents=["t04_reduce_outliers", "t05_topic_evolution"],
        parameter_override={
            "General/upstream_task_ids": (
                "${t04_reduce_outliers.id},${t02_preprocess.id},"
                "${t05_topic_evolution.id},${t03_embed.id},${t04_train_bertopic.id}"
            ),
        },
        execution_queue="default",
    )
    pipe.add_step(
        name="t07_validate_model",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t07_validate_model",
        parents=["t04_reduce_outliers"],
        parameter_override={
            "General/upstream_task_ids": "${t04_reduce_outliers.id},${t03_embed.id}",
        },
        execution_queue="default",
    )
    pipe.add_step(
        name="t08_promote_model",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t08_promote_model",
        parents=["t07_validate_model", "t06_topic_labeling"],
        parameter_override={
            "General/upstream_task_ids": (
                "${t07_validate_model.id},${t06_topic_labeling.id},"
                "${t05_topic_evolution.id},${t04_reduce_outliers.id},${t04_train_bertopic.id}"
            ),
        },
        execution_queue="default",
    )

    pipe.start_locally(run_pipeline_steps_locally=False)
    print("Pipeline enqueued. Monitor at ClearML UI.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--sample-size", type=int, default=0, help="0 = all records")
    args = parser.parse_args()

    run_pipeline(args.start_date, args.end_date, args.sample_size)
