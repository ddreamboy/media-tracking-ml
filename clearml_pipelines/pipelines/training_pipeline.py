"""Training + Validation + Promotion Pipeline: t01 → t02 → t03 → t04 → t05 → t06 → t07 → t08"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone

from clearml.automation import PipelineController

from shared.clearml_utils import is_training_in_progress
from shared.config import (
    CLEARML_PROJECT_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PROVIDER,
)

# Artifact dependency map:
# t01 → t02(raw_data.parquet)
# t02 → t03(preprocessed.parquet)
# t02,t03 → t04(preprocessed.parquet, embeddings.npy, embedding_meta.json)
# t04,t03 → t05(bertopic_model.model, embedding_meta.json)
# t04,t02,t05,t03 → t06(bertopic_model.model, topic_embeddings.npy, preprocessed.parquet, evolution_report.json, embeddings.npy)
# t04 → t07(training_meta.json)
# t07,t06,t05,t04 → t08(validation_report.json, topic_map_llm.csv, evolution_report.json, bertopic_model.model, ...)


def run_pipeline(
    start_date: str = None,
    end_date: str = None,
    sample_size: int = 0,
):
    if is_training_in_progress():
        print("Training already in progress — aborting to prevent parallel runs")
        return

    if end_date is None:
        end_date = datetime.now(timezone.utc).date().isoformat()
    if start_date is None:
        from datetime import timedelta

        start_date = (
            (datetime.now(timezone.utc) - timedelta(days=180)).date().isoformat()
        )

    print(
        f"Starting Training Pipeline: {start_date} → {end_date}, "
        f"sample_size={sample_size or 'all'}"
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
        parameter_override={"General/sample_size": sample_size},
        execution_queue="default",
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
        },
        execution_queue="gpu",
    )
    pipe.add_step(
        name="t05_topic_evolution",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t05_topic_evolution",
        parents=["t04_train_bertopic"],
        parameter_override={
            "General/upstream_task_ids": "${t04_train_bertopic.id},${t03_embed.id}",
        },
        execution_queue="default",
    )
    pipe.add_step(
        name="t06_topic_labeling",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t06_topic_labeling",
        parents=["t04_train_bertopic", "t05_topic_evolution"],
        parameter_override={
            "General/upstream_task_ids": (
                "${t04_train_bertopic.id},${t02_preprocess.id},"
                "${t05_topic_evolution.id},${t03_embed.id}"
            ),
        },
        execution_queue="default",
    )
    pipe.add_step(
        name="t07_validate_model",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t07_validate_model",
        parents=["t04_train_bertopic"],
        parameter_override={
            "General/upstream_task_ids": "${t04_train_bertopic.id}",
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
                "${t05_topic_evolution.id},${t04_train_bertopic.id}"
            ),
        },
        execution_queue="default",
    )

    pipe.start(queue="default")
    print("Pipeline enqueued. Monitor at ClearML UI.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--sample-size", type=int, default=0, help="0 = all records")
    args = parser.parse_args()

    run_pipeline(args.start_date, args.end_date, args.sample_size)
