"""Validation & Promotion Pipeline: t07 → t08 (standalone, accepts training task IDs)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clearml.automation import PipelineController

from shared.config import CLEARML_PROJECT_NAME


def run_pipeline(
    t04_task_id: str,
    t05_task_id: str,
    t06_task_id: str,
):
    """
    Run validation + promotion on top of an existing training run.

    Args:
        t04_task_id: ClearML task ID of completed t04_train_bertopic
        t05_task_id: ClearML task ID of completed t05_topic_evolution
        t06_task_id: ClearML task ID of completed t06_topic_labeling
    """
    print(
        f"Starting Validation Pipeline: "
        f"t04={t04_task_id}, t05={t05_task_id}, t06={t06_task_id}"
    )

    pipe = PipelineController(
        project=CLEARML_PROJECT_NAME,
        name="Validation & Promotion Pipeline",
        version="1.0",
        add_pipeline_tags=True,
    )
    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="t07_validate_model",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t07_validate_model",
        parameter_override={
            "General/upstream_task_ids": t04_task_id,
        },
        execution_queue="default",
    )
    pipe.add_step(
        name="t08_promote_model",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t08_promote_model",
        parents=["t07_validate_model"],
        parameter_override={
            "General/upstream_task_ids": (
                f"${{t07_validate_model.id}},{t06_task_id},{t05_task_id},{t04_task_id}"
            ),
        },
        execution_queue="default",
    )

    pipe.start_locally(run_pipeline_steps_locally=False)
    print("Pipeline enqueued. Monitor at ClearML UI.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--t04-task-id", required=True)
    parser.add_argument("--t05-task-id", required=True)
    parser.add_argument("--t06-task-id", required=True)
    args = parser.parse_args()

    run_pipeline(args.t04_task_id, args.t05_task_id, args.t06_task_id)
