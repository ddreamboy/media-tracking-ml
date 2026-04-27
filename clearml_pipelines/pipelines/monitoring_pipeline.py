"""Monitoring Pipeline: t09 (runs standalone on schedule)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clearml.automation import PipelineController

from shared.config import CLEARML_PROJECT_NAME, DRIFT_WINDOW_DAYS


def run_pipeline(drift_window_days: int = DRIFT_WINDOW_DAYS):
    print(f"Starting Monitoring Pipeline: drift_window_days={drift_window_days}")

    pipe = PipelineController(
        project=CLEARML_PROJECT_NAME,
        name="Monitoring Pipeline",
        version="1.0",
        add_pipeline_tags=True,
    )
    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="t09_collect_metrics",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="t09_collect_metrics",
        parameter_override={
            "General/drift_window_days": drift_window_days,
        },
        execution_queue="default",
    )

    pipe.start_locally(run_pipeline_steps_locally=False)
    print("Pipeline enqueued. Monitor at ClearML UI.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drift-window-days", type=int, default=DRIFT_WINDOW_DAYS
    )
    args = parser.parse_args()

    run_pipeline(args.drift_window_days)
