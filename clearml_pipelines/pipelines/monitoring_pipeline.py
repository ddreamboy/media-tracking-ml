"""Pipeline 3: Monitoring Pipeline - runs t09 on schedule (every 24h)"""

from clearml import Task
from clearml.automation import TriggerScheduler
from clearml.automation.controller import PipelineDecorator
from shared.config import CLEARML_PROJECT_NAME, DRIFT_WINDOW_DAYS


@PipelineDecorator.component(
    return_values=["monitoring_report_path"],
    execution_queue="default",
    task_type=Task.TaskTypes.monitor,
)
def step_collect_metrics(drift_window_days: int) -> str:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "tasks/t09_collect_metrics.py"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"WARNING: t09_collect_metrics failed:\n{result.stderr}")
    return "/tmp/monitoring_report.json"


@PipelineDecorator.pipeline(
    name="Monitoring Pipeline",
    project=CLEARML_PROJECT_NAME,
    version="1.0",
    pipeline_execution_queue="default",
)
def monitoring_pipeline(drift_window_days: int = DRIFT_WINDOW_DAYS):
    report_path = step_collect_metrics(drift_window_days)
    return report_path


def schedule_monitoring(cron_expression: str = "0 3 * * *"):
    """Schedule monitoring pipeline to run daily at 3 AM UTC"""
    scheduler = TriggerScheduler()
    scheduler.add_task(
        schedule_task_id=None,
        schedule_function=lambda: run_pipeline_once(),
        cron_expression=cron_expression,
        task_name="Monitoring Pipeline",
        project_name=CLEARML_PROJECT_NAME,
    )
    scheduler.start()
    print(f"Monitoring pipeline scheduled: {cron_expression}")


def run_pipeline_once():
    print("Starting Monitoring Pipeline")
    PipelineDecorator.run_locally()
    monitoring_pipeline(drift_window_days=DRIFT_WINDOW_DAYS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule", action="store_true", help="Schedule pipeline (24h interval)"
    )
    parser.add_argument("--cron", default="0 3 * * *")
    args = parser.parse_args()

    if args.schedule:
        schedule_monitoring(args.cron)
    else:
        run_pipeline_once()
