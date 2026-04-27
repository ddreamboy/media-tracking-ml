import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import importlib

TASK_MODULES = [
    "t01_data_fetch",
    "t02_preprocess",
    "t03_embed",
    "t04_train_bertopic",
    "t05_topic_evolution",
    "t06_topic_labeling",
    "t07_validate_model",
    "t08_promote_model",
    "t09_collect_metrics",
]


def cmd_task(args):
    module = importlib.import_module(f"tasks.{args.task}")
    module.main()


def cmd_train(args):
    from pipelines.training_pipeline import run_pipeline

    run_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        sample_size=args.sample_size,
    )


def cmd_validate(args):
    from pipelines.validation_pipeline import run_pipeline

    run_pipeline(
        t04_task_id=args.t04_task_id,
        t05_task_id=args.t05_task_id,
        t06_task_id=args.t06_task_id,
    )


def cmd_monitor(args):
    from pipelines.monitoring_pipeline import run_pipeline
    from shared.config import DRIFT_WINDOW_DAYS

    run_pipeline(drift_window_days=args.drift_window_days or DRIFT_WINDOW_DAYS)


def cmd_register(args):
    from clearml import Task
    from shared.config import CLEARML_PROJECT_NAME

    # Path to each task script relative to the repo root.
    # Task.create() records the git repo + this path so PipelineController
    # can clone the repo and run the script on the agent — without executing anything now.
    tasks = [
        ("t01_data_fetch",    Task.TaskTypes.data_processing),
        ("t02_preprocess",    Task.TaskTypes.data_processing),
        ("t03_embed",         Task.TaskTypes.training),
        ("t04_train_bertopic",Task.TaskTypes.training),
        ("t05_topic_evolution",Task.TaskTypes.data_processing),
        ("t06_topic_labeling", Task.TaskTypes.data_processing),
        ("t07_validate_model", Task.TaskTypes.data_processing),
        ("t08_promote_model",  Task.TaskTypes.data_processing),
        ("t09_collect_metrics",Task.TaskTypes.monitor),
    ]

    for task_name, task_type in tasks:
        t = Task.create(
            project_name=CLEARML_PROJECT_NAME,
            task_name=task_name,
            task_type=task_type,
            script=f"clearml_pipelines/tasks/{task_name}.py",
            working_directory="clearml_pipelines",
            add_task_init_call=False,
        )
        print(f"Registered  {task_name:30s}  id={t.id}")


def main():
    parser = argparse.ArgumentParser(description="ClearML pipeline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    # run a single task
    p_task = sub.add_parser("task", help="Run a single task")
    p_task.add_argument("task", choices=TASK_MODULES)
    p_task.set_defaults(func=cmd_task)

    # training pipeline
    p_train = sub.add_parser("train", help="Run training pipeline (t01-t08)")
    p_train.add_argument("--start-date", dest="start_date", default=None)
    p_train.add_argument("--end-date", dest="end_date", default=None)
    p_train.add_argument("--sample-size", dest="sample_size", type=int, default=0)
    p_train.set_defaults(func=cmd_train)

    # validation/promotion pipeline
    p_val = sub.add_parser("validate", help="Run validation+promotion pipeline (t07-t08)")
    p_val.add_argument("--t04-task-id", dest="t04_task_id", required=True)
    p_val.add_argument("--t05-task-id", dest="t05_task_id", required=True)
    p_val.add_argument("--t06-task-id", dest="t06_task_id", required=True)
    p_val.set_defaults(func=cmd_validate)

    # monitoring pipeline
    p_mon = sub.add_parser("monitor", help="Run monitoring pipeline (t09)")
    p_mon.add_argument("--drift-window-days", dest="drift_window_days", type=int, default=None)
    p_mon.set_defaults(func=cmd_monitor)

    # register all tasks in ClearML (no execution)
    p_reg = sub.add_parser("register", help="Register all tasks in ClearML without running them")
    p_reg.set_defaults(func=cmd_register)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
