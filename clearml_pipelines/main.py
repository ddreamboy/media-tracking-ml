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


def _get_https_repo_url() -> str:
    import re
    import subprocess

    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    url = result.stdout.strip()
    # git@github.com:user/repo.git -> https://github.com/user/repo.git
    url = re.sub(r"^git@([^:]+):", r"https://\1/", url)
    return url


def cmd_hpo(args):
    from hpo.run_hpo import main as hpo_main
    import sys

    argv = []
    if args.n_trials is not None:
        argv += ["--n-trials", str(args.n_trials)]
    if args.sample_size is not None:
        argv += ["--sample-size", str(args.sample_size)]
    if args.study_name is not None:
        argv += ["--study-name", args.study_name]
    if args.preprocess_task_id:
        argv += ["--preprocess-task-id", args.preprocess_task_id]
    if args.embed_task_id:
        argv += ["--embed-task-id", args.embed_task_id]
    if args.data_path:
        argv += ["--data-path", args.data_path]
    if args.embeddings_path:
        argv += ["--embeddings-path", args.embeddings_path]

    sys.argv = [sys.argv[0]] + argv
    hpo_main()


def cmd_register(args):
    from clearml import Task
    from shared.config import CLEARML_PROJECT_NAME

    repo_url = _get_https_repo_url()
    print(f"Registering tasks from repo: {repo_url}")

    tasks = [
        ("t01_data_fetch", Task.TaskTypes.data_processing),
        ("t02_preprocess", Task.TaskTypes.data_processing),
        ("t03_embed", Task.TaskTypes.training),
        ("t04_train_bertopic", Task.TaskTypes.training),
        ("t05_topic_evolution", Task.TaskTypes.data_processing),
        ("t06_topic_labeling", Task.TaskTypes.data_processing),
        ("t07_validate_model", Task.TaskTypes.data_processing),
        ("t08_promote_model", Task.TaskTypes.data_processing),
        ("t09_collect_metrics", Task.TaskTypes.monitor),
    ]

    for task_name, task_type in tasks:
        t = Task.create(
            project_name=CLEARML_PROJECT_NAME,
            task_name=task_name,
            task_type=task_type,
            repo=repo_url,
            script=f"clearml_pipelines/tasks/{task_name}.py",
            working_directory=".",
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
    p_val = sub.add_parser(
        "validate", help="Run validation+promotion pipeline (t07-t08)"
    )
    p_val.add_argument("--t04-task-id", dest="t04_task_id", required=True)
    p_val.add_argument("--t05-task-id", dest="t05_task_id", required=True)
    p_val.add_argument("--t06-task-id", dest="t06_task_id", required=True)
    p_val.set_defaults(func=cmd_validate)

    # monitoring pipeline
    p_mon = sub.add_parser("monitor", help="Run monitoring pipeline (t09)")
    p_mon.add_argument(
        "--drift-window-days", dest="drift_window_days", type=int, default=None
    )
    p_mon.set_defaults(func=cmd_monitor)

    # HPO
    p_hpo = sub.add_parser("hpo", help="Run BERTopic hyperparameter optimisation")
    p_hpo.add_argument("--n-trials", dest="n_trials", type=int, default=None)
    p_hpo.add_argument("--sample-size", dest="sample_size", type=int, default=None)
    p_hpo.add_argument("--study-name", dest="study_name", default=None)
    p_hpo.add_argument("--preprocess-task-id", dest="preprocess_task_id", default=None)
    p_hpo.add_argument("--embed-task-id", dest="embed_task_id", default=None)
    p_hpo.add_argument("--data-path", dest="data_path", default=None)
    p_hpo.add_argument("--embeddings-path", dest="embeddings_path", default=None)
    p_hpo.set_defaults(func=cmd_hpo)

    # register all tasks in ClearML (no execution)
    p_reg = sub.add_parser(
        "register", help="Register all tasks in ClearML without running them"
    )
    p_reg.set_defaults(func=cmd_register)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
