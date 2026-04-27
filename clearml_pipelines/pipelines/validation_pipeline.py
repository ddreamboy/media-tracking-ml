"""Pipeline 2: Validation & Promotion Pipeline - orchestrates t07->t08"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clearml import Task
from clearml.automation.controller import PipelineDecorator
from shared.config import CLEARML_PROJECT_NAME


def _run_task(script_name: str):
    import os
    import subprocess
    import sys

    repo_root = os.path.abspath(".")
    pipelines_dir = os.path.join(repo_root, "clearml_pipelines")
    task_script = os.path.join(pipelines_dir, "tasks", script_name)
    env = {**os.environ, "PYTHONPATH": pipelines_dir}

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
    return_values=["validation_report_path"],
    execution_queue="default",
    task_type=Task.TaskTypes.data_processing,
)
def step_validate(training_meta_path: str) -> str:
    _run_task("t07_validate_model.py")
    return "/tmp/validation_report.json"


@PipelineDecorator.component(
    return_values=["promotion_done"],
    execution_queue="default",
    task_type=Task.TaskTypes.data_processing,
)
def step_promote(
    validation_report_path: str,
    topic_map_path: str,
    evolution_report_path: str,
    model_path: str,
    topic_emb_path: str,
    training_meta_path: str,
) -> bool:
    _run_task("t08_promote_model.py")
    return True


@PipelineDecorator.pipeline(
    name="Validation & Promotion Pipeline",
    project=CLEARML_PROJECT_NAME,
    version="1.0",
    pipeline_execution_queue="default",
)
def validation_pipeline(
    training_meta_path: str,
    topic_map_path: str,
    evolution_report_path: str,
    model_path: str,
    topic_emb_path: str,
):
    validation_report_path = step_validate(training_meta_path)
    promotion_done = step_promote(
        validation_report_path,
        topic_map_path,
        evolution_report_path,
        model_path,
        topic_emb_path,
        training_meta_path,
    )
    return promotion_done


def run_pipeline(
    training_meta_path: str,
    topic_map_path: str,
    evolution_report_path: str,
    model_path: str,
    topic_emb_path: str,
):
    print("Starting Validation & Promotion Pipeline")
    validation_pipeline(
        training_meta_path=training_meta_path,
        topic_map_path=topic_map_path,
        evolution_report_path=evolution_report_path,
        model_path=model_path,
        topic_emb_path=topic_emb_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training-meta", required=True)
    parser.add_argument("--topic-map", required=True)
    parser.add_argument("--evolution-report", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--topic-emb", required=True)
    args = parser.parse_args()

    run_pipeline(
        args.training_meta,
        args.topic_map,
        args.evolution_report,
        args.model_path,
        args.topic_emb,
    )
