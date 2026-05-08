"""
Utility: register bertopic_model artifacts from completed t08 tasks as ClearML
Model objects with the 'production' / 'archived' tags.

Run once to populate the Model Registry so models appear in ClearML UI:
    uv run python register_production_models.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from clearml import Model, Task
from shared.config import CLEARML_PROJECT_NAME, TAG_PRODUCTION, TAG_ARCHIVED


def find_promote_tasks():
    tasks = Task.get_tasks(
        project_name=CLEARML_PROJECT_NAME,
        task_name="t08_promote_model",
        task_filter={"status": ["completed"]},
    )
    tasks.sort(key=lambda t: t.data.completed or "", reverse=True)
    return tasks


def ensure_model_registered(task: Task, desired_tag: str) -> Model | None:
    artifact = task.artifacts.get("bertopic_model.model")
    if artifact is None:
        print(f"  [SKIP] task {task.id}: no bertopic_model.model artifact")
        return None

    run_date = (task.data.completed or "")[:10]
    model_name = f"BERTopic-{run_date}"

    meta_artifact = task.artifacts.get("training_meta.json")
    meta = meta_artifact.get() if meta_artifact else {}

    # Re-use existing output model if t08 already registered one
    existing_outputs = task.get_models().get("output", [])
    if existing_outputs:
        model_obj = existing_outputs[0]
        print(f"  Using existing output model: {model_obj.id} ({model_obj.name})")
    else:
        model_obj = Model.import_model(
            weights_url=artifact.url,
            config_dict=meta,
            label_enumeration={},
            name=model_name,
            project=CLEARML_PROJECT_NAME,
            tags=[],
        )
        print(f"  Imported new model: {model_obj.id} ({model_obj.name})")

    current_tags = list(model_obj.tags or [])
    # Remove old production/archived tags, set desired
    cleaned = [t for t in current_tags if t not in (TAG_PRODUCTION, TAG_ARCHIVED)]
    cleaned.append(desired_tag)
    model_obj.tags = cleaned
    print(f"  Tags set: {cleaned}")
    return model_obj


def main():
    tasks = find_promote_tasks()
    if not tasks:
        print("No completed t08_promote_model tasks found.")
        return

    print(f"Found {len(tasks)} completed t08_promote_model task(s).\n")

    latest = tasks[0]
    print(f"Latest task → production: {latest.id}")
    prod_model = ensure_model_registered(latest, TAG_PRODUCTION)

    for old_task in tasks[1:]:
        print(f"\nOlder task → archived: {old_task.id}")
        ensure_model_registered(old_task, TAG_ARCHIVED)

    print("\nDone. ClearML UI → Model Registry → filter tag='production'")
    if prod_model:
        print(f"Production model ID: {prod_model.id}")


if __name__ == "__main__":
    main()
