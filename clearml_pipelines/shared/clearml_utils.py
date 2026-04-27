import json
import warnings
from typing import Optional

from clearml import Task, Model

from .config import (
    CLEARML_PROJECT_NAME,
    TAG_PRODUCTION,
    TAG_ARCHIVED,
    TAG_TRAINING_IN_PROGRESS,
)


def get_production_model() -> Optional[Model]:
    models = Model.query_models(
        project_name=CLEARML_PROJECT_NAME,
        tags=[TAG_PRODUCTION],
    )
    if not models:
        return None
    # Sort by creation time, return most recent
    models.sort(key=lambda m: m.created, reverse=True)
    return models[0]


def get_production_artifact(artifact_name: str):
    model = get_production_model()
    if model is None:
        return None
    task = Task.get_task(task_id=model.task)
    if task is None:
        return None
    artifact = task.artifacts.get(artifact_name)
    if artifact is None:
        return None
    return artifact.get()


def is_training_in_progress() -> bool:
    tasks = Task.get_tasks(
        project_name=CLEARML_PROJECT_NAME,
        tags=[TAG_TRAINING_IN_PROGRESS],
        task_filter={"status": ["in_progress", "queued"]},
    )
    return len(tasks) > 0


def get_best_hparams(default_path: str) -> dict:
    try:
        prod_model = get_production_model()
        if prod_model is not None:
            task = Task.get_task(task_id=prod_model.task)
            if task is not None:
                artifact = task.artifacts.get("best_hparams.json")
                if artifact is not None:
                    return artifact.get()
    except Exception as e:
        warnings.warn(f"Could not load hparams from Model Registry: {e}")

    with open(default_path) as f:
        return json.load(f)


def get_heterogeneous_topic_ids() -> list[int]:
    try:
        prod_model = get_production_model()
        if prod_model is not None:
            task = Task.get_task(task_id=prod_model.task)
            if task is not None:
                artifact = task.artifacts.get("heterogeneous_topic_ids.json")
                if artifact is not None:
                    data = artifact.get()
                    return data if isinstance(data, list) else []
    except Exception as e:
        warnings.warn(f"Could not load heterogeneous_topic_ids: {e}")
    return []


def get_production_embedding_model_name() -> Optional[str]:
    try:
        prod_model = get_production_model()
        if prod_model is not None:
            task = Task.get_task(task_id=prod_model.task)
            if task is not None:
                artifact = task.artifacts.get("training_meta.json")
                if artifact is not None:
                    meta = artifact.get()
                    return meta.get("embedding_model_name")
    except Exception as e:
        warnings.warn(f"Could not load production embedding model name: {e}")
    return None


def tag_model_as_production(new_model: Model) -> None:
    existing = Model.query_models(
        project_name=CLEARML_PROJECT_NAME,
        tags=[TAG_PRODUCTION],
    )
    for m in existing:
        current_tags = list(m.system_tags or []) + list(m.tags or [])
        new_tags = [t for t in current_tags if t != TAG_PRODUCTION]
        new_tags.append(TAG_ARCHIVED)
        m.tags = new_tags

    tags = list(new_model.tags or [])
    if TAG_PRODUCTION not in tags:
        tags.append(TAG_PRODUCTION)
    new_model.tags = tags
