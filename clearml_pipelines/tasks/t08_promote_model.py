"""Task t08: Promote model to production and push to HF Hub"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


import json
from datetime import datetime, timezone

from clearml import Model, Task
from shared.clearml_utils import get_artifact, tag_model_as_production
from shared.config import (
    CLEARML_PROJECT_NAME,
    HF_TOKEN,
    TAG_FAILED_VALIDATION,
)
from shared.hf_utils import push_artifacts_to_hub


def main():
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="t08_promote_model",
        task_type=Task.TaskTypes.data_processing,
    )
    logger = task.get_logger()

    task.connect({"upstream_task_ids": ""})

    # Load validation report
    val_report_path = get_artifact(task, "validation_report.json")
    with open(val_report_path) as f:
        validation_report = json.load(f)

    verdict = validation_report["verdict"]

    if verdict != "passed":
        print(f"Validation failed - skipping promotion. Verdict: {verdict}")
        task.connect({"promotion_blocked": True, "reason": verdict}, name="promotion")
        logger.report_scalar(
            "promotion", "promoted_to_production", value=0, iteration=0
        )

        # Tag failed model
        try:
            current_task_id = task.id
            failed_model = Model(
                model_id=current_task_id,
                project=CLEARML_PROJECT_NAME,
            )
            existing_tags = list(failed_model.tags or [])
            existing_tags.append(TAG_FAILED_VALIDATION)
            failed_model.tags = existing_tags
        except Exception as e:
            print(f"WARNING: could not tag failed model: {e}")

        task.close()
        return

    # Load required artifacts
    evolution_path = get_artifact(task, "evolution_report.json")
    topic_map_path = get_artifact(task, "topic_map_llm.csv")
    model_path = get_artifact(task, "bertopic_model.model")
    topic_emb_path = get_artifact(task, "topic_embeddings.npy")
    training_meta_path = get_artifact(task, "training_meta.json")

    with open(evolution_path, encoding="utf-8") as f:
        evolution_report = json.load(f)

    with open(training_meta_path) as f:
        training_meta = json.load(f)

    # Tag model in ClearML registry
    tag_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_tag = f"model-{tag_date}"

    try:
        # Tag current ClearML task's output model as production
        output_models = task.get_models()["output"]
        if output_models:
            new_model_obj = output_models[0]
            tag_model_as_production(new_model_obj)
            task.connect({"new_model_tag": new_tag}, name="promotion")
    except Exception as e:
        print(f"WARNING: could not tag model in registry: {e}")

    # Deprecated topic IDs
    deprecated_ids = [
        rec.get("matched_old_topic_id")
        for rec in evolution_report.get("topic_mapping", [])
        if rec.get("evolution_type") == "Disappeared"
    ]
    deprecated_ids = [d for d in deprecated_ids if d is not None]

    logger.report_scalar("promotion", "promoted_to_production", value=1, iteration=0)
    logger.report_scalar(
        "promotion",
        "deprecated_topic_ids_count",
        value=len(deprecated_ids),
        iteration=0,
    )
    task.connect(
        {"new_model_tag": new_tag, "promoted_to_production": True}, name="promotion"
    )

    # Push to HF Hub
    if HF_TOKEN:
        files_to_push = {
            "topic_map_llm.csv": topic_map_path,
            # topic_embeddings_gemini.npy - semantic matcher artifact name for ml_service
            "topic_embeddings_gemini.npy": topic_emb_path,
            "training_meta.json": training_meta_path,
            "evolution_report.json": evolution_path,
            "bertopic_model.model": model_path,
        }

        try:
            hf_url = push_artifacts_to_hub(files_to_push, tag=new_tag)
            task.connect({"hf_hub_url": hf_url}, name="promotion")
        except Exception as e:
            print(f"WARNING: HF Hub push failed: {e}")
    else:
        print("HF_TOKEN not set - skipping HF Hub push")

    # Save heterogeneous_topic_ids for next cycle
    import pandas as pd

    try:
        topic_map_df = pd.read_csv(topic_map_path)
        het_ids = topic_map_df[topic_map_df["is_heterogeneous"] == True][
            "Topic"
        ].tolist()
        het_ids = [int(x) for x in het_ids]
    except Exception:
        het_ids = []

    import json as _json

    het_path = "/tmp/heterogeneous_topic_ids.json"
    with open(het_path, "w") as f:
        _json.dump(het_ids, f)
    task.upload_artifact("heterogeneous_topic_ids.json", artifact_object=het_path)

    print(
        f"Done. Promoted: {new_tag}, deprecated_ids={len(deprecated_ids)}, het_ids={len(het_ids)}"
    )
    task.close()


if __name__ == "__main__":
    main()
