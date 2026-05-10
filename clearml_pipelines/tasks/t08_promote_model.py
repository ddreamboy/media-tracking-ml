"""Task t08: Promote model to production and push to HF Hub"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from clearml import Model, OutputModel, Task
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

    # Register model in ClearML Models registry
    tag_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_tag = f"model-{tag_date}"

    try:
        # Модель сохранена как директория - пакуем в zip для регистрации
        zip_path = os.path.join(tempfile.gettempdir(), f"bertopic_model_{tag_date}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f_path in Path(model_path).rglob("*"):
                if f_path.is_file():
                    zf.write(f_path, f_path.relative_to(model_path))

        with open(training_meta_path) as f:
            _meta = json.load(f)

        output_model = OutputModel(
            task=task,
            name=f"BERTopic {new_tag}",
            tags=["production"],
            framework="BERTopic",
            config_dict={
                "embedding_model": _meta.get("embedding_model_name", ""),
                "noise_ratio": _meta.get("metrics", {}).get("noise_ratio", ""),
                "corpus_size": _meta.get("corpus_size", ""),
            },
        )
        output_model.update_weights(weights_filename=zip_path)
        output_model.publish()
        tag_model_as_production(output_model)
        print(f"Registered OutputModel: {output_model.id}  tag={new_tag}")
    except Exception as e:
        print(f"WARNING: could not register model in ClearML registry: {e}")

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

    het_path = os.path.join(tempfile.gettempdir(), "heterogeneous_topic_ids.json")
    with open(het_path, "w") as f:
        _json.dump(het_ids, f)
    task.upload_artifact("heterogeneous_topic_ids.json", artifact_object=het_path)

    print(
        f"Done. Promoted: {new_tag}, deprecated_ids={len(deprecated_ids)}, het_ids={len(het_ids)}"
    )
    task.close()


if __name__ == "__main__":
    main()
