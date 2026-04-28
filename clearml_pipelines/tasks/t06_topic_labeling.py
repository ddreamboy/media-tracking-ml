"""Task t06: Topic labeling via ring-based sampling + LLM"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile

import json

import numpy as np
import pandas as pd
from bertopic import BERTopic
from clearml import Task
from shared.clearml_utils import get_artifact, get_artifact_optional, get_production_model
from shared.config import (
    CLEARML_PROJECT_NAME,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)
from shared.topic_labeling import compute_ring_boundaries, label_all_topics


def load_inherited_labels(prod_task) -> dict[int, dict]:
    if prod_task is None:
        return {}
    try:
        artifact = prod_task.artifacts.get("topic_map_llm.csv")
        if artifact is None:
            return {}
        path = artifact.get_local_copy()
        df = pd.read_csv(path)
        result = {}
        for _, row in df.iterrows():
            topic_id = int(row["Topic"])
            result[topic_id] = {
                "l1": str(row.get("l1", "")),
                "l2": str(row.get("l2", "")),
                "l3": str(row.get("l3", "")),
                "heterogeneous": bool(row.get("is_heterogeneous", False)),
                "coverage": float(row.get("coverage", 0.0))
                if pd.notna(row.get("coverage"))
                else None,
            }
        return result
    except Exception as e:
        print(f"WARNING: could not load inherited labels: {e}")
        return {}


def main():
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="t06_topic_labeling",
        task_type=Task.TaskTypes.data_processing,
    )
    logger = task.get_logger()

    params = task.connect(
        {
            "upstream_task_ids": "",
            "core_docs_count": 5,
            "main_docs_count": 3,
            "periphery_docs_count": 2,
            "llm_model": LLM_MODEL,
        }
    )

    core_docs = int(params["core_docs_count"])
    main_docs = int(params["main_docs_count"])
    periphery_docs = int(params["periphery_docs_count"])
    llm_model = params["llm_model"]
    llm_base_url = task.get_parameter("Args/llm_base_url") or LLM_BASE_URL
    llm_api_key = task.get_parameter("Args/llm_api_key") or LLM_API_KEY

    # Load artifacts
    model_path = get_artifact(task, "bertopic_model.model")
    emb_topics_path = get_artifact(task, "topic_embeddings.npy")
    preprocessed_path = get_artifact(task, "preprocessed.parquet")
    evolution_path = get_artifact(task, "evolution_report.json")

    model = BERTopic.load(model_path)
    df = pd.read_parquet(preprocessed_path)

    with open(evolution_path, encoding="utf-8") as f:
        evolution_report = json.load(f)

    # Get stable topic ids for label inheritance
    stable_topic_ids: set[int] = set()
    for rec in evolution_report.get("topic_mapping", []):
        if (
            rec.get("evolution_type") == "Stable"
            and rec.get("new_topic_id") is not None
        ):
            stable_topic_ids.add(int(rec["new_topic_id"]))

    # Load prod labels for inheritance
    prod_model_obj = get_production_model()
    inherited_labels: dict[int, dict] = {}
    if prod_model_obj is not None:
        prod_task = Task.get_task(task_id=prod_model_obj.task)
        inherited_labels = load_inherited_labels(prod_task)

    # Need topics assignment to compute rings - run transform
    docs_clean = df["text_clean"].tolist()
    docs_lemm = df["text_lemm"].tolist()

    # We need doc embeddings for ring sampling
    # Reuse already-computed embeddings if available in pipeline context
    emb_artifact = get_artifact_optional(task, "embeddings.npy")
    if emb_artifact:
        embeddings_path = emb_artifact.get_local_copy()
        doc_embeddings = np.load(embeddings_path)
    else:
        # No embeddings available - skip ring computation; use top words only
        print("WARNING: embeddings.npy not available, using fallback topic labeling")
        doc_embeddings = None

    topic_ids = sorted([t for t in model.get_topics().keys() if t != -1])

    if doc_embeddings is not None:
        # Transform docs to get topic assignments
        topics, _ = model.transform(docs_lemm, embeddings=doc_embeddings)

        q_low, q_high = compute_ring_boundaries(
            model, topic_ids, doc_embeddings, topics
        )
    else:
        topics = [-1] * len(df)
        q_low, q_high = 0.25, 0.60

    records = label_all_topics(
        model=model,
        topic_ids=topic_ids,
        doc_embeddings=doc_embeddings
        if doc_embeddings is not None
        else np.zeros((len(df), 1)),
        topics=topics,
        docs=docs_clean,
        q_low=q_low,
        q_high=q_high,
        core_docs=core_docs,
        main_docs=main_docs,
        periphery_docs=periphery_docs,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        inherited_labels=inherited_labels,
        stable_topic_ids=stable_topic_ids,
    )

    # Fill evolution_type from evolution_report (Stable/Emerged/Merged/Split/Disappeared)
    evo_map: dict[int, str] = {}
    for rec in evolution_report.get("topic_mapping", []):
        tn = rec.get("new_topic_id")
        if tn is not None:
            evo_map[int(tn)] = rec.get("evolution_type", "")

    for rec in records:
        rec["evolution_type"] = evo_map.get(rec["Topic"], "Emerged")

    df_labels = pd.DataFrame(records)

    # label_source: "inherited" = label reused from production, "llm" = freshly generated
    topics_by_llm = int((df_labels["label_source"] == "llm").sum())
    topics_inherited = int((df_labels["label_source"] == "inherited").sum())
    heterogeneous_count = int(df_labels["is_heterogeneous"].sum())
    avg_coverage = (
        float(df_labels["coverage"].dropna().mean()) if len(df_labels) > 0 else 0.0
    )
    total_llm_calls = (
        int(df_labels["_llm_calls"].iloc[-1])
        if "_llm_calls" in df_labels.columns
        else topics_by_llm
    )

    logger.report_scalar(
        "labeling", "topics_labeled_by_llm", value=topics_by_llm, iteration=0
    )
    logger.report_scalar(
        "labeling",
        "topics_inherited_from_production",
        value=topics_inherited,
        iteration=0,
    )
    logger.report_scalar(
        "labeling", "heterogeneous_topics_count", value=heterogeneous_count, iteration=0
    )
    logger.report_scalar("labeling", "avg_coverage", value=avg_coverage, iteration=0)
    logger.report_scalar(
        "labeling", "total_llm_calls", value=total_llm_calls, iteration=0
    )

    # HTML examples table - show 5 topics from each evolution type
    try:
        sample_rows = []
        for evo_type in ["Stable", "Emerged", "Split", "Merged"]:
            subset = df_labels[df_labels["evolution_type"] == evo_type].head(5)
            sample_rows.append(subset)
        if sample_rows:
            sample_df = pd.concat(sample_rows).head(20)
            html = sample_df[
                [
                    "Topic",
                    "Human_Label",
                    "is_heterogeneous",
                    "coverage",
                    "evolution_type",
                ]
            ].to_html()
            logger.report_text("label_examples", html)
    except Exception:
        pass

    # Spec-required columns first, then extras (label_source, l1/l2/l3 for ml_service use)
    out_cols = [
        "Topic",
        "Human_Label",
        "is_heterogeneous",
        "coverage",
        "evolution_type",
        "zone_boundaries_q25_q60",
        # non-spec extras that are useful for ml_service and auditing:
        "label_source",
        "l1",
        "l2",
        "l3",
        "core_topic",
        "periphery_topic",
        "count",
    ]
    out_cols = [c for c in out_cols if c in df_labels.columns]
    out_path = os.path.join(tempfile.gettempdir(), "topic_map_llm.csv")
    df_labels[out_cols].to_csv(out_path, index=False, encoding="utf-8")
    task.upload_artifact("topic_map_llm.csv", artifact_object=out_path)

    print(
        f"Done. LLM calls={total_llm_calls}, inherited={topics_inherited}, het={heterogeneous_count}"
    )
    task.close()


if __name__ == "__main__":
    main()
