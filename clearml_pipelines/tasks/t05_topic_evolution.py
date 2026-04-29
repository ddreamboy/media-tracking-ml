"""Task t05: Topic evolution analysis (Momeni taxonomy)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from clearml import Task
from shared.clearml_utils import get_artifact, get_artifact_optional, get_production_model
from shared.config import (
    CLEARML_PROJECT_NAME,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)
from shared.topic_evolution import (
    make_cold_start_evolution,
    make_embedding_changed_evolution,
    run_topic_evolution,
)


def main():
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="t05_topic_evolution",
        task_type=Task.TaskTypes.data_processing,
    )
    logger = task.get_logger()

    params = task.connect(
        {
            "upstream_task_ids": "",
            "cosine_threshold_high": 0.7,
            "cosine_threshold_low": 0.4,
            "jaccard_threshold": 0.3,
            "llm_model": LLM_MODEL,
        }
    )

    cosine_high = float(params["cosine_threshold_high"])
    cosine_low = float(params["cosine_threshold_low"])
    jaccard_thr = float(params["jaccard_threshold"])
    llm_model = params["llm_model"]
    llm_base_url = task.get_parameter("Args/llm_base_url") or LLM_BASE_URL
    llm_api_key = task.get_parameter("Args/llm_api_key") or LLM_API_KEY

    # Load new model
    new_model_path = get_artifact(task, "bertopic_model.model")
    new_model = BERTopic.load(new_model_path)

    # Load embedding_meta to check if embedding changed
    emb_meta_artifact = get_artifact_optional(task, "embedding_meta.json")
    embedding_changed = False
    if emb_meta_artifact:
        try:
            emb_meta = emb_meta_artifact.get()
            embedding_changed = bool(emb_meta.get("embedding_changed", False))
        except Exception:
            pass

    # Check cold start
    prod_model_obj = get_production_model()
    cold_start = prod_model_obj is None

    if cold_start:
        print("Cold start - no production model found. All topics = Emerged.")
        evolution_report = make_cold_start_evolution(new_model)
    elif embedding_changed:
        print("Embedding model changed - skipping evolution. All topics = Emerged.")
        evolution_report = make_embedding_changed_evolution(new_model)
    else:
        print("Running topic evolution analysis...")
        prod_task = Task.get_task(task_id=prod_model_obj.task)
        prod_model_path = prod_task.artifacts["bertopic_model.model"].get_local_copy()
        try:
            prod_model = BERTopic.load(prod_model_path)
        except Exception as e:
            print(f"WARNING: could not load production model ({e}), falling back to cold start")
            evolution_report = make_cold_start_evolution(new_model)
            prod_model = None

        if prod_model is not None:
            evolution_report = run_topic_evolution(
                new_model=new_model,
                prod_model=prod_model,
                cosine_threshold_high=cosine_high,
                cosine_threshold_low=cosine_low,
                jaccard_threshold=jaccard_thr,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
            )

        # Cosine similarity heatmap
        sim_matrix = evolution_report.pop("_sim_matrix", None)
        new_ids = evolution_report.pop("_new_topic_ids", [])
        prod_ids = evolution_report.pop("_prod_topic_ids", [])

        if sim_matrix is not None:
            show_n = min(30, len(new_ids), len(prod_ids))
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                sim_matrix[:show_n, :show_n],
                ax=ax,
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
                xticklabels=[f"P{t}" for t in prod_ids[:show_n]],
                yticklabels=[f"N{t}" for t in new_ids[:show_n]],
            )
            ax.set_title(
                f"Cosine Similarity: new vs production (first {show_n} topics)"
            )
            plt.tight_layout()
            heatmap_path = os.path.join(tempfile.gettempdir(), "cosine_heatmap.png")
            fig.savefig(heatmap_path)
            plt.close(fig)
            logger.report_image(
                "evolution", "cosine_heatmap", iteration=0, local_path=heatmap_path
            )

        evolution_report.pop("_sim_matrix", None)
        evolution_report.pop("_new_topic_ids", None)
        evolution_report.pop("_prod_topic_ids", None)

    summary = evolution_report.get("summary", {})
    for evo_type, count in summary.items():
        logger.report_scalar("evolution", evo_type, value=count, iteration=0)

    logger.report_scalar(
        "evolution",
        "llm_border_cases_total",
        value=evolution_report.get("llm_border_cases_total", 0),
        iteration=0,
    )
    logger.report_scalar(
        "evolution",
        "llm_same_topic_verdict",
        value=evolution_report.get("llm_same_topic_verdict", 0),
        iteration=0,
    )
    logger.report_scalar(
        "evolution",
        "llm_different_topic_verdict",
        value=evolution_report.get("llm_different_topic_verdict", 0),
        iteration=0,
    )

    report_path = os.path.join(tempfile.gettempdir(), "evolution_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(evolution_report, f, indent=2, ensure_ascii=False)
    task.upload_artifact("evolution_report.json", artifact_object=report_path)

    print(f"Done. Summary: {summary}")
    task.close()


if __name__ == "__main__":
    main()
